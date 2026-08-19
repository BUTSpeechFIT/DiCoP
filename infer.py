#!/usr/bin/env python
"""Target-speaker ASR inference: diarization + audio -> STM.

For every speaker in every session, the diarization is turned into the
silence/target/non-target/overlap mask that conditions the encoder, and the audio is decoded
once per speaker. The result is written as a single STM, byte-compatible with the `hyp.stm`
that training-time validation produces.

The diarization and audio come from either an RTTM plus an audio directory:

    python infer.py \\
        --rttm /path/to/rttms/ \\
        --audio-dir /path/to/audio/ \\
        --output hyp.stm \\
        --checkpoint /path/to/best.ckpt

or a Lhotse CutSet, which carries both:

    python infer.py \\
        --cuts /path/to/cuts.jsonl.gz \\
        --output hyp.stm \\
        --checkpoint /path/to/best.ckpt

A pre-segmented cutset holds several cuts per recording; those are decoded separately and
their words merged onto the recording's timeline, so one STM session comes out either way.

Long-form notes. Full-session decoding is the default and is what the model was trained and
evaluated with (`pos_emb_max_len` covers an hour at 12.5 Hz). It is also the memory-hungry
option: a 30-minute session is ~22.5k encoder frames, so full self-attention over it is only
comfortable with scaled dot-product attention (`--use-sdpa`, on by default). If a session
still does not fit, either restrict the attention window (`--att-context-size 128,128`), switch
to windowed local attention (`-O model.encoder.self_attention_model=rel_pos_local_attn`), or
decode in windows (`--chunk-seconds 120`), which costs accuracy because the conditioning
sees less of the surrounding non-target speech.

The same applies to batching speakers. `--per-speaker-batching` decodes all of a session's
speakers in one pass off the single spectrogram the audio was loaded and preprocessed into, which
is the fast way to run short units and `--chunk-seconds` windows, but it multiplies the encoder's
activation memory by the speaker count, so on full-session long-form the default of one speaker
per pass is the safe choice.
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nemo.utils import logging

from src.data.lhotse_utils import (
    cut_segments,
    cut_session_id,
    cut_source_label,
    load_cut_audio,
    load_cutset,
    require_supported_cut,
)
from src.data.stno import (
    SpeechSegment,
    create_stno_masks,
    frame_rate_from_downsampling_factor,
    num_encoder_frames,
    speaker_activity_mask,
)
from utils.audio import DEFAULT_AUDIO_EXTENSIONS, load_session_audio, resolve_audio_path
from utils.inference import DEFAULT_CONFIG_PATH, InferenceRuntime, load_inference_config
from utils.rttm import load_rttm, speakers_in
from utils.stm import WordSpan, spans_to_segments, write_stm


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rttm", type=Path, help="RTTM file, or directory of *.rttm. Needs --audio-dir.")
    source.add_argument(
        "--cuts",
        type=Path,
        help="Lhotse CutSet (.jsonl.gz). Supplies both the audio paths and the diarization, so "
        "--audio-dir and its lookup options are not used.",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None, help="Directory containing the audio. Required with --rttm."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .stm path. A directory gets hyp.stm inside it."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt / .nemo, or an NGC/HF model id.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Base config YAML.")
    parser.add_argument(
        "-O",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override a config key, e.g. -O model.encoder.self_attention_model=rel_pos_local_attn. "
        "Repeatable. Applied after a checkpoint's own config, so it decides the architecture the "
        "weights load into.",
    )

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Target speakers decoded per forward pass (default 1), so a session with N speakers "
        "never uses more than N regardless of what is asked for. Features are computed once per "
        "session either way. Worth raising for short units and for --chunk-seconds decoding; "
        "full-session long-form already saturates the GPU at 1, where it buys little and costs "
        "memory roughly linearly in the batch.",
    )
    parser.add_argument(
        "--per-speaker-batching",
        action="store_true",
        help="Decode every speaker of a unit in one forward pass, so the batch follows the unit "
        "instead of a fixed --batch-size. The audio is loaded and the features computed once "
        "either way; this only removes the per-speaker passes over them. Peak memory scales with "
        "the session's speaker count, so it pairs best with --chunk-seconds decoding. Cannot be "
        "combined with --batch-size.",
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Decode in windows of this length instead of whole sessions. 0 (default) means "
        "full-session decoding, which is what the model was trained for.",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=10.0,
        help="Context shared between consecutive windows. Words are attributed to one window only.",
    )

    parser.add_argument(
        "--audio-ext",
        default=",".join(DEFAULT_AUDIO_EXTENSIONS),
        help="Comma-separated audio extensions to try.",
    )
    parser.add_argument(
        "--audio-glob",
        default=None,
        help="Glob relative to --audio-dir with a {session} placeholder, for layouts the plain "
        "{session}{ext} lookups miss, e.g. '{session}/audio.wav'.",
    )
    parser.add_argument(
        "--channel-selector",
        default="average",
        help="'average' mixes multi-channel audio down (the training setup), an integer picks "
        "one channel, 'none' leaves it as-is.",
    )

    parser.add_argument("--sessions", default=None, help="Comma-separated subset of session ids.")
    parser.add_argument("--session-list", type=Path, default=None, help="File with one session id per line.")
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=0.0,
        help="Skip RTTM speakers with less total speech than this. Each speaker costs a decoding "
        "pass, so this suppresses spurious diarization output cheaply.",
    )

    parser.add_argument("--stm-granularity", default="word", choices=["word", "segment"])
    parser.add_argument(
        "--att-context-size",
        default=None,
        help="Override the encoder attention context as 'left,right', e.g. '128,128'. Reduces "
        "memory on very long sessions at some cost in accuracy.",
    )
    sdpa = parser.add_mutually_exclusive_group()
    sdpa.add_argument("--use-sdpa", dest="use_sdpa", action="store_true", default=None)
    sdpa.add_argument("--no-use-sdpa", dest="use_sdpa", action="store_false")
    parser.add_argument(
        "--continue-on-fail", action="store_true", help="Log and skip sessions that fail instead of aborting."
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    """h:mm:ss, so a multi-hour decode is readable without counting digits."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def resolve_output_path(output: Path) -> Path:
    if output.is_dir() or output.suffix == "":
        return output / "hyp.stm"
    return output


def selected_sessions(all_sessions, sessions_arg, session_list_path):
    """Intersect the input's sessions with an explicit selection, if one was given."""
    if sessions_arg is None and session_list_path is None:
        return sorted(all_sessions)

    requested = []
    if sessions_arg:
        requested.extend(s.strip() for s in sessions_arg.split(",") if s.strip())
    if session_list_path:
        requested.extend(
            line.strip() for line in session_list_path.read_text().splitlines() if line.strip()
        )

    missing = [s for s in requested if s not in all_sessions]
    if missing:
        raise ValueError(f"Sessions not present in the input: {', '.join(sorted(missing))}")
    return sorted(dict.fromkeys(requested))


@dataclass
class DecodeUnit:
    """One contiguous stretch of audio to decode, with the diarization covering it.

    An RTTM gives one unit per session. A CutSet gives one unit per cut, and a pre-segmented
    cutset holds many cuts per recording — hence `session_id` (the STM key, shared across a
    recording's cuts) being distinct from `label` (this unit, for logging) and `time_offset`
    (where this unit sits on the recording's timeline).
    """

    session_id: str
    label: str
    segments: List[SpeechSegment]  # unit-relative
    time_offset: float  # added to decoded word times
    audio_path: Optional[Path] = None  # RTTM input
    cut: Optional[object] = None  # CutSet input

    def load_audio(self, runtime, channel_selector):
        if self.cut is not None:
            return load_cut_audio(self.cut, runtime.featurizer, channel_selector)
        return load_session_audio(self.audio_path, runtime.featurizer, channel_selector)

    @property
    def source(self):
        return self.audio_path if self.cut is None else cut_source_label(self.cut)


def build_rttm_units(args, extensions) -> List[DecodeUnit]:
    """One decode unit per RTTM session, with audio resolved under --audio-dir."""
    if args.audio_dir is None:
        raise ValueError("--audio-dir is required with --rttm.")

    rttm_sessions = load_rttm(args.rttm)
    session_ids = selected_sessions(rttm_sessions, args.sessions, args.session_list)
    logging.info("Decoding %d sessions from %s", len(session_ids), args.rttm)

    units = []
    for session_id in session_ids:
        units.append(
            DecodeUnit(
                session_id=session_id,
                label=session_id,
                segments=rttm_sessions[session_id],
                time_offset=0.0,
                audio_path=resolve_audio_path(
                    session_id, args.audio_dir, extensions=extensions, glob_template=args.audio_glob
                ),
            )
        )
    return units


def build_cut_units(args) -> List[DecodeUnit]:
    """One decode unit per cut, grouped under the recording they belong to."""
    cuts = list(load_cutset(args.cuts))
    for cut in cuts:
        require_supported_cut(cut)

    by_session = {}
    for cut in cuts:
        by_session.setdefault(cut_session_id(cut), []).append(cut)

    session_ids = selected_sessions(by_session, args.sessions, args.session_list)
    logging.info("Decoding %d cuts across %d sessions from %s", len(cuts), len(session_ids), args.cuts)

    units = []
    for session_id in session_ids:
        # Sorted so a recording's windows are decoded, and logged, in time order.
        for cut in sorted(by_session[session_id], key=lambda c: c.start):
            units.append(
                DecodeUnit(
                    session_id=session_id,
                    label=cut.id,
                    segments=cut_segments(cut),
                    # Supervision times are cut-relative, so shift this cut's words onto the
                    # recording's timeline, the same rule training-time scoring applies.
                    time_offset=cut.start,
                    cut=cut,
                )
            )
    return units


def build_windows(num_samples, chunk_seconds, overlap_seconds, sample_rate, samples_per_frame):
    """Split a recording into decoding windows aligned to whole encoder frames.

    Aligning both the window start and the hop to `samples_per_frame` keeps the audio and the
    mask on the same frame grid, so a mask slice always corresponds to its window exactly.

    Returns:
        A list of `(begin_sample, end_sample, emit_from, emit_until)`, where the emit bounds are
        the times (in seconds) whose words this window owns. Consecutive windows share context
        but split ownership at the midpoint of their overlap, so no word is emitted twice.
    """
    if chunk_seconds <= 0:
        return [(0, num_samples, 0.0, math.inf)]

    window = max(samples_per_frame, int(round(chunk_seconds * sample_rate / samples_per_frame)) * samples_per_frame)
    overlap = max(0, int(round(overlap_seconds * sample_rate / samples_per_frame)) * samples_per_frame)
    overlap = min(overlap, window - samples_per_frame)
    hop = window - overlap

    windows = []
    begin = 0
    while begin < num_samples:
        end = min(begin + window, num_samples)
        is_first = begin == 0
        is_last = end >= num_samples
        emit_from = 0.0 if is_first else (begin + overlap / 2) / sample_rate
        emit_until = math.inf if is_last else (end - overlap / 2) / sample_rate
        windows.append((begin, end, emit_from, emit_until))
        if is_last:
            break
        begin += hop

    return windows


def speaker_batches(speakers, per_speaker_batching, batch_size):
    """Group target speakers into the sets decoded in one forward pass.

    Every group shares one spectrogram, so grouping only trades memory against passes; it never
    changes what is decoded. `per_speaker_batching` makes that group the whole unit, so all of a
    session's speakers are inferred simultaneously, each with its own STNO mask.

    Yields:
        Lists of speaker ids, in the order given.
    """
    size = len(speakers) if per_speaker_batching else batch_size
    for start in range(0, len(speakers), max(size, 1)):
        yield speakers[start : start + size]


def decode_session(runtime, audio, segments, speakers, args):
    """Decode every target speaker of one session.

    Returns:
        `{speaker: [WordSpan, ...]}` with times absolute within the recording.
    """
    frame_rate = frame_rate_from_downsampling_factor(runtime.sample_rate, runtime.audio_downsampling_factor)
    samples_per_frame = runtime.audio_downsampling_factor
    windows = build_windows(
        audio.shape[0], args.chunk_seconds, args.chunk_overlap_seconds, runtime.sample_rate, samples_per_frame
    )

    spans_by_speaker = defaultdict(list)

    for begin, end, emit_from, emit_until in windows:
        window_audio = audio[begin:end]
        window_offset = begin / runtime.sample_rate
        num_frames = num_encoder_frames(window_audio.shape[0], samples_per_frame)

        # Shift the diarization into window-relative time before rasterizing, so the mask lines
        # up with the window's own frames.
        shifted = [
            type(segment)(
                start=segment.start - window_offset, duration=segment.duration, speaker=segment.speaker
            )
            for segment in segments
        ]
        activity = speaker_activity_mask(shifted, speakers, num_frames, frame_rate)

        processed_signal, processed_signal_length = runtime.preprocess(window_audio)

        for batch_speakers in speaker_batches(speakers, args.per_speaker_batching, args.batch_size):
            masks = torch.stack(
                [create_stno_masks(activity, speakers.index(speaker)) for speaker in batch_speakers]
            )
            batch_spans = runtime.decode(processed_signal, processed_signal_length, masks)

            for speaker, spans in zip(batch_speakers, batch_spans):
                for span in spans:
                    start = window_offset + span.start
                    # Ownership is decided by the word's start time, so overlapping windows
                    # cannot emit the same word twice.
                    if emit_from <= start < emit_until:
                        spans_by_speaker[speaker].append(
                            WordSpan(text=span.text, start=start, end=window_offset + span.end)
                        )

    for speaker in spans_by_speaker:
        spans_by_speaker[speaker].sort(key=lambda span: (span.start, span.end))
    return spans_by_speaker


def main():
    args = parse_args()

    if args.per_speaker_batching and args.batch_size is not None:
        raise ValueError(
            "--per-speaker-batching decides the batch from the unit's speaker count; "
            "--batch-size cannot also be set."
        )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f"--batch-size must be at least 1, got {args.batch_size}")
    if args.batch_size is None:
        args.batch_size = 1

    output_path = resolve_output_path(args.output)
    extensions = [e if e.startswith(".") else f".{e}" for e in args.audio_ext.split(",") if e]
    channel_selector = None if args.channel_selector.lower() == "none" else args.channel_selector
    if channel_selector is not None and channel_selector.isdigit():
        channel_selector = int(channel_selector)

    att_context_size = None
    if args.att_context_size:
        att_context_size = [int(v) for v in args.att_context_size.replace(" ", "").split(",")]
        if len(att_context_size) != 2:
            raise ValueError(f"--att-context-size needs two values, got {args.att_context_size!r}")

    if args.cuts is not None:
        # A cutset carries its own audio paths, so the lookup options would be silent no-ops.
        ignored = [
            name
            for name, value in (("--audio-dir", args.audio_dir), ("--audio-glob", args.audio_glob))
            if value is not None
        ]
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} {'is' if len(ignored) == 1 else 'are'} not used with "
                f"--cuts; the cutset already carries the audio paths."
            )

    units = build_cut_units(args) if args.cuts is not None else build_rttm_units(args, extensions)

    cfg = load_inference_config(args.checkpoint, config_path=args.config)
    runtime = InferenceRuntime(
        cfg,
        device_name=args.device,
        precision=args.precision,
        att_context_size=att_context_size,
        use_sdpa=args.use_sdpa,
        overrides=args.overrides,
    )

    if args.chunk_seconds <= 0 and args.per_speaker_batching:
        logging.warning(
            "--per-speaker-batching with full-session decoding puts every speaker of a session "
            "in one forward pass, so peak memory scales with the speaker count and can exhaust "
            "the GPU on long recordings; --chunk-seconds, --att-context-size, or a fixed "
            "--batch-size keep it bounded."
        )
    elif args.chunk_seconds <= 0 and args.batch_size > 1:
        logging.warning(
            "--batch-size %d with full-session decoding can exhaust GPU memory on long "
            "recordings; 1 is the safe choice.",
            args.batch_size,
        )

    # Accumulated across units, since a recording may be split over several cuts.
    spans_by_session = defaultdict(lambda: defaultdict(list))
    speakers_by_session = defaultdict(list)
    failed = []

    # Units differ wildly in length, so units/s says little on its own; the postfix carries the
    # number that does travel between sets — audio seconds decoded per wall-clock second.
    decode_started = time.monotonic()
    audio_seconds = 0.0
    progress = tqdm(units, desc="decoding", unit="unit", dynamic_ncols=True)

    for index, unit in enumerate(progress, start=1):
        speakers = speakers_in(unit.segments, args.min_speech_seconds)
        if not speakers:
            logging.warning("%s has no speaker above --min-speech-seconds; skipping", unit.label)
            continue

        try:
            audio = unit.load_audio(runtime, channel_selector)
            logging.info(
                "[%d/%d] %s: %.1f s, %d speakers (%s)",
                index,
                len(units),
                unit.label,
                audio.shape[0] / runtime.sample_rate,
                len(speakers),
                unit.source,
            )
            spans_by_speaker = decode_session(runtime, audio, unit.segments, speakers, args)
        except Exception as exc:
            if not args.continue_on_fail:
                raise
            logging.error("%s failed: %s", unit.label, exc)
            failed.append(unit.label)
            continue

        audio_seconds += audio.shape[0] / runtime.sample_rate
        progress.set_postfix_str(
            f"{audio_seconds / max(time.monotonic() - decode_started, 1e-9):.1f}x realtime"
        )

        for speaker in speakers:
            if speaker not in speakers_by_session[unit.session_id]:
                speakers_by_session[unit.session_id].append(speaker)
            spans_by_session[unit.session_id][speaker].extend(
                WordSpan(text=span.text, start=span.start + unit.time_offset, end=span.end + unit.time_offset)
                for span in spans_by_speaker.get(speaker, [])
            )

        words = sum(len(spans_by_speaker.get(speaker, [])) for speaker in speakers)
        logging.info("[%d/%d] %s: %d words decoded", index, len(units), unit.label, words)

    decode_elapsed = time.monotonic() - decode_started
    progress.close()

    session_segments = {}
    session_speakers = {}
    for session_id, per_speaker in spans_by_session.items():
        speakers = sorted(speakers_by_session[session_id])
        session_speakers[session_id] = speakers
        session_segments[session_id] = [
            segment
            for speaker in speakers
            # Cuts are decoded in time order, but sort anyway so the STM is monotonic per speaker.
            for segment in spans_to_segments(
                sorted(per_speaker.get(speaker, []), key=lambda s: (s.start, s.end)),
                session_id,
                speaker,
                args.stm_granularity,
            )
        ]

    num_lines = write_stm(output_path, session_segments, session_speakers)
    print(f"Wrote {num_lines} STM lines for {len(session_segments)} sessions to {output_path}")
    # Model loading is deliberately outside this: it is a fixed cost that says nothing about
    # how fast the set decodes, and on a Hub id it is mostly download time.
    print(
        f"Decoded {format_duration(audio_seconds)} of audio in {format_duration(decode_elapsed)} "
        f"({audio_seconds / max(decode_elapsed, 1e-9):.1f}x realtime)"
    )
    if failed:
        print(f"{len(failed)} units failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
