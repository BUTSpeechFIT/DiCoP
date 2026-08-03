#!/usr/bin/env python
"""Export a Lhotse CutSet as the wavs and RTTMs `infer.py --rttm --audio-dir` consumes.

The two inference routes express the same thing — a cutset carries the audio and the
diarization, an RTTM plus an audio directory carries the same two — so decoding an exported
cutset must reproduce decoding the cutset itself. This script writes the export, and
`scripts/run_rttm_parity.sh` checks that the two decodes agree.

    python scripts/cutset_to_wav_rttm.py --cuts cuts.jsonl.gz --output export/
    python infer.py --rttm export/rttm --audio-dir export/audio \\
                    --output hyp.stm --checkpoint best.ckpt

Two details are what make the export an exact re-expression rather than an approximation:

- **The audio is rendered through the loader `infer.py` uses.** `load_cut_audio` already applies
  the featurizer's resampling and the channel selection, so the written wav *is* the tensor the
  cutset route decodes, and the RTTM route's `load_session_audio` reads it back unchanged. That
  round-trip is checked per session unless `--no-verify` is passed.
- **Segment times keep full float precision.** The STNO mask is rasterized with
  `int(t * frame_rate)` on 80 ms frames, so rounding a segment edge can move it into the
  neighbouring frame: `%.3f` changes the mask on ~90% of NOTSOFAR sessions. `--precision full`
  (the default) writes shortest-round-trip floats, which `float()` recovers exactly. A smaller
  `--precision N` gives a conventional-looking RTTM and gives up that guarantee.

One cut per recording spanning the whole recording — the shape of the NOTSOFAR, AMI and LibriMix
evaluation cutsets — is the case that round-trips exactly. A pre-segmented cutset (several cuts
per recording, or a cut starting past 0) is still exported, as the whole recording with its
segments on the recording's timeline, but the RTTM route then decodes each session in one pass
where the cutset route decodes window by window; the two are not expected to agree and the
affected recordings are reported.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer import selected_sessions
from src.data.lhotse_utils import (
    cut_audio_path,
    cut_segments,
    cut_session_id,
    load_cut_audio,
    load_cutset,
    require_supported_cut,
)
from src.data.stno import SpeechSegment
from utils.audio import create_audio_featurizer, load_session_audio

SUBTYPES = {"int16": "PCM_16", "float32": "FLOAT"}

# A cut is taken to span its recording if it starts at 0 and reaches the end. The tolerance
# absorbs the rounding in a duration derived from a sample count.
WHOLE_RECORDING_TOLERANCE = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuts", type=Path, required=True, help="Lhotse CutSet (.jsonl.gz).")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory. Gets audio/{session}.wav and rttm/{session}.rttm, which is the "
        "layout infer.py resolves with no --audio-glob.",
    )
    parser.add_argument("--sessions", default=None, help="Comma-separated subset of session ids.")
    parser.add_argument("--session-list", type=Path, default=None, help="File with one session id per line.")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Target sample rate; must match the model's."
    )
    parser.add_argument(
        "--channel-selector",
        default="average",
        help="Applied while rendering, exactly as in infer.py: 'average' mixes multi-channel audio "
        "down, an integer picks one channel, 'none' leaves it as-is. Use the same value for the "
        "decode.",
    )
    parser.add_argument(
        "--dtype",
        default="int16",
        choices=sorted(SUBTYPES),
        help="Sample format of the written wavs. int16 is bit-exact for 16-bit sources at the "
        "target rate (the usual case, and checked by the verify step); float32 is exact for any "
        "source, at twice the size.",
    )
    parser.add_argument(
        "--precision",
        default="full",
        help="Decimal places for RTTM times, or 'full' (default) for shortest-round-trip floats, "
        "which reproduce the cutset route's mask exactly.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip reading each written wav back and comparing it to the rendered audio.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-export sessions whose wav and RTTM already exist."
    )

    args = parser.parse_args()
    if args.precision != "full":
        try:
            args.precision = int(args.precision)
        except ValueError:
            parser.error(f"--precision takes an integer or 'full', got {args.precision!r}")
        if args.precision < 0:
            parser.error("--precision must not be negative")
    return args


def format_time(value, precision):
    """RTTM time field: shortest round-trip float by default, fixed-point when asked."""
    return repr(float(value)) if precision == "full" else f"{value:.{precision}f}"


def rttm_lines(session_id, segments, precision):
    for segment in sorted(segments, key=lambda s: (s.start, s.speaker)):
        start = format_time(segment.start, precision)
        duration = format_time(segment.duration, precision)
        yield f"SPEAKER {session_id} 1 {start} {duration} <NA> <NA> {segment.speaker} <NA> <NA>\n"


def spans_whole_recording(cut) -> bool:
    """Whether decoding this cut alone is the same as decoding its whole recording.

    A `MixedCut` has no recording — the mixture is all there is to render — so it qualifies as
    long as it starts at 0, which is where `infer.py` would put its words.
    """
    recording = getattr(cut, "recording", None)
    if recording is None:
        return cut.start == 0
    return cut.start == 0 and abs(cut.duration - recording.duration) <= WHOLE_RECORDING_TOLERANCE


def render_session(cuts, featurizer, channel_selector):
    """The audio and the recording-relative segments to export for one session.

    Returns:
        `(audio, segments, equivalent)`. `equivalent` is False when the session had to be widened
        to the whole recording because its cuts do not tile it as one full-length cut, which is
        the case where a single-pass RTTM decode is not the same computation as the cutset route's
        per-cut decode.
    """
    if len(cuts) == 1 and spans_whole_recording(cuts[0]):
        cut = cuts[0]
        return load_cut_audio(cut, featurizer, channel_selector), cut_segments(cut), True

    # Several windows of one recording, or a window that starts past 0: export the recording and
    # put every cut's supervisions on its timeline, the way infer.py merges their words.
    audio = load_session_audio(cut_audio_path(cuts[0]), featurizer, channel_selector)
    segments = [
        SpeechSegment(
            start=segment.start + cut.start, duration=segment.duration, speaker=segment.speaker
        )
        for cut in cuts
        for segment in cut_segments(cut)
    ]
    return audio, segments, False


def main():
    args = parse_args()

    channel_selector = None if args.channel_selector.lower() == "none" else args.channel_selector
    if channel_selector is not None and channel_selector.isdigit():
        channel_selector = int(channel_selector)

    by_session = defaultdict(list)
    for cut in load_cutset(args.cuts):
        require_supported_cut(cut)
        by_session[cut_session_id(cut)].append(cut)
    for cuts in by_session.values():
        cuts.sort(key=lambda cut: cut.start)

    session_ids = selected_sessions(by_session, args.sessions, args.session_list)

    audio_dir = args.output / "audio"
    rttm_dir = args.output / "rttm"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    featurizer = create_audio_featurizer(args.sample_rate)
    print(f"Exporting {len(session_ids)} sessions from {args.cuts} to {args.output}")

    exported = skipped = total_segments = 0
    total_seconds = 0.0
    total_bytes = 0
    widened = []
    unverified = []
    dropped_zero_duration = 0

    for index, session_id in enumerate(session_ids, start=1):
        audio_path = audio_dir / f"{session_id}.wav"
        rttm_path = rttm_dir / f"{session_id}.rttm"
        if not args.force and audio_path.is_file() and rttm_path.is_file():
            skipped += 1
            continue

        audio, segments, equivalent = render_session(by_session[session_id], featurizer, channel_selector)
        if not equivalent:
            widened.append(session_id)

        # The RTTM parser drops non-positive durations, so drop them here too rather than write
        # lines that will not come back.
        kept = [segment for segment in segments if segment.duration > 0]
        dropped_zero_duration += len(segments) - len(kept)

        sf.write(str(audio_path), audio.numpy(), args.sample_rate, subtype=SUBTYPES[args.dtype])
        rttm_path.write_text("".join(rttm_lines(session_id, kept, args.precision)))

        if args.verify:
            reloaded = load_session_audio(audio_path, featurizer, channel_selector)
            if not torch.equal(audio, reloaded):
                deviation = (
                    (audio - reloaded).abs().max().item()
                    if audio.shape == reloaded.shape
                    else float("nan")
                )
                unverified.append((session_id, audio.shape[0], reloaded.shape[0], deviation))

        speakers = {segment.speaker for segment in kept}
        seconds = audio.shape[0] / args.sample_rate
        exported += 1
        total_segments += len(kept)
        total_seconds += seconds
        total_bytes += audio_path.stat().st_size
        print(
            f"[{index}/{len(session_ids)}] {session_id}: {seconds:.1f} s, "
            f"{len(speakers)} speakers, {len(kept)} segments -> {audio_path.name}"
        )

    print(
        f"\nExported {exported} sessions ({total_seconds / 3600:.2f} h, "
        f"{total_bytes / 1e9:.2f} GB) and {total_segments} RTTM segments to {args.output}"
    )
    if skipped:
        print(f"Skipped {skipped} sessions that were already exported (--force to redo them)")
    if dropped_zero_duration:
        print(
            f"Dropped {dropped_zero_duration} segments with a non-positive duration, which an "
            f"RTTM cannot carry; the cutset route keeps them, so a speaker with only such "
            f"segments would be decoded there and not here"
        )
    if widened:
        print(
            f"{len(widened)} sessions are not a single recording-spanning cut and were exported "
            f"as the whole recording: {', '.join(widened[:5])}"
            f"{' ...' if len(widened) > 5 else ''}\n"
            f"  Decoding these from the RTTM is one pass over the recording, where --cuts decodes "
            f"each cut separately, so the two will not agree."
        )
    if args.verify:
        if unverified:
            print(f"\nVERIFY FAILED for {len(unverified)} sessions:")
            for session_id, written, read_back, deviation in unverified[:10]:
                print(
                    f"  {session_id}: {written} samples written, {read_back} read back, "
                    f"max abs diff {deviation:.3e}"
                )
            print("  Try --dtype float32, which is exact for any source.")
            return 1
        if exported:
            print("Verified: every written wav reads back bit-identical to the rendered audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
