#!/usr/bin/env python
"""Write a reference STM from a DiCoP manifest, for scoring `infer.py` output.

Reads either manifest format DiCoP takes: a NeMo JSON Lines manifest, or a Lhotse CutSet
(`.jsonl.gz`). Build the reference from the same file the decode was run on, and the session ids
come out the way `infer.py` wrote them, so the two STMs line up without any renaming.

Pairs with `manifest_to_rttm.py`: the two derive the reference transcript and the oracle
diarization from the same manifest, so

    meeteval-wer cpwer  -r ref.stm -h hyp.stm
    meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar 5

reproduces what training-time validation reports.

Times are put on the recording's absolute timeline -- shifted by each row's `offset`, or by
`cut.start` for a cutset -- matching `manifest_to_rttm.py` and the training-time metric. Text
normalization is applied by default so the reference matches what the model is trained and
scored against.

    python scripts/manifest_to_stm.py --manifest manifest.jsonl --output ref.stm
    python scripts/manifest_to_stm.py --manifest cuts.jsonl.gz --output ref.stm
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meeteval.io.seglst import SegLstSegment

from src.data.text_norm import get_text_norm
from utils.stm import write_stm

# Mirrors LHOTSE_SUFFIXES in src/data/dataloader.py; kept local because importing that module
# pulls in NeMo, which this script has no other use for.
LHOTSE_SUFFIXES = ('.jsonl.gz', '.jsonl.gzip')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Input manifest: NeMo JSON Lines, or a Lhotse CutSet (.jsonl.gz).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .stm path.")
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "nemo", "lhotse"],
        help="Input format. 'auto' reads a .jsonl.gz as a CutSet and anything else as a NeMo manifest.",
    )
    parser.add_argument(
        "--text-norm",
        default="whisper_nsf",
        help="Text normalizer to apply, or 'none'. Must match what infer.py / training used.",
    )
    parser.add_argument(
        "--session-id-from",
        default="auto",
        choices=["auto", "field", "stem"],
        help="Where the session id comes from; see manifest_to_rttm.py. NeMo manifests only -- "
        "a cutset always keys on its recording id.",
    )

    args = parser.parse_args()
    if is_lhotse(args.manifest, args.format) and args.session_id_from != "auto":
        parser.error("--session-id-from applies to NeMo manifests only; a cutset keys on its recording id.")
    return args


def is_lhotse(manifest: Path, fmt: str) -> bool:
    if fmt != "auto":
        return fmt == "lhotse"
    return str(manifest).endswith(LHOTSE_SUFFIXES)


def session_id_for(entry, mode):
    stem = Path(entry["audio_filepath"]).stem
    if mode == "stem":
        return stem
    if mode == "field":
        if "session_id" not in entry:
            raise KeyError(f"No `session_id` field for {entry['audio_filepath']}")
        return entry["session_id"]
    return entry.get("session_id") or stem


def nemo_sessions(manifest: Path, text_norm, session_id_mode: str):
    """Reference segments from a NeMo JSON Lines manifest, keyed by session id."""
    per_session = defaultdict(list)
    skipped_empty = 0

    with manifest.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            session_id = session_id_for(entry, session_id_mode)
            offset = entry.get("offset") or 0.0

            for segment in entry.get("text") or []:
                words = text_norm(segment["text"])
                if not words.strip():
                    # Normalization can empty a segment out; the training dataset drops these too.
                    skipped_empty += 1
                    continue
                start = offset + float(segment["start"])
                per_session[session_id].append(
                    SegLstSegment(
                        session_id=session_id,
                        speaker=segment["speaker"],
                        words=words,
                        start_time=start,
                        end_time=start + float(segment["duration"]),
                    )
                )

    return per_session, skipped_empty


def lhotse_sessions(manifest: Path, text_norm):
    """Reference segments from a Lhotse CutSet, keyed and timed exactly as `infer.py` writes them.

    The session is the cut's recording id, and supervision times are cut-relative, so `cut.start`
    puts them on the recording's timeline. A pre-segmented cutset's windows therefore compose into
    one session rather than one pseudo-session each, and a `MixedCut` -- which has no recording and
    starts at 0 -- keys on its own id, which is how LibriMix and LibriSpeechMix sessions are named.
    """
    # Imported here so the NeMo path above does not pay for the torch import.
    from src.data.lhotse_utils import cut_session_id, load_cutset

    per_session = defaultdict(list)
    skipped_empty = 0

    for cut in load_cutset(manifest):
        session_id = cut_session_id(cut)
        for supervision in cut.supervisions:
            words = text_norm(supervision.text or "")
            if not words.strip():
                skipped_empty += 1
                continue
            start = cut.start + supervision.start
            per_session[session_id].append(
                SegLstSegment(
                    session_id=session_id,
                    speaker=supervision.speaker,
                    words=words,
                    start_time=start,
                    end_time=start + supervision.duration,
                )
            )

    return per_session, skipped_empty


def main():
    args = parse_args()
    text_norm = (lambda x: x) if args.text_norm.lower() == "none" else get_text_norm(args.text_norm)

    if is_lhotse(args.manifest, args.format):
        per_session, skipped_empty = lhotse_sessions(args.manifest, text_norm)
    else:
        per_session, skipped_empty = nemo_sessions(args.manifest, text_norm, args.session_id_from)

    session_segments = {
        session_id: sorted(segments, key=lambda s: (s["start_time"], s["speaker"]))
        for session_id, segments in per_session.items()
    }
    num_lines = write_stm(args.output, session_segments)

    print(f"Wrote {num_lines} segments across {len(session_segments)} sessions to {args.output}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} segments that normalized to empty text")


if __name__ == "__main__":
    main()
