#!/usr/bin/env python
"""Write a reference STM from a DiCoP manifest, for scoring `infer.py` output.

Pairs with `manifest_to_rttm.py`: the two derive the reference transcript and the oracle
diarization from the same manifest, so

    meeteval-wer cpwer  -r ref.stm -h hyp.stm
    meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar 5

reproduces what training-time validation reports.

Times are shifted by each row's `offset` onto the recording's absolute timeline, matching
`manifest_to_rttm.py` and the training-time metric. Text normalization is applied by default so
the reference matches what the model is trained and scored against.

    python scripts/manifest_to_stm.py --manifest manifest.jsonl --output ref.stm
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import meeteval
from meeteval.io.seglst import SegLST, SegLstSegment

from src.data.text_norm import get_text_norm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest (JSON Lines).")
    parser.add_argument("--output", type=Path, required=True, help="Output .stm path.")
    parser.add_argument(
        "--text-norm",
        default="whisper_nsf",
        help="Text normalizer to apply, or 'none'. Must match what infer.py / training used.",
    )
    parser.add_argument(
        "--session-id-from",
        default="auto",
        choices=["auto", "field", "stem"],
        help="Where the session id comes from; see manifest_to_rttm.py.",
    )
    return parser.parse_args()


def session_id_for(entry, mode):
    stem = Path(entry["audio_filepath"]).stem
    if mode == "stem":
        return stem
    if mode == "field":
        if "session_id" not in entry:
            raise KeyError(f"No `session_id` field for {entry['audio_filepath']}")
        return entry["session_id"]
    return entry.get("session_id") or stem


def main():
    args = parse_args()
    text_norm = (lambda x: x) if args.text_norm.lower() == "none" else get_text_norm(args.text_norm)

    per_session = defaultdict(list)
    skipped_empty = 0

    with args.manifest.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            session_id = session_id_for(entry, args.session_id_from)
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

    segments = []
    for session_id in sorted(per_session):
        segments.extend(sorted(per_session[session_id], key=lambda s: (s["start_time"], s["speaker"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meeteval.io.dump(SegLST(segments=segments), str(args.output))

    print(f"Wrote {len(segments)} segments across {len(per_session)} sessions to {args.output}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} segments that normalized to empty text")


if __name__ == "__main__":
    main()
