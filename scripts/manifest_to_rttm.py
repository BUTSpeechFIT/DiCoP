#!/usr/bin/env python
"""Derive an oracle RTTM from a DiCoP manifest.

Turns the manifest's diarized `text` segments into the RTTM that `infer.py` consumes. With
this you can run the inference path on a manifest you already have references for, which is
both the end-to-end smoke test and the oracle-diarization control: the mask `infer.py` builds
from this RTTM is the same one training builds from the manifest.

Segment `start` values are relative to the row's `offset`, so the offset is added to put
everything on the recording's absolute timeline. Rows sharing a recording therefore merge into
one RTTM session, which is what full-session decoding expects.

    python scripts/manifest_to_rttm.py --manifest manifest.jsonl --output rttms/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest (JSON Lines).")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path. A directory (or extension-less path) gets one {session}.rttm per "
        "session; a path ending in .rttm gets a single combined file.",
    )
    parser.add_argument(
        "--session-id-from",
        default="auto",
        choices=["auto", "field", "stem"],
        help="Where the session id comes from: the manifest `session_id` field, the audio file "
        "stem, or 'auto' (the field when present, else the stem).",
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

    sessions = defaultdict(list)
    with args.manifest.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            session_id = session_id_for(entry, args.session_id_from)
            offset = entry.get("offset") or 0.0

            for segment in entry.get("text") or []:
                duration = float(segment["duration"])
                if duration <= 0:
                    continue
                sessions[session_id].append(
                    (offset + float(segment["start"]), duration, segment["speaker"])
                )

    for session_id in sessions:
        sessions[session_id].sort(key=lambda item: (item[0], item[2]))

    def rttm_lines(session_id):
        for start, duration, speaker in sessions[session_id]:
            yield (
                f"SPEAKER {session_id} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    combined = args.output.suffix == ".rttm"
    if combined:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as sink:
            for session_id in sorted(sessions):
                sink.writelines(rttm_lines(session_id))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for session_id in sorted(sessions):
            with (args.output / f"{session_id}.rttm").open("w") as sink:
                sink.writelines(rttm_lines(session_id))

    total = sum(len(v) for v in sessions.values())
    where = args.output if combined else f"{args.output}/{{session}}.rttm"
    print(f"Wrote {total} segments across {len(sessions)} sessions to {where}")


if __name__ == "__main__":
    main()
