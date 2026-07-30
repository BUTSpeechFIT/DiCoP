#!/usr/bin/env python
"""Repoint a manifest's `audio_filepath` entries at a different audio root.

Manifests often outlive the machine they were written on. This rewrites the audio paths in
place-free fashion (a new manifest is written) by looking each session up under a new root,
and reports which sessions could not be resolved rather than failing the whole file.

Examples:
    # AMI, where the audio now lives at {root}/{session}/audio.wav
    python scripts/retarget_manifest.py \\
        --manifest /home/jovyan/data/nemo_manifests/ami-sdm_test_sc_cutset.jsonl \\
        --audio-dir /home/jovyan/data/chime9/ami/ami_fixed_matching_lips \\
        --audio-glob '{session}/audio.wav' \\
        --session-from-parent-of-parent \\
        --output /tmp/dicop_smoke/ami_test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.audio import DEFAULT_AUDIO_EXTENSIONS, resolve_audio_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest (JSON Lines).")
    parser.add_argument("--audio-dir", type=Path, required=True, help="New audio root to search.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest path.")
    parser.add_argument(
        "--audio-glob",
        default=None,
        help="Glob relative to --audio-dir with a {session} placeholder, "
        "e.g. '{session}/audio.wav'. Tried after the plain {session}{ext} lookups.",
    )
    parser.add_argument(
        "--audio-ext",
        default=",".join(DEFAULT_AUDIO_EXTENSIONS),
        help="Comma-separated audio extensions to try.",
    )
    parser.add_argument(
        "--session-from-parent-of-parent",
        action="store_true",
        help="Derive the session id from the grandparent directory of the old path "
        "(AMI: .../IS1009c/audio/IS1009c.Array1-01.wav -> IS1009c) instead of the file stem.",
    )
    parser.add_argument(
        "--set-session-id",
        action="store_true",
        help="Also write the derived session id into a `session_id` field.",
    )
    parser.add_argument(
        "--keep-unresolved",
        action="store_true",
        help="Keep sessions whose audio was not found, leaving their path unchanged.",
    )
    return parser.parse_args()


def derive_session_id(old_path: str, from_grandparent: bool) -> str:
    path = Path(old_path)
    if from_grandparent:
        # .../<session>/audio/<session>.Array1-01.wav
        return path.parent.parent.name
    return path.stem


def main():
    args = parse_args()
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.audio_ext.split(",") if ext]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Segmented manifests repeat each session on many lines, and the recursive-glob tier of
    # resolution walks the whole audio tree, so resolve each session id only once.
    resolution_cache = {}

    def resolve_cached(session_id):
        if session_id not in resolution_cache:
            try:
                resolution_cache[session_id] = resolve_audio_path(
                    session_id, args.audio_dir, extensions=extensions, glob_template=args.audio_glob
                )
            except FileNotFoundError:
                resolution_cache[session_id] = None
        return resolution_cache[session_id]

    kept, dropped = 0, 0
    with args.manifest.open() as source, args.output.open("w") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            session_id = derive_session_id(entry["audio_filepath"], args.session_from_parent_of_parent)

            audio_path = resolve_cached(session_id)
            if audio_path is None:
                if not args.keep_unresolved:
                    dropped += 1
                    continue
            else:
                entry["audio_filepath"] = str(audio_path)

            if args.set_session_id:
                entry["session_id"] = session_id

            sink.write(json.dumps(entry) + "\n")
            kept += 1

    unresolved = sorted(sid for sid, path in resolution_cache.items() if path is None)
    resolved = len(resolution_cache) - len(unresolved)
    print(f"Resolved {resolved}/{len(resolution_cache)} sessions; wrote {kept} lines to {args.output}")
    if unresolved:
        action = "kept with original paths" if args.keep_unresolved else f"{dropped} lines dropped"
        print(f"Could not resolve {len(unresolved)} sessions ({action}): {', '.join(unresolved)}")


if __name__ == "__main__":
    main()
