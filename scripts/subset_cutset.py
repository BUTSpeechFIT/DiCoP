#!/usr/bin/env python
"""Take a smaller, evenly spread CutSet out of a larger one.

Written for validation during training. A DiCoP validation epoch decodes every (cut, speaker)
pair one at a time, so a full dev set is expensive to run every epoch -- Libri2Mix dev-clean is
3000 cuts, i.e. 6000 greedy decodes. `trainer.limit_val_batches` cannot shorten it: the
reference side is built from the whole manifest regardless of what was decoded, so a truncated
pass leaves most sessions without a hypothesis and meeteval refuses to score. Shrinking the
*cutset* is the way to shorten validation.

Selection is a fixed stride rather than the first N cuts, because cutsets tend to be written in
recipe order -- LibriMix follows its metadata CSV -- so the head of the file is not a
representative sample. The stride is deterministic, so the same input and `--num` always give
the same subset and metrics stay comparable across runs.

    python scripts/subset_cutset.py --cuts dev.jsonl.gz --output dev300.jsonl.gz --num 300
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lhotse_utils import load_cutset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuts", type=Path, required=True, help="Input Lhotse CutSet (.jsonl.gz / .jsonl).")
    parser.add_argument("--output", type=Path, required=True, help="Output CutSet path.")
    parser.add_argument("--num", type=int, required=True, help="Number of cuts to keep.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output.")

    args = parser.parse_args()
    if args.num <= 0:
        parser.error(f"--num must be positive, got {args.num}")
    return args


def main():
    args = parse_args()

    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        print(f"{args.output} already exists; not rebuilding it (--force to overwrite)")
        return

    from lhotse import CutSet

    cuts = list(load_cutset(args.cuts))
    if not cuts:
        raise SystemExit(f"{args.cuts} contains no cuts.")

    # A stride of at least 1, then truncate: `len // num` overshoots slightly when the two do not
    # divide evenly, which costs a few cuts off the end rather than returning more than asked for.
    stride = max(1, len(cuts) // args.num)
    kept = cuts[::stride][: args.num]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(kept).to_file(args.output)

    print(f"Wrote {len(kept)} of {len(cuts)} cuts (every {stride}) to {args.output}")


if __name__ == "__main__":
    main()
