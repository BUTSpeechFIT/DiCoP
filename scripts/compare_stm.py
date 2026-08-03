#!/usr/bin/env python
"""Compare two STM files line by line, and say how much any difference is worth.

Written for the RTTM-vs-CutSet parity check — the same recordings decoded through
`infer.py --cuts` and through `infer.py --rttm --audio-dir` on an export of that cutset should
produce the same hypothesis — but nothing here is specific to that: any two STMs over the same
sessions can be compared.

    python scripts/compare_stm.py a.stm b.stm

Equality is exact by default, because for the parity check anything else would hide the thing
being tested. `--time-tolerance` allows a per-word timing slack, `--ignore-times` compares only
the words. When the two differ, the report names the sessions and speakers involved, shows the
first differing lines, and gives the cpWER of B against A so a difference of a few words is not
read as a difference of a few hundred.

Exit status is 0 when the two agree.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import meeteval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="First STM, treated as the reference when scoring.")
    parser.add_argument("b", type=Path, help="Second STM.")
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.0,
        help="Seconds of slack allowed on a word's start and end. Default 0, i.e. exact.",
    )
    parser.add_argument("--ignore-times", action="store_true", help="Compare the words only.")
    parser.add_argument(
        "--max-examples", type=int, default=10, help="Differing lines to print. Default 10."
    )
    parser.add_argument(
        "--no-cpwer",
        dest="cpwer",
        action="store_false",
        help="Skip scoring B against A, which is otherwise reported when the two differ.",
    )
    return parser.parse_args()


def by_speaker(seglst):
    """`{(session_id, speaker): [segment, ...]}`, keeping each STM's own line order."""
    grouped = defaultdict(list)
    for segment in seglst:
        grouped[(segment["session_id"], segment["speaker"])].append(segment)
    return grouped


def describe(segment):
    return (
        f"{segment['session_id']} {segment['speaker']} "
        f"{float(segment['start_time']):.2f}-{float(segment['end_time']):.2f} {segment['words']!r}"
    )


def segments_differ(left, right, time_tolerance, ignore_times):
    if left["words"] != right["words"]:
        return True
    if ignore_times:
        return False
    return (
        abs(float(left["start_time"]) - float(right["start_time"])) > time_tolerance
        or abs(float(left["end_time"]) - float(right["end_time"])) > time_tolerance
    )


def main():
    args = parse_args()

    a = meeteval.io.load(args.a).to_seglst()
    b = meeteval.io.load(args.b).to_seglst()
    print(f"A: {len(a)} lines  {args.a}")
    print(f"B: {len(b)} lines  {args.b}")

    grouped_a, grouped_b = by_speaker(a), by_speaker(b)
    sessions_a = {session for session, _ in grouped_a}
    sessions_b = {session for session, _ in grouped_b}

    only_a = sorted(sessions_a - sessions_b)
    only_b = sorted(sessions_b - sessions_a)
    keys_only_a = sorted(set(grouped_a) - set(grouped_b))
    keys_only_b = sorted(set(grouped_b) - set(grouped_a))

    examples = []
    differing_lines = 0
    differing_sessions = set()

    for key in sorted(set(grouped_a) & set(grouped_b)):
        left, right = grouped_a[key], grouped_b[key]
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                differing_lines += 1
                differing_sessions.add(key[0])
                missing_from, present = ("A", right[index]) if index >= len(left) else ("B", left[index])
                if len(examples) < args.max_examples:
                    examples.append(f"  only in {'B' if missing_from == 'A' else 'A'}: {describe(present)}")
                continue
            if segments_differ(left[index], right[index], args.time_tolerance, args.ignore_times):
                differing_lines += 1
                differing_sessions.add(key[0])
                if len(examples) < args.max_examples:
                    examples.append(f"  A: {describe(left[index])}\n  B: {describe(right[index])}")

    for key in keys_only_a:
        differing_lines += len(grouped_a[key])
        differing_sessions.add(key[0])
    for key in keys_only_b:
        differing_lines += len(grouped_b[key])
        differing_sessions.add(key[0])

    print(f"\nSessions: {len(sessions_a)} in A, {len(sessions_b)} in B, {len(sessions_a & sessions_b)} shared")
    if only_a:
        print(f"  only in A ({len(only_a)}): {', '.join(only_a[:5])}{' ...' if len(only_a) > 5 else ''}")
    if only_b:
        print(f"  only in B ({len(only_b)}): {', '.join(only_b[:5])}{' ...' if len(only_b) > 5 else ''}")
    for label, keys in (("A", keys_only_a), ("B", keys_only_b)):
        if keys:
            shown = ', '.join(f"{session}/{speaker}" for session, speaker in keys[:5])
            print(f"  (session, speaker) pairs only in {label} ({len(keys)}): {shown}"
                  f"{' ...' if len(keys) > 5 else ''}")

    if differing_lines == 0 and not only_a and not only_b:
        criterion = "words" if args.ignore_times else f"words and times (tolerance {args.time_tolerance})"
        print(f"\nIDENTICAL: {len(a)} lines agree on {criterion}")
        return 0

    print(f"\nDIFFERENT: {differing_lines} lines across {len(differing_sessions)} sessions")
    if examples:
        print("\n".join(examples))
        if differing_lines > len(examples):
            print(f"  ... and more (--max-examples {args.max_examples})")

    if args.cpwer:
        # A supplementary number, so a scorer that refuses the input (invalid timings, say)
        # must not cost us the line-level report above.
        try:
            error_rates = meeteval.wer.cpwer(reference=a, hypothesis=b)
        except Exception as exc:
            print(f"\ncpWER of B against A: not scorable ({type(exc).__name__}: {exc})")
            return 1
        overall = meeteval.wer.combine_error_rates(error_rates)
        print(
            f"\ncpWER of B against A: {overall.error_rate * 100:.4f}% "
            f"({overall.errors} errors over {overall.length} words)"
        )
        worst = sorted(error_rates.items(), key=lambda item: -item[1].errors)[:5]
        for session_id, rate in worst:
            if rate.errors:
                print(f"  {session_id}: {rate.errors} errors / {rate.length} words")
    return 1


if __name__ == "__main__":
    sys.exit(main())
