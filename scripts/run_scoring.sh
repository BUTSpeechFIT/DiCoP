#!/usr/bin/env bash
#
# Export a reference STM for every decoded set and score it with meeteval.
#
#   scripts/run_scoring.sh
#   scripts/run_scoring.sh --datasets ami,notsofar
#   scripts/run_scoring.sh --decode-dir exps/decode --no-score
#
# Every dataset directory left behind by run_inference.sh -- {name}/{hyp.stm,infer.log} -- gets a
# `ref.stm` beside its `hyp.stm`, plus meeteval's `hyp_cpwer.json` and `hyp_tcpwer.json`. A
# cpWER/tcpWER table over all of them is printed at the end. A set that fails is reported there
# rather than aborting the rest.
#
# The reference is built from the manifest named in the run's own `infer.log`, not from a table
# kept here. That is deliberate: the reference can then only ever come from the file that was
# actually decoded, so session ids and timelines match `hyp.stm` by construction -- which the
# paired NeMo manifests would not, since they carry no `session_id` and several are empty.
#
# The RTTM route logs its RTTM directory instead of a manifest, so those sets cannot be resolved
# from the log; point them at their NeMo manifest with `--manifest NAME=PATH`. AliMeeting and
# AIShell-4 are Chinese, so they also want `--text-norm none`.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DECODE_DIR="${DECODE_DIR:-$REPO_ROOT/exps/decode-local}"
PYTHON="${PYTHON:-python}"
MEETEVAL="${MEETEVAL:-meeteval-wer}"

SELECTION="all"
TEXT_NORM="whisper_nsf"
COLLAR=5
SCORE=1
FORCE=0
DRY_RUN=0
LIST_ONLY=0
declare -A MANIFEST_OVERRIDE=()

usage() {
  cat <<EOF
Export reference STMs for the decoded sets and score them with meeteval.

Usage: $(basename "$0") [options]

Options:
  --decode-dir DIR      Directory of {name}/hyp.stm runs. Default: $DECODE_DIR
  --datasets LIST       Comma-separated names or name prefixes, or 'all'. Default: $SELECTION
  --text-norm NAME      Normalizer for the reference, or 'none'. Default: $TEXT_NORM
  --collar N            tcpWER collar, in seconds. Default: $COLLAR
  --manifest NAME=PATH  Source manifest for one set, when infer.log does not name one.
                        Repeatable.
  --no-score            Only write ref.stm; skip meeteval.
  --force               Rebuild a ref.stm that already exists.
  --dry-run             Print the commands without running them.
  --list                Show the decoded sets and their source manifests, then exit.
  -h, --help            This message.

Scoring reproduces the README recipe, which is what training-time validation reports:

  meeteval-wer cpwer  -r ref.stm -h hyp.stm
  meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar $COLLAR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --decode-dir) DECODE_DIR="$2"; shift 2 ;;
    --datasets) SELECTION="$2"; shift 2 ;;
    --text-norm) TEXT_NORM="$2"; shift 2 ;;
    --collar) COLLAR="$2"; shift 2 ;;
    --manifest)
      [[ "$2" == *=* ]] || { echo "--manifest takes NAME=PATH, got: $2" >&2; exit 2; }
      MANIFEST_OVERRIDE["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --no-score) SCORE=0; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# A set is selected by its exact name, by a prefix of its name up to a '-', or by 'all'.
is_selected() {
  local name="$1" token
  IFS=',' read -r -a tokens <<<"$SELECTION"
  for token in "${tokens[@]}"; do
    [[ -z "$token" ]] && continue
    [[ "$token" == "all" || "$name" == "$token" || "$name" == "$token"-* ]] && return 0
  done
  return 1
}

# infer.py logs the cutset it decoded: "Decoding 18 cuts across 18 sessions from <path>". The
# RTTM route logs its RTTM directory instead, which is not a manifest, so it deliberately does
# not match here and the set is left to --manifest.
manifest_from_log() {
  local log="$1"
  [[ -f "$log" ]] || return 0
  sed -n 's/.*Decoding [0-9]* cuts across [0-9]* sessions from \(.*\)$/\1/p' "$log" | tail -1
}

manifest_for() {
  local name="$1"
  if [[ -n "${MANIFEST_OVERRIDE[$name]:-}" ]]; then
    echo "${MANIFEST_OVERRIDE[$name]}"
    return 0
  fi
  manifest_from_log "$DECODE_DIR/$name/infer.log"
}

# meeteval writes its average to {parent}/{stem}_cpwer.json; pull the headline number out as a
# percentage for the summary table.
error_rate() {
  local path="$1"
  [[ -f "$path" ]] || { echo "n/a"; return 0; }
  "$PYTHON" -c 'import json, sys; rate = json.load(open(sys.argv[1]))["error_rate"]; print(f"{rate * 100:.2f}")' \
    "$path" 2>/dev/null || echo "n/a"
}

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

if [[ ! -d "$DECODE_DIR" ]]; then
  echo "Decode directory not found: $DECODE_DIR" >&2
  exit 2
fi
if [[ $SCORE -eq 1 && $DRY_RUN -eq 0 ]] && ! command -v "$MEETEVAL" >/dev/null; then
  echo "$MEETEVAL is not on PATH; install meeteval or pass --no-score." >&2
  exit 2
fi

selected=()
for dir in "$DECODE_DIR"/*/; do
  name="$(basename "$dir")"
  # A directory without a hypothesis is not a finished decode.
  [[ -s "$dir/hyp.stm" ]] || continue
  is_selected "$name" && selected+=("$name")
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No decoded set under $DECODE_DIR matches --datasets '$SELECTION'." >&2
  exit 2
fi

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%-32s %-8s %s\n' NAME REF MANIFEST
  for name in "${selected[@]}"; do
    manifest="$(manifest_for "$name")"
    ref_state=missing
    [[ -s "$DECODE_DIR/$name/ref.stm" ]] && ref_state=present
    printf '%-32s %-8s %s\n' "$name" "$ref_state" "${manifest:-<unresolved>}"
  done
  exit 0
fi

echo "Decode dir : $DECODE_DIR"
echo "Text norm  : $TEXT_NORM"
echo "Datasets   : ${#selected[@]} selected from '$SELECTION'"
echo

exported=()
skipped=()
failed=()
rows=()

index=0
for name in "${selected[@]}"; do
  index=$((index + 1))
  dir="$DECODE_DIR/$name"
  hyp="$dir/hyp.stm"
  ref="$dir/ref.stm"

  echo "[$index/${#selected[@]}] $name"

  manifest="$(manifest_for "$name")"
  if [[ -z "$manifest" ]]; then
    echo "  skip: no source manifest in $dir/infer.log (RTTM route? pass --manifest $name=PATH)"
    skipped+=("$name")
    continue
  fi
  if [[ ! -s "$manifest" ]]; then
    echo "  skip: manifest missing or empty ($manifest)"
    skipped+=("$name")
    continue
  fi

  if [[ -s "$ref" && $FORCE -eq 0 ]]; then
    echo "  ref.stm already exists (--force to rebuild)"
  else
    echo "  reference from $manifest"
    if ! run "$PYTHON" "$REPO_ROOT/scripts/manifest_to_stm.py" \
        --manifest "$manifest" --output "$ref" --text-norm "$TEXT_NORM"; then
      echo "  FAILED: could not build the reference"
      failed+=("$name")
      continue
    fi
    exported+=("$name")
  fi

  if [[ $SCORE -eq 0 ]]; then
    continue
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    run "$MEETEVAL" cpwer -r "$ref" -h "$hyp"
    run "$MEETEVAL" tcpwer -r "$ref" -h "$hyp" --collar "$COLLAR"
    continue
  fi

  scored=1
  "$MEETEVAL" cpwer -r "$ref" -h "$hyp" || scored=0
  "$MEETEVAL" tcpwer -r "$ref" -h "$hyp" --collar "$COLLAR" || scored=0
  if [[ $scored -eq 0 ]]; then
    echo "  FAILED: meeteval could not score $name"
    failed+=("$name")
    continue
  fi

  rows+=("$(printf '%-32s %8s %8s' "$name" "$(error_rate "$dir/hyp_cpwer.json")" \
    "$(error_rate "$dir/hyp_tcpwer.json")")")
  echo
done

echo "==================== summary ===================="
if [[ ${#rows[@]} -gt 0 ]]; then
  printf '%-32s %8s %8s\n' DATASET cpWER tcpWER
  printf '%s\n' "${rows[@]}"
  echo
fi
echo "exported: ${#exported[@]} ${exported[*]+(${exported[*]})}"
echo "skipped : ${#skipped[@]} ${skipped[*]+(${skipped[*]})}"
echo "failed  : ${#failed[@]} ${failed[*]+(${failed[*]})}"
[[ ${#failed[@]} -eq 0 ]]
