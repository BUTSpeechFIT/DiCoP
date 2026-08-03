#!/usr/bin/env bash
#
# Check that decoding an exported cutset through the RTTM route reproduces decoding the cutset.
#
#   scripts/run_rttm_parity.sh --checkpoint /path/to/best.ckpt
#   scripts/run_rttm_parity.sh --checkpoint best.ckpt --datasets ami-sdm-dev
#   scripts/run_rttm_parity.sh --checkpoint best.ckpt --sessions sdm_MTG_30860_sc_meetup_0
#
# `infer.py` takes its diarization and audio either from a Lhotse CutSet (--cuts) or from an RTTM
# plus an audio directory (--rttm --audio-dir). Only the first is exercised by run_inference.sh,
# yet the second is the route a real diarizer's output takes. For each selected set this script:
#
#   1. exports the cutset to wavs and RTTMs   (scripts/cutset_to_wav_rttm.py)
#   2. decodes that export with --rttm        -> {name}/hyp_rttm.stm
#   3. takes the --cuts hypothesis from --cutset-decode-dir, or decodes one
#   4. compares the two                       (scripts/compare_stm.py)
#
# The two hypotheses are expected to be *identical*, not merely close: the export renders its
# audio through the same loader infer.py uses and writes RTTM times at full float precision, so
# both routes see the same samples and the same STNO mask. A set whose cutset is pre-segmented
# (several cuts per recording) is exported as whole recordings and will not match -- the export
# step says so.
#
# Exported audio is roughly the size of the corpus (~4 GB for the two NOTSOFAR sets); `--clean`
# deletes it again once a set has compared equal.
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_ROOT="${MANIFEST_ROOT:-/home/jovyan/mt-asr-data-prep/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/exps/parity}"
CUTSET_DECODE_DIR="${CUTSET_DECODE_DIR:-$REPO_ROOT/exps/decode-local}"
PYTHON="${PYTHON:-python}"

CHECKPOINT=""
DEVICE="auto"
SELECTION="notsofar-sdm-dev1,notsofar-sdm-eval"
SESSIONS=""
FORCE=0
FORCE_CUTSET=0
CLEAN=0
DRY_RUN=0
LIST_ONLY=0
EXTRA_ARGS=()

# name|cutset under $MANIFEST_ROOT. The names match run_inference.sh, so a cutset hypothesis it
# already produced is found under --cutset-decode-dir by name.
DATASETS=(
  "notsofar-sdm-dev1|notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl.gz"
  "notsofar-sdm-eval|notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl.gz"
  "ami-sdm-dev|ami/ami-sdm_cutset_dev.jsonl.gz"
  "ami-sdm-test|ami/ami-sdm_cutset_test.jsonl.gz"
  "ami-ihm-mix-dev|ami/ami-ihm-mix_cutset_dev.jsonl.gz"
  "ami-ihm-mix-test|ami/ami-ihm-mix_cutset_test.jsonl.gz"
  "librimix-2mix-dev-clean|librimix/librimix_cutset_libri2mix_dev-clean.jsonl.gz"
  "librimix-2mix-test-clean|librimix/librimix_cutset_libri2mix_test-clean.jsonl.gz"
  "librispeechmix-dev-clean-2mix|librispeechmix/librispeechmix_cutset_dev-clean-2mix.jsonl.gz"
  "librispeechmix-test-clean-2mix|librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz"
)

usage() {
  cat <<EOF
Check the RTTM route against the CutSet route on an export of the same cutset.

Usage: $(basename "$0") --checkpoint PATH [options] [-- <extra infer.py args>]

Options:
  --checkpoint PATH       .ckpt / .nemo / NGC / HF model id. Required.
  --datasets LIST         Comma-separated names or name prefixes, or 'all'.
                          Default: $SELECTION
  --output-dir DIR        Where {name}/ goes. Default: $OUTPUT_DIR
  --cutset-decode-dir DIR Where an existing {name}/hyp.stm from the --cuts route is looked for.
                          Default: $CUTSET_DECODE_DIR
  --manifest-root DIR     mt-asr-data-prep manifests. Default: $MANIFEST_ROOT
  --device SPEC           auto, cpu, cuda, cuda:N. Default: $DEVICE
  --sessions LIST         Comma-separated session ids; both routes decode just those. Useful for
                          a quick check before committing to a whole set. Implies
                          --force-cutset-decode, since a stored hypothesis covers the whole set.
  --force                 Redo the export and the RTTM decode even if they exist.
  --force-cutset-decode   Decode the --cuts side here instead of reusing a stored hypothesis.
  --clean                 Delete the exported audio once a set has compared equal.
  --dry-run               Print the commands without running them.
  --list                  Show the known datasets and exit.
  -h, --help              This message.

Everything after '--' is appended to both infer.py calls, so the two routes stay comparable.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --datasets) SELECTION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cutset-decode-dir) CUTSET_DECODE_DIR="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --force-cutset-decode) FORCE_CUTSET=1; shift ;;
    --clean) CLEAN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# A row is selected by its exact name, by a prefix of its name up to a '-', or by 'all'.
is_selected() {
  local name="$1" token
  IFS=',' read -r -a tokens <<<"$SELECTION"
  for token in "${tokens[@]}"; do
    [[ -z "$token" ]] && continue
    [[ "$token" == "all" || "$name" == "$token" || "$name" == "$token"-* ]] && return 0
  done
  return 1
}

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%-32s %-8s %-8s %s\n' NAME CUTSET CUTS-HYP MANIFEST
  for row in "${DATASETS[@]}"; do
    IFS='|' read -r name manifest <<<"$row"
    state=missing
    [[ -s "$MANIFEST_ROOT/$manifest" ]] && state=ok
    hyp=missing
    [[ -s "$CUTSET_DECODE_DIR/$name/hyp.stm" ]] && hyp=ok
    printf '%-32s %-8s %-8s %s\n' "$name" "$state" "$hyp" "$manifest"
  done
  echo
  echo "Default selection: $SELECTION  (use --datasets all for every row above)"
  exit 0
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required." >&2
  usage >&2
  exit 2
fi
if [[ "$CHECKPOINT" == *.ckpt || "$CHECKPOINT" == *.nemo ]] && [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

# A stored cutset hypothesis covers the whole set, so it cannot be compared against a decode of
# a few sessions; decode the cutset side here instead.
if [[ -n "$SESSIONS" && $FORCE_CUTSET -eq 0 ]]; then
  FORCE_CUTSET=1
  echo "--sessions given: decoding the cutset side here rather than reusing a whole-set hypothesis."
fi

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

# Run a command, showing it live and keeping a log; the command's own status is returned.
run_logged() {
  local log="$1"; shift
  if [[ $DRY_RUN -eq 1 ]]; then
    run "$@"
    return 0
  fi
  "$@" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

selected=()
for row in "${DATASETS[@]}"; do
  IFS='|' read -r name _ <<<"$row"
  is_selected "$name" && selected+=("$row")
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No dataset matches --datasets '$SELECTION'. Try --list." >&2
  exit 2
fi

echo "Checkpoint  : $CHECKPOINT"
echo "Output dir  : $OUTPUT_DIR"
echo "Cuts hyp dir: $CUTSET_DECODE_DIR"
echo "Datasets    : ${#selected[@]} selected from '$SELECTION'"
[[ -n "$SESSIONS" ]] && echo "Sessions    : $SESSIONS"
echo

identical=()
different=()
failed=()
rows=()

index=0
for row in "${selected[@]}"; do
  IFS='|' read -r name manifest <<<"$row"
  index=$((index + 1))

  cutset="$MANIFEST_ROOT/$manifest"
  out_dir="$OUTPUT_DIR/$name"
  export_dir="$out_dir/export"
  hyp_rttm="$out_dir/hyp_rttm.stm"
  hyp_cuts="$CUTSET_DECODE_DIR/$name/hyp.stm"

  echo "[$index/${#selected[@]}] $name"

  if [[ ! -s "$cutset" ]]; then
    echo "  FAILED: cutset missing or empty ($cutset)"
    failed+=("$name")
    continue
  fi

  [[ $DRY_RUN -eq 0 ]] && mkdir -p "$out_dir"

  # 1. export the cutset to wavs and RTTMs.
  export_cmd=("$PYTHON" "$REPO_ROOT/scripts/cutset_to_wav_rttm.py"
    --cuts "$cutset" --output "$export_dir")
  [[ -n "$SESSIONS" ]] && export_cmd+=(--sessions "$SESSIONS")
  [[ $FORCE -eq 1 ]] && export_cmd+=(--force)
  echo "  export -> $export_dir"
  if ! run_logged "$out_dir/export.log" "${export_cmd[@]}"; then
    echo "  FAILED: export"
    failed+=("$name")
    continue
  fi

  # 2. decode the export through the RTTM route.
  if [[ -s "$hyp_rttm" && $FORCE -eq 0 ]]; then
    echo "  reusing $hyp_rttm (--force to re-decode)"
  else
    rttm_cmd=("$PYTHON" "$REPO_ROOT/infer.py"
      --rttm "$export_dir/rttm" --audio-dir "$export_dir/audio"
      --output "$hyp_rttm" --checkpoint "$CHECKPOINT" --device "$DEVICE")
    [[ -n "$SESSIONS" ]] && rttm_cmd+=(--sessions "$SESSIONS")
    rttm_cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
    echo "  decode --rttm -> $hyp_rttm"
    started=$SECONDS
    if ! run_logged "$out_dir/infer_rttm.log" "${rttm_cmd[@]}"; then
      echo "  FAILED: RTTM decode; see $out_dir/infer_rttm.log"
      failed+=("$name")
      continue
    fi
    echo "  RTTM decode done in $((SECONDS - started))s"
  fi

  # 3. the cutset side: a stored hypothesis, or one decoded here.
  if [[ $FORCE_CUTSET -eq 1 || ! -s "$hyp_cuts" ]]; then
    hyp_cuts="$out_dir/hyp_cuts.stm"
    if [[ -s "$hyp_cuts" && $FORCE -eq 0 && $FORCE_CUTSET -eq 0 ]]; then
      echo "  reusing $hyp_cuts"
    else
      cuts_cmd=("$PYTHON" "$REPO_ROOT/infer.py"
        --cuts "$cutset" --output "$hyp_cuts" --checkpoint "$CHECKPOINT" --device "$DEVICE")
      [[ -n "$SESSIONS" ]] && cuts_cmd+=(--sessions "$SESSIONS")
      cuts_cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
      echo "  decode --cuts -> $hyp_cuts"
      started=$SECONDS
      if ! run_logged "$out_dir/infer_cuts.log" "${cuts_cmd[@]}"; then
        echo "  FAILED: cutset decode; see $out_dir/infer_cuts.log"
        failed+=("$name")
        continue
      fi
      echo "  cutset decode done in $((SECONDS - started))s"
    fi
  else
    echo "  cutset hypothesis: $hyp_cuts"
  fi

  # 4. compare.
  compare_log="$out_dir/compare.txt"
  echo "  compare"
  run_logged "$compare_log" "$PYTHON" "$REPO_ROOT/scripts/compare_stm.py" "$hyp_cuts" "$hyp_rttm"
  status=$?

  if [[ $DRY_RUN -eq 1 ]]; then
    continue
  fi

  sessions_compared="$(sed -n 's/^Sessions: \([0-9]*\) in A.*/\1/p' "$compare_log" | tail -1)"
  if [[ $status -eq 0 ]]; then
    identical+=("$name")
    rows+=("$(printf '%-32s %8s %-11s %10s %8s' "$name" "${sessions_compared:-?}" identical 0 "0.0000")")
    if [[ $CLEAN -eq 1 ]]; then
      echo "  removing $export_dir/audio"
      rm -rf "$export_dir/audio"
    fi
  else
    different+=("$name")
    lines="$(sed -n 's/^DIFFERENT: \([0-9]*\) lines.*/\1/p' "$compare_log" | tail -1)"
    cpwer="$(sed -n 's/^cpWER of B against A: \([0-9.]*\)%.*/\1/p' "$compare_log" | tail -1)"
    rows+=("$(printf '%-32s %8s %-11s %10s %8s' "$name" "${sessions_compared:-?}" differs \
      "${lines:-?}" "${cpwer:-?}")")
  fi
  echo
done

echo "==================== parity summary ===================="
if [[ ${#rows[@]} -gt 0 ]]; then
  printf '%-32s %8s %-11s %10s %8s\n' DATASET SESSIONS RESULT DIFF-LINES cpWER%
  printf '%s\n' "${rows[@]}"
  echo
fi
echo "identical: ${#identical[@]} ${identical[*]+(${identical[*]})}"
echo "differs  : ${#different[@]} ${different[*]+(${different[*]})}"
echo "failed   : ${#failed[@]} ${failed[*]+(${failed[*]})}"
[[ ${#different[@]} -eq 0 && ${#failed[@]} -eq 0 ]]
