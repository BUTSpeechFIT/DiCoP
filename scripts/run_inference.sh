#!/usr/bin/env bash
#
# Run DiCoP inference over the evaluation sets prepared in mt-asr-data-prep.
#
#   scripts/run_inference.sh --checkpoint /path/to/best.ckpt
#   scripts/run_inference.sh --checkpoint best.ckpt --datasets ami,notsofar
#   scripts/run_inference.sh --checkpoint best.ckpt --datasets all -- --chunk-seconds 120
#
# One `hyp.stm` per dataset lands in {output-dir}/{name}/, next to the run's log. Sets are
# decoded one after another; a failing set is reported at the end instead of aborting the rest.
#
# Two input routes, because the prepared cutsets are not all decodable as they stand:
#
#   cuts  The cutset goes straight to `infer.py --cuts`. Works for MonoCut and MixedCut (the
#         LibriMix / LibriSpeechMix mixtures, which Lhotse renders on the fly).
#   rttm  The paired NeMo manifest is turned into an oracle RTTM (scripts/manifest_to_rttm.py)
#         and decoded with `--rttm --audio-dir`. This is the route for the multi-channel
#         corpora, whose cutsets are MultiCut and which `infer.py` refuses; the array is mixed
#         down to mono by `--channel-selector average` at load time instead.
#
# Deliberately absent, because there is no correct way to run them here:
#
#   ami-mdm, notsofar1-mdm   MultiCut over one file *per microphone*, so no in-file downmix is
#                            possible. Their first array channel is exactly the sdm audio these
#                            sets already decode, so a first-channel run would be duplicate work.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_ROOT="${MANIFEST_ROOT:-/home/jovyan/mt-asr-data-prep/manifests}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/mt-asr-data-prep/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/exps/decode}"
PYTHON="${PYTHON:-python}"

CHECKPOINT=""
DEVICE="auto"
# Multi-talker meeting corpora. LibriSpeech and LibriSpeechMix are opt-in: they are thousands of
# short single-speaker recordings, so they cost far more wall time than they are worth by default.
SELECTION="ami,notsofar,alimeeting,aishell4"
DRY_RUN=0
LIST_ONLY=0
FORCE=0
EXTRA_ARGS=()

# name|route|manifest under $MANIFEST_ROOT|audio dir under $DATA_ROOT (rttm only)|extra infer.py args
DATASETS=(
  "ami-sdm-dev|cuts|ami/ami-sdm_cutset_dev.jsonl.gz||"
  "ami-sdm-test|cuts|ami/ami-sdm_cutset_test.jsonl.gz||"
  "ami-ihm-mix-dev|cuts|ami/ami-ihm-mix_cutset_dev.jsonl.gz||"
  "ami-ihm-mix-test|cuts|ami/ami-ihm-mix_cutset_test.jsonl.gz||"
  "notsofar-sdm-dev1|cuts|notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl.gz||"
  "notsofar-sdm-eval|cuts|notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl.gz||"
  "alimeeting-far-eval|rttm|ali_meeting/alimeeting-sdm_eval_nemo_manifest.jsonl|ali_meeting/Eval_Ali/Eval_Ali_far/audio_dir|--channel-selector average"
  "alimeeting-far-test|rttm|ali_meeting/alimeeting-sdm_test_nemo_manifest.jsonl|ali_meeting/Test_Ali/Test_Ali_far/audio_dir|--channel-selector average"
  "aishell4-test|rttm|aishell4/aishell4_test_nemo_manifest.jsonl|aishell4/test/wav|--channel-selector average"
  "librispeech-dev-clean|cuts|librispeech/librispeech_cutset_dev-clean.jsonl.gz||"
  "librispeech-dev-other|cuts|librispeech/librispeech_cutset_dev-other.jsonl.gz||"
  "librispeech-test-clean|cuts|librispeech/librispeech_cutset_test-clean.jsonl.gz||"
  "librispeech-test-other|cuts|librispeech/librispeech_cutset_test-other.jsonl.gz||"
  "librispeechmix-dev-clean-1mix|cuts|librispeechmix/librispeechmix_cutset_dev-clean-1mix.jsonl.gz||"
  "librispeechmix-test-clean-1mix|cuts|librispeechmix/librispeechmix_cutset_test-clean-1mix.jsonl.gz||"
  "librispeechmix-dev-clean-2mix|cuts|librispeechmix/librispeechmix_cutset_dev-clean-2mix.jsonl.gz||"
  "librispeechmix-test-clean-2mix|cuts|librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz||"
  "librispeechmix-dev-clean-3mix|cuts|librispeechmix/librispeechmix_cutset_dev-clean-3mix.jsonl.gz||"
  "librispeechmix-test-clean-3mix|cuts|librispeechmix/librispeechmix_cutset_test-clean-3mix.jsonl.gz||"
  # LibriMix also ships _noisy variants of each of these; add them here if they are wanted.
  "librimix-2mix-dev-clean|cuts|librimix/librimix_cutset_libri2mix_dev-clean.jsonl.gz||"
  "librimix-2mix-test-clean|cuts|librimix/librimix_cutset_libri2mix_test-clean.jsonl.gz||"
  "librimix-3mix-dev-clean|cuts|librimix/librimix_cutset_libri3mix_dev-clean.jsonl.gz||"
  "librimix-3mix-test-clean|cuts|librimix/librimix_cutset_libri3mix_test-clean.jsonl.gz||"
)

usage() {
  cat <<EOF
Run DiCoP inference over the mt-asr-data-prep evaluation sets.

Usage: $(basename "$0") --checkpoint PATH [options] [-- <extra infer.py args>]

Options:
  --checkpoint PATH     .ckpt / .nemo / NGC / HF model id. Required.
  --datasets LIST       Comma-separated names or name prefixes, or 'all'.
                        Default: $SELECTION
  --output-dir DIR      Where {name}/hyp.stm goes. Default: $OUTPUT_DIR
  --manifest-root DIR   mt-asr-data-prep manifests. Default: $MANIFEST_ROOT
  --data-root DIR       mt-asr-data-prep audio. Default: $DATA_ROOT
  --device SPEC         auto, cpu, cuda, cuda:N. Default: $DEVICE
  --force               Re-decode sets that already have a hyp.stm.
  --dry-run             Print the commands without running them.
  --list                Show the known datasets and exit.
  -h, --help            This message.

Everything after '--' is appended to every infer.py call, e.g.

  $(basename "$0") --checkpoint best.ckpt -- --chunk-seconds 120 --continue-on-fail
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --datasets) SELECTION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
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
  printf '%-32s %-5s %-8s %s\n' NAME ROUTE MANIFEST SOURCE
  for row in "${DATASETS[@]}"; do
    IFS='|' read -r name route manifest _ _ <<<"$row"
    state=missing
    [[ -f "$MANIFEST_ROOT/$manifest" && -s "$MANIFEST_ROOT/$manifest" ]] && state=ok
    printf '%-32s %-5s %-8s %s\n' "$name" "$route" "$state" "$manifest"
  done
  echo
  echo "Default selection: $SELECTION  (use --datasets all for every row above)"
  echo "Not listed: ami-mdm / notsofar1-mdm (per-microphone MultiCut, first channel == sdm)."
  exit 0
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required." >&2
  usage >&2
  exit 2
fi
# Only a local file is checkable; an NGC/HF id is resolved by infer.py itself.
if [[ "$CHECKPOINT" == *.ckpt || "$CHECKPOINT" == *.nemo ]] && [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -d "$MANIFEST_ROOT" ]]; then
  echo "Manifest root not found: $MANIFEST_ROOT" >&2
  exit 2
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

decoded=()
skipped=()
failed=()

selected=()
for row in "${DATASETS[@]}"; do
  IFS='|' read -r name _ _ _ _ <<<"$row"
  is_selected "$name" && selected+=("$row")
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No dataset matches --datasets '$SELECTION'. Try --list." >&2
  exit 2
fi

echo "Checkpoint : $CHECKPOINT"
echo "Output dir : $OUTPUT_DIR"
echo "Datasets   : ${#selected[@]} selected from '$SELECTION'"
echo

index=0
for row in "${selected[@]}"; do
  IFS='|' read -r name route manifest audio_subdir extra <<<"$row"
  index=$((index + 1))

  manifest_path="$MANIFEST_ROOT/$manifest"
  out_dir="$OUTPUT_DIR/$name"
  hyp="$out_dir/hyp.stm"
  log="$out_dir/infer.log"

  echo "[$index/${#selected[@]}] $name"

  if [[ ! -s "$manifest_path" ]]; then
    echo "  skip: manifest missing or empty ($manifest_path)"
    skipped+=("$name")
    continue
  fi
  if [[ -s "$hyp" && $FORCE -eq 0 ]]; then
    echo "  skip: $hyp already exists (--force to re-decode)"
    skipped+=("$name")
    continue
  fi

  [[ $DRY_RUN -eq 0 ]] && mkdir -p "$out_dir"

  # Word-split the per-dataset extras; they are fixed strings in the table above, not user input.
  read -r -a extra_args <<<"$extra"

  cmd=("$PYTHON" "$REPO_ROOT/infer.py")
  if [[ "$route" == "cuts" ]]; then
    cmd+=(--cuts "$manifest_path")
  else
    audio_dir="$DATA_ROOT/$audio_subdir"
    if [[ ! -d "$audio_dir" ]]; then
      echo "  skip: audio directory missing ($audio_dir)"
      skipped+=("$name")
      continue
    fi
    rttm_dir="$out_dir/rttm"
    echo "  oracle RTTM from $manifest"
    if ! run "$PYTHON" "$REPO_ROOT/scripts/manifest_to_rttm.py" \
        --manifest "$manifest_path" --output "$rttm_dir"; then
      echo "  FAILED: could not build the RTTM"
      failed+=("$name")
      continue
    fi
    cmd+=(--rttm "$rttm_dir" --audio-dir "$audio_dir")
  fi
  cmd+=(--output "$hyp" --checkpoint "$CHECKPOINT" --device "$DEVICE")
  cmd+=("${extra_args[@]+"${extra_args[@]}"}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

  started=$SECONDS
  if [[ $DRY_RUN -eq 1 ]]; then
    run "${cmd[@]}"
    decoded+=("$name")
    continue
  fi

  # tee so a long decode is watchable and still leaves a log; pipefail would report tee's status.
  set +e
  "${cmd[@]}" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  elapsed=$((SECONDS - started))
  if [[ $status -ne 0 ]]; then
    echo "  FAILED after ${elapsed}s (exit $status); see $log"
    failed+=("$name")
  else
    echo "  done in ${elapsed}s -> $hyp"
    decoded+=("$name")
  fi
  echo
done

echo "==================== summary ===================="
echo "decoded: ${#decoded[@]} ${decoded[*]+(${decoded[*]})}"
echo "skipped: ${#skipped[@]} ${skipped[*]+(${skipped[*]})}"
echo "failed : ${#failed[@]} ${failed[*]+(${failed[*]})}"
[[ ${#failed[@]} -eq 0 ]]
