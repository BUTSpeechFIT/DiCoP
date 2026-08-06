#!/usr/bin/env bash
#
# Fine-tune DiCoP on LibriMix, starting from an already-trained DiCoP checkpoint.
#
#   scripts/run_finetune.sh
#   scripts/run_finetune.sh --train-split train-clean-360 --max-epochs 3
#   scripts/run_finetune.sh --dry-run
#   scripts/run_finetune.sh --val-cuts 20 -- trainer.max_steps=20     # smoke test
#
# Everything after '--' is appended verbatim as extra Hydra overrides, so anything in
# conf/dicop.yaml can be reached without a flag here.
#
# LibriMix cutset names are fully systematic, so there is no dataset table -- the paths are
# templated from --n-src, --train-split / --dev-split and --noisy:
#
#   $MANIFEST_ROOT/librimix/librimix_cutset_libri${n}mix_${split}${_noisy}.jsonl.gz
#
# The `_30s` variants are deliberately not used. For LibriMix they are a length *filter*, not a
# windowing, and no Libri2Mix cut reaches 30 s (longest measured: 18.2 s), so they select exactly
# the same cuts as the plain files.
#
# Two things differ from a from-scratch `python train.py`, both because this is a fine-tune:
#
#   init   `+init_from_ptl_ckpt` loads weights only (train.py:71, strict=True). The `+` is
#          required: the key is commented out in conf/dicop.yaml, so Hydra has to add it.
#   lr     conf/dicop.yaml ships lr=0.5 / warmup=2000, a from-scratch schedule, and no optimizer
#          state is restored so it would restart at that peak and wreck a converged checkpoint.
#          NoamAnnealing is lr = base * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5); at
#          d_model=1024 (0.03125) and warmup=500 the peak is base * 500^-0.5 * 0.03125
#          = base * 1.40e-3, so the default base=0.02 peaks at ~2.8e-5 and decays as step^-0.5.
#
# Validation runs on an evenly spread subset of the dev set by default (--val-cuts), because a
# full Libri2Mix dev epoch is 3000 cuts x 2 speakers = 6000 one-at-a-time greedy decodes. That
# makes `val/cp_wer` -- which checkpoints are selected on -- subset-relative and not directly
# comparable to a full-set number; rescore the finished checkpoint with run_inference.sh and
# run_scoring.sh. `trainer.limit_val_batches` is not an alternative, see scripts/subset_cutset.py.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_ROOT="${MANIFEST_ROOT:-/home/jovyan/mt-asr-data-prep/manifests}"
PYTHON="${PYTHON:-python}"

INIT_CKPT="${INIT_CKPT:-/home/jovyan/NeMo/misc/dicop_stno_jsalt_pretrained.ckpt}"
NSRC=2
TRAIN_SPLIT="train-clean-100"
DEV_SPLIT="dev-clean"
NOISY=0
VAL_CUTS=300
EXP_DIR=""
RUN_NAME=""
LR=0.02
WARMUP_STEPS=500
MAX_EPOCHS=5
BATCH_SIZE=16
ACCUM=1
DEVICES=-1
EVAL_AT_START=0
RESUME=0
WANDB=0
SKIP_CHECKS=0
DRY_RUN=0
LIST_ONLY=0
EXTRA_ARGS=()

usage() {
  cat <<EOF
Fine-tune DiCoP on LibriMix from an existing DiCoP checkpoint.

Usage: $(basename "$0") [options] [-- <extra Hydra overrides>]

Data:
  --n-src N             2 for Libri2Mix, 3 for Libri3Mix. Default: $NSRC
  --train-split NAME    train-clean-100 or train-clean-360. Default: $TRAIN_SPLIT
  --dev-split NAME      Validation split. Default: $DEV_SPLIT
  --noisy               Use the WHAM _noisy variants of both splits.
  --val-cuts N          Validate on an evenly spread N-cut subset of the dev set;
                        0 uses the whole thing. Default: $VAL_CUTS
  --manifest-root DIR   mt-asr-data-prep manifests. Default: $MANIFEST_ROOT

Run:
  --init-ckpt PATH      Checkpoint to fine-tune. Default: $INIT_CKPT
  --exp-dir DIR         Default: $REPO_ROOT/exps/libri{n}mix-ft
  --name NAME           Run name under the exp dir. Default: dicop-libri{n}mix-{train-split}
  --resume              Continue an interrupted run in the same exp dir.
  --eval-at-start       Validate once before training. Confirms the warm start.
  --wandb               Log to Weights & Biases instead of TensorBoard only.

Optimization:
  --lr F                NoamAnnealing base lr; see the header for the peak. Default: $LR
  --warmup-steps N      Default: $WARMUP_STEPS
  --max-epochs N        Default: $MAX_EPOCHS
  --batch-size N        Training batch size. Default: $BATCH_SIZE
  --accum N             Gradient accumulation steps. Default: $ACCUM
  --devices SPEC        Lightning devices; -1 is all visible GPUs. Default: $DEVICES

Other:
  --skip-checks         Skip the preflight checks.
  --dry-run             Print the commands without running them.
  --list                Show the resolved paths and settings, then exit.
  -h, --help            This message.

If it runs out of memory, halve --batch-size and double --accum to hold the effective batch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-src) NSRC="$2"; shift 2 ;;
    --train-split) TRAIN_SPLIT="$2"; shift 2 ;;
    --dev-split) DEV_SPLIT="$2"; shift 2 ;;
    --noisy) NOISY=1; shift ;;
    --val-cuts) VAL_CUTS="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --init-ckpt) INIT_CKPT="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --eval-at-start) EVAL_AT_START=1; shift ;;
    --wandb) WANDB=1; shift ;;
    --lr) LR="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --accum) ACCUM="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --skip-checks) SKIP_CHECKS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

suffix=""
[[ $NOISY -eq 1 ]] && suffix="_noisy"
CORPUS="libri${NSRC}mix"
TRAIN_CUTS="$MANIFEST_ROOT/librimix/librimix_cutset_${CORPUS}_${TRAIN_SPLIT}${suffix}.jsonl.gz"
DEV_CUTS="$MANIFEST_ROOT/librimix/librimix_cutset_${CORPUS}_${DEV_SPLIT}${suffix}.jsonl.gz"

[[ -z "$EXP_DIR" ]] && EXP_DIR="$REPO_ROOT/exps/${CORPUS}-ft"
[[ -z "$RUN_NAME" ]] && RUN_NAME="dicop-${CORPUS}-${TRAIN_SPLIT}${suffix}"

# The subset is keyed by its source and size, so switching --dev-split or --val-cuts builds a new
# one rather than silently reusing the previous set under a stale name.
VAL_CUTS_PATH="$DEV_CUTS"
if [[ "$VAL_CUTS" != "0" ]]; then
  VAL_CUTS_PATH="$EXP_DIR/val_${DEV_SPLIT}${suffix}_${VAL_CUTS}.jsonl.gz"
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

settings() {
  echo "Corpus     : $CORPUS ${TRAIN_SPLIT}${suffix} -> ${DEV_SPLIT}${suffix}"
  echo "Train cuts : $TRAIN_CUTS"
  echo "Dev cuts   : $DEV_CUTS"
  if [[ "$VAL_CUTS" != "0" ]]; then
    echo "Validation : $VAL_CUTS-cut subset -> $VAL_CUTS_PATH"
  else
    echo "Validation : full dev set"
  fi
  echo "Init ckpt  : $INIT_CKPT"
  echo "Exp dir    : $EXP_DIR/$RUN_NAME"
  echo "Optim      : lr=$LR warmup=$WARMUP_STEPS epochs=$MAX_EPOCHS batch=$BATCH_SIZE accum=$ACCUM devices=$DEVICES"
}

if [[ $LIST_ONLY -eq 1 ]]; then
  settings
  for path in "$TRAIN_CUTS" "$DEV_CUTS"; do
    state=missing
    [[ -s "$path" ]] && state=ok
    printf '%-8s %s\n' "$state" "$path"
  done
  exit 0
fi

if [[ $SKIP_CHECKS -eq 0 ]]; then
  for path in "$TRAIN_CUTS" "$DEV_CUTS"; do
    if [[ ! -s "$path" ]]; then
      echo "Cutset missing or empty: $path" >&2
      echo "Check --manifest-root (currently $MANIFEST_ROOT), --n-src, and the split names." >&2
      exit 2
    fi
  done
  # Only a local file is checkable; anything else is left to train.py to resolve.
  if [[ "$INIT_CKPT" == *.ckpt || "$INIT_CKPT" == *.nemo ]] && [[ ! -f "$INIT_CKPT" ]]; then
    echo "Checkpoint not found: $INIT_CKPT" >&2
    exit 2
  fi
  # The TDT loss is a numba CUDA kernel, so a broken toolchain fails on the first backward pass,
  # well after the checkpoint and the cutsets have been loaded. Catch it up front instead.
  #
  # `cuda.is_available()` alone does not catch it. Under numba 0.66 it returns True and the
  # forward kernels compile, then the *gradient* kernel dies with "Signature mismatch: 2 argument
  # types given, but function takes 1 arguments": 0.66 implements the `min`/`max` builtins as a
  # vararg overload (`def impl(*x)`) that the CUDA device-function compiler cannot handle, and
  # NeMo clamps with `min(g, clamp)` / `max(g, -clamp)` in compute_tdt_grad_kernel. So the probe
  # compiles and launches that exact construct rather than just asking whether CUDA is there.
  # Skipped under --dry-run, which only prints a command and needs no working runtime.
  if [[ $DRY_RUN -eq 0 ]]; then
    probe_status=0
    probe="$("$PYTHON" -W ignore - <<'PY' 2>&1
import sys

try:
    import numpy as np
    from numba import cuda
except Exception as exc:
    sys.exit(f"{type(exc).__name__}: {exc}")

if not cuda.is_available():
    sys.exit("numba reports no usable CUDA device or toolchain")

@cuda.jit
def clamp_probe(out, clamp):
    i = cuda.grid(1)
    g = out[i]
    g = min(g, clamp)
    g = max(g, -clamp)
    out[i] = g

try:
    buf = cuda.to_device(np.zeros(1, dtype=np.float32))
    clamp_probe[1, 1](buf, 1.0)
    cuda.synchronize()
except Exception as exc:
    sys.exit(f"{type(exc).__name__}: {exc}")
PY
)" || probe_status=$?
    if [[ $probe_status -ne 0 ]]; then
      echo "numba cannot compile the TDT gradient kernel, and training needs it:" >&2
      echo "  ${probe##*$'\n'}" >&2
      echo "A 'Signature mismatch' here means numba 0.66; pin numba==0.65.1." >&2
      echo "A missing libnvvm.so means no CUDA toolkit; conda install -c nvidia cuda-nvcc." >&2
      echo "See the Setup section of README.md. Pass --skip-checks to run anyway." >&2
      exit 2
    fi
  fi
fi

[[ $DRY_RUN -eq 0 ]] && mkdir -p "$EXP_DIR"

if [[ "$VAL_CUTS" != "0" ]]; then
  if [[ -s "$VAL_CUTS_PATH" ]]; then
    echo "Reusing validation subset $VAL_CUTS_PATH"
  else
    echo "Building the $VAL_CUTS-cut validation subset"
    run "$PYTHON" "$REPO_ROOT/scripts/subset_cutset.py" \
      --cuts "$DEV_CUTS" --output "$VAL_CUTS_PATH" --num "$VAL_CUTS"
  fi
fi

cmd=("$PYTHON" "$REPO_ROOT/train.py")
cmd+=("+init_from_ptl_ckpt=$INIT_CKPT")
cmd+=("model.train_ds.manifest_filepath=$TRAIN_CUTS")
# Named, so the curve is val/$CORPUS/cp_wer rather than the subset file's stem. `val/cp_wer`,
# which checkpoints are selected on, is logged alongside it and is the same number here.
cmd+=("model.validation_ds.manifest_filepath={$CORPUS:'$VAL_CUTS_PATH'}")
cmd+=("model.train_ds.batch_size=$BATCH_SIZE")
cmd+=("model.optim.lr=$LR")
cmd+=("model.optim.sched.warmup_steps=$WARMUP_STEPS")
cmd+=("trainer.max_epochs=$MAX_EPOCHS")
cmd+=("trainer.accumulate_grad_batches=$ACCUM")
cmd+=("trainer.devices=$DEVICES")
cmd+=("exp_manager.exp_dir=$EXP_DIR")
cmd+=("exp_manager.name=$RUN_NAME")
[[ $EVAL_AT_START -eq 1 ]] && cmd+=("evaluate_at_start=true")
[[ $WANDB -eq 1 ]] && cmd+=("exp_manager.create_wandb_logger=true")
if [[ $RESUME -eq 1 ]]; then
  cmd+=("exp_manager.resume_if_exists=true" "exp_manager.resume_ignore_no_checkpoint=true")
fi
cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

settings
echo

if [[ $DRY_RUN -eq 1 ]]; then
  run "${cmd[@]}"
  exit 0
fi

# tee so a long run is watchable and still leaves a log; pipefail would report tee's status.
log="$EXP_DIR/train.log"
started=$SECONDS
set +e
"${cmd[@]}" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
elapsed=$((SECONDS - started))

if [[ $status -ne 0 ]]; then
  echo "FAILED after ${elapsed}s (exit $status); see $log" >&2
  exit $status
fi

# Two things push the checkpoints below $EXP_DIR/$RUN_NAME: exp_manager nests each run under a
# timestamped version directory, and the monitored metric `val/cp_wer` has a '/' in it, so
# Lightning's filename template turns the part before it into another directory
# (checkpoints/dicop-...--val/cp_wer=...-last.ckpt). Locate the newest .ckpt and report the
# directory it is actually in rather than reconstructing a path that does not exist.
ckpt_dir="$(find "$EXP_DIR/$RUN_NAME" -name '*.ckpt' -printf '%T@ %h\n' 2>/dev/null |
            sort -rn | head -1 | cut -d' ' -f2-)"
[[ -z "$ckpt_dir" ]] && ckpt_dir="$EXP_DIR/$RUN_NAME/<version>/checkpoints"

echo
echo "done in ${elapsed}s -> $EXP_DIR/$RUN_NAME"
echo "val/cp_wer above is subset-relative; score the best checkpoint on the full sets with:"
echo "  scripts/run_inference.sh --checkpoint $ckpt_dir/<best>.ckpt \\"
echo "      --datasets librimix-${NSRC}mix --output-dir $REPO_ROOT/exps/decode-ft"
echo "  scripts/run_scoring.sh --decode-dir $REPO_ROOT/exps/decode-ft"
