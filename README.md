# DiCoP — Diarization-Conditioned Parakeet

Target-speaker ASR built on NVIDIA [Parakeet-TDT-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2).
Given a recording and a diarization, the model transcribes **one chosen speaker at a time**,
using the diarization to tell it which speech to attend to and which to suppress.

Conditioning happens inside the encoder. Every frame is labelled with one of four classes —
**S**ilence, **T**arget, **N**on-target, **O**verlap — and each of the 24 FastConformer layers
applies a small learned transform per class before the layer runs (an FDDT block). At
initialization the target and overlap transforms are the identity while silence and non-target
are attenuated, so the model suppresses interfering speech before it has been trained at all.

The result is that a whole meeting is decoded per speaker in a single pass, with no
segmentation, no speaker embeddings, and no separation front-end.

## Setup

```bash
conda create -n dicop python=3.11 -y && conda activate dicop
# Install cuda-toolkit if not already installed. It is only necessary for training.
pip install -r requirements.txt
```

Training additionally needs a working **numba CUDA** toolchain, because the TDT loss is a numba
kernel. Inference does not need it. Two things have to hold, and `scripts/run_finetune.sh` checks
both before it starts:

```bash
python -c "from numba import cuda; print(cuda.is_available())"   # must print True
```

- If that reports a missing `libnvvm.so`, install a CUDA toolkit (`conda install -c nvidia cuda-nvcc`).
- `cuda.is_available()` printing `True` is not sufficient. **numba must be 0.65.1**, as pinned in
  `requirements.txt`. numba 0.66 implements the `min`/`max` builtins as a vararg overload its CUDA
  target cannot compile, and NeMo's TDT *gradient* kernel clamps with `min(g, clamp)` /
  `max(g, -clamp)`. The forward pass compiles fine, so a too-new numba only surfaces on the first
  backward pass, as `TypeError: Signature mismatch: 2 argument types given, but function takes 1
  arguments`.

## Manifest formats

Two are supported interchangeably, selected by the file suffix (`.jsonl` → NeMo, `.jsonl.gz` →
Lhotse) or by an explicit `use_lhotse` in the dataset config. They express the same thing — the
same data in either form produces identical dataset items — so you can switch without changing
results.

### NeMo JSON Lines

JSON Lines, **one line per session**, with `text` holding the diarized reference segments rather
than a single string:

```json
{"audio_filepath": "/path/IS1009a/audio.wav",
 "duration": 839.0,
 "offset": 0.0,
 "session_id": "IS1009a",
 "text": [{"start": 54.95, "duration": 0.91, "speaker": "FIE088", "text": "okay"},
          {"start": 55.70, "duration": 0.33, "speaker": "FIO089", "text": "hello"}]}
```

- `start` is relative to `offset`, so a segmented manifest can hold many windows of one recording.
- `session_id` is optional and defaults to the audio file stem. It is what identifies a recording
  in STM output, so set it when the filename is not the recording id
  (`IS1009a.Array1-01.wav` → `IS1009a`).
- One dataset item is a **(session, target speaker)** pair. Training samples one random target
  speaker per session per epoch; validation and test enumerate every speaker exactly once.

### Lhotse CutSet

A `CutSet` (`.jsonl.gz`) can be used directly for training, validation and inference — no
conversion step. The fields map one to one, and both formats keep segment times **relative to the
start of their window**, so a pre-segmented cutset works the same way a segmented NeMo manifest
does:

| NeMo manifest | Lhotse cut |
|---|---|
| `offset` | `cut.start` |
| `text[].start` | `supervision.start` |
| `duration` | `cut.duration` |
| `session_id` | `cut.recording.id` |

Keying on the *recording* rather than the cut is what makes the many windows of a pre-segmented
cutset compose back into one STM session on a single timeline.

Two differences from the NeMo path worth knowing:

- **Training enumerates every (cut, speaker) pair**, whereas the NeMo path samples one random
  target speaker per session per epoch. A Lhotse epoch is correspondingly longer.
- **No `MultiCut`.** Multi-channel arrays raise a clear error naming the cut and its type, rather
  than being silently mishandled: choosing between their channels is the cutset author's call.
  `MixedCut` (LibriMix, LibriSpeechMix) *is* supported, for training as well as inference — it
  names no file, so it is rendered with `cut.load_audio()` instead of going through NeMo's
  featurizer, and resampled if the corpus rate differs. Because it is rendered rather than read,
  `channel_selector`, `trim` and any waveform `augmentor` do not apply to it, and having no
  recording it scores as its own session under `cut.id`.

Lhotse is used purely as a manifest format: the CutSet is read into a plain map-style dataset and
batched by the same DataLoader as the NeMo path. Lhotse's own sampler stack is deliberately not
used — it hands a whole `CutSet` to `__getitem__`, which would break both the per-(cut, speaker)
fan-out and the integer-indexed reference lookup that multi-talker scoring depends on.

`scripts/retarget_manifest.py` rewrites the audio paths of an existing NeMo manifest against a new
audio root, which is useful when a manifest outlives the machine it was written on:

```bash
python scripts/retarget_manifest.py \
    --manifest manifests/ami-sdm_test.jsonl \
    --audio-dir /data/ami --audio-glob '{session}/audio.wav' \
    --session-from-parent-of-parent --set-session-id \
    --output manifests/ami-sdm_test_local.jsonl
```

## Inference: diarization + audio → STM

```bash
python infer.py \
    --rttm /path/to/rttms/ \
    --audio-dir /path/to/audio/ \
    --output hyp.stm \
    --checkpoint /path/to/best.ckpt
```

`--rttm` takes a single RTTM or a directory searched recursively for `*.rttm`; segments for the
same session id are merged. Each RTTM session id is matched to audio by trying
`{audio-dir}/{session}{ext}`, then `{audio-dir}/**/{session}{ext}`, then `--audio-glob` — so for
a layout like `/data/ami/IS1009a/audio.wav` pass `--audio-glob '{session}/audio.wav'`.

Alternatively pass a **Lhotse CutSet**, which already carries both the audio paths and the
diarization, so no `--audio-dir` is needed:

```bash
python infer.py \
    --cuts /path/to/cuts.jsonl.gz \
    --output hyp.stm \
    --checkpoint /path/to/best.ckpt
```

A pre-segmented cutset holds several cuts per recording; each is decoded separately and their
words are merged onto the recording's timeline, so one STM session comes out either way. All the
options below apply to both inputs.

Output is one STM, byte-compatible with the `hyp.stm` that training-time validation writes:

```
IS1009a 1 FIE088 54.88 55.12 okay
```

Useful options:

| Option | Purpose |
|---|---|
| `--stm-granularity word\|segment` | One line per word (default, tightest tcpWER timings) or grouped on >0.5 s pauses. |
| `--batch-size N` | Target speakers per forward pass. Features are computed once per session either way. Keep at 1 for long-form. |
| `--min-speech-seconds S` | Skip speakers with less total speech than `S`. Each speaker costs a decoding pass, so this cheaply suppresses spurious diarization output. |
| `--sessions` / `--session-list` | Decode a subset. |
| `--chunk-seconds S` | Decode in windows instead of whole sessions. See below. |
| `--att-context-size L,R` | Restrict encoder attention, e.g. `128,128`. |
| `-O` / `--override KEY=VALUE` | Override any config key, repeatable. Applied *after* a checkpoint's own stored config, so it decides the architecture the weights load into — which is what makes it able to change things a `.ckpt` would otherwise pin. |
| `--continue-on-fail` | Log and skip failing sessions instead of aborting. |

### Long-form decoding

Full-session decoding is the default and is what the model was trained and evaluated with
(`pos_emb_max_len` covers an hour at the encoder's 12.5 Hz frame rate). It is also the
memory-hungry option: a 30-minute session is roughly 22.5k encoder frames, so full self-attention
over it is only comfortable with scaled dot-product attention, which is on by default
(`encoder.use_pytorch_sdpa: true`, overridable with `--no-use-sdpa`).

If a session still will not fit, in increasing order of accuracy cost:

1. `--att-context-size 128,128` — bounds the attention window.
2. `-O model.encoder.self_attention_model=rel_pos_local_attn -O model.encoder.att_context_size=[256,256]`
   — swaps in NeMo's overlapping-window local attention, which bounds the cost rather than just
   masking a full attention matrix. The trained attention weights carry over unchanged, so the
   checkpoint still loads with `strict=True`.
3. `--chunk-seconds 120 --chunk-overlap-seconds 10` — decodes in overlapping windows aligned to
   whole encoder frames, attributing each word to exactly one window so seams cannot duplicate
   or drop words. This costs real accuracy because the conditioning sees less of the surrounding
   non-target speech: on three AMI-SDM test sessions, cpWER went from 0.185 to 0.260.

### Scoring

`scripts/manifest_to_rttm.py` and `scripts/manifest_to_stm.py` derive an oracle RTTM and a
reference STM from the same manifest, which gives both an end-to-end check of the inference path
and the oracle-diarization control — the mask `infer.py` builds from that RTTM is the one training
builds from the manifest:

```bash
python scripts/manifest_to_rttm.py --manifest test.jsonl --output rttms/
python scripts/manifest_to_stm.py  --manifest test.jsonl --output ref.stm
python infer.py --rttm rttms/ --audio-dir /data/ami --audio-glob '{session}/audio.wav' \
                --output hyp.stm --checkpoint best.ckpt

meeteval-wer cpwer  -r ref.stm -h hyp.stm
meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar 5
```

`manifest_to_stm.py` reads either manifest format, so a decode run from a cutset takes its
reference from that same cutset. Sessions are then keyed on the recording id exactly as
`infer.py` keys them, which the paired NeMo manifest would not reproduce — it carries no
`session_id`, so its sessions fall back to the audio file stem:

```bash
python scripts/manifest_to_stm.py --manifest ami-sdm_cutset_dev.jsonl.gz --output ref.stm
```

`scripts/run_scoring.sh` does this for a whole sweep: for every `{name}/hyp.stm` under a decode
directory it recovers the source manifest from that run's `infer.log`, writes `ref.stm` beside
the hypothesis, and prints a cpWER/tcpWER table over all the sets.

```bash
scripts/run_scoring.sh --decode-dir exps/decode-local
```

The `meeteval-wer` CLI needs `simplejson` (in `requirements.txt`); the Python API
(`meeteval.wer.cpwer`) does not.

## Evaluation

`scripts/run_inference.sh` decodes a checkpoint over a whole set of prepared evaluation corpora,
one after another, leaving `{output-dir}/{name}/hyp.stm` next to that run's `infer.log`;
`scripts/run_scoring.sh` then scores every one of them. A failing set is reported at the end
rather than aborting the rest.

```bash
scripts/run_inference.sh --checkpoint /path/to/best.ckpt --output-dir exps/decode-local
scripts/run_scoring.sh   --decode-dir exps/decode-local
```

`--datasets` takes names or name prefixes (`--datasets ami,notsofar`, `--datasets all`), `--list`
prints the known sets and their resolved paths, `--dry-run` prints the commands only, and
everything after `--` is appended to every `infer.py` call. Both scripts read their corpora from a
prepared `mt-asr-data-prep` checkout — `MANIFEST_ROOT` and `DATA_ROOT` (or `--manifest-root` /
`--data-root`) point them elsewhere, and the defaults are machine-local paths that will not exist
on your machine.

Sets whose cutsets are `MultiCut` take the RTTM route instead: their paired NeMo manifest is turned
into an oracle RTTM and the array is mixed down with `--channel-selector average`. `run_scoring.sh`
cannot recover a source manifest from those runs' logs, so name it with `--manifest NAME=PATH`;
AliMeeting and AIShell-4 also want `--text-norm none`.

### Results

Oracle diarization, cpWER and tcpWER (collar 5) in percent, `whisper_nsf` normalization on both
sides — i.e. exactly what the two commands above produce, decoded with the pretrained DiCoP
checkpoint (`dicop_stno_jsalt_pretrained.ckpt`). AMI's half-hour sessions used windowed local
attention
(`-O model.encoder.self_attention_model=rel_pos_local_attn -O model.encoder.att_context_size=[256,256]`)
to bound memory; every other set is full-context, full-session.

| Set | Sessions | cpWER | tcpWER |
|---|---|---|---|
| AMI-SDM dev / test | 18 / 16 | 13.98 / 15.97 | 14.26 / 16.51 |
| AMI-IHM-mix dev / test | 18 / 16 | 11.20 / 11.75 | 11.41 / 12.15 |
| NOTSOFAR-SDM dev1 / eval | 177 / 160 | 17.44 / 17.56 | 17.93 / 17.94 |
| LibriSpeechMix 2mix dev / test | 2703 / 2620 | 2.62 / 2.54 | 2.62 / 2.54 |
| LibriSpeechMix 3mix dev / test | 2703 / 2620 | 6.79 / 6.34 | 6.80 / 6.35 |
| Libri2Mix dev / test clean | 3000 | 4.13 / 4.40 | 4.16 / 4.41 |
| Libri3Mix dev / test clean | 3000 | 30.93 / 33.12 | 31.00 / 33.19 |

Libri3Mix is the outlier because three fully overlapped speakers at t=0 is the hardest condition
the mask can describe and the furthest from the meeting data the checkpoint was trained on; see
the fine-tuning note below.

### Checking the two inference routes agree

A cutset and an RTTM-plus-audio-directory express the same thing, so decoding an export of a cutset
must reproduce decoding the cutset. `scripts/run_rttm_parity.sh` checks that end to end: it exports
each set with `scripts/cutset_to_wav_rttm.py` (audio rendered through the loader `infer.py` uses,
RTTM times at full float precision, because `%.3f` rounding moves segment edges into neighbouring
80 ms mask frames), decodes the export with `--rttm --audio-dir`, and compares the result against
the `--cuts` hypothesis with `scripts/compare_stm.py`.

```bash
scripts/run_rttm_parity.sh --checkpoint best.ckpt --datasets notsofar-sdm-dev1 --clean
```

The two hypotheses are expected to be **identical**, not merely close. `compare_stm.py` also works
standalone on any two STMs over the same sessions, and reports the cpWER of one against the other
so a handful of differing words is not read as a wholesale mismatch.

## Training

```bash
python train.py \
    model.train_ds.manifest_filepath=manifests/train.jsonl \
    model.validation_ds.manifest_filepath=manifests/dev.jsonl \
    exp_manager.exp_dir=exps/
```

Lhotse CutSets work the same way — the format is picked up from the suffix:

```bash
python train.py \
    model.train_ds.manifest_filepath=manifests/ami-sdm_cutset_train_30s.jsonl.gz \
    model.validation_ds.manifest_filepath=manifests/ami-sdm_cutset_dev.jsonl.gz \
    exp_manager.exp_dir=exps/
```

Note that validation cannot be truncated with `trainer.limit_val_batches` when the validation
set spans several recordings: meeteval refuses to score if more than 10% of recordings have no
hypothesis, and a truncated pass leaves most of them undecoded.

The tokenizer and the initial encoder/decoder/joint weights both come from
`init_from_pretrained` (default `nvidia/parakeet-tdt-0.6b-v2`); only the FDDT blocks start from
scratch. A clean warm start reports the 192 FDDT tensors as the **only** missing keys and nothing
unexpected — `train.py` warns if anything else is missing, which means a topology mismatch.

Validation reports multi-talker **cpWER** and **tcpWER** (collar 5) rather than token WER, and
writes `ref.stm` / `hyp.stm` into `{log_dir}/preds_{epoch}_{step}/` each validation epoch.
Checkpoints are selected on `val/cp_wer`. Logging is TensorBoard by default; set
`exp_manager.create_wandb_logger=true` for W&B.

All settings live in [conf/dicop.yaml](conf/dicop.yaml). Notable ones:

| Key | Notes |
|---|---|
| `model.fddt_lr_multiplier` | Separate learning rate for the FDDT parameters. **Leave at 1.0 to reproduce the released checkpoints** — see below. |
| `model.text_norm` | Normalizer applied to references before tokenization and to both sides at scoring (`whisper_nsf`). |
| `model.encoder.fddt_*` | FDDT shape and initialization. Defaults match the original implementation. |
| `model.log_training_wer` | Off by default; a greedy decode every logged step is expensive. |
| `model.encoder.reduction` | Must stay `null`: it changes the frame rate mid-stack, which would misalign the mask. The encoder refuses to run with both. |

### A note on `fddt_lr_multiplier`

The original in-NeMo implementation built a separate FDDT parameter group but then discarded it
on the code path every shipped config took, so the multiplier was silently a no-op and the
released checkpoints trained at a uniform learning rate despite being launched with
`fddt_lr_multiplier=100`. DiCoP implements it correctly and defaults it to **1.0**, which
reproduces those runs. Values above 1 are the intended-but-untested behaviour, and switching away
from 1.0 emits two parameter groups, which optimizer state saved by a 1.0 run cannot be resumed
into.

### Fine-tuning on LibriMix

`scripts/run_finetune.sh` continues training an existing DiCoP checkpoint on Libri2Mix or
Libri3Mix, the counterpart to `run_inference.sh` on the training side:

```bash
scripts/run_finetune.sh --init-ckpt /path/to/dicop.ckpt
scripts/run_finetune.sh --train-split train-clean-360 --noisy --max-epochs 3
scripts/run_finetune.sh --dry-run          # print the train.py line and stop
```

Cutset paths are templated from `--n-src`, `--train-split` / `--dev-split` and `--noisy`, so no
dataset table is needed; `--list` shows the resolved paths and whether they exist. Everything after
`--` is passed to `train.py` as extra Hydra overrides.

Two defaults differ from `conf/dicop.yaml`, both because this is a fine-tune rather than a fresh
run:

- **`+init_from_ptl_ckpt`** loads weights only. The `+` is required — the key is commented out in
  the YAML, so Hydra has to add it rather than override it.
- **`lr=0.02`, `warmup_steps=500`** replace the from-scratch `0.5` / `2000`. No optimizer state is
  restored, so the schedule restarts from step 0 and the from-scratch peak would wreck a converged
  checkpoint. Under NoamAnnealing at `d_model=1024` these peak at ~2.8e-5.

Validation defaults to an evenly spread 300-cut subset of the dev set (`--val-cuts`, `0` for the
full set), built by `scripts/subset_cutset.py`. A full Libri2Mix dev epoch is 3000 cuts × 2
speakers = 6000 one-at-a-time greedy decodes, and as noted above `trainer.limit_val_batches`
cannot shorten it. The trade-off is that `val/cp_wer` — which checkpoints are selected on — is
then subset-relative, so rescore the finished checkpoint on the full sets:

```bash
scripts/run_inference.sh --checkpoint exps/libri2mix-ft/<run>/checkpoints/<best>.ckpt \
    --datasets librimix-2mix --output-dir exps/decode-ft
scripts/run_scoring.sh --decode-dir exps/decode-ft
```

Note that LibriMix mixtures are fully overlapped — every track starts at t=0 — so the STNO mask is
overlap-dominated. That is a narrower condition than meeting data, and gains here need not carry
over to AMI or NOTSOFAR.

## Checkpoints

`--checkpoint` accepts a Lightning `.ckpt`, a `.nemo` archive, or an NGC/HuggingFace model id.
For a `.ckpt` or a `.nemo`, the architecture config is taken from the checkpoint itself and merged
over the YAML, so the model matches the weights even if the config has since changed; weights then
load with `strict=True`. A model id is instead fetched and restored by NeMo, which builds the
encoder before any override could reach its constructor: `--att-context-size` and
`model.encoder.self_attention_model` still take effect (the attention layers are rebuilt and the
trained weights carried across), and anything else is reported as ignored rather than applied.

Checkpoints produced by the original in-NeMo implementation load unchanged — the encoder's old
module path is aliased to this repository's class, and the parameter names are identical.

`transcribe()` is deliberately disabled on these models. NeMo's transcription path cannot supply
a mask, and an unconditioned encoder returns a fluent transcript of whoever is loudest, which
looks correct but is not target-speaker output. Use `infer.py`, or `transcribe_stno()` directly.

### Publishing to HuggingFace

Nothing training writes is publishable as-is. The tokenizer is injected as an object rather than
built from `cfg.tokenizer`, so it is never registered as an artifact and the `.nemo` that
`exp_manager` writes holds only a config and weights — `restore_from()` cannot read it, which is
why loading one locally re-fetches the tokenizer from `init_from_pretrained` every time.

`scripts/export_to_hf.py` bundles the tokenizer in, names the archive the way NeMo expects to
find it in a Hub repo, generates a model card, and proves the round trip by restoring the result
and comparing it tensor by tensor against the source:

```bash
python scripts/export_to_hf.py \
    --checkpoint exps/run/checkpoints/best.ckpt \
    --output-dir exps/hf-export/dicop-parakeet-tdt-0.6b \
    --repo-id ORG/dicop-parakeet-tdt-0.6b \
    --push-to-hub
```

Authentication is the standard HuggingFace chain (`HF_TOKEN`, or `hf auth login`). Drop
`--push-to-hub` to build the bundle without uploading, or add `--dry-run` to see what an upload
would contain. The archive filename has to match the repository name — NeMo resolves a Hub id by
downloading `{repo-name}.nemo` from the repo root — so `--repo-id` decides it.

A published checkpoint still needs this repository to load: the encoder's `_target_` points
outside the `nemo` package, so a consumer has to import DiCoP and call
`allow_external_nemo_targets()`. The generated model card says so.

## Repository layout

```
train.py                           training entrypoint (Hydra)
infer.py                           RTTM + audio, or a Lhotse CutSet -> STM
conf/dicop.yaml                    the single config
src/data/stno.py                   the STNO mask math, shared by training and inference
src/data/dataset.py                (session, target speaker) dataset, NeMo manifests
src/data/collections.py            NeMo manifest parsing
src/data/lhotse_dataset.py         (cut, target speaker) dataset, Lhotse CutSets
src/data/lhotse_utils.py           cut adapters, shared by training and inference
src/data/dataloader.py             picks the dataset class per manifest format, builds the loader
src/data/text_norm/                the text normalizers (`whisper_nsf` and friends)
src/metrics/meeteval_mt_wer.py     cpWER / tcpWER + STM dumping
src/model/asr_model.py             EncDecRNNTModelSTNO
src/model/asr_bpe_model.py         EncDecRNNTBPEModelSTNO
src/model/modules/fddt.py          the per-class frame transform
src/model/modules/stno_encoder.py  FastConformer + per-layer FDDT
utils/{rttm,audio,stm}.py          RTTM parsing, audio resolution, STM writing
utils/inference.py                 checkpoint loading and the batched decoding runtime
utils/nemo.py                      NeMo interop: external `_target_`s, legacy aliases, ckpt loading
scripts/export_to_hf.py            checkpoint -> self-contained .nemo bundle for the HF Hub
scripts/run_{inference,scoring}.sh decode a checkpoint over the eval sets, then score them
scripts/run_finetune.sh            LibriMix fine-tuning runner
scripts/run_rttm_parity.sh         checks the --cuts and --rttm routes decode identically,
                                   with cutset_to_wav_rttm.py and compare_stm.py
scripts/                           manifest -> RTTM / STM, manifest retargeting, cutset subsetting
```

DiCoP depends on the pip-installed `nemo_toolkit` and subclasses it rather than forking it.
`src/data/stno.py` is deliberately the single source of truth for the mask, so the training
dataset and `infer.py` cannot drift apart.

## Contact

If you have any questions, reach out to: [iklement@fit.vut.cz](mailto:iklement@fit.vut.cz)
