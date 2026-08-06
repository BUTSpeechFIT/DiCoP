# DiCoP — Diarization-Conditioned Parakeet

This repository implements Target-speaker ASR built on NVIDIA [Parakeet-TDT-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). It uses the STNO masks introduced in [DiCoW](https://github.com/BUTSpeechFIT/DiCoW) (hence the name DiCoP).

A trained checkpoint is published on the HuggingFace Hub:
**[BUT-FIT/DiCoP_v0.1](https://huggingface.co/BUT-FIT/DiCoP_v0.1)**.

---

## Checkpoints

| Model | Description | Link |
| --- | --- | --- |
| **DiCoP v0.1** | Fine-tuned Nvidia Parakeet with STNO masks on AMI, NOTSOFAR, Libri2Mix | [Hugging Face](https://huggingface.co/BUT-FIT/DiCoP_v0.1) |

---

## Setup

```bash
conda create -n dicop python=3.11 -y && conda activate dicop
pip install -r requirements.txt
```

Training requires numba CUDA setup (the TDT loss uses a numba kernel). Not necessary for inference.

```bash
python -c "from numba import cuda; print(cuda.is_available())"   # must print True
```

If that fails, install a CUDA toolkit: `conda install nvidia::cuda-toolkit`.

---

## Data preparation

Training and evaluation need a [Lhotse](https://github.com/lhotse-speech/lhotse) CutSet
(`.jsonl.gz`) that points to prepared audio. Manifests for the standard multi-talker datasets (AMI, NOTSOFAR, LibriSpeechMix,
LibriMix, ...) can be prepared using the following repository:

**[mt-asr-data-prep](https://github.com/BUTSpeechFIT/mt-asr-data-prep)**

Follow the instructions to prepare the multi-talker data.

---

## Configuration

This codebases inherits [Hydra](https://hydra.cc/) configuration management from [NeMo framework](https://github.com/NVIDIA-NeMo/Speech). All settings are in one file: [conf/dicop.yaml](conf/dicop.yaml). Edit it directly (recommended) or pass the changes through the command line: e.g.:
```
python train.py model.train_ds.manifest_filepath=...
```

Main parts of the config:

| Key | What it does |
|---|---|
| `name` | Name of the current run, used for logging and checkpoint directory. The default value is: `dicop`. |
| `init_from_pretrained` | Which pretrained model to start from (default `nvidia/parakeet-tdt-0.6b-v2`). |
| `model.train_ds` / `validation_ds` / `test_ds` | Manifest path, batch size, and other loader settings for each data split. |
| `model.encoder` | Model architecture, including the `fddt_*` settings for diarization conditioning. |
| `model.optim` | Optimizer and learning rate schedule. |
| `trainer` | PyTorch Lightning settings: number of GPUs, precision, max epochs, etc. |
| `exp_manager` | Where checkpoints and logs are written. |

Most of the parameters are inherited or set to their default values. Config value `???` requires the config key to be set either through the command line or directly in the `.yaml` file.

In `config/dicop.yaml`, the following keys need to be set:
| Key | Description |
|---|---|
| `model.train_ds.manifest_filepath` | Single Lhotse CutSet manifest path or a list of paths that will be concatenated. |
| `model.validation_ds.manifest_filepath` | Single Lhotse CutSet manifest path or a list of paths or a dictionary of `dataset_name` -> `path`. Every manifest is evaluated as a dataset of its own, and metrics are logged per-dataset + aggregate (e.g., `val/ami/cp_wer` and `val/cp_wer` - the accumulated WER across all the datasets). Without a dictionary, dataset names are the manifest file names without their suffixes. A dictionary value may itself be a list, which scores those manifests as one dataset. |
| `exp_manager.exp_dir` | Experiment directory. The run will be logged in `{exp_dir}/dicop` by default. |

---

## Inference: diarization + audio → STM

The basic inference script uses RTTMs to guide the inference. RTTM session ID must match the waveform name! For more details, look at: [utils/audio.py:L26](https://github.com/BUTSpeechFIT/DiCoP/blob/main/utils/audio.py#L26C1-L26C24).

- `--checkpoint` accepts a Hub id, a local `.ckpt`, or a `.nemo` file.
- `--rttm` takes a single RTTM file or a directory (searched recursively for `*.rttm`).

```bash
python infer.py \
    --rttm /path/to/rttms/ \
    --audio-dir /path/to/audio/ \
    --output hyp.stm \
    --checkpoint BUT-FIT/DiCoP_v0.1
```

You can also pass a Lhotse CutSet, which already contains both audio paths and diarization:

```bash
python infer.py \
    --cuts /path/to/cuts.jsonl.gz \
    --output hyp.stm \
    --checkpoint BUT-FIT/DiCoP_v0.1
```

Useful options:

| Option | What it does |
|---|---|
| `--stm-granularity word\|segment` | One line per word (default) or grouped by pauses. |
| `--batch-size N` | How many target speakers are decoded per forward pass. |
| `--min-speech-seconds S` | Skip speakers with less than `S` seconds of total speech. |
| `--chunk-seconds S` | Decode in windows instead of whole sessions (for long recordings). |
| `-O KEY=VALUE` | Override any config key, can be repeated. |

### Scoring

By default, we use **Whisper CHiME-8 normalization** to stay compatible with the DiCoW model. To run the scoring, we first transform the Lhotse CutSet manifest to STM and then run `meeteval` scoring tool to obtain `tcp` and `cp` Word Error Rate.

```bash
python scripts/manifest_to_rttm.py --manifest test.jsonl --output rttms/
python scripts/manifest_to_stm.py  --manifest test.jsonl --output ref.stm
python infer.py --rttm rttms/ --audio-dir /data/ami --output hyp.stm --checkpoint BUT-FIT/DiCoP_v0.1

meeteval-wer cpwer  -r ref.stm -h hyp.stm
meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar 5
```

To evaluate on a full set of prepared corpora at once:

```bash
scripts/run_inference.sh --checkpoint BUT-FIT/DiCoP_v0.1 --output-dir exps/decode-local
scripts/run_scoring.sh   --decode-dir exps/decode-local
```

---

## Training

The minimum you need to set is a training Lhotse manifest, a validation Lhotse manifest, and an output experiment
directory. Everything else uses the defaults in [conf/dicop.yaml](conf/dicop.yaml).

```bash
python train.py \
    model.train_ds.manifest_filepath=manifests/train_cuts.jsonl.gz \
    model.validation_ds.manifest_filepath=manifests/dev_cuts.jsonl.gz \
    exp_manager.exp_dir=exps/
```

Another example where more parameters are changed:

```bash
python train.py \
    model.train_ds.manifest_filepath=manifests/train.jsonl \
    model.validation_ds.manifest_filepath=manifests/dev.jsonl \
    exp_manager.exp_dir=exps/ \
    model.train_ds.batch_size=8 \
    trainer.devices=2 \
    trainer.max_epochs=100 \
    exp_manager.create_wandb_logger=true
```

`train_ds` can be set to a list of manifest paths. In such a case, it pools the data together and trains on the compound set:
```bash
python train.py \
    model.train_ds.manifest_filepath=[manifests/train_cuts_1.jsonl.gz,manifests/train_cuts_1.jsonl.gz] \
    model.validation_ds.manifest_filepath=manifests/dev_cuts.jsonl.gz \
    exp_manager.exp_dir=exps/
```

If you want to use multiple datasets for validation, you can set `validation_ds` to a list of manifests, or a dictionary where the key is the dataset name and the value is the manifest path. In such a case, the codebase automatically logs `val/{dataset_name}/{metric}` for all the dataset separately (either the manifest name or the dictionary key). Also, `val/{metric}` is then a properly-aggregated metric over all the datasets (for `cp_wer`, we aggregate the error components and divide it by the total reference length).

---

## Publishing to HuggingFace

The following script loads the checkpoint and publishes it to the HuggingFace along with a Readme.md. If you want to change the Readme text, look at the script source code.
```bash
python scripts/export_to_hf.py \
    --checkpoint exps/run/checkpoints/best.ckpt \
    --output-dir exps/hf-export/dicop-parakeet-tdt-0.6b \
    --repo-id ORG/dicop-parakeet-tdt-0.6b \
    --push-to-hub
```

It needs HuggingFace auth (`HF_TOKEN` or `hf auth login`). Drop `--push-to-hub` flag to build the export without uploading it to the HuggingFace Hub.

---

## Results

Oracle diarization, cpWER and tcpWER (collar 5) in percent, decoded with
[BUT-FIT/DiCoP_v0.1](https://huggingface.co/BUT-FIT/DiCoP_v0.1):

| Set | Sessions | cpWER | tcpWER |
|---|---|---|---|
| AMI-SDM dev / test | 18 / 16 | 13.98 / 15.97 | 14.26 / 16.51 |
| AMI-IHM-mix dev / test | 18 / 16 | 11.20 / 11.75 | 11.41 / 12.15 |
| NOTSOFAR-SDM dev1 / eval | 177 / 160 | 17.44 / 17.56 | 17.93 / 17.94 |
| LibriSpeechMix 2mix dev / test | 2703 / 2620 | 2.62 / 2.54 | 2.62 / 2.54 |
| LibriSpeechMix 3mix dev / test | 2703 / 2620 | 6.79 / 6.34 | 6.80 / 6.35 |
| Libri2Mix dev / test clean | 3000 | 4.13 / 4.40 | 4.16 / 4.41 |
| Libri3Mix dev / test clean | 3000 | 30.93 / 33.12 | 31.00 / 33.19 |

---

## Contact

[iklement@fit.vut.cz](mailto:iklement@fit.vut.cz)
