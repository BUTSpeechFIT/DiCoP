#!/usr/bin/env python
"""Export a DiCoP checkpoint as a self-contained `.nemo` bundle for the HuggingFace Hub.

`infer.py --checkpoint` already accepts an NGC/HF model id, but nothing DiCoP writes can be
published as-is. The tokenizer is the reason: `EncDecRNNTBPEModelSTNO` takes a tokenizer *object*
injected from the pretrained Parakeet rather than building one from `cfg.tokenizer`, so it never
registers tokenizer artifacts, and the `.nemo` that `exp_manager` writes holds only
`model_config.yaml` and `model_weights.ckpt`. `restore_from()` cannot read that, which is why
`utils.inference` unpacks such archives by hand and re-fetches the tokenizer every time.

This script loads a checkpoint the same way `infer.py` does, attaches the tokenizer as proper
NeMo artifacts, writes an archive named the way NeMo expects to find it in a Hub repo, and
restores it back to prove the round trip is exact.

    python scripts/export_to_hf.py \\
        --checkpoint exps/run/checkpoints/best.ckpt \\
        --output-dir exps/hf-export/dicop-parakeet-tdt-0.6b

Add `--repo-id ORG/NAME --push-to-hub` to upload. Authentication is the standard HuggingFace
chain (`HF_TOKEN`, or `hf auth login`); there is no token flag. The archive is named after the
repository because NeMo resolves a Hub id by downloading `{repo-name}.nemo` from the repo root --
any other name sends it down the "download the whole repo" path instead.

The published model still needs this repository: the encoder's `_target_` points outside the
`nemo` package, so a consumer has to import DiCoP and call `allow_external_nemo_targets()`. The
generated model card says so.
"""

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf, open_dict

from utils.nemo import allow_external_nemo_targets

allow_external_nemo_targets()

from nemo.utils import logging

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNO
from utils.inference import DEFAULT_CONFIG_PATH, load_asr_model, load_inference_config
from utils.nemo import register_legacy_nemo_aliases

# The three files a NeMo BPE tokenizer is restored from, as (config key, name on disk, required).
# `_setup_tokenizer` registers the first two unconditionally and tolerates a missing third, so
# an archive without `tokenizer.vocab` still restores.
TOKENIZER_ARTIFACTS = (
    ("model_path", "tokenizer.model", True),
    ("vocab_path", "vocab.txt", True),
    ("spe_tokenizer_vocab", "tokenizer.vocab", False),
)

BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
PROJECT_URL = "https://github.com/BUT-FIT/DiCoP"

MODEL_CARD_TEMPLATE = """\
---
license: {license}
language:
- en
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: {base_model}
tags:
- automatic-speech-recognition
- target-speaker-asr
- speaker-diarization
- multi-talker
- parakeet
- nemo
- rnnt
- tdt
---

# DiCoP — Diarization-Conditioned Parakeet

Target-speaker ASR built on [{base_model}](https://huggingface.co/{base_model}). Given audio and
a diarization, it transcribes **one speaker at a time**.

The conditioning lives inside the encoder. Every frame is labelled silence / target /
non-target / overlap (STNO), and each Conformer layer applies a small learned per-class
transform — an FDDT block — before the layer runs. A whole meeting is decoded per speaker in one
pass: no segmentation, no speaker embeddings, no separation front-end.

| | |
|---|---|
| Base model | [{base_model}](https://huggingface.co/{base_model}) |
| Parameters | {num_params} |
| Encoder | {num_layers} × FastConformer, d_model {d_model} |
| Encoder frame rate | {frame_rate} Hz ({frame_ms} ms) |
| Vocabulary | {vocab_size} BPE tokens |
| Decoder | TDT (token-and-duration transducer) |
| Sample rate | {sample_rate} Hz |

## Usage

This checkpoint **cannot be loaded by `nemo_toolkit` alone**. Its encoder `_target_` points at a
class that lives in the [DiCoP repository]({project_url}), and NeMo only resolves `_target_`s
inside the `nemo` package unless that check is relaxed.

```bash
git clone {project_url} && cd DiCoP
pip install -r requirements.txt

python infer.py \\
    --checkpoint {model_id} \\
    --rttm /path/to/rttms/ --audio-dir /path/to/audio/ \\
    --output hyp.stm
```

To drive the model directly:

```python
import sys
sys.path.insert(0, "/path/to/DiCoP")

from utils.nemo import allow_external_nemo_targets, register_legacy_nemo_aliases

allow_external_nemo_targets()
register_legacy_nemo_aliases()

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNO

model = EncDecRNNTBPEModelSTNO.from_pretrained("{model_id}")
```

`transcribe()` is deliberately disabled on this model. NeMo's transcription path cannot supply a
mask, and an unconditioned encoder returns a fluent transcript of whoever is loudest — which
looks correct but is not target-speaker output. Use `infer.py`, or `transcribe_stno()` with an
STNO mask you build yourself (see `src/data/stno.py`).

## Citation

```
@misc{{klement2026descriptionchime9mcorecchallenge,
      title={{BUT System Description for CHiME-9 MCoRec Challenge}},
      author={{Dominik Klement and Alexander Polok and Nguyen Hai Phong and Prachi Singh and Lukáš Burget}},
      year={{2026}},
      eprint={{2604.27436}},
      archivePrefix={{arXiv}},
      primaryClass={{eess.AS}},
      url={{https://arxiv.org/abs/2604.27436}},
}}
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt / .nemo, or an NGC/HF model id.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the bundle into.")
    parser.add_argument(
        "--repo-id",
        help="Target Hub repo, ORG/NAME. Decides the archive filename, which NeMo requires to "
        "match the repository name. Defaults to the output directory's name.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Base config YAML.")
    parser.add_argument(
        "-O",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override a config key before the model is built, e.g. "
        "-O model.encoder.use_pytorch_sdpa=true. Repeatable. Applied after the checkpoint's own "
        "config, so it decides the architecture that gets exported.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Directory holding tokenizer.model / vocab.txt / tokenizer.vocab. By default these "
        "come from the source .nemo if it carries them, else from `init_from_pretrained`.",
    )
    parser.add_argument("--license", default="cc-by-4.0", help="License for the model card. Default cc-by-4.0.")
    parser.add_argument("--model-card", type=Path, help="Use this file as the model card instead of generating one.")
    parser.add_argument(
        "--no-model-card", dest="model_card_enabled", action="store_false", help="Do not write a model card."
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the bundle. Requires --repo-id.")
    parser.add_argument("--private", action="store_true", help="Create the repo private. Only on first push.")
    parser.add_argument("--commit-message", default="Upload DiCoP checkpoint", help="Commit message for the upload.")
    parser.add_argument("--dry-run", action="store_true", help="Build the bundle but do not upload; list the files.")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip restoring the archive back and comparing it against the source model. The "
        "check roughly doubles peak memory, but it is the only thing that proves the bundle works.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing bundle.")

    args = parser.parse_args()
    if args.push_to_hub and not args.repo_id:
        parser.error("--push-to-hub needs --repo-id ORG/NAME")
    if args.repo_id and args.repo_id.count("/") != 1:
        parser.error(f"--repo-id must be ORG/NAME, got {args.repo_id!r}")
    if args.model_card and not args.model_card_enabled:
        parser.error("--model-card and --no-model-card are mutually exclusive")
    return args


def tokenizer_pieces(tokenizer) -> list:
    """The tokenizer's vocabulary as an ordered list of pieces.

    Not `tokenizer.tokenizer.get_vocab()`: that method is monkeypatched onto the sentencepiece
    processor by NeMo's `_setup_tokenizer`, so it exists on a tokenizer taken from a restored
    model but not on one built straight from a `.model` file. `ids_to_tokens` works on both.
    """
    return [tokenizer.ids_to_tokens([index])[0] for index in range(tokenizer.vocab_size)]


def extract_tokenizer_files(archive_path: Path, dest: Path) -> dict:
    """Copy a `.nemo`'s tokenizer artifacts into `dest` under their canonical names.

    The names inside an archive are uuid-prefixed (`705f..._tokenizer.model`), and which file
    plays which role is recorded in the config rather than the filename, so the config is what
    is read here.

    Returns:
        `{config key: written path}`, empty when the archive registers no tokenizer.
    """
    with tarfile.open(archive_path, "r:*") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        if "model_config.yaml" not in members:
            raise KeyError(f"{archive_path} has no model_config.yaml; contents: {sorted(members)}")

        config = OmegaConf.load(archive.extractfile(members["model_config.yaml"]))
        tokenizer_cfg = config.get("tokenizer") or {}

        written = {}
        for key, filename, _ in TOKENIZER_ARTIFACTS:
            source = tokenizer_cfg.get(key)
            if not source:
                continue

            member_name = Path(str(source).split("nemo:")[-1]).name
            if member_name not in members:
                raise KeyError(f"{archive_path} references {member_name} but does not contain it.")

            target = dest / filename
            with archive.extractfile(members[member_name]) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written[key] = target

    # A partial tokenizer is worse than none: falling through to `init_from_pretrained` would
    # silently pair one archive's weights with another's vocabulary.
    if written:
        missing = [name for key, name, required in TOKENIZER_ARTIFACTS if required and key not in written]
        if missing:
            raise KeyError(f"{archive_path} registers a tokenizer but not {', '.join(missing)}.")

    return written


def resolve_tokenizer_dir(explicit: Path, checkpoint: str, cfg, dest: Path) -> Path:
    """Produce a directory holding the tokenizer files, trying the cheapest source first."""
    if explicit is not None:
        missing = [name for _, name, required in TOKENIZER_ARTIFACTS if required and not (explicit / name).is_file()]
        if missing:
            raise SystemExit(f"--tokenizer-dir {explicit} is missing {', '.join(missing)}")
        logging.info("Using tokenizer files from %s", explicit)
        return explicit

    source = Path(checkpoint).expanduser()
    if source.is_file() and source.suffix == ".nemo" and extract_tokenizer_files(source, dest):
        logging.info("Took the tokenizer from the source archive %s", source)
        return dest

    pretrained = cfg.get("init_from_pretrained")
    if not pretrained:
        raise SystemExit(
            "The checkpoint carries no tokenizer and the config has no `init_from_pretrained` to "
            "take one from. Pass --tokenizer-dir."
        )

    logging.info("Taking the tokenizer from %s", pretrained)
    archive = EncDecRNNTBPEModelSTNO.from_pretrained(pretrained, return_model_file=True)
    if not extract_tokenizer_files(Path(archive), dest):
        raise SystemExit(f"{pretrained} registers no tokenizer artifacts; pass --tokenizer-dir.")
    return dest


def attach_tokenizer(model, tokenizer_dir: Path) -> None:
    """Register the tokenizer files so `save_to` bundles them into the archive.

    Registration is done by hand rather than through `_setup_tokenizer`, which would build a
    fresh tokenizer and bind it to `model.tokenizer` while `model.decoding`, `model.wer` and
    `model.meeteval_mt_wer` still held the old object — and the decoder and joint were sized from
    that old object's vocabulary. Registering the artifacts leaves the loaded model untouched.

    At save time `_handle_artifacts` copies each file into the archive under a uuid name and
    `_update_artifact_paths` writes `nemo:<uuid>_<name>` into the stored config, creating the
    keys, so `cfg.tokenizer` only has to carry `dir` and `type` here. On restore, `cfg.tokenizer`
    is present and no tokenizer object is passed, so the model builds one from the archive.
    """
    # `dir` is deliberately not the directory the files are read from here: the artifacts are
    # registered by absolute path, and `dir` is only a fallback for a tokenizer whose files are
    # not registered. Writing the real path would bake a dead local temp directory into a
    # published config for no benefit.
    with open_dict(model.cfg):
        model.cfg.tokenizer = OmegaConf.create({"dir": ".", "type": "bpe"})

    for key, filename, required in TOKENIZER_ARTIFACTS:
        path = tokenizer_dir / filename
        if not path.is_file():
            if required:
                raise SystemExit(f"Tokenizer directory {tokenizer_dir} is missing {filename}")
            logging.warning("No %s alongside the tokenizer; the archive will restore without it.", filename)
            continue
        model.register_artifact(f"tokenizer.{key}", str(path))

    # A mismatched tokenizer produces a model that loads cleanly and decodes garbage, so the
    # bundled files are checked against the tokenizer the weights were actually trained with.
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    bundled = tokenizer_pieces(SentencePieceTokenizer(model_path=str(tokenizer_dir / "tokenizer.model")))
    loaded = tokenizer_pieces(model.tokenizer)
    if bundled != loaded:
        raise SystemExit(
            f"The tokenizer in {tokenizer_dir} does not match the model's: "
            f"{len(bundled)} vs {len(loaded)} pieces. Pass the right --tokenizer-dir."
        )
    logging.info("Bundled tokenizer matches the model's (%d pieces)", len(loaded))


def write_model_card(path: Path, model, model_id: str, license_name: str) -> None:
    num_params = sum(parameter.numel() for parameter in model.parameters())
    path.write_text(
        MODEL_CARD_TEMPLATE.format(
            license=license_name,
            base_model=BASE_MODEL,
            project_url=PROJECT_URL,
            model_id=model_id,
            num_params=f"{num_params / 1e6:.0f}M",
            num_layers=model.cfg.encoder.n_layers,
            d_model=model.cfg.encoder.d_model,
            frame_rate=f"{1 / model.embed_duration:.1f}",
            frame_ms=f"{model.embed_duration * 1000:.0f}",
            vocab_size=model.tokenizer.vocab_size,
            sample_rate=model.cfg.sample_rate,
        ),
        encoding="utf-8",
    )


def verify_round_trip(archive: Path, model) -> None:
    """Restore the archive and require it to reproduce the source model exactly."""
    register_legacy_nemo_aliases()

    logging.info("Verifying %s by restoring it", archive)
    restored = EncDecRNNTBPEModelSTNO.restore_from(str(archive), map_location="cpu")

    original_state, restored_state = model.state_dict(), restored.state_dict()
    missing = sorted(set(original_state) - set(restored_state))
    unexpected = sorted(set(restored_state) - set(original_state))
    if missing or unexpected:
        raise SystemExit(f"State dict mismatch. Missing: {missing[:5]}. Unexpected: {unexpected[:5]}.")

    differing = [key for key, value in original_state.items() if not torch.equal(value, restored_state[key])]
    if differing:
        raise SystemExit(f"{len(differing)} tensors differ after the round trip, e.g. {differing[:5]}")

    if tokenizer_pieces(restored.tokenizer) != tokenizer_pieces(model.tokenizer):
        raise SystemExit("The restored tokenizer differs from the source model's.")

    for attribute in ("audio_downsampling_factor", "embed_duration"):
        if getattr(restored, attribute) != getattr(model, attribute):
            raise SystemExit(
                f"{attribute} differs after the round trip: "
                f"{getattr(model, attribute)} vs {getattr(restored, attribute)}"
            )

    logging.info(
        "Verified: %d tensors identical, %d tokenizer pieces identical",
        len(original_state),
        restored.tokenizer.vocab_size,
    )


def push(output_dir: Path, repo_id: str, private: bool, commit_message: str) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:
        raise SystemExit(
            f"Not authenticated with HuggingFace ({type(exc).__name__}). "
            f"Run `hf auth login`, or set HF_TOKEN."
        ) from exc

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        url = api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message=commit_message)
    except HfHubHTTPError as exc:
        raise SystemExit(f"Upload failed: {exc}") from exc

    print(f"\nPushed to https://huggingface.co/{repo_id}")
    print(f"Commit: {url}")


def main():
    args = parse_args()

    # NeMo finds a model in a Hub repo by downloading `{repo-name}.nemo` from its root; any other
    # name falls through to a different, slower code path that unpacks the whole repository.
    stem = args.repo_id.split("/")[-1] if args.repo_id else args.output_dir.name
    archive = args.output_dir / f"{stem}.nemo"

    if archive.exists() and not args.force:
        raise SystemExit(f"{archive} already exists (--force to overwrite)")

    # Before the model is loaded, so an unwritable destination costs seconds rather than minutes.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_inference_config(args.checkpoint, args.config)
    model = load_asr_model(cfg, overrides=args.overrides)

    # The tokenizer files must outlive `save_to`, which is what reads them.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer_dir = resolve_tokenizer_dir(args.tokenizer_dir, args.checkpoint, cfg, Path(tmp_dir))
        attach_tokenizer(model, tokenizer_dir)
        model.save_to(str(archive))

    logging.info("Wrote %s (%.1f MiB)", archive, archive.stat().st_size / 1024**2)

    # A copy of the config next to the archive, so the export can be inspected without untarring.
    with tarfile.open(archive, "r:*") as bundle:
        member = next(m for m in bundle.getmembers() if Path(m.name).name == "model_config.yaml")
        with bundle.extractfile(member) as src, open(args.output_dir / "model_config.yaml", "wb") as out:
            shutil.copyfileobj(src, out)

    model_id = args.repo_id or stem
    if args.model_card:
        shutil.copyfile(args.model_card, args.output_dir / "README.md")
    elif args.model_card_enabled:
        write_model_card(args.output_dir / "README.md", model, model_id, args.license)

    if not args.skip_verify:
        verify_round_trip(archive, model)

    print(f"\nBundle: {args.output_dir}")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path.name}  ({path.stat().st_size / 1024**2:.1f} MiB)")

    if args.push_to_hub and not args.dry_run:
        push(args.output_dir, args.repo_id, args.private, args.commit_message)
        print(f"\nUse it with: --checkpoint {args.repo_id}")
    elif args.push_to_hub:
        print(f"\nDry run: would upload the files above to https://huggingface.co/{args.repo_id}")
    else:
        target = args.repo_id or f"ORG/{stem}"
        print(f"\nTo publish: rerun with --repo-id {target} --push-to-hub")


if __name__ == "__main__":
    main()
