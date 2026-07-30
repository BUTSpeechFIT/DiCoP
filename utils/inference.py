"""Model loading and the batched decoding runtime used by `infer.py`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.nemo import allow_external_nemo_targets

allow_external_nemo_targets()

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNO
from utils.audio import create_audio_featurizer
from utils.nemo import register_legacy_nemo_aliases
from utils.stm import WordSpan

__all__ = ['InferenceRuntime', 'WordSpan', 'load_asr_model', 'select_device']

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "dicop.yaml"

# Config sections that describe the *training run* rather than the architecture. They are
# dropped from a checkpoint's stored config so the YAML's versions win.
CHECKPOINT_CFG_KEYS_TO_DROP = (
    "train_ds",
    "validation_ds",
    "test_ds",
    "nemo_version",
    "target",
    "labels",
    "tokenizer",
    "decoder",
    "joint",
    "decoding",
    # Present in configs inherited from the hybrid-CTC template but unused by a pure RNNT model.
    "aux_ctc",
    "interctc",
)

ENCODER_TARGET = "src.model.modules.stno_encoder.ConformerEncoderSTNO"


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def load_inference_config(
    checkpoint: Union[str, Path, None] = None,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> DictConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if checkpoint is not None:
        with open_dict(cfg):
            cfg.checkpoint = str(checkpoint)
    return cfg


def _normalize_model_source(model_source: Union[str, Path]) -> Union[Path, str]:
    """Return a `Path` for a local file, or the string unchanged for an NGC/HF model id."""
    source_text = str(model_source).strip()
    if not source_text:
        raise ValueError("Model source must be specified.")

    local_path = Path(source_text).expanduser()
    if local_path.is_file():
        return local_path.resolve()
    return source_text


def _disable_dataset_configs(cfg: DictConfig) -> None:
    """Blank out the dataset manifests so no dataloader is built during inference.

    `ModelPT.__init__` calls `setup_training_data` / `setup_validation_data`, which resolve the
    dataset configs. The shipped YAML marks the training manifest mandatory (`???`), so leaving
    it in place raises `MissingMandatoryValue` here. `get_stno_dataloader` returns `None` for a
    null manifest, which is what makes this safe.
    """
    for section in ("train_ds", "validation_ds", "test_ds"):
        if section in cfg.model:
            with open_dict(cfg.model[section]):
                cfg.model[section].manifest_filepath = None


def _apply_config_overrides(cfg: DictConfig, overrides: Optional[Sequence[str]] = None) -> None:
    """Merge `key=value` dotlist overrides into the config.

    Applied *after* a checkpoint's stored config has been merged in, so an override wins over
    the architecture the checkpoint was saved with. That ordering is the point: without it a
    `.ckpt` silently pins every `model.encoder.*` key and editing the YAML has no effect.
    """
    if not overrides:
        return

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--override expects KEY=VALUE, got {override!r}")

    with open_dict(cfg):
        cfg.merge_with(OmegaConf.from_dotlist(list(overrides)))
    for override in overrides:
        logging.info("Config override applied: %s", override)


def _apply_encoder_overrides(
    cfg: DictConfig,
    att_context_size: Optional[Sequence[int]] = None,
    use_sdpa: Optional[bool] = None,
) -> None:
    """Adjust attention settings before the encoder is built.

    Both of these decide whether a long-form forward pass fits in memory, and neither can be
    changed after construction, so they are applied to the config rather than the module.
    """
    with open_dict(cfg.model.encoder):
        if att_context_size is not None:
            cfg.model.encoder.att_context_size = list(att_context_size)
            logging.info("Encoder att_context_size set to %s", list(att_context_size))
        if use_sdpa is not None:
            cfg.model.encoder.use_pytorch_sdpa = bool(use_sdpa)
            logging.info("Encoder use_pytorch_sdpa set to %s", bool(use_sdpa))


def _read_lightning_checkpoint(path: Path):
    """Extract `(model_config, state_dict)` from a Lightning `.ckpt`."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError(
            f"{path} does not look like a Lightning checkpoint "
            f"(needs 'hyper_parameters' and 'state_dict')."
        )
    return checkpoint["hyper_parameters"]["cfg"], checkpoint["state_dict"]


def _read_nemo_archive(path: Path):
    """Extract `(model_config, state_dict)` from a `.nemo` archive.

    Unpacked by hand rather than through `restore_from` because these archives contain only
    `model_config.yaml` and `model_weights.ckpt`: the tokenizer is injected at construction
    time rather than registered as an artifact, so it is never bundled, and `restore_from`
    fails trying to resolve the training machine's tokenizer path. Reading the two members
    directly lets the tokenizer come from `init_from_pretrained`, as it does for a `.ckpt`.
    """
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with tarfile.open(path, "r:*") as archive:
            members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
            for required in ("model_config.yaml", "model_weights.ckpt"):
                if required not in members:
                    raise KeyError(
                        f"{path} is missing {required}; contents: {sorted(members)}"
                    )
                archive.extract(members[required], path=tmp_path, filter="data")

        config_path = next(tmp_path.rglob("model_config.yaml"))
        weights_path = next(tmp_path.rglob("model_weights.ckpt"))
        model_cfg = OmegaConf.load(config_path)
        state_dict = torch.load(weights_path, weights_only=False, map_location="cpu")

    return model_cfg, state_dict


def load_asr_model(
    cfg: DictConfig,
    att_context_size: Optional[Sequence[int]] = None,
    use_sdpa: Optional[bool] = None,
    overrides: Optional[Sequence[str]] = None,
) -> EncDecRNNTBPEModelSTNO:
    """Build the model from `cfg.checkpoint`.

    Three checkpoint kinds are supported:

      - a Lightning `.ckpt` or a `.nemo` archive: the architecture config is taken from the
        checkpoint and merged over the YAML, so the model matches the weights even if the YAML
        has since changed; the tokenizer comes from `cfg.init_from_pretrained`, and the weights
        load with `strict=True`.
      - an NGC or HuggingFace model id: fetched and restored by NeMo.

    Args:
        cfg: The full config, with `cfg.checkpoint` set.
        att_context_size: Optional `[left, right]` attention context override.
        use_sdpa: Optional override for scaled dot-product attention.
        overrides: Optional `key=value` dotlist entries, applied over the checkpoint's own
            config so they decide the architecture the weights are loaded into.

    Returns:
        The model, on CPU.
    """
    # Checkpoints from the in-NeMo implementation name the encoder by its old module path.
    register_legacy_nemo_aliases()

    model_source = _normalize_model_source(cfg.checkpoint)

    if isinstance(model_source, str):
        logging.info("Restoring pretrained model %s", model_source)
        model = EncDecRNNTBPEModelSTNO.from_pretrained(model_source, map_location="cpu")
        _warn_if_overrides_ignored(att_context_size, use_sdpa, overrides, model)
        return model

    if model_source.suffix == ".nemo":
        logging.info("Loading .nemo archive %s", model_source)
        stored_cfg, state_dict = _read_nemo_archive(model_source)
    else:
        logging.info("Loading Lightning checkpoint %s", model_source)
        stored_cfg, state_dict = _read_lightning_checkpoint(model_source)

    checkpoint_cfg = dict(stored_cfg)
    for key in CHECKPOINT_CFG_KEYS_TO_DROP:
        checkpoint_cfg.pop(key, None)

    # Point the encoder at this repository's class. Harmless when the alias above already
    # resolves the old path, but it keeps the resulting config self-describing.
    checkpoint_cfg["encoder"]["_target_"] = ENCODER_TARGET

    cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(checkpoint_cfg))
    _disable_dataset_configs(cfg)
    # Overrides go on last, so they beat the checkpoint's stored architecture, and the dedicated
    # flags go on last of all, so an explicit --att-context-size still wins over a dotlist entry.
    _apply_config_overrides(cfg, overrides)
    _apply_encoder_overrides(cfg, att_context_size, use_sdpa)

    pretrained_name = cfg.get("init_from_pretrained")
    if not pretrained_name:
        raise ValueError(
            "Loading a .ckpt needs `init_from_pretrained` in the config to supply the tokenizer."
        )
    logging.info("Taking the tokenizer from %s", pretrained_name)
    tokenizer = ASRModel.from_pretrained(model_name=pretrained_name, map_location="cpu").tokenizer

    model = EncDecRNNTBPEModelSTNO(cfg=cfg.model, tokenizer=tokenizer)
    model.load_state_dict(state_dict, strict=True)
    logging.info("Checkpoint weights loaded (strict)")
    return model


def _warn_if_overrides_ignored(att_context_size, use_sdpa, overrides, model) -> None:
    """`.nemo` and pretrained restores build the encoder themselves, before overrides apply.

    The attention settings are the exception: `change_attention_model` rebuilds the positional
    encoding and the attention layers and carries the trained weights across, so those can still
    be honoured here. Every other override is already baked into the restored module and can
    only be reported.
    """
    remaining = dict(entry.split("=", 1) for entry in (overrides or []) if "=" in entry)

    def take(key):
        """Pop `key`, typed the way the YAML would read it, so `[256,256]` arrives as a list."""
        if key not in remaining:
            return None
        return OmegaConf.select(OmegaConf.from_dotlist([f"{key}={remaining.pop(key)}"]), key)

    self_attention_model = take("model.encoder.self_attention_model")
    override_context = take("model.encoder.att_context_size")
    if att_context_size is None and override_context is not None:
        att_context_size = list(override_context)

    if self_attention_model is not None or att_context_size is not None:
        model.change_attention_model(
            self_attention_model=self_attention_model,
            att_context_size=list(att_context_size) if att_context_size is not None else None,
        )
        logging.info(
            "Encoder attention changed post-restore: self_attention_model=%s, att_context_size=%s",
            self_attention_model or "unchanged",
            list(att_context_size) if att_context_size is not None else "unchanged",
        )

    if remaining:
        logging.warning(
            "These overrides cannot be applied to an already-restored model and were ignored: "
            "%s. They are constructor arguments; use a .ckpt if you need them.",
            ", ".join(sorted(remaining)),
        )

    if use_sdpa is not None and bool(use_sdpa) != bool(model.cfg.encoder.get("use_pytorch_sdpa", False)):
        logging.warning(
            "--use-sdpa cannot be applied to an already-restored model; it is a constructor "
            "argument. The encoder keeps use_pytorch_sdpa=%s. Convert the checkpoint to .ckpt, "
            "or edit the archive's config, if you need to change it.",
            model.cfg.encoder.get("use_pytorch_sdpa", False),
        )


class InferenceRuntime:
    """Holds the loaded model plus everything needed to turn audio into timed words."""

    def __init__(
        self,
        cfg: DictConfig,
        device_name: str = "auto",
        precision: str = "bf16",
        att_context_size: Optional[Sequence[int]] = None,
        use_sdpa: Optional[bool] = None,
        overrides: Optional[Sequence[str]] = None,
    ):
        self.cfg = cfg
        self.device = select_device(device_name)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.model = load_asr_model(
            cfg, att_context_size=att_context_size, use_sdpa=use_sdpa, overrides=overrides
        )

        # Word timestamps are what place hypotheses on the time axis; alignments are not needed
        # and are costly on long-form audio.
        with open_dict(self.model.cfg.decoding):
            self.model.cfg.decoding.compute_timestamps = True
            self.model.cfg.decoding.preserve_alignments = False
        self.model.change_decoding_strategy(self.model.cfg.decoding, verbose=False)

        self.model.to(self.device)
        self.model.eval()

        self.sample_rate = int(self.cfg.model.sample_rate)
        self.audio_downsampling_factor = int(self.model.audio_downsampling_factor)
        self.embed_duration = float(self.model.embed_duration)
        self.featurizer = create_audio_featurizer(self.sample_rate)

    @torch.inference_mode()
    def preprocess(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-mel features once for a whole recording.

        The features do not depend on which speaker is being targeted, so a session with N
        speakers computes them once rather than N times.

        Returns:
            `(processed_signal, processed_signal_length)`, batch size 1.
        """
        signal = audio.to(self.device).unsqueeze(0)
        length = torch.tensor([audio.shape[0]], dtype=torch.int64, device=self.device)
        return self.model.preprocessor(input_signal=signal, length=length)

    @torch.inference_mode()
    def decode(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
        stno_masks: torch.Tensor,
    ) -> List[List[WordSpan]]:
        """Decode one batch of target speakers sharing the same features.

        Args:
            processed_signal: `(1, D, T)` features, expanded internally to the batch size.
            processed_signal_length: `(1,)` feature frame count.
            stno_masks: `(B, 4, T_enc)`, one mask per target speaker.

        Returns:
            Per target speaker, the list of decoded words with times relative to the start of
            the supplied features.
        """
        batch_size = stno_masks.shape[0]
        # `expand` rather than `repeat`: the features are read-only here, so there is no need
        # to materialize B copies of a potentially very long spectrogram.
        signal = processed_signal.expand(batch_size, -1, -1)
        lengths = processed_signal_length.expand(batch_size)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            hypotheses = self.model.transcribe_stno(
                stno_mask=stno_masks.to(self.device),
                processed_signal=signal,
                processed_signal_length=lengths,
                return_hypotheses=True,
            )

        return [self._hypothesis_to_spans(hyp) for hyp in hypotheses]

    def _hypothesis_to_spans(self, hypothesis) -> List[WordSpan]:
        timestamps = getattr(hypothesis, "timestamp", None)
        if not timestamps or "word" not in timestamps:
            raise RuntimeError(
                "The model returned no word timestamps. `decoding.compute_timestamps` must be true."
            )

        spans = []
        for word_info in timestamps["word"]:
            text = str(word_info.get("word", "")).strip()
            if not text or text == "<unk>":
                continue
            start = float(word_info.get("start_offset", 0)) * self.embed_duration
            end = float(word_info.get("end_offset", 0)) * self.embed_duration
            spans.append(WordSpan(text=text, start=start, end=max(start, end)))
        return spans
