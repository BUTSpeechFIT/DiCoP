"""Small NeMo interop helpers."""

from pathlib import Path
from typing import Union

import torch

from nemo.utils import logging

__all__ = ['allow_external_nemo_targets', 'load_ptl_ckpt', 'register_legacy_nemo_aliases']


def allow_external_nemo_targets() -> None:
    """Let NeMo instantiate `_target_`s that live in this repository.

    NeMo restricts `_target_` resolution to modules inside the `nemo` package. DiCoP's
    checkpoints and configs point at `src.model.modules.stno_encoder.ConformerEncoderSTNO`,
    so that check has to be relaxed before any config is instantiated or restored.
    """
    import nemo.core.classes

    nemo.core.classes.common._is_target_allowed = lambda _: True


def register_legacy_nemo_aliases() -> None:
    """Make checkpoints written by the in-NeMo implementation loadable unchanged.

    Those checkpoints record `_target_: nemo.collections.asr.modules.ConformerEncoderSTNO`,
    a class that only existed in the NeMo fork. Binding the name to DiCoP's equivalent lets
    both `.ckpt` and `.nemo` files instantiate without rewriting their stored configs. The
    parameter names are identical, so the weights line up exactly.
    """
    import nemo.collections.asr.modules as nemo_asr_modules

    from src.model.modules.stno_encoder import ConformerEncoderSTNO

    if not hasattr(nemo_asr_modules, 'ConformerEncoderSTNO'):
        nemo_asr_modules.ConformerEncoderSTNO = ConformerEncoderSTNO
        logging.debug("Registered ConformerEncoderSTNO alias for legacy checkpoints")


def load_ptl_ckpt(model, checkpoint_path: Union[str, Path], strict: bool = True) -> None:
    """Load weights from a Lightning `.ckpt` into an already-constructed model.

    Replaces NeMo's `maybe_init_from_pretrained_checkpoint` for the `init_from_ptl_ckpt` case.
    NeMo's version loads with `weights_only=True` (which cannot read these checkpoints, whose
    `hyper_parameters` hold an OmegaConf object) and `strict=False` (which hides a genuinely
    mismatched checkpoint). Doing it here avoids patching the installed NeMo package.

    Args:
        model: The model to load into.
        checkpoint_path: Path to a `.ckpt` file.
        strict: Require an exact parameter match. Keep this on unless you are deliberately
            loading a partially-matching checkpoint.
    """
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' not in checkpoint:
        raise KeyError(f"{checkpoint_path} has no 'state_dict' key; is it a Lightning checkpoint?")

    result = model.load_state_dict(checkpoint['state_dict'], strict=strict)
    if result.missing_keys:
        logging.warning("Missing keys when loading %s: %s", checkpoint_path, result.missing_keys)
    if result.unexpected_keys:
        logging.warning("Unexpected keys when loading %s: %s", checkpoint_path, result.unexpected_keys)
    logging.info("Restored weights from Lightning checkpoint %s", checkpoint_path)
