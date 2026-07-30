"""Train the diarization-conditioned Parakeet target-speaker ASR model.

    python train.py +exp_dir="exps/"

The tokenizer and the initial encoder/decoder/joint weights both come from
`init_from_pretrained` (a pretrained Parakeet); only the FDDT blocks start from scratch.

Validation reports multi-talker cpWER/tcpWER and writes `ref.stm` / `hyp.stm` under the
experiment's log directory. For inference on new audio, see `infer.py`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from utils.nemo import allow_external_nemo_targets, load_ptl_ckpt

allow_external_nemo_targets()

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNO


def maybe_load_pretrained_model(model_name: str | None) -> ASRModel | None:
    if not model_name:
        return None
    return ASRModel.from_pretrained(model_name=model_name, map_location="cpu")


@hydra_runner(config_path="conf", config_name="dicop")
def main(cfg):
    logging.info("Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    pretrained_model = maybe_load_pretrained_model(cfg.get("init_from_pretrained"))

    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecRNNTBPEModelSTNO(
        cfg=cfg.model,
        trainer=trainer,
        tokenizer=pretrained_model.tokenizer if pretrained_model is not None else None,
    )

    if pretrained_model is not None:
        # Non-strict on purpose: the FDDT blocks have no counterpart in stock Parakeet and
        # should be the *only* missing keys. Anything else means a topology mismatch.
        missing, unexpected = asr_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        unexpected_fddt = [key for key in missing if '.fddts.' not in key]
        logging.info("Missing keys (%d, of which %d are not FDDT): %s", len(missing), len(unexpected_fddt), missing)
        logging.info("Unexpected keys: %s", unexpected)
        if unexpected_fddt or unexpected:
            logging.warning(
                "Warm start from %s did not match cleanly. Non-FDDT missing keys: %s. Unexpected keys: %s",
                cfg.get("init_from_pretrained"),
                unexpected_fddt,
                unexpected,
            )
        del pretrained_model

    if cfg.get("init_from_ptl_ckpt"):
        load_ptl_ckpt(asr_model, cfg.init_from_ptl_ckpt, strict=True)

    if cfg.get("evaluate_at_start", False):
        trainer.validate(asr_model)

    trainer.fit(asr_model)

    if cfg.model.get("test_ds") is not None and cfg.model.test_ds.get("manifest_filepath") is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
