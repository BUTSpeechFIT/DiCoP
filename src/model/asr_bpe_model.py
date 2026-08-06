"""Sub-word (BPE) variant of the diarization-conditioned RNNT/TDT model.

Inherits `(EncDecRNNTModelSTNO, ASRBPEMixin)` rather than NeMo's `EncDecRNNTBPEModel`, for
two reasons. First, `EncDecRNNTBPEModel.__init__` hard-requires a `cfg.tokenizer` section,
whereas DiCoP injects the tokenizer object taken from the pretrained Parakeet model. Second,
with a diamond base the C3 order puts `EncDecRNNTBPEModel` ahead of `EncDecRNNTModel`, and
since `EncDecRNNTModelSTNO` defines no `__init__`, attribute lookup would fall straight
through to the tokenizer-requiring constructor we are trying to avoid.
"""

from typing import Dict, Optional

from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils

from src.data.dataloader import get_stno_dataloader
from src.metrics.meeteval_mt_wer import MeetevalMTWER
from src.model.asr_model import EncDecRNNTModelSTNO

__all__ = ['EncDecRNNTBPEModelSTNO']


class EncDecRNNTBPEModelSTNO(EncDecRNNTModelSTNO, ASRBPEMixin):
    """Diarization-conditioned RNNT/TDT model with sub-word tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, tokenizer=None):
        """
        Args:
            cfg: The `model` section of the config.
            trainer: The Lightning trainer, when training.
            tokenizer: A pre-built tokenizer, normally taken from the pretrained Parakeet
                checkpoint. When omitted, `cfg.tokenizer` is used to build one.
        """
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg and tokenizer is None:
            raise ValueError("`cfg` must have a `tokenizer` config, or a `tokenizer` must be passed.")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        if tokenizer is None:
            self._setup_tokenizer(cfg.tokenizer)
        else:
            self.tokenizer = tokenizer

        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Size the decoder and joint to the tokenizer before the modules are instantiated.
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        super().__init__(cfg=cfg, trainer=trainer)

        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        # Samples of audio per encoder frame: 16000 * 0.01 * 8 = 1280, i.e. 12.5 Hz / 80 ms.
        # This sets both the STNO mask frame rate and the scale for decoded word timestamps.
        self.audio_downsampling_factor = int(
            self.cfg.sample_rate * self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor
        )
        self.embed_duration = self.audio_downsampling_factor / self.cfg.sample_rate  # seconds

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # The multi-talker metrics are not built here: there is one per evaluation dataloader and
        # the dataloaders are set up by `ModelPT.__init__` above, before `self.decoding` exists.
        # `EncDecRNNTModelSTNO._build_eval_metrics` builds them at each evaluation epoch start.

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

    def _build_meeteval_metric(self) -> MeetevalMTWER:
        meeteval_cfg = self._cfg.get('meeteval', {}) or {}
        return MeetevalMTWER(
            decoding=self.decoding,
            dist_sync_on_step=False,
            log_prediction=self._cfg.get('log_prediction', False),
            embed_duration=self.embed_duration,
            output_per_word_timestamps=meeteval_cfg.get('output_per_word_timestamps', True),
            tcp_collar=meeteval_cfg.get('tcp_collar', 5),
            text_norm_type=self._cfg.get('text_norm', 'whisper_nsf'),
        )

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose: bool = True):
        """Rebuild the decoding object, and the metrics that hold a reference to it.

        `infer.py` uses this to turn on `compute_timestamps`, which the STM output needs.

        Args:
            decoding_cfg: New decoding config; `None` re-applies the current one.
            verbose: Log the resulting config.
        """
        if decoding_cfg is None:
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Fill in every hyperparameter the decoding config declares.
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # The multi-talker metrics decode through `self.decoding` too, but they are rebuilt at
        # every evaluation epoch start, so they pick the new strategy up on their own.

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.joint.temperature = decoding_cfg.get('temperature', 1.0)

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info("Changed decoding strategy to \n%s", OmegaConf.to_yaml(self.cfg.decoding))

    def _setup_dataloader_from_config(self, config: Optional[Dict], val: bool = False):
        """Build a `(session, target speaker)` dataloader. See `src/data/dataloader.py`."""
        return get_stno_dataloader(
            config=config,
            tokenizer=self.tokenizer,
            audio_downsampling_factor=int(
                self.cfg.sample_rate * self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor
            ),
            text_norm_type=self._cfg.get('text_norm', 'whisper_nsf'),
            val=val,
        )

    def _setup_transcribe_dataloader(self, config: Dict):
        raise NotImplementedError(
            "EncDecRNNTBPEModelSTNO requires diarization conditioning. Use infer.py instead."
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """No pretrained models are published under this class yet."""
        return []
