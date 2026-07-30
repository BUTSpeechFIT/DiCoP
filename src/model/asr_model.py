"""Diarization-conditioned RNNT/TDT target-speaker ASR model.

`EncDecRNNTModelSTNO` subclasses NeMo's `EncDecRNNTModel` and changes only what the STNO
conditioning requires:

  - `forward` threads the (B, 4, T) STNO mask into the encoder;
  - the training/validation batches carry the mask, the manifest row id and the target
    speaker index alongside the usual audio and transcript;
  - validation scores multi-talker cpWER/tcpWER with `MeetevalMTWER` instead of token WER,
    and writes `ref.stm` / `hyp.stm` per validation epoch;
  - `transcribe()` is disabled because NeMo's transcription path cannot supply a mask, and
    silently decoding without conditioning would produce plausible-looking wrong output.
    `transcribe_stno()` is the conditioned replacement, used by `infer.py`.

Everything else - the module construction, the TDT loss, optimizer setup, export - is
inherited from NeMo unchanged.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.classes.common import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import LengthsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['EncDecRNNTModelSTNO']


class EncDecRNNTModelSTNO(EncDecRNNTModel):
    """Base class for diarization-conditioned encoder-decoder RNNT models."""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        types = dict(super().input_types)
        types["stno_mask"] = NeuralType(('B', 'S', 'T'), MaskType(), optional=True)
        types["stno_mask_length"] = NeuralType(tuple('B'), LengthsType(), optional=True)
        return types

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        stno_mask=None,
        stno_mask_length=None,
    ):
        """Encoder forward pass, conditioned on the diarization-derived STNO mask.

        As in the base class this is only the first of the three RNNT forward steps; the
        prediction and joint networks are applied in `training_step` / `validation_pass`.

        Args:
            input_signal: Raw audio batch of shape (B, T).
            input_signal_length: Per-example sample counts, shape (B,).
            processed_signal: Pre-computed features of shape (B, D, T); mutually exclusive
                with `input_signal`.
            processed_signal_length: Per-example feature frame counts.
            stno_mask: (B, 4, T_enc) silence/target/non-target/overlap indicators at the
                encoder frame rate. `None` disables conditioning entirely.
            stno_mask_length: Per-example unpadded mask lengths, shape (B,).

        Returns:
            `(encoded, encoded_len)` with `encoded` of shape (B, D, T_enc).
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_length,
        )
        return encoded, encoded_len

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, _, _ = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Off by default: a greedy decode of every logged step is expensive, and cpWER on the
        # validation set is the metric this model is actually selected on.
        log_training_wer = self.cfg.get('log_training_wer', False)

        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if log_training_wer and (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            compute_wer = log_training_wer and (sample_id + 1) % log_every_n_steps == 0

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, _, _, stno_mask, stno_mask_len, utt_ids, spk_ids = batch

        encoded, encoded_len = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_len,
        )
        del signal

        best_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )
        return list(zip(utt_ids.cpu().tolist(), spk_ids.cpu().tolist(), best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, utt_ids, spk_ids = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        del signal

        tensorboard_logs = {}

        if self.compute_eval_loss:
            decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            tensorboard_logs['val_loss'] = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

        # Multi-talker scoring replaces token WER: hypotheses are accumulated per
        # (session, speaker) here and scored once per epoch in `multi_validation_epoch_end`.
        self.meeteval_mt_wer.update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            utt_ids=utt_ids,
            spk_ids=spk_ids,
        )

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def _stm_output_dir(self) -> str:
        log_dir = self.trainer.log_dir if self.trainer is not None else '.'
        path = os.path.join(log_dir or '.', f'preds_{self.current_epoch}_{self.trainer.global_step}')
        os.makedirs(path, exist_ok=True)
        return path

    def _dataloader_collection(self, dataloader, dataloader_idx: int):
        """The reference collection backing a (possibly multi-) dataloader.

        Both manifest formats expose the fields the multi-talker metric needs, under different
        names: the Lhotse dataset builds them on the fly from its cuts, the NeMo one keeps them
        on its manifest processor.
        """
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[dataloader_idx]
        dataset = dataloader.dataset

        if hasattr(dataset, 'segments_collection'):
            return dataset.segments_collection
        if hasattr(dataset, 'manifest_processor'):
            return dataset.manifest_processor.collection
        raise AttributeError(
            f"{type(dataset).__name__} exposes neither `segments_collection` nor "
            f"`manifest_processor`, so multi-talker scoring cannot find its references."
        )

    def _multi_epoch_end(self, outputs, dataloader, dataloader_idx: int, prefix: str):
        """Score the accumulated hypotheses and dump the STMs. Shared by val and test."""
        loss_key = f'{prefix}_loss'
        if self.compute_eval_loss:
            loss_mean = torch.stack([x[loss_key] for x in outputs]).mean()
            loss_log = {loss_key: loss_mean}
        else:
            loss_log = {}

        save_stm_path = self._stm_output_dir()
        logging.info("Saving %s predictions to %s", prefix, save_stm_path)

        cp_res, tcp_res = self.meeteval_mt_wer.compute(
            self._dataloader_collection(dataloader, dataloader_idx), save_stm_path=save_stm_path
        )
        self.meeteval_mt_wer.reset()

        metrics = {}
        for name, res in (('cp', cp_res), ('tcp', tcp_res)):
            for key in ('wer', 'ins', 'del', 'sub', 'len'):
                metrics[f'{prefix}/{name}_{key}'] = float(res[key])

        return {**loss_log, 'log': {**loss_log, **metrics}}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self._multi_epoch_end(outputs, self._validation_dl, dataloader_idx, prefix='val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self._multi_epoch_end(outputs, self._test_dl, dataloader_idx, prefix='test')

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        # Long-form sessions leave the allocator badly fragmented between phases.
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # Greedy decoding of variable-length long-form audio is incompatible with the captured
        # cuda graphs, which assume a fixed shape.
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self, attribute_path="decoding.decoding")
        torch.cuda.empty_cache()

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """Build the validation loader with one item per (session, speaker).

        `val=True` makes evaluation exhaustive and deterministic, which is what the
        multi-talker metrics need; training instead samples one target speaker per session.
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, val=True)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """Build the test loader with one item per (session, speaker). See `setup_validation_data`."""
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, val=True)

    def setup_optimizer_param_groups(self):
        """Optionally give the FDDT parameters their own learning rate.

        Note on reproducibility: the original NeMo implementation built this group but then
        discarded it on the code path every shipped config took, so `fddt_lr_multiplier` was
        silently a no-op and the released checkpoints trained at a uniform rate. The default of
        1.0 reproduces that; values above 1 are the intended-but-untested behaviour.

        At 1.0 this defers to the base class so a *single* parameter group is emitted, which
        keeps `exp_manager.resume_if_exists` able to load optimizer state saved by those runs.
        """
        multiplier = float(self.cfg.get('fddt_lr_multiplier', 1.0))
        if multiplier == 1.0:
            super().setup_optimizer_param_groups()
            return

        # '.fddts.' rather than 'fddt' so this cannot catch an unrelated parameter by substring.
        fddt_params = [p for name, p in self.named_parameters() if '.fddts.' in name]
        other_params = [p for name, p in self.named_parameters() if '.fddts.' not in name]

        base_lr = self.cfg.optim.lr
        logging.info(
            "Using a separate FDDT parameter group: %d FDDT tensors at lr=%s, %d others at lr=%s",
            len(fddt_params),
            base_lr * multiplier,
            len(other_params),
            base_lr,
        )
        self._optimizer_param_groups = [
            {'params': other_params},
            {'params': fddt_params, 'lr': base_lr * multiplier},
        ]

    """ Transcription related methods """

    def transcribe(self, *args, **kwargs):
        """Disabled: NeMo's transcription path cannot supply the STNO mask.

        Without a mask the encoder runs unconditioned and returns a fluent transcript of
        whoever is loudest, which looks correct but is not target-speaker output. Use
        `infer.py` (RTTM + audio directory -> STM) or `transcribe_stno` instead.
        """
        raise NotImplementedError(
            "EncDecRNNTModelSTNO requires diarization conditioning, which transcribe() cannot "
            "provide. Use infer.py for RTTM-driven inference, or transcribe_stno() directly."
        )

    @torch.inference_mode()
    def transcribe_stno(
        self,
        stno_mask: torch.Tensor,
        input_signal: Optional[torch.Tensor] = None,
        input_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        return_hypotheses: bool = True,
    ) -> List[Hypothesis]:
        """Decode a batch of target speakers given their STNO masks.

        Either `input_signal`/`input_signal_length` or `processed_signal`/
        `processed_signal_length` must be given. Callers decoding several target speakers of
        the same session should pass `processed_signal`, since the features do not depend on
        the target speaker and recomputing them per speaker is wasteful.

        Args:
            stno_mask: (B, 4, T_enc) mask, one row group per target speaker.
            return_hypotheses: Return `Hypothesis` objects, which carry the word timestamps
                that `infer.py` needs. With `False`, only the text is returned.

        Returns:
            One hypothesis per batch element, in input order.
        """
        was_training = self.training
        self.eval()
        try:
            encoded, encoded_len = self.forward(
                input_signal=input_signal,
                input_signal_length=input_signal_length,
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                stno_mask=stno_mask,
            )
            return self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=return_hypotheses,
            )
        finally:
            if was_training:
                self.train()
