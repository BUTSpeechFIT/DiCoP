"""Diarization-conditioned RNNT/TDT target-speaker ASR model.

`EncDecRNNTModelSTNO` subclasses NeMo's `EncDecRNNTModel` and changes only what the STNO
conditioning requires:

  - `forward` threads the (B, 4, T) STNO mask into the encoder;
  - the training/validation batches carry the mask, the manifest row id and the target
    speaker index alongside the usual audio and transcript;
  - validation scores multi-talker cpWER/tcpWER with `MeetevalMTWER` instead of token WER,
    one dataset at a time, and writes `ref.stm` / `hyp.stm` per dataset per validation epoch;
  - `transcribe()` is disabled because NeMo's transcription path cannot supply a mask, and
    silently decoding without conditioning would produce plausible-looking wrong output.
    `transcribe_stno()` is the conditioned replacement, used by `infer.py`.

Everything else - the module construction, the TDT loss, optimizer setup, export - is
inherited from NeMo unchanged.
"""

import copy
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

from src.data.lhotse_utils import named_manifests
from src.metrics.meeteval_mt_wer import pool_results

__all__ = ['EncDecRNNTModelSTNO']

# Where each evaluation stage keeps its state on the model. The dataloaders, NeMo's own display
# names and its "which loader owns the unprefixed metrics" index are NeMo's attributes; the
# dataset names and the per-dataloader metrics are this model's.
EVAL_STAGES = {
    'validation': {
        'prefix': 'val',
        'dataloaders': '_validation_dl',
        'nemo_names': '_validation_names',
        'dl_idx': '_val_dl_idx',
        'names': '_validation_dataset_names',
        'metrics': '_validation_metrics',
    },
    'test': {
        'prefix': 'test',
        'dataloaders': '_test_dl',
        'nemo_names': '_test_names',
        'dl_idx': '_test_dl_idx',
        'names': '_test_dataset_names',
        'metrics': '_test_metrics',
    },
}


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

            logs_dict = {
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
                logs_dict.update({'training_batch_wer': scores.float() / words})

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

            logs_dict = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                logs_dict.update({'training_batch_wer': wer})

        self.log_dict(logs_dict)

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

        logs_dict = {}

        if self.compute_eval_loss:
            decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            logs_dict['val_loss'] = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

        # Multi-talker scoring replaces token WER: hypotheses are accumulated per
        # (session, speaker) here and scored once per epoch in `_eval_epoch_end`.
        self._eval_metric(dataloader_idx).update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            utt_ids=utt_ids,
            spk_ids=spk_ids,
        )

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return logs_dict

    def _stm_output_dir(self, name: Optional[str] = None) -> str:
        """Where one dataset's `ref.stm` / `hyp.stm` go, in `run_inference.sh`'s `{name}/` layout."""
        log_dir = self.trainer.log_dir if self.trainer is not None else '.'
        parts = [log_dir or '.', f'preds_{self.current_epoch}_{self.trainer.global_step}']
        if name:
            parts.append(name)
        path = os.path.join(*parts)
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

    def _score_eval_dataloader(self, outputs, stage: str, dataloader_idx: int, name: Optional[str]):
        """Score one dataset's accumulated hypotheses and dump its STMs.

        Returns the log entries for this dataset, keyed `val/<name>/cp_wer` and so on, together
        with the raw cpWER and tcpWER counts, which `_eval_epoch_end` sums into the pooled number
        checkpoints are selected on.
        """
        stage_attrs = EVAL_STAGES[stage]
        prefix = stage_attrs['prefix']
        scope = f'{prefix}/{name}' if name else prefix

        save_stm_path = self._stm_output_dir(name)
        logging.info("Saving %s predictions to %s", scope, save_stm_path)

        metric = self._eval_metrics(stage)[dataloader_idx]
        cp_res, tcp_res = metric.compute(
            self._dataloader_collection(getattr(self, stage_attrs['dataloaders']), dataloader_idx),
            save_stm_path=save_stm_path,
        )
        metric.reset()

        logs = {}
        if self.compute_eval_loss:
            logs[f'{scope}/loss'] = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
        for metric_name, res in (('cp', cp_res), ('tcp', tcp_res)):
            for key in ('wer', 'ins', 'del', 'sub', 'len'):
                logs[f'{scope}/{metric_name}_{key}'] = float(res[key])

        return logs, cp_res, tcp_res

    def _eval_epoch_end(self, stage: str):
        """Score every dataset of `stage` separately, then log them and their pooled aggregate.

        Replaces `ModelPT.on_validation_epoch_end`, which prefixes the second and later
        dataloaders' metrics as `<name>_val/cp_wer` and has nowhere to put a number pooled across
        them. `exp_manager` selects checkpoints on `val/cp_wer`, so that number has to exist
        however many datasets are configured — with one it is simply that dataset's own.
        """
        stage_attrs = EVAL_STAGES[stage]
        prefix = stage_attrs['prefix']
        outputs = self.validation_step_outputs if stage == 'validation' else self.test_step_outputs
        if not outputs:
            return {}

        # A single dataloader reports a flat list of step outputs, several report one list each.
        per_dataloader = [outputs] if isinstance(outputs[0], dict) else list(outputs)
        names = getattr(self, stage_attrs['names'], None) or [None] * len(per_dataloader)

        logs = {}
        if self.compute_eval_loss:
            losses = [step[f'{prefix}_loss'] for outs in per_dataloader for step in outs]
            logs[f'{prefix}_loss'] = torch.stack(losses).mean()

        results = []
        for dataloader_idx, dataloader_outputs in enumerate(per_dataloader):
            dataloader_logs, cp_res, tcp_res = self._score_eval_dataloader(
                dataloader_outputs, stage, dataloader_idx, names[dataloader_idx]
            )
            logs.update(dataloader_logs)
            results.append((cp_res, tcp_res))
            dataloader_outputs.clear()  # free memory

        if any(names):
            for metric_name, pooled in (
                ('cp', pool_results([cp_res for cp_res, _ in results])),
                ('tcp', pool_results([tcp_res for _, tcp_res in results])),
            ):
                for key in ('wer', 'ins', 'del', 'sub', 'len'):
                    logs[f'{prefix}/{metric_name}_{key}'] = float(pooled[key])

        # `sync_dist` matches the `sync_metrics=True` NeMo passes here; the scores are already
        # reduced across ranks by the metric itself, so it only averages equal values.
        self.log_dict(logs, on_epoch=True, sync_dist=True)
        return {'log': logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """NeMo's per-dataloader hook, kept to its contract.

        `on_validation_epoch_end` scores directly instead, because it also needs the raw counts
        each dataset contributes to the pooled aggregate.
        """
        names = getattr(self, EVAL_STAGES['validation']['names'], None)
        logs, _, _ = self._score_eval_dataloader(
            outputs, 'validation', dataloader_idx, names[dataloader_idx] if names else None
        )
        return {'log': logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        """See `multi_validation_epoch_end`."""
        names = getattr(self, EVAL_STAGES['test']['names'], None)
        logs, _, _ = self._score_eval_dataloader(
            outputs, 'test', dataloader_idx, names[dataloader_idx] if names else None
        )
        return {'log': logs}

    def on_validation_epoch_end(self):
        """See `_eval_epoch_end`."""
        # All `ASRModel.on_validation_epoch_end` does before delegating to the logging this
        # replaces: validation runs inside training, which continues with the graphs disabled.
        self.disable_cuda_graphs()
        return self._eval_epoch_end('validation')

    def on_test_epoch_end(self):
        """See `_eval_epoch_end`. Unlike validation, nothing resumes after testing."""
        return self._eval_epoch_end('test')

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
        self._build_eval_metrics('validation')

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._build_eval_metrics('test')

    def _eval_stage(self) -> str:
        """Which evaluation stage the trainer is currently running."""
        return 'test' if self._trainer is not None and self.trainer.testing else 'validation'

    def _eval_metric(self, dataloader_idx: int):
        """The hypothesis accumulator for one dataloader of the running evaluation stage.

        `validation_pass` serves both `validation_step` and `test_step`, which is why the stage is
        read off the trainer rather than passed in.
        """
        return self._eval_metrics(self._eval_stage())[dataloader_idx]

    def _eval_metrics(self, stage: str):
        """One `MeetevalMTWER` per dataloader of `stage`, built on first use.

        Lightning runs the dataloaders one after another but calls the epoch end only once they
        have all finished, so a single shared accumulator would pool every dataset's hypotheses
        and then score that pool against each dataset's references in turn. One accumulator per
        dataloader keeps them apart.
        """
        metrics = getattr(self, EVAL_STAGES[stage]['metrics'], None)
        return metrics if metrics is not None else self._build_eval_metrics(stage)

    def _build_eval_metrics(self, stage: str):
        """Build `stage`'s accumulators, one per dataloader.

        Not built in `setup_multiple_validation_data`: NeMo calls that from `ModelPT.__init__`,
        before `self.decoding` — which the metric decodes through — exists. Rebuilding them at
        every epoch start also keeps them in step with `change_decoding_strategy`.
        """
        stage_attrs = EVAL_STAGES[stage]
        names = getattr(self, stage_attrs['names'], None)
        dataloaders = getattr(self, stage_attrs['dataloaders'], None) or []
        count = len(names) if names else max(len(dataloaders), 1)

        # A plain list, not a `ModuleList`: these hold no parameters and their metric states are
        # non-persistent, so registering them would only add and remove state_dict entries as the
        # configured datasets change.
        metrics = [self._build_meeteval_metric().to(self.device) for _ in range(count)]
        setattr(self, stage_attrs['metrics'], metrics)
        return metrics

    def _setup_named_eval_dataloaders(self, config: Optional[Union[DictConfig, Dict]], stage: str) -> None:
        """Build one dataloader per named evaluation dataset.

        Replaces `nemo.utils.model_utils.resolve_{validation,test}_dataloaders`. That also turns a
        list of manifests into one dataloader each, but it names them itself and NeMo then logs
        `<stem>_val/cp_wer` where this model logs `val/<stem>/cp_wer`; and it has no mapping form,
        which is the only way to name a dataset or to pool several manifests under one name.
        """
        stage_attrs = EVAL_STAGES[stage]
        setattr(self, stage_attrs['dataloaders'], None)
        setattr(self, stage_attrs['names'], None)
        setattr(self, stage_attrs['metrics'], None)
        setattr(self, stage_attrs['nemo_names'], None)
        setattr(self, stage_attrs['dl_idx'], 0)

        if config is None or config.get('manifest_filepath') is None:
            return

        # Preserved as written, before the per-dataset copies below narrow it to one manifest.
        self._update_dataset_config(dataset_name=stage, config=config)

        pairs = named_manifests(config['manifest_filepath'])
        setup = self.setup_validation_data if stage == 'validation' else self.setup_test_data

        dataloaders = []
        try:
            # `_update_dataset_config` is a no-op in this mode, which is what stops the per-dataset
            # copies from replacing the config just preserved.
            self._multi_dataset_mode = True
            for _, manifest in pairs:
                dataset_config = copy.deepcopy(config)
                dataset_config['manifest_filepath'] = manifest
                setup(dataset_config)
                dataloaders.append(getattr(self, stage_attrs['dataloaders']))
        finally:
            self._multi_dataset_mode = False

        names = [name for name, _ in pairs]
        logging.info("Evaluating %s on %d dataset(s): %s", stage, len(names), ', '.join(names))
        setattr(self, stage_attrs['dataloaders'], dataloaders)
        setattr(self, stage_attrs['names'], names)
        # NeMo's own convention for the same thing, in case anything reads it: its display names
        # end in an underscore, since it uses them as a metric-key prefix.
        setattr(self, stage_attrs['nemo_names'], [f'{name}_' for name in names])

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

    def setup_multiple_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """See `_setup_named_eval_dataloaders`."""
        self._setup_named_eval_dataloaders(val_data_config, 'validation')

    def setup_multiple_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """See `_setup_named_eval_dataloaders`."""
        self._setup_named_eval_dataloaders(test_data_config, 'test')

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
