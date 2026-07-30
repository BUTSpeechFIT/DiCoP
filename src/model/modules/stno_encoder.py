"""FastConformer encoder with per-layer FDDT diarization conditioning.

`ConformerEncoderSTNO` is a thin subclass of NeMo's stock `ConformerEncoder`. The whole
delta is:

  1. one `FDDT` block per Conformer layer, built in `__init__`
  2. two extra input ports, `stno_mask` (B, 4, T) and `stno_mask_length` (B,)
  3. the mask applied to the hidden states immediately before each Conformer layer

Because the FDDT modules are registered as `fddts.{i}.…`, the state dict stays a strict
superset of stock Parakeet's: `nvidia/parakeet-tdt-0.6b-v2` loads with the FDDT parameters
reported as the only missing keys, and trained STNO checkpoints load `strict=True`.

`forward` and `forward_internal` are copied verbatim from `nemo_toolkit==2.7.3`
(`nemo/collections/asr/modules/conformer_encoder.py`, lines 548-760), with the two
conditioning hunks marked `# DiCoP:`. `tests/test_encoder_parity.py` asserts that with
`stno_mask=None` this encoder is bit-identical to the stock one, which is what catches
upstream drift when NeMo is bumped.

Two alternatives to copying `forward_internal` were considered and rejected:
  - wrapping each `ConformerLayer` in an FDDT-applying module renames the state dict keys
    to `layers.{i}.layer.…`, so no existing checkpoint would load;
  - a `forward_pre_hook` on each layer is semantically equivalent (stochastic depth captures
    `original_signal` before the layer call, interctc capture happens after it, and adapters
    live inside `ConformerLayer.forward`), but needs hidden mutable `self._stno_mask` state
    between `forward` and the layer calls, which is non-reentrant and invisible to a reader.
"""

from collections import OrderedDict

import random

import torch
import torch.nn as nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import LengthsType, MaskType, NeuralType

from src.model.modules.fddt import FDDT

__all__ = ['ConformerEncoderSTNO']


class ConformerEncoderSTNO(ConformerEncoder):
    """`ConformerEncoder` conditioned on a frame-level silence/target/non-target/overlap mask.

    Args:
        fddt_non_target_rate: initial scale of the silence and non-target transforms. The
            target and overlap transforms always start at identity, so at initialization the
            encoder passes target frames through and attenuates the rest.
        fddt_is_diagonal: use element-wise (diagonal) transforms rather than full
            `d_model x d_model` matrices.
        fddt_bias_only: use additive biases only, ignoring `fddt_is_diagonal`.
        fddt_use_silence, fddt_use_target, fddt_use_non_target, fddt_use_overlap: disable
            individual branches; a disabled branch passes its frames through unchanged.

    The defaults reproduce the values that were hardcoded in the original NeMo
    implementation, so checkpoints trained there instantiate identically.
    """

    def __init__(
        self,
        *args,
        fddt_non_target_rate: float = 0.5,
        fddt_is_diagonal: bool = True,
        fddt_bias_only: bool = False,
        fddt_use_silence: bool = True,
        fddt_use_target: bool = True,
        fddt_use_non_target: bool = True,
        fddt_use_overlap: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.fddts = nn.ModuleList(
            [
                FDDT(
                    self.d_model,
                    non_target_rate=fddt_non_target_rate,
                    is_diagonal=fddt_is_diagonal,
                    bias_only=fddt_bias_only,
                    use_silence=fddt_use_silence,
                    use_target=fddt_use_target,
                    use_overlap=fddt_use_overlap,
                    use_non_target=fddt_use_non_target,
                )
                for _ in range(len(self.layers))
            ]
        )

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        types = OrderedDict(super().input_types)
        types["stno_mask"] = NeuralType(('B', 'S', 'T'), MaskType(), optional=True)
        types["stno_mask_length"] = NeuralType(tuple('B'), LengthsType(), optional=True)
        return types

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        types = OrderedDict(super().input_types_for_export)
        types["stno_mask"] = NeuralType(('B', 'S', 'T'), MaskType(), optional=True)
        types["stno_mask_length"] = NeuralType(tuple('B'), LengthsType(), optional=True)
        return types

    @typecheck()
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        stno_mask=None,
        stno_mask_length=None,
    ):
        """
        Forward function for the ConformerEncoderSTNO accepting an audio signal, its corresponding
        length, and the diarization-derived STNO mask.
        The `audio_signal` input supports two formats depending on the `bypass_pre_encode` boolean flag.
        This determines the required format of the input variable `audio_signal`:
        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)

        `stno_mask` has shape (batch, 4, n_encoder_frames) and holds, per frame, the
        silence / target / non-target / overlap indicators. It is reconciled against the actual
        subsampled length inside `forward_internal`. Passing `None` disables conditioning
        entirely, which makes this encoder equivalent to the stock `ConformerEncoder`.
        `stno_mask_length` is accepted for interface symmetry with the dataloader but unused:
        the mask is zero-padded on the right, and padded frames are masked out downstream anyway.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"If bypass_pre_encode is False, audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
            )
        if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
            raise ValueError(
                f"If bypass_pre_encode is True, audio_signal should have shape "
                f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
            )

        if bypass_pre_encode:
            self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
            stno_mask=stno_mask,
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        stno_mask=None,
    ):
        """
        The `audio_signal` input supports two formats depending on the `bypass_pre_encode` boolean flag.
        This determines the required format of the input variable `audio_signal`:
        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)

        `bypass_pre_encode=True` is used in cases where frame-level, context-independent embeddings are
        needed to be saved or reused (e.g., speaker cache in streaming speaker diarization).
        """
        # DiCoP: the mask is aligned to one contiguous window of encoder frames at a single frame
        # rate. Streaming caches offset that window and `reduction` changes the frame rate mid-stack,
        # so both would silently misalign the conditioning. Fail loudly instead.
        if stno_mask is not None:
            if cache_last_channel is not None:
                raise NotImplementedError(
                    "ConformerEncoderSTNO does not support cache-aware streaming with an STNO mask."
                )
            if self.reduction_position is not None:
                raise NotImplementedError(
                    "ConformerEncoderSTNO does not support encoder `reduction` with an STNO mask."
                )

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)

            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)
                # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
                if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                    audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                    length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

            if self.reduction_position is not None and cache_last_channel is not None:
                raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        # DiCoP: the dataloader rasterizes the mask by rounding the sample count up to a whole
        # number of encoder frames, which can disagree with the subsampling output by a frame or
        # two. Reconcile here rather than trying to predict the subsampling arithmetic.
        if stno_mask is not None:
            if stno_mask.shape[-1] > audio_signal.shape[1]:
                stno_mask = stno_mask[:, :, : audio_signal.shape[1]]
            elif stno_mask.shape[-1] < audio_signal.shape[1]:
                stno_mask = nn.functional.pad(stno_mask, (0, audio_signal.shape[1] - stno_mask.shape[-1]))

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None

            # DiCoP: apply the diarization conditioning. Note this happens *after*
            # `original_signal` is captured, so stochastic depth keeps its pre-FDDT baseline.
            if stno_mask is not None:
                audio_signal = self.fddts[lth](audio_signal, stno_mask)

            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length
