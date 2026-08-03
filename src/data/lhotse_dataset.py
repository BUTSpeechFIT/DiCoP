"""Multi-talker target-speaker ASR dataset over a Lhotse CutSet.

The Lhotse counterpart of `src/data/dataset.py`. It emits the **same 8-tuple**, so the collate,
the training and validation steps, and the multi-talker metric are all shared unchanged between
the two manifest formats; only the container differs.

One dataset item is a **(cut, target speaker)** pair, enumerated exhaustively for both training
and validation. This differs from the NeMo-manifest dataset, which samples one random target
speaker per session per training epoch: a Lhotse epoch therefore covers every speaker of every
cut, and is correspondingly longer.

`MonoCut` and `MixedCut` are both accepted, the same pair `infer.py` decodes. A `MixedCut` (the
LibriMix / LibriSpeechMix mixtures) names no audio file — it is a recipe for summing its tracks —
so `load_cut_audio` renders it rather than reading a path, and `channel_selector` and `trim` do
not apply to it. It also has no `recording`, so it becomes its own scoring session by `cut.id`.
`MultiCut` is still rejected; see `require_supported_cut`.

See `src/data/lhotse_utils.py` for why Lhotse's own dataloading stack is not used and how cut
times map onto the NeMo manifest's.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType, VoidType
from nemo.utils import logging

from src.data.dataset import _TokenizerWrapper, speech_collate_fn
from src.data.lhotse_utils import cut_session_id, load_cut_audio, load_cutset, require_supported_cut
from src.data.stno import (
    SpeechSegment,
    create_stno_masks,
    frame_rate_from_downsampling_factor,
    num_encoder_frames,
    speaker_activity_mask,
)

__all__ = ['LhotseToBPEAndSTNODataset']


def _segments_from_tokenized(segments: List[Dict]) -> List[SpeechSegment]:
    """Tokenized supervisions as `SpeechSegment`s, for mask rasterization.

    Built from the tokenized list rather than from the cut so that supervisions dropped for
    tokenizing to empty are absent from the mask too, matching the NeMo-manifest path.
    """
    return [
        SpeechSegment(start=s['start'], duration=s['duration'], speaker=s['speaker']) for s in segments
    ]


class LhotseToBPEAndSTNODataset(Dataset):
    """Yields `(audio, tokens, stno_mask)` for one (cut, target speaker) pair.

    Args:
        manifest_filepath: Path to a Lhotse CutSet (`.jsonl.gz` / `.jsonl`).
        tokenizer: A `nemo.collections.common.tokenizers.TokenizerSpec` subclass.
        sample_rate: Sample rate to resample loaded audio to.
        int_values: If True, load samples as 32-bit integers.
        augmentor: Optional `AudioAugmentor` applied to the loaded waveform. `MonoCut` only —
            a `MixedCut` is rendered by Lhotse rather than by the featurizer, which is where the
            augmentor lives, so it is skipped there and warned about.
        max_duration: Drop cuts longer than this.
        min_duration: Drop cuts shorter than this.
        max_utts: Limit the number of cuts kept (0 means no limit).
        trim: Trim leading/trailing silence. `MonoCut` only: a `MixedCut`'s own bounds already
            define its extent.
        use_start_end_token: Add [BOS]/[EOS] around the target transcript.
        channel_selector: Select or average channels of multi-channel audio. `MonoCut` only:
            a rendered `MixedCut` is mono by construction.
        audio_downsampling_factor: Samples per encoder frame; sets the STNO mask frame rate.
        text_norm_type: Text normalization applied before tokenization (`None` disables it).
        val: Accepted for interface symmetry with the NeMo-manifest dataset. It does not change
            enumeration here, since every (cut, speaker) pair is emitted either way.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'stno_masks': NeuralType(('B', 'S', 'T'), MaskType()),
            'stno_mask_length': NeuralType(tuple('B'), LengthsType()),
            'utterance_id': NeuralType(tuple('B'), VoidType()),
            'speaker_id': NeuralType(tuple('B'), VoidType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, List[str]],
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        channel_selector: Optional[ChannelSelectorType] = None,
        audio_downsampling_factor: int = 1,
        text_norm_type: Optional[str] = 'whisper_nsf',
        val: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = 0

        if isinstance(manifest_filepath, (list, tuple)):
            if len(manifest_filepath) != 1:
                raise ValueError(
                    f"A Lhotse dataset takes a single CutSet path, got {len(manifest_filepath)}."
                )
            manifest_filepath = manifest_filepath[0]

        # Materialized once for O(1) indexing. Cuts are metadata only, so this is cheap.
        self.cuts = list(load_cutset(manifest_filepath))
        self.parser = _TokenizerWrapper(tokenizer, text_norm_type)

        # Tokenize up front: the transcript, the STNO mask and the scoring references all read
        # the same tokenized supervisions, so they cannot disagree.
        self.tokenized: Dict[int, List[Dict]] = {}
        self.session_ids: Dict[int, str] = {}
        self.offsets: Dict[int, float] = {}
        self.spk_cut_list: List[tuple] = []

        num_duration_filtered = 0
        num_empty = 0
        num_mixed = 0
        total_duration = 0.0

        for index, cut in enumerate(self.cuts):
            require_supported_cut(cut)

            if (min_duration is not None and cut.duration < min_duration) or (
                max_duration is not None and cut.duration > max_duration
            ):
                num_duration_filtered += 1
                continue

            segments = []
            for supervision in cut.supervisions:
                token_ids = self.parser(supervision.text or '')
                if not token_ids:
                    continue
                segments.append(
                    {
                        'start': supervision.start,
                        'duration': supervision.duration,
                        'speaker': supervision.speaker,
                        'text': token_ids,
                    }
                )

            if not segments:
                num_empty += 1
                continue

            # `index` is the cut's position in the CutSet, assigned before any filtering, so a
            # dropped cut leaves a hole rather than renumbering the rest. That keeps the emitted
            # utt_id joinable with `segments_collection`, exactly as NeMo manifest ids survive
            # the removal of untokenizable rows.
            self.tokenized[index] = segments
            self.session_ids[index] = cut_session_id(cut)
            self.offsets[index] = cut.start
            total_duration += cut.duration
            if type(cut).__name__ == 'MixedCut':
                num_mixed += 1

            for speaker in sorted({segment['speaker'] for segment in segments}):
                self.spk_cut_list.append((index, speaker))

            if max_utts and len(self.tokenized) >= max_utts:
                break

        if num_duration_filtered:
            logging.info("Filtered %d cuts by duration", num_duration_filtered)
        if num_empty:
            logging.info("Dropped %d cuts with no tokenizable transcript", num_empty)
        if num_mixed:
            logging.info("%d of the kept cuts are MixedCut and are rendered by Lhotse", num_mixed)
        # The augmentor lives on the featurizer, which the MixedCut branch of `load_cut_audio`
        # does not go through. Saying so is cheap; silently training unaugmented would not be.
        if num_mixed and augmentor is not None:
            logging.warning(
                "An augmentor is configured but %d cuts are MixedCut, which are rendered by "
                "Lhotse rather than by the featurizer the augmentor is attached to. Those cuts "
                "will not be augmented.",
                num_mixed,
            )
        logging.info(
            "CutSet loaded with %d cuts totalling %.2f hours -> %d (cut, speaker) items",
            len(self.tokenized),
            total_duration / 3600,
            len(self.spk_cut_list),
        )

        self.val = val
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.channel_selector = channel_selector
        self.audio_downsampling_factor = audio_downsampling_factor
        self.frame_rate = frame_rate_from_downsampling_factor(sample_rate, audio_downsampling_factor)

    def __len__(self):
        return len(self.spk_cut_list)

    def __getitem__(self, index):
        cut_index, target_speaker = self.spk_cut_list[index]
        cut = self.cuts[cut_index]
        segments = self.tokenized[cut_index]

        features = load_cut_audio(cut, self.featurizer, self.channel_selector, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()

        speakers = sorted({segment['speaker'] for segment in segments})
        target_idx = speakers.index(target_speaker)

        num_frames = num_encoder_frames(int(fl), self.audio_downsampling_factor)
        spk_activity_mask = speaker_activity_mask(
            _segments_from_tokenized(segments), speakers, num_frames, self.frame_rate
        )
        stno_mask = create_stno_masks(spk_activity_mask, target_idx)

        # The transcript is every segment of the target speaker, in cut order.
        target_tokens = []
        for segment in segments:
            if segment['speaker'] == target_speaker:
                target_tokens.extend(segment['text'])

        t, tl = self._add_bos_eos(target_tokens)

        return (
            f,
            fl,
            torch.tensor(t).long(),
            torch.tensor(tl).long(),
            stno_mask,
            torch.tensor(stno_mask.shape[-1]).long(),
            cut_index,
            target_idx,
        )

    def _add_bos_eos(self, token_ids: List[int]):
        t, tl = token_ids, len(token_ids)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1
        return t, tl

    @property
    def segments_collection(self):
        """Reference segments in the shape `MeetevalMTWER.compute` expects.

        Mirrors the NeMo path's `manifest_processor.collection`: `.id` matches the emitted
        `utt_id`, and `.offset` is `cut.start` so a recording's windows land on one timeline.
        """
        return [
            SimpleNamespace(
                id=cut_index,
                session_id=self.session_ids[cut_index],
                offset=self.offsets[cut_index],
                text_tokens=segments,
            )
            for cut_index, segments in sorted(self.tokenized.items())
        ]

    def _collate_fn(self, batch):
        return speech_collate_fn(batch, pad_id=self.pad_id)
