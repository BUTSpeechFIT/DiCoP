"""Multi-talker target-speaker ASR dataset.

Ported from `nemo/collections/asr/data/audio_to_text_and_stno.py`, keeping only the code
this model actually runs: the collate function, the manifest processor, and the BPE dataset.
NeMo's `_AudioTextDataset` / `AudioToBPEAndSTNODataset` split exists upstream only so the
character-level dataset can share the base; with that gone the two are merged here.

One dataset item is a **(session, target speaker)** pair:

  - `train`: one random target speaker per session per epoch, so an epoch sees every session
    once and the speaker choice is resampled each time.
  - `val=True`: every (session, speaker) pair exactly once, so validation is deterministic
    and covers all speakers. This is why `setup_validation_data` / `setup_test_data` pass
    `val=True`.

The transcript is the concatenation of the target speaker's segments; the STNO mask is built
from *all* speakers' segments so the model knows where the interfering speech is.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType, VoidType
from nemo.utils import logging

from src.data.collections import ASRAudioTextSTNO
from src.data.stno import (
    SpeechSegment,
    create_stno_masks,
    frame_rate_from_downsampling_factor,
    num_encoder_frames,
    speaker_activity_mask,
)
from src.data.text_norm import get_text_norm

__all__ = ['AudioToBPEAndSTNODataset', 'ASRManifestProcessor', 'speech_collate_fn']


def speech_collate_fn(batch, pad_id: int):
    """Collate a batch of (audio, tokens, STNO mask) tuples, zero-padding to the batch maximum.

    Args:
        batch: sequence of 8-tuples as returned by `AudioToBPEAndSTNODataset.__getitem__`.
        pad_id: token id used to pad transcripts.

    Returns:
        `(audio_signal, audio_lengths, tokens, tokens_lengths, stno_masks, stno_mask_lengths,
        utt_ids, spk_ids)`, where `utt_ids` indexes the manifest row and `spk_ids` indexes the
        target speaker within that session's sorted speaker list.
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) != 8:
        raise ValueError(f"Expected 8 tensors in the batch, got {len(packed_batch)}")

    _, audio_lengths, _, tokens_lengths, _, stno_mask_lengths, utt_ids, spk_ids = packed_batch

    max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()
    max_stno_mask_len = max(stno_mask_lengths).item()

    audio_signal, tokens, stno_masks = [], [], []
    for sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, _, _ in batch:
        sig_len = sig_len.item()
        if sig_len < max_audio_len:
            sig = torch.nn.functional.pad(sig, (0, max_audio_len - sig_len))
        audio_signal.append(sig)

        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            tokens_i = torch.nn.functional.pad(tokens_i, (0, max_tokens_len - tokens_i_len), value=pad_id)
        tokens.append(tokens_i)

        stno_mask_i_len = stno_mask_i_len.item()
        if stno_mask_i_len < max_stno_mask_len:
            stno_mask_i = torch.nn.functional.pad(stno_mask_i, (0, max_stno_mask_len - stno_mask_i_len))
        stno_masks.append(stno_mask_i)

    return (
        torch.stack(audio_signal),
        torch.stack(audio_lengths),
        torch.stack(tokens),
        torch.stack(tokens_lengths),
        torch.stack(stno_masks),
        torch.stack(stno_mask_lengths),
        torch.tensor(utt_ids, dtype=torch.int32),
        torch.tensor(spk_ids, dtype=torch.int32),
    )


class ASRManifestProcessor:
    """Loads a multi-talker manifest and drops sessions with no usable transcript."""

    def __init__(
        self,
        manifest_filepath: Union[str, List[str]],
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
    ):
        self.parser = parser

        self.collection = ASRAudioTextSTNO(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            parse_func=manifest_parse_func,
        )

        # A session whose every segment tokenized to nothing cannot supply a target transcript.
        ids_to_pop = [i for i in range(len(self.collection)) if not self.collection[i].text_tokens]
        if ids_to_pop:
            logging.info(
                "Dropping %d/%d sessions with no tokenizable transcript", len(ids_to_pop), len(self.collection)
            )
            for i in reversed(ids_to_pop):
                self.collection.pop(i)
        logging.info("Manifest loaded with %d sessions", len(self.collection))

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        return self.process_text_by_sample(self.collection[index])

    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        return self.process_text_by_sample(self.collection[manifest_idx])

    def process_text_by_sample(self, text_tokens: List[int]) -> Tuple[List[int], int]:
        t, tl = text_tokens, len(text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


class _TokenizerWrapper:
    """Applies text normalization before tokenizing, so refs and hyps are normalized alike."""

    def __init__(self, tokenizer, text_norm_type: Optional[str] = 'whisper_nsf'):
        self.is_aggregate = isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer)
        self._tokenizer = tokenizer
        self.text_norm = get_text_norm(text_norm_type) if text_norm_type else (lambda x: x)

    def __call__(self, *args):
        if isinstance(args[0], List) and self.is_aggregate:
            t = []
            for span in args[0]:
                t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
            return t

        args = tuple(self.text_norm(x) for x in args)
        return self._tokenizer.text_to_ids(*args)


class AudioToBPEAndSTNODataset(Dataset):
    """Yields `(audio, tokens, stno_mask)` for one (session, target speaker) pair.

    Args:
        manifest_filepath: Path(s) to the JSON Lines manifest; comma-separated string allowed.
        tokenizer: A `nemo.collections.common.tokenizers.TokenizerSpec` subclass.
        sample_rate: Sample rate to resample loaded audio to.
        int_values: If True, load samples as 32-bit integers.
        augmentor: Optional `AudioAugmentor` applied to the loaded waveform.
        max_duration: Drop sessions longer than this.
        min_duration: Drop sessions shorter than this.
        max_utts: Limit the number of sessions.
        trim: Whether to trim leading/trailing silence.
        use_start_end_token: Add [BOS]/[EOS] around the target transcript.
        channel_selector: Select or average channels of multi-channel audio.
        manifest_parse_func: Optional custom manifest line parser.
        audio_downsampling_factor: Samples per encoder frame; sets the STNO mask frame rate.
        text_norm_type: Text normalization applied before tokenization (`None` disables it).
        val: If True, enumerate every (session, speaker) pair instead of sampling one speaker.
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
        manifest_parse_func: Optional[Callable] = None,
        audio_downsampling_factor: int = 1,
        text_norm_type: Optional[str] = 'whisper_nsf',
        val: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(",")

        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=_TokenizerWrapper(tokenizer, text_norm_type),
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            manifest_parse_func=manifest_parse_func,
        )

        self.val = val
        if val:
            # Deterministic, exhaustive: one item per (session, speaker).
            self.per_spk_collection = [
                (sample, speaker)
                for sample in self.manifest_processor.collection
                for speaker in sorted({segment['speaker'] for segment in sample.text_tokens})
            ]

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.channel_selector = channel_selector
        self.audio_downsampling_factor = audio_downsampling_factor
        self.frame_rate = frame_rate_from_downsampling_factor(sample_rate, audio_downsampling_factor)

    def __len__(self):
        return len(self.per_spk_collection) if self.val else len(self.manifest_processor.collection)

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        return self._process_sample(index)

    def _process_sample(self, index):
        if self.val:
            sample, target_speaker = self.per_spk_collection[index]
        else:
            sample = self.manifest_processor.collection[index]
            target_speaker = None

        offset = sample.offset if sample.offset is not None else 0

        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        speakers = sorted({segment['speaker'] for segment in sample.text_tokens})
        if target_speaker is None:
            target_speaker = random.choice(speakers)
        target_idx = speakers.index(target_speaker)

        num_frames = num_encoder_frames(int(fl), self.audio_downsampling_factor)
        segments = [
            SpeechSegment(start=segment['start'], duration=segment['duration'], speaker=segment['speaker'])
            for segment in sample.text_tokens
        ]
        spk_activity_mask = speaker_activity_mask(segments, speakers, num_frames, self.frame_rate)
        stno_mask = create_stno_masks(spk_activity_mask, target_idx)

        # The transcript is every segment of the target speaker, in manifest order.
        target_tokens = []
        for segment in sample.text_tokens:
            if segment['speaker'] == target_speaker:
                target_tokens.extend(segment['text'])

        t, tl = self.manifest_processor.process_text_by_sample(target_tokens)

        return (
            f,
            fl,
            torch.tensor(t).long(),
            torch.tensor(tl).long(),
            stno_mask,
            torch.tensor(stno_mask.shape[-1]).long(),
            sample.id,
            target_idx,
        )

    def _collate_fn(self, batch):
        return speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)
