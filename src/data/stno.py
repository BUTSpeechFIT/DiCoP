"""Silence / Target / Non-target / Overlap (STNO) mask construction.

This is the single source of truth for how diarization information is turned into the
4-channel frame-level mask that conditions the FDDT blocks inside the encoder. Both the
training dataset (`src/data/dataset.py`, from manifest segments) and inference
(`infer.py`, from an RTTM) build their masks here so the two paths cannot drift apart.

The mask lives at the *encoder* frame rate, i.e. after the FastConformer subsampling:

    frame_rate = sample_rate / audio_downsampling_factor
               = 16000 / (16000 * 0.01 * 8) = 16000 / 1280 = 12.5 Hz
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

__all__ = [
    'SpeechSegment',
    'create_stno_masks',
    'frame_rate_from_downsampling_factor',
    'num_encoder_frames',
    'speaker_activity_mask',
    'stno_mask_for_speaker',
]


@dataclass(frozen=True)
class SpeechSegment:
    """A single diarization segment: who speaks when, in seconds."""

    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def frame_rate_from_downsampling_factor(sample_rate: int, audio_downsampling_factor: int) -> float:
    """Encoder frame rate in Hz (12.5 for 16 kHz audio and 8x FastConformer subsampling)."""
    return sample_rate / audio_downsampling_factor


def num_encoder_frames(num_samples: int, audio_downsampling_factor: int) -> int:
    """Number of encoder frames a waveform of `num_samples` samples subsamples to.

    Rounds up, matching the dataset's original computation. The encoder reconciles any
    remaining off-by-one against the actual subsampled length, see `ConformerEncoderSTNO`.
    """
    remainder = num_samples % audio_downsampling_factor
    if remainder != 0:
        num_samples = num_samples + (audio_downsampling_factor - remainder)
    return int(num_samples / audio_downsampling_factor)


def speaker_activity_mask(
    segments: Iterable[SpeechSegment],
    speakers: Sequence[str],
    num_frames: int,
    frame_rate: float,
) -> torch.Tensor:
    """Rasterize diarization segments into a binary (num_speakers, num_frames) mask.

    Segments whose speaker is not in `speakers` are ignored.
    """
    speaker_to_idx = {speaker: i for i, speaker in enumerate(speakers)}
    mask = torch.zeros((len(speakers), num_frames))

    for segment in segments:
        idx = speaker_to_idx.get(segment.speaker)
        if idx is None:
            continue
        begin = int(segment.start * frame_rate)
        end = int((segment.start + segment.duration) * frame_rate)
        mask[idx, begin:end] = 1

    return mask


def create_stno_masks(spk_mask: torch.Tensor, s_index: int) -> torch.Tensor:
    """Turn a (num_speakers, T) activity mask into the (4, T) STNO mask for speaker `s_index`.

    The four channels are mutually exclusive and sum to 1 at every frame:
      0 silence     - nobody speaks
      1 target      - the target speaker speaks alone
      2 non-target  - somebody else speaks but the target does not
      3 overlap     - the target speaks together with somebody else

    Only the target row is distinguished from the rest, so the result does not depend on
    how the remaining speakers are ordered.
    """
    non_target_mask = torch.ones(spk_mask.shape[0], dtype=torch.bool)
    non_target_mask[s_index] = False
    sil_frames = (1 - spk_mask).prod(dim=0)
    anyone_else = (1 - spk_mask[non_target_mask]).prod(dim=0)
    target_spk = spk_mask[s_index] * anyone_else
    non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
    overlapping_speech = spk_mask[s_index] - target_spk
    return torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], dim=0)


def stno_mask_for_speaker(
    segments: Iterable[SpeechSegment],
    speakers: Sequence[str],
    target_speaker: str,
    num_frames: int,
    frame_rate: float,
) -> torch.Tensor:
    """Convenience wrapper: diarization segments -> (4, num_frames) mask for one target speaker."""
    if target_speaker not in speakers:
        raise ValueError(f"Target speaker {target_speaker!r} is not in {list(speakers)!r}")

    spk_mask = speaker_activity_mask(segments, speakers, num_frames, frame_rate)
    return create_stno_masks(spk_mask, speakers.index(target_speaker))
