"""Locating and loading session audio for inference.

An RTTM identifies recordings by an opaque file id, and corpora lay their audio out in
whatever shape they like: `{id}.wav` next to the RTTM, `{id}/audio.wav`, or something only
the user knows. `resolve_audio_path` tries the common shapes and then a user-supplied glob,
and reports what it tried when it fails.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

__all__ = [
    'DEFAULT_AUDIO_EXTENSIONS',
    'create_audio_featurizer',
    'load_session_audio',
    'resolve_audio_path',
]

DEFAULT_AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.opus', '.sph')


def resolve_audio_path(
    session_id: str,
    audio_dir: Union[str, Path],
    extensions: Sequence[str] = DEFAULT_AUDIO_EXTENSIONS,
    glob_template: Optional[str] = None,
) -> Path:
    """Find the audio file for `session_id` under `audio_dir`.

    Tried in order:
      1. `audio_dir/{session_id}{ext}` for each extension;
      2. `audio_dir/**/{session_id}{ext}` for each extension;
      3. `audio_dir/<glob_template with {session} substituted>`, if given.

    Args:
        session_id: Recording id, normally the RTTM's second column.
        audio_dir: Root directory to search.
        extensions: Extensions to try, in preference order.
        glob_template: Glob relative to `audio_dir` containing `{session}`, for layouts the
            first two tiers do not cover, e.g. `'{session}/audio.wav'` or `'**/{session}/*.wav'`.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: If nothing matched. The message lists every pattern tried.
    """
    audio_dir = Path(audio_dir)
    attempted = []

    for extension in extensions:
        candidate = audio_dir / f"{session_id}{extension}"
        attempted.append(str(candidate))
        if candidate.is_file():
            return candidate

    for extension in extensions:
        pattern = f"**/{session_id}{extension}"
        attempted.append(str(audio_dir / pattern))
        matches = sorted(audio_dir.glob(pattern))
        if matches:
            return matches[0]

    if glob_template:
        pattern = glob_template.format(session=session_id)
        attempted.append(str(audio_dir / pattern))
        matches = sorted(audio_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No audio found for session {session_id!r}. Tried:\n  " + "\n  ".join(attempted)
        + "\nPass --audio-glob to describe the layout, e.g. --audio-glob '{session}/audio.wav'."
    )


def create_audio_featurizer(sample_rate: int = 16000) -> WaveformFeaturizer:
    """The same loader the training dataset uses, so resampling behaviour is identical."""
    return WaveformFeaturizer(sample_rate=sample_rate, int_values=False, augmentor=None)


def load_session_audio(
    audio_path: Union[str, Path],
    featurizer: WaveformFeaturizer,
    channel_selector: Union[str, int, None] = 'average',
) -> torch.Tensor:
    """Load a whole recording as a mono waveform.

    Args:
        audio_path: Path to the audio file.
        featurizer: From `create_audio_featurizer`.
        channel_selector: `'average'` mixes multi-channel recordings down (matching the
            AMI-SDM training setup); an int picks a single channel; `None` leaves it alone.

    Returns:
        A 1-D float tensor of samples.
    """
    return featurizer.process(
        str(audio_path),
        offset=0,
        duration=0,
        trim=False,
        channel_selector=channel_selector,
    )
