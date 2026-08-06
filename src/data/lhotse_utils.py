"""Adapters from Lhotse cuts to the shapes DiCoP works in.

Lhotse is used here purely as a **manifest format**: a `CutSet` is read eagerly and indexed
like any other list. Lhotse's own dataloading stack (`DynamicBucketingSampler` and friends,
which hand a whole `CutSet` to `__getitem__`) is deliberately not used, because it would break
both the per-(cut, speaker) fan-out and the integer-indexed reference lookup that multi-talker
scoring depends on.

The two manifest formats line up exactly, which is what makes them interchangeable:

    NeMo manifest              Lhotse cut
    ---------------------------------------------
    offset                     cut.start
    text[].start               supervision.start        (both cut-relative)
    duration                   cut.duration
    session_id                 cut.recording.id

`supervision.start` being cut-relative is easy to miss: it coincides with recording-absolute
time only in full-recording cuts, where `cut.start == 0`. Pre-segmented cutsets (the `_30s`
variants) have a non-zero `cut.start` and supervisions that restart from 0.

`MonoCut` and `MixedCut` are supported; `MultiCut` (multi-channel arrays) is rejected with a
clear error rather than silently mishandled, because picking or mixing its channels is a choice
this loader should not make silently.

The two supported kinds read their audio differently. A `MonoCut` names one file, so it goes
through NeMo's `WaveformFeaturizer` and is bit-identical to the NeMo-manifest path. A `MixedCut`
(LibriMix, LibriSpeechMix) names no file at all — it is a recipe for summing its tracks, and the
mixture only exists once Lhotse renders it — so it is read with `cut.load_audio()` and resampled
afterwards if needed.
"""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import List, Tuple, Union

import torch

from src.data.stno import SpeechSegment

__all__ = [
    'cut_audio_path',
    'cut_segments',
    'cut_session_id',
    'cut_source_label',
    'cut_speakers',
    'load_cut_audio',
    'load_cutset',
    'manifest_paths',
    'manifest_stem',
    'named_manifests',
    'require_monocut',
    'require_supported_cut',
]

# Stripped, in this order, to turn a manifest path into a dataset name. `Path.stem` alone leaves
# the `.jsonl` of a `.jsonl.gz` cutset behind.
MANIFEST_SUFFIXES = (('.gz', '.gzip'), ('.jsonl', '.json'))


def manifest_paths(manifest_filepath) -> List[str]:
    """The manifest paths a `manifest_filepath` config value names, in order.

    Accepts a single path, a comma-separated string (the convention the NeMo-manifest dataset
    already follows), or any sequence — including the `ListConfig` a Hydra list override such as
    `model.train_ds.manifest_filepath=[a.jsonl.gz,b.jsonl.gz]` produces, which is not a `list`
    and so is missed by an `isinstance(..., list)` test.

    All of these name **one** dataset, which reads them all. A mapping is rejected: it names one
    dataset per entry, which only the evaluation sections support (see `named_manifests`).
    """
    if isinstance(manifest_filepath, Mapping):
        raise ValueError(
            f"A mapping of {len(manifest_filepath)} named manifests cannot build a single "
            f"dataset. Only `validation_ds` and `test_ds` evaluate one dataset per name; pass a "
            f"list or a comma-separated string here to read them as one set. "
            f"Got {list(manifest_filepath)}."
        )

    if isinstance(manifest_filepath, (str, Path)):
        candidates = str(manifest_filepath).split(',')
    else:
        candidates = [str(path) for path in manifest_filepath]

    paths = [path.strip() for path in candidates if str(path).strip()]
    if not paths:
        raise ValueError(f"No manifest path in {manifest_filepath!r}.")
    return paths


def manifest_stem(path: Union[str, Path]) -> str:
    """The dataset name a manifest path implies: its file name without the manifest suffixes.

    `.../librimix_cutset_dev-clean.jsonl.gz` -> `librimix_cutset_dev-clean`.
    """
    name = Path(str(path).strip()).name
    for group in MANIFEST_SUFFIXES:
        for suffix in group:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
    return name


def named_manifests(manifest_filepath) -> List[Tuple[str, object]]:
    """The `(name, manifest)` pairs an evaluation `manifest_filepath` names, in order.

    Every manifest is a dataset of its own, inferred and scored separately under `val/<name>/`.
    A mapping names them explicitly, and is the only form that pools several manifests into one
    score, since its value is passed through untouched and may itself be a list or a
    comma-separated string. Any other form goes through `manifest_paths`, and each path is named
    after its file stem.
    """
    if isinstance(manifest_filepath, Mapping):
        pairs = [(str(name), manifest) for name, manifest in manifest_filepath.items()]
        if not pairs:
            raise ValueError(f"No manifest in {manifest_filepath!r}.")
    else:
        pairs = [(manifest_stem(path), path) for path in manifest_paths(manifest_filepath)]

    for name, manifest in pairs:
        if not name or '/' in name:
            raise ValueError(
                f"{name!r} (from {manifest!r}) cannot name a dataset: the name becomes a metric "
                f"key segment (`val/<name>/cp_wer`) and a predictions subdirectory, so it must be "
                f"non-empty and free of '/'."
            )

    counts = Counter(name for name, _ in pairs)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Two evaluation manifests resolved to the same dataset name: {duplicates}. Names are "
            f"taken from the file stem, which collides for identically-named files in different "
            f"directories. Name them explicitly instead, as "
            f"manifest_filepath=\"{{ami:'/a/dev.jsonl.gz',notsofar:'/b/dev.jsonl.gz'}}\"."
        )
    return pairs


def load_cutset(path: Union[str, Path]):
    """Read a Lhotse CutSet manifest (`.jsonl.gz` or `.jsonl`) eagerly."""
    from lhotse import CutSet

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CutSet manifest not found: {path}")
    return CutSet.from_file(path)


def require_monocut(cut) -> None:
    """Reject cuts that cannot go through the single-source, single-file audio path.

    `MultiCut` carries a list of channels and `MixedCut` has no `recording` at all, so neither
    names the one file this path needs. `MixedCut` is still decodable — see
    `require_supported_cut` and `load_cut_audio` — just not by way of a file path.
    """
    if type(cut).__name__ != 'MonoCut':
        raise NotImplementedError(
            f"Cut {cut.id!r} is a {type(cut).__name__}; this path supports MonoCut only. "
            f"Convert the cutset to single-channel cuts first "
            f"(e.g. `cut.with_channels(...)` for a MultiCut, or mix down a MixedCut)."
        )


def require_supported_cut(cut) -> None:
    """Reject cut types the loader cannot handle at all.

    Wider than `require_monocut`: a `MixedCut` is fine here because `load_cut_audio` can render
    it. A `MultiCut` is not, because choosing between its channels — one? averaged? beamformed? —
    is a decision that belongs to whoever prepared the cutset, not to this loader.
    """
    if type(cut).__name__ not in ('MonoCut', 'MixedCut'):
        raise NotImplementedError(
            f"Cut {cut.id!r} is a {type(cut).__name__}; DiCoP supports MonoCut and MixedCut. "
            f"Convert the cutset to single-channel cuts first, e.g. `cut.with_channels(...)`."
        )


def cut_speakers(cut) -> List[str]:
    """The cut's speakers, sorted.

    The sort order matters beyond determinism: the target speaker is identified downstream by
    its *index* into this list, and multi-talker scoring recovers the label by sorting the
    reference speakers the same way. Both sides must agree.
    """
    return sorted({supervision.speaker for supervision in cut.supervisions})


def cut_segments(cut) -> List[SpeechSegment]:
    """The cut's supervisions as `SpeechSegment`s, in cut-relative time."""
    return [
        SpeechSegment(
            start=supervision.start, duration=supervision.duration, speaker=supervision.speaker
        )
        for supervision in cut.supervisions
    ]


def cut_session_id(cut) -> str:
    """The recording a cut belongs to, used as the STM session id.

    Keyed on the *recording* rather than the cut so that the many windows of a pre-segmented
    cutset compose into a single session on one timeline, instead of each window becoming its
    own pseudo-session. Falls back to `cut.id` for cuts with no recording attached.
    """
    recording = getattr(cut, 'recording', None)
    recording_id = getattr(recording, 'id', None)
    return recording_id if recording_id else cut.id


def cut_audio_path(cut) -> str:
    """Path to the cut's audio file."""
    require_monocut(cut)
    sources = cut.recording.sources
    if not sources:
        raise ValueError(f"Cut {cut.id!r} has a recording with no sources.")
    return sources[0].source


def cut_source_label(cut) -> str:
    """Where a cut's audio comes from, for logging.

    Unlike `cut_audio_path` this never raises, because a `MixedCut` has no single file to name.
    """
    if type(cut).__name__ == 'MixedCut':
        return f"<mixture of {len(cut.tracks)} tracks>"
    return cut_audio_path(cut)


def load_cut_audio(cut, featurizer, channel_selector=None, trim: bool = False) -> torch.Tensor:
    """Load a cut's audio as a mono waveform.

    A `MonoCut` deliberately reads through NeMo's `WaveformFeaturizer` rather than
    `cut.load_audio()`, so that resampling, channel selection and augmentation behave identically
    to the NeMo-manifest dataset. `tests/test_lhotse_dataset.py` asserts the two produce
    bit-identical audio.

    A `MixedCut` has no file to hand the featurizer — the mixture exists only once Lhotse sums
    the tracks — so it is rendered by `cut.load_audio()` and resampled here if the corpus is not
    already at the featurizer's rate. `channel_selector` and `trim` do not apply: the render is
    mono by construction, and the cut's own bounds already define the extent.

    Args:
        cut: A `MonoCut` or a `MixedCut`.
        featurizer: A `nemo...WaveformFeaturizer`, which fixes the target sample rate.
        channel_selector: Channel index, or `'average'` to mix multi-channel files down.
        trim: Trim leading/trailing silence.

    Returns:
        A 1-D float tensor of samples covering `[cut.start, cut.start + cut.duration)`.
    """
    require_supported_cut(cut)

    if type(cut).__name__ == 'MixedCut':
        audio = torch.from_numpy(cut.load_audio()).float().reshape(-1)
        if cut.sampling_rate != featurizer.sample_rate:
            import torchaudio

            audio = torchaudio.functional.resample(audio, cut.sampling_rate, featurizer.sample_rate)
        return audio

    return featurizer.process(
        cut_audio_path(cut),
        offset=cut.start,
        duration=cut.duration,
        trim=trim,
        orig_sr=cut.recording.sampling_rate,
        channel_selector=channel_selector,
    )
