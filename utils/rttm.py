"""RTTM parsing.

RTTM is whitespace-separated, one segment per line, with the fields:

    SPEAKER <file-id> <channel> <start> <duration> <ortho> <stype> <speaker-id> <conf> <slat>

Only `SPEAKER` lines matter here; everything else (`SPKR-INFO`, comments) is skipped. The
recording is identified by field 1, and diarizers occasionally leave it as `<NA>`, in which
case the filename stem is used instead.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Union

from nemo.utils import logging

from src.data.stno import SpeechSegment

__all__ = ['load_rttm', 'parse_rttm_file', 'speakers_in']

NA_VALUES = {'<NA>', 'NA', '-', ''}


def _is_missing(value: str) -> bool:
    return value in NA_VALUES


def parse_rttm_file(path: Union[str, Path], default_session_id: str = None) -> Dict[str, List[SpeechSegment]]:
    """Parse one RTTM file into `{session_id: [SpeechSegment, ...]}`.

    Args:
        path: Path to the RTTM file.
        default_session_id: Used for lines whose file-id field is `<NA>`. Defaults to the
            file's stem.

    Returns:
        Segments per session, sorted by start time.
    """
    path = Path(path)
    default_session_id = default_session_id or path.stem
    sessions = defaultdict(list)
    skipped_zero_duration = 0

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue

            fields = line.split()
            if fields[0] != 'SPEAKER':
                continue
            if len(fields) < 8:
                raise ValueError(
                    f"{path}:{line_number}: a SPEAKER line needs at least 8 fields, got "
                    f"{len(fields)}: {line!r}"
                )

            session_id = default_session_id if _is_missing(fields[1]) else fields[1]
            speaker = fields[7]
            if _is_missing(speaker):
                raise ValueError(f"{path}:{line_number}: missing speaker id: {line!r}")

            try:
                start = float(fields[3])
                duration = float(fields[4])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: unparseable start/duration: {line!r}") from exc

            if duration <= 0:
                skipped_zero_duration += 1
                continue

            sessions[session_id].append(
                SpeechSegment(start=start, duration=duration, speaker=speaker)
            )

    if skipped_zero_duration:
        logging.warning("%s: skipped %d segments with non-positive duration", path, skipped_zero_duration)

    return {sid: sorted(segments, key=lambda s: (s.start, s.speaker)) for sid, segments in sessions.items()}


def load_rttm(path: Union[str, Path]) -> Dict[str, List[SpeechSegment]]:
    """Load a single RTTM file or a directory of them.

    A directory is searched recursively for `*.rttm`, and segments for the same session id are
    merged across files.

    Args:
        path: An RTTM file, or a directory containing `*.rttm` files.

    Returns:
        `{session_id: [SpeechSegment, ...]}`, each list sorted by start time.
    """
    path = Path(path)

    if path.is_file():
        sessions = parse_rttm_file(path)
    elif path.is_dir():
        rttm_files = sorted(path.rglob("*.rttm"))
        if not rttm_files:
            raise FileNotFoundError(f"No .rttm files found under {path}")

        sessions = defaultdict(list)
        for rttm_file in rttm_files:
            for session_id, segments in parse_rttm_file(rttm_file).items():
                sessions[session_id].extend(segments)
        sessions = {
            sid: sorted(segments, key=lambda s: (s.start, s.speaker)) for sid, segments in sessions.items()
        }
        logging.info("Loaded %d RTTM files describing %d sessions", len(rttm_files), len(sessions))
    else:
        raise FileNotFoundError(f"RTTM path not found: {path}")

    if not sessions:
        raise ValueError(f"No SPEAKER segments found in {path}")

    return sessions


def speakers_in(segments: Iterable[SpeechSegment], min_speech_seconds: float = 0.0) -> List[str]:
    """The sorted speakers of a session, optionally dropping barely-active ones.

    Args:
        segments: The session's diarization segments.
        min_speech_seconds: Drop speakers with less total speech than this. Useful to suppress
            spurious diarization speakers, which would otherwise each cost a decoding pass.

    Returns:
        Sorted speaker labels.
    """
    totals = defaultdict(float)
    for segment in segments:
        totals[segment.speaker] += segment.duration

    return sorted(
        speaker for speaker, total in totals.items() if total >= min_speech_seconds
    )
