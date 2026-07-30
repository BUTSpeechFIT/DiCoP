"""STM output via meeteval.

Writing through `meeteval.io.dump` rather than formatting lines by hand keeps `infer.py`'s
output byte-compatible with the `hyp.stm` that training-time validation produces, so the two
can be scored and compared with the same commands.

Granularity mirrors the metric's `output_per_word_timestamps` option:

  - `word` (default): one line per word. Gives tcpWER the tightest timings and matches what
    validation writes.
  - `segment`: words grouped into utterances on pauses longer than `SEGMENT_SPLIT_GAP_SECONDS`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import meeteval
from meeteval.io.seglst import SegLST, SegLstSegment

__all__ = ['SEGMENT_SPLIT_GAP_SECONDS', 'WordSpan', 'spans_to_segments', 'write_stm']


@dataclass(frozen=True)
class WordSpan:
    """One decoded word with its start and end time, in seconds."""

    text: str
    start: float
    end: float

# Matches the training-time metric, so segment-mode output is consistent between the two.
SEGMENT_SPLIT_GAP_SECONDS = 0.5


def spans_to_segments(
    spans: Sequence[WordSpan],
    session_id: str,
    speaker: str,
    granularity: str = 'word',
) -> List[SegLstSegment]:
    """Convert decoded words into STM segments for one (session, speaker).

    Args:
        spans: Decoded words with absolute times.
        session_id: STM session/file id.
        speaker: STM speaker label.
        granularity: `'word'` for one segment per word, `'segment'` to group on pauses.

    Returns:
        The STM segments, possibly empty.
    """
    if granularity not in ('word', 'segment'):
        raise ValueError(f"granularity must be 'word' or 'segment', got {granularity!r}")

    if not spans:
        return []

    if granularity == 'word':
        return [
            SegLstSegment(
                session_id=session_id,
                speaker=speaker,
                words=span.text,
                start_time=span.start,
                end_time=span.end,
            )
            for span in spans
        ]

    segments = []
    group = [spans[0]]
    for span in spans[1:]:
        if span.start - group[-1].end > SEGMENT_SPLIT_GAP_SECONDS:
            segments.append(_group_to_segment(group, session_id, speaker))
            group = [span]
        else:
            group.append(span)
    segments.append(_group_to_segment(group, session_id, speaker))
    return segments


def _group_to_segment(group: Sequence[WordSpan], session_id: str, speaker: str) -> SegLstSegment:
    return SegLstSegment(
        session_id=session_id,
        speaker=speaker,
        words=' '.join(span.text for span in group),
        start_time=group[0].start,
        end_time=group[-1].end,
    )


def write_stm(
    output_path: Union[str, Path],
    session_segments: Dict[str, List[SegLstSegment]],
    session_speakers: Dict[str, Iterable[str]] = None,
) -> int:
    """Write an STM file.

    Args:
        output_path: Destination `.stm` path; parent directories are created.
        session_segments: `{session_id: [SegLstSegment, ...]}`, in the order to be written.
        session_speakers: Optional `{session_id: speakers}`. Any (session, speaker) pair with
            no segments gets one empty-transcript line, so a scorer charges the deletions
            rather than reporting the pair as absent. This is what the training-time metric does.

    Returns:
        The number of lines written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = []
    for session_id in sorted(session_segments):
        session = session_segments[session_id]
        segments.extend(session)

        if session_speakers is None:
            continue
        covered = {segment['speaker'] for segment in session}
        for speaker in sorted(set(session_speakers.get(session_id, ())) - covered):
            segments.append(
                SegLstSegment(
                    session_id=session_id, speaker=speaker, words='', start_time=0.0, end_time=1.0
                )
            )

    meeteval.io.dump(SegLST(segments=segments), str(output_path))
    return len(segments)
