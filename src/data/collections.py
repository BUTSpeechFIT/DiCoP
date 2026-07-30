"""Manifest parsing for multi-talker, per-segment-annotated sessions.

Ported from `nemo/collections/common/parts/preprocessing/collections.py` (classes
`AudioTextSTNO` and `ASRAudioTextSTNO`) so that DiCoP does not have to patch the installed
NeMo package. Only the multi-talker collection is kept; the single-utterance siblings live
in NeMo and are used unchanged elsewhere.

The manifest is JSON Lines with **one line per session**, and `text` holding a *list* of
diarized segments rather than a single string:

    {"audio_filepath": "/path/IS1009c.Array1-01.wav",
     "duration": 1820.842625,
     "offset": 0.0,
     "session_id": "IS1009c",            # optional, defaults to the audio file stem
     "text": [{"start": 43.77, "duration": 0.32, "speaker": "FIO087", "text": "HI"},
              {"start": 55.60, "duration": 0.37, "speaker": "FIO087", "text": "MM-HMM"},
              ...]}

`start` is relative to `offset`. NeMo's own `manifest.item_iter` passes a list-valued `text`
through untouched, so it is reused here as-is; the per-segment tokenization happens below.
"""

import collections as py_collections
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging

__all__ = ['AudioTextSTNO', 'ASRAudioTextSTNO', 'parse_stno_manifest_item']


def parse_stno_manifest_item(line: str, manifest_file: str) -> Dict[str, Any]:
    """Parse one manifest line into the fields `AudioTextSTNO` needs.

    NeMo's own `manifest.__parse_item` rebuilds the item from a fixed key list and so would
    silently drop `session_id`. Parsing here instead keeps the schema DiCoP expects explicit
    and avoids depending on a private NeMo function.
    """
    item = json.loads(line)

    if 'audio_filepath' in item:
        audio_file = item.pop('audio_filepath')
    elif 'audio_filename' in item:
        audio_file = item.pop('audio_filename')
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has a line without an `audio_filepath` key: {line}"
        )

    if 'duration' not in item:
        raise ValueError(f"Manifest file {manifest_file} has a line without a `duration` key: {line}")

    text = item.get('text', [])
    if isinstance(text, str):
        raise ValueError(
            f"Manifest file {manifest_file} has a string `text` field. DiCoP expects a list of "
            f"diarized segments, each `{{start, duration, speaker, text}}`. Offending line: {line}"
        )

    return dict(
        # Relative paths are resolved against the manifest's own directory.
        audio_file=get_full_path(audio_file=audio_file, manifest_file=manifest_file),
        duration=item['duration'],
        text=text,
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
        token_labels=item.get('token_labels', None),
        lang=item.get('lang', None),
        session_id=item.get('session_id', None),
    )


class AudioTextSTNO(py_collections.UserList):
    """List of sessions, each with per-speaker tokenized segments."""

    OUTPUT_TYPE = py_collections.namedtuple(
        typename='AudioTextSTNOEntity',
        field_names='id audio_file duration text_tokens offset text_raw speaker orig_sr lang session_id',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[List[Dict]],
        offsets: List[Optional[float]],
        speakers: List[Optional[str]],
        orig_sampling_rates: List[Optional[int]],
        token_labels: List[Optional[int]],
        langs: List[Optional[str]],
        session_ids: List[Optional[str]],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates a multi-talker manifest with duration filters and tokenization.

        Args:
            ids: List of example positions.
            audio_files: List of audio files.
            durations: List of float durations.
            texts: Per session, the list of `{start, duration, speaker, text}` segments.
            offsets: List of duration offsets or None.
            speakers: List of optional session-level speaker ids (unused by this model).
            orig_sampling_rates: List of original sampling rates of audio files.
            token_labels: Pre-tokenized labels, bypassing `parser` when present.
            langs: List of language ids, one per sample, or None.
            session_ids: Session identifier used for STM output; falls back to the audio stem.
            parser: Callable converting a segment's text to token ids.
            min_duration: Minimum duration to keep an entry (default: None).
            max_duration: Maximum duration to keep an entry (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: Sort samples by duration. Incompatible with index_by_file_id.
            index_by_file_id: If True, build a mapping from file stem to index in data.
        """
        output_type = self.OUTPUT_TYPE
        all_has_duration = True
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, duration, offset, text, speaker, orig_sr, token_label, lang, session_id in zip(
            ids,
            audio_files,
            durations,
            offsets,
            texts,
            speakers,
            orig_sampling_rates,
            token_labels,
            langs,
            session_ids,
        ):
            if duration is None:
                all_has_duration = False
            # Duration filters.
            if duration is not None and min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if duration is not None and max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if token_label is not None:
                text_tokens = token_label
            elif text:
                # Tokenize each diarized segment separately, keeping its timing and speaker.
                # Segments that tokenize to nothing are dropped from both views of the text.
                text_tokens = []
                kept_text = []
                for text_entry in text:
                    parsed = parser(text_entry['text'])
                    if parsed:
                        text_tokens.append(
                            {
                                'start': text_entry['start'],
                                'duration': text_entry['duration'],
                                'speaker': text_entry['speaker'],
                                'text': parsed,
                            }
                        )
                        kept_text.append(text_entry)
                text = kept_text
            else:
                text_tokens = []

            total_duration += duration if duration is not None else 0.0

            if not session_id:
                session_id = os.path.splitext(os.path.basename(audio_file))[0]

            data.append(
                output_type(
                    id_, audio_file, duration, text_tokens, offset, text, speaker, orig_sr, lang, session_id
                )
            )
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping.setdefault(file_id, []).append(len(data) - 1)

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)
        if not all_has_duration:
            logging.info("Not all audios have duration information, the total number of hours is inaccurate.")
        super().__init__(data)


class ASRAudioTextSTNO(AudioTextSTNO):
    """`AudioTextSTNO` collector reading NeMo-style JSON Lines manifests."""

    def __init__(
        self,
        manifests_files: Union[str, List[str]],
        parse_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        """Parse per-session audio paths, durations and diarized transcripts.

        Args:
            manifests_files: Either a single manifest path or a list of such.
            parse_func: Optional custom manifest line parser. Defaults to
                `parse_stno_manifest_item`.
            *args: Args to pass to the `AudioTextSTNO` constructor.
            **kwargs: Kwargs to pass to the `AudioTextSTNO` constructor.
        """
        ids, audio_files, durations, texts, offsets = [], [], [], [], []
        speakers, orig_srs, token_labels, langs, session_ids = [], [], [], [], []

        if parse_func is None:
            parse_func = parse_stno_manifest_item

        for item in manifest.item_iter(manifests_files, parse_func=parse_func):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            texts.append(item['text'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])
            token_labels.append(item['token_labels'])
            langs.append(item['lang'])
            session_ids.append(item.get('session_id'))

        super().__init__(
            ids,
            audio_files,
            durations,
            texts,
            offsets,
            speakers,
            orig_srs,
            token_labels,
            langs,
            session_ids,
            *args,
            **kwargs,
        )
