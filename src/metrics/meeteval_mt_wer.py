"""Multi-talker cpWER / tcpWER via meeteval, with STM dumping.

Ported from `nemo/collections/asr/metrics/meeteval_mt_wer.py` on the NeMo `ts_asr` branch,
with these changes:

  - `get_world_size` is defined locally; the NeMo function it imported does not exist in
    `nemo_toolkit==2.7.3`.
  - STM segments carry the real session id and the real speaker label instead of the
    manifest row index and the speaker's position in the sorted speaker list. cpWER and
    tcpWER are both speaker-permutation invariant, so this cannot change any reported
    number - it only makes `ref.stm` / `hyp.stm` readable and joinable with other tooling.
  - A duplicated (session, speaker) pair is marked processed *before* the empty-hypothesis
    early exits, so DDP's batch padding cannot double-count empty predictions.
  - A gt/prediction session count mismatch warns and lists the missing sessions instead of
    asserting, which previously aborted a whole evaluation.
  - `tcp_collar`, `output_per_word_timestamps` and the text normalizer are configurable.

The per-rank sharding of the scoring work is kept: ranks score disjoint session subsets and
the insertion/deletion/substitution/length counts are summed, which is identical to scoring
everything on one rank but much faster, because cpWER's speaker-permutation search is the
expensive part.

Requires `compute_timestamps=True` on the decoding config: word timestamps are what place
the hypotheses on the time axis for tcpWER and for the STM.
"""

from math import ceil
from typing import Dict, List, Optional, Union

import meeteval
import torch
from meeteval.io.seglst import SegLST, SegLstSegment
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging
from nemo.utils.get_rank import get_rank, is_global_rank_zero

from src.data.text_norm import get_text_norm

__all__ = ['MeetevalMTWER']

# Hypotheses whose words are farther apart than this are split into separate STM segments
# when `output_per_word_timestamps` is False.
SEGMENT_SPLIT_GAP_SECONDS = 0.5


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


class MeetevalMTWER(Metric):
    """Accumulates per-(session, speaker) hypotheses and scores them with cpWER / tcpWER.

    Args:
        decoding: The model's decoding object; also supplies the tokenizer.
        embed_duration: Seconds per encoder frame (0.08 for 12.5 Hz), used to turn decoded
            frame offsets into times.
        output_per_word_timestamps: Emit one STM segment per word (the default, and what makes
            tcpWER timings tightest) rather than grouping words into segments on pauses.
        tcp_collar: tcpWER collar in seconds.
        text_norm_type: Normalizer applied to both references and hypotheses before scoring.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding],
        use_cer=False,
        log_prediction=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        fold_consecutive=True,
        sync_on_compute=True,
        embed_duration=0.08,  # 80 ms - 12.5 Hz with 8x downsampling conformer
        output_per_word_timestamps=True,
        tcp_collar=5,
        text_norm_type='whisper_nsf',
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index
        self.embed_duration = embed_duration
        self.output_per_word_timestamps = output_per_word_timestamps
        self.tcp_collar = tcp_collar

        self.text_norm = get_text_norm(text_norm_type) if text_norm_type else (lambda x: x)

        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions,
                encoded_lengths=predictions_lengths,
                return_hypotheses=True,
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
                return_hypotheses=True,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("preds_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("preds_word_timestamps", default=[], dist_reduce_fx="cat")
        self.add_state("preds_word_timestamps_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("utt_ids", default=[], dist_reduce_fx="cat")
        self.add_state("spk_ids", default=[], dist_reduce_fx="cat")

    def _hypothesis_to_states(self, hypothesis, device):
        """Re-tokenize a hypothesis and extract its word timestamps as flat tensors.

        Re-tokenizing the decoded *text* (rather than keeping `y_sequence`) is what collapses
        the RNNT output into the token sequence the text actually corresponds to.
        """
        token_ids = torch.tensor(
            self.decoding.tokenizer.text_to_ids(hypothesis.text), dtype=torch.int32, device=device
        )

        if not hypothesis.text:
            return (
                token_ids,
                torch.tensor(len(token_ids), dtype=torch.int32, device=device),
                torch.tensor([], dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
            )

        words = hypothesis.text.split()
        word_timestamps = hypothesis.timestamp['word']
        if len(words) != len(word_timestamps):
            raise ValueError(
                f"Number of words must match number of word timestamps: "
                f"{len(words)} != {len(word_timestamps)}"
            )

        return (
            token_ids,
            torch.tensor(len(token_ids), dtype=torch.int32, device=device),
            torch.tensor(
                [[w['start_offset'], w['end_offset']] for w in word_timestamps],
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(len(word_timestamps), dtype=torch.int32, device=device),
        )

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        utt_ids: torch.Tensor,
        spk_ids: torch.Tensor,
    ):
        with torch.no_grad():
            device = predictions.device
            empty = (
                torch.tensor([], dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
                torch.tensor([], dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
            )

            try:
                decoded = self.decode(predictions, predictions_lengths)
                states = [self._hypothesis_to_states(hyp, device) for hyp in decoded]
            except Exception as exc:
                # NeMo occasionally returns a hypothesis whose word timestamps do not line up
                # with its text. Retry one utterance at a time and treat the broken ones as
                # empty hypotheses so the rest of the batch still scores.
                logging.error("MeetevalMTWER: batched decoding failed: %s", exc)
                logging.warning("MeetevalMTWER: falling back to one-by-one decoding.")
                states = []
                for i in range(len(predictions)):
                    try:
                        decoded = self.decode(predictions[i : i + 1], predictions_lengths[i : i + 1])
                        states.append(self._hypothesis_to_states(decoded[0], device))
                    except Exception as inner_exc:
                        logging.error("MeetevalMTWER: skipping utterance %d: %s", i, inner_exc)
                        states.append(empty)

            # Every utterance must be recorded, faulty ones included, or meeteval will later
            # complain about missing hypotheses.
            if not len(states) == len(predictions) == len(utt_ids) == len(spk_ids):
                raise ValueError(
                    f"Decoded {len(states)} hypotheses for {len(predictions)} predictions, "
                    f"{len(utt_ids)} utterance ids and {len(spk_ids)} speaker ids."
                )

            for token_ids, num_tokens, word_timestamps, num_word_timestamps in states:
                self.preds.append(token_ids)
                self.preds_lengths.append(num_tokens)
                self.preds_word_timestamps.append(word_timestamps)
                self.preds_word_timestamps_lengths.append(num_word_timestamps)
            self.utt_ids.extend(utt_ids.detach())
            self.spk_ids.extend(spk_ids.detach())

    @staticmethod
    def _process_metric_res(res):
        output = {'wer': 0, 'ins': 0, 'del': 0, 'sub': 0, 'len': 0}
        for i in res:
            output['len'] += res[i].length
            output['ins'] += res[i].insertions
            output['del'] += res[i].deletions
            output['sub'] += res[i].substitutions
        return output

    @staticmethod
    def _reduce_res(res_all_ranks: List[Dict]):
        res = {k: 0 for k in res_all_ranks[0]}
        for res_rank in res_all_ranks:
            for k in res:
                res[k] += res_rank[k]
        return res

    def _reference_segments(self, targets_collection) -> Dict[int, List[SegLstSegment]]:
        """Build reference STM segments from the whole dataset, keyed by manifest row id.

        Segment `start` values are relative to the row's `offset`, so the offset is added to
        put every row on its recording's absolute timeline. That matters for segmented
        manifests, where many rows share one recording: without it their windows would all
        start at zero and pile up on top of each other. For full-session manifests the offset
        is zero and this is a no-op.
        """
        gt_segments = {}
        for sample in targets_collection:
            offset = sample.offset or 0.0
            for segment in sample.text_tokens:
                start = offset + segment['start']
                gt_segments.setdefault(sample.id, []).append(
                    SegLstSegment(
                        session_id=sample.session_id,
                        speaker=segment['speaker'],
                        words=self.text_norm(self.decoding.decode_tokens_to_str(segment['text'])),
                        start_time=start,
                        end_time=start + segment['duration'],
                    )
                )
        return gt_segments

    def _words_to_segments(
        self, words, word_timestamps, session_id, speaker, offset: float = 0.0
    ) -> List[SegLstSegment]:
        """Turn one hypothesis's words into STM segments, per word or grouped on pauses.

        `offset` shifts the window-relative decoded times onto the recording's absolute
        timeline, matching what `_reference_segments` does for the references.
        """
        if self.output_per_word_timestamps:
            return [
                SegLstSegment(
                    session_id=session_id,
                    speaker=speaker,
                    words=self.text_norm(word),
                    start_time=offset + start * self.embed_duration,
                    end_time=offset + end * self.embed_duration,
                )
                for word, (start, end) in zip(words, word_timestamps)
            ]

        segments = []
        group = [(words[0], *word_timestamps[0])]

        def flush(group):
            return SegLstSegment(
                session_id=session_id,
                speaker=speaker,
                words=self.text_norm(' '.join(w[0] for w in group)),
                start_time=offset + group[0][1] * self.embed_duration,
                end_time=offset + group[-1][2] * self.embed_duration,
            )

        for word, (start, end) in zip(words[1:], word_timestamps[1:]):
            if (start - group[-1][2]) * self.embed_duration > SEGMENT_SPLIT_GAP_SECONDS:
                segments.append(flush(group))
                group = [(word, start, end)]
            else:
                group.append((word, start, end))

        segments.append(flush(group))
        return segments

    def _hypothesis_segments(self, session_of, speakers_of, offset_of) -> Dict[int, List[SegLstSegment]]:
        """Build hypothesis STM segments from the accumulated states, keyed by manifest row id."""
        preds = dim_zero_cat(self.preds)
        preds_lengths = dim_zero_cat(self.preds_lengths)
        preds_word_timestamps = dim_zero_cat(self.preds_word_timestamps)
        preds_word_timestamps_lengths = dim_zero_cat(self.preds_word_timestamps_lengths)
        utt_ids = dim_zero_cat(self.utt_ids)
        spk_ids = dim_zero_cat(self.spk_ids)

        # When the dataset size is not divisible by the number of ranks, the distributed sampler
        # pads the last batch by repeating samples. Score each (session, speaker) pair once.
        already_processed_pairs = set()
        pred_segments = {}
        current_start = 0
        current_ts_start = 0

        for i in range(len(preds_lengths)):
            utt_id = utt_ids[i].item()
            spk_id = spk_ids[i].item()
            num_tokens = int(preds_lengths[i])
            num_word_timestamps = int(preds_word_timestamps_lengths[i])
            token_start, current_start = current_start, current_start + num_tokens
            ts_start, current_ts_start = current_ts_start, current_ts_start + num_word_timestamps

            if (utt_id, spk_id) in already_processed_pairs:
                continue
            already_processed_pairs.add((utt_id, spk_id))

            if utt_id not in session_of:
                logging.warning("Skipping prediction for unknown manifest row %s", utt_id)
                continue

            session_id = session_of[utt_id]
            speaker = speakers_of[utt_id][spk_id]
            offset = offset_of[utt_id]
            pred_segments.setdefault(utt_id, [])

            transcript = (
                self.decoding.decode_tokens_to_str(preds[token_start:current_start].detach().cpu())
                if num_tokens > 0
                else ''
            )
            words = transcript.split()

            if not words:
                # meeteval needs the (session, speaker) pair to exist even when nothing was said,
                # otherwise the deletions it should be charged for go uncounted.
                pred_segments[utt_id].append(
                    SegLstSegment(
                        session_id=session_id,
                        speaker=speaker,
                        words='',
                        start_time=offset,
                        end_time=offset + 1,
                    )
                )
                continue

            word_timestamps = preds_word_timestamps[ts_start:current_ts_start]
            if len(word_timestamps) != len(words):
                raise ValueError(
                    f"Number of word timestamps must match number of words: "
                    f"{len(word_timestamps)} != {len(words)}"
                )

            pred_segments[utt_id].extend(
                self._words_to_segments(words, word_timestamps.tolist(), session_id, speaker, offset)
            )

        return pred_segments

    def compute(self, targets_collection: List[Dict], save_stm_path: Optional[str] = None):
        """Score every accumulated hypothesis against the dataset's references.

        Args:
            targets_collection: The validation dataset's manifest collection, i.e.
                `dataloader.dataset.manifest_processor.collection`.
            save_stm_path: Directory to write `ref.stm` and `hyp.stm` into. Rank zero only.

        Returns:
            `(cp_result, tcp_result)` dicts with `wer`, `ins`, `del`, `sub` and `len`.
        """
        # Keyed by manifest row id (`sample.id`), which is what `update` records, and which is
        # not the position in the collection once empty sessions have been dropped.
        session_of = {sample.id: sample.session_id for sample in targets_collection}
        speakers_of = {
            sample.id: sorted({segment['speaker'] for segment in sample.text_tokens})
            for sample in targets_collection
        }
        offset_of = {sample.id: (sample.offset or 0.0) for sample in targets_collection}

        gt_segments = self._reference_segments(targets_collection)
        pred_segments = self._hypothesis_segments(session_of, speakers_of, offset_of)

        gt_segment_ids = sorted(gt_segments.keys())
        pred_segment_ids = sorted(pred_segments.keys())

        if len(gt_segment_ids) != len(pred_segment_ids):
            missing_rows = set(gt_segment_ids) - set(pred_segment_ids)
            missing_sessions = sorted({session_of.get(uid, uid) for uid in missing_rows})
            logging.warning(
                "References cover %d manifest rows but predictions cover %d; %d rows have no "
                "prediction, across recordings: %s%s. (Expected when limit_val_batches is set.)",
                len(gt_segment_ids),
                len(pred_segment_ids),
                len(missing_rows),
                missing_sessions[:10],
                f" and {len(missing_sessions) - 10} more" if len(missing_sessions) > 10 else "",
            )

        # Split the scoring work across ranks; the counts are summed afterwards, so the result
        # is the same as scoring everything on one rank.
        world_size = get_world_size()
        num_elems_per_rank = ceil(len(gt_segment_ids) / world_size)
        begin_idx = get_rank() * num_elems_per_rank
        end_idx = begin_idx + num_elems_per_rank

        gt_seg_lst = SegLST(
            segments=[seg for uid in gt_segment_ids[begin_idx:end_idx] for seg in gt_segments[uid]]
        )
        pred_seg_lst = SegLST(
            segments=[seg for uid in pred_segment_ids[begin_idx:end_idx] for seg in pred_segments[uid]]
        )

        res_cp = self._process_metric_res(meeteval.wer.cpwer(reference=gt_seg_lst, hypothesis=pred_seg_lst))
        res_tcp = self._process_metric_res(
            meeteval.wer.tcpwer(reference=gt_seg_lst, hypothesis=pred_seg_lst, collar=self.tcp_collar)
        )

        res_both_all_ranks = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(res_both_all_ranks, (res_cp, res_tcp))
        else:
            res_both_all_ranks[0] = (res_cp, res_tcp)

        res_cp = self._reduce_res([res[0] for res in res_both_all_ranks])
        res_tcp = self._reduce_res([res[1] for res in res_both_all_ranks])
        for res in (res_cp, res_tcp):
            res['wer'] = (res['sub'] + res['ins'] + res['del']) / res['len'] if res['len'] else 0.0

        if save_stm_path is not None and is_global_rank_zero():
            gt_seglist = SegLST(segments=[seg for uid in gt_segment_ids for seg in gt_segments[uid]])
            hyp_seglist = SegLST(segments=[seg for uid in pred_segment_ids for seg in pred_segments[uid]])
            meeteval.io.dump(gt_seglist, f'{save_stm_path}/ref.stm')
            meeteval.io.dump(hyp_seglist, f'{save_stm_path}/hyp.stm')

        return res_cp, res_tcp
