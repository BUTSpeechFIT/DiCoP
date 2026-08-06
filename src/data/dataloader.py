"""Dataloader construction for the multi-talker STNO dataset.

Collapses NeMo's `get_audio_to_text_bpe_dataset_from_config` -> `get_bpe_dataset` ->
`EncDecRNNTBPEModel._setup_dataloader_from_config` chain into one function. All the branches
that chain carried (tarred, DALI, bucketing, concat, HuggingFace, code-switching) are
unreachable for this model and are not reproduced.
"""

from typing import Optional

import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.utils import logging

from src.data.dataset import AudioToBPEAndSTNODataset
from src.data.lhotse_dataset import LhotseToBPEAndSTNODataset
from src.data.lhotse_utils import manifest_paths

__all__ = ['get_stno_dataloader', 'uses_lhotse_manifest']

LHOTSE_SUFFIXES = ('.jsonl.gz', '.jsonl.gzip')


def uses_lhotse_manifest(config: DictConfig) -> bool:
    """Whether this dataset section points at a Lhotse CutSet rather than a NeMo manifest.

    An explicit `use_lhotse` always wins. Otherwise the format is inferred from the suffix of the
    first manifest, since a CutSet is conventionally `.jsonl.gz` while DiCoP's NeMo manifests are
    plain `.jsonl`. Set `use_lhotse: false` if you keep NeMo manifests gzipped.

    Several manifests may be given (as a list, or comma-separated), in which case they must all
    be the same format — one dataset reads them all.
    """
    if config.get('use_lhotse') is not None:
        return bool(config['use_lhotse'])

    manifest = config.get('manifest_filepath')
    if not manifest:
        return False

    paths = manifest_paths(manifest)
    is_lhotse = [path.endswith(LHOTSE_SUFFIXES) for path in paths]
    if any(is_lhotse) and not all(is_lhotse):
        raise ValueError(
            f"Mixed manifest formats in {list(paths)}: a dataset reads either Lhotse CutSets "
            f"({'/'.join(LHOTSE_SUFFIXES)}) or NeMo manifests, not both. Set `use_lhotse` "
            f"explicitly if the suffixes are misleading."
        )
    return is_lhotse[0]


def get_stno_dataloader(
    config: DictConfig,
    tokenizer,
    audio_downsampling_factor: int,
    text_norm_type: Optional[str] = 'whisper_nsf',
    val: bool = False,
) -> Optional[torch.utils.data.DataLoader]:
    """Build a dataloader over `(session, target speaker)` pairs.

    Args:
        config: One of the `model.{train,validation,test}_ds` sections.
        tokenizer: Tokenizer shared with the model's decoding object.
        audio_downsampling_factor: Samples per encoder frame (1280 for this architecture).
        text_norm_type: Text normalization applied before tokenization.
        val: Enumerate every (session, speaker) pair rather than sampling one speaker per
            session. Set for validation and test so scoring is deterministic and complete.

    Returns:
        A `DataLoader`, or `None` if no manifest was configured.
    """
    if config.get('manifest_filepath') is None:
        return None

    augmentor = process_augmentations(config['augmentor']) if config.get('augmentor') is not None else None

    # Both datasets take the same arguments and emit the same 8-tuple; only the manifest
    # container differs.
    is_lhotse = uses_lhotse_manifest(config)
    dataset_cls = LhotseToBPEAndSTNODataset if is_lhotse else AudioToBPEAndSTNODataset
    logging.info(
        "Reading %s as a %s manifest", config['manifest_filepath'], "Lhotse CutSet" if is_lhotse else "NeMo"
    )

    dataset = dataset_cls(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        channel_selector=config.get('channel_selector', None),
        audio_downsampling_factor=audio_downsampling_factor,
        text_norm_type=config.get('txt_norm_type', text_norm_type),
        val=val,
    )

    if len(dataset) == 0:
        logging.warning("Dataset built from %s is empty.", config['manifest_filepath'])

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset._collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=config.get('shuffle', False),
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', None) if config.get('num_workers', 0) > 0 else None,
    )
