"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is aligned with the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""
import json
from pathlib import Path

from nemo_text_processing.text_normalization.normalize import Normalizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .basic import BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizerNSF
from .remove_disfluencies import remove_disfluencies

__all__ = [
    "BasicTextNormalizer",
    "EnglishTextNormalizerNSF",
    "get_text_norm",
]

SPELLING_CORRECTIONS_PATH = Path(__file__).with_name("english.json")


def _load_spelling_corrections() -> dict[str, str]:
    with SPELLING_CORRECTIONS_PATH.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _build_whisper_basic_rm_disf():
    normalizer = EnglishTextNormalizer({})

    def whisper_basic_rm_disf(text):
        return remove_disfluencies(normalizer(text))

    return whisper_basic_rm_disf


def _build_nemo_en_cased():
    return Normalizer(input_case="cased", lang="en").normalize


def get_text_norm(t_norm: str):
    if t_norm == "whisper_basic":
        return EnglishTextNormalizer({})
    if t_norm == "whisper_basic_rm_disf":
        return _build_whisper_basic_rm_disf()

    builders = {
        "whisper": lambda: EnglishTextNormalizer(_load_spelling_corrections()),
        "whisper_nsf": EnglishTextNormalizerNSF,
        "nemo_en_cased": _build_nemo_en_cased,
    }
    try:
        return builders[t_norm]()
    except KeyError:
        raise ValueError(f"Unsupported text normalization type: {t_norm}")
