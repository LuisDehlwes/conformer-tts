"""Text frontend: clean -> phonemize -> tokenize."""

from __future__ import annotations

from .cleaners import clean_text
from .symbols import BOS_ID, EOS_ID, SYMBOL_TO_ID, UNK_ID

try:
    from phonemizer import phonemize as _phonemize
    from phonemizer.separator import Separator

    _HAS_PHONEMIZER = True
except ImportError:  # pragma: no cover
    _HAS_PHONEMIZER = False


_LANG_MAP = {"de": "de", "en": "en-us"}


def text_to_phonemes(text: str, language: str = "de") -> str:
    if not _HAS_PHONEMIZER:
        raise RuntimeError(
            "phonemizer not installed. `pip install phonemizer` and install eSpeak NG."
        )
    lang = _LANG_MAP.get(language, language)
    out = _phonemize(
        text,
        language=lang,
        backend="espeak",
        separator=Separator(phone="", word=" ", syllable=""),
        strip=True,
        preserve_punctuation=True,
        with_stress=False,
    )
    return out


def encode(
    text: str,
    cleaners: list[str],
    language: str = "de",
    use_phonemes: bool = True,
    add_bos_eos: bool = True,
) -> list[int]:
    text = clean_text(text, cleaners)
    if use_phonemes:
        text = text_to_phonemes(text, language=language)
    ids = [SYMBOL_TO_ID.get(ch, UNK_ID) for ch in text]
    if add_bos_eos:
        ids = [BOS_ID, *ids, EOS_ID]
    return ids
