"""Text cleaners for German and English."""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

_whitespace_re = re.compile(r"\s+")

_de_abbrev = {
    "Dr.": "Doktor",
    "Prof.": "Professor",
    "z.B.": "zum Beispiel",
    "z. B.": "zum Beispiel",
    "u.a.": "unter anderem",
    "etc.": "et cetera",
    "ggf.": "gegebenenfalls",
    "bzw.": "beziehungsweise",
    "ca.": "circa",
    "Nr.": "Nummer",
    "Mr.": "Mister",
    "Mrs.": "Misses",
}


def collapse_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()


def expand_abbreviations(text: str, mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def basic_cleaners(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return collapse_whitespace(text)


def english_cleaners(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = text.lower()
    return collapse_whitespace(text)


def german_cleaners(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = expand_abbreviations(text, _de_abbrev)
    # keep umlauts; lower-case
    text = text.lower()
    return collapse_whitespace(text)


CLEANERS = {
    "basic_cleaners": basic_cleaners,
    "english_cleaners": english_cleaners,
    "german_cleaners": german_cleaners,
}


def clean_text(text: str, cleaners: list[str]) -> str:
    for c in cleaners:
        if c not in CLEANERS:
            raise ValueError(f"Unknown cleaner: {c}")
        text = CLEANERS[c](text)
    return text
