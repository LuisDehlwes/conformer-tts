"""Test text cleaners (no espeak required)."""

from conformer_tts.text.cleaners import clean_text


def test_german_cleaners_expands_abbreviations() -> None:
    out = clean_text("Dr. Müller, z.B. heute.", ["german_cleaners"])
    assert "doktor" in out
    assert "zum beispiel" in out


def test_basic_cleaners_collapses_whitespace() -> None:
    out = clean_text("hello   world  ", ["basic_cleaners"])
    assert out == "hello world"
