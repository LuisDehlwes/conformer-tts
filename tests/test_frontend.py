"""Test text frontend: clean -> encode (without phonemizer/eSpeak)."""

from __future__ import annotations

from conformer_tts.text import BOS_ID, EOS_ID, PAD_ID, SYMBOL_TO_ID, VOCAB_SIZE
from conformer_tts.text.frontend import encode


def test_vocab_specials_distinct() -> None:
    ids = {PAD_ID, SYMBOL_TO_ID["<unk>"], BOS_ID, EOS_ID}
    assert len(ids) == 4
    assert PAD_ID == 0


def test_encode_graphemes_with_bos_eos() -> None:
    ids = encode(
        "hallo welt",
        cleaners=["basic_cleaners"],
        language="de",
        use_phonemes=False,
        add_bos_eos=True,
    )
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert all(0 <= i < VOCAB_SIZE for i in ids)


def test_encode_unknown_falls_back_to_unk() -> None:
    ids = encode(
        "ZZ\u0001ZZ",  # control char triggers UNK
        cleaners=["basic_cleaners"],
        language="de",
        use_phonemes=False,
        add_bos_eos=False,
    )
    assert SYMBOL_TO_ID["<unk>"] in ids
