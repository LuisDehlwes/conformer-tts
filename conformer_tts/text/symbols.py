"""Phoneme/grapheme symbol set used as the model vocabulary."""

# Index 0 reserved for padding, 1 for unknown.
PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"

_special = [PAD, UNK, BOS, EOS]

_punct = list(" !?,.;:'-\"()")

# IPA + ASCII fallback. Covers common DE/EN phonemes from espeak-ng.
_phonemes = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "äöüÄÖÜß"
    "ɐɑɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘"
)

SYMBOLS: list[str] = _special + _punct + _phonemes

SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

VOCAB_SIZE = len(SYMBOLS)
PAD_ID = SYMBOL_TO_ID[PAD]
UNK_ID = SYMBOL_TO_ID[UNK]
BOS_ID = SYMBOL_TO_ID[BOS]
EOS_ID = SYMBOL_TO_ID[EOS]
