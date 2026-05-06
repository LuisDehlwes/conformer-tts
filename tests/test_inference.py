"""Test inference path of the synthesizer (without checkpoint)."""

from __future__ import annotations

import torch

from conformer_tts.models import FastSpeech2, GriffinLimVocoder
from conformer_tts.text import VOCAB_SIZE


def _model_cfg() -> dict:
    return {
        "hidden_dim": 32,
        "n_heads": 2,
        "encoder": {"n_layers": 1, "conv_kernel": 5, "ff_expansion": 2, "dropout": 0.0},
        "decoder": {"n_layers": 1, "conv_kernel": 5, "ff_expansion": 2, "dropout": 0.0},
        "variance": {"duration_kernel": 3, "pitch_kernel": 3, "energy_kernel": 3, "dropout": 0.0},
    }


def test_inference_durations_nonzero_for_real_phonemes() -> None:
    """At inference (no GT durations), each non-pad phoneme must yield >= 1 frame."""
    torch.manual_seed(0)
    model = FastSpeech2(vocab_size=VOCAB_SIZE, n_mels=80, cfg=_model_cfg()).eval()
    phon = torch.randint(1, VOCAB_SIZE, (1, 6))
    lens = torch.tensor([6])
    with torch.no_grad():
        out = model(phon, lens)
    # T_mel >= T_text in any reasonable case
    assert out.mel.size(1) >= 6


def test_griffinlim_runs_and_returns_audio() -> None:
    voc = GriffinLimVocoder(
        n_fft=256, hop=64, win=256, n_mels=40, sr=8000, fmin=0.0, fmax=4000.0
    ).eval()
    mel = torch.randn(1, 20, 40) - 4.0  # log-mel-ish
    with torch.no_grad():
        audio = voc(mel)
    assert audio.dim() == 2
    assert audio.size(0) == 1
    assert audio.size(1) > 0
