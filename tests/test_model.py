"""Smoke tests: forward pass, collate, text encode."""

from __future__ import annotations

import torch

from conformer_tts.data import collate
from conformer_tts.models import FastSpeech2
from conformer_tts.text import VOCAB_SIZE


def _cfg() -> dict:
    return {
        "hidden_dim": 64,
        "n_heads": 4,
        "encoder": {"n_layers": 2, "conv_kernel": 7, "ff_expansion": 2, "dropout": 0.0},
        "decoder": {"n_layers": 2, "conv_kernel": 7, "ff_expansion": 2, "dropout": 0.0},
        "variance": {"duration_kernel": 3, "pitch_kernel": 3, "energy_kernel": 3, "dropout": 0.0},
    }


def test_forward_with_gt_durations() -> None:
    torch.manual_seed(0)
    model = FastSpeech2(vocab_size=VOCAB_SIZE, n_mels=80, cfg=_cfg())
    B, T = 2, 7
    phon = torch.randint(1, VOCAB_SIZE, (B, T))
    lens = torch.tensor([T, T - 1])
    dur = torch.randint(1, 4, (B, T))
    pitch = torch.randn(B, T)
    energy = torch.randn(B, T)
    out = model(phon, lens, durations=dur, pitch=pitch, energy=energy)
    assert out.mel.dim() == 3 and out.mel.size(0) == B
    assert out.log_duration.shape == (B, T)


def test_forward_inference_path() -> None:
    torch.manual_seed(0)
    model = FastSpeech2(vocab_size=VOCAB_SIZE, n_mels=80, cfg=_cfg()).eval()
    phon = torch.randint(1, VOCAB_SIZE, (1, 5))
    lens = torch.tensor([5])
    with torch.no_grad():
        out = model(phon, lens)
    assert out.mel.size(0) == 1
    assert out.mel.size(2) == 80


def test_collate() -> None:
    a = {
        "id": "a",
        "phonemes": torch.tensor([1, 2, 3]),
        "mel": torch.randn(10, 80),
        "duration": torch.tensor([3, 4, 3]),
        "pitch": torch.randn(3),
        "energy": torch.randn(3),
    }
    b = {
        "id": "b",
        "phonemes": torch.tensor([4, 5]),
        "mel": torch.randn(7, 80),
        "duration": torch.tensor([4, 3]),
        "pitch": torch.randn(2),
        "energy": torch.randn(2),
    }
    out = collate([a, b])
    assert out["phonemes"].shape == (2, 3)
    assert out["mel"].shape == (2, 10, 80)
    assert out["text_lengths"].tolist() == [3, 2]
    assert out["mel_lengths"].tolist() == [10, 7]
