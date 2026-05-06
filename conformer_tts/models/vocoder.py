"""Vocoder wrapper. Supports HiFi-GAN checkpoints or Griffin-Lim fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class GriffinLimVocoder(nn.Module):
    """Simple Griffin-Lim fallback. Quality is mediocre but no checkpoint required."""

    def __init__(self, n_fft: int, hop: int, win: int, n_mels: int, sr: int, fmin: float, fmax: float) -> None:
        super().__init__()
        import librosa

        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        inv = np.linalg.pinv(mel_basis)
        self.register_buffer("inv_mel", torch.from_numpy(inv).float())
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, n_mels) log-mel
        mel_lin = torch.matmul(torch.exp(mel), self.inv_mel.t()).clamp(min=1e-10)
        spec = mel_lin.transpose(1, 2)  # (B, n_freq, T)
        # Griffin-Lim
        angles = torch.exp(2j * torch.pi * torch.rand_like(spec))
        S = spec.to(torch.complex64) * angles
        for _ in range(32):
            audio = torch.istft(
                S, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                window=self.window, return_complex=False,
            )
            S_new = torch.stft(
                audio, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                window=self.window, return_complex=True,
            )
            S = spec.to(torch.complex64) * (S_new / (S_new.abs() + 1e-10))
        audio = torch.istft(
            S, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window, return_complex=False,
        )
        return audio


def load_vocoder(cfg: dict, audio_cfg: dict) -> nn.Module:
    """Load HiFi-GAN if checkpoint provided, else Griffin-Lim fallback."""
    vtype = cfg.get("type", "hifigan")
    ckpt = cfg.get("checkpoint")
    if vtype == "hifigan" and ckpt and Path(ckpt).exists():
        # Lazy import to keep base install slim. The HiFi-GAN module is expected to expose
        # `Generator` (e.g., from https://github.com/jik876/hifi-gan).
        try:
            from .hifigan import Generator  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "HiFi-GAN module not bundled. Drop `hifigan.py` from jik876/hifi-gan into "
                "conformer_tts/models/ or set vocoder.checkpoint to null for Griffin-Lim."
            ) from e
        gen = Generator()  # config-aware Generator should be passed; placeholder here
        state = torch.load(ckpt, map_location="cpu")
        gen.load_state_dict(state.get("generator", state))
        gen.eval()
        return gen
    return GriffinLimVocoder(
        n_fft=audio_cfg["n_fft"],
        hop=audio_cfg["hop_length"],
        win=audio_cfg["win_length"],
        n_mels=audio_cfg["n_mels"],
        sr=audio_cfg["sample_rate"],
        fmin=audio_cfg["f_min"],
        fmax=audio_cfg["f_max"],
    )
