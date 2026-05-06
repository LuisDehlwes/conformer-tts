"""FastSpeech2 acoustic model with Conformer encoder/decoder.

Predicts mel-spectrograms from phoneme IDs. Uses variance adaptors for duration,
pitch, and energy. Reference: Ren et al., "FastSpeech 2" (2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerStack


@dataclass
class FS2Output:
    mel: torch.Tensor              # (B, T_mel, n_mels)
    log_duration: torch.Tensor     # (B, T_text) log-domain prediction
    pitch: torch.Tensor            # (B, T_text)
    energy: torch.Tensor           # (B, T_text)
    mel_mask: torch.Tensor         # (B, T_mel) bool, True = pad


class VariancePredictor(nn.Module):
    """1D-conv stack predicting a scalar per phoneme step."""

    def __init__(self, dim: int, kernel: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=pad),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel, padding=pad),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.transpose(1, 2)
        # Conv -> ReLU
        h = F.relu(self.net[0](h))
        h = self.net[2](h.transpose(1, 2))   # LayerNorm on (B,T,C)
        h = self.net[3](h)
        h = h.transpose(1, 2)
        h = F.relu(self.net[4](h))
        h = self.net[6](h.transpose(1, 2))
        h = self.net[7](h)
        return self.proj(h).squeeze(-1)


def length_regulate(
    x: torch.Tensor, durations: torch.Tensor, max_len: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand encoder outputs by integer durations (per phoneme).

    Returns (expanded, mel_lengths).
    """
    out = []
    mel_lens = []
    for seq, dur in zip(x, durations):
        # clamp negatives, ensure ints
        d = torch.clamp(dur, min=0).long()
        expanded = torch.repeat_interleave(seq, d, dim=0)
        out.append(expanded)
        mel_lens.append(expanded.size(0))
    mel_lens_t = torch.tensor(mel_lens, device=x.device, dtype=torch.long)
    if max_len is not None:
        target_len = max_len
    else:
        target_len = int(mel_lens_t.max().item()) if mel_lens_t.numel() > 0 else 0
    target_len = max(target_len, 1)
    padded = x.new_zeros(len(out), target_len, x.size(-1))
    for i, e in enumerate(out):
        n = min(e.size(0), target_len)
        if n > 0:
            padded[i, :n] = e[:n]
    # mel_lens cannot exceed target_len after truncation
    mel_lens_t = torch.clamp(mel_lens_t, max=target_len)
    return padded, mel_lens_t


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int, cfg: dict) -> None:
        super().__init__()
        dim = cfg["hidden_dim"]
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)

        enc_cfg = cfg["encoder"]
        dec_cfg = cfg["decoder"]
        var_cfg = cfg["variance"]
        n_heads = cfg["n_heads"]

        self.encoder = ConformerStack(
            in_dim=dim,
            dim=dim,
            n_layers=enc_cfg["n_layers"],
            n_heads=n_heads,
            ff_expansion=enc_cfg["ff_expansion"],
            conv_kernel=enc_cfg["conv_kernel"],
            dropout=enc_cfg["dropout"],
        )
        self.duration_predictor = VariancePredictor(
            dim, var_cfg["duration_kernel"], var_cfg["dropout"]
        )
        self.pitch_predictor = VariancePredictor(
            dim, var_cfg["pitch_kernel"], var_cfg["dropout"]
        )
        self.energy_predictor = VariancePredictor(
            dim, var_cfg["energy_kernel"], var_cfg["dropout"]
        )
        self.pitch_embed = nn.Conv1d(1, dim, 9, padding=4)
        self.energy_embed = nn.Conv1d(1, dim, 9, padding=4)

        self.decoder = ConformerStack(
            in_dim=dim,
            dim=dim,
            n_layers=dec_cfg["n_layers"],
            n_heads=n_heads,
            ff_expansion=dec_cfg["ff_expansion"],
            conv_kernel=dec_cfg["conv_kernel"],
            dropout=dec_cfg["dropout"],
        )
        self.mel_proj = nn.Linear(dim, n_mels)

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return ar >= lengths.unsqueeze(1)  # True = pad

    def forward(
        self,
        phonemes: torch.Tensor,         # (B, T_text)
        text_lengths: torch.Tensor,     # (B,)
        durations: torch.Tensor | None = None,   # (B, T_text), training only
        pitch: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        max_mel_len: int | None = None,
    ) -> FS2Output:
        B, T = phonemes.shape
        text_mask = self._padding_mask(text_lengths, T)

        x = self.embed(phonemes)
        x = self.encoder(x, key_padding_mask=text_mask)

        log_dur = self.duration_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)

        # Use ground truth at train time, predictions at inference.
        # Training target for log_dur is log(d + 1), so invert with exp(.) - 1.
        if durations is not None:
            use_dur = durations
        else:
            pred_dur = (torch.exp(log_dur) - 1.0).round().clamp(min=0).long()
            # Avoid dropping all phonemes: keep non-padded positions at >= 1
            non_pad = (~text_mask).long()
            pred_dur = torch.where(non_pad.bool(), pred_dur.clamp(min=1), pred_dur)
            use_dur = pred_dur
        use_pitch = pitch if pitch is not None else pitch_pred
        use_energy = energy if energy is not None else energy_pred

        # Variance embedding (added at phoneme level before length regulation)
        p_emb = self.pitch_embed(use_pitch.unsqueeze(1)).transpose(1, 2)
        e_emb = self.energy_embed(use_energy.unsqueeze(1)).transpose(1, 2)
        x = x + p_emb + e_emb

        expanded, mel_lengths = length_regulate(x, use_dur, max_len=max_mel_len)
        mel_mask = self._padding_mask(mel_lengths, expanded.size(1))

        dec = self.decoder(expanded, key_padding_mask=mel_mask)
        mel = self.mel_proj(dec)

        return FS2Output(
            mel=mel,
            log_duration=log_dur,
            pitch=pitch_pred,
            energy=energy_pred,
            mel_mask=mel_mask,
        )
