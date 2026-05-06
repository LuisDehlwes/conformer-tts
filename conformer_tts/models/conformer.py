"""Conformer encoder block (shared design with conformer-stt).

Conformer = Macaron FFN + Multi-Head Self-Attention + Conv module + FFN + LayerNorm.
Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for ASR" (2020).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        inner = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (absolute; sufficient for FastSpeech-style TTS)."""

    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(out)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise -> GLU -> depthwise -> BN -> Swish -> pointwise."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.dw = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = Swish()
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.norm(x).transpose(1, 2)  # (B, C, T)
        h = self.pw1(h)
        h = F.glu(h, dim=1)
        h = self.dw(h)
        h = self.bn(h)
        h = self.act(h)
        h = self.pw2(h)
        h = self.dropout(h)
        return h.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.conv = ConvModule(dim, conv_kernel, dropout)
        self.ff2 = FeedForward(dim, ff_expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerStack(nn.Module):
    """Stack of N Conformer blocks with input projection + positional encoding."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.pos = RelPositionalEncoding(dim)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(dim, n_heads, ff_expansion, conv_kernel, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x
