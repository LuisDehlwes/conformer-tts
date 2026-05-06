"""End-to-end synthesizer: text -> mel (FastSpeech2) -> audio (vocoder)."""

from __future__ import annotations

from pathlib import Path

import torch

from ..models import FastSpeech2, load_vocoder
from ..text import VOCAB_SIZE, encode
from ..training import TTSLightningModule


class Synthesizer:
    def __init__(self, ckpt_path: str | Path, cfg: dict, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        module = TTSLightningModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location=self.device
        )
        module.eval()
        self.model: FastSpeech2 = module.model.to(self.device)

        self.vocoder = load_vocoder(cfg["vocoder"], cfg["audio"]).to(self.device)
        self.vocoder.eval()

    @torch.no_grad()
    def synthesize(self, text: str, language: str | None = None) -> torch.Tensor:
        ids = encode(
            text,
            cleaners=self.cfg["text"]["cleaners"],
            language=language or self.cfg["text"]["language"],
            use_phonemes=self.cfg["text"]["use_phonemes"],
        )
        phonemes = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        lengths = torch.tensor([phonemes.size(1)], device=self.device)

        out = self.model(phonemes=phonemes, text_lengths=lengths)
        mel = out.mel  # (1, T, n_mels)

        audio = self.vocoder(mel.transpose(1, 2) if hasattr(self.vocoder, "remove_weight_norm") else mel)
        # GriffinLimVocoder returns (B, samples); HiFi-GAN returns (B, 1, samples)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio.squeeze(0).cpu()
