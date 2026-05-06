"""Lightning module: FastSpeech2 + multi-task loss."""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import FastSpeech2
from ..text import VOCAB_SIZE


class TTSLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = FastSpeech2(
            vocab_size=VOCAB_SIZE,
            n_mels=cfg["audio"]["n_mels"],
            cfg=cfg["model"],
        )

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True = pad
        valid = (~mask).unsqueeze(-1) if pred.dim() == 3 else (~mask)
        diff = (pred - target) ** 2
        diff = diff * valid
        denom = valid.sum().clamp(min=1)
        return diff.sum() / denom

    def forward_step(self, batch: dict) -> dict:
        out = self.model(
            phonemes=batch["phonemes"],
            text_lengths=batch["text_lengths"],
            durations=batch["duration"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            max_mel_len=batch["mel"].shape[1],
        )
        text_mask = self._padding_mask(batch["text_lengths"], batch["phonemes"].shape[1])
        mel_mask = out.mel_mask

        # Mel L1
        mel_pred = out.mel[:, : batch["mel"].shape[1]]
        m_mask = mel_mask[:, : batch["mel"].shape[1]]
        mel_l1 = (
            (mel_pred - batch["mel"]).abs() * (~m_mask).unsqueeze(-1)
        ).sum() / (~m_mask).sum().clamp(min=1) / batch["mel"].shape[-1]

        # Duration in log-domain
        log_dur_target = torch.log(batch["duration"].float() + 1.0)
        dur_loss = F.mse_loss(
            out.log_duration[~text_mask], log_dur_target[~text_mask]
        )
        pitch_loss = F.mse_loss(out.pitch[~text_mask], batch["pitch"][~text_mask])
        energy_loss = F.mse_loss(out.energy[~text_mask], batch["energy"][~text_mask])

        loss = mel_l1 + dur_loss + 0.1 * pitch_loss + 0.1 * energy_loss
        return {
            "loss": loss,
            "mel_l1": mel_l1.detach(),
            "dur": dur_loss.detach(),
            "pitch": pitch_loss.detach(),
            "energy": energy_loss.detach(),
        }

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return ar >= lengths.unsqueeze(1)

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        out = self.forward_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in out.items()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return out["loss"]

    def validation_step(self, batch: dict, _: int) -> torch.Tensor:
        out = self.forward_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in out.items()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return out["loss"]

    def configure_optimizers(self):
        tcfg = self.cfg["training"]
        opt = torch.optim.AdamW(
            self.parameters(), lr=tcfg["lr"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01
        )

        warmup = tcfg["warmup_steps"]

        def lr_lambda(step: int) -> float:
            step = max(1, step)
            return min(step ** -0.5, step * warmup ** -1.5) * math.sqrt(warmup)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
