"""Train FastSpeech2 with PyTorch Lightning."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from conformer_tts.data import TTSDataset, collate
from conformer_tts.training import TTSLightningModule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="Path to .ckpt to resume from")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    dcfg = cfg["data"]
    tcfg = cfg["training"]

    train_ds = TTSDataset(dcfg["root"], dcfg["train_meta"])
    val_ds = TTSDataset(dcfg["root"], dcfg["val_meta"])

    train_dl = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=dcfg["num_workers"] > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=dcfg["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=dcfg["num_workers"] > 0,
    )

    module = TTSLightningModule(cfg)

    out_dir = Path(tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="step{step:08d}-loss{val/loss:.3f}",
        every_n_train_steps=tcfg["save_every_n_steps"],
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(str(out_dir), name="tb")

    trainer = pl.Trainer(
        max_steps=tcfg["max_steps"],
        precision=tcfg["precision"],
        gradient_clip_val=tcfg["grad_clip"],
        log_every_n_steps=tcfg["log_every_n_steps"],
        val_check_interval=tcfg["val_check_interval"],
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
