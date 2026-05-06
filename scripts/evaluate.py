"""Evaluate a trained checkpoint on the validation set.

Reports:
  - Mel L1 (lower is better; primary acoustic-quality proxy)
  - Duration MAE in frames
  - Pitch / Energy MSE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from conformer_tts.data import TTSDataset, collate
from conformer_tts.training import TTSLightningModule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default=None, help="Override split CSV (default: val_meta from config)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = args.split or cfg["data"]["val_meta"]

    ds = TTSDataset(cfg["data"]["root"], split)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    module = TTSLightningModule.load_from_checkpoint(
        args.ckpt, cfg=cfg, map_location=args.device
    ).eval()
    model = module.model.to(args.device)

    n = 0
    mel_l1_sum = 0.0
    dur_mae_sum = 0.0
    pitch_mse_sum = 0.0
    energy_mse_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(dl, desc="eval"):
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                phonemes=batch["phonemes"],
                text_lengths=batch["text_lengths"],
                durations=batch["duration"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                max_mel_len=batch["mel"].shape[1],
            )
            T = batch["mel"].shape[1]
            text_mask = TTSLightningModule._padding_mask(batch["text_lengths"], batch["phonemes"].shape[1])
            m_mask = out.mel_mask[:, :T]

            mel_pred = out.mel[:, :T]
            mel_l1 = ((mel_pred - batch["mel"]).abs() * (~m_mask).unsqueeze(-1)).sum() \
                / (~m_mask).sum().clamp(min=1) / batch["mel"].shape[-1]

            pred_dur = (torch.exp(out.log_duration) - 1.0).clamp(min=0)
            dur_mae = ((pred_dur - batch["duration"].float()).abs() * (~text_mask)).sum() \
                / (~text_mask).sum().clamp(min=1)

            pitch_mse = ((out.pitch - batch["pitch"]) ** 2 * (~text_mask)).sum() \
                / (~text_mask).sum().clamp(min=1)
            energy_mse = ((out.energy - batch["energy"]) ** 2 * (~text_mask)).sum() \
                / (~text_mask).sum().clamp(min=1)

            B = batch["phonemes"].size(0)
            mel_l1_sum += float(mel_l1) * B
            dur_mae_sum += float(dur_mae) * B
            pitch_mse_sum += float(pitch_mse) * B
            energy_mse_sum += float(energy_mse) * B
            n += B

    print(f"\nEvaluated {n} utterances on '{split}':")
    print(f"  Mel L1       : {mel_l1_sum / n:.4f}")
    print(f"  Duration MAE : {dur_mae_sum / n:.3f} frames")
    print(f"  Pitch MSE    : {pitch_mse_sum / n:.4f}")
    print(f"  Energy MSE   : {energy_mse_sum / n:.4f}")


if __name__ == "__main__":
    main()
