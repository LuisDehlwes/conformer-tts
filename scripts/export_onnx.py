"""Export the FastSpeech2 acoustic model to ONNX.

Note: only the acoustic model (text -> mel) is exported. Vocoder export is left
out because HiFi-GAN ONNX requires a config-aware Generator. For full pipelines,
deploy the vocoder via TorchScript or a separate HiFi-GAN ONNX file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from conformer_tts.training import TTSLightningModule


class _AcousticWrapper(torch.nn.Module):
    """Wraps FastSpeech2 to a fixed signature for ONNX tracing."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, phonemes: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        out = self.model(phonemes=phonemes, text_lengths=text_lengths)
        return out.mel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models/exported/tts_acoustic.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module = TTSLightningModule.load_from_checkpoint(
        args.ckpt, cfg=cfg, map_location="cpu"
    ).eval()
    wrapped = _AcousticWrapper(module.model).eval()

    dummy_phon = torch.randint(1, 32, (1, 24), dtype=torch.long)
    dummy_lens = torch.tensor([24], dtype=torch.long)

    torch.onnx.export(
        wrapped,
        (dummy_phon, dummy_lens),
        str(out_path),
        input_names=["phonemes", "text_lengths"],
        output_names=["mel"],
        dynamic_axes={
            "phonemes": {0: "batch", 1: "T_text"},
            "text_lengths": {0: "batch"},
            "mel": {0: "batch", 1: "T_mel"},
        },
        opset_version=args.opset,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
