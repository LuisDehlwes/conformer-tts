"""CLI: synthesize a wav from text."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import yaml

from conformer_tts.inference import Synthesizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    synth = Synthesizer(args.ckpt, cfg, device=args.device)
    audio = synth.synthesize(args.text)
    sf.write(args.out, audio.numpy(), cfg["audio"]["sample_rate"])
    print(f"Wrote {args.out}  ({audio.numel() / cfg['audio']['sample_rate']:.2f}s)")


if __name__ == "__main__":
    main()
