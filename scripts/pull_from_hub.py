"""Download a conformer-tts model from the HuggingFace Hub and synthesize.

Usage:
    python scripts/pull_from_hub.py --repo-id <user>/conformer-tts-de \
        --text "Hallo Welt" --out hello.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import yaml

from conformer_tts.inference import Synthesizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--config-name", default="default.yaml")
    ap.add_argument("--ckpt-name", default="last.ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e

    ckpt = hf_hub_download(args.repo_id, args.ckpt_name, cache_dir=args.cache_dir)
    cfg_path = hf_hub_download(args.repo_id, args.config_name, cache_dir=args.cache_dir)

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    synth = Synthesizer(ckpt, cfg, device=args.device)
    audio = synth.synthesize(args.text)
    sf.write(args.out, audio.numpy(), cfg["audio"]["sample_rate"])
    print(f"Wrote {args.out}  ({audio.numel() / cfg['audio']['sample_rate']:.2f}s)")


if __name__ == "__main__":
    main()
