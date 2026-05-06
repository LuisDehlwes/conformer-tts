"""Upload a trained TTS checkpoint + config to the HuggingFace Hub.

Usage:
    huggingface-cli login
    python scripts/push_to_hub.py \
        --ckpt runs/default/last.ckpt \
        --config configs/default.yaml \
        --repo-id <user>/conformer-tts-de
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


_MODEL_CARD = """---
license: mit
language:
- {lang}
library_name: pytorch
tags:
- text-to-speech
- tts
- fastspeech2
- conformer
pipeline_tag: text-to-speech
---

# conformer-tts

FastSpeech2 with Conformer encoder/decoder. Companion to
[conformer-stt](https://github.com/LuisDehlwes/conformer-stt).

## Usage

```python
from conformer_tts.inference import Synthesizer
import yaml, soundfile as sf

cfg = yaml.safe_load(open("default.yaml"))
synth = Synthesizer("last.ckpt", cfg, device="cuda")
audio = synth.synthesize("Hallo Welt")
sf.write("hello.wav", audio.numpy(), cfg["audio"]["sample_rate"])
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-id", required=True, help="e.g. user/conformer-tts-de")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message", default="Upload conformer-tts checkpoint")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lang = cfg.get("text", {}).get("language", "de")

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        shutil.copy(args.ckpt, td_path / "last.ckpt")
        shutil.copy(args.config, td_path / Path(args.config).name)
        (td_path / "README.md").write_text(_MODEL_CARD.format(lang=lang), encoding="utf-8")

        api.upload_folder(
            folder_path=str(td_path),
            repo_id=args.repo_id,
            commit_message=args.commit_message,
        )
    print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
