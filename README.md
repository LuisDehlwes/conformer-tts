# Conformer-TTS — Text-to-Speech Engine

A from-scratch German/English Text-to-Speech engine built with PyTorch using a
**FastSpeech2 acoustic model with Conformer encoder/decoder**, plus a HiFi-GAN
vocoder hook. Companion to [conformer-stt](https://github.com/LuisDehlwes/conformer-stt).

Together they form a full voice loop for LegacyAI: **hear → understand → speak**, all on-prem.

## Features

- **Conformer Encoder & Decoder** — same blocks as `conformer-stt` (Macaron FFN + MHSA + Conv module)
- **FastSpeech2** — non-autoregressive acoustic model with variance adaptors (duration, pitch, energy)
- **DE / EN text frontend** — cleaners (abbrev expansion, normalization) + `phonemizer` (eSpeak NG)
- **PyTorch Lightning** training with mixed precision and Noam LR schedule
- **HiFi-GAN hook + Griffin-Lim fallback** — runs without a vocoder checkpoint out of the box
- **REST API** — FastAPI `/synthesize` endpoint returning WAV
- **HuggingFace Hub** — push and pull trained models
- **ONNX export** of the acoustic model
- **Multiple model sizes** — small (testing), medium (default), large (best quality)
- **Test suite** — pytest (forward-pass, collate, vocoder, frontend, cleaners)

## Project structure

```
conformer-tts/
├── configs/
│   ├── default_small.yaml         # ~5M params, CPU smoke test
│   ├── default.yaml               # ~30M params, default training target
│   └── default_large.yaml         # ~60M params, multi-GPU
├── conformer_tts/
│   ├── data/                      # audio (mel/pitch/energy), dataset, collate
│   ├── inference/                 # Synthesizer + FastAPI server
│   ├── models/                    # Conformer blocks, FastSpeech2, vocoder
│   ├── text/                      # cleaners, phonemizer frontend, symbols
│   └── training/                  # Lightning module + losses
├── scripts/
│   ├── preprocess.py              # wav + text -> mel/pitch/energy/phonemes/durations
│   ├── train.py                   # training entry point
│   ├── synthesize.py              # text -> WAV CLI
│   ├── evaluate.py                # Mel L1 + duration MAE on val
│   ├── export_onnx.py             # acoustic model -> ONNX
│   ├── push_to_hub.py             # upload to HuggingFace Hub
│   └── pull_from_hub.py           # download from Hub & synthesize
├── tests/                         # pytest suite
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Prerequisites

- **Python 3.10–3.12** (PyTorch wheels)
- **eSpeak NG** for phonemization (system package):
  - Windows: https://github.com/espeak-ng/espeak-ng/releases (installer)
  - Linux: `sudo apt install espeak-ng`
  - macOS: `brew install espeak-ng`
- A GPU with ≥ 12 GB VRAM is recommended for training the default config.

## Quick start

### 1. Install

```bash
git clone https://github.com/LuisDehlwes/conformer-tts.git
cd conformer-tts
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate

# Install PyTorch (adjust cu124 to your CUDA version, or use cpu for CPU-only)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare data

**Option A — Thorsten-Voice (DE):** download the 22.05 kHz set from
[thorsten-voice.de](https://www.thorsten-voice.de/) and extract to `data/thorsten/`
so the layout is:

```
data/thorsten/
    wavs/<id>.wav
    metadata.csv         # "<id>|<text>" or "<id>|<text>|<text_norm>"
```

Then run:

```bash
python -m scripts.preprocess --config configs/default.yaml --data-root data/thorsten
```

**Option B — LJSpeech (EN):** same layout under `data/ljspeech/`. Switch
`configs/default.yaml` `text.language` to `en` and `text.cleaners` to
`["english_cleaners"]`.

> ⚠️ The current preprocess uses **uniform durations** (frames spread evenly
> across phonemes). For best quality, run [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/)
> and replace `data/thorsten/durations/` with MFA-derived per-phoneme frame counts.
> See [TTS_ENGINE_PLAN.md](TTS_ENGINE_PLAN.md) — Phase 2.

### 3. Train

```bash
# Smoke test (CPU, minutes):
python -m scripts.train --config configs/default_small.yaml

# Default (~30M params, single GPU):
python -m scripts.train --config configs/default.yaml

# Large (~60M params, multi-GPU):
python -m scripts.train --config configs/default_large.yaml
```

Checkpoints land in `runs/<name>/`. Resume with `--resume runs/<name>/last.ckpt`.

### 4. Evaluate

```bash
python -m scripts.evaluate \
    --config configs/default.yaml \
    --ckpt runs/default/last.ckpt
```

Reports Mel L1, duration MAE (frames), pitch / energy MSE.

### 5. Synthesize a WAV

```bash
python -m scripts.synthesize \
    --config configs/default.yaml \
    --ckpt runs/default/last.ckpt \
    --text "Hallo Welt, ich bin LegacyAI." \
    --out hello.wav
```

### 6. Run the REST API

```bash
TTS_CONFIG=configs/default.yaml \
TTS_CKPT=runs/default/last.ckpt \
uvicorn conformer_tts.inference.server:app --host 0.0.0.0 --port 8001
```

```bash
# health
curl http://localhost:8001/health

# synthesize
curl -X POST http://localhost:8001/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Hallo Welt"}' \
     --output hello.wav
```

### 7. Export to ONNX (acoustic model)

```bash
python -m scripts.export_onnx \
    --config configs/default.yaml \
    --ckpt runs/default/last.ckpt \
    --out models/exported/tts_acoustic.onnx
```

### 8. Push to / pull from HuggingFace Hub

```bash
pip install huggingface_hub
huggingface-cli login

# Push:
python -m scripts.push_to_hub \
    --ckpt runs/default/last.ckpt \
    --config configs/default.yaml \
    --repo-id your-user/conformer-tts-de

# Pull & synthesize:
python -m scripts.pull_from_hub \
    --repo-id your-user/conformer-tts-de \
    --text "Hallo Welt" --out hello.wav
```

## Architecture

```
Text (UTF-8)
  → Cleaners (DE/EN: abbrev expansion, normalization)
  → Phonemizer (eSpeak NG via `phonemizer`)
  → Symbol IDs
  → FastSpeech2 acoustic model:
      ├── Embedding
      ├── Conformer Encoder (N×)
      ├── Variance Adaptors (duration / pitch / energy)
      ├── Length Regulator
      └── Conformer Decoder (N×)
  → Mel Spectrogram (80 bins, 22.05 kHz)
  → Vocoder
      ├── HiFi-GAN  (preferred)
      └── Griffin-Lim  (fallback, no checkpoint)
  → Waveform (22.05 kHz, mono)
```

### Model sizes

| Config | hidden | layers (enc/dec) | heads | Params (≈) | Use |
|---|---|---|---|---|---|
| `default_small.yaml`  | 128 | 2 | 2 | ~5M  | Smoke test / CPU |
| `default.yaml`        | 256 | 4 | 4 | ~30M | Default |
| `default_large.yaml`  | 384 | 6 | 6 | ~60M | Best quality |

## Training configuration

| Item | Default |
|---|---|
| Optimizer | AdamW (lr=1e-4, β=(0.9, 0.98), wd=0.01) |
| Scheduler | Noam (4000 warmup) |
| Loss | Mel L1 + Duration MSE (log+1) + 0.1 · (Pitch + Energy) MSE |
| Mixed precision | FP16 (`16-mixed`) |
| Gradient clipping | 1.0 |
| Audio | 22.05 kHz, 80 mels, n_fft=1024, hop=256 |

## Hardware recommendations

| GPU | VRAM | Batch | Time to ~200k steps (default) |
|---|---|---|---|
| RTX 4070 | 12 GB | 16 | ~24h |
| RTX 4080 | 16 GB | 32 | ~14h |
| RTX 6000 | 48 GB | 64 | ~6h |
| 2× RTX 6000 | 96 GB | 128 | ~3h |

## Datasets

| Dataset | Language | Hours | License | Link |
|---|---|---|---|---|
| Thorsten-Voice | DE | ~23h | CC0 | https://www.thorsten-voice.de/ |
| LJSpeech | EN | ~24h | Public domain | https://keithito.com/LJ-Speech-Dataset/ |

Custom datasets: provide `wavs/<id>.wav` and a `metadata.csv` with
`<id>|<text>` (or `<id>|<text>|<text_norm>`).

## Testing

```bash
pytest tests/ -v
```

## Vocoder

The default vocoder is **Griffin-Lim** (works immediately without a checkpoint,
quality is mediocre). To use **HiFi-GAN**:

1. Drop the `Generator` module from
   [jik876/hifi-gan](https://github.com/jik876/hifi-gan) into
   `conformer_tts/models/hifigan.py`.
2. Set `vocoder.checkpoint` in the config to the `.pt` path.
3. The synthesizer will load it automatically.

## Roadmap

See [TTS_ENGINE_PLAN.md](TTS_ENGINE_PLAN.md) for the full roadmap (Phase 1 / 2 / 3).

## License

MIT
