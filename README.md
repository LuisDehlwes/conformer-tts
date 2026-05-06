# conformer-tts

Conformer-based Text-to-Speech for LegacyAI. Companion repo to `conformer-stt`.

## Architecture

- **Text frontend**: graphemes → cleaned text → phonemes (`phonemizer` + `espeak-ng`)
- **Acoustic model**: FastSpeech2 with **Conformer encoder/decoder** → mel-spectrogram
- **Vocoder**: HiFi-GAN (pretrained, swappable) → 22.05 kHz waveform
- **Training**: PyTorch Lightning, mixed precision
- **Serving**: FastAPI + WebSocket streaming

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"
# system deps for phonemizer:
#   Windows: install eSpeak NG from https://github.com/espeak-ng/espeak-ng/releases
#   Linux:   sudo apt install espeak-ng

# 2. Preprocess (Thorsten DE or LJSpeech)
python -m scripts.preprocess --config configs/default.yaml --data-root <path>

# 3. Train
python -m scripts.train --config configs/default.yaml

# 4. Synthesize
python -m scripts.synthesize --text "Hallo Welt" --ckpt runs/last.ckpt --out hello.wav

# 5. Serve
uvicorn conformer_tts.inference.server:app --host 0.0.0.0 --port 8001
```

## Datasets

- **DE**: [Thorsten-Voice 22.05kHz](https://www.thorsten-voice.de/) (~23h, CC0)
- **EN**: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)

## Status

- [x] Repo skeleton
- [x] Conformer encoder (shared with STT)
- [x] FastSpeech2 acoustic model
- [x] Text frontend (DE/EN)
- [x] Dataset + preprocess
- [x] Lightning training loop
- [x] HiFi-GAN inference wrapper
- [x] FastAPI server
- [ ] Trained checkpoint (run training)
- [ ] Streaming inference

## License

MIT
