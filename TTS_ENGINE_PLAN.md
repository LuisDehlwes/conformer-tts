# Conformer-TTS — Engine Plan

Companion repo to [conformer-stt](https://github.com/LuisDehlwes/conformer-stt). Together
they form a full **voice loop** for LegacyAI: hear (STT) → understand → speak (TTS).

## Vision

A self-hosted, on-prem German Text-to-Speech engine with the same Conformer
architecture used in `conformer-stt`. No OpenAI-TTS, no cloud dependency, no data
leakage. Custom voices possible in Phase 3.

## Architecture

```
Text (UTF-8)
  → Cleaners (DE/EN: abbrev expansion, normalization)
  → Phonemizer (eSpeak NG via `phonemizer`)
  → Symbol IDs (vocab ≈ 220, IPA + ASCII)
  → FastSpeech2 acoustic model:
      ├── Embedding
      ├── Conformer Encoder (N×)
      ├── Variance Adaptors (duration / pitch / energy)
      ├── Length Regulator
      └── Conformer Decoder (N×)
  → Mel Spectrogram (80 bins, 22.05 kHz)
  → Vocoder
      ├── HiFi-GAN  (preferred, swappable)
      └── Griffin-Lim  (fallback, no checkpoint required)
  → Waveform (22.05 kHz, mono)
```

## Model sizes

| Config | hidden | n_layers (enc/dec) | n_heads | Params (≈) | Use |
|---|---|---|---|---|---|
| `default_small.yaml`  | 128 | 2 | 2 | ~5M  | Smoke test, CPU |
| `default.yaml`        | 256 | 4 | 4 | ~30M | Default training target |
| `default_large.yaml`  | 384 | 6 | 6 | ~60M | Best quality (multi-GPU) |

## Roadmap

### Phase 1 — Skeleton + smoke train (current)

- [x] Repo skeleton, Conformer blocks, FastSpeech2, variance adaptors
- [x] DE/EN text frontend (cleaners + phonemizer)
- [x] Preprocess pipeline (Thorsten / LJSpeech)
- [x] PyTorch Lightning training loop with Noam schedule
- [x] FastAPI inference server
- [x] Griffin-Lim fallback vocoder
- [x] Bug fixes: length regulator, duration formula, vocoder buffers
- [ ] First end-to-end training on Thorsten (uniform durations) — verifies pipeline

### Phase 2 — Real quality

- [ ] **Forced alignment** with Montreal Forced Aligner (MFA) → real per-phoneme durations
- [ ] HiFi-GAN integration: bundle pretrained 22.05 kHz checkpoint, document weight loading
- [ ] Mel statistics normalization (CMVN) for stable training
- [ ] Per-speaker embedding stub (multi-speaker prep)
- [ ] Validation audio samples logged to TensorBoard
- [ ] Evaluate script: Mel L1 + Duration MAE on val (✓ done)

### Phase 3 — Production / custom voices

- [ ] Streaming inference (chunked mel synthesis)
- [ ] Speaker embedding from short reference audio (Voice Cloning Lite)
- [ ] HiFi-GAN fine-tune on target speaker
- [ ] ONNX export of full pipeline
- [ ] Dockerfile + Helm chart for K8s deployment
- [ ] Benchmark suite (RTF, MOS proxy, latency)

## Datasets

| Dataset | Language | Hours | License | Link |
|---|---|---|---|---|
| Thorsten-Voice | DE | ~23h | CC0 | https://www.thorsten-voice.de/ |
| LJSpeech | EN | ~24h | Public domain | https://keithito.com/LJ-Speech-Dataset/ |

## Hardware

| GPU | VRAM | Batch | Steps to converge (default) |
|---|---|---|---|
| RTX 4070 | 12 GB | 16 | ~24h |
| RTX 4080 | 16 GB | 32 | ~14h |
| RTX 6000 | 48 GB | 64 | ~6h |
| 2× RTX 6000 | 96 GB | 128 | ~3h |

Training data ≈ 200k steps for default config on Thorsten 23h.

## Strategic value

- **Architectural consistency** with `conformer-stt`: same Conformer blocks, shared utilities, single skill set.
- **Data sovereignty**: on-prem / EU-hosted, audit-friendly.
- **Customizable**: branded / per-speaker voices in Phase 3.
- **No vendor lock-in**: replaces OpenAI / ElevenLabs / Azure TTS.
