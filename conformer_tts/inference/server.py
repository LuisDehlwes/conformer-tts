"""FastAPI server exposing /synthesize endpoint."""

from __future__ import annotations

import io
import os
from pathlib import Path

import soundfile as sf
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from .synthesizer import Synthesizer

CONFIG_PATH = os.environ.get("TTS_CONFIG", "configs/default.yaml")
CKPT_PATH = os.environ.get("TTS_CKPT", "runs/default/last.ckpt")
DEVICE = os.environ.get("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class SynthRequest(BaseModel):
    text: str
    language: str | None = None  # override config


app = FastAPI(title="conformer-tts", version="0.1.0")
_synth: Synthesizer | None = None
_cfg: dict | None = None


def _ensure_loaded() -> Synthesizer:
    global _synth, _cfg
    if _synth is None:
        cfg_path = Path(CONFIG_PATH)
        if not cfg_path.exists():
            raise HTTPException(500, f"Config not found: {cfg_path}")
        _cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        ckpt = Path(CKPT_PATH)
        if not ckpt.exists():
            raise HTTPException(
                500, f"Checkpoint not found: {ckpt}. Train first or set TTS_CKPT."
            )
        _synth = Synthesizer(str(ckpt), _cfg, device=DEVICE)
    return _synth


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE, "ckpt_exists": Path(CKPT_PATH).exists()}


@app.post("/synthesize")
def synthesize(req: SynthRequest) -> Response:
    synth = _ensure_loaded()
    language = req.language or synth.cfg["text"]["language"]
    audio = synth.synthesize(req.text, language=language)
    sr = synth.cfg["audio"]["sample_rate"]

    buf = io.BytesIO()
    sf.write(buf, audio.numpy(), sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
