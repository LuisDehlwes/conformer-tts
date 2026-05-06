"""Preprocess a Thorsten / LJSpeech-style dataset.

Expected input layout::

    <data_root>/
        wavs/<id>.wav
        metadata.csv     # "<id>|<text>" lines (LJSpeech) or "<id>|<text>|<text_norm>" (Thorsten)

Output: extends <data_root>/ with mels/, durations/, pitch/, energy/, phonemes/
plus train/val splits.

Note: Duration extraction here uses a simple uniform fallback. For high quality,
run Montreal Forced Aligner (MFA) and replace the durations folder.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from conformer_tts.data.audio import (
    AudioConfig,
    aggregate_per_phoneme,
    compute_energy,
    compute_log_mel,
    compute_pitch,
    load_wav,
    normalize_pitch_energy,
)
from conformer_tts.text import encode


def parse_metadata(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        uid = parts[0].strip()
        # prefer normalized column if present (Thorsten/LJSpeech)
        text = parts[-1].strip() if len(parts) >= 3 else parts[1].strip()
        if uid and text:
            items.append((uid, text))
    return items


def uniform_durations(n_phon: int, n_frames: int) -> np.ndarray:
    """Fallback duration: spread frames evenly across phonemes.

    NOT suitable for high-quality TTS; replace with MFA-aligned durations.
    """
    if n_phon == 0:
        return np.zeros(0, dtype=np.int64)
    base = n_frames // n_phon
    rem = n_frames - base * n_phon
    durs = np.full(n_phon, base, dtype=np.int64)
    durs[:rem] += 1
    return durs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    audio_cfg = AudioConfig(**cfg["audio"])
    text_cfg = cfg["text"]
    root = Path(args.data_root)

    for sub in ("mels", "durations", "pitch", "energy", "phonemes"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    items = parse_metadata(root / args.metadata)
    random.Random(args.seed).shuffle(items)
    n_val = max(1, int(len(items) * args.val_fraction))
    val, train = items[:n_val], items[n_val:]

    def write_split(name: str, rows: list[tuple[str, str]]) -> None:
        (root / f"{name}.csv").write_text(
            "\n".join(uid for uid, _ in rows), encoding="utf-8"
        )

    print(f"Total: {len(items)} | Train: {len(train)} | Val: {len(val)}")

    n_ok = 0
    for uid, text in tqdm(items, desc="preprocess"):
        wav_path = root / "wavs" / f"{uid}.wav"
        if not wav_path.exists():
            continue
        try:
            wav = load_wav(str(wav_path), audio_cfg.sample_rate)
            mel = compute_log_mel(wav, audio_cfg)              # (T_mel, n_mels)
            energy = compute_energy(wav, audio_cfg)            # (T_mel,)
            pitch = compute_pitch(wav, audio_cfg)              # (~T_mel,)

            # Align lengths
            T = mel.shape[0]
            energy = energy[:T] if energy.numel() >= T else torch.nn.functional.pad(
                energy, (0, T - energy.numel())
            )
            pitch = pitch[:T] if pitch.numel() >= T else torch.nn.functional.pad(
                pitch, (0, T - pitch.numel())
            )

            phon_ids = encode(
                text,
                cleaners=text_cfg["cleaners"],
                language=text_cfg["language"],
                use_phonemes=text_cfg["use_phonemes"],
            )
            phon = np.array(phon_ids, dtype=np.int64)
            durations = uniform_durations(phon.size, T)

            pitch_phon = aggregate_per_phoneme(pitch, torch.from_numpy(durations))
            energy_phon = aggregate_per_phoneme(energy, torch.from_numpy(durations))
            pitch_phon = normalize_pitch_energy(pitch_phon)
            energy_phon = normalize_pitch_energy(energy_phon)

            np.save(root / "mels" / f"{uid}.npy", mel.numpy().astype(np.float32))
            np.save(root / "durations" / f"{uid}.npy", durations)
            np.save(root / "pitch" / f"{uid}.npy", pitch_phon.numpy().astype(np.float32))
            np.save(root / "energy" / f"{uid}.npy", energy_phon.numpy().astype(np.float32))
            np.save(root / "phonemes" / f"{uid}.npy", phon)
            n_ok += 1
        except Exception as exc:  # pragma: no cover
            print(f"[skip] {uid}: {exc}")

    write_split(cfg["data"]["train_meta"].replace(".csv", ""), train)
    write_split(cfg["data"]["val_meta"].replace(".csv", ""), val)
    print(f"Wrote {n_ok}/{len(items)} utterances.")


if __name__ == "__main__":
    main()
