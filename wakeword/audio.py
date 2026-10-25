"""Audio I/O and basic waveform utilities (numpy, mono float32 in [-1, 1])."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import torch
    import torchaudio.functional as AF

    y = AF.resample(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), sr_in, sr_out)
    return y.numpy()


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def load_audio(path: str | Path, sr: int = SAMPLE_RATE, offset_s: float = 0.0,
               duration_s: float | None = None) -> np.ndarray:
    """Load (a slice of) an audio file as mono float32 at `sr`."""
    path = str(path)
    try:
        info = sf.info(path)
        start = int(offset_s * info.samplerate)
        frames = -1 if duration_s is None else int(duration_s * info.samplerate)
        x, file_sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=False)
    except Exception:  # formats libsndfile cannot read -> ffmpeg fallback
        x, file_sr = _ffmpeg_read(path, sr)
        if offset_s or duration_s is not None:
            s = int(offset_s * file_sr)
            e = None if duration_s is None else s + int(duration_s * file_sr)
            x = x[s:e]
    x = to_mono(np.asarray(x, dtype=np.float32))
    return resample(x, file_sr, sr).astype(np.float32)


def _ffmpeg_read(path: str, sr: int) -> tuple[np.ndarray, int]:
    import subprocess

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy(), sr


def audio_duration(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        return len(load_audio(path)) / SAMPLE_RATE


def save_audio(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def rms_db(x: np.ndarray, eps: float = 1e-10) -> float:
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + eps))


def trim_silence(x: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 35.0,
                 frame_ms: float = 20.0, pad_ms: float = 80.0) -> np.ndarray:
    """Energy-based trim of leading/trailing silence (relative to loudest frame)."""
    if len(x) == 0:
        return x
    hop = max(1, int(sr * frame_ms / 1000))
    n = int(np.ceil(len(x) / hop))
    padded = np.pad(x, (0, n * hop - len(x)))
    energy = 10 * np.log10(np.mean(padded.reshape(n, hop) ** 2, axis=1) + 1e-10)
    active = np.where(energy > energy.max() - top_db)[0]
    if len(active) == 0:
        return x
    pad = int(sr * pad_ms / 1000)
    s = max(0, active[0] * hop - pad)
    e = min(len(x), (active[-1] + 1) * hop + pad)
    return x[s:e]


def fit_to_length(x: np.ndarray, n: int, mode: str = "random", rng: np.random.Generator | None = None,
                  keep: str = "end") -> np.ndarray:
    """Pad (with zeros) or crop `x` to exactly n samples.

    mode="random": random placement/crop; mode="center"; mode="end": right-align.
    When cropping a clip that is longer than the window, `keep` decides which part survives
    in non-random modes.
    """
    rng = rng or np.random.default_rng()
    L = len(x)
    if L == n:
        return x
    if L > n:
        if mode == "random":
            s = int(rng.integers(0, L - n + 1))
        elif mode == "center":
            s = (L - n) // 2
        else:
            s = L - n if keep == "end" else 0
        return x[s:s + n]
    out = np.zeros(n, dtype=np.float32)
    if mode == "random":
        s = int(rng.integers(0, n - L + 1))
    elif mode == "center":
        s = (n - L) // 2
    else:
        s = n - L
    out[s:s + L] = x
    return out


def list_audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())
