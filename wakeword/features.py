"""Feature front-ends.

* ``logmel``: log-Mel filterbank (default 40 bins, 25 ms window, 10 ms hop) -> (B, 1, n_mels, T)
* ``oww``: frozen openWakeWord front-end (Google speech-embedding model, ONNX) -> (B, T_emb, 96).
  This is "approach C" (transfer learning from a pretrained speech embedding).
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .audio import SAMPLE_RATE


class LogMel(nn.Module):
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = 40, win_ms: float = 25.0,
                 hop_ms: float = 10.0, n_fft: int = 512, f_min: float = 20.0, f_max: float | None = 7600.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=int(sample_rate * win_ms / 1000),
            hop_length=int(sample_rate * hop_ms / 1000), f_min=f_min, f_max=f_max, n_mels=n_mels,
            power=2.0, center=True,
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, 1, n_mels, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return torch.log(self.mel(wav) + 1e-6).unsqueeze(1)


OWW_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
OWW_FILES = ("melspectrogram.onnx", "embedding_model.onnx")


def ensure_oww_models(model_dir: str | Path = "pretrained/oww") -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in OWW_FILES:
        dst = model_dir / name
        if not dst.exists():
            print(f"[oww] downloading {name} -> {dst}")
            urllib.request.urlretrieve(f"{OWW_RELEASE}/{name}", dst)
    return model_dir


class OWWEmbedding:
    """Frozen openWakeWord front-end: audio -> 32-bin mel (10 ms) -> 96-d embedding every 80 ms.

    A 2 s window yields 16 embedding frames. Sessions are created lazily per process so the
    object is safe to use inside DataLoader workers.
    """

    WINDOW = 76
    STEP = 8

    def __init__(self, model_dir: str | Path = "pretrained/oww", threads: int = 1):
        self.model_dir = ensure_oww_models(model_dir)
        self.threads = threads
        self._local = threading.local()

    def _sessions(self):
        if not hasattr(self._local, "mel"):
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1
            self._local.mel = ort.InferenceSession(str(self.model_dir / OWW_FILES[0]), opts,
                                                   providers=["CPUExecutionProvider"])
            self._local.emb = ort.InferenceSession(str(self.model_dir / OWW_FILES[1]), opts,
                                                   providers=["CPUExecutionProvider"])
        return self._local.mel, self._local.emb

    def __getstate__(self):  # do not pickle ORT sessions into workers
        return {"model_dir": self.model_dir, "threads": self.threads}

    def __setstate__(self, state):
        self.model_dir = state["model_dir"]
        self.threads = state["threads"]
        self._local = threading.local()

    def embed_one(self, wav: np.ndarray) -> np.ndarray:
        mel_s, emb_s = self._sessions()
        x = (np.asarray(wav, dtype=np.float32) * 32767.0)[None, :]
        spec = np.squeeze(mel_s.run(None, {"input": x})[0]) / 10.0 + 2.0  # (frames, 32)
        n = (spec.shape[0] - self.WINDOW) // self.STEP + 1
        if n <= 0:
            raise ValueError("audio too short for the openWakeWord embedding front-end (>~0.8 s needed)")
        wins = np.stack([spec[i * self.STEP: i * self.STEP + self.WINDOW] for i in range(n)])[..., None]
        out = emb_s.run(None, {emb_s.get_inputs()[0].name: wins.astype(np.float32)})[0]
        return out.reshape(n, -1).astype(np.float32)  # (T_emb, 96)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, T_emb, 96)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        arr = wav.detach().cpu().numpy()
        return torch.from_numpy(np.stack([self.embed_one(a) for a in arr]))


def build_frontend(cfg) -> object:
    fcfg = cfg.features
    if fcfg.type == "logmel":
        return LogMel(sample_rate=cfg.audio.sample_rate, n_mels=fcfg.n_mels, win_ms=fcfg.win_ms,
                      hop_ms=fcfg.hop_ms, n_fft=fcfg.n_fft)
    if fcfg.type == "oww":
        return OWWEmbedding(model_dir=fcfg.get("model_dir", "pretrained/oww"))
    raise ValueError(f"unknown feature type: {fcfg.type}")


def feature_shape(cfg, frontend=None) -> tuple[int, ...]:
    """Per-sample feature shape for the configured window length."""
    frontend = frontend or build_frontend(cfg)
    n = int(cfg.audio.clip_seconds * cfg.audio.sample_rate)
    with torch.no_grad():
        f = frontend(torch.zeros(1, n))
    return tuple(f.shape[1:])
