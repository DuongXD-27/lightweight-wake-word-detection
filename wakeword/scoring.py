"""Checkpoint loading and batched scoring of clips / long streams (PyTorch or ONNX back-end)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import SAMPLE_RATE, audio_duration, load_audio
from .config import Cfg, _wrap
from .features import LogMel, OWWEmbedding, build_frontend
from .models import build_model


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg: Cfg = _wrap(ckpt["cfg"])
    frontend = build_frontend(cfg)
    model = build_model(cfg, tuple(ckpt["in_shape"]))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg, frontend, ckpt


class Scorer:
    """Scores fixed-size windows. ``model_fn`` maps a feature batch (torch) -> probabilities (np)."""

    def __init__(self, cfg: Cfg, frontend, model_fn, batch_size: int = 256):
        self.cfg = cfg
        self.frontend = frontend
        self.model_fn = model_fn
        self.sr = cfg.audio.sample_rate
        self.n = int(cfg.audio.clip_seconds * self.sr)
        self.batch_size = batch_size

    # ---- constructors
    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu", **kw) -> Scorer:
        model, cfg, frontend, ckpt = load_checkpoint(path, device)

        @torch.no_grad()
        def fn(feats: torch.Tensor) -> np.ndarray:
            return torch.sigmoid(model(feats.to(device))).cpu().numpy()

        s = cls(cfg, frontend, fn, **kw)
        s.ckpt = ckpt
        s.model = model
        return s

    @classmethod
    def from_onnx(cls, onnx_path, cfg: Cfg, threads: int = 1, **kw) -> Scorer:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def fn(feats: torch.Tensor) -> np.ndarray:
            logits = sess.run(None, {name: feats.numpy().astype(np.float32)})[0]
            return 1.0 / (1.0 + np.exp(-logits.reshape(-1)))

        return cls(cfg, build_frontend(cfg), fn, **kw)

    # ---- scoring
    def score_windows(self, wavs: np.ndarray) -> np.ndarray:
        """wavs: (B, N) -> probs (B,)"""
        out = []
        for i in range(0, len(wavs), self.batch_size):
            feats = self.frontend(torch.from_numpy(np.ascontiguousarray(wavs[i:i + self.batch_size])))
            out.append(self.model_fn(feats))
        return np.concatenate(out) if out else np.zeros(0)

    def score_one(self, wav: np.ndarray) -> float:
        return float(self.score_windows(wav[None, :])[0])

    def score_array(self, audio: np.ndarray, hop_s: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Slide the window over a waveform. Returns (window_end_times_s, probs)."""
        hop = max(1, int(hop_s * self.sr))
        if len(audio) < self.n:
            audio = np.concatenate([np.zeros(self.n - len(audio), dtype=np.float32), audio])
        starts = np.arange(0, len(audio) - self.n + 1, hop)
        if isinstance(self.frontend, LogMel):
            return self._score_array_logmel(audio, starts)
        if isinstance(self.frontend, OWWEmbedding):
            return self._score_array_oww(audio, hop_s)
        probs = []
        for i in range(0, len(starts), self.batch_size):
            idx = starts[i:i + self.batch_size]
            probs.append(self.score_windows(np.stack([audio[s:s + self.n] for s in idx])))
        return (starts + self.n) / self.sr, np.concatenate(probs)

    def _score_array_logmel(self, audio: np.ndarray, starts: np.ndarray):
        """Fast path: compute the log-mel once for the whole stream and slice frame windows."""
        fcfg = self.cfg.features
        hop_len = int(self.sr * fcfg.hop_ms / 1000)
        T = int(self.n // hop_len) + 1
        mel = self.frontend(torch.from_numpy(audio))[0]  # (1, F, T_total)
        probs = []
        for i in range(0, len(starts), self.batch_size):
            idx = starts[i:i + self.batch_size] // hop_len
            idx = np.minimum(idx, mel.shape[-1] - T)
            batch = torch.stack([mel[:, :, j:j + T] for j in idx])
            probs.append(self.model_fn(batch))
        return (starts + self.n) / self.sr, np.concatenate(probs)

    def _score_array_oww(self, audio: np.ndarray, hop_s: float):
        """Fast path: embed the whole stream once (one 96-d vector per 80 ms) and slide over embeddings."""
        emb = self.frontend.embed_one(audio)  # (T_total, 96)
        step_s = OWWEmbedding.STEP * 0.01
        T = self.frontend.embed_one(np.zeros(self.n, np.float32)).shape[0]
        stride = max(1, int(round(hop_s / step_s)))
        idx = np.arange(0, max(1, emb.shape[0] - T + 1), stride)
        probs = []
        for i in range(0, len(idx), self.batch_size):
            batch = torch.from_numpy(np.stack([emb[j:j + T] for j in idx[i:i + self.batch_size]]))
            probs.append(self.model_fn(batch))
        times = (idx * step_s) + self.cfg.audio.clip_seconds
        return times, np.concatenate(probs)

    def score_file(self, path, hop_s: float = 0.1, pad_before_s: float = 0.0, pad_after_s: float = 0.0,
                   block_s: float = 600.0) -> tuple[np.ndarray, np.ndarray]:
        """Score a (possibly multi-hour) file in blocks. Returns (times, probs)."""
        dur = audio_duration(path)
        times_all, probs_all = [], []
        overlap = self.cfg.audio.clip_seconds
        start = 0.0
        while start < dur:
            seg = load_audio(path, self.sr, offset_s=max(0.0, start - overlap) if start > 0 else 0.0,
                             duration_s=block_s + (overlap if start > 0 else 0.0))
            if start == 0 and pad_before_s:
                seg = np.concatenate([np.zeros(int(pad_before_s * self.sr), np.float32), seg])
            if start + block_s >= dur and pad_after_s:
                seg = np.concatenate([seg, np.zeros(int(pad_after_s * self.sr), np.float32)])
            t, p = self.score_array(seg, hop_s)
            offset = (start - overlap) if start > 0 else -pad_before_s
            t = t + offset
            if times_all:
                keep = t > times_all[-1][-1] + 1e-6
                t, p = t[keep], p[keep]
            times_all.append(t)
            probs_all.append(p)
            start += block_s
        return np.concatenate(times_all), np.concatenate(probs_all)


def default_scorer(path: str | Path, device: str = "cpu") -> Scorer:
    path = Path(path)
    if path.suffix == ".onnx":
        cand = Path(str(path).replace(".int8", "")).with_suffix(".pt")
        meta = cand if cand.exists() else None
        if meta is None:
            raise FileNotFoundError("ONNX scoring needs the matching .pt checkpoint next to it for the config")
        _, cfg, _, _ = load_checkpoint(meta)
        return Scorer.from_onnx(path, cfg)
    return Scorer.from_checkpoint(path, device)


__all__ = ["Scorer", "load_checkpoint", "default_scorer", "SAMPLE_RATE"]
