"""Turning per-window scores into wake events.

The detector slides a ``clip_seconds`` window over the audio with a hop of ``hop_s``. Scores are
smoothed with a causal moving average over ``smooth_k`` windows; a wake event fires when the
smoothed score crosses ``threshold``. After an event the detector is muted for ``refractory_s``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def smooth_scores(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(probs) == 0:
        return probs
    c = np.cumsum(np.insert(probs.astype(np.float64), 0, 0.0))
    out = np.empty(len(probs))
    for i in range(len(probs)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def detect_events(times: np.ndarray, smoothed: np.ndarray, threshold: float, refractory_s: float) -> list[float]:
    idx = np.nonzero(smoothed >= threshold)[0]
    events: list[float] = []
    last = -np.inf
    for i in idx:
        t = float(times[i])
        if t - last >= refractory_s:
            events.append(t)
            last = t
    return events


def count_events_sweep(times: np.ndarray, smoothed: np.ndarray, thresholds: np.ndarray,
                       refractory_s: float) -> np.ndarray:
    return np.array([len(detect_events(times, smoothed, float(t), refractory_s)) for t in thresholds])


@dataclass
class DetectorParams:
    threshold: float = 0.5
    smooth_k: int = 3
    refractory_s: float = 2.0
    hop_s: float = 0.1


class StreamingDetector:
    """Real-time detector. Feed arbitrary-size chunks with ``push``; returns events found in them.

    ``score_fn(window: np.ndarray) -> float`` scores one ``window_samples`` long waveform.
    """

    def __init__(self, score_fn, window_samples: int, sample_rate: int, params: DetectorParams):
        self.score_fn = score_fn
        self.n = window_samples
        self.sr = sample_rate
        self.p = params
        self.hop = max(1, int(params.hop_s * sample_rate))
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.n, dtype=np.float32)
        self.pending = 0
        self.t = 0.0
        self.hist: deque = deque(maxlen=max(1, self.p.smooth_k))
        self.last_event = -np.inf
        self.last_score = 0.0
        self.last_smoothed = 0.0

    def push(self, chunk: np.ndarray) -> list[dict]:
        events = []
        chunk = np.asarray(chunk, dtype=np.float32)
        pos = 0
        while pos < len(chunk):
            take = min(self.hop - self.pending, len(chunk) - pos)
            self.buf = np.concatenate([self.buf[take:], chunk[pos:pos + take]])
            self.pending += take
            pos += take
            self.t += take / self.sr
            if self.pending >= self.hop:
                self.pending = 0
                s = float(self.score_fn(self.buf))
                self.hist.append(s)
                self.last_score = s
                self.last_smoothed = float(np.mean(self.hist))
                if self.last_smoothed >= self.p.threshold and self.t - self.last_event >= self.p.refractory_s:
                    self.last_event = self.t
                    events.append({"time": self.t, "score": self.last_smoothed})
        return events
