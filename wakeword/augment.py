"""Waveform and spectrogram augmentation.

Waveform-level (numpy, per sample, applied inside the Dataset):
    speed perturbation, light pitch shift, room impulse response (reverb),
    background noise mixing at a random SNR, random gain.
Feature-level (torch, per batch): SpecAugment-style time/frequency masking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from .audio import SAMPLE_RATE, list_audio_files, load_audio


class AudioBank:
    """Lazily-loaded pool of audio files (noise, RIRs, babble speech) with a bounded cache."""

    def __init__(self, files: list[Path], sr: int = SAMPLE_RATE, max_cache: int = 512,
                 max_seconds: float | None = 30.0):
        self.files = list(files)
        self.sr = sr
        self.max_cache = max_cache
        self.max_seconds = max_seconds
        self._cache: dict[int, np.ndarray] = {}

    @classmethod
    def from_dirs(cls, dirs, **kw) -> AudioBank:
        files: list[Path] = []
        for d in dirs or []:
            files.extend(list_audio_files(d))
        return cls(files, **kw)

    def __len__(self) -> int:
        return len(self.files)

    def get(self, i: int) -> np.ndarray:
        if i in self._cache:
            return self._cache[i]
        x = load_audio(self.files[i], self.sr, duration_s=self.max_seconds)
        if len(self._cache) < self.max_cache:
            self._cache[i] = x
        return x

    def sample(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        x = self.get(int(rng.integers(len(self.files))))
        if n is None:
            return x
        if len(x) >= n:
            s = int(rng.integers(0, len(x) - n + 1))
            return x[s:s + n]
        reps = int(np.ceil(n / max(1, len(x))))
        return np.tile(x, reps)[:n]


def speed_perturb(x: np.ndarray, factor: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Change speed (and pitch) by resampling, like Kaldi speed perturbation."""
    if abs(factor - 1.0) < 1e-3:
        return x
    from fractions import Fraction

    from scipy.signal import resample_poly

    # faster speech = fewer samples; small up/down integers keep the polyphase filter cheap
    fr = Fraction(factor).limit_denominator(40)
    return resample_poly(x, fr.denominator, fr.numerator).astype(np.float32)


def pitch_shift(x: np.ndarray, semitones: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return x
    import torchaudio.functional as AF

    y = AF.pitch_shift(torch.from_numpy(x).unsqueeze(0), sr, n_steps=semitones)
    return y.squeeze(0).numpy().astype(np.float32)


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    rir = rir[np.argmax(np.abs(rir)):]  # align direct path
    rir = rir / (np.linalg.norm(rir) + 1e-8)
    y = fftconvolve(x, rir)[: len(x)]
    peak_in, peak_out = np.max(np.abs(x)) + 1e-8, np.max(np.abs(y)) + 1e-8
    return (y * (peak_in / peak_out)).astype(np.float32)


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    px = np.mean(x ** 2) + 1e-10
    pn = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return (x + scale * noise).astype(np.float32)


class WaveAugment:
    def __init__(self, cfg, noise_bank: AudioBank | None = None, rir_bank: AudioBank | None = None,
                 babble_bank: AudioBank | None = None, sr: int = SAMPLE_RATE):
        self.c = cfg
        self.noise = noise_bank if noise_bank is not None and len(noise_bank) else None
        self.rir = rir_bank if rir_bank is not None and len(rir_bank) else None
        self.babble = babble_bank if babble_bank is not None and len(babble_bank) else None
        self.sr = sr

    def pre(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Length-changing augmentations, applied to the raw clip before windowing."""
        c = self.c
        if rng.random() < c.speed_prob:
            x = speed_perturb(x, float(rng.uniform(*c.speed_range)), self.sr)
        if rng.random() < c.pitch_prob:
            x = pitch_shift(x, float(rng.uniform(*c.pitch_semitones)), self.sr)
        return x

    def post(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Length-preserving augmentations, applied to the fixed-size window."""
        c = self.c
        if self.rir is not None and rng.random() < c.rir_prob:
            x = apply_rir(x, self.rir.sample(rng))
        if rng.random() < c.noise_prob:
            bank = self.noise
            if self.babble is not None and (bank is None or rng.random() < c.babble_frac):
                bank = self.babble
            if bank is not None:
                x = mix_at_snr(x, bank.sample(rng, len(x)), float(rng.uniform(*c.snr_db)))
        if rng.random() < c.gain_prob:
            x = x * (10 ** (float(rng.uniform(*c.gain_db)) / 20))
        peak = np.max(np.abs(x))
        if peak > 1.0:
            x = x / peak
        return x.astype(np.float32)


class SpecAugment(torch.nn.Module):
    """Time/frequency masking on (B, 1, F, T) log-mel or (B, T, D) embedding features."""

    def __init__(self, freq_masks: int = 2, freq_width: int = 6, time_masks: int = 2, time_width: int = 15):
        super().__init__()
        self.fm, self.fw, self.tm, self.tw = freq_masks, freq_width, time_masks, time_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = x.clone()
        is_mel = x.dim() == 4
        B = x.shape[0]
        F = x.shape[2] if is_mel else x.shape[2]
        T = x.shape[3] if is_mel else x.shape[1]
        fill = x.mean()
        for b in range(B):
            for _ in range(self.fm):
                w = int(torch.randint(0, self.fw + 1, (1,)))
                f0 = int(torch.randint(0, max(1, F - w), (1,)))
                if is_mel:
                    x[b, :, f0:f0 + w, :] = fill
                else:
                    x[b, :, f0:f0 + w] = fill
            for _ in range(self.tm):
                w = int(torch.randint(0, min(self.tw, max(1, T // 4)) + 1, (1,)))
                t0 = int(torch.randint(0, max(1, T - w), (1,)))
                if is_mel:
                    x[b, :, :, t0:t0 + w] = fill
                else:
                    x[b, t0:t0 + w, :] = fill
        return x
