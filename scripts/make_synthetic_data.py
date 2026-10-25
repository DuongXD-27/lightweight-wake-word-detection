#!/usr/bin/env python
"""Generate a tiny *synthetic* dataset with the exact folder layout of the real one.

It is only for smoke tests / CI / trying the demo before real recordings exist. The "wake phrase"
is a fixed sequence of 4 tone syllables; hard negatives are partial / substituted / reordered
sequences; "speakers" differ in base pitch and speaking rate.

    python scripts/make_synthetic_data.py --out data_synth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wakeword.audio import SAMPLE_RATE, save_audio

SR = SAMPLE_RATE
WAKE = [1.0, 1.6, 1.25, 2.0]          # relative pitch of the 4 "syllables" (Alo-GPT-à-em)
SYLLABLES = [0.8, 1.0, 1.25, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]


def syllable(f0: float, dur: float, rng) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    env = np.sin(np.pi * t / dur) ** 2
    x = sum((0.6 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) for k in (1, 2, 3))
    return (x * env).astype(np.float32)


def utterance(rel: list[float], base: float, rate: float, rng, gap: float | None = None) -> np.ndarray:
    parts = []
    for i, r in enumerate(rel):
        parts.append(syllable(base * r * rng.uniform(0.97, 1.03), rng.uniform(0.16, 0.24) / rate, rng))
        g = gap if (gap is not None and i == 1) else rng.uniform(0.02, 0.08) / rate
        parts.append(np.zeros(int(g * SR), np.float32))
    x = np.concatenate(parts)
    x = np.concatenate([np.zeros(int(0.15 * SR), np.float32), x, np.zeros(int(0.15 * SR), np.float32)])
    return 0.5 * x / (np.abs(x).max() + 1e-8)


def hard_variants(rng) -> dict[str, list[float]]:
    sub = list(WAKE)
    sub[rng.integers(4)] = float(rng.choice([s for s in SYLLABLES if s not in WAKE]))
    k = int(rng.integers(1, 4))
    return {
        "hard_partial": WAKE[:k] if rng.random() < 0.5 else WAKE[k:],
        "hard_substitution": sub,
        "hard_reorder": list(rng.permutation(WAKE)) if True else WAKE,
    }


def random_speech(rng, base: float, n_syl: int | None = None) -> np.ndarray:
    n = n_syl or int(rng.integers(2, 9))
    rel = [float(rng.choice(SYLLABLES)) for _ in range(n)]
    if rel[:4] == WAKE:
        rel[0] = 0.8
    return utterance(rel, base, rng.uniform(0.8, 1.2), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--speakers", type=int, default=8)
    ap.add_argument("--per-speaker", type=int, default=12)
    ap.add_argument("--long-minutes", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    raw = Path(args.out) / "raw"

    speakers = [f"spk{i:02d}" for i in range(args.speakers)] + ["ext_guest"]
    for spk in speakers:
        base = rng.uniform(120, 260)
        for j in range(args.per_speaker):
            rate = rng.uniform(0.8, 1.25)
            save_audio(raw / "real" / spk / "positive" / f"p{j:03d}.wav", utterance(WAKE, base, rate, rng))
            for cat, rel in hard_variants(rng).items():
                save_audio(raw / "real" / spk / cat / f"h{j:03d}.wav", utterance(rel, base, rate, rng))
            if j % 3 == 0:
                save_audio(raw / "real" / spk / "hard_long_pause" / f"lp{j:03d}.wav",
                           utterance(WAKE, base, rate, rng, gap=rng.uniform(1.5, 2.0)))
                save_audio(raw / "real" / spk / "neg_length_matched" / f"n{j:03d}.wav",
                           random_speech(rng, base, 4))
        save_audio(raw / "real" / spk / "challenge_mid_pause" / "c000.wav",
                   utterance(WAKE, base, 1.0, rng, gap=1.0))
    for v in ("voiceA", "voiceB"):
        base = rng.uniform(140, 240)
        for j in range(args.per_speaker):
            save_audio(raw / "tts" / v / "positive" / f"t{j:03d}.wav", utterance(WAKE, base, rng.uniform(0.8, 1.2), rng))
            save_audio(raw / "tts" / v / "hard_substitution" / f"t{j:03d}.wav",
                       utterance(hard_variants(rng)["hard_substitution"], base, 1.0, rng))
    for c in range(2):
        for s in range(6):
            base = rng.uniform(100, 280)
            for j in range(10):
                save_audio(raw / "speech" / f"corpus{c}" / f"s{s}" / f"u{j:03d}.wav", random_speech(rng, base))
    for j in range(30):
        n = rng.standard_normal(int(SR * rng.uniform(1, 4))).astype(np.float32)
        if j % 2:
            n = np.cumsum(n) / 50  # brown-ish noise
        save_audio(raw / "noise" / f"noise{j:03d}.wav", 0.3 * n / (np.abs(n).max() + 1e-8))
    for split in ("train", "val", "test"):
        chunks, total = [], 0
        while total < args.long_minutes * 60 * SR:
            x = random_speech(rng, rng.uniform(100, 280)) if rng.random() < 0.7 else np.zeros(int(SR * rng.uniform(0.3, 2)), np.float32)
            chunks.append(x)
            total += len(x)
        long = np.concatenate(chunks)
        long += 0.01 * rng.standard_normal(len(long)).astype(np.float32)
        save_audio(raw / "long" / split / f"stream_{split}.wav", long)
    rir_dir = Path(args.out) / "rir"
    for j in range(5):
        t = np.arange(int(SR * 0.4)) / SR
        rir = rng.standard_normal(len(t)) * np.exp(-t / rng.uniform(0.03, 0.12))
        rir[0] = 3.0
        save_audio(rir_dir / f"rir{j}.wav", (rir / np.abs(rir).max()).astype(np.float32))
    print(f"synthetic dataset written to {args.out}")


if __name__ == "__main__":
    main()
