#!/usr/bin/env python
"""Build hours of continuous negative audio (for FA/hour and hard-negative mining) by chaining
negative speech utterances with random gaps and background noise.

Real long recordings (TV, podcasts, meetings, a room with people talking) are better — put those
directly in data/raw/long/<split>/. This script fills the gap and respects the speaker split, so
no utterance used here leaks across splits.

    python scripts/make_long_negatives.py --hours 2 --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wakeword.audio import SAMPLE_RATE, list_audio_files, load_audio, save_audio
from wakeword.augment import mix_at_snr
from wakeword.config import load_config
from wakeword.data import assign_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--file-minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    raw = Path(cfg.paths.raw)

    speech = []
    for corpus in sorted((raw / "speech").glob("*")):
        for spk in sorted(corpus.glob("*")):
            if assign_split(f"{corpus.name}:{spk.stem}", cfg.split.val_ratio, cfg.split.test_ratio, "speech") == args.split:
                speech += list_audio_files(spk) if spk.is_dir() else [spk]
    noise = [p for p in list_audio_files(raw / "noise")
             if assign_split(p.as_posix(), cfg.split.val_ratio, cfg.split.test_ratio, "noise") == args.split]
    if not speech:
        sys.exit("no negative speech for this split under data/raw/speech")

    out_dir = raw / "long" / args.split
    n_files = int(np.ceil(args.hours * 60 / args.file_minutes))
    for k in range(n_files):
        target = int(args.file_minutes * 60 * SAMPLE_RATE)
        chunks, total = [], 0
        while total < target:
            x = load_audio(speech[rng.integers(len(speech))])
            x *= 10 ** (rng.uniform(-12, 3) / 20)
            gap = np.zeros(int(rng.exponential(1.0) * SAMPLE_RATE), np.float32)
            chunks += [x, gap]
            total += len(x) + len(gap)
        stream = np.concatenate(chunks)[:target]
        if noise:
            bed = np.concatenate([load_audio(noise[rng.integers(len(noise))]) for _ in range(20)])
            bed = np.tile(bed, int(np.ceil(target / len(bed))))[:target]
            stream = mix_at_snr(stream, bed, float(rng.uniform(5, 25)))
        stream /= max(1.0, np.abs(stream).max())
        save_audio(out_dir / f"synthetic_long_{k:02d}.wav", stream)
        print(f"wrote {out_dir / f'synthetic_long_{k:02d}.wav'} ({args.file_minutes} min)")


if __name__ == "__main__":
    main()
