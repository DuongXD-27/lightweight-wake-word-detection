#!/usr/bin/env python
"""Compose "Alo GPT" + <pause> + "à em" from recorded/TTS parts to probe the pause policy.

  pause <= short_max (0.5 s)   -> positive_pause      (should wake)
  pause >= long_min  (1.5 s)   -> hard_long_pause     (should NOT wake)
  in between                   -> challenge_mid_pause (reported separately)

Parts come from data/tts_parts/<voice>/part_head__*.wav and part_tail__*.wav (generate_tts.py) and/or
data/parts/<speaker>/part_head*.wav, part_tail*.wav recorded with the collection app.
Outputs go to data/raw/<tts|real>/<voice_or_speaker>/<category>/compose_*.wav, so they inherit the
speaker split of their source voice.

    python scripts/compose_pauses.py --per-voice 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wakeword.audio import SAMPLE_RATE, list_audio_files, load_audio, save_audio, trim_silence


def compose(head: np.ndarray, tail: np.ndarray, pause_s: float, rng) -> np.ndarray:
    floor = rng.standard_normal(int(pause_s * SAMPLE_RATE)).astype(np.float32) * 1e-4
    return np.concatenate([head, floor, tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["tts:data/tts_parts", "real:data/parts"])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--per-voice", type=int, default=30)
    ap.add_argument("--short-max", type=float, default=0.5)
    ap.add_argument("--long-min", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    total = 0
    for spec in args.sources:
        kind, root = spec.split(":", 1)
        for vdir in sorted(Path(root).glob("*")) if Path(root).exists() else []:
            heads = [f for f in list_audio_files(vdir) if f.name.startswith("part_head")]
            tails = [f for f in list_audio_files(vdir) if f.name.startswith("part_tail")]
            if not heads or not tails:
                continue
            for k in range(args.per_voice):
                h = trim_silence(load_audio(heads[rng.integers(len(heads))]))
                t = trim_silence(load_audio(tails[rng.integers(len(tails))]))
                r = rng.random()
                if r < 0.4:
                    pause, cat = rng.uniform(0.05, args.short_max), "positive_pause"
                elif r < 0.8:
                    pause, cat = rng.uniform(args.long_min, args.long_min + 1.0), "hard_long_pause"
                else:
                    pause, cat = rng.uniform(args.short_max + 0.1, args.long_min - 0.1), "challenge_mid_pause"
                dst = Path(args.raw) / kind / vdir.name / cat / f"compose_{k:04d}_p{pause:.2f}.wav"
                save_audio(dst, compose(h, t, pause, rng))
                total += 1
    print(f"composed {total} clips")


if __name__ == "__main__":
    main()
