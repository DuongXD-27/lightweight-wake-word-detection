#!/usr/bin/env python
"""Scan data/raw (+ data/mined) and write a speaker-disjoint manifest CSV.

    python scripts/build_manifest.py -c configs/base.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakeword.config import load_config
from wakeword.data import check_leakage, scan_raw, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-mined", action="store_true", help="ignore data/mined (e.g. for the V1 model)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    df = scan_raw(cfg.paths.raw, None if args.no_mined else cfg.paths.mined, cfg.split)
    if df.empty:
        sys.exit(f"No audio found under {cfg.paths.raw}. See README 'Data layout'.")
    out = Path(cfg.paths.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(summarize(df).to_string())
    print("\nSpeakers per split:")
    print(df[df.source.isin(["real", "tts"])].groupby("split").speaker.unique().to_string())
    leaks = check_leakage(df)
    if leaks:
        print(f"\n!! WARNING: speakers in several splits: {leaks}")
    pos = df[(df.label == 1) & (df.source.isin(["real", "tts"]))]
    if len(pos):
        too_long = (pos.duration > cfg.audio.clip_seconds + 0.5).mean()
        print(f"\nPositive duration (s): median {pos.duration.median():.2f}, p95 {pos.duration.quantile(.95):.2f}"
              f" | window = {cfg.audio.clip_seconds}s"
              + (f"  (!! {too_long:.0%} positives > window+0.5s before trimming)" if too_long > 0.1 else ""))
    hours = df[df.source == "long"].groupby("split").duration.sum() / 3600
    if len(hours):
        print("\nLong negative audio (hours):", hours.round(2).to_dict())
    print(f"\nwrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
