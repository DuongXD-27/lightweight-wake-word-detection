#!/usr/bin/env python
"""Hard-negative mining (Model V1 -> V2).

Runs a trained model over long NEGATIVE audio of the *train* split (never test!), keeps windows
whose score exceeds ``--min-score``, applies non-maximum suppression and writes the top-K windows
as new training clips under data/mined/<tag>/. Rebuild the manifest afterwards and retrain.

    python scripts/mine_hard_negatives.py --checkpoint runs/abl_a3_plus_hardneg/best.pt --tag v1
    python scripts/build_manifest.py -c configs/tc_resnet.yaml
    python scripts/train.py -c configs/ablation/a4_plus_mined.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from wakeword.audio import load_audio, save_audio
from wakeword.data import read_manifest
from wakeword.scoring import Scorer
from wakeword.streaming import smooth_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument("--nms-s", type=float, default=1.0, help="min distance between mined windows")
    ap.add_argument("--include-speech", action="store_true",
                    help="also mine from short neg_speech clips of the train split")
    args = ap.parse_args()

    scorer = Scorer.from_checkpoint(args.checkpoint)
    cfg = scorer.cfg
    manifest = read_manifest(cfg.paths.manifest)
    sources = ["long"] + (["speech"] if args.include_speech else [])
    files = manifest[(manifest.split == "train") & (manifest.label == 0) & manifest.source.isin(sources)]
    if files.empty:
        sys.exit("No train-split negative audio to mine from (data/raw/long/train or --include-speech).")

    cands = []
    for row in files.itertuples():
        t, p = scorer.score_file(row.path, cfg.eval.hop_s)
        s = smooth_scores(p, cfg.eval.smooth_k)
        order = np.argsort(-s)
        taken: list[float] = []
        for i in order:
            if s[i] < args.min_score:
                break
            if all(abs(t[i] - u) >= args.nms_s for u in taken):
                taken.append(float(t[i]))
                cands.append((float(s[i]), row.path, float(t[i])))
    cands.sort(reverse=True)
    cands = cands[: args.top_k]

    out_dir = Path(cfg.paths.mined) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    win = cfg.audio.clip_seconds
    rows = []
    for k, (score, path, t_end) in enumerate(cands):
        start = max(0.0, t_end - win)
        x = load_audio(path, cfg.audio.sample_rate, offset_s=start, duration_s=win)
        dst = out_dir / f"mined_{k:05d}.wav"
        save_audio(dst, x, cfg.audio.sample_rate)
        rows.append({"file": dst.name, "score": score, "source": path, "start_s": start})
    pd.DataFrame(rows).to_csv(out_dir / "mined_index.csv", index=False)
    hours = files.duration.sum() / 3600
    print(f"mined {len(rows)} windows (score >= {args.min_score}) from {hours:.2f} h -> {out_dir}")
    print("Tip: listen to a few — if real wake phrases slipped into 'negative' audio, delete them before retraining.")


if __name__ == "__main__":
    main()
