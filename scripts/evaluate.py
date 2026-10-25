#!/usr/bin/env python
"""Full evaluation of a trained model.

1. clip-level metrics on the test split (AUC, EER, FRR/FAR at the validation threshold) + per category
2. streaming FRR on test positives (sliding window, smoothing, refractory) for a sweep of thresholds
3. false activations per hour on long negative audio (data/raw/long/<split>)
4. DET curve (FRR vs FA/h), operating points for FA/h budgets, challenge-set trigger rates
5. efficiency: params, MACs, CPU latency per window

    python scripts/evaluate.py --checkpoint runs/tc_resnet/best.pt
    python scripts/evaluate.py --checkpoint runs/tc_resnet/best.pt --onnx runs/tc_resnet/best.int8.onnx --tag int8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from wakeword.data import WakeWordDataset, read_manifest
from wakeword.metrics import clip_metrics, det_curve, operating_point, per_category_rates
from wakeword.scoring import Scorer
from wakeword.streaming import count_events_sweep, smooth_scores


def latency_ms(scorer: Scorer, n_runs: int = 50) -> dict:
    torch.set_num_threads(1)
    x = np.random.randn(1, scorer.n).astype(np.float32) * 0.1
    feats = scorer.frontend(torch.from_numpy(x))
    for _ in range(5):
        scorer.model_fn(feats)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        feats = scorer.frontend(torch.from_numpy(x))
    t_feat = (time.perf_counter() - t0) / n_runs * 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        scorer.model_fn(feats)
    t_model = (time.perf_counter() - t0) / n_runs * 1000
    return {"feature_ms": t_feat, "model_ms": t_model, "total_ms": t_feat + t_model}


def plot_det(det: list[dict], path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa = np.array([d["fa_per_hour"] for d in det])
    frr = np.array([d["frr"] for d in det])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.maximum(fa, 1e-2), frr * 100, marker=".")
    ax.set_xscale("log")
    ax.set_xlabel("False activations per hour (log)")
    ax.set_ylabel("False rejection rate (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", default=None, help="score with this ONNX model instead of PyTorch")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-long-hours", type=float, default=None, help="cap long audio (for quick runs)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = Scorer.from_checkpoint(args.checkpoint, args.device)
    cfg, ckpt = base.cfg, base.ckpt
    scorer = Scorer.from_onnx(args.onnx, cfg) if args.onnx else base
    e = cfg.eval
    tag = args.tag or ("onnx" if args.onnx else "torch")
    out_dir = Path(args.checkpoint).parent / f"eval_{args.split}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest or cfg.paths.manifest)
    thr0 = float(ckpt.get("threshold", 0.5))
    report: dict = {"checkpoint": str(args.checkpoint), "backend": tag, "val_threshold": thr0,
                    "model": cfg.model.name, "features": cfg.features.type,
                    "params": ckpt.get("params"), "macs": ckpt.get("macs")}

    # ---------------------------------------------------------------- 1. clip level
    ds = WakeWordDataset(manifest, cfg, args.split, scorer.frontend, train=False, include_challenge=True)
    wavs = np.stack([ds.get_waveform(i) for i in range(len(ds))]) if len(ds) else np.zeros((0, scorer.n))
    probs = scorer.score_windows(wavs) if len(ds) else np.zeros(0)
    df = ds.df.assign(prob=probs)
    df.to_csv(out_dir / "clip_scores.csv", index=False)
    main_df = df[df.label >= 0]
    report["clip"] = clip_metrics(main_df.label.values, main_df.prob.values, thr0)
    report["clip_real_only"] = clip_metrics(main_df[main_df.source != "tts"].label.values,
                                            main_df[main_df.source != "tts"].prob.values, thr0)
    report["clip_per_category"] = per_category_rates(df.category.values, df.prob.values, thr0)
    real_pos = df[(df.label == 1) & (df.source == "real")]
    report["clip_frr_per_speaker"] = {s: float((g.prob < thr0).mean()) for s, g in real_pos.groupby("speaker")}

    # ---------------------------------------------------------------- 2./3. streaming
    thresholds = np.linspace(0.01, 0.99, e.n_thresholds)
    # recorded clips: positives, hard negatives and challenge items (each padded with silence)
    stream_items = df[df.source.isin(["real", "tts"]) & ((df.label != 0) | df.category.str.startswith("hard_"))]
    detected = {}
    for row in stream_items.itertuples():
        t, p = scorer.score_file(row.path, e.hop_s, e.pad_before_s, e.pad_after_s)
        s = smooth_scores(p, e.smooth_k)
        detected[row.path] = np.array([s.max() >= th for th in thresholds]) if len(s) else np.zeros(len(thresholds), bool)
    stream_items = stream_items.assign(_det=[detected[p] for p in stream_items.path])

    def rate(sub):
        return np.stack(sub._det.values).mean(axis=0) if len(sub) else np.full(len(thresholds), np.nan)

    pos_real = stream_items[(stream_items.label == 1) & (stream_items.source == "real")]
    pos_all = stream_items[stream_items.label == 1]
    frr_curve = 1 - rate(pos_real if len(pos_real) else pos_all)

    long_df = manifest[(manifest.source == "long") & (manifest.split == e.long_split)]
    total_events = np.zeros(len(thresholds))
    hours = 0.0
    for row in long_df.itertuples():
        if args.max_long_hours and hours >= args.max_long_hours:
            break
        t, p = scorer.score_file(row.path, e.hop_s)
        s = smooth_scores(p, e.smooth_k)
        total_events += count_events_sweep(t, s, thresholds, e.refractory_s)
        hours += row.duration / 3600
    fa_curve = total_events / hours if hours > 0 else np.full(len(thresholds), np.nan)
    det = det_curve(frr_curve, fa_curve, thresholds)
    report["long_audio_hours"] = hours
    report["det"] = det
    report["operating_points"] = {str(b): operating_point(det, b) for b in e.fa_budgets} if hours > 0 else {}
    i0 = int(np.argmin(np.abs(thresholds - thr0)))
    report["stream_at_val_threshold"] = {
        "threshold": float(thresholds[i0]), "frr_real": float(frr_curve[i0]),
        "fa_per_hour": float(fa_curve[i0]) if hours > 0 else None,
        "per_category_trigger_rate": {c: float(rate(g)[i0]) for c, g in stream_items.groupby("category")},
    }

    # ---------------------------------------------------------------- 5. efficiency
    report["latency"] = latency_ms(scorer)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame(det).to_csv(out_dir / "det.csv", index=False)
    if hours > 0:
        plot_det(det, out_dir / "det.png", f"{cfg.experiment} ({tag})")

    # ---------------------------------------------------------------- summary
    c = report["clip"]
    print(f"\n== {cfg.experiment} [{tag}] on {args.split} ==")
    print(f"clip: AUC={c.get('auc', float('nan')):.4f} EER={c.get('eer', float('nan')):.4f} "
          f"FRR={c['frr']:.4f} FAR={c['far']:.4f} @thr={thr0:.3f}")
    sv = report["stream_at_val_threshold"]
    print(f"stream @thr={sv['threshold']:.2f}: FRR(real)={sv['frr_real']:.4f} FA/h={sv['fa_per_hour']} "
          f"(long audio {hours:.2f} h)")
    for b, op in report["operating_points"].items():
        if op:
            print(f"  FA/h <= {b}: FRR={op['frr']:.4f} @thr={op['threshold']:.2f}")
    print("per-category stream trigger rate:", {k: round(v, 3) for k, v in sv["per_category_trigger_rate"].items()})
    lat = report["latency"]
    print(f"latency/window: feat {lat['feature_ms']:.2f} ms + model {lat['model_ms']:.2f} ms (1 CPU thread)")
    print(f"report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
