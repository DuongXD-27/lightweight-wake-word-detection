#!/usr/bin/env python
"""Train a wake-word model.

    python scripts/train.py -c configs/tc_resnet.yaml [--set train.epochs=10 tracking.enabled=true]

Outputs in runs/<experiment>/: best.pt, last.pt, config.yaml, history.csv, train_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wakeword.augment import SpecAugment
from wakeword.config import load_config, save_config
from wakeword.data import WakeWordDataset, read_manifest
from wakeword.features import build_frontend, feature_shape
from wakeword.metrics import clip_metrics, threshold_for_far
from wakeword.models import build_model, count_macs, count_params
from wakeword.tracking import Tracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, labels, idx = [], [], []
    for x, y, i in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
        idx.append(i.numpy())
    if not probs:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idx)


def evaluate_val(model, loader, device, target_far: float) -> dict:
    probs, labels, _ = predict(model, loader, device)
    if len(probs) == 0:
        return {}
    thr = threshold_for_far(labels, probs, target_far)
    m = clip_metrics(labels, probs, thr)
    out = {"val_frr_at_far": m["frr"], "val_far": m["far"], "val_threshold": thr}
    if "auc" in m:
        out.update(val_auc=m["auc"], val_eer=m["eer"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=5")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = pick_device(cfg.train.device)
    out_dir = Path(cfg.paths.runs) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    manifest_path = Path(cfg.paths.manifest)
    if not manifest_path.exists():
        sys.exit(f"Manifest {manifest_path} not found. Run: python scripts/build_manifest.py -c {args.config}")
    manifest = read_manifest(manifest_path)

    frontend = build_frontend(cfg)
    in_shape = feature_shape(cfg, frontend)
    train_ds = WakeWordDataset(manifest, cfg, "train", frontend, train=True)
    val_ds = WakeWordDataset(manifest, cfg, "val", frontend, train=False, seed=cfg.seed)
    if len(train_ds) == 0:
        sys.exit("Empty training split - check data/raw and the manifest.")
    print(f"[data] train groups: {train_ds.group_counts()}  | val items: {len(val_ds)}")

    bs = cfg.train.batch_size
    nw = cfg.data.num_workers
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=train_ds.make_sampler(cfg.train.steps_per_epoch * bs),
                          num_workers=nw, pin_memory=device == "cuda", drop_last=True,
                          persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=nw)

    model = build_model(cfg, in_shape).to(device)
    n_params, macs = count_params(model), count_macs(model, in_shape)
    print(f"[model] {cfg.model.name}: {n_params:,} params, {macs / 1e6:.2f} M MACs / window, input {in_shape}")

    sa = cfg.aug.specaug
    specaug = SpecAugment(sa.freq_masks, sa.freq_width, sa.time_masks, sa.time_width) if (
        cfg.aug.enabled and sa.enabled) else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total = cfg.train.epochs * cfg.train.steps_per_epoch
    warm = cfg.train.warmup_epochs * cfg.train.steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    pos_weight = torch.tensor(cfg.train.pos_weight, device=device)
    ls = cfg.train.label_smoothing

    tracker = Tracker(cfg, cfg.experiment)
    tracker.set_tags({"model": cfg.model.name, "features": cfg.features.type})
    tracker.log_metrics({"params": n_params, "macs": macs})

    higher_better = cfg.train.select_metric == "val_auc"
    best_score, best_epoch, bad = (-np.inf if higher_better else np.inf), -1, 0
    best_key = (best_score, -np.inf if higher_better else np.inf)
    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        if specaug is not None:
            specaug.train()
        t0, losses = time.time(), []
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            if specaug is not None:
                x = specaug(x)
            logits = model(x)
            target = y * (1 - ls) + 0.5 * ls
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())

        val = evaluate_val(model, val_dl, device, cfg.train.target_val_far)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               "time_s": time.time() - t0, **val}
        history.append(row)
        tracker.log_metrics(row, step=epoch)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        score = val.get(cfg.train.select_metric, row["loss"])
        if math.isnan(score):  # e.g. no positives in val -> fall back to loss
            score = row["loss"]
        # ties (common for FRR on small val sets) are broken by AUC
        tie = val.get("val_auc", 0.0) if not higher_better else -row["loss"]
        key = (score, -tie) if not higher_better else (score, tie)
        improved = (key > best_key) if higher_better else (key < best_key)
        ckpt = {"cfg": cfg.to_dict(), "model": model.state_dict(), "in_shape": list(in_shape), "epoch": epoch,
                "threshold": float(val.get("val_threshold", 0.5)), "val": val, "params": n_params, "macs": macs}
        torch.save(ckpt, out_dir / "last.pt")
        if improved:
            best_score, best_key, best_epoch, bad = score, key, epoch, 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.train.early_stop_patience:
                print(f"[train] early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"experiment": cfg.experiment, "best_epoch": best_epoch, "best_" + cfg.train.select_metric: best_score,
               "params": n_params, "macs": macs, "in_shape": list(in_shape)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    tracker.log_artifact(out_dir / "history.csv")
    tracker.log_artifact(out_dir / "config.yaml")
    tracker.end()
    print(f"[train] done. best epoch {best_epoch} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
