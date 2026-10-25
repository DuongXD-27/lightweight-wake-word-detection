#!/usr/bin/env python
"""Run the whole experiment matrix (cross-platform replacement for a Makefile).

    python scripts/run_experiments.py                   # everything
    python scripts/run_experiments.py --only models     # approaches A/B/C + CRNN
    python scripts/run_experiments.py --only ablation   # data ablations a0..a4 (incl. hard-negative mining)
    python scripts/run_experiments.py --extra --set tracking.enabled=true

Each experiment: train -> evaluate (PyTorch) -> export ONNX/INT8 -> evaluate INT8 (log-mel models).
Finally writes runs/comparison.md for the report.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MODELS = ["configs/ds_cnn.yaml", "configs/tc_resnet.yaml", "configs/crnn.yaml", "configs/oww_head.yaml"]
ABLATION = ["configs/ablation/a0_real_only.yaml", "configs/ablation/a1_plus_tts.yaml",
            "configs/ablation/a2_plus_aug.yaml", "configs/ablation/a3_plus_hardneg.yaml"]


def sh(*args):
    print("\n$", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True)


def experiment_dir(cfg_path: str, overrides: list[str]) -> Path:
    sys.path.insert(0, str(ROOT))
    from wakeword.config import load_config

    c = load_config(ROOT / cfg_path, overrides)
    return Path(c.paths.runs) / c.experiment, c


def run_one(cfg_path: str, overrides: list[str], skip_train: bool, export: bool = True):
    out, c = experiment_dir(cfg_path, overrides)
    if not skip_train or not (ROOT / out / "best.pt").exists():
        sh("scripts/train.py", "-c", cfg_path, "--set", *overrides)
    ckpt = out / "best.pt"
    sh("scripts/evaluate.py", "--checkpoint", ckpt)
    if export and c.features.type == "logmel":
        sh("scripts/export_onnx.py", "--checkpoint", ckpt)
        sh("scripts/evaluate.py", "--checkpoint", ckpt, "--onnx", out / "best.int8.onnx", "--tag", "int8")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["models", "ablation"], default=None)
    ap.add_argument("--skip-train", action="store_true", help="reuse existing best.pt when present")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every experiment")
    args = ap.parse_args()
    ov = args.set

    sh("scripts/build_manifest.py", "-c", "configs/base.yaml", "--no-mined", "--set", *ov)
    if args.only in (None, "models"):
        for cfg_path in MODELS:
            run_one(cfg_path, ov, args.skip_train)
    if args.only in (None, "ablation"):
        v1 = None
        for cfg_path in ABLATION:
            v1 = run_one(cfg_path, ov, args.skip_train, export=False)
        # V1 -> mine hard negatives on train-split long audio -> V2
        sh("scripts/mine_hard_negatives.py", "--checkpoint", v1, "--tag", "v1")
        sh("scripts/build_manifest.py", "-c", "configs/base.yaml", "--set", *ov)
        run_one("configs/ablation/a4_plus_mined.yaml", ov, args.skip_train, export=False)
    _, c = experiment_dir("configs/base.yaml", ov)
    sh("scripts/compare_runs.py", "--runs", c.paths.runs, "--out", Path(c.paths.runs) / "comparison.md")


if __name__ == "__main__":
    main()
