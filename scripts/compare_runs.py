#!/usr/bin/env python
"""Collect every runs/*/eval_*/report.json (+ STT baseline) into one comparison table for the report.

    python scripts/compare_runs.py --runs runs --out runs/comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", default="0.5", help="FA/h budget for the 'FRR @ budget' column")
    args = ap.parse_args()
    rows = []
    for rep in sorted(Path(args.runs).glob("*/eval_*/report.json")):
        r = json.loads(rep.read_text())
        exp = rep.parent.parent.name
        op = (r.get("operating_points") or {}).get(args.budget) or {}
        sv = r.get("stream_at_val_threshold", {})
        cats = sv.get("per_category_trigger_rate", {})
        hard = {k: v for k, v in cats.items() if k.startswith("hard_")}
        export = rep.parent.parent / "export_info.json"
        size_kb = None
        if export.exists():
            e = json.loads(export.read_text())
            size_kb = e["onnx_int8_kb"] if r.get("backend") == "int8" else e["onnx_fp32_kb"]
        rows.append({
            "experiment": exp, "backend": r.get("backend"), "model": r.get("model"), "features": r.get("features"),
            "params": r.get("params"), "MMACs": (r.get("macs") or 0) / 1e6, "size_kb": size_kb,
            "AUC": r["clip"].get("auc"), "EER": r["clip"].get("eer"),
            "FRR@val_thr": sv.get("frr_real"), "FA/h@val_thr": sv.get("fa_per_hour"),
            f"FRR@{args.budget}FA/h": op.get("frr"),
            "hard_neg_trigger(mean)": sum(hard.values()) / len(hard) if hard else None,
            "latency_ms": r.get("latency", {}).get("total_ms"),
        })
    stt = Path(args.runs) / "stt_baseline" / "report.json"
    if stt.exists():
        r = json.loads(stt.read_text())
        cats = r.get("per_category_trigger_rate", {})
        hard = [v["trigger_rate"] for k, v in cats.items() if k.startswith("hard_")]
        rows.append({"experiment": "stt_baseline", "backend": "torch", "model": r["model"], "features": "-",
                     "params": r["params"], "FRR@val_thr": r.get("clip_frr"), "FA/h@val_thr": r.get("fa_per_hour"),
                     "hard_neg_trigger(mean)": sum(hard) / len(hard) if hard else None,
                     "latency_ms": None, "RTF": r.get("rtf_streaming")})
    if not rows:
        print("no reports found")
        return
    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False, floatfmt=".4g")
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        df.to_csv(Path(args.out).with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
