#!/usr/bin/env python
"""Export a checkpoint to ONNX (FP32) and INT8 (static QDQ quantisation with calibration data).

The ONNX graph takes features (B, *in_shape) and returns logits; feature extraction stays in
numpy/torch (log-mel) or the frozen openWakeWord ONNX front-end.

    python scripts/export_onnx.py --checkpoint runs/tc_resnet/best.pt
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
import torch

from wakeword.data import WakeWordDataset, read_manifest
from wakeword.scoring import load_checkpoint


def bench(sess, x: np.ndarray, n: int = 200) -> float:
    name = sess.get_inputs()[0].name
    for _ in range(10):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib", type=int, default=300, help="calibration windows for INT8")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import onnxruntime as ort

    ckpt_path = Path(args.checkpoint)
    model, cfg, frontend, ckpt = load_checkpoint(ckpt_path)
    in_shape = tuple(ckpt["in_shape"])
    fp32 = ckpt_path.with_suffix(".onnx")
    int8 = ckpt_path.with_suffix(".int8.onnx")

    dummy = torch.zeros(1, *in_shape)
    torch.onnx.export(model, dummy, str(fp32), input_names=["features"], output_names=["logit"],
                      dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}}, opset_version=args.opset,
                      dynamo=False)

    # parity check
    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    x = torch.randn(4, *in_shape)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"features": x.numpy()})[0].reshape(-1)
    print(f"[onnx] fp32 max |diff| vs torch: {np.abs(ref - got).max():.2e}")

    # calibration data = real feature windows from the train split (no augmentation)
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    feats = []
    try:
        manifest = read_manifest(cfg.paths.manifest)
        ds = WakeWordDataset(manifest, cfg, "train", frontend, train=False)
        idx = np.random.default_rng(0).choice(len(ds), size=min(args.calib, len(ds)), replace=False)
        feats = [ds[int(i)][0].numpy()[None] for i in idx]
    except Exception as e:  # pragma: no cover
        print(f"[int8] manifest unavailable ({e}); calibrating on random features")
    if not feats:
        feats = [np.random.randn(1, *in_shape).astype(np.float32) for _ in range(32)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"features": f.astype(np.float32)} for f in feats])

        def get_next(self):
            return next(self.it, None)

    pre = ckpt_path.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(fp32), str(pre))
        src = pre
    except Exception:
        src = fp32
    quantize_static(str(src), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    if pre.exists():
        pre.unlink()

    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    xb = np.concatenate(feats[:32]).astype(np.float32)
    p32 = 1 / (1 + np.exp(-s32.run(None, {"features": xb})[0].reshape(-1)))
    p8 = 1 / (1 + np.exp(-s8.run(None, {"features": xb})[0].reshape(-1)))
    one = xb[:1]
    info = {
        "params": ckpt.get("params"), "macs": ckpt.get("macs"),
        "pytorch_state_kb": sum(v.numel() * v.element_size() for v in ckpt["model"].values()) / 1024,
        "onnx_fp32_kb": fp32.stat().st_size / 1024, "onnx_int8_kb": int8.stat().st_size / 1024,
        "fp32_ms_per_window": bench(s32, one), "int8_ms_per_window": bench(s8, one),
        "int8_vs_fp32_max_prob_diff": float(np.abs(p32 - p8).max()),
    }
    (ckpt_path.parent / "export_info.json").write_text(json.dumps(info, indent=2))
    for k, v in info.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"[onnx] wrote {fp32} and {int8}")


if __name__ == "__main__":
    main()
