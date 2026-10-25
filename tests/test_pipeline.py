"""End-to-end smoke test on synthetic data: data -> manifest -> train -> evaluate -> export -> mine."""
import json
import subprocess
import sys

import pytest
from conftest import ROOT

PY = sys.executable


def run(args, cwd):
    r = subprocess.run([PY, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.slow
def test_end_to_end(tmp_path):
    s = lambda p: str(ROOT / p)  # noqa: E731
    run([s("scripts/make_synthetic_data.py"), "--out", "data_synth", "--speakers", "5", "--per-speaker", "4",
         "--long-minutes", "0.5"], tmp_path)
    run([s("scripts/build_manifest.py"), "-c", s("configs/smoke.yaml")], tmp_path)
    run([s("scripts/train.py"), "-c", s("configs/smoke.yaml"), "--set", "train.steps_per_epoch=3"], tmp_path)
    ckpt = tmp_path / "runs_smoke" / "smoke" / "best.pt"
    assert ckpt.exists()
    run([s("scripts/evaluate.py"), "--checkpoint", str(ckpt)], tmp_path)
    rep = json.loads((ckpt.parent / "eval_test_torch" / "report.json").read_text())
    assert "det" in rep and rep["long_audio_hours"] > 0
    run([s("scripts/export_onnx.py"), "--checkpoint", str(ckpt), "--calib", "16"], tmp_path)
    assert (ckpt.parent / "best.int8.onnx").exists()
    run([s("scripts/evaluate.py"), "--checkpoint", str(ckpt), "--onnx", str(ckpt.parent / "best.int8.onnx"),
         "--tag", "int8"], tmp_path)
    run([s("scripts/mine_hard_negatives.py"), "--checkpoint", str(ckpt), "--min-score", "0.0", "--top-k", "5"],
        tmp_path)
    run([s("scripts/compare_runs.py"), "--runs", "runs_smoke"], tmp_path)
