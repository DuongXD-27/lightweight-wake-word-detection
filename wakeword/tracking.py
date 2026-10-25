"""Optional MLflow experiment tracking (no-op when mlflow is missing or disabled)."""
from __future__ import annotations

import os
from pathlib import Path


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
    return out


class Tracker:
    def __init__(self, cfg, run_name: str):
        self.enabled = bool(cfg.get_path("tracking.enabled", False))
        self.mlflow = None
        if not self.enabled:
            return
        try:
            import mlflow
        except ImportError:
            print("[tracking] mlflow not installed -> tracking disabled")
            self.enabled = False
            return
        self.mlflow = mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.get_path("tracking.uri", "file:./mlruns")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.get_path("tracking.experiment", "wakeword"))
        mlflow.start_run(run_name=run_name)
        params = _flatten(cfg.to_dict())
        for i in range(0, len(params), 90):  # mlflow limits params per call
            mlflow.log_params(dict(list(params.items())[i:i + 90]))

    def log_metrics(self, metrics: dict, step: int | None = None):
        if self.enabled:
            clean = {k: float(v) for k, v in _flatten(metrics).items()
                     if isinstance(v, (int, float)) and v == v}
            self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path):
        if self.enabled and Path(path).exists():
            self.mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict):
        if self.enabled:
            self.mlflow.set_tags(tags)

    def end(self):
        if self.enabled:
            self.mlflow.end_run()
