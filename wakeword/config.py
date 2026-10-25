"""Tiny YAML config system with `_base_` inheritance and dotted CLI overrides.

Example:
    cfg = load_config("configs/tc_resnet.yaml", overrides=["train.epochs=5", "model.width=1.5"])
    cfg.train.epochs  # -> 5
"""
from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        def conv(x):
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, list):
                return [conv(v) for v in x]
            return x

        return conv(self)


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            # a section that switches its `name` (e.g. model: ds_cnn -> tc_resnet) is replaced, not merged
            if "name" in v and "name" in out[k] and v["name"] != out[k]["name"]:
                out[k] = copy.deepcopy(v)
            else:
                out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, _load_yaml((path.parent / b).resolve()))
    return deep_merge(merged, data)


def parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like key.sub=value, got: {item}")
    key, raw = item.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Cfg:
    data = _load_yaml(Path(path).resolve())
    for item in overrides or []:
        keys, value = parse_override(item)
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return _wrap(data)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plain = cfg.to_dict() if isinstance(cfg, Cfg) else cfg
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plain, f, allow_unicode=True, sort_keys=False)
