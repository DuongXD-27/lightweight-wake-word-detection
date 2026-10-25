#!/usr/bin/env python
"""Download public corpora used as negatives / augmentation material.

    python scripts/download_data.py esc50 rirs          # noise + room impulse responses
    python scripts/download_data.py musan               # 11 GB: noise/music (+English speech)
    python scripts/download_data.py hf --repo AILAB-VNUHCM/vivos --name vivos   # Vietnamese speech (HF hub)

Everything lands in the layout expected by scripts/build_manifest.py:
    esc50  -> data/raw/noise/esc50/*.wav
    musan  -> data/raw/noise/musan/{noise,music}/... and data/raw/speech/musan_en/<spk>/...
    rirs   -> data/rir/...
    hf     -> data/raw/speech/<name>/<speaker>/*.wav

Check each dataset's licence before redistributing anything derived from it.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

URLS = {
    "esc50": "https://github.com/karoldvl/ESC-50/archive/master.zip",
    "rirs": "https://www.openslr.org/resources/28/rirs_noises.zip",
    "musan": "https://www.openslr.org/resources/17/musan.tar.gz",
}
CACHE = Path("data/_downloads")


def fetch(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / url.rsplit("/", 1)[-1]
    if dst.exists():
        return dst
    print(f"downloading {url}")

    def hook(n, bs, total):
        if total > 0 and n % 200 == 0:
            print(f"\r  {min(100, n * bs * 100 // total)}%", end="", flush=True)

    urllib.request.urlretrieve(url, dst, hook)
    print()
    return dst


def esc50():
    z = fetch(URLS["esc50"])
    out = Path("data/raw/noise/esc50")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith(".wav") and "/audio/" in name:
                (out / Path(name).name).write_bytes(zf.read(name))
    print(f"ESC-50 -> {out}")


def rirs():
    z = fetch(URLS["rirs"])
    out = Path("data/rir")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith(".wav") and ("simulated_rirs" in name or "real_rirs" in name) and "noise" not in Path(name).name:
                dst = out / name.replace("/", "_")
                dst.write_bytes(zf.read(name))
    print(f"RIRs -> {out}")


def musan():
    t = fetch(URLS["musan"])
    with tarfile.open(t) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".wav"):
                continue
            parts = Path(m.name).parts  # musan/<type>/<sub>/<file>
            if len(parts) < 3:
                continue
            kind = parts[1]
            if kind in ("noise", "music"):
                dst = Path("data/raw/noise/musan") / kind / Path(m.name).name
            else:  # English speech: still a useful "random speech" negative
                spk = Path(m.name).stem.rsplit("-", 1)[0]
                dst = Path("data/raw/speech/musan_en") / spk / Path(m.name).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as f:
                dst.write_bytes(f.read())
    print("MUSAN extracted")


def hf(repo: str, name: str, max_files: int | None):
    """Generic Hugging Face dataset import (tar archives or parquet with an audio column)."""
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo, repo_type="dataset", cache_dir=str(CACHE / "hf")))
    out = Path("data/raw/speech") / name
    n = 0

    def speaker_of(fname: str, fallback: str = "unknown") -> str:
        stem = Path(fname).stem
        return stem.split("_")[0] if "_" in stem else fallback

    for arc in list(local.rglob("*.tar.gz")) + list(local.rglob("*.tar")):
        with tarfile.open(arc) as tf:
            for m in tf.getmembers():
                if m.name.lower().endswith((".wav", ".flac", ".mp3")):
                    dst = out / speaker_of(m.name, Path(m.name).parent.name) / Path(m.name).name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(m) as f:
                        dst.write_bytes(f.read())
                    n += 1
                    if max_files and n >= max_files:
                        break
    if n == 0:
        import pandas as pd

        for pq in local.rglob("*.parquet"):
            df = pd.read_parquet(pq)
            col = next((c for c in df.columns if c.lower() == "audio"), None)
            if col is None:
                continue
            for i, row in df.iterrows():
                a = row[col]
                path = a.get("path") or f"{pq.stem}_{i}.wav"
                spk = str(row["speaker_id"]) if "speaker_id" in df.columns else speaker_of(path)
                data, sr = sf.read(io.BytesIO(a["bytes"]))
                dst = out / spk / (Path(path).stem + ".wav")
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, np.asarray(data, dtype=np.float32), sr)
                n += 1
                if max_files and n >= max_files:
                    break
    print(f"{repo}: {n} files -> {out}" if n else f"!! no audio found in {repo}; import it manually into {out}/<speaker>/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["esc50", "rirs", "musan", "hf"])
    ap.add_argument("--repo", default="AILAB-VNUHCM/vivos")
    ap.add_argument("--name", default="vivos")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--clean-cache", action="store_true")
    args = ap.parse_args()
    for w in args.what:
        try:
            if w == "hf":
                hf(args.repo, args.name, args.max_files)
            else:
                globals()[w]()
        except Exception as e:
            print(f"!! {w} failed: {e}\n   Download manually (see README > Data) and place files in the layout above.")
    if args.clean_cache:
        shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
