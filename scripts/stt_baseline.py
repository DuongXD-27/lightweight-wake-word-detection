#!/usr/bin/env python
"""Approach D - "just use Speech-to-Text": Whisper/PhoWhisper + text matching as an always-on trigger.

Quantifies the proposal's argument: compute (params, real-time factor), FRR, FAR on hard negatives
and false activations per hour, compared with the small keyword-spotting models.

    pip install -r requirements-stt.txt
    python scripts/stt_baseline.py --model vinai/PhoWhisper-tiny --manifest data/manifests/manifest.csv --max-long-hours 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from wakeword.audio import SAMPLE_RATE, audio_duration, load_audio
from wakeword.metrics import per_category_rates

GPT = r"(g ?p ?t|gi ?pi ?ti|chi ?pi ?ti|ji ?pi ?ti|go ?pe ?te|gi ?bi ?ti|z ?p ?t)"
WAKE_RE = re.compile(rf"\ba ?lo\b[ ,.!]*{GPT}[ ,.!]*\ba\b[ ,.!]*\bem\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_wake(text: str) -> bool:
    return bool(WAKE_RE.search(normalize(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/PhoWhisper-tiny")
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk-s", type=float, default=4.0)
    ap.add_argument("--hop-s", type=float, default=2.0)
    ap.add_argument("--refractory-s", type=float, default=2.0)
    ap.add_argument("--max-long-hours", type=float, default=0.5)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/stt_baseline")
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=args.model, device=args.device,
                   torch_dtype=torch.float32)
    n_params = sum(p.numel() for p in asr.model.parameters())

    def transcribe(x: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        out = asr({"raw": x, "sampling_rate": SAMPLE_RATE},
                  generate_kwargs={"language": "vi", "task": "transcribe"} if "whisper" in args.model.lower() else {})
        return out["text"], time.perf_counter() - t0

    man = pd.read_csv(args.manifest)
    clips = man[(man.split == args.split) & man.source.isin(["real", "tts"])]
    if args.max_clips:
        clips = clips.sample(min(args.max_clips, len(clips)), random_state=0)
    texts, hits, secs, audio_s = [], [], 0.0, 0.0
    for row in clips.itertuples():
        x = load_audio(row.path)
        txt, dt = transcribe(x)
        texts.append(txt)
        hits.append(is_wake(txt))
        secs += dt
        audio_s += len(x) / SAMPLE_RATE
    clips = clips.assign(text=texts, triggered=hits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips.to_csv(out / "clip_transcripts.csv", index=False)

    pos = clips[clips.label == 1]
    neg = clips[clips.label == 0]
    report = {
        "model": args.model, "params": int(n_params),
        "clip_frr": float(1 - pos.triggered.mean()) if len(pos) else None,
        "clip_far": float(neg.triggered.mean()) if len(neg) else None,
        "per_category_trigger_rate": per_category_rates(clips.category.values, clips.triggered.astype(float).values, 0.5),
        "rtf_clips": secs / max(audio_s, 1e-9),
    }

    long_df = man[(man.source == "long") & (man.split == args.split)]
    hours, events, t_long = 0.0, 0, 0.0
    for row in long_df.itertuples():
        dur = audio_duration(row.path)
        t, last = 0.0, -1e9
        while t + args.chunk_s <= dur and hours + t / 3600 < args.max_long_hours:
            x = load_audio(row.path, offset_s=t, duration_s=args.chunk_s)
            txt, dt = transcribe(x)
            t_long += dt
            if is_wake(txt) and t - last >= args.refractory_s:
                events += 1
                last = t
            t += args.hop_s
        hours += t / 3600
        if hours >= args.max_long_hours:
            break
    report.update(long_audio_hours=hours, fa_per_hour=events / hours if hours else None,
                  rtf_streaming=t_long / max(hours * 3600, 1e-9),
                  note="always-on STT must transcribe every chunk; RTF_streaming > chunk overlap factor means it "
                       "cannot keep up in real time on this CPU")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "per_category_trigger_rate"}, indent=2))


if __name__ == "__main__":
    main()
