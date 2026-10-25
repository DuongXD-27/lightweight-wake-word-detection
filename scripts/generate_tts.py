#!/usr/bin/env python
"""Synthetic speech for positives, hard negatives and length-matched negatives.

Back-ends (install with `pip install -r requirements-tts.txt`):
  * edge   - Microsoft Edge neural voices (vi-VN-HoaiMyNeural female, vi-VN-NamMinhNeural male); needs internet
  * mms    - facebook/mms-tts-vie (VITS, offline after first download); stochastic -> varied takes

Each prompt is rendered with several speaking rates / pitches. Output:
    data/raw/tts/<voice>/<category>/<prompt_id>__<variant>.wav
    data/tts_parts/<voice>/<part>__<variant>.wav    (building blocks for scripts/compose_pauses.py)

    python scripts/generate_tts.py --backend edge mms --variants 6
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wakeword.audio import SAMPLE_RATE, load_audio, save_audio, trim_silence
from wakeword.phrases import tts_prompts

EDGE_VOICES = {"edge_hoaimy": "vi-VN-HoaiMyNeural", "edge_namminh": "vi-VN-NamMinhNeural"}


def out_path(root: Path, parts_root: Path, voice: str, prompt, variant: str) -> Path:
    if prompt.prompt_id.startswith("part_"):
        return parts_root / voice / f"{prompt.prompt_id}__{variant}.wav"
    return root / voice / prompt.category / f"{prompt.prompt_id}__{variant}.wav"


def run_edge(prompts, n_variants: int, root: Path, parts_root: Path, rng):
    import edge_tts

    async def synth(text, voice, rate, pitch, dst):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3 = tmp.name
        await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(mp3)
        x = load_audio(mp3, SAMPLE_RATE)  # ffmpeg fallback decodes mp3
        save_audio(dst, trim_silence(x), SAMPLE_RATE)
        Path(mp3).unlink(missing_ok=True)

    async def main():
        for vname, voice in EDGE_VOICES.items():
            for p in prompts:
                for k in range(n_variants):
                    rate = f"{int(rng.integers(-25, 26)):+d}%"
                    pitch = f"{int(rng.integers(-15, 16)):+d}Hz"
                    dst = out_path(root, parts_root, vname, p, f"v{k}_r{rate}_p{pitch}".replace("%", "").replace("Hz", ""))
                    if dst.exists():
                        continue
                    try:
                        await synth(p.text, voice, rate, pitch, dst)
                    except Exception as e:
                        print(f"[edge] failed {p.prompt_id}: {e}")
            print(f"[edge] {vname} done")

    asyncio.run(main())


def run_mms(prompts, n_variants: int, root: Path, parts_root: Path, rng, device: str = "cpu"):
    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").to(device).eval()
    sr = model.config.sampling_rate
    for p in prompts:
        for k in range(n_variants):
            rate = float(rng.uniform(0.8, 1.25))
            model.speaking_rate = rate
            model.noise_scale = float(rng.uniform(0.5, 0.9))
            torch.manual_seed(int(rng.integers(1 << 30)))
            dst = out_path(root, parts_root, "mms_vie", p, f"v{k}_r{rate:.2f}")
            if dst.exists():
                continue
            with torch.no_grad():
                wav = model(**tok(p.text.lower(), return_tensors="pt").to(device)).waveform[0].cpu().numpy()
            from wakeword.audio import resample

            save_audio(dst, trim_silence(resample(wav.astype(np.float32), sr, SAMPLE_RATE)), SAMPLE_RATE)
    print("[mms] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", nargs="+", default=["edge"], choices=["edge", "mms"])
    ap.add_argument("--variants", type=int, default=6, help="takes per prompt per voice")
    ap.add_argument("--root", default="data/raw/tts")
    ap.add_argument("--parts-root", default="data/tts_parts")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these categories")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    prompts = [p for p in tts_prompts() if not args.only or p.category in args.only or p.prompt_id.startswith("part_")]
    root, parts = Path(args.root), Path(args.parts_root)
    if "edge" in args.backend:
        run_edge(prompts, args.variants, root, parts, rng)
    if "mms" in args.backend:
        run_mms(prompts, args.variants, root, parts, rng)


if __name__ == "__main__":
    main()
