#!/usr/bin/env python
"""Gradio data-collection app: guides a speaker through the prompt list and saves each take in the
layout used by the training pipeline. Share it (``--share``) to collect unseen speakers for the test set.

    python apps/collect_app.py --out data/raw/real --share

Saved files:
    data/raw/real/<speaker>/<category>/<prompt_id>__<env>__t<k>.wav
    data/parts/<speaker>/part_head__*.wav, part_tail__*.wav     (for scripts/compose_pauses.py)
    data/raw/real/<speaker>/meta.csv                             (speaker + recording conditions)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np

from wakeword.audio import SAMPLE_RATE, resample, save_audio
from wakeword.phrases import HARD_CATEGORIES, Prompt, default_prompts

PARTS = [Prompt("part_head", "Alo GPT", "hard_partial", "Chỉ nói 'Alo GPT' rồi dừng", 3),
         Prompt("part_tail", "à em", "hard_partial", "Chỉ nói 'à em'", 3)]


def build_queue(mode: str) -> list[tuple[Prompt, int]]:
    prompts = default_prompts() + PARTS
    if mode.startswith("Nhanh"):
        picked, seen = [], set()
        for p in prompts:
            if p.category == "positive":
                picked.append(Prompt(p.prompt_id, p.text, p.category, p.hint, 2))
            elif p.category in HARD_CATEGORIES and p.category not in seen:
                seen.add(p.category)
                picked.append(Prompt(p.prompt_id, p.text, p.category, p.hint, 1))
            elif p.category in ("positive_command", "positive_pause") and p.category not in seen:
                seen.add(p.category)
                picked.append(Prompt(p.prompt_id, p.text, p.category, p.hint, 2))
        prompts = picked
    return [(p, k) for p in prompts for k in range(p.repeats)]


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip().lower())
    return s.strip("_")


def render(queue, idx):
    if not queue:
        return "<h3>Nhập thông tin rồi bấm <b>Bắt đầu</b></h3>", ""
    if idx >= len(queue):
        return "<h2>✅ Xong! Cảm ơn bạn đã đóng góp dữ liệu.</h2>", f"{len(queue)}/{len(queue)}"
    p, k = queue[idx]
    hint = f"<br><i>{p.hint}</i>" if p.hint else ""
    html = (f"<div style='font-size:30px;font-weight:600;padding:18px;border:2px solid #888;border-radius:12px'>"
            f"{p.text}</div><div style='margin-top:6px'>Loại: <code>{p.category}</code> · lần {k + 1}/{p.repeats}{hint}</div>")
    return html, f"{idx + 1}/{len(queue)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/real")
    ap.add_argument("--parts", default="data/parts")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    out_root, parts_root = Path(args.out), Path(args.parts)

    def start(name, external, gender, region, device, env, mode, consent):
        if not consent:
            raise gr.Error("Cần đồng ý cho phép sử dụng bản ghi cho mục đích học tập.")
        spk = slug(name)
        if not spk:
            raise gr.Error("Nhập tên/ID người nói.")
        if external and not spk.startswith("ext_"):
            spk = "ext_" + spk
        meta = out_root / spk / "meta.csv"
        meta.parent.mkdir(parents=True, exist_ok=True)
        new = not meta.exists()
        with open(meta, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "speaker", "gender", "region", "device", "env", "mode"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), spk, gender, region, device, env, mode])
        queue = build_queue(mode)
        info = {"speaker": spk, "env": slug(env)}
        html, prog = render(queue, 0)
        return queue, 0, info, html, prog, None, f"Người nói: **{spk}** — {len(queue)} câu"

    def save(audio, queue, idx, info):
        if not queue or idx >= len(queue):
            return idx, *render(queue, idx), None, "Chưa bắt đầu hoặc đã xong."
        if audio is None:
            raise gr.Error("Hãy thu âm trước khi lưu.")
        sr, data = audio
        x = data.astype(np.float32)
        if data.dtype.kind == "i":
            x /= np.iinfo(data.dtype).max
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = resample(x, sr, SAMPLE_RATE)
        if len(x) < 0.3 * SAMPLE_RATE or np.abs(x).max() < 1e-3:
            raise gr.Error("Bản ghi quá ngắn hoặc im lặng — thu lại nhé.")
        p, k = queue[idx]
        stamp = dt.datetime.now().strftime("%H%M%S")
        if p.prompt_id.startswith("part_"):
            dst = parts_root / info["speaker"] / f"{p.prompt_id}__{info['env']}__t{k}_{stamp}.wav"
        else:
            dst = out_root / info["speaker"] / p.category / f"{p.prompt_id}__{info['env']}__t{k}_{stamp}.wav"
        save_audio(dst, x, SAMPLE_RATE)
        idx += 1
        return idx, *render(queue, idx), None, f"Đã lưu `{dst.as_posix()}` ({len(x) / SAMPLE_RATE:.1f}s)"

    def skip(queue, idx):
        idx = min(idx + 1, len(queue))
        return idx, *render(queue, idx), None, "Đã bỏ qua."

    def back(queue, idx):
        idx = max(0, idx - 1)
        return idx, *render(queue, idx), None, "Quay lại câu trước (bản ghi cũ vẫn giữ)."

    with gr.Blocks(title="Thu âm dữ liệu wake word") as demo:
        gr.Markdown("# 🎤 Thu âm dữ liệu cho wake word “Alo GPT à em”\n"
                    "Mỗi câu: bấm record → nói → stop → **Lưu & tiếp**. Nói tự nhiên, thay đổi tốc độ/âm lượng theo gợi ý.")
        queue, idx, info = gr.State([]), gr.State(0), gr.State({})
        with gr.Row():
            name = gr.Textbox(label="Tên / ID người nói (không dấu)", placeholder="vd: duong")
            external = gr.Checkbox(label="Người ngoài nhóm (chỉ dùng để test)")
            gender = gr.Dropdown(["Nam", "Nữ", "Khác"], label="Giới tính")
            region = gr.Dropdown(["Bắc", "Trung", "Nam"], label="Giọng vùng miền")
        with gr.Row():
            device = gr.Dropdown(["Laptop mic", "Điện thoại", "Tai nghe", "Mic rời"], label="Thiết bị",
                                 allow_custom_value=True)
            env = gr.Dropdown(["clean", "noisy_fan", "noisy_tv", "noisy_people", "far_field", "reverb_room"],
                              value="clean", label="Môi trường", allow_custom_value=True)
            mode = gr.Radio(["Đầy đủ (team, ~25 phút)", "Nhanh (người ngoài, ~5 phút)"],
                            value="Nhanh (người ngoài, ~5 phút)", label="Chế độ")
        consent = gr.Checkbox(label="Tôi đồng ý cho nhóm dùng bản ghi này cho mục đích học tập/nghiên cứu.")
        start_btn = gr.Button("Bắt đầu", variant="primary")
        who = gr.Markdown()
        prompt_html = gr.HTML(render([], 0)[0])
        progress = gr.Textbox(label="Tiến độ", interactive=False)
        rec = gr.Audio(sources=["microphone"], type="numpy", label="Thu âm")
        with gr.Row():
            back_btn = gr.Button("⬅ Quay lại")
            skip_btn = gr.Button("Bỏ qua")
            save_btn = gr.Button("💾 Lưu & tiếp", variant="primary")
        msg = gr.Markdown()

        start_btn.click(start, [name, external, gender, region, device, env, mode, consent],
                        [queue, idx, info, prompt_html, progress, rec, who])
        save_btn.click(save, [rec, queue, idx, info], [idx, prompt_html, progress, rec, msg])
        skip_btn.click(skip, [queue, idx], [idx, prompt_html, progress, rec, msg])
        back_btn.click(back, [queue, idx], [idx, prompt_html, progress, rec, msg])

    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
