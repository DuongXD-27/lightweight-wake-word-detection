#!/usr/bin/env python
"""Gradio demo: wake-word detection for "Alo GPT à em" + voice-assistant session state machine.

    python apps/demo_app.py --runs runs            # picks up runs/*/best.pt (and *.onnx)
    python apps/demo_app.py --checkpoint runs/tc_resnet/best.pt --share

Tabs
  1. Phân tích audio  - upload / record a clip, see the wake score over time, events and sessions
  2. Live mic         - real-time streaming detection; wake word opens a GPT session (mock by default)
  3. Model            - size, MACs, latency, validation metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np

from wakeword.audio import SAMPLE_RATE, load_audio, resample
from wakeword.scoring import Scorer, load_checkpoint
from wakeword.session import AssistantSession, State, make_backend, simulate
from wakeword.streaming import DetectorParams, StreamingDetector, detect_events, smooth_scores

_SCORERS: dict[str, Scorer] = {}


def find_models(runs: Path, extra: list[str]) -> list[str]:
    found = [str(p) for p in sorted(runs.glob("*/best.pt"))] if runs.exists() else []
    found += [str(p) for p in sorted(runs.glob("*/best*.onnx"))] if runs.exists() else []
    return list(dict.fromkeys([*extra, *found]))


def get_scorer(path: str) -> Scorer:
    if path not in _SCORERS:
        if path.endswith(".onnx"):
            ckpt = Path(path.replace(".int8", "")).with_suffix(".pt")
            _, cfg, _, meta = load_checkpoint(ckpt)
            s = Scorer.from_onnx(path, cfg)
            s.ckpt = meta
        else:
            s = Scorer.from_checkpoint(path)
        _SCORERS[path] = s
    return _SCORERS[path]


def to_float_mono(sr: int, data: np.ndarray) -> np.ndarray:
    x = np.asarray(data)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(data.dtype).max
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return resample(x, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else x


# ---------------------------------------------------------------------------- offline analysis
def analyze(audio_path, model_path, threshold, smooth_k, refractory, timeout):
    if not model_path:
        raise gr.Error("Chưa có model. Train trước (scripts/train.py) hoặc truyền --checkpoint.")
    if audio_path is None:
        raise gr.Error("Hãy upload hoặc thu một đoạn audio.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = get_scorer(model_path)
    x = load_audio(audio_path, SAMPLE_RATE)
    pad = np.zeros(int(0.5 * SAMPLE_RATE), np.float32)
    t0 = time.perf_counter()
    times, probs = scorer.score_array(np.concatenate([pad, x, pad]), hop_s=0.05)
    elapsed = time.perf_counter() - t0
    times = times - 0.5
    sm = smooth_scores(probs, int(smooth_k))
    events = detect_events(times, sm, float(threshold), float(refractory))
    dur = len(x) / SAMPLE_RATE
    hop = int(0.1 * SAMPLE_RATE)
    frames = x[: len(x) // hop * hop].reshape(-1, hop) if len(x) >= hop else x[None, :]
    activity = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12) > -40  # speech keeps session alive
    sessions = simulate(events, dur, timeout_s=float(timeout), activity=activity)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    tx = np.arange(len(x)) / SAMPLE_RATE
    a1.plot(tx, x, lw=0.4, color="#555")
    a1.set_ylabel("waveform")
    a2.plot(times, probs, lw=0.8, alpha=0.35, label="raw score")
    a2.plot(times, sm, lw=1.6, label=f"smoothed (k={int(smooth_k)})")
    a2.axhline(threshold, ls="--", color="crimson", lw=1, label=f"threshold {threshold:.2f}")
    for s, e, _ in sessions:
        for ax in (a1, a2):
            ax.axvspan(s, e, color="gold", alpha=0.18)
    for ev in events:
        for ax in (a1, a2):
            ax.axvline(ev, color="green", lw=1.5)
    a2.set_ylim(-0.02, 1.02)
    a2.set_xlim(0, max(dur, 0.1))
    a2.set_xlabel("time (s) — vạch xanh: wake event, nền vàng: phiên GPT đang mở")
    a2.set_ylabel("P(wake)")
    a2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    rows = [f"| {i + 1} | {t:.2f} s | {sm[np.argmin(np.abs(times - t))]:.3f} |" for i, t in enumerate(events)]
    table = "| # | thời điểm | score |\n|---|---|---|\n" + "\n".join(rows) if rows else "_Không phát hiện wake word._"
    sess_txt = "\n".join(f"- Phiên {i + 1}: {s:.2f}s → {e:.2f}s ({why})" for i, (s, e, why) in enumerate(sessions))
    summary = (f"**{len(events)} wake event(s)** · max score {sm.max() if len(sm) else 0:.3f} · "
               f"audio {dur:.1f}s scored in {elapsed * 1000:.0f} ms ({len(times)} windows)\n\n{table}\n\n"
               + (f"**Phiên hội thoại (timeout {timeout:.0f}s):**\n{sess_txt}" if sessions else ""))
    return fig, summary


# ------------------------------------------------------------------------------- live streaming
def new_live_state():
    return {"det": None, "sess": None, "model": None, "t": 0.0, "scores": []}


def live_step(chunk, state, model_path, threshold, smooth_k, refractory, timeout):
    if chunk is None or not model_path:
        return state, "<b>Chưa có audio / model</b>", "", ""
    sr, data = chunk
    x = to_float_mono(sr, data)
    scorer = get_scorer(model_path)
    if state is None or state.get("model") != model_path or state.get("det") is None:
        state = new_live_state()
        state["model"] = model_path
        params = DetectorParams(float(threshold), int(smooth_k), float(refractory), hop_s=0.1)
        state["det"] = StreamingDetector(scorer.score_one, scorer.n, SAMPLE_RATE, params)
        state["sess"] = AssistantSession(timeout_s=float(timeout), backend=make_backend())
    det: StreamingDetector = state["det"]
    sess: AssistantSession = state["sess"]
    det.p.threshold, det.p.refractory_s, sess.timeout_s = float(threshold), float(refractory), float(timeout)
    events = det.push(x)
    state["t"] = det.t
    for ev in events:
        sess.on_wake(ev["time"])
    sess.on_audio(det.t, x, SAMPLE_RATE)
    state["scores"] = (state["scores"] + [det.last_smoothed])[-60:]

    if sess.state == State.SESSION:
        status = (f"<div style='padding:14px;border-radius:10px;background:#1b7f3b;color:white;font-size:20px'>"
                  f"🟢 SESSION — đang hội thoại với GPT · còn {sess.remaining(det.t):.1f}s</div>")
    else:
        status = ("<div style='padding:14px;border-radius:10px;background:#444;color:white;font-size:20px'>"
                  "⚪ IDLE — đang chờ “Alo GPT à em”</div>")
    bar = "".join("█" if s >= threshold else ("▆" if s > threshold * 0.6 else "▁") for s in state["scores"])
    meter = f"score {det.last_smoothed:.3f} (thr {threshold:.2f})  `{bar}`"
    return state, status, meter, "\n".join(sess.log[-12:])


def reset_live():
    return new_live_state(), "", "", ""


# -------------------------------------------------------------------------------------- model tab
def model_info(model_path):
    if not model_path:
        return "_Chưa có model._"
    s = get_scorer(model_path)
    ck = getattr(s, "ckpt", {}) or {}
    cfg = s.cfg
    info = {
        "model": cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model),
        "features": cfg.features.type, "window_s": cfg.audio.clip_seconds,
        "params": ck.get("params"), "MACs/window": ck.get("macs"), "epoch": ck.get("epoch"),
        "val": ck.get("val"), "threshold (val)": ck.get("threshold"),
    }
    t0 = time.perf_counter()
    for _ in range(20):
        s.score_one(np.zeros(s.n, np.float32))
    info["latency_ms/window (feat+model)"] = round((time.perf_counter() - t0) / 20 * 1000, 2)
    exp = Path(model_path).parent / "export_info.json"
    if exp.exists():
        info["export"] = json.loads(exp.read_text())
    return "```json\n" + json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n```"


def default_threshold(model_path):
    try:
        return round(float(get_scorer(model_path).ckpt.get("threshold", 0.5)), 2)
    except Exception:
        return 0.5


# ------------------------------------------------------------------------------------------- UI
def build_ui(models: list[str]) -> gr.Blocks:
    with gr.Blocks(title="Alo GPT à em — Wake Word Demo") as demo:
        gr.Markdown("# 🎙️ Lightweight Wake Word Detection — “Alo GPT à em”\n"
                    "Mô hình nhỏ chạy liên tục trên thiết bị; chỉ khi nghe đúng câu kích hoạt mới mở phiên GPT. "
                    "Trong phiên có thể nói nhiều lượt mà không cần lặp lại wake word.")
        with gr.Row():
            model_dd = gr.Dropdown(models, value=models[0] if models else None, label="Model checkpoint",
                                   allow_custom_value=True, scale=3)
            thr = gr.Slider(0.05, 0.99, value=default_threshold(models[0]) if models else 0.5, step=0.01,
                            label="Threshold", scale=2)
            smooth = gr.Slider(1, 8, value=3, step=1, label="Smoothing (windows)", scale=1)
            refr = gr.Slider(0.5, 5, value=2.0, step=0.5, label="Refractory (s)", scale=1)
            tout = gr.Slider(3, 30, value=8, step=1, label="Session timeout (s)", scale=1)
        model_dd.change(default_threshold, model_dd, thr)

        with gr.Tab("Phân tích audio"):
            with gr.Row():
                audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
                run_btn = gr.Button("Phân tích", variant="primary")
            plot = gr.Plot(label="Wake score theo thời gian")
            summary = gr.Markdown()
            run_btn.click(analyze, [audio_in, model_dd, thr, smooth, refr, tout], [plot, summary])
            audio_in.change(analyze, [audio_in, model_dd, thr, smooth, refr, tout], [plot, summary])

        with gr.Tab("Live mic"):
            gr.Markdown("Bấm record và nói **“Alo GPT à em”**. Sau khi kích hoạt, nói tiếp các câu lệnh; "
                        "phiên tự đóng khi im lặng quá timeout.")
            live_state = gr.State(new_live_state())
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
            status = gr.HTML()
            meter = gr.Markdown()
            log = gr.Textbox(label="Session log", lines=10)
            reset = gr.Button("Reset")
            mic.stream(live_step, [mic, live_state, model_dd, thr, smooth, refr, tout],
                       [live_state, status, meter, log], stream_every=0.25, time_limit=600)
            reset.click(reset_live, None, [live_state, status, meter, log])

        with gr.Tab("Model"):
            info_btn = gr.Button("Xem thông tin model")
            info = gr.Markdown()
            info_btn.click(model_info, model_dd, info)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    models = find_models(Path(args.runs), args.checkpoint)
    if not models:
        print("!! no checkpoints found - the UI will start but needs a model path")
    build_ui(models).queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
