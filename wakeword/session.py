"""Voice-assistant session state machine around the wake-word detector.

IDLE --(wake word)--> SESSION --(timeout without speech | user says goodbye)--> IDLE

Inside SESSION the user can speak several turns without repeating the wake word; every detected
speech turn refreshes the timeout. The GPT voice back-end is pluggable (``SessionBackend``); the
default ``MockBackend`` only logs, so the demo runs offline.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class State(str, Enum):
    IDLE = "IDLE"
    SESSION = "SESSION"


class SessionBackend:
    def open(self) -> str:
        return "session opened"

    def on_turn(self, audio: np.ndarray, sr: int) -> str:
        return ""

    def close(self, reason: str) -> str:
        return f"session closed ({reason})"


class MockBackend(SessionBackend):
    def open(self) -> str:
        return "🔔 Đã mở phiên hội thoại với GPT (mock). Bạn có thể nói nhiều lượt."

    def on_turn(self, audio: np.ndarray, sr: int) -> str:
        return f"🗣️ Nhận một lượt nói dài {len(audio) / sr:.1f}s (mock: sẽ gửi tới GPT voice)."

    def close(self, reason: str) -> str:
        return f"💤 Đóng phiên ({reason}). Quay lại chờ wake word."


class OpenAIBackend(MockBackend):
    """Optional: transcribe each turn with OpenAI and answer with a chat model (needs OPENAI_API_KEY).

    Kept intentionally small; replace with the Realtime API for true speech-to-speech.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # optional dependency

        self.client = OpenAI()
        self.model = model
        self.history = []

    def open(self) -> str:
        self.history = [{"role": "system", "content": "Bạn là trợ lý giọng nói tiếng Việt, trả lời ngắn gọn."}]
        return "🔔 Đã mở phiên hội thoại với GPT."

    def on_turn(self, audio: np.ndarray, sr: int) -> str:
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.name = "turn.wav"
        buf.seek(0)
        text = self.client.audio.transcriptions.create(model="whisper-1", file=buf, language="vi").text
        self.history.append({"role": "user", "content": text})
        reply = self.client.chat.completions.create(model=self.model, messages=self.history)
        answer = reply.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return f"🧑 {text}\n🤖 {answer}"


def make_backend(name: str | None = None) -> SessionBackend:
    name = name or os.environ.get("WAKEWORD_BACKEND", "mock")
    if name == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception as e:  # pragma: no cover
            print(f"[session] OpenAI backend unavailable ({e}); falling back to mock")
    return MockBackend()


@dataclass
class AssistantSession:
    timeout_s: float = 8.0
    backend: SessionBackend = field(default_factory=MockBackend)
    vad_db: float = -40.0
    state: State = State.IDLE
    last_activity: float = 0.0
    log: list = field(default_factory=list)
    _turn: list = field(default_factory=list)
    _ignore_until: float = 0.0

    def _emit(self, t: float, msg: str):
        self.log.append(f"[{t:7.2f}s] {msg}")

    def on_wake(self, t: float):
        if self.state == State.IDLE:
            self.state = State.SESSION
            self.last_activity = t
            self._turn = []
            self._ignore_until = t + 0.4  # tail of the wake phrase itself is not a user turn
            self._emit(t, self.backend.open())
        else:  # wake word inside a session just refreshes it
            self.last_activity = t

    def on_audio(self, t: float, chunk: np.ndarray, sr: int):
        """Call with every chunk; handles turn segmentation (energy VAD) and the timeout."""
        if self.state != State.SESSION or t < self._ignore_until:
            return
        level = 10 * np.log10(np.mean(chunk.astype(np.float64) ** 2) + 1e-12) if len(chunk) else -120
        if level > self.vad_db:
            self.last_activity = t
            self._turn.append(chunk)
        elif self._turn and t - self.last_activity > 0.8:  # end of a user turn
            audio = np.concatenate(self._turn)
            self._turn = []
            if len(audio) > 0.3 * sr:
                reply = self.backend.on_turn(audio, sr)
                if reply:
                    self._emit(t, reply)
        if t - self.last_activity > self.timeout_s:
            self.close(t, "timeout")

    def close(self, t: float, reason: str = "user"):
        if self.state == State.SESSION:
            self.state = State.IDLE
            self._turn = []
            self._emit(t, self.backend.close(reason))

    def remaining(self, t: float) -> float:
        return max(0.0, self.timeout_s - (t - self.last_activity)) if self.state == State.SESSION else 0.0


def simulate(events: list[float], duration: float, timeout_s: float = 8.0,
             activity: np.ndarray | None = None, hop_s: float = 0.1) -> list[tuple[float, float, str]]:
    """Offline: given wake events (s) and optional per-hop speech activity, return session intervals."""
    intervals = []
    sess_start, last = None, None
    ev = sorted(events)
    t = 0.0
    i = 0
    while t <= duration + 1e-9:
        while i < len(ev) and ev[i] <= t:
            if sess_start is None:
                sess_start = ev[i]
            last = ev[i]
            i += 1
        if sess_start is not None:
            k = int(t / hop_s)
            if activity is not None and k < len(activity) and activity[k]:
                last = t
            if t - last > timeout_s:
                intervals.append((sess_start, t, "timeout"))
                sess_start = None
        t += hop_s
    if sess_start is not None:
        intervals.append((sess_start, duration, "open"))
    return intervals


def now() -> float:
    return time.monotonic()
