import numpy as np
import pytest
import torch
from conftest import ROOT

from wakeword.augment import SpecAugment, WaveAugment, mix_at_snr, speed_perturb
from wakeword.config import load_config
from wakeword.features import LogMel, feature_shape
from wakeword.metrics import clip_metrics, operating_point, threshold_for_far
from wakeword.models import build_model, count_macs, count_params
from wakeword.phrases import CATEGORY_LABEL, default_prompts, tts_prompts
from wakeword.session import AssistantSession, State, simulate
from wakeword.streaming import DetectorParams, StreamingDetector, detect_events, smooth_scores


def cfg(name="base.yaml", *ov):
    return load_config(ROOT / "configs" / name, list(ov))


def test_config_inheritance_and_override():
    c = cfg("tc_resnet.yaml", "train.epochs=3", "model.width=2.0")
    assert c.train.epochs == 3 and c.model.width == 2.0
    assert "channels" not in c.model  # model section replaced when `name` changes
    assert cfg("ablation/a0_real_only.yaml").aug.enabled is False


def test_logmel_shape():
    c = cfg()
    assert feature_shape(c) == (1, 40, 201)
    x = LogMel()(torch.randn(3, 32000))
    assert x.shape == (3, 1, 40, 201)


@pytest.mark.parametrize("conf", ["ds_cnn.yaml", "tc_resnet.yaml", "crnn.yaml"])
def test_models_forward_and_small(conf):
    c = cfg(conf)
    m = build_model(c, (1, 40, 201))
    out = m(torch.randn(2, 1, 40, 201))
    assert out.shape == (2,)
    assert count_params(m) < 300_000
    assert count_macs(m, (1, 40, 201)) > 0


def test_emb_head():
    c = cfg("oww_head.yaml")
    m = build_model(c, (16, 96))
    assert m(torch.randn(4, 16, 96)).shape == (4,)


def test_augment_preserves_length():
    c = cfg()
    aug = WaveAugment(c.aug)
    rng = np.random.default_rng(0)
    x = np.random.randn(32000).astype(np.float32) * 0.1
    assert len(aug.post(x, rng)) == 32000
    assert abs(len(speed_perturb(x, 1.1)) - 32000 / 1.1) < 50
    y = mix_at_snr(x, np.random.randn(32000).astype(np.float32), 10.0)
    assert y.shape == x.shape
    sa = SpecAugment().train()
    assert sa(torch.randn(2, 1, 40, 201)).shape == (2, 1, 40, 201)


def test_streaming_detector_refractory():
    scores = iter([0.1, 0.2, 0.9, 0.95, 0.9, 0.1, 0.1] + [0.9] * 30)
    det = StreamingDetector(lambda w: next(scores), 1600, 16000, DetectorParams(0.5, 1, 1.0, 0.1))
    events = det.push(np.zeros(1600 * 30, np.float32))
    times = [e["time"] for e in events]
    assert len(times) >= 2 and all(b - a >= 1.0 - 1e-6 for a, b in zip(times, times[1:]))


def test_offline_events_and_smoothing():
    t = np.arange(10) * 0.1
    s = smooth_scores(np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1.0]), 2)
    assert detect_events(t, s, 0.75, 0.5) == [pytest.approx(0.3), pytest.approx(0.9)]


def test_metrics():
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.3, 0.1, 0.2, 0.7, 0.05])
    m = clip_metrics(labels, probs, 0.5)
    assert m["frr"] == pytest.approx(1 / 3) and m["far"] == pytest.approx(0.25)
    thr = threshold_for_far(labels, probs, 0.0)
    assert np.mean(probs[labels == 0] >= thr) == 0
    det = [{"threshold": 0.3, "frr": 0.1, "fa_per_hour": 2.0}, {"threshold": 0.6, "frr": 0.2, "fa_per_hour": 0.3}]
    assert operating_point(det, 0.5)["threshold"] == 0.6


def test_session_state_machine():
    s = AssistantSession(timeout_s=2.0)
    s.on_wake(1.0)
    assert s.state == State.SESSION
    s.on_audio(2.5, np.zeros(1600, np.float32), 16000)
    assert s.state == State.SESSION
    s.on_audio(3.5, np.zeros(1600, np.float32), 16000)
    assert s.state == State.IDLE
    assert simulate([1.0], 10.0, timeout_s=2.0)[0][2] == "timeout"


def test_prompts_consistent():
    for p in default_prompts() + tts_prompts():
        assert p.category in CATEGORY_LABEL


def test_stt_matcher():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from stt_baseline import is_wake
    assert is_wake("A lô, Gi Pi Ti à em, bật nhạc")
    assert not is_wake("Alo GPT à anh")
    assert not is_wake("Alo em à")
