"""Wake-word metrics: FRR, FAR, false activations per hour, DET curve, operating points."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def clip_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    pos, neg = labels == 1, labels == 0
    out = {
        "threshold": float(threshold),
        "frr": float(np.mean(probs[pos] < threshold)) if pos.any() else float("nan"),
        "far": float(np.mean(probs[neg] >= threshold)) if neg.any() else float("nan"),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
    }
    if pos.any() and neg.any():
        out["auc"] = float(roc_auc_score(labels, probs))
        fpr, tpr, thr = roc_curve(labels, probs)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[i] + fnr[i]) / 2)
        out["eer_threshold"] = float(thr[i])
    return out


def threshold_for_far(labels: np.ndarray, probs: np.ndarray, target_far: float) -> float:
    """Smallest threshold whose clip-level FAR on negatives is <= target (used on validation)."""
    neg = np.sort(np.asarray(probs)[np.asarray(labels) == 0])[::-1]
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(target_far * len(neg)))
    return float(min(0.999, neg[k] + 1e-6)) if k < len(neg) else float(neg[-1])


def per_category_rates(categories, probs, threshold: float) -> dict:
    """Trigger rate per category (for positives = 1 - FRR, for negatives = FAR)."""
    categories = np.asarray(categories)
    probs = np.asarray(probs)
    return {c: {"trigger_rate": float(np.mean(probs[categories == c] >= threshold)),
                "n": int(np.sum(categories == c))}
            for c in sorted(set(categories.tolist()))}


def det_curve(frr_by_thr: np.ndarray, fa_per_hour_by_thr: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    return [{"threshold": float(t), "frr": float(f), "fa_per_hour": float(a)}
            for t, f, a in zip(thresholds, frr_by_thr, fa_per_hour_by_thr)]


def operating_point(det: list[dict], max_fa_per_hour: float) -> dict | None:
    """Lowest-FRR point with FA/h <= budget (ties -> highest threshold)."""
    ok = [d for d in det if d["fa_per_hour"] <= max_fa_per_hour]
    if not ok:
        return None
    best = min(ok, key=lambda d: (d["frr"], -d["threshold"]))
    return best
