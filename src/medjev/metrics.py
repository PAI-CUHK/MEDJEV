from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=-1)


def fit_temperature(logits, y):
    logits = np.asarray(logits)
    result = minimize_scalar(lambda log_t: log_loss(y, probabilities(logits, np.exp(log_t)), labels=list(range(logits.shape[1]))),
                             bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 2 or not len(y) or not np.isfinite(p).all() or np.min(p) < 0 or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probability matrix")
    k = p.shape[1]
    if np.any(y < 0) or np.any(y >= k):
        raise ValueError("Labels outside the candidate set")
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == y
    ece = 0.
    reliability = []
    for i in range(10):
        mask = (conf >= i / 10) & (conf < (i + 1) / 10 if i < 9 else conf <= 1)
        if mask.any():
            acc, confidence = float(correct[mask].mean()), float(conf[mask].mean())
            ece += float(mask.mean()) * abs(acc - confidence)
            reliability.append({"lo": i / 10, "n": int(mask.sum()), "accuracy": acc, "confidence": confidence})
    order = np.argsort(-conf, kind="stable")
    risks = np.cumsum(~correct[order]) / np.arange(1, len(y) + 1)
    coverage = {}
    for c in [.25, .5, .75, 1.]:
        n = max(1, int(np.ceil(c * len(y))))
        coverage[str(c)] = float((~correct[order[:n]]).mean())
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0)),
            "nll": float(log_loss(y, p, labels=list(range(k)))),
            "brier": float(np.mean(np.sum((p - np.eye(k)[y]) ** 2, axis=1))),
            "ece_10": ece, "aurc": float(risks.mean()), "risk_at_coverage": coverage,
            "confusion": confusion_matrix(y, pred, labels=list(range(k))).tolist(),
            "reliability": reliability}


def paired_bootstrap(y, a, b, groups, repeats=1000, seed=17):
    """Document/premise-group paired interval, NOT a patient interval."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    keys = np.unique(groups)
    ix = [np.flatnonzero(groups == g) for g in keys]
    a_correct = (a.argmax(1) == y).astype(float)
    b_correct = (b.argmax(1) == y).astype(float)
    deltas = []
    for _ in range(repeats):
        ids = np.concatenate([ix[k] for k in rng.integers(0, len(ix), len(ix))])
        deltas.append(float((a_correct[ids] - b_correct[ids]).mean()))
    return {"metric": "accuracy_difference_a_minus_b", "point": float((a_correct-b_correct).mean()),
            "ci95": np.quantile(deltas, [.025, .975]).tolist(), "repeats": repeats, "group_unit": "exact_premise"}
