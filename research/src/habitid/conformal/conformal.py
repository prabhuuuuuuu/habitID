"""Class-conditional split conformal prediction for the human / automated decision.

Nonconformity s(x, c) = 1 - p_c(x). For each class c, q_c is the ceil((n_c + 1)(1 - eps)) / n_c
empirical quantile of calibration scores of class c. Prediction set C(x) = {c : s(x, c) <= q_c}.
Verdict map: {human} -> VERIFIED, {both} -> REVIEW, {bot} -> SUSPICIOUS, empty -> REVIEW.
Class labels: 1 = human, 0 = bot.
"""
from __future__ import annotations

import math

import numpy as np


def class_thresholds(p_h_cal: np.ndarray, y_cal: np.ndarray, eps: float) -> dict[int, float]:
    scores = {1: 1.0 - p_h_cal, 0: p_h_cal}  # s(x,1) = 1 - p_h ; s(x,0) = 1 - p_bot = p_h
    q = {}
    for c in (0, 1):
        s = scores[c][y_cal == c]
        n = len(s)
        level = math.ceil((n + 1) * (1.0 - eps)) / n
        q[c] = float("inf") if level > 1.0 else float(np.quantile(s, level, method="higher"))
    return q


def prediction_sets(p_h: np.ndarray, q: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    in_h = (1.0 - p_h) <= q[1]
    in_b = p_h <= q[0]
    return in_h, in_b


def verdicts(in_h: np.ndarray, in_b: np.ndarray) -> np.ndarray:
    out = np.full(len(in_h), "REVIEW", dtype="U10")
    out[in_h & ~in_b] = "VERIFIED"
    out[in_b & ~in_h] = "SUSPICIOUS"
    return out


def summarize(in_h: np.ndarray, in_b: np.ndarray, y_human: np.ndarray, letter: np.ndarray | None = None) -> dict:
    """Empirical coverage per class, abstention rate, and bot coverage per attacker letter."""
    out = {
        "coverage_human": float(in_h[y_human == 1].mean()),
        "coverage_bot": float(in_b[y_human == 0].mean()),
        "abstention": float((in_h == in_b).mean()),
        "verified_rate_human": float((in_h & ~in_b)[y_human == 1].mean()),
        "suspicious_rate_bot": float((in_b & ~in_h)[y_human == 0].mean()),
    }
    if letter is not None:
        per = {}
        for L in np.unique(letter[y_human == 0]):
            m = (letter == L) & (y_human == 0)
            per[int(L)] = float(in_b[m].mean())
        out["coverage_bot_by_letter"] = per
    return out
