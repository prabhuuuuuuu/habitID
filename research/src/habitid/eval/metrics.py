"""Detection metrics. Positive class = automated (bot) session; `score` is higher for bots."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auc(bot: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(bot, score))


def eer(bot: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(bot, score)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def tpr_at_fpr(bot: np.ndarray, score: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(bot, score)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def pr_auc_base_rate(bot: np.ndarray, score: np.ndarray, base_rate: float) -> float:
    """Average precision when automated sessions occur at `base_rate` in deployment.

    Reweights the test set (bots get weight base_rate/n_bot, humans (1-base_rate)/n_human); no resampling.
    """
    n1 = bot.sum()
    n0 = len(bot) - n1
    w = np.where(bot == 1, base_rate / n1, (1 - base_rate) / n0)
    return float(average_precision_score(bot, score, sample_weight=w))


def ece(p_human: np.ndarray, y_human: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_human, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p_human[m].mean() - y_human[m].mean())
    return float(total)


def reliability(p_human: np.ndarray, y_human: np.ndarray, bins: int = 15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_human, edges[1:-1]), 0, bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        conf.append(float(p_human[m].mean()) if m.any() else float("nan"))
        acc.append(float(y_human[m].mean()) if m.any() else float("nan"))
        cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def detection_summary(bot: np.ndarray, score: np.ndarray, base_rates, fpr_target: float) -> dict:
    out = {"AUC": auc(bot, score), "EER": eer(bot, score), f"TPR@{fpr_target:.0%}FPR": tpr_at_fpr(bot, score, fpr_target)}
    for br in base_rates:
        out[f"PR-AUC@{br:g}"] = pr_auc_base_rate(bot, score, br)
    return out
