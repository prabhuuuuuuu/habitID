"""Subject-cluster bootstrap (paired across methods) and Holm-Bonferroni adjustment.

Clusters: human sessions cluster by their typing subject; automated sessions cluster by generator
instance (50 sessions per instance, per family and fold). Resampling draws clusters with replacement
separately within the human and automated classes, so class balance is preserved.
"""
from __future__ import annotations

import numpy as np


class ClusterSampler:
    def __init__(self, cluster: np.ndarray, label: np.ndarray):
        self.groups = {}
        for c in (0, 1):
            m = label == c
            ids = np.unique(cluster[m])
            self.groups[c] = [np.where(m & (cluster == i))[0] for i in ids]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for c in (0, 1):
            g = self.groups[c]
            pick = rng.integers(0, len(g), size=len(g))
            parts += [g[j] for j in pick]
        return np.concatenate(parts)


def bootstrap_indices(cluster: np.ndarray, label: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    s = ClusterSampler(cluster, label)
    return [s.draw(rng) for _ in range(n)]


def percentile_ci(vals: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_p_value(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (centred at 0 with the sign of the estimate)."""
    p_le = np.mean(diff <= 0)
    p_ge = np.mean(diff >= 0)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
