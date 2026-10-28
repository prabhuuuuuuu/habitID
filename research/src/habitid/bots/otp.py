"""OTP latency model: ASSUMED Gaussian mixtures over log latency (configs/otp_assumptions.yaml)."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def components(otp_cfg: dict, side: str, human_mult: float = 1.0, bot_near_weight: float | None = None):
    """Return (weights, mu_log, sigma_log) arrays for side in {'human', 'automated'}.

    human_mult scales the medians of human components (sensitivity analysis).
    bot_near_weight overrides the weight of the automated near-zero component (the other
    automated component takes the complement).
    """
    comps = otp_cfg[side]["components"]
    w = np.array([c["weight"] for c in comps], dtype=float)
    med = np.array([c["median_s"] for c in comps], dtype=float)
    sig = np.array([c["sigma_log"] for c in comps], dtype=float)
    if side == "human":
        med = med * human_mult
    if side == "automated" and bot_near_weight is not None:
        w = np.array([bot_near_weight, 1.0 - bot_near_weight])
    return w / w.sum(), np.log(med), sig


def sample_latency(rng: np.random.Generator, n: int, otp_cfg: dict, side: str,
                   human_mult: float = 1.0, bot_near_weight: float | None = None) -> np.ndarray:
    w, mu, sg = components(otp_cfg, side, human_mult, bot_near_weight)
    k = rng.choice(len(w), size=n, p=w)
    x = rng.normal(mu[k], sg[k])
    lo, hi = otp_cfg["clip_s"]
    return np.clip(np.exp(x), lo, hi)


def log_pdf(logt: np.ndarray, otp_cfg: dict, side: str, human_mult: float = 1.0,
            bot_near_weight: float | None = None) -> np.ndarray:
    w, mu, sg = components(otp_cfg, side, human_mult, bot_near_weight)
    x = logt[:, None]
    lp = np.log(w)[None, :] - 0.5 * np.log(2 * np.pi) - np.log(sg)[None, :] - 0.5 * ((x - mu) / sg) ** 2
    return logsumexp(lp, axis=1)


def llr_numpy(otp_s: np.ndarray, otp_cfg: dict, human_mult: float = 1.0,
              bot_near_weight: float | None = None) -> np.ndarray:
    """log p(log t | human) - log p(log t | automated), clipped (numpy reference implementation)."""
    lo, hi = otp_cfg["clip_s"]
    logt = np.log(np.clip(otp_s, lo, hi))
    llr = log_pdf(logt, otp_cfg, "human", human_mult) - log_pdf(logt, otp_cfg, "automated", 1.0, bot_near_weight)
    c = otp_cfg["llr_clip"]
    return np.clip(llr, -c, c)
