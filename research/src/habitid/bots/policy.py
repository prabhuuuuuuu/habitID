"""Session composition.

A session = (typing sequence, motion window, OTP latency, validity gate).
Human sessions cross-pair a typing session and a motion window drawn from the SAME role
(train/cal/test) of the SAME fold: subjects are never shared across roles. Within-person
dependence across modalities is not preserved (documented independence assumption).

Bot sessions follow a joint policy. A bot session has a family letter:
  A/B/C/D  joint family: typing family letter, motion family letter, OTP automated (type 'full'),
           or only typing automated ('typing_only'), or only motion automated ('motion_only').
  O        OTP automated only ('otp_only'); typing and motion are human.
  D*       test-only adaptive variant of D (best-of-N against the trained detector), added at evaluation.
"""
from __future__ import annotations

import numpy as np

from ..bots.otp import sample_latency
from ..data.validity import synthetic_gate
from ..features.features import flight

LETTER_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "O": 5, "D*": 6}
TYPE_CODE = {"human": 0, "full": 1, "typing_only": 2, "motion_only": 3, "otp_only": 4}


def valid_typing(h: np.ndarray, d: np.ndarray, bounds: dict) -> np.ndarray:
    f = flight(h, d)
    ok = np.ones(len(h), dtype=bool)
    ok &= ~((h < bounds["hold"][0]) | (h > bounds["hold"][1])).any(1)
    ok &= ~((d < bounds["dd"][0]) | (d > bounds["dd"][1])).any(1)
    ok &= ~((f < bounds["flight"][0]) | (f > bounds["flight"][1])).any(1)
    return ok


def sample_valid_typing(gen, n: int, rng: np.random.Generator, bounds: dict):
    """Rejection sampling so that generated bot sessions obey the same validity bounds as humans."""
    H, D = [], []
    got = 0
    while got < n:
        h, d = gen.sample(max(2 * (n - got), 64), rng)
        ok = valid_typing(h, d, bounds)
        H.append(h[ok]); D.append(d[ok]); got += int(ok.sum())
    return np.concatenate(H)[:n], np.concatenate(D)[:n]


def draw_humans(n: int, rng: np.random.Generator, pool: dict, motion_windows: np.ndarray,
                motion_subject: np.ndarray, otp_cfg: dict, gate_rate: float, cluster_fn=None) -> dict:
    """Pool: dict with h, d, subject, dataset. motion_windows / motion_subject: role-restricted motion pool."""
    replace = n > len(pool["h"])
    ti = rng.choice(len(pool["h"]), size=n, replace=replace)
    mi = rng.choice(len(motion_windows), size=n, replace=True)
    h, d = pool["h"][ti], pool["d"][ti]
    return {
        "h": h, "d": d, "windows": motion_windows[mi], "otp_s": sample_latency(rng, n, otp_cfg, "human"),
        "label": np.ones(n, dtype=np.int64), "letter": np.zeros(n, dtype=np.int64),
        "type": np.zeros(n, dtype=np.int64), "subject": pool["subject"][ti],
        "motion_subject": motion_subject[mi], "dataset": pool["dataset"][ti],
        "cluster": pool["subject"][ti].astype(np.int64),
        "gate": synthetic_gate(rng, n, gate_rate),
    }


def draw_bots(n: int, rng: np.random.Generator, typing_gens: dict, motion_gens: dict, pool: dict,
              motion_windows: np.ndarray, motion_subject: np.ndarray, otp_cfg: dict, bounds: dict,
              mix: dict, gate_rate: float, cluster_base: int, motion_T: int) -> dict:
    types = rng.choice(list(mix), size=n, p=list(mix.values()))
    letters = np.where(types == "otp_only", "O", rng.choice(["A", "B", "C", "D"], size=n))
    typed = np.isin(types, ["full", "typing_only"])
    moved = np.isin(types, ["full", "motion_only"])
    otp_bot = np.isin(types, ["full", "otp_only"])

    h = np.zeros((n, 10)); d = np.zeros((n, 9))
    W = np.zeros((n, motion_T, motion_windows.shape[-1]), dtype=np.float32)
    subj_m = np.full(n, -1)
    if (~typed).any():
        ti = rng.choice(len(pool["h"]), size=int((~typed).sum()), replace=True)
        h[~typed], d[~typed] = pool["h"][ti], pool["d"][ti]
    for L in ["A", "B", "C", "D"]:
        m = typed & (letters == L)
        if m.any():
            h[m], d[m] = sample_valid_typing(typing_gens[L], int(m.sum()), rng, bounds)
    if (~moved).any():
        mi = rng.choice(len(motion_windows), size=int((~moved).sum()))
        W[~moved] = motion_windows[mi]
        subj_m[~moved] = motion_subject[mi]
    for L in ["A", "B", "C", "D"]:
        m = moved & (letters == L)
        if m.any():
            W[m] = motion_gens[L].sample(int(m.sum()), motion_T, rng)
    otp = np.zeros(n)
    otp[otp_bot] = sample_latency(rng, int(otp_bot.sum()), otp_cfg, "automated")
    otp[~otp_bot] = sample_latency(rng, int((~otp_bot).sum()), otp_cfg, "human")
    code = np.array([LETTER_CODE[x] for x in letters])
    return {
        "h": h, "d": d, "windows": W, "otp_s": otp,
        "label": np.zeros(n, dtype=np.int64), "letter": code,
        "type": np.array([TYPE_CODE[x] for x in types]), "subject": np.full(n, -1),
        "motion_subject": subj_m, "dataset": np.full(n, -1),
        "cluster": cluster_base + code * 100 + np.arange(n) // 50,
        "gate": synthetic_gate(rng, n, gate_rate),
    }


def concat_splits(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}
