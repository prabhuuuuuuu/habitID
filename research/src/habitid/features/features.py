"""Feature transforms. All normalization statistics are fitted on training-fold human sessions only."""
from __future__ import annotations

import numpy as np


def flight(h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Standard flight time (up of k1 to down of k2) = DD - hold(k1)."""
    return d - h[:, :-1]


class TypingNorm:
    """Log-transform holds and DDs, asinh-transform flights, z-score per keypress position."""

    def __init__(self, flight_scale: float):
        self.scale = flight_scale
        self.mu = {}
        self.sd = {}

    def _raw(self, h, d):
        f = flight(h, d)
        return np.log(h), np.log(d), np.arcsinh(f / self.scale)

    def fit(self, h: np.ndarray, d: np.ndarray) -> "TypingNorm":
        lh, ld, af = self._raw(h, d)
        for key, arr in (("h", lh), ("d", ld), ("f", af)):
            self.mu[key] = arr.mean(0)
            self.sd[key] = arr.std(0) + 1e-6
        return self

    def transform(self, h: np.ndarray, d: np.ndarray, use_flight: bool = True):
        """Return (seq, vec): seq (N, 10, 4) = [z_hold, z_dd, z_flight, is_last]; vec (N, 28)."""
        lh, ld, af = self._raw(h, d)
        zh = (lh - self.mu["h"]) / self.sd["h"]
        zd = (ld - self.mu["d"]) / self.sd["d"]
        zf = (af - self.mu["f"]) / self.sd["f"]
        if not use_flight:
            zf = np.zeros_like(zf)
        n = len(h)
        seq = np.zeros((n, h.shape[1], 4), dtype=np.float32)
        seq[:, :, 0] = zh
        seq[:, :-1, 1] = zd
        seq[:, :-1, 2] = zf
        seq[:, -1, 3] = 1.0
        vec = np.concatenate([zh, zd, zf], axis=1).astype(np.float32)
        return seq, vec

    def state(self) -> dict:
        return {"scale": self.scale, "mu": {k: v.tolist() for k, v in self.mu.items()},
                "sd": {k: v.tolist() for k, v in self.sd.items()}}


class MotionNorm:
    """Per-channel z-score fitted on training-fold human windows."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, windows: np.ndarray) -> "MotionNorm":
        flat = windows.reshape(-1, windows.shape[-1])
        self.mean = flat.mean(0).astype(np.float32)
        self.std = (flat.std(0) + 1e-6).astype(np.float32)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return ((windows - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def typing_handcrafted(h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """CV, mean, std of flight; hold and DD statistics; WPM-equivalent (log)."""
    f = flight(h, d)
    mf = f.mean(1)
    cv = f.std(1) / np.maximum(np.abs(mf), 1e-6)
    wpm = 120.0 / d.sum(1)
    return np.column_stack([mf, f.std(1), cv, h.mean(1), h.std(1), d.mean(1), d.std(1), np.log(wpm)])


def motion_handcrafted(windows: np.ndarray) -> np.ndarray:
    """Per-channel variance, acceleration energy, rotation energy (raw units)."""
    var = windows.var(1)
    e_acc = (windows[:, :, :3] ** 2).mean((1, 2))
    e_rot = (windows[:, :, 3:] ** 2).mean((1, 2))
    return np.column_stack([var, e_acc, e_rot])


def otp_handcrafted(otp_s: np.ndarray) -> np.ndarray:
    return np.log(np.clip(otp_s, 1e-3, None))[:, None]
