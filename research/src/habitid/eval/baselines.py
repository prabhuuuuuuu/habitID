"""Baselines: habitID-v0 (hand-weighted additive score), validity gate only, LR / GBT on hand-crafted
features, one-class detectors on typing (Isolation Forest, one-class SVM, population scaled-Manhattan)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..features.features import flight, motion_handcrafted, otp_handcrafted, typing_handcrafted


def _ramp(x, a0, a1, a2, a3):
    """0 at a0, 1 at a1, 1 until a2, 0 at a3; linear between."""
    up = (x - a0) / (a1 - a0)
    down = (a3 - x) / (a3 - a2)
    return np.clip(np.minimum(up, down), 0.0, 1.0)


def v0_score(h, d, windows, otp_s, gate, v0: dict):
    """S0 = sum_i w_i s_i with the thresholds in configs/v0_weights.yaml. Returns (S0, verdict array)."""
    w = v0["weights"]
    f = flight(h, d)
    cv = f.std(1) / np.maximum(np.abs(f.mean(1)), 1e-9)
    s_cv = _ramp(cv, *v0["typing"]["cv_ramp"])
    wpm = 120.0 / d.sum(1)
    lo, hi = v0["typing"]["wpm_plausible"]
    s_wpm = ((wpm >= lo) & (wpm <= hi)).astype(float)
    s_typing = 0.5 * s_cv + 0.5 * s_wpm
    mag = np.sqrt((windows[:, :, :3] ** 2).sum(-1))
    cvm = mag.std(1) / np.maximum(mag.mean(1), 1e-9)
    s_motion = _ramp(cvm, *v0["motion"]["cv_ramp"])
    a0, a1, a2, a3 = v0["otp"]["ramp_s"]
    s_otp = np.clip(np.minimum((otp_s - a0) / (a1 - a0), (a3 - otp_s) / (a3 - a2)), 0.0, 1.0)
    s_aadhaar = np.where(gate, v0["aadhaar_valid_score"], 0.0)
    s_sim = np.full(len(h), v0["sim_age_neutral"])
    s_phone = np.full(len(h), v0["phone_format_score"])
    S0 = (w["typing"] * s_typing + w["motion"] * s_motion + w["aadhaar"] * s_aadhaar + w["otp"] * s_otp
          + w["sim_age"] * s_sim + w["phone_format"] * s_phone)
    vt, rm = v0["verdict"]["verified_min"], v0["verdict"]["review_min"]
    verdict = np.where(S0 >= vt, "VERIFIED", np.where(S0 >= rm, "REVIEW", "SUSPICIOUS"))
    return S0, verdict


def handcrafted(h, d, windows, otp_s) -> np.ndarray:
    return np.hstack([typing_handcrafted(h, d), motion_handcrafted(windows), otp_handcrafted(otp_s)])


def fit_lr_gbt(train: dict, test: dict, seed: int):
    Xtr = handcrafted(train["h"], train["d"], train["windows"], train["otp_s"])
    Xte = handcrafted(test["h"], test["d"], test["windows"], test["otp_s"])
    y = 1 - train["label"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(Xtr, y)
    gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    gbt.fit(Xtr, y)
    return lr.predict_proba(Xte)[:, 1], gbt.predict_proba(Xte)[:, 1]


def fit_oneclass(train_vec_h: np.ndarray, test_vec: np.ndarray, seed: int, svm_n: int = 4000):
    """Fit on human training typing vectors only. Scores: higher = more anomalous (bot-like)."""
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(train_vec_h)
    s_iso = -iso.score_samples(test_vec)
    rng = np.random.default_rng(seed)
    sub = train_vec_h[rng.choice(len(train_vec_h), size=min(svm_n, len(train_vec_h)), replace=False)]
    svm = OneClassSVM(nu=0.1, gamma="scale").fit(sub)
    s_svm = -svm.decision_function(test_vec)
    med = np.median(train_vec_h, 0)
    mad = np.median(np.abs(train_vec_h - med), 0) * 1.4826 + 1e-6
    s_smd = np.mean(np.abs(test_vec - med) / mad, axis=1)
    return s_iso, s_svm, s_smd
