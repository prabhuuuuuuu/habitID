"""Load per-run prediction files into aligned test views. Positive (bot) score is higher for bots."""
from __future__ import annotations

import json

import numpy as np

from ..calibrate.calibration import p_human
from ..utils.config import RESULTS

FOLDS = range(5)


def run_path(variant: str, fold: int, seed: int):
    return RESULTS / "runs" / variant / f"fold{fold}_seed{seed}.npz"


def run_meta(variant: str, fold: int, seed: int) -> dict:
    return json.loads((RESULTS / "runs" / variant / f"fold{fold}_seed{seed}.json").read_text(encoding="utf-8"))


def neural_test(variant: str, fold: int, seed: int, split: str = "test", star: bool = False) -> dict:
    """Return the test (or D*) split of a neural run. `score` = -logit (higher = more bot)."""
    z = np.load(run_path(variant, fold, seed))
    pre = "star_" if star else f"{split}_"
    logit = z[f"{pre}logit"]
    T = float(z["T"])
    label = z[f"{pre}y"].astype(int)
    return {
        "score": -logit, "logit": logit, "p_h_raw": p_human(logit, 1.0), "p_h_cal": p_human(logit, T),
        "T": T, "label": label, "letter": z[f"{pre}letter"].astype(int), "type": z[f"{pre}type"].astype(int),
        "cluster": z[f"{pre}cluster"].astype(np.int64), "gate": z[f"{pre}gate"].astype(bool),
        "dataset": z[f"{pre}dataset"].astype(int), "otp_s": z[f"{pre}otp_s"],
        "contrib_otp": z[f"{pre}contrib_otp"], "fold": np.full(len(label), fold),
        "alpha": z["alpha"], "alpha0": z["alpha0"],
    }


def baseline_test(method: str, fold: int, seed: int, root: str = "baselines") -> dict:
    z = np.load(RESULTS / "runs" / root / method / f"fold{fold}_seed{seed}.npz")
    return {
        "score": z["test_score"].astype(float), "label": z["test_label"].astype(int),
        "letter": z["test_letter"].astype(int), "type": z["test_type"].astype(int),
        "cluster": z["test_cluster"].astype(np.int64), "gate": z["test_gate"].astype(bool),
        "dataset": z["test_dataset"].astype(int), "fold": np.full(len(z["test_label"]), fold),
    }


def pooled(loader, seed: int, folds=FOLDS, **kw) -> dict:
    """Concatenate the test views over folds for one seed. Each subject appears in exactly one test fold."""
    parts = []
    for k in folds:
        try:
            parts.append(loader(fold=k, seed=seed, **kw))
        except FileNotFoundError:
            # Only lam10 (fold 4 cut, see results/timing/cut_log.txt) has missing folds; the count is reported.
            print(f"  [pooled] fold {k} missing for {kw or seed}: skipped", flush=True)
    if not parts:
        raise RuntimeError("no folds available")
    keys = [k for k, v in parts[0].items() if isinstance(v, np.ndarray) and v.shape[:1] == (len(parts[0]["label"]),)]
    out = {}
    for key in keys:
        out[key] = np.concatenate([p[key] for p in parts])
    out["T"] = parts[0].get("T", 1.0)
    return out


def gate_filter(view: dict, keep_all: bool = False) -> dict:
    """Apply the validity gate: failed sessions are rejected before scoring (keep_all for the gate-only baseline)."""
    if keep_all:
        return view
    m = view["gate"]
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in view.items()}
