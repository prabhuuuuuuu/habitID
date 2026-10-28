"""Canonical typing sessions (CMU + KeyRecs fixed-text) and motion windows (MotionSense sit/std).

Typing session = 10 keypress positions. Holds h[0..9]; DD d[0..8]; standard flight f = d - h[:9].
KeyRecs stored UD/UU/DU are NOT used (sign convention, see DEVIATIONS D2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import INTERIM, PROC, RAW, RESULTS
from ..utils.runtime import write_json


def _bounds_bad(h, d, f, b) -> np.ndarray:
    bad_h = (h < b["hold"][0]) | (h > b["hold"][1])
    bad_d = (d < b["dd"][0]) | (d > b["dd"][1])
    bad_f = (f < b["flight"][0]) | (f > b["flight"][1])
    return bad_h.any(1) | bad_d.any(1) | bad_f.any(1)


def load_cmu(cfg: dict) -> dict:
    df = pd.read_csv(RAW / "DSL-StrongPasswordData.csv")
    n_keys = cfg["typing"]["n_keys"]
    hold = [c for c in df.columns if c.startswith("H.")][:n_keys]
    dd = [c for c in df.columns if c.startswith("DD.")][: n_keys - 1]
    timing = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
    corrupt = (df[timing].abs() > cfg["typing"]["corrupt_abs_s"]).any(axis=1).to_numpy()
    return {
        "h": df[hold].to_numpy(np.float64), "d": df[dd].to_numpy(np.float64),
        "subject": df["subject"].astype(str).to_numpy(),
        "session": (df["sessionIndex"] * 100 + df["rep"]).to_numpy(),
        "corrupt": corrupt, "dataset": "CMU", "n_rows": int(len(df)),
    }


def load_keyrecs_fixed(cfg: dict) -> dict:
    df = pd.read_csv(RAW / "fixed-text.csv")
    X = df.iloc[:, 3:49]  # 10 holds interleaved with 9 x (DD, DU, UD, UU)
    n_keys = cfg["typing"]["n_keys"]
    hold_pos = [5 * i for i in range(n_keys)]
    dd_pos = [5 * i + 1 for i in range(n_keys - 1)]
    corrupt = (X.abs() > cfg["typing"]["corrupt_abs_s"]).any(axis=1).to_numpy()
    return {
        "h": X.iloc[:, hold_pos].to_numpy(np.float64), "d": X.iloc[:, dd_pos].to_numpy(np.float64),
        "subject": df["participant"].astype(str).to_numpy(),
        "session": (df["session"] * 1000 + df["repetition"]).to_numpy(),
        "corrupt": corrupt, "dataset": "KeyRecs-fixed", "n_rows": int(len(df)),
    }


def build_typing(cfg: dict) -> dict:
    """Write data/processed/typing_canonical.npz and results/phase1/typing_counts.json."""
    b = cfg["typing"]["bounds_s"]
    parts = [load_cmu(cfg), load_keyrecs_fixed(cfg)]
    names = sorted({s for p in parts[:1] for s in np.unique(p["subject"])})
    kr_names = sorted(np.unique(parts[1]["subject"]))
    all_names = [f"CMU:{s}" for s in names] + [f"KeyRecs:{s}" for s in kr_names]
    idx_of = {n: i for i, n in enumerate(all_names)}
    H, D, SUB, DS, SESS, CORR = [], [], [], [], [], []
    counts = {}
    for ds_id, p in enumerate(parts):
        prefix = "CMU" if ds_id == 0 else "KeyRecs"
        sub = np.array([idx_of[f"{prefix}:{s}"] for s in p["subject"]])
        H.append(p["h"]); D.append(p["d"]); SUB.append(sub)
        DS.append(np.full(len(sub), ds_id)); SESS.append(p["session"]); CORR.append(p["corrupt"])
    H = np.concatenate(H); D = np.concatenate(D); SUB = np.concatenate(SUB)
    DS = np.concatenate(DS); SESS = np.concatenate(SESS); CORR = np.concatenate(CORR)
    F = D - H[:, :-1]
    oob = _bounds_bad(H, D, F, b) & ~CORR
    keep = ~CORR & ~oob
    for ds_id, p in enumerate(parts):
        m = DS == ds_id
        kept_subj = np.bincount(SUB[m & keep], minlength=len(all_names))
        kept_subj = kept_subj[kept_subj > 0]
        counts[p["dataset"]] = {
            "rows_raw": p["n_rows"], "corrupt_dropped": int(CORR[m].sum()),
            "out_of_bounds_dropped": int(oob[m].sum()), "kept": int(keep[m].sum()),
            "subjects_kept": int(len(kept_subj)),
            "sessions_per_subject_min": int(kept_subj.min()), "sessions_per_subject_max": int(kept_subj.max()),
        }
    counts["total_kept"] = int(keep.sum())
    counts["bounds_s"] = b
    counts["note"] = ("10-key canonical representation: CMU Return key dropped; "
                      "KeyRecs flight = DD - H(k1) (DEVIATIONS D2); stored KeyRecs UD/UU/DU not used.")
    out = PROC / "typing_canonical.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, h=H[keep], d=D[keep], f=F[keep], subject=SUB[keep], dataset=DS[keep],
                        session=SESS[keep], subject_names=np.array(all_names))
    write_json(RESULTS / "phase1" / "typing_counts.json", counts)
    return counts


def build_motion(cfg: dict) -> dict:
    """Non-overlapping windows of `window_s` seconds from MotionSense sit/std trials."""
    root = INTERIM / "motionsense" / "A_DeviceMotion_data" / "A_DeviceMotion_data"
    win = int(round(cfg["motion"]["fs_hz"] * cfg["motion"]["window_s"]))
    chans = cfg["motion"]["channels"]
    W, S, A, T = [], [], [], []
    per_act = {}
    for act_idx, act in enumerate(cfg["motion"]["activities"]):
        for trial_dir in sorted(root.glob(f"{act}_*")):
            trial = int(trial_dir.name.split("_")[1])
            for f in sorted(trial_dir.glob("sub_*.csv")):
                sub = int(f.stem.split("_")[1]) - 1
                x = pd.read_csv(f, usecols=chans).to_numpy(np.float32)
                n = len(x) // win
                if n == 0:
                    continue
                W.append(x[: n * win].reshape(n, win, len(chans)))
                S += [sub] * n; A += [act_idx] * n; T += [trial] * n
                per_act[act] = per_act.get(act, 0) + n
    W = np.concatenate(W)
    out = PROC / "motion_canonical.npz"
    np.savez_compressed(out, windows=W, subject=np.array(S), activity=np.array(A), trial=np.array(T))
    counts = {"windows": int(len(W)), "window_samples": win, "channels": chans,
              "windows_per_activity": per_act, "subjects": int(len(set(S))),
              "windows_per_subject_min": int(np.bincount(S).min()),
              "windows_per_subject_max": int(np.bincount(S).max()),
              "fs_hz_assumed": cfg["motion"]["fs_hz"], "fs_verified": False}
    write_json(RESULTS / "phase1" / "motion_counts.json", counts)
    return counts
