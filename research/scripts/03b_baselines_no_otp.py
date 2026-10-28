"""Baselines WITHOUT the OTP term (typing + motion only). The OTP latency is an assumed-parameter signal
(no public data), so the no-OTP versions are the primary scientific comparison.

Writes results/runs/baselines_no_otp/<method>/fold{k}_seed{s}.npz in the same format as 03_baselines.py.
  lr_no_otp, gbt_no_otp: same models as 03_baselines.py, OTP feature column removed.
  habitid_v0_no_otp    : S0 with the 15-point OTP term removed (ranking-equivalent to the reduced sum).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from habitid.eval.baselines import _ramp, handcrafted, v0_score  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_v0  # noqa: E402
from habitid.utils.runtime import phase_timer  # noqa: E402

KEYS = ("label", "letter", "type", "cluster", "gate", "dataset", "otp_s")


def save(method, k, seed, score, te):
    out = RESULTS / "runs" / "baselines_no_otp" / method
    out.mkdir(parents=True, exist_ok=True)
    arrs = {"test_score": score, "fold": np.full(len(score), k)}
    for key in KEYS:
        arrs[f"test_{key}"] = te[key]
    np.savez_compressed(out / f"fold{k}_seed{seed}.npz", **arrs)


def main():
    cfg = load_cfg()
    v0 = load_v0()
    seeds = cfg["project"]["train_seeds"]
    with phase_timer("phase4_baselines_no_otp"):
        for k in range(cfg["project"]["n_folds"]):
            f = np.load(PROC / "folds" / f"fold{k}.npz")
            tr = {"h": f["train_h"], "d": f["train_d"], "windows": f["train_windows"], "otp_s": f["train_otp_s"],
                  "label": f["train_label"]}
            te = {key: f[f"test_{key}"] for key in ("h", "d", "windows", "otp_s", "label", "letter", "type",
                                                    "cluster", "gate", "dataset")}
            te["gate"] = te["gate"].astype(bool)
            Xtr = handcrafted(tr["h"], tr["d"], tr["windows"], tr["otp_s"])[:, :-1]
            Xte = handcrafted(te["h"], te["d"], te["windows"], te["otp_s"])[:, :-1]
            y = 1 - tr["label"]
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, y)
            save("lr_no_otp", k, 0, lr.predict_proba(Xte)[:, 1], te)
            for seed in seeds:
                g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(Xtr, y)
                save("gbt_no_otp", k, seed, g.predict_proba(Xte)[:, 1], te)
            S0, _ = v0_score(te["h"], te["d"], te["windows"], te["otp_s"], te["gate"], v0)
            a0, a1, a2, a3 = v0["otp"]["ramp_s"]
            s_otp = _ramp(te["otp_s"], a0, a1, a2, a3)
            save("habitid_v0_no_otp", k, 0, 100.0 - (S0 - v0["weights"]["otp"] * s_otp), te)
            print(f"fold {k} no-OTP baselines done", flush=True)


if __name__ == "__main__":
    main()
