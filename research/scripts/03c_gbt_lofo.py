"""Leave-one-family-out for the GBDT baseline, using the same protocol as habitID-Net (scripts/05):
family f is removed from training AND calibration-side scoring, and the model is tested on humans vs family f.
GBDT has no calibration step of its own; its predicted probabilities are used as-is for ECE.
Writes results/runs/baselines_lofo/gbt_lofo_{L}/fold{k}_seed0.npz (test split, same format as 03_baselines)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from habitid.bots.policy import LETTER_CODE  # noqa: E402
from habitid.eval.baselines import handcrafted  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg  # noqa: E402
from habitid.utils.runtime import phase_timer  # noqa: E402

KEYS = ("label", "letter", "type", "cluster", "gate", "dataset", "otp_s")


def main():
    cfg = load_cfg()
    with phase_timer("phase4_gbt_lofo"):
        for k in range(cfg["project"]["n_folds"]):
            f = np.load(PROC / "folds" / f"fold{k}.npz")
            Xtr_all = handcrafted(f["train_h"], f["train_d"], f["train_windows"], f["train_otp_s"])
            Xcal_all = handcrafted(f["cal_h"], f["cal_d"], f["cal_windows"], f["cal_otp_s"])
            Xte = handcrafted(f["test_h"], f["test_d"], f["test_windows"], f["test_otp_s"])
            ytr_all = 1 - f["train_label"]
            ycal_all = 1 - f["cal_label"]
            for L in "ABCD":
                code = LETTER_CODE[L]
                mtr = f["train_letter"] != code
                mca = f["cal_letter"] != code
                g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
                g.fit(np.vstack([Xtr_all[mtr], Xcal_all[mca]]), np.concatenate([ytr_all[mtr], ycal_all[mca]]))
                prob = g.predict_proba(Xte)[:, 1]
                out = RESULTS / "runs" / "baselines_lofo" / f"gbt_lofo_{L}"
                out.mkdir(parents=True, exist_ok=True)
                arrs = {"test_score": prob, "fold": np.full(len(prob), k)}
                for key in KEYS:
                    arrs[f"test_{key}"] = f[f"test_{key}"]
                np.savez_compressed(out / f"fold{k}_seed0.npz", **arrs)
            print(f"fold {k} GBT-LOFO done", flush=True)


if __name__ == "__main__":
    main()
