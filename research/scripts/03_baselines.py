"""Phase 4 (baselines): habitID-v0, validity gate only, LR / GBT on hand-crafted features,
Isolation Forest / one-class SVM / scaled-Manhattan on typing (fit on human training data only).

Writes results/runs/baselines/<method>/fold{k}_seed{s}.npz with the test-split predictions.
Deterministic methods are written once (seed 0)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from habitid.eval.baselines import fit_lr_gbt, fit_oneclass, v0_score  # noqa: E402
from habitid.features.features import TypingNorm  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_v0  # noqa: E402
from habitid.utils.runtime import phase_timer  # noqa: E402

KEYS = ("label", "letter", "type", "cluster", "gate", "dataset", "otp_s")


def save(method: str, k: int, seed: int, score: np.ndarray, te: dict, verdict=None):
    out = RESULTS / "runs" / "baselines" / method
    out.mkdir(parents=True, exist_ok=True)
    arrs = {"test_score": score, "fold": np.full(len(score), k)}
    for key in KEYS:
        arrs[f"test_{key}"] = te[key]
    if verdict is not None:
        arrs["test_verdict"] = verdict.astype("U10")
    np.savez_compressed(out / f"fold{k}_seed{seed}.npz", **arrs)


def main():
    cfg = load_cfg()
    v0 = load_v0()
    seeds = cfg["project"]["train_seeds"]
    folds = [int(x) for x in sys.argv[1:]] or list(range(cfg["project"]["n_folds"]))
    with phase_timer("phase4_baselines"):
        for k in folds:
            f = np.load(PROC / "folds" / f"fold{k}.npz")
            tr = {"h": f["train_h"], "d": f["train_d"], "windows": f["train_windows"], "otp_s": f["train_otp_s"],
                  "label": f["train_label"]}
            te = {key: f[f"test_{key}"] for key in ("h", "d", "windows", "otp_s", "label", "letter", "type",
                                                    "cluster", "gate", "dataset")}
            te["gate"] = te["gate"].astype(bool)

            S0, verdict = v0_score(te["h"], te["d"], te["windows"], te["otp_s"], te["gate"], v0)
            save("habitid_v0", k, 0, 100.0 - S0, te, verdict)
            save("gate_only", k, 0, (~te["gate"]).astype(float), te)

            for seed in seeds:
                lr_p, gbt_p = fit_lr_gbt(tr, te, seed)
                save("lr_handcrafted", k, seed, lr_p, te) if seed == seeds[0] else None
                save("gbt_handcrafted", k, seed, gbt_p, te)

            tnorm = TypingNorm(cfg["typing"]["flight_asinh_scale"])
            hm = tr["label"] == 1
            tnorm.fit(tr["h"][hm], tr["d"][hm])
            _, vec_h = tnorm.transform(tr["h"][hm], tr["d"][hm])
            _, vec_te = tnorm.transform(te["h"], te["d"])
            for seed in seeds:
                s_iso, s_svm, s_smd = fit_oneclass(vec_h, vec_te, seed)
                save("isolation_forest", k, seed, s_iso, te)
                save("one_class_svm", k, seed, s_svm, te)
                if seed == seeds[0]:
                    save("scaled_manhattan", k, 0, s_smd, te)
            print(f"fold {k} baselines done", flush=True)


if __name__ == "__main__":
    main()
