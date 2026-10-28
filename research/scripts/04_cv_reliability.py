"""Phase 4 (experiment 5): reliability of the flight-time CV as a function of the number of keys n.

For CMU (11 keys, raw file) and KeyRecs-fixed (10 keys, canonical), truncate sessions to their first n keys
(n-1 flight times) and report:
  - bootstrap standard error of the per-session CV (200 within-session resamples of the flight times),
  - share of human sessions whose CV lies in the claimed 0.2-1.0 window,
  - AUC of the CV alone for human vs family-B (humanized) bots, with the direction fixed a priori
    (humans are assumed to have more variable flight times).
The humanized family is fitted on half of the subjects and humans are evaluated on the other half.
Descriptive: this is a reliability study, not a detection estimate for the full system.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from habitid.data.canonical import _bounds_bad  # noqa: E402
from habitid.utils.config import PROC, RAW, RESULTS, load_cfg  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402


def load_cmu_raw():
    df = pd.read_csv(RAW / "DSL-StrongPasswordData.csv")
    H = df[[c for c in df.columns if c.startswith("H.")]].to_numpy(float)
    D = df[[c for c in df.columns if c.startswith("DD.")]].to_numpy(float)
    return H, D, df["subject"].to_numpy()


def load_keyrecs_canonical():
    z = np.load(PROC / "typing_canonical.npz")
    m = z["dataset"] == 1
    return z["h"][m], z["d"][m], z["subject"][m]


def cv_of(f):
    return f.std(1) / np.maximum(np.abs(f.mean(1)), 1e-9)


def boot_se(f, rng, B=200):
    N, m = f.shape
    idx = rng.integers(0, m, size=(N, B, m))
    vals = np.take_along_axis(np.repeat(f[:, None, :], B, axis=1), idx, axis=2)
    cvb = vals.std(2) / np.maximum(np.abs(vals.mean(2)), 1e-9)
    return cvb.std(1)


def humanized_flights(h_fit, d_fit, n, rng, N, bounds):
    lh, ld = np.log(h_fit[:, :n]).ravel(), np.log(d_fit[:, : n - 1]).ravel()
    mh, sh, md, sd = lh.mean(), lh.std(), ld.mean(), ld.std()
    out_h, out_d = [], []
    got = 0
    while got < N:
        h = np.exp(rng.normal(mh, sh, size=(2 * N, n)))
        d = np.exp(rng.normal(md, sd, size=(2 * N, n - 1)))
        f = d - h[:, : n - 1]
        ok = ~_bounds_bad(h, d, f, bounds)
        out_h.append(h[ok]); out_d.append(d[ok]); got += int(ok.sum())
    h = np.concatenate(out_h)[:N]; d = np.concatenate(out_d)[:N]
    return d - h[:, : n - 1]


def analyse(name, H, D, subj, n_list, cfg, rng):
    b = cfg["typing"]["bounds_s"]
    rng_sub = np.random.default_rng(7)
    uniq = np.unique(subj)
    half = set(rng_sub.permutation(uniq)[: len(uniq) // 2].tolist())
    fit_m = np.array([s in half for s in subj])
    rows = []
    for n in n_list:
        f_all = D[:, : n - 1] - H[:, : n - 1]
        cv_all = cv_of(f_all)
        se = boot_se(f_all[:4000], rng)
        hum_eval = ~fit_m
        cv_h = cv_all[hum_eval][:4000]
        f_bot = humanized_flights(H[fit_m], D[fit_m], n, rng, len(cv_h), b)
        cv_b = cv_of(f_bot)
        # Family A (naive): fixed intervals with jitter; constants from configs/default.yaml, no data fitted.
        gcfg = cfg["generators"]
        ch = rng.uniform(*gcfg["naive_hold_range"], size=(len(cv_h), 1))
        cd = rng.uniform(*gcfg["naive_dd_range"], size=(len(cv_h), 1))
        hA = ch + rng.normal(0, gcfg["naive_jitter_s"], size=(len(cv_h), n))
        dA = cd + rng.normal(0, gcfg["naive_jitter_s"], size=(len(cv_h), n - 1))
        cv_a = cv_of(dA - hA[:, : n - 1])
        out = {}
        for fam, cv_bot in (("A", cv_a), ("B", cv_b)):
            y = np.r_[np.ones(len(cv_h)), np.zeros(len(cv_bot))]
            s = np.r_[cv_h, cv_bot]
            auc = roc_auc_score(y, s)
            boots = []
            for _ in range(200):
                i = rng.integers(0, len(y), len(y))
                if y[i].min() == y[i].max():
                    continue
                boots.append(roc_auc_score(y[i], s[i]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[fam] = (float(auc), float(lo), float(hi), float(np.median(cv_bot)))
        rows.append({
            "dataset": name, "n_keys": n, "n_flights": n - 1,
            "median_cv_human": float(np.median(cv_h)),
            "median_boot_se_cv": float(np.median(se)),
            "mean_boot_se_cv": float(np.mean(se)),
            "relative_se_median": float(np.median(se / np.maximum(np.median(cv_all), 1e-9))),
            "share_human_cv_in_0.2_1.0": float(np.mean((cv_all >= 0.2) & (cv_all <= 1.0))),
            "auc_cv_human_vs_A": out["A"][0], "auc_ci95_A": [out["A"][1], out["A"][2]],
            "median_cv_A": out["A"][3],
            "auc_cv_human_vs_B": out["B"][0], "auc_ci95_B": [out["B"][1], out["B"][2]],
            "median_cv_B": out["B"][3], "n_human_eval": int(len(cv_h)), "n_bot": int(len(cv_b)),
            "note_B": "B is fitted to human marginals (means/SDs of log hold and log DD on the other half of subjects), so near-0.5 AUC is partly by construction.",
        })
        print(name, n, "A", round(out["A"][0], 3), "B", round(out["B"][0], 3), round(float(np.median(se)), 3), flush=True)
    return rows


def main():
    cfg = load_cfg()
    rng = np.random.default_rng(99)
    with phase_timer("phase4_cv_reliability"):
        H, D, subj = load_cmu_raw()
        rows = analyse("CMU", H, D, subj, range(4, 12), cfg, rng)
        Hk, Dk, sk = load_keyrecs_canonical()
        rows += analyse("KeyRecs-fixed", Hk, Dk, sk, range(4, 11), cfg, rng)
    write_json(RESULTS / "cv_reliability" / "cv_reliability.json", rows)


if __name__ == "__main__":
    main()
