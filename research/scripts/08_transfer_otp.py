"""Phase 6 (extra): cross-dataset typing transfer (CMU <-> KeyRecs-fixed, with and without flight features)
and the 3 x 3 OTP-model sensitivity grid.

Transfer protocol (typing-only detector):
  source subjects are split 60/20/20 (train/cal/in-domain test). Attackers A-D are fitted on source-train
  humans only. Target humans are never used for fitting; they are scored against source-fitted attackers.
OTP sensitivity (post-hoc on the trained full model, folds x seed 0):
  the test OTP latencies are redrawn under the perturbed 'true' world; the detector keeps the nominal LLR.
  logit_new = logit - alpha_otp * LLR_nominal(t_old) + alpha_otp * LLR_nominal(t_new).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import numpy as np  # noqa: E402  (already imported above; kept explicit)
from sklearn.metrics import roc_auc_score  # noqa: E402

from habitid.bots.otp import llr_numpy, sample_latency  # noqa: E402
from habitid.bots.policy import valid_typing  # noqa: E402
from habitid.bots.typing_gen import ARGen, HumanizedGen, NaiveGen, ReplayGen  # noqa: E402
from habitid.calibrate.calibration import p_human  # noqa: E402
from habitid.eval.io import neural_test  # noqa: E402
from habitid.eval.metrics import detection_summary, ece  # noqa: E402
from habitid.stats.bootstrap import bootstrap_indices, percentile_ci  # noqa: E402
from habitid.train.train import predict, tensors_from_split, train_variant  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_otp  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402

DATASET_CODE = {"CMU": 0, "KeyRecs": 1}


def _split_subjects(subjects: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subjects)
    n = len(perm)
    a, b = int(round(0.6 * n)), int(round(0.8 * n))
    return set(perm[:a].tolist()), set(perm[a:b].tolist()), set(perm[b:].tolist())


def _bots(gens, n, rng, bounds, src_pool):
    letters = ["A", "B", "C", "D"]
    h, d, L = [], [], []
    per = n // 4
    for i, letter in enumerate(letters):
        m = per if i < 3 else n - 3 * per
        g = gens[letter]
        hh, dd = [], []
        got = 0
        while got < m:
            a, b_ = g.sample(max(2 * (m - got), 64), rng)
            ok = valid_typing(a, b_, bounds)
            hh.append(a[ok]); dd.append(b_[ok]); got += int(ok.sum())
        h.append(np.concatenate(hh)[:m]); d.append(np.concatenate(dd)[:m]); L += [i + 1] * m
    return np.concatenate(h), np.concatenate(d), np.array(L)


def _fold_dict(h, d, label, letter, cluster, n_dummy=None):
    n = len(h)
    z = {
        "h": h, "d": d, "windows": np.zeros((n, 150, 6), dtype=np.float32), "otp_s": np.full(n, 8.0),
        "label": label, "letter": letter, "type": np.where(label == 1, 0, 1), "cluster": cluster,
        "gate": np.ones(n, dtype=bool), "subject": np.full(n, -1), "dataset": np.full(n, -1),
    }
    return z


def transfer(cfg, src: str, tgt: str, use_flight: bool, seed: int, t_all):
    rng = np.random.default_rng(500 + seed)
    bounds = cfg["typing"]["bounds_s"]
    ds_all = t_all["dataset"]
    subj_all = t_all["subject"]
    src_subj = np.unique(subj_all[ds_all == DATASET_CODE[src]])
    tgt_subj = np.unique(subj_all[ds_all == DATASET_CODE[tgt]])
    s_tr, s_cal, s_te = _split_subjects(src_subj, 11)
    take = lambda subjs: np.isin(subj_all, list(subjs)) & (ds_all == DATASET_CODE[src])  # noqa: E731
    m_tr, m_cal, m_te = take(s_tr), take(s_cal), take(s_te)
    H, D = t_all["h"], t_all["d"]
    hs_tr, ds_tr = H[m_tr], D[m_tr]
    tn_gens = {"A": NaiveGen(cfg), "B": HumanizedGen(cfg).fit(hs_tr, ds_tr),
               "C": ReplayGen(cfg).fit(hs_tr, ds_tr), "D": ARGen(cfg, seed=seed).fit(hs_tr, ds_tr)}
    n_h = cfg["sizes"]["train_h"]
    idx_tr = rng.choice(np.where(m_tr)[0], size=min(n_h, m_tr.sum()), replace=False)
    idx_cal = rng.choice(np.where(m_cal)[0], size=min(cfg["sizes"]["cal_h"], m_cal.sum()), replace=False)
    idx_te = rng.choice(np.where(m_te)[0], size=min(cfg["sizes"]["test_h"], m_te.sum()), replace=False)
    btr = _bots(tn_gens, len(idx_tr), rng, bounds, None)
    bcal = _bots(tn_gens, len(idx_cal), rng, bounds, None)
    bte = _bots(tn_gens, len(idx_te), rng, bounds, None)
    tgt_idx = rng.choice(np.where(ds_all == DATASET_CODE[tgt])[0], size=cfg["sizes"]["test_h"], replace=False)

    def build(idx_h, bots, label_set=True):
        h = np.concatenate([H[idx_h], bots[0]]); d = np.concatenate([D[idx_h], bots[1]])
        lab = np.concatenate([np.ones(len(idx_h), dtype=np.int64), np.zeros(len(bots[0]), dtype=np.int64)])
        let = np.concatenate([np.zeros(len(idx_h), dtype=np.int64), bots[2]])
        clu = np.concatenate([subj_all[idx_h].astype(np.int64), 90_000_000 + np.arange(len(bots[0])) // 50])
        return _fold_dict(h, d, lab, let, clu)

    tr, cal, te_src = build(idx_tr, btr), build(idx_cal, bcal), build(idx_te, bte)
    fold = {}
    for pre, z in (("train", tr), ("cal", cal), ("test", te_src)):
        for k, v in z.items():
            fold[f"{pre}_{k}"] = v
    fold["train_motion_pool"] = np.zeros((10, 150, 6), dtype=np.float32)
    fold["fold_index"] = np.array(0)
    variant = {"active": ["typing"], "lam_prior": cfg["training"]["lambda_prior"],
               "lam_adv": cfg["training"]["lambda_adv"], "use_flight": use_flight}
    otp_cfg = load_otp()
    res = train_variant(fold, cfg, otp_cfg, None, variant, seed)
    model, tnorm, mnorm = res["model"], res["tnorm"], res["mnorm"]
    T = float(res["preds"]["T"])
    out = {}
    tgt_bots = _bots(tn_gens, len(tgt_idx), rng, bounds, None)
    tgt_fold = build(tgt_idx, tgt_bots)
    for name, z in (("in_domain", te_src), ("cross_domain", tgt_fold)):
        td = {f"test_{k}": v for k, v in z.items()}
        td["test_windows"] = td["test_windows"]
        tt = tensors_from_split(td, "test", tnorm, mnorm, use_flight)
        p = predict(model, tt)
        logit = p["logit"]
        y = z["label"]
        bot = 1 - y
        met = detection_summary(bot, -logit, cfg["evaluation"]["base_rates"], cfg["evaluation"]["fpr_target"])
        met["ECE"] = ece(p_human(logit, T), y, cfg["evaluation"]["ece_bins"])
        met["n_human"] = int((y == 1).sum()); met["n_bot"] = int((y == 0).sum())
        idx = bootstrap_indices(z["cluster"], y, 1000, 77 + seed)
        aucs = np.array([roc_auc_score(bot[i], -logit[i]) for i in idx if bot[i].min() != bot[i].max()])
        lo, hi = percentile_ci(aucs)
        met["AUC_CI95"] = [lo, hi]
        out[name] = met
    return out


def transfer_block(cfg):
    t = np.load(PROC / "typing_canonical.npz")
    t_all = {k: t[k] for k in ("h", "d", "subject", "dataset")}
    t_all["subject"] = t_all["subject"]
    res = {}
    for src, tgt in (("CMU", "KeyRecs"), ("KeyRecs", "CMU")):
        for use_flight in (True, False):
            key = f"{src}->{tgt}|flight={'on' if use_flight else 'off'}"
            runs = []
            for seed in cfg["project"]["train_seeds"]:
                runs.append(transfer(cfg, src, tgt, use_flight, seed, t_all))
                print(key, seed, round(runs[-1]["cross_domain"]["AUC"], 3), flush=True)
            res[key] = {}
            for setting in ("in_domain", "cross_domain"):
                vals = {m: [r[setting][m] for r in runs] for m in ("AUC", "EER", "TPR@1%FPR", "PR-AUC@0.01", "PR-AUC@0.001", "ECE")}
                res[key][setting] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for m, v in vals.items()}
    return res


def otp_grid(cfg):
    otp_cfg = load_otp()
    sens = otp_cfg["sensitivity"]
    grid = [(m, w) for m in sens["human_median_mult"] for w in sens["bot_near_weight"]]
    results = {}
    for m_mult, w_near in grid:
        rec = {"AUC": [], "TPR@1%FPR": [], "PR-AUC@0.01": [], "PR-AUC@0.001": []}
        for k in range(cfg["project"]["n_folds"]):
            z = neural_test("full", k, 0)
            rng = np.random.default_rng(900 + k)
            n = len(z["label"])
            otp_new = np.empty(n)
            is_human = z["label"] == 1
            auto_otp = (~is_human) & np.isin(z["type"], [1, 4])  # bots with an automated OTP component
            other_bot = (~is_human) & ~auto_otp
            otp_new[is_human | other_bot] = sample_latency(rng, int((is_human | other_bot).sum()), otp_cfg,
                                                           "human", human_mult=m_mult)
            otp_new[auto_otp] = sample_latency(rng, int(auto_otp.sum()), otp_cfg, "automated",
                                               bot_near_weight=w_near)
            alpha_otp = float(z["alpha"][-1])
            llr_old = llr_numpy(z["otp_s"], otp_cfg)
            llr_new = llr_numpy(otp_new, otp_cfg)
            logit_new = z["logit"] - alpha_otp * llr_old + alpha_otp * llr_new
            v = {"label": z["label"], "letter": z["letter"], "gate": z["gate"], "score": -logit_new}
            g = v["gate"]
            bot = 1 - v["label"][g]
            met = detection_summary(bot, v["score"][g], cfg["evaluation"]["base_rates"], cfg["evaluation"]["fpr_target"])
            rec["AUC"].append(met["AUC"]); rec["TPR@1%FPR"].append(met["TPR@1%FPR"])
            rec["PR-AUC@0.01"].append(met["PR-AUC@0.01"]); rec["PR-AUC@0.001"].append(met["PR-AUC@0.001"])
        results[f"human_mult={m_mult}|bot_near_weight={w_near}"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                                                                    for k, v in rec.items()}
    means = [v["AUC"]["mean"] for v in results.values()]
    tprs = [v["TPR@1%FPR"]["mean"] for v in results.values()]
    summary = {"AUC_range": [float(min(means)), float(max(means))],
               "TPR@1%FPR_range": [float(min(tprs)), float(max(tprs))],
               "nominal_cell": results[f"human_mult=1.0|bot_near_weight=0.7"]}
    return {"grid": results, "summary": summary,
            "note": "detector keeps nominal OTP parameters; the simulated world uses each grid cell; pooled over 5 folds"}


def main():
    """--part transfer | otp | all. Each part writes its own JSON; the paper script merges them."""
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_cfg()
    if part in ("transfer", "all"):
        with phase_timer("phase6_transfer"):
            write_json(RESULTS / "tables" / "transfer.json", {"transfer": transfer_block(cfg)})
    if part in ("otp", "all"):
        with phase_timer("phase6_otp_sensitivity"):
            write_json(RESULTS / "tables" / "otp_sensitivity.json", {"otp_sensitivity": otp_grid(cfg)})
    print("done", part)


if __name__ == "__main__":
    main()
