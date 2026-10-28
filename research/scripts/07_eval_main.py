"""Phase 6 (main): metrics, subject-cluster bootstrap CIs, paired tests (Holm), LOFO, conformal, calibration,
learned weights, and ablations. Writes results/tables/*.json. All numbers come from results/runs/.

Gate policy: sessions failing the validity gate are rejected before scoring for every method
(the gate-only baseline is scored on all sessions, since it is the gate).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from habitid.calibrate.calibration import p_human  # noqa: E402
from habitid.conformal.conformal import class_thresholds, prediction_sets, summarize, verdicts  # noqa: E402
from habitid.eval.io import baseline_test, gate_filter, neural_test, pooled, run_meta  # noqa: E402
from habitid.eval.metrics import detection_summary, ece, reliability  # noqa: E402
from habitid.stats.bootstrap import bootstrap_indices, holm, percentile_ci  # noqa: E402
from habitid.utils.config import RESULTS, load_cfg  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

LETTERS = {1: "A", 2: "B", 3: "C", 4: "D", 5: "O", 6: "D*"}
SEEDS = [0, 1, 2]
FOLDS = range(5)


def view_for(method: str, k: int, s: int, split: str = "test"):
    if method in ("habitid_net", "noprior", "noadv", "typing_only", "motion_only", "otp_only", "no_typing",
                  "no_motion", "no_otp", "lam10", "noadv_seed0", "noprior_seed0"):
        variant = {"habitid_net": "full", "noadv_seed0": "noadv", "noprior_seed0": "noprior"}.get(method, method)
        return neural_test(variant, k, s, split=split)
    if method.startswith("lofo_"):
        return neural_test(method, k, s, split=split)
    return baseline_test(method, k, s)


def gated(v):
    return gate_filter(v)


def metrics_fold(view, cfg, with_ece=True):
    v = gated(view)
    bot = 1 - v["label"]
    out = detection_summary(bot, v["score"], cfg["evaluation"]["base_rates"], cfg["evaluation"]["fpr_target"])
    if with_ece and "p_h_cal" in v:
        out["ECE"] = ece(v["p_h_cal"], v["label"], cfg["evaluation"]["ece_bins"])
        out["ECE_uncalibrated"] = ece(v["p_h_raw"], v["label"], cfg["evaluation"]["ece_bins"])
    return out


def mean_std(vals):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def main_table(cfg):
    methods = {
        "habitid_v0": ("base", [0]), "gate_only": ("base", [0]), "lr_handcrafted": ("base", [0]),
        "gbt_handcrafted": ("base", SEEDS), "isolation_forest": ("base", SEEDS),
        "one_class_svm": ("base", SEEDS), "scaled_manhattan": ("base", [0]),
        "noprior": ("neural", SEEDS), "noadv": ("neural", SEEDS), "habitid_net": ("neural", SEEDS),
        "typing_only": ("neural", SEEDS), "motion_only": ("neural", SEEDS), "otp_only": ("neural", SEEDS),
    }
    rows = {}
    for name, (kind, seeds) in methods.items():
        per = {m: [] for m in ("AUC", "EER", "TPR@1%FPR", "PR-AUC@1", "PR-AUC@0.1", "ECE", "ECE_uncalibrated")}
        for s in seeds:
            for k in FOLDS:
                v = view_for(name, k, s)
                if name == "gate_only":
                    v = gate_filter(v, keep_all=True)
                v = with_probs(v, name)
                m = metrics_fold(v, cfg, with_ece=(kind == "neural" or name in PROB_BASELINES))
                per["AUC"].append(m["AUC"]); per["EER"].append(m["EER"])
                per["TPR@1%FPR"].append(m["TPR@1%FPR"])
                per["PR-AUC@1"].append(m["PR-AUC@0.01"]); per["PR-AUC@0.1"].append(m["PR-AUC@0.001"])
                per["ECE"].append(m.get("ECE")); per["ECE_uncalibrated"].append(m.get("ECE_uncalibrated"))
        rows[name] = {key: mean_std(vals) for key, vals in per.items()}
        rows[name]["n_runs"] = len(seeds) * len(list(FOLDS))
        print(name, {k: round(v["mean"], 3) for k, v in rows[name].items() if isinstance(v, dict)}, flush=True)
    return rows


def pooled_seed0(method: str, seed: int = 0):
    v = pooled(lambda fold, seed, **kw: view_for(method, fold, seed), seed)
    return v


def bootstrap_block(cfg, methods_for_ci):
    """Cluster bootstrap on seed-0 pooled test sessions. Same resample indices for all methods (paired)."""
    base = gated(pooled_seed0("habitid_net"))
    n_res = cfg["evaluation"]["bootstrap_resamples"]
    idx_list = bootstrap_indices(base["cluster"], base["label"], n_res, cfg["evaluation"]["bootstrap_seed"])
    views = {m: gated(pooled_seed0(m)) for m in methods_for_ci}
    for m, vv in views.items():
        assert np.array_equal(vv["label"], base["label"]) and np.array_equal(vv["cluster"], base["cluster"]), m
    out, auc_boot = {}, {}
    for m, vv in views.items():
        bot = 1 - vv["label"]
        s = vv["score"]
        aucs = np.array([roc_auc_score(bot[i], s[i]) if bot[i].min() != bot[i].max() else np.nan for i in idx_list])
        auc_boot[m] = aucs
        lo, hi = percentile_ci(aucs)
        out[m] = {"AUC": float(roc_auc_score(bot, s)), "AUC_CI95": [lo, hi]}
    comparisons = {"habitid_net - habitid_v0": ("habitid_net", "habitid_v0"),
                   "habitid_net - noprior": ("habitid_net", "noprior"),
                   "habitid_net - noadv": ("habitid_net", "noadv"),
                   "habitid_net - gbt_handcrafted": ("habitid_net", "gbt_handcrafted"),
                   "habitid_net - lr_handcrafted": ("habitid_net", "lr_handcrafted")}
    diffs, pvals = {}, {}
    for name, (a, b) in comparisons.items():
        d = auc_boot[a] - auc_boot[b]
        d = d[~np.isnan(d)]
        lo, hi = percentile_ci(d)
        est = out[a]["AUC"] - out[b]["AUC"]
        p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
        diffs[name] = {"delta_AUC": float(est), "CI95": [lo, hi], "p_raw": p}
        pvals[name] = p
    adj = holm(pvals)
    for name in diffs:
        diffs[name]["p_holm"] = float(adj[name])
    return {"AUC_seed0_pooled": out, "paired_differences": diffs, "n_resamples": n_res,
            "clusters": {"human_subjects": int(len(np.unique(base["cluster"][base["label"] == 1]))),
                         "bot_clusters": int(len(np.unique(base["cluster"][base["label"] == 0])))}}


def per_letter_auc(view):
    v = gated(view)
    out = {}
    for code, name in LETTERS.items():
        m = (v["label"] == 0) & (v["letter"] == code)
        if m.sum() == 0:
            continue
        sel = m | (v["label"] == 1)
        out[name] = float(roc_auc_score(1 - v["label"][sel], v["score"][sel]))
    return out


def lofo_block(cfg):
    res = {}
    for variant in ("lofo_full", "lofo_noadv"):
        per_L = {}
        for L in "ABCD":
            name = f"{variant}_{L}"
            aucs, tprs, dets = [], [], []
            for s in [0]:
                v = gated(pooled(lambda fold, seed, **kw: neural_test(name, fold, seed), s))
                sel = (v["label"] == 1) | (v["letter"] == {"A": 1, "B": 2, "C": 3, "D": 4}[L])
                bot = 1 - v["label"][sel]
                sc = v["score"][sel]
                d = detection_summary(bot, sc, cfg["evaluation"]["base_rates"], cfg["evaluation"]["fpr_target"])
                aucs.append(d["AUC"]); tprs.append(d["TPR@1%FPR"])
                dets.append({k: round(float(x), 4) for k, x in d.items()})
            per_L[L] = {"AUC": float(np.mean(aucs)), "TPR@1%FPR": float(np.mean(tprs)), "detail": dets[0],
                        "per_letter_all_eval": per_letter_auc(pooled(lambda fold, seed, **kw: neural_test(name, fold, seed), 0))}
        worst = min(per_L, key=lambda L: per_L[L]["AUC"])
        res[variant] = {"per_family": per_L, "worst_family": worst,
                        "worst_AUC": per_L[worst]["AUC"], "worst_TPR@1%FPR": per_L[worst]["TPR@1%FPR"]}
        print(variant, {L: round(per_L[L]["AUC"], 3) for L in per_L}, flush=True)
    return res


def lofo_stats(cfg):
    """LOFO with seed-0 pooled sessions: the AUC of humans vs the held-out family only."""
    return lofo_block(cfg)


def conformal_block(cfg, variant_prefix, excluded=None, seeds=SEEDS):
    out = {}
    for eps in cfg["evaluation"]["conformal_eps"]:
        acc = {"coverage_human": [], "coverage_bot": [], "abstention": [], "verified_rate_human": [],
               "suspicious_rate_bot": []}
        by_letter = {}
        for s in seeds:
            for k in FOLDS:
                name = variant_prefix
                cal = gated(neural_test(name, k, s, split="cal"))
                te = gated(neural_test(name, k, s, split="test"))
                q = class_thresholds(cal["p_h_cal"], cal["label"], eps)
                in_h, in_b = prediction_sets(te["p_h_cal"], q)
                sm = summarize(in_h, in_b, te["label"], te["letter"])
                for key in acc:
                    acc[key].append(sm[key])
                for L, c in sm["coverage_bot_by_letter"].items():
                    by_letter.setdefault(LETTERS[L], []).append(c)
        out[str(eps)] = {"nominal": 1 - eps, **{k: mean_std(v) for k, v in acc.items()},
                         "coverage_bot_by_letter": {L: mean_std(v) for L, v in by_letter.items()}}
    return out


def conformal_lofo(cfg):
    out = {}
    for L in "ABCD":
        name = f"lofo_full_{L}"
        per = {}
        for eps in cfg["evaluation"]["conformal_eps"]:
            cov_held, cov_h, abst = [], [], []
            for k in FOLDS:
                cal = gated(neural_test(name, k, 0, split="cal"))
                te = gated(neural_test(name, k, 0, split="test"))
                q = class_thresholds(cal["p_h_cal"], cal["label"], eps)
                in_h, in_b = prediction_sets(te["p_h_cal"], q)
                held = (te["label"] == 0) & (te["letter"] == {"A": 1, "B": 2, "C": 3, "D": 4}[L])
                cov_held.append(in_b[held]); cov_h.append(in_h[te["label"] == 1]); abst.append(in_h == in_b)
            per[str(eps)] = {"nominal": 1 - eps,
                             "coverage_held_out_family_bots": float(np.concatenate(cov_held).mean()),
                             "coverage_human": float(np.concatenate(cov_h).mean()),
                             "abstention": float(np.concatenate(abst).mean())}
        out[L] = per
    return out


def reliability_block(cfg):
    out = {}
    for name, variant in [("habitID-Net", "full"), ("no prior", "noprior"), ("no adv.", "noadv")]:
        vals = {"ECE_uncal": [], "ECE_TS": [], "T": []}
        for s in SEEDS:
            pr_u, pr_c, ys = [], [], []
            for k in FOLDS:
                v = gated(neural_test(variant, k, s))
                pr_u.append(v["p_h_raw"]); pr_c.append(v["p_h_cal"]); ys.append(v["label"])
                vals["T"].append(v["T"])
            pu, pc, y = np.concatenate(pr_u), np.concatenate(pr_c), np.concatenate(ys)
            vals["ECE_uncal"].append(ece(pu, y, cfg["evaluation"]["ece_bins"]))
            vals["ECE_TS"].append(ece(pc, y, cfg["evaluation"]["ece_bins"]))
        out[name] = {k: mean_std(v) for k, v in vals.items()}
        if variant == "full":
            pu = np.concatenate([gated(neural_test("full", k, 0))["p_h_raw"] for k in FOLDS])
            pc = np.concatenate([gated(neural_test("full", k, 0))["p_h_cal"] for k in FOLDS])
            y = np.concatenate([gated(neural_test("full", k, 0))["label"] for k in FOLDS])
            out["reliability_full_seed0"] = {
                "uncalibrated": {"conf": reliability(pu, y, cfg["evaluation"]["ece_bins"])[0].tolist(),
                                 "acc": reliability(pu, y, cfg["evaluation"]["ece_bins"])[1].tolist(),
                                 "count": reliability(pu, y, cfg["evaluation"]["ece_bins"])[2].tolist()},
                "temperature_scaled": {"conf": reliability(pc, y, cfg["evaluation"]["ece_bins"])[0].tolist(),
                                       "acc": reliability(pc, y, cfg["evaluation"]["ece_bins"])[1].tolist(),
                                       "count": reliability(pc, y, cfg["evaluation"]["ece_bins"])[2].tolist()}}
    return out


def weights_block():
    out = {}
    for name in ["full", "noprior", "lam10", "noadv", "typing_only", "motion_only", "otp_only",
                 "no_typing", "no_motion", "no_otp", "lofo_full_A", "lofo_full_B", "lofo_full_C", "lofo_full_D"]:
        rec = []
        for k in FOLDS:
            for s in [0, 1, 2] if name in ("full", "noprior", "noadv", "typing_only", "motion_only", "otp_only") else [0]:
                p = RESULTS / "runs" / name / f"fold{k}_seed{s}.json"
                if not p.exists():
                    continue
                m = run_meta(name, k, s)
                rec.append({"fold": k, "seed": s, "alpha": m["alpha"], "alpha0": m["alpha0"],
                            "prior_distance": m["prior_distance"], "lam_prior": m["lam_prior"],
                            "active": m["active"], "T": m["T"]})
        if not rec:
            continue
        A = np.array([r["alpha"] for r in rec]) if rec else None
        out[name] = {"active": rec[0]["active"], "lam_prior": rec[0]["lam_prior"],
                     "alpha0": rec[0]["alpha0"], "alpha_mean": np.mean([r["alpha"] for r in rec], axis=0).tolist(),
                     "alpha_std": np.std([r["alpha"] for r in rec], axis=0, ddof=0).tolist(),
                     "prior_distance_mean": float(np.mean([r["prior_distance"] for r in rec])),
                     "prior_distance_std": float(np.std([r["prior_distance"] for r in rec], ddof=0)),
                     "T_mean": float(np.mean([r["T"] for r in rec])), "n": len(rec)}
    return out


def ablation_block(cfg):
    res = {}
    base = {}
    for name, variant in [("full (seed 0)", "full"), ("no prior (lambda=0)", "noprior"),
                          ("lambda=10", "lam10"), ("no adversarial training", "noadv"),
                          ("no typing encoder", "no_typing"), ("no motion encoder", "no_motion"),
                          ("no OTP term", "no_otp"), ("typing only", "typing_only"),
                          ("motion only", "motion_only"), ("OTP only", "otp_only")]:
        per_letter = per_letter_auc(pooled(lambda fold, seed, **kw: neural_test(variant, fold, seed), 0))
        worst_in = min(per_letter.values())
        base[name] = {"per_letter_AUC": per_letter, "worst_letter_AUC_in_dist": float(worst_in),
                      "worst_letter_in_dist": min(per_letter, key=per_letter.get)}
        print(name, round(worst_in, 3), flush=True)
    # LOFO worst case (true held-out) for full and no-adversarial
    lf = lofo_block(cfg)
    res["in_distribution_worst_letter"] = base
    res["lofo_worst_case"] = {"full": lf["lofo_full"]["worst_AUC"], "noadv": lf["lofo_noadv"]["worst_AUC"],
                              "full_worst_family": lf["lofo_full"]["worst_family"],
                              "noadv_worst_family": lf["lofo_noadv"]["worst_family"]}
    # no generator-D augmentation: model trained without family D, evaluated on all letters
    pl = per_letter_auc(pooled(lambda fold, seed, **kw: neural_test("lofo_full_D", fold, seed), 0))
    res["no_generatorD_augmentation"] = {"per_letter_AUC": pl, "worst_letter_AUC": float(min(pl.values())),
                                         "worst_letter": min(pl, key=pl.get)}
    # no calibration: AUC unchanged by monotone TS; report ECE; no conformal: fixed p>=0.5 verdict
    fv = gated(pooled(lambda fold, seed, **kw: neural_test("full", fold, seed), 0))
    pred_bot = fv["p_h_raw"] < 0.5
    res["fixed_threshold_verdict_p0.5"] = {
        "bot_detection_rate": float(pred_bot[fv["label"] == 0].mean()),
        "human_false_flag_rate": float(pred_bot[fv["label"] == 1].mean()),
    }
    return res


def verdict_block(cfg):
    out = {}
    for eps in [0.1]:
        counts = {"human": {"VERIFIED": [], "REVIEW": [], "SUSPICIOUS": []}, "bot": {"VERIFIED": [], "REVIEW": [], "SUSPICIOUS": []}}
        for s in SEEDS:
            for k in FOLDS:
                cal = gated(neural_test("full", k, s, split="cal"))
                te = gated(neural_test("full", k, s, split="test"))
                q = class_thresholds(cal["p_h_cal"], cal["label"], eps)
                in_h, in_b = prediction_sets(te["p_h_cal"], q)
                vv = verdicts(in_h, in_b)
                for cls, m in (("human", te["label"] == 1), ("bot", te["label"] == 0)):
                    for name in ("VERIFIED", "REVIEW", "SUSPICIOUS"):
                        counts[cls][name].append(float((vv[m] == name).mean()))
        out[str(eps)] = {cls: {k: mean_std(v) for k, v in d.items()} for cls, d in counts.items()}
    return out


def v0_verdicts(cfg):
    """habitID-v0 verdict rates (gate-passing sessions, per fold, seed 0)."""
    out = {}
    for cls, lab in (("human", 1), ("bot", 0)):
        for name in ("VERIFIED", "REVIEW", "SUSPICIOUS"):
            vals = []
            for k in FOLDS:
                z = np.load(RESULTS / "runs" / "baselines" / "habitid_v0" / f"fold{k}_seed0.npz")
                m = (z["test_label"] == lab) & z["test_gate"].astype(bool)
                vals.append(float((z["test_verdict"][m] == name).mean()))
            out.setdefault(cls, {})[name] = mean_std(vals)
    return out


def baseline_surprises(cfg):
    """Five-fold pooled AUC with subject-cluster bootstrap CI and per-family AUC for the baselines.

    Score direction is fixed a priori (higher = more bot) for every method; no sign is flipped post hoc."""
    out = {}
    for m in ["habitid_v0", "lr_handcrafted", "gbt_handcrafted", "isolation_forest", "one_class_svm",
              "scaled_manhattan"]:
        v = gated(pooled(lambda fold, seed, **kw: baseline_test(m, fold, seed), 0))
        bot = 1 - v["label"]
        idx = bootstrap_indices(v["cluster"], v["label"], cfg["evaluation"]["bootstrap_resamples"],
                                cfg["evaluation"]["bootstrap_seed"])
        aucs = np.array([roc_auc_score(bot[i], v["score"][i]) for i in idx])
        lo, hi = percentile_ci(aucs)
        out[m] = {"AUC_pooled_5_folds": float(roc_auc_score(bot, v["score"])), "AUC_CI95": [lo, hi],
                  "per_family_AUC_vs_human": per_letter_auc(pooled(lambda fold, seed, **kw: baseline_test(m, fold, seed), 0))}
        print(m, round(out[m]["AUC_pooled_5_folds"], 3), flush=True)
    return out


PROB_BASELINES = {"gbt_handcrafted", "lr_handcrafted", "gbt_no_otp", "lr_no_otp"}


def with_probs(view, name):
    """Probabilistic baselines: predicted bot probability -> human probability for ECE."""
    if name in PROB_BASELINES:
        p = 1.0 - view["score"]
        view["p_h_cal"] = p
        view["p_h_raw"] = p
    return view


def per_type_block(cfg):
    """AUC of humans vs each attacker type. Types: 1 full (typing+motion+OTP automated), 2 typing-only,
    3 motion-only, 4 OTP-only (by construction). Pooled AUC is capped by the share of single-signal bots."""
    out = {}
    for v in ["full", "noprior", "noadv", "typing_only", "motion_only", "otp_only", "no_otp"]:
        seeds = [0] if v == "no_otp" else SEEDS
        per_seed = []
        for s in seeds:
            z = gated(pooled(lambda fold, seed, **kw: neural_test(v, fold, seed), s))
            lab, ty = z["label"], z["type"]
            rec = {}
            for t, nm in [(1, "fully_automated"), (2, "typing_bot"), (3, "motion_bot"), (4, "otp_bot_by_construction")]:
                m = (lab == 1) | (ty == t)
                if (ty == t).any():
                    rec[nm] = float(roc_auc_score(1 - lab[m], z["score"][m]))
            bots = lab == 0
            rec["share_single_signal_bots"] = float(((ty >= 2) & (ty <= 4) & bots).sum() / bots.sum())
            per_seed.append(rec)
        out[v] = {k: mean_std([r[k] for r in per_seed if k in r]) for k in per_seed[0]}
        out[v]["n_seeds"] = len(seeds)
    return out


def no_otp_block(cfg):
    """Primary scientific comparison: typing + motion only (no OTP term, no OTP-based bot sessions)."""
    methods = [("neural", "no_otp", [0]), ("neural", "motion_only", SEEDS), ("neural", "typing_only", SEEDS),
               ("base", "lr_no_otp", [0]), ("base", "gbt_no_otp", SEEDS), ("base", "habitid_v0_no_otp", [0])]
    rows = {}
    for kind, name, seeds in methods:
        per = {m: [] for m in ("AUC", "EER", "TPR@1%FPR", "PR-AUC@1", "PR-AUC@0.1", "ECE")}
        for s in seeds:
            for k in FOLDS:
                v = neural_test(name, k, s) if kind == "neural" else baseline_test(name, k, s, root="baselines_no_otp")
                v = with_probs(v, name)
                m = metrics_fold(v, cfg, with_ece=(kind == "neural" or name in PROB_BASELINES))
                per["AUC"].append(m["AUC"]); per["EER"].append(m["EER"]); per["TPR@1%FPR"].append(m["TPR@1%FPR"])
                per["PR-AUC@1"].append(m["PR-AUC@0.01"]); per["PR-AUC@0.1"].append(m["PR-AUC@0.001"])
                per["ECE"].append(m.get("ECE"))
        rows[name] = {key: mean_std(vals) for key, vals in per.items()}
        rows[name]["n_runs"] = len(seeds) * len(list(FOLDS))
        print("no-OTP", name, round(rows[name]["AUC"]["mean"], 3), flush=True)
    return rows


def framing_block(cfg, lofo_tables):
    """GBDT vs habitID-Net on LOFO (GBDT refit without the held-out family) and calibration."""
    out = {"habitID_full": {}, "gbdt": {}}
    fam_codes = {"A": 1, "B": 2, "C": 3, "D": 4}
    for L in "ABCD":
        v = gated(pooled(lambda fold, seed, **kw: baseline_test(f"gbt_lofo_{L}", fold, seed, root="baselines_lofo"), 0))
        sel = (v["label"] == 1) | (v["letter"] == fam_codes[L])
        out["gbdt"][L] = float(roc_auc_score(1 - v["label"][sel], v["score"][sel]))
    out["habitID_full"] = {L: lofo_tables["lofo_full"]["per_family"][L]["AUC"] for L in "ABCD"}
    out["worst_habitID_full"] = float(min(out["habitID_full"].values()))
    out["worst_gbdt"] = float(min(out["gbdt"].values()))
    out["worst_family_gbdt"] = min(out["gbdt"], key=out["gbdt"].get)
    out["worst_family_habitID_full"] = min(out["habitID_full"], key=out["habitID_full"].get)
    return out


def main():
    cfg = load_cfg()
    with phase_timer("phase6_eval_main"):
        tables = {}
        tables["baseline_surprises"] = baseline_surprises(cfg)
        tables["main"] = main_table(cfg)
        tables["per_type"] = per_type_block(cfg)
        tables["no_otp"] = no_otp_block(cfg)
        tables["bootstrap"] = bootstrap_block(cfg, ["habitid_net", "noprior", "noadv", "habitid_v0",
                                                    "gbt_handcrafted", "lr_handcrafted"])
        tables["lofo"] = lofo_block(cfg)
        tables["framing_lofo"] = framing_block(cfg, tables["lofo"])
        tables["conformal_in_dist_full"] = conformal_block(cfg, "full")
        tables["conformal_lofo"] = conformal_lofo(cfg)
        tables["calibration"] = reliability_block(cfg)
        tables["verdicts_full_eps0.1"] = verdict_block(cfg)
        tables["weights"] = weights_block()
        tables["ablations"] = ablation_block(cfg)
        tables["v0_verdicts"] = v0_verdicts(cfg)
        tables["dstar"] = dstar_block()
    write_json(RESULTS / "tables" / "main_eval.json", tables)
    print("wrote results/tables/main_eval.json")


def dstar_block():
    out = {}
    for name in ("full", "lofo_full_D"):
        vals, auc_list = [], []
        for s in [0]:
            for k in FOLDS:
                te = gated(neural_test(name, k, s))
                st = gated(neural_test(name, k, s, star=True))
                if len(st["label"]) == 0:
                    continue
                hum = te["label"] == 1
                score = np.concatenate([te["score"][hum], st["score"]])
                bot = np.concatenate([np.zeros(hum.sum()), np.ones(len(st["label"]))])
                auc_list.append(float(roc_auc_score(bot, score)))
                vals.append(float((st["p_h_cal"] < 0.5).mean()))
        out[name] = {"AUC_humans_vs_Dstar": mean_std(auc_list), "Dstar_flag_rate_p0.5": mean_std(vals)}
    return out


if __name__ == "__main__":
    main()
