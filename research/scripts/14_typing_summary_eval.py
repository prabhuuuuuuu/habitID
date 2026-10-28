"""Post-hoc exploratory variant typing_summary_full vs the original full model and GBDT (seed 0, five folds).

Reports, with no selection on test metrics: per-family typing-only-bot AUC (A-D); pooled AUC with the OTP term
(subject-cluster bootstrap CI, 2000 resamples, same resampling as scripts/07); ECE (temperature-scaled, from the
calibration split); typing-evidence contribution medians for humans vs typing-only bots.
The without-OTP version of the variant was not trained (not requested); it is reported as not run.
Writes results/tables/typing_summary_exploratory.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from habitid.eval.io import baseline_test, neural_test, run_path  # noqa: E402
from habitid.eval.metrics import ece  # noqa: E402
from habitid.stats.bootstrap import bootstrap_indices, percentile_ci  # noqa: E402
from habitid.utils.config import load_cfg  # noqa: E402
from habitid.utils.runtime import write_json  # noqa: E402

FAM = {"A": 1, "B": 2, "C": 3, "D": 4}
TYPING_ONLY = 2


def pooled(variant):
    parts = [neural_test(variant, k, 0) for k in range(5)]
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("score", "label", "letter", "type", "p_h_cal", "cluster")}
    out["contrib_typing"] = np.concatenate([np.load(run_path(variant, k, 0))["test_contrib_typing"] for k in range(5)])
    return out


def gbdt_pooled():
    parts = [baseline_test("gbt_handcrafted", k, 0) for k in range(5)]
    return {key: np.concatenate([p[key] for p in parts]) for key in ("score", "label", "letter", "type", "cluster")}


def per_family_typing_only(score, label, letter, typ):
    out = {}
    for L, code in FAM.items():
        m = (label == 1) | ((typ == TYPING_ONLY) & (letter == code))
        out[L] = float(roc_auc_score(1 - label[m], score[m]))
    return out


def summarize(v, cfg):
    bot = 1 - v["label"]
    idx = bootstrap_indices(v["cluster"], v["label"], cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["bootstrap_seed"])
    aucs = np.array([roc_auc_score(bot[i], v["score"][i]) for i in idx])
    lo, hi = percentile_ci(aucs)
    res = {"pooled_AUC_with_OTP": float(roc_auc_score(bot, v["score"])), "pooled_AUC_CI95": [lo, hi],
           "per_family_typing_only_AUC": per_family_typing_only(v["score"], v["label"], v["letter"], v["type"])}
    if "p_h_cal" in v:
        res["ECE_temperature_scaled"] = float(ece(v["p_h_cal"], v["label"], cfg["evaluation"]["ece_bins"]))
    return res


def main():
    cfg = load_cfg()
    out = {"seed": 0, "folds": 5, "status": "post-hoc exploratory; not the main model; not tuned on test metrics"}
    exp = pooled("typing_summary_full")
    orig = pooled("full")
    gb = gbdt_pooled()
    out["typing_summary_full"] = summarize(exp, cfg)
    out["typing_summary_full"]["typing_evidence_median_human"] = float(np.median(exp["contrib_typing"][exp["label"] == 1]))
    out["typing_summary_full"]["typing_evidence_median_typing_only_bot"] = float(
        np.median(exp["contrib_typing"][(exp["label"] == 0) & (exp["type"] == TYPING_ONLY)]))
    out["typing_summary_full"]["without_OTP"] = "not trained (outside requested scope)"
    out["original_full"] = summarize(orig, cfg)
    out["original_full"]["typing_evidence_median_human"] = float(np.median(orig["contrib_typing"][orig["label"] == 1]))
    out["original_full"]["typing_evidence_median_typing_only_bot"] = float(
        np.median(orig["contrib_typing"][(orig["label"] == 0) & (orig["type"] == TYPING_ONLY)]))
    out["gbdt_all_features"] = summarize(gb, cfg)
    out["gbdt_all_features"]["ECE_probabilities_as_output"] = float(ece(1 - gb["score"], gb["label"], cfg["evaluation"]["ece_bins"]))
    out["gbdt_all_features"]["note"] = "GBDT ECE uses its predicted probabilities as output (no temperature scaling), as in the main table"
    write_json(Path(__file__).resolve().parents[1] / "results" / "tables" / "typing_summary_exploratory.json", out)
    print({k: (v.get("pooled_AUC_with_OTP") if isinstance(v, dict) else v) for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
