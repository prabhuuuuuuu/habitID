"""Typing-only-bot diagnosis (QA). Sessions where ONLY typing is automated (motion and OTP human).

For each family letter A-D (type = typing_only): AUC of humans vs that family's typing-only bots, for
  (a) the fused habitID-Net (full variant, seed 0; OTP term included),
  (b) the typing head alone (typing_only variant, seed 0),
  (c) GBDT on hand-crafted typing features only (fitted per fold on the train split, seed 0),
  (d) logistic regression on the same typing features (fitted per fold, seed 0).
Also: learned typing weight alpha (full model), and the distribution of the typing evidence
contribution alpha_typ * g_typ for humans vs typing-only bots (fused model).
Reads results/runs and data/processed/folds only. Writes results/tables/typing_only_diagnosis.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from habitid.features.features import typing_handcrafted  # noqa: E402
from habitid.eval.io import neural_test  # noqa: E402
from habitid.utils.config import PROC, RESULTS  # noqa: E402
from habitid.utils.runtime import write_json  # noqa: E402

FAM = {"A": 1, "B": 2, "C": 3, "D": 4}
TYPING_ONLY = 2


def pooled_neural(variant):
    from habitid.eval.io import run_path
    parts = [neural_test(variant, k, 0) for k in range(5)]
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("score", "label", "letter", "type")}
    out["contrib_typing"] = np.concatenate([np.load(run_path(variant, k, 0))["test_contrib_typing"] for k in range(5)])
    out["alpha"] = parts[0]["alpha"]
    return out


def per_family_auc(score, label, letter, typ):
    out = {}
    for L, code in FAM.items():
        m = (label == 1) | ((typ == TYPING_ONLY) & (letter == code))
        out[L] = float(roc_auc_score(1 - label[m], score[m]))
    return out


def classical_scores():
    """GBDT and LR on typing features only, fitted per fold on the train split (all families), seed 0."""
    s_gbt, s_lr, lab, let, typ = [], [], [], [], []
    for k in range(5):
        f = np.load(PROC / "folds" / f"fold{k}.npz")
        Xtr = typing_handcrafted(f["train_h"], f["train_d"])
        Xte = typing_handcrafted(f["test_h"], f["test_d"])
        y = 1 - f["train_label"]
        gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr, y)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, y)
        s_gbt.append(gbt.predict_proba(Xte)[:, 1]); s_lr.append(lr.predict_proba(Xte)[:, 1])
        lab.append(f["test_label"]); let.append(f["test_letter"]); typ.append(f["test_type"])
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    return cat(s_gbt), cat(s_lr), cat(lab), cat(let), cat(typ)


def main():
    out = {"seed": 0, "note": "bot score = automated probability / -logit; AUC humans vs typing-only bots of one family"}
    full = pooled_neural("full")
    typo = pooled_neural("typing_only")
    gbt, lr, lab, let, typ = classical_scores()
    out["a_fused_habitID_net_with_OTP"] = per_family_auc(full["score"], full["label"], full["letter"], full["type"])
    out["b_typing_head_only"] = per_family_auc(typo["score"], typo["label"], typo["letter"], typo["type"])
    out["c_gbdt_typing_features_only"] = per_family_auc(gbt, lab, let, typ)
    out["d_lr_typing_features_only"] = per_family_auc(lr, lab, let, typ)
    out["alpha_typing_full"] = float(full["alpha"][0])
    out["alpha_typing_typing_only_model"] = float(typo["alpha"][0])
    contrib = full["contrib_typing"]
    dist = {}
    for name, m in (("human", full["label"] == 1), ("typing_only_bot", (full["label"] == 0) & (full["type"] == TYPING_ONLY))):
        q = np.quantile(contrib[m], [0.1, 0.25, 0.5, 0.75, 0.9])
        dist[name] = {"q10": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "q90": float(q[4])}
    out["typing_evidence_contribution_fused"] = dist
    out["typing_only_bot_share_of_bots"] = float(((full["label"] == 0) & (full["type"] == TYPING_ONLY)).sum() / (full["label"] == 0).sum())
    write_json(RESULTS / "tables" / "typing_only_diagnosis.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main()
