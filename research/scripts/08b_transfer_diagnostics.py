"""Diagnosis of the cross-dataset typing transfer (CMU <-> KeyRecs-fixed). Does NOT alter the detector.

For each direction (source -> target) and mode, one typing-only detector is trained on the source
training split (seed 0), identical to scripts/08_transfer_otp.py, and scored on:
  in_domain  : held-out SOURCE subjects, source-fitted attackers A-D (control: should be well above 0.5)
  cross      : TARGET humans (never used for fitting) vs source-fitted attackers A-D
Modes:
  on   : features with flight (standard, source normalization statistics)   [same as main transfer run]
  off  : flight features zeroed (sensitivity: D1 run)
  pds  : per-dataset standardization. The target pool (target humans + source-fitted bots, UNLABELLED,
         1:1 mix) defines its own normalization statistics; in_domain keeps source statistics.
Positive class = human (label 1). Bot score = -logit (higher = more bot).
Writes results/transfer_diag/<src>_to_<tgt>_<mode>.npz and results/tables/transfer_diag.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from habitid.bots.typing_gen import ARGen, HumanizedGen, NaiveGen, ReplayGen  # noqa: E402
from habitid.bots.policy import valid_typing  # noqa: E402
from habitid.eval.metrics import detection_summary  # noqa: E402
from habitid.features.features import TypingNorm  # noqa: E402
from habitid.train.train import predict, tensors_from_split, train_variant  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_otp  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("t08", Path(__file__).resolve().parent / "08_transfer_otp.py")
t08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t08)  # reuses _split_subjects, _bots, _fold_dict, DATASET_CODE


def run(cfg, src, tgt, mode, seed=0):
    rng = np.random.default_rng(500 + seed)
    bounds = cfg["typing"]["bounds_s"]
    t = np.load(PROC / "typing_canonical.npz")
    H, D, subj_all, ds_all = t["h"], t["d"], t["subject"], t["dataset"]
    use_flight = mode != "off"
    src_subj = np.unique(subj_all[ds_all == t08.DATASET_CODE[src]])
    s_tr, s_cal, s_te = t08._split_subjects(src_subj, 11)
    take = lambda subjs: np.isin(subj_all, list(subjs)) & (ds_all == t08.DATASET_CODE[src])  # noqa: E731
    m_tr, m_cal, m_te = take(s_tr), take(s_cal), take(s_te)
    hs, dsrc = H[m_tr], D[m_tr]
    gens = {"A": NaiveGen(cfg), "B": HumanizedGen(cfg).fit(hs, dsrc), "C": ReplayGen(cfg).fit(hs, dsrc),
            "D": ARGen(cfg, seed=seed).fit(hs, dsrc)}
    n_h = cfg["sizes"]["train_h"]
    idx_tr = rng.choice(np.where(m_tr)[0], size=min(n_h, m_tr.sum()), replace=False)
    idx_cal = rng.choice(np.where(m_cal)[0], size=min(cfg["sizes"]["cal_h"], m_cal.sum()), replace=False)
    idx_te = rng.choice(np.where(m_te)[0], size=min(cfg["sizes"]["test_h"], m_te.sum()), replace=False)
    tgt_idx = rng.choice(np.where(ds_all == t08.DATASET_CODE[tgt])[0], size=cfg["sizes"]["test_h"], replace=False)
    btr = t08._bots(gens, len(idx_tr), rng, bounds, None)
    bcal = t08._bots(gens, len(idx_cal), rng, bounds, None)
    bte = t08._bots(gens, len(idx_te), rng, bounds, None)
    btgt = t08._bots(gens, len(tgt_idx), rng, bounds, None)

    def build(idx_h, bots):
        h = np.concatenate([H[idx_h], bots[0]]); d = np.concatenate([D[idx_h], bots[1]])
        lab = np.concatenate([np.ones(len(idx_h), dtype=np.int64), np.zeros(len(bots[0]), dtype=np.int64)])
        let = np.concatenate([np.zeros(len(idx_h), dtype=np.int64), bots[2]])
        clu = np.concatenate([subj_all[idx_h].astype(np.int64), 90_000_000 + np.arange(len(bots[0])) // 50])
        return t08._fold_dict(h, d, lab, let, clu)

    tr, cal, te_src = build(idx_tr, btr), build(idx_cal, bcal), build(idx_te, bte)
    tgt_fold = build(tgt_idx, btgt)
    fold = {}
    for pre, z in (("train", tr), ("cal", cal), ("test", te_src)):
        for k, v in z.items():
            fold[f"{pre}_{k}"] = v
    fold["train_motion_pool"] = np.zeros((10, 150, 6), dtype=np.float32)
    fold["fold_index"] = np.array(0)
    variant = {"active": ["typing"], "lam_prior": cfg["training"]["lambda_prior"],
               "lam_adv": cfg["training"]["lambda_adv"], "use_flight": use_flight}
    res = train_variant(fold, cfg, load_otp(), None, variant, seed)
    model, tnorm_src, mnorm = res["model"], res["tnorm"], res["mnorm"]

    out = {}
    # in-domain control (source statistics)
    tt = tensors_from_split({f"test_{k}": v for k, v in te_src.items()}, "test", tnorm_src, mnorm, use_flight)
    p_in = predict(model, tt)["logit"]
    # cross-dataset
    if mode == "pds":
        pool_h = H[tgt_idx]; pool_d = D[tgt_idx]
        pool = TypingNorm(cfg["typing"]["flight_asinh_scale"]).fit(
            np.concatenate([pool_h, btgt[0]]), np.concatenate([pool_d, btgt[1]]))
        tn_cross = pool
    else:
        tn_cross = tnorm_src
    tc = tensors_from_split({f"test_{k}": v for k, v in tgt_fold.items()}, "test", tn_cross, mnorm, use_flight)
    p_cr = predict(model, tc)["logit"]
    for name, z, lg in (("in_domain", te_src, p_in), ("cross", tgt_fold, p_cr)):
        y = z["label"]; bot = 1 - y
        met = detection_summary(bot, -lg, cfg["evaluation"]["base_rates"], cfg["evaluation"]["fpr_target"])
        out[name] = {"AUC": float(met["AUC"]), "EER": float(met["EER"]), "TPR@1%FPR": float(met["TPR@1%FPR"]),
                     "n_human": int((y == 1).sum()), "n_bot": int((y == 0).sum()),
                     "human_logit_median": float(np.median(lg[y == 1])),
                     "bot_logit_median": float(np.median(lg[y == 0]))}
    name = f"{src}_to_{tgt}_{mode}"
    d = RESULTS / "transfer_diag"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}.npz", in_logit=p_in, in_label=te_src["label"], in_letter=te_src["letter"],
                        cross_logit=p_cr, cross_label=tgt_fold["label"], cross_letter=tgt_fold["letter"],
                        T=np.array(res["preds"]["T"]))
    out["T"] = float(res["preds"]["T"]); out["best_epoch"] = int(res["meta"]["best_epoch"])
    out["epochs_run"] = int(res["meta"]["epochs_run"])
    return out


def main():
    import torch
    torch.set_num_threads(2)
    cfg = load_cfg()
    which = sys.argv[1:]
    jobs = [("CMU", "KeyRecs", "on"), ("CMU", "KeyRecs", "off"), ("CMU", "KeyRecs", "pds"),
            ("KeyRecs", "CMU", "on"), ("KeyRecs", "CMU", "off"), ("KeyRecs", "CMU", "pds")]
    if which:
        jobs = [j for j in jobs if f"{j[0]}_to_{j[1]}_{j[2]}" in which]
    res = {}
    with phase_timer("phase6_transfer_diagnostics"):
        for src, tgt, mode in jobs:
            key = f"{src}_to_{tgt}_{mode}"
            res[key] = run(cfg, src, tgt, mode)
            print(key, json.dumps({k: (round(v["AUC"], 3) if isinstance(v, dict) else v) for k, v in res[key].items()}), flush=True)
            out_path = RESULTS / "tables" / "transfer_diag.json"
            prev = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
            prev.update({key: res[key]})
            write_json(out_path, prev)


if __name__ == "__main__":
    main()
