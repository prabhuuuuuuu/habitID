"""Build the per-fold corpora: human sessions (cross-paired within role) and attacker sessions.

All generators are fitted on the TRAIN role of the fold only. Calibration and test bots are drawn
from those fitted generators, so no test-subject data enter any fitting step.
"""
from __future__ import annotations

import numpy as np
import torch

from ..bots.motion_gen import ConvARGen, GaussGen, ReplayMotionGen, ZeroGen
from ..bots.policy import concat_splits, draw_bots, draw_humans
from ..bots.typing_gen import ARGen, HumanizedGen, NaiveGen, ReplayGen
from ..utils.runtime import set_seed


def build_fold(cfg: dict, otp_cfg: dict, typing: dict, motion: dict, fold_idx: int,
               typ_roles: dict, mot_roles: dict) -> tuple[dict, dict]:
    """typing: dict(h,d,subject,dataset); motion: dict(windows, subject). Returns (arrays, generator_states)."""
    set_seed(1000 + fold_idx)
    rng = np.random.default_rng(2000 + fold_idx)
    gate_rate = cfg["gate"]["invalid_rate"]
    bounds = cfg["typing"]["bounds_s"]
    mix = cfg["bot_mix"]
    T = int(round(cfg["motion"]["fs_hz"] * cfg["motion"]["window_s"]))

    def tpool(subjects):
        m = np.isin(typing["subject"], subjects)
        return {"h": typing["h"][m], "d": typing["d"][m], "subject": typing["subject"][m],
                "dataset": typing["dataset"][m]}

    def mpool(subjects):
        m = np.isin(motion["subject"], subjects)
        return motion["windows"][m], motion["subject"][m]

    pools = {r: tpool(typ_roles[r]) for r in ("train", "cal", "test")}
    mwin = {r: mpool(mot_roles[r]) for r in ("train", "cal", "test")}

    tr_pool = pools["train"]
    humans_tr = np.ones(len(tr_pool["h"]), dtype=bool)
    gen_t = {
        "A": NaiveGen(cfg),
        "B": HumanizedGen(cfg).fit(tr_pool["h"], tr_pool["d"]),
        "C": ReplayGen(cfg).fit(tr_pool["h"], tr_pool["d"]),
        "D": ARGen(cfg, seed=fold_idx).fit(tr_pool["h"], tr_pool["d"]),
    }
    W_tr = mwin["train"][0]
    gen_m = {
        "A": ZeroGen(cfg).fit(W_tr),
        "B": GaussGen().fit(W_tr),
        "C": ReplayMotionGen(cfg).fit(W_tr),
        "D": ConvARGen(cfg, seed=fold_idx).fit(W_tr),
    }

    out, role_meta = {}, {}
    for r_i, role in enumerate(("train", "cal", "test")):
        n_h = cfg["sizes"][f"{role}_h"] if role != "train" else cfg["sizes"]["train_h"]
        pool = pools[role]
        W, S = mwin[role]
        hum = draw_humans(n_h, rng, pool, W, S, otp_cfg, gate_rate)
        n_b = int(round(cfg["sizes"]["bot_per_human"] * n_h))
        bots = draw_bots(n_b, rng, gen_t, gen_m, pool, W, S, otp_cfg, bounds, mix, gate_rate,
                         cluster_base=(fold_idx + 1) * 10_000_000 + r_i * 1_000_000, motion_T=T)
        bots["windows"] = bots["windows"].astype(np.float32)
        hum["windows"] = hum["windows"].astype(np.float32)
        sp = concat_splits([hum, bots])
        sp["motion_subject"] = sp["motion_subject"]
        for k, v in sp.items():
            out[f"{role}_{k}"] = v
        role_meta[role] = {
            "humans": int(n_h), "bots": int(n_b),
            "bot_types": {k: int((bots["type"] == c).sum()) for k, c in
                          [("full", 1), ("typing_only", 2), ("motion_only", 3), ("otp_only", 4)]},
            "bot_letters": {k: int((bots["letter"] == c).sum()) for k, c in
                            [("A", 1), ("B", 2), ("C", 3), ("D", 4), ("O", 5)]},
            "typing_pool_sessions": int(len(pool["h"])), "typing_pool_subjects": int(len(np.unique(pool["subject"]))),
            "motion_windows": int(len(W)), "motion_subjects": int(len(np.unique(S))),
            "gate_pass": int(sp["gate"].sum()), "gate_fail": int((~sp["gate"]).sum()),
        }
    out["train_motion_pool"] = mwin["train"][0].astype(np.float32)
    out["train_motion_pool_subject"] = mwin["train"][1]
    out["fold_index"] = np.array(fold_idx)
    meta = {"fold": fold_idx, "roles": role_meta,
            "typing_roles": {k: [int(x) for x in v] for k, v in typ_roles.items()},
            "motion_roles": {k: [int(x) for x in v] for k, v in mot_roles.items()},
            "generatorD_nll": [float(x) for x in gen_t["D"].train_nll],
            "motionD_nll": [float(x) for x in gen_m["D"].train_nll]}
    gen_state = {"D": gen_t["D"].state_dict(), "MD": gen_m["D"].state_dict(),
                 "replay_pool_h": gen_t["C"].h, "replay_pool_d": gen_t["C"].d,
                 "meta": meta}
    return out, gen_state, meta
