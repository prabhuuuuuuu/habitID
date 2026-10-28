"""Phase 5: train habitID-Net variants. One prediction file + checkpoint + metadata per (variant, fold, seed).

Existing outputs are skipped, so the script can be resumed. The first planned run (fold 0, seed 0) of each
variant doubles as the timing probe; its runtime is recorded in results/timing/probe.json.

Usage:
  python 05_train_habitid.py --group main          # 5 folds x 3 seeds for the main variants
  python 05_train_habitid.py --group ablation      # seed 0 for ablations, lambda sweep and LOFO
  python 05_train_habitid.py --variants full --folds 0 --seeds 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from habitid.bots.typing_gen import ARGen  # noqa: E402
from habitid.train.train import train_variant  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_otp  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402

ALL = ["A", "B", "C", "D"]
ALL_ACTIVE = ["typing", "motion", "otp"]
VARIANTS = {
    "full": dict(active=ALL_ACTIVE, lam_prior=None, lam_adv=None, dstar=True, group="main"),
    "noprior": dict(active=ALL_ACTIVE, lam_prior=0.0, lam_adv=None, group="main"),
    "noadv": dict(active=ALL_ACTIVE, lam_prior=None, lam_adv=0.0, group="main"),
    "typing_only": dict(active=["typing"], lam_prior=None, lam_adv=None, group="main"),
    "motion_only": dict(active=["motion"], lam_prior=None, lam_adv=None, group="main"),
    "otp_only": dict(active=["otp"], lam_prior=None, lam_adv=None, group="main"),
    "lam10": dict(active=ALL_ACTIVE, lam_prior=10.0, lam_adv=None, group="ablation"),
    "no_typing": dict(active=["motion", "otp"], lam_prior=None, lam_adv=None, group="ablation"),
    "no_motion": dict(active=["typing", "otp"], lam_prior=None, lam_adv=None, group="ablation"),
    "no_otp": dict(active=["typing", "motion"], lam_prior=None, lam_adv=None, group="ablation"),
}
# POST-HOC EXPLORATORY (not the main model): typing encoder with summary statistics. Motivated by the
# typing-only-bot diagnosis. Same settings as "full" except the summary option; reported separately.
VARIANTS["typing_summary_full"] = dict(active=ALL_ACTIVE, lam_prior=None, lam_adv=None, typing_summary=True,
                                       group="exploratory")
for L in ALL:
    VARIANTS[f"lofo_full_{L}"] = dict(active=ALL_ACTIVE, lam_prior=None, lam_adv=None, exclude=[L],
                                      dstar=True, group="lofo")
    VARIANTS[f"lofo_noadv_{L}"] = dict(active=ALL_ACTIVE, lam_prior=None, lam_adv=0.0, exclude=[L], group="lofo")


def run_one(name: str, k: int, seed: int, fold: dict, cfg: dict, otp_cfg: dict, gen_d, suffix: str = "") -> float:
    v = dict(VARIANTS[name])
    out_dir = RESULTS / "runs" / (name + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fold{k}_seed{seed}"
    if (out_dir / f"{stem}.npz").exists():
        print(f"skip {name} {stem} (exists)", flush=True)
        return 0.0
    v.setdefault("lam_prior", None)
    v.setdefault("lam_adv", None)
    if v["lam_prior"] is None:
        v["lam_prior"] = cfg["training"]["lambda_prior"]
    if v["lam_adv"] is None:
        v["lam_adv"] = cfg["training"]["lambda_adv"]
    v["extras"] = []
    t0 = time.time()
    res = train_variant(fold, cfg, otp_cfg, gen_d, v, seed)
    dt = time.time() - t0
    np.savez_compressed(out_dir / f"{stem}.npz", **res["preds"])
    meta = res["meta"]
    meta.update({"variant": name, "fold": k, "wall_s": round(dt, 2)})
    write_json(out_dir / f"{stem}.json", meta)
    ck = RESULTS / "checkpoints" / (name + suffix)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state": res["model"].state_dict(), "active": v["active"], "tnorm": res["tnorm"].state(),
                "mnorm": res["mnorm"].state(), "T": float(res["preds"]["T"]), "use_flight": v.get("use_flight", True)},
               ck / f"{stem}.pt")
    print(f"{name} {stem}: {dt:.1f}s epochs={meta['epochs_run']} best={meta['best_epoch']} "
          f"alpha={np.round(meta['alpha'], 3).tolist()} T={meta['T']:.3f}", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["main", "ablation", "lofo", "all"], default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (parallel workers)")
    ap.add_argument("--epoch-cap", type=int, default=None, help="override training.epochs (sensitivity reruns)")
    ap.add_argument("--suffix", default="", help="output directory suffix, e.g. _cap48")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_cfg()
    if args.epoch_cap:
        cfg["training"]["epochs"] = args.epoch_cap
    otp_cfg = load_otp()
    pc = cfg["project"]
    if args.variants:
        names = args.variants
    elif args.group == "all":
        names = list(VARIANTS)
    else:
        names = [n for n, v in VARIANTS.items() if v["group"] == (args.group or "main")]
    folds = args.folds if args.folds is not None else list(range(pc["n_folds"]))
    seeds_main = pc["train_seeds"]
    seeds_abl = pc["seed_ablation"]
    with phase_timer(f"phase5_train_{args.group or 'custom'}"):
        for k in folds:
            fold = dict(np.load(PROC / "folds" / f"fold{k}.npz"))
            fold["fold_index"] = np.array(k)
            gst = torch.load(PROC / "bots" / f"fold{k}_generators.pt", weights_only=False)
            gen_d = ARGen(cfg, seed=k).load_state(gst["D"], cfg)
            for name in names:
                seeds = seeds_main if VARIANTS[name]["group"] == "main" else seeds_abl
                if args.seeds:
                    seeds = args.seeds
                for s in seeds:
                    run_one(name, k, s, fold, cfg, otp_cfg, gen_d, args.suffix)


if __name__ == "__main__":
    main()
