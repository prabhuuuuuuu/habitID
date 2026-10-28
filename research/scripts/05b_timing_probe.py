"""Timing probe (required before the full runs). Trains each variant type once on fold 0, seed 0 (these are
real, reusable runs), records seconds per epoch and per fold, and projects the total runtime of the plan.

Writes results/timing/probe.json. The planned runs are:
  main     6 variants x 5 folds x 3 seeds
  ablation 4 variants x 5 folds x 1 seed (lambda=10, leave-one-signal-out x3)
  LOFO     8 variants x 5 folds x 1 seed (full and no-adversarial, held out per family A-D)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from habitid.utils.config import RESULTS, load_cfg  # noqa: E402
from habitid.utils.runtime import write_json  # noqa: E402

PROBE = ["full", "noprior", "noadv", "typing_only", "motion_only", "otp_only", "lam10", "no_typing", "no_motion",
         "no_otp", "lofo_full_D"]
PLAN = {  # variant -> (proxy probe variant, number of runs)
    **{v: (v, 15) for v in ["full", "noprior", "noadv", "typing_only", "motion_only", "otp_only"]},
    **{v: (v, 5) for v in ["lam10", "no_typing", "no_motion", "no_otp"]},  # each probed directly
    **{f"lofo_full_{L}": ("lofo_full_D", 5) for L in "ABCD"},
    **{f"lofo_noadv_{L}": ("noadv", 5) for L in "ABCD"},
}


def main():
    cfg = load_cfg()
    budget_h = 3.0
    measured = {}
    script = Path(__file__).resolve().parent / "05_train_habitid.py"
    for v in PROBE:
        t0 = time.time()
        subprocess.run([sys.executable, str(script), "--variants", v, "--folds", "0", "--seeds", "0"], check=True)
        wall = time.time() - t0
        meta = json.loads((RESULTS / "runs" / v / "fold0_seed0.json").read_text(encoding="utf-8"))
        epochs = [h["epoch_s"] for h in meta["history"]]
        measured[v] = {"fold_wall_s": meta["wall_s"], "epochs_run": meta["epochs_run"],
                       "mean_epoch_s": sum(epochs) / len(epochs), "process_wall_s": round(wall, 1)}
        print(v, measured[v], flush=True)
    total_s = 0.0
    breakdown = {}
    for v, (proxy, n) in PLAN.items():
        cost = measured[proxy]["fold_wall_s"] * n
        breakdown[v] = {"proxy": proxy, "runs": n, "projected_s": round(cost, 1)}
        total_s += cost
    report = {"measured": measured, "plan": breakdown, "projected_total_s": round(total_s, 1),
              "projected_total_h": round(total_s / 3600, 2), "budget_h": budget_h,
              "exceeds_budget": total_s / 3600 > budget_h,
              "note": "fold 0 / seed 0 runs of the probe variants are reused by the full plan"}
    write_json(RESULTS / "timing" / "probe.json", report)
    print(f"projected total: {report['projected_total_h']} h (budget {budget_h} h)")


if __name__ == "__main__":
    main()
