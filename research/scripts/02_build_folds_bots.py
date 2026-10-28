"""Phase 2-3: subject-grouped folds, attacker generators (fitted on training roles only), and
per-fold session corpora. Writes data/processed/folds/fold{k}.npz and bots/fold{k}_generators.pt.
Runs the leakage assertions for every fold and modality before anything is saved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from habitid.bots.corpus import build_fold  # noqa: E402
from habitid.data.folds import build_motion_folds, build_typing_folds, check_disjoint  # noqa: E402
from habitid.utils.config import PROC, RESULTS, load_cfg, load_otp  # noqa: E402
from habitid.utils.runtime import phase_timer, write_json  # noqa: E402


def main():
    cfg = load_cfg()
    otp_cfg = load_otp()
    pc = cfg["project"]
    with phase_timer("phase2_folds_bots"):
        t = np.load(PROC / "typing_canonical.npz")
        typing = {"h": t["h"], "d": t["d"], "subject": t["subject"], "dataset": t["dataset"]}
        m = np.load(PROC / "motion_canonical.npz")
        motion = {"windows": m["windows"], "subject": m["subject"]}

        tfolds = build_typing_folds(typing["subject"], typing["dataset"], pc["n_folds"], pc["fold_seed"], pc["cal_frac"])
        mfolds = build_motion_folds(motion["subject"], pc["n_folds"], pc["fold_seed"], pc["cal_frac"])
        for k in range(pc["n_folds"]):
            check_disjoint(tfolds[k]); check_disjoint(mfolds[k])
        (PROC / "folds").mkdir(parents=True, exist_ok=True)
        (PROC / "bots").mkdir(parents=True, exist_ok=True)

        all_meta = []
        for k in range(pc["n_folds"]):
            arrays, gstate, meta = build_fold(cfg, otp_cfg, typing, motion, k, tfolds[k], mfolds[k])
            np.savez_compressed(PROC / "folds" / f"fold{k}.npz", **arrays)
            torch.save(gstate, PROC / "bots" / f"fold{k}_generators.pt")
            all_meta.append(meta)
            print(f"fold {k}: " + ", ".join(f"{r}: {v['humans']}h/{v['bots']}b" for r, v in meta["roles"].items()), flush=True)
        write_json(RESULTS / "phase2" / "fold_meta.json", all_meta)


if __name__ == "__main__":
    main()
