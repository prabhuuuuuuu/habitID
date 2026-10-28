"""Epoch-cap sensitivity (D35): capped noadv runs rerun with the cap raised from 24 to 48 epochs.
Compares, for the same fold and seed, the 24-epoch run (results/runs/noadv) and the 48-epoch rerun
(results/runs/noadv_cap48). Reads only results/. Writes results/tables/cap_sensitivity.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from habitid.eval.io import neural_test, run_meta  # noqa: E402
from habitid.utils.config import RESULTS  # noqa: E402
from habitid.utils.runtime import write_json  # noqa: E402


def auc_of(variant, k, s):
    v = neural_test(variant, k, s)
    return float(roc_auc_score(1 - v["label"], v["score"]))


def main():
    out = {"runs": {}}
    cap48 = sorted((RESULTS / "runs" / "noadv_cap48").glob("fold*_seed*.json"))
    for p in cap48:
        k = int(p.name.split("_")[0][4:]); s = int(p.name.split("_")[1][4:-5])
        m24 = run_meta("noadv", k, s); m48 = json.loads(p.read_text(encoding="utf-8"))
        out["runs"][f"fold{k}_seed{s}"] = {
            "epochs_24_cap": m24["epochs_run"], "best_epoch_24": m24["best_epoch"], "AUC_24": auc_of("noadv", k, s),
            "epochs_48_cap": m48["epochs_run"], "best_epoch_48": m48["best_epoch"],
            "AUC_48": float(roc_auc_score(1 - neural_test("noadv_cap48", k, s)["label"], neural_test("noadv_cap48", k, s)["score"])),
            "stopped_by_patience_at_48": bool(m48["epochs_run"] < 48),
        }
    if out["runs"]:
        d = [r["AUC_48"] - r["AUC_24"] for r in out["runs"].values()]
        out["mean_AUC_change_48_minus_24"] = float(np.mean(d))
        out["n_runs"] = len(d)
        out["n_stopped_by_patience_at_48"] = int(sum(r["stopped_by_patience_at_48"] for r in out["runs"].values()))
    write_json(RESULTS / "tables" / "cap_sensitivity.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
