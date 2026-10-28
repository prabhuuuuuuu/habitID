"""Integrity check of all training and baseline outputs. Read-only on results/.

Checks: expected run matrix present (variant x fold x seed), no NaN/inf in any saved array, training
finished (early stop vs epoch cap), and that results/pilot_cap12 is not referenced by any aggregate script.
Writes results/tables/verification.json.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from habitid.utils.config import RESULTS, load_cfg  # noqa: E402
from habitid.utils.runtime import write_json  # noqa: E402

MAIN = ["full", "noprior", "noadv", "typing_only", "motion_only", "otp_only"]
ABL = ["lam10", "no_typing", "no_motion", "no_otp"]
LOFO = [f"lofo_full_{L}" for L in "ABCD"] + [f"lofo_noadv_{L}" for L in "ABCD"]
BASE = ["habitid_v0", "gate_only", "lr_handcrafted", "gbt_handcrafted", "isolation_forest", "one_class_svm",
        "scaled_manhattan"]


def expected(cfg):
    seeds = cfg["project"]["train_seeds"]
    seeds_abl = cfg["project"]["seed_ablation"]
    exp = {}
    for v in MAIN:
        for k in range(5):
            for s in seeds:
                exp[(v, k, s)] = True
    for v in ABL + LOFO:
        for k in range(5):
            for s in seeds_abl:
                exp[(v, k, s)] = True
    return exp


def main():
    cfg = load_cfg()
    cap = cfg["training"]["epochs"]
    missing, nonfinite, errors = [], [], []
    per_variant = {}
    capped = []
    for (v, k, s) in expected(cfg):
        stem = f"fold{k}_seed{s}"
        npz, js = RESULTS / "runs" / v / f"{stem}.npz", RESULTS / "runs" / v / f"{stem}.json"
        ck = RESULTS / "checkpoints" / v / f"{stem}.pt"
        if not (npz.exists() and js.exists() and ck.exists()):
            missing.append(f"{v} {stem}")
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            z = np.load(npz)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{v} {stem}: {e}")
            continue
        for key in z.files:
            arr = z[key]
            if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                nonfinite.append(f"{v} {stem} {key}")
        for h in meta["history"]:
            vals = [h["train_loss"], h["cal_bce"], h["b"]] + list(h["alpha"])
            if not np.isfinite(vals).all():
                nonfinite.append(f"{v} {stem} history@{h['epoch']}")
        hit_cap = meta["epochs_run"] >= cap  # reached the epoch cap (stopped by cap, not by patience)
        if hit_cap:
            capped.append(f"{v} {stem} best_epoch={meta['best_epoch']}")
        rec = per_variant.setdefault(v, {"runs": 0, "early_stopped": 0, "hit_epoch_cap_at_best": 0})
        rec["runs"] += 1
        rec["early_stopped"] += int(meta["epochs_run"] < cap)
        rec["hit_epoch_cap_at_best"] += int(hit_cap)
        rec["T_range"] = [min(rec.get("T_range", [9e9, 0])[0], meta["T"]), max(rec.get("T_range", [9e9, 0])[1], meta["T"])]
    base_missing, base_nonfinite = [], []
    for m in BASE:
        d = RESULTS / "runs" / "baselines" / m
        for k in range(5):
            seeds = [0] if m in ("habitid_v0", "gate_only", "lr_handcrafted", "scaled_manhattan") else cfg["project"]["train_seeds"]
            for s in seeds:
                p = d / f"fold{k}_seed{s}.npz"
                if not p.exists():
                    base_missing.append(f"{m} fold{k} seed{s}")
                    continue
                z = np.load(p)
                if any(z[f].dtype.kind == "f" and not np.isfinite(z[f]).all() for f in z.files):
                    base_nonfinite.append(f"{m} fold{k} seed{s}")
    # pilot exclusion: no aggregate script may read the pilot folder
    pilot_refs = []
    for py in (Path(__file__).resolve().parents[1] / "scripts").glob("*.py"):
        if py.name == Path(__file__).name:
            continue
        if re.search(r"pilot", py.read_text(encoding="utf-8")):
            pilot_refs.append(py.name)
    for py in (Path(__file__).resolve().parents[1] / "src").rglob("*.py"):
        if re.search(r"pilot_cap12", py.read_text(encoding="utf-8")):
            pilot_refs.append(str(py.name))
    import hashlib
    manifest = {str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((RESULTS / "runs" / "baselines").glob("*/*.npz"))}
    write_json(RESULTS / "tables" / "baseline_sha256.json", manifest)
    report = {
        "expected_runs": len(expected(cfg)),
        "missing_runs": missing, "missing_baseline_files": base_missing,
        "nonfinite": nonfinite, "baseline_nonfinite": base_nonfinite, "read_or_load_errors": errors,
        "hit_epoch_cap_at_best_epoch": capped,
        "per_variant": per_variant,
        "pilot_cap12_referenced_by": pilot_refs,
        "epoch_cap": cap,
    }
    write_json(RESULTS / "tables" / "verification.json", report)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items() if k != "per_variant"}, indent=1))


if __name__ == "__main__":
    main()
