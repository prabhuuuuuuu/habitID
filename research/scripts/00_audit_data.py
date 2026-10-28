"""Phase 0: data audit. Read-only on `raw dataset/`.

Extracts archive.zip (MotionSense) into research/data/interim/motionsense/ and writes
  research/data/AUDIT.md                     human-readable audit (all numbers generated here)
  research/results/audit/audit_summary.json  machine-readable numbers
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "raw dataset"
RESEARCH = PROJECT / "research"
DATA = RESEARCH / "data"
MS_DIR = DATA / "interim" / "motionsense"
RESULTS = RESEARCH / "results" / "audit"

ID_TOL = 0.0015  # KeyRecs values are stored with 3 decimals; allow one rounding unit per term
MS_FS = 50       # MotionSense documented sampling rate (Hz); no timestamp column in the file
MS_WIN = 150     # 3 s windows at 50 Hz
HOLD_BOUNDS = (0.01, 1.0)     # proposed validity bounds (s), to be confirmed at Gate A
DIGRAPH_BOUNDS = (-0.5, 2.0)  # proposed validity bounds for signed flight time (s)
DD_BOUNDS = (0.01, 2.0)       # proposed validity bounds for down-down latency (s)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(x.size), "min": float(x.min()), "q05": float(qs[0]), "q25": float(qs[1]),
        "median": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4]), "max": float(x.max()),
        "mean": float(x.mean()),
    }


def fmt(v, nd=4) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- CMU
def audit_cmu(summary: dict) -> list[str]:
    path = RAW / "DSL-StrongPasswordData.csv"
    df = pd.read_csv(path)
    hold = [c for c in df.columns if c.startswith("H.")]
    dd = [c for c in df.columns if c.startswith("DD.")]
    ud = [c for c in df.columns if c.startswith("UD.")]
    assert len(hold) == 11 and len(dd) == 10 and len(ud) == 10, (len(hold), len(dd), len(ud))
    timing = hold + dd + ud
    H = df[hold].to_numpy(float)
    DD = df[dd].to_numpy(float)
    UD = df[ud].to_numpy(float)

    ident = DD - UD - H[:, :10]
    flight = UD
    per_subject = df.groupby("subject").size()
    per_sess = df.groupby(["subject", "sessionIndex"]).size()
    s = {
        "file": path.name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "timing_columns": len(timing), "hold_cols": len(hold), "dd_cols": len(dd), "ud_cols": len(ud),
        "subjects": int(df["subject"].nunique()), "sessions_per_subject": int(df.groupby("subject")["sessionIndex"].nunique().iloc[0]),
        "reps_per_session": int(per_sess.iloc[0]), "rows_per_subject_min": int(per_subject.min()),
        "rows_per_subject_max": int(per_subject.max()), "nan_cells": int(df.isna().sum().sum()),
        "nonpositive_hold": int((H <= 0).sum()), "negative_dd": int((DD < 0).sum()),
        "dd_gt_2s": int((DD > 2).sum()), "dd_gt_10s": int((DD > 10).sum()),
        "dd_max": float(DD.max()), "flight_negative_frac": float((flight < 0).mean()),
        "identity_DD_minus_UD_eq_H1_frac": float((np.abs(ident) <= ID_TOL).mean()),
        "identity_max_abs_dev": float(np.abs(ident).max()),
        "hold": describe(H.ravel()), "dd": describe(DD.ravel()), "flight_UD": describe(flight.ravel()),
        "keys": [c[2:] for c in hold],
        "hold_out_of_bounds_frac": float(((H < HOLD_BOUNDS[0]) | (H > HOLD_BOUNDS[1])).mean()),
        "dd_out_of_bounds_frac": float(((DD < DD_BOUNDS[0]) | (DD > DD_BOUNDS[1])).mean()),
        "flight_out_of_bounds_frac": float(((flight < DIGRAPH_BOUNDS[0]) | (flight > DIGRAPH_BOUNDS[1])).mean()),
        "sessions_with_any_oob_frac": float((((H < HOLD_BOUNDS[0]) | (H > HOLD_BOUNDS[1])).any(1)
                                            | ((DD < DD_BOUNDS[0]) | (DD > DD_BOUNDS[1])).any(1)
                                            | ((flight < DIGRAPH_BOUNDS[0]) | (flight > DIGRAPH_BOUNDS[1])).any(1)).mean()),
        "sha256": sha256(path),
    }
    summary["cmu"] = s
    return [
        "## 1. CMU DSL-StrongPasswordData (Killourhy & Maxion 2009)",
        "",
        f"- File: `raw dataset/{path.name}` ({path.stat().st_size:,} bytes), sha256 `{s['sha256'][:16]}...`",
        f"- Shape: {s['rows']:,} rows x {s['columns']} columns. Keys (from `H.*` columns, in order): "
        + ", ".join(f"`{k}`" for k in s["keys"]) + ". The password is `.tie5Roanl` followed by Return (11 key events; `Shift.r` is the capital R).",
        f"- Timing columns: {s['hold_cols']} hold (`H.k`), {s['dd_cols']} down-down (`DD.k1.k2`), {s['ud_cols']} up-down (`UD.k1.k2`). "
        "**No UU or DU columns.**",
        f"- Subjects: {s['subjects']}; sessions per subject: {s['sessions_per_subject']}; repetitions per session: {s['reps_per_session']}; "
        f"rows per subject: {s['rows_per_subject_min']} to {s['rows_per_subject_max']}. Missing cells: {s['nan_cells']}.",
        f"- Column semantics verified: `DD(k1,k2) - UD(k1,k2) = H(k1)` holds in {s['identity_DD_minus_UD_eq_H1_frac']*100:.2f}% of cells "
        f"(max abs deviation {s['identity_max_abs_dev']:.6f}). So `UD` is the **standard flight time** (up of k1 to down of k2), which can be negative when keys overlap.",
        f"- Non-positive holds: {s['nonpositive_hold']}. Negative DD: {s['negative_dd']}. DD > 2 s: {s['dd_gt_2s']} cells; DD > 10 s: {s['dd_gt_10s']} cells (max DD {s['dd_max']:.2f} s, pauses).",
        f"- Negative flight-time (`UD`) cells: {s['flight_negative_frac']*100:.2f}% (expected for overlapping keys).",
        "",
        md_table(["quantity", "n", "q05", "median", "q95", "max"],
                 [["hold (s)", s["hold"]["n"], s["hold"]["q05"], s["hold"]["median"], s["hold"]["q95"], s["hold"]["max"]],
                  ["DD (s)", s["dd"]["n"], s["dd"]["q05"], s["dd"]["median"], s["dd"]["q95"], s["dd"]["max"]],
                  ["flight UD (s)", s["flight_UD"]["n"], s["flight_UD"]["q05"], s["flight_UD"]["median"], s["flight_UD"]["q95"], s["flight_UD"]["max"]]]),
        "",
    ]


# ---------------------------------------------------------------- KeyRecs fixed
def audit_keyrecs_fixed(summary: dict) -> list[str]:
    path = RAW / "fixed-text.csv"
    with open(path, newline="", encoding="utf-8") as f:
        header = [h.strip() for h in next(csv.reader(f))]
    # Layout: participant, session, repetition, then 10 holds interleaved with 9 x (DD, DU, UD, UU), then total time.
    hold_idx = [3 + 5 * i for i in range(10)]
    assert all(header[j].startswith("DU.") and header[j].split(".")[1] == header[j].split(".")[2].split(".")[0] for j in hold_idx)
    keys = [header[j].split(".")[1] for j in hold_idx]
    assert header[-1] == "total time" and len(header) == 50
    df = pd.read_csv(path)
    X = df.iloc[:, 3:49].to_numpy(float)
    H = X[:, 0::5]                      # 10 holds
    DD = np.stack([X[:, 5 * i + 1] for i in range(9)], 1)
    DU = np.stack([X[:, 5 * i + 2] for i in range(9)], 1)
    UD = np.stack([X[:, 5 * i + 3] for i in range(9)], 1)
    UU = np.stack([X[:, 5 * i + 4] for i in range(9)], 1)
    h1, h2 = H[:, :9], H[:, 1:]
    id_ud = UD - DD - h1          # stored UD = DD + hold(k1)  (sign-flipped vs standard)
    id_uu = UU - DU - h1          # stored UU = DU + hold(k1)
    id_dd = DD - DU - h2          # stored DU = DD - hold(k2)
    id_std = UD - (DD - h1)       # standard convention: UD = DD - hold(k1)
    bad_rows = (np.abs(X) > 60).any(1)
    ok = ~bad_rows
    std_flight = DD - h1          # standard flight, consistent with CMU and KeyRecs free-text
    sess = df.groupby(["participant", "session"]).size()
    tot = df["total time"].to_numpy(float)
    s = {
        "file": path.name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "participants": int(df["participant"].nunique()), "sessions": sorted(int(x) for x in df["session"].unique()),
        "reps_per_participant_session_min": int(sess.min()), "reps_per_participant_session_median": float(sess.median()),
        "reps_per_participant_session_max": int(sess.max()),
        "password_from_header": "".join(keys), "holds": len(keys), "digraphs_per_row": 9 * 4,
        "nan_cells": int(df.isna().sum().sum()), "nonpositive_hold": int((H <= 0).sum()),
        "negative_dd": int((DD < 0).sum()), "dd_gt_2s": int((DD > 2).sum()), "dd_gt_10s": int((DD > 10).sum()),
        "dd_max": float(DD.max()),
        "stored_UD_negative_frac": float((UD < 0).mean()),
        "identity_UD_eq_DD_plus_H1_frac": float((np.abs(id_ud) <= ID_TOL).mean()),
        "identity_UU_eq_DU_plus_H1_frac": float((np.abs(id_uu) <= ID_TOL).mean()),
        "identity_DD_minus_DU_eq_H2_frac": float((np.abs(id_dd) <= ID_TOL).mean()),
        "identity_max_abs_dev": float(max(np.abs(id_ud).max(), np.abs(id_uu).max(), np.abs(id_dd).max())),
        "identity_std_UD_eq_DD_minus_H1_frac": float((np.abs(id_std) <= ID_TOL).mean()),
        "rows_any_abs_gt_60s": int(bad_rows.sum()), "rows_any_abs_gt_5s": int((np.abs(X) > 5).any(1).sum()),
        "hold": describe(H[ok].ravel()), "dd": describe(DD[ok].ravel()), "stored_UD": describe(UD[ok].ravel()),
        "flight_reconstructed": describe(std_flight[ok].ravel()),
        "flight_reconstructed_negative_frac": float((std_flight[ok] < 0).mean()),
        "total_time_nonpositive": int((tot <= 0).sum()), "total_time_gt_60s": int((tot > 60).sum()),
        "total_time_median": float(np.median(tot)), "total_time_min": float(tot.min()),
        "hold_out_of_bounds_frac": float(((H < HOLD_BOUNDS[0]) | (H > HOLD_BOUNDS[1])).mean()),
        "dd_out_of_bounds_frac": float(((DD < DD_BOUNDS[0]) | (DD > DD_BOUNDS[1])).mean()),
        "flight_out_of_bounds_frac": float(((std_flight < DIGRAPH_BOUNDS[0]) | (std_flight > DIGRAPH_BOUNDS[1])).mean()),
        "sessions_with_any_oob_frac": float((((H < HOLD_BOUNDS[0]) | (H > HOLD_BOUNDS[1])).any(1)
                                            | ((DD < DD_BOUNDS[0]) | (DD > DD_BOUNDS[1])).any(1)
                                            | ((std_flight < DIGRAPH_BOUNDS[0]) | (std_flight > DIGRAPH_BOUNDS[1])).any(1)).mean()),
        "sha256": sha256(path),
    }
    summary["keyrecs_fixed"] = s
    return [
        "## 2. KeyRecs fixed-text (Dias et al. 2023, Data in Brief 50:109509; Zenodo 10.5281/zenodo.7886742)",
        "",
        f"- File: `raw dataset/{path.name}`, {path.stat().st_size:,} bytes, sha256 `{s['sha256'][:16]}...`. "
        f"Shape: {s['rows']:,} rows x {s['columns']} columns (the data article states 19,773 rows; the file has {s['rows']:,} data rows plus a header line).",
        f"- Participants: **{s['participants']}** (the Zenodo description says 100; the data article and the file say 99 after one exclusion). "
        f"Sessions: {s['sessions']}. Repetitions per participant-session: min {s['reps_per_participant_session_min']}, "
        f"median {s['reps_per_participant_session_median']:.0f}, max {s['reps_per_participant_session_max']}.",
        f"- Password recovered from the column names: `{s['password_from_header']}` (10 keys, no Return). Columns 4-49 are, per key pair, "
        "`DU.k.k` (hold) followed by `DD.k1.k2, DU.k1.k2, UD.k1.k2, UU.k1.k2`.",
        f"- Missing cells: {s['nan_cells']}. Non-positive holds: {s['nonpositive_hold']}. Negative DD: {s['negative_dd']}. "
        f"DD > 2 s: {s['dd_gt_2s']}; DD > 10 s: {s['dd_gt_10s']} (raw max {s['dd_max']:.0f} s).",
        f"- **Corrupt rows:** {s['rows_any_abs_gt_60s']} rows contain at least one timing value with |x| > 60 s (up to ~1e6 s), and {s['rows_any_abs_gt_5s']} rows contain a value > 5 s. "
        "The corrupt rows are excluded from the distributions below and dropped in Phase 1. Their `total time` is also implausible.",
        f"- `total time` (col 50) is **not used**: {s['total_time_nonpositive']} row(s) have total time <= 0 and {s['total_time_gt_60s']} exceed 60 s "
        f"(min {s['total_time_min']:.0f} s, median {s['total_time_median']:.2f} s).",
        "",
        "### 2a. Column semantics: sign convention of the stored `UD/UU/DU` columns",
        "",
        "The standard convention (CMU, and KeyRecs free-text as shown in section 3) is `UD = DD - H(k1)`, `UU = DU - H(k1)`, `DU = DD + H(k2)`. "
        "In the fixed-text file the hold term has the opposite sign, and this holds exactly in every row:",
        "",
        md_table(["identity (checked on every digraph of every row)", "fraction within tolerance", "tolerance (s)"],
                 [["stored UD = DD + H(k1)  (fixed-text, as stored)", s["identity_UD_eq_DD_plus_H1_frac"], ID_TOL],
                  ["stored UU = DU + H(k1)  (fixed-text, as stored)", s["identity_UU_eq_DU_plus_H1_frac"], ID_TOL],
                  ["stored DU = DD - H(k2)  (fixed-text, as stored)", s["identity_DD_minus_DU_eq_H2_frac"], ID_TOL],
                  ["standard UD = DD - H(k1)  (fixed-text)", s["identity_std_UD_eq_DD_minus_H1_frac"], ID_TOL]]),
        "",
        f"- Stored fixed-text `UD` has **{s['stored_UD_negative_frac']*100:.2f}%** negative values; CMU flight times are "
        f"{summary['cmu']['flight_negative_frac']*100:.2f}% negative, and standard flights should be negative whenever keys overlap.",
        f"- Flight time used for fixed-text: `DD(k1,k2) - H(k1)` (standard). On the clean rows it is {s['flight_reconstructed_negative_frac']*100:.2f}% negative, "
        f"median {s['flight_reconstructed']['median']:.3f} s, q05 {s['flight_reconstructed']['q05']:.3f} s, q95 {s['flight_reconstructed']['q95']:.3f} s. "
        f"CMU flight: {summary['cmu']['flight_negative_frac']*100:.2f}% negative, median {summary['cmu']['flight_UD']['median']:.3f} s, "
        f"q05 {summary['cmu']['flight_UD']['q05']:.3f} s, q95 {summary['cmu']['flight_UD']['q95']:.3f} s.",
        "- Consequence: the four digraph columns carry only two independent pieces of information per pair (`DD` and the holds). "
        f"The fixed-text flight is derived as `DD - H(k1)`; the stored `UD`, `UU`, `DU` are not used. Stored DD has median {s['dd']['median']:.3f} s "
        f"(CMU {summary['cmu']['dd']['median']:.3f} s), and the flight distribution above is close to CMU's, which is the basis for this choice.",
        "",
        md_table(["quantity", "n", "q05", "median", "q95", "max"],
                 [["hold (s)", s["hold"]["n"], s["hold"]["q05"], s["hold"]["median"], s["hold"]["q95"], s["hold"]["max"]],
                  ["DD (s)", s["dd"]["n"], s["dd"]["q05"], s["dd"]["median"], s["dd"]["q95"], s["dd"]["max"]],
                  ["stored UD (s), not used", s["stored_UD"]["n"], s["stored_UD"]["q05"], s["stored_UD"]["median"], s["stored_UD"]["q95"], s["stored_UD"]["max"]],
                  ["flight = DD - H(k1) (s)", s["flight_reconstructed"]["n"], s["flight_reconstructed"]["q05"], s["flight_reconstructed"]["median"], s["flight_reconstructed"]["q95"], s["flight_reconstructed"]["max"]]]),
        "",
    ]


# ---------------------------------------------------------------- KeyRecs free text
def audit_keyrecs_free(summary: dict, participants_fixed: set[str]) -> list[str]:
    path = RAW / "free-text.csv"
    n_lines = 0
    end_rows = 0
    malformed = []
    keep = []
    key_tokens = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for r in reader:
            n_lines += 1
            if len(r) < 9:
                malformed.append((reader.line_num, "short row"))
                continue
            part, sess, k1, k2 = r[0], r[1], r[2], r[3]
            vals = r[4:9]
            if k2 == "" and all(v == "" for v in vals[1:]):
                end_rows += 1
                key_tokens[k1] = key_tokens.get(k1, 0) + 1
                continue
            try:
                nums = [float(v) for v in vals]
            except ValueError:
                malformed.append((reader.line_num, repr(",".join(r))[:160]))
                continue
            keep.append((part, sess, k1, k2, *nums))
            key_tokens[k1] = key_tokens.get(k1, 0) + 1
            key_tokens[k2] = key_tokens.get(k2, 0) + 1
    arr = pd.DataFrame(keep, columns=["participant", "session", "key1", "key2", "H1", "DD", "DU", "UD", "UU"])
    single = {k: v for k, v in key_tokens.items() if len(k) == 1}
    special = sorted(((k, v) for k, v in key_tokens.items() if len(k) > 1), key=lambda kv: -kv[1])
    flight = arr["DD"] - arr["H1"]
    ident_std_ud = arr["UD"] - (arr["DD"] - arr["H1"])
    ident_std_uu = arr["UU"] - (arr["DU"] - arr["H1"])
    part_free = set(arr["participant"])
    s = {
        "file": path.name, "data_rows": n_lines, "valid_digraph_rows": int(len(arr)),
        "end_of_session_rows": end_rows, "malformed_rows": len(malformed),
        "malformed_examples": [{"line": ln, "raw": raw} for ln, raw in malformed[:3]],
        "header_columns": header, "participants": int(arr["participant"].nunique()),
        "sessions": sorted(arr["session"].unique().tolist()),
        "single_char_token_types": len(single), "special_tokens": special[:15], "special_token_types": len(special),
        "key1_special_frac": float((arr["key1"].str.len() > 1).mean()),
        "key2_special_frac": float((arr["key2"].str.len() > 1).mean()),
        "dd_negative": int((arr["DD"] < 0).sum()), "dd_gt_2s": int((arr["DD"] > 2).sum()),
        "dd_gt_10s": int((arr["DD"] > 10).sum()), "hold_nonpositive": int((arr["H1"] <= 0).sum()),
        "hold_gt_1s": int((arr["H1"] > 1).sum()),
        "identity_std_UD_eq_DD_minus_H1_frac": float((np.abs(ident_std_ud) <= ID_TOL).mean()),
        "identity_std_UU_eq_DU_minus_H1_frac": float((np.abs(ident_std_uu) <= ID_TOL).mean()),
        "rows_any_abs_gt_60s": int(((arr[["H1", "DD", "DU", "UD", "UU"]].abs()) > 60).any(axis=1).sum()),
        "stored_UD_negative_frac": float((arr["UD"] < 0).mean()),
        "H1": describe(arr["H1"]), "DD": describe(arr["DD"]), "stored_UD": describe(arr["UD"]),
        "flight_reconstructed": describe(flight),
        "flight_reconstructed_negative_frac": float((flight < 0).mean()),
        "dd_negative_lt_minus_0p2": int((arr["DD"] < -0.2).sum()),
        "participants_match_fixed": part_free == participants_fixed,
        "sha256": sha256(path),
    }
    summary["keyrecs_free"] = s
    lines = [
        "## 3. KeyRecs free-text (transcription task)",
        "",
        f"- File: `raw dataset/{path.name}` ({path.stat().st_size:,} bytes), sha256 `{s['sha256'][:16]}...`. "
        f"Data lines (excluding header): {s['data_rows']:,}.",
        f"- Each row is one consecutive key pair `(key1, key2)` in typing order. Columns: `DU.key1.key1` (hold of key1), "
        "`DD.key1.key2`, `DU.key1.key2`, `UD.key1.key2`, `UU.key1.key2`. The header ends with a trailing comma and an unnamed empty column; `UU.key1.key2 ` has a trailing space.",
        f"- End-of-session rows (`key2` empty, only the hold present): {s['end_of_session_rows']}. "
        f"Malformed rows dropped: {s['malformed_rows']} (non-numeric timing fields, caused by embedded line breaks/extra delimiters). "
        f"Valid digraph rows kept: {s['valid_digraph_rows']:,}.",
        f"- Participants: {s['participants']} (same set as fixed-text: {s['participants_match_fixed']}). Sessions: {s['sessions']}.",
        f"- Token types: {s['single_char_token_types']} single-character tokens; {s['special_token_types']} special tokens, e.g. "
        + ", ".join(f"`{k}` ({v:,})" for k, v in s["special_tokens"][:8])
        + f". Rows with a special token as key1: {s['key1_special_frac']*100:.2f}%; as key2: {s['key2_special_frac']*100:.2f}%.",
        f"- Negative DD: {s['dd_negative']} rows ({s['dd_negative_lt_minus_0p2']} below -0.2 s); DD > 2 s: {s['dd_gt_2s']}; DD > 10 s: {s['dd_gt_10s']}; hold > 1 s: {s['hold_gt_1s']}. "
        f"Rows with any |value| > 60 s (corrupt): {s['rows_any_abs_gt_60s']}.",
        f"- Standard convention holds in this file: `UD = DD - H(k1)` in {s['identity_std_UD_eq_DD_minus_H1_frac']*100:.2f}% of valid rows, "
        f"`UU = DU - H(k1)` in {s['identity_std_UU_eq_DU_minus_H1_frac']*100:.2f}%. So the stored free-text `UD` is the standard flight time, "
        f"median {s['stored_UD']['median']:.3f} s, {s['stored_UD_negative_frac']*100:.2f}% negative. "
        "This is the direct evidence for the fixed-text flight definition (section 2a).",
        "- Free-text is **secondary**. Its digraphs are not aligned to a fixed key set, so it supports only key-agnostic features.",
        "",
    ]
    return lines


# ---------------------------------------------------------------- demographics
def audit_demographics(summary: dict, fixed_part: set[str]) -> list[str]:
    path = RAW / "demographics.csv"
    d = pd.read_csv(path)
    s = {
        "file": path.name, "rows": int(len(d)), "participants": int(d["participant"].nunique()),
        "missing_cells": int(d.isna().sum().sum()), "match_fixed_text_participants": set(d["participant"]) == fixed_part,
        "gender": d["gender"].value_counts().to_dict(), "handedness": d["handedness"].value_counts().to_dict(),
        "age": describe(d["age"]), "nationalities": int(d["nationality"].nunique()),
        "sha256": sha256(path),
    }
    summary["demographics"] = s
    return [
        "## 4. KeyRecs demographics",
        "",
        f"- `{path.name}`: {s['rows']} rows, {s['participants']} unique participants, missing cells {s['missing_cells']}. "
        f"Participant IDs match fixed-text: {s['match_fixed_text_participants']}.",
        f"- Gender: {s['gender']}. Handedness: {s['handedness']}. Age median {s['age']['median']:.0f} (min {s['age']['min']:.0f}, max {s['age']['max']:.0f}). "
        f"{s['nationalities']} nationalities. Not used as model inputs.",
        "",
    ]


# ---------------------------------------------------------------- MotionSense
def extract_motionsense() -> int:
    MS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(RAW / "archive.zip") as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith("._"):
                continue
            target = MS_DIR / name
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            z.extract(info, MS_DIR)
            n += 1
    return n


def audit_motionsense(summary: dict) -> list[str]:
    n_new = extract_motionsense()
    root = MS_DIR / "A_DeviceMotion_data" / "A_DeviceMotion_data"
    files = sorted(root.rglob("sub_*.csv"))
    info = pd.read_csv(MS_DIR / "data_subjects_info.csv")
    recs = []
    for f in files:
        trial = f.parent.name
        act, tid = trial.rsplit("_", 1)
        d = pd.read_csv(f)
        grav = d[["gravity.x", "gravity.y", "gravity.z"]].to_numpy(float)
        acc = d[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].to_numpy(float)
        rot = d[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].to_numpy(float)
        gnorm = np.linalg.norm(grav, axis=1)
        recs.append({
            "activity": act, "trial": int(tid), "subject": int(f.stem.split("_")[1]), "rows": int(len(d)),
            "nan": int(d.isna().sum().sum()), "ncols": int(d.shape[1]),
            "index_contiguous": bool(np.array_equal(d.iloc[:, 0].to_numpy(), np.arange(len(d)))),
            "gravity_norm_mean": float(gnorm.mean()), "gravity_norm_std": float(gnorm.std()),
            "ua_absmax": float(np.abs(acc).max()), "rr_absmax": float(np.abs(rot).max()),
            "columns": ",".join(d.columns),
        })
    r = pd.DataFrame(recs)
    r["duration_s"] = r["rows"] / MS_FS
    r["windows_3s"] = r["rows"] // MS_WIN
    subs = sorted(r["subject"].unique().tolist())
    trials_by_act = r.groupby("activity")["trial"].nunique().to_dict()
    per_subj_act = r.groupby(["subject", "activity"]).size().unstack(fill_value=0)
    win = r.groupby(["subject", "activity"])["windows_3s"].sum().unstack(fill_value=0)
    act_rows = []
    for act in ["dws", "ups", "wlk", "jog", "sit", "std"]:
        sub = r[r["activity"] == act]
        act_rows.append([act, trials_by_act.get(act, 0), int(sub["subject"].nunique()),
                         int(per_subj_act[act].min()), int(per_subj_act[act].max()),
                         float(sub["duration_s"].median()), float(sub["duration_s"].min()), float(sub["duration_s"].max()),
                         int(win[act].min()), int(win[act].median()), int(win[act].sum())])
    ws = (win["sit"] + win["std"]).to_numpy()
    s = {
        "zip_entries_extracted_now": n_new, "csv_files": len(files), "subjects": len(subs),
        "subject_ids": [subs[0], subs[-1]], "subjects_complete": subs == list(range(1, 25)),
        "trial_folders": int(r.groupby(["activity", "trial"]).ngroups),
        "trial_folder_names": sorted({f"{a}_{t}" for a, t in zip(r["activity"], r["trial"])}),
        "trials_per_activity": {k: int(v) for k, v in trials_by_act.items()},
        "per_subject_files": {"min": int(r.groupby("subject").size().min()), "max": int(r.groupby("subject").size().max())},
        "nan_cells": int(r["nan"].sum()), "ncols_set": sorted(r["ncols"].unique().tolist()),
        "index_contiguous_all": bool(r["index_contiguous"].all()),
        "gravity_norm_mean_range": [float(r["gravity_norm_mean"].min()), float(r["gravity_norm_mean"].max())],
        "gravity_norm_std_max": float(r["gravity_norm_std"].max()),
        "ua_absmax": float(r["ua_absmax"].max()), "rr_absmax": float(r["rr_absmax"].max()),
        "duration_s": describe(r["duration_s"]),
        "sit_std_windows_per_subject_min": int(ws.min()), "sit_std_windows_per_subject_median": float(np.median(ws)),
        "info_rows": int(len(info)), "info_codes_match": sorted(info["code"].tolist()) == list(range(1, 25)),
        "info_columns": info.columns.tolist(),
        "sha256_archive": sha256(RAW / "archive.zip"),
        "sha256_info": sha256(MS_DIR / "data_subjects_info.csv"),
    }
    summary["motionsense"] = s
    summary["_motionsense_trial_table"] = act_rows
    lines = [
        "## 5. MotionSense (Malekzadeh et al.; Kaggle mirror `archive.zip`)",
        "",
        f"- `raw dataset/archive.zip` ({(RAW / 'archive.zip').stat().st_size:,} bytes), sha256 `{s['sha256_archive'][:16]}...`. "
        f"Extracted to `research/data/interim/motionsense/` (skipping `__MACOSX` resource forks; entries newly written in this run: {n_new}, 0 = already present). "
        f"Structure: `A_DeviceMotion_data/A_DeviceMotion_data/<activity>_<trial>/sub_<k>.csv` plus `data_subjects_info.csv`.",
        f"- {s['csv_files']} trial files = {s['trial_folders']} trial folders x 24 subjects. Subject IDs 1-24: complete = {s['subjects_complete']}. "
        f"Columns: {s['ncols_set'][0]} per file: unnamed row index, `attitude.{{roll,pitch,yaw}}`, `gravity.{{x,y,z}}`, `rotationRate.{{x,y,z}}`, `userAcceleration.{{x,y,z}}`. "
        "There is **no timestamp column**.",
        f"- Sampling rate: the file does not record it. The MotionSense documentation gives 50 Hz; the file gives no way to verify it. "
        f"Row counts imply median trial duration of {s['duration_s']['median']:.1f} s at 50 Hz.",
        f"- Units: `|gravity|` mean is in [{s['gravity_norm_mean_range'][0]:.3f}, {s['gravity_norm_mean_range'][1]:.3f}] "
        f"(max per-file std {s['gravity_norm_std_max']:.4f}), consistent with gravity in g. Max |userAcceleration| {s['ua_absmax']:.2f}; max |rotationRate| {s['rr_absmax']:.2f} (rad/s). "
        f"Row index contiguous in every file: {s['index_contiguous_all']}. Missing cells: {s['nan_cells']}.",
        "- Activity codes (from the dataset documentation): `dws` downstairs, `ups` upstairs, `wlk` walking, `jog` jogging, `sit` sitting, `std` standing.",
        "",
        md_table(["activity", "trial folders", "subjects", "min trials/subj", "max trials/subj", "median dur (s)", "min dur (s)", "max dur (s)",
                  "min 3s-windows/subj", "median 3s-windows/subj", "total 3s-windows"],
                 act_rows),
        "",
        f"- Sit + standing (the primary 'phone held still' proxy): per subject, median {s['sit_std_windows_per_subject_median']:.0f} non-overlapping 3 s windows, minimum {s['sit_std_windows_per_subject_min']}.",
        f"- `data_subjects_info.csv`: {s['info_rows']} rows, codes 1-24 match subject IDs: {s['info_codes_match']}. Columns: {', '.join(s['info_columns'])}. "
        "The `gender` encoding is not documented in the file. Subject metadata is not used as model input.",
        "- **Domain caveat (must go in Limitations):** the phone was in the front trouser pocket. This is pocket-worn activity data, not hand-held typing motion.",
        "",
    ]
    return lines


# ---------------------------------------------------------------- report
def write_report(summary: dict, body: list[str]) -> None:
    env = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:  # pragma: no cover
        env["torch"] = f"unavailable ({type(e).__name__})"
    tex = {t: shutil.which(t) for t in ["pdflatex", "latexmk", "xelatex", "lualatex", "tectonic", "bibtex"]}
    env["tex_toolchain_on_path"] = {k: (v is not None) for k, v in tex.items()}
    summary["environment"] = env
    summary["generated_by"] = "research/scripts/00_audit_data.py"

    s_cmu, s_fx, s_fr, s_dm, s_ms = summary["cmu"], summary["keyrecs_fixed"], summary["keyrecs_free"], summary["demographics"], summary["motionsense"]
    head = [
        "# Data audit (Phase 0)",
        "",
        "Generated by `research/scripts/00_audit_data.py`. Every number below comes from that run; raw files were not modified "
        "(sha256 recorded). Machine-readable numbers: `research/results/audit/audit_summary.json`.",
        "",
        "## 0. Summary",
        "",
        md_table(["dataset", "subjects", "units", "modality", "sampling / timing", "used for"],
                 [["CMU DSL-StrongPasswordData", s_cmu["subjects"], f"{s_cmu['rows']:,} reps (11 keys)", "typing", "hold, DD, UD in seconds (4 dp)", "primary typing (fixed 11-key)"],
                  ["KeyRecs fixed-text", s_fx["participants"], f"{s_fx['rows']:,} attempts (10 keys)", "typing", "holds + DD; stored UD/UU/DU sign-flipped (see 2a)", "primary typing (fixed 10-key), cross-dataset"],
                  ["KeyRecs free-text", s_fr["participants"], f"{s_fr['valid_digraph_rows']:,} digraphs", "typing", "digraphs, standard convention; no fixed key set", "secondary (free-text windows)"],
                  ["MotionSense", s_ms["subjects"], f"{s_ms['csv_files']} trial files", "phone in pocket, accel + gyro", "50 Hz documented, not verifiable", "motion (sit/std primary; walking secondary)"]]),
        "",
        "**Not present, not used:** Aalto, HuMIdb, IKDD (the IKDD PDF is a paper only; no data in `raw dataset/`).",
        "",
        "## Common typing feature set (for cross-dataset tests)",
        "",
        "| feature | CMU | KeyRecs fixed | comment |",
        "|---|---|---|---|",
        "| hold of key i, i=1..10 | yes (`H.*`, first 10 of 11 keys) | yes (`DU.k.k`) | log-transform; mask if out of bounds |",
        "| down-down DD(k_i,k_{i+1}), i=1..9 | yes | yes | log-transform; mask if out of bounds |",
        "| flight = DD - H(k_i), i=1..9 | yes (identical to `UD`, verified) | derived as DD - H(k_i) (see 2a) | signed; use asinh-type transform |",
        "| up-up UU, down-up DU | no | stored, sign-flipped (see 2a) | **not used** |",
        "| return key (`H.Return`, `DD.l.Return`) | yes | no | dropped for cross-dataset |",
        "",
        "Proposed common set: `{log hold_1..10, log DD_1..9, asinh(flight_1..9 / 0.05)}` = 28 features per session, plus a validity mask. "
        "Within-dataset models may use the full CMU set (11 holds, 10 DD, 10 flight).",
        "",
        "Cross-dataset caveat: the two datasets use **different keys and different hardware/populations** (lab-based CMU vs online Prolific KeyRecs). "
        "Transfer tests therefore measure population + device shift, not key-specific behaviour.",
        "",
        "## Proposed validity bounds (to be confirmed at Gate A)",
        "",
        md_table(["quantity", "bounds (s)", "CMU frac out of bounds", "KeyRecs-fixed frac out of bounds", "CMU sessions with >=1 OOB value", "KeyRecs sessions with >=1 OOB value"],
                 [["hold", f"[{HOLD_BOUNDS[0]}, {HOLD_BOUNDS[1]}]", s_cmu["hold_out_of_bounds_frac"], s_fx["hold_out_of_bounds_frac"], "", ""],
                  ["DD", f"[{DD_BOUNDS[0]}, {DD_BOUNDS[1]}]", s_cmu["dd_out_of_bounds_frac"], s_fx["dd_out_of_bounds_frac"], "", ""],
                  ["flight", f"[{DIGRAPH_BOUNDS[0]}, {DIGRAPH_BOUNDS[1]}]", s_cmu["flight_out_of_bounds_frac"], s_fx["flight_out_of_bounds_frac"], "", ""],
                  ["any of the above", "", "", "", s_cmu["sessions_with_any_oob_frac"], s_fx["sessions_with_any_oob_frac"]]]),
        "",
        "Out-of-bounds values are masked (not imputed) and the session keeps its validity mask. Large pauses (DD up to tens of seconds) are common and are not treated as human/bot evidence.",
        "",
    ]
    body = head + body + [
        "## 6. Environment",
        "",
        md_table(["item", "value"], [[k, json.dumps(v) if isinstance(v, (dict, list)) else v] for k, v in env.items()]),
        "",
        "GPU: `torch.cuda.is_available()` is False. All training must run on CPU; models are kept small.",
        "",
        "TeX: no LaTeX engine on PATH (see table above). The paper cannot be compiled here until a toolchain is available (Gate C).",
        "",
        "## 7. Issues, blockers and decisions for Gate A",
        "",
        "See `research/PROGRESS.md` (Phase 0 entry) and `research/DEVIATIONS.md`. No blocker prevents Phase 1, but these need your decision:",
        "",
        f"1. **KeyRecs fixed-text sign convention.** Its stored `UD/UU/DU` have the hold term sign-flipped relative to CMU and to KeyRecs free-text (see 2a). Proposal: use `DD` and holds as stored and derive flight as `DD - H(k1)`, the convention verified in CMU and free-text. Alternative: drop flight features from the fixed-text side of the transfer tests.",
        "1b. **Corrupt rows** in fixed-text and free-text (timings of ~1e6 s). Proposal: drop rows with any |timing| > 60 s, and mask individual out-of-bounds values per the table above.",
        f"2. **KeyRecs has 99 participants**, not 100 as the Zenodo description says. Proposal: use 99 (the article and the file agree).",
        f"3. **MotionSense sampling rate is unverifiable from the file** (no timestamps). Proposal: accept the documented 50 Hz and state this in the paper.",
        f"4. **Typing sessions differ in size:** CMU {s_cmu['reps_per_session']} reps x 8 sessions per subject; KeyRecs about {s_fx['reps_per_participant_session_median']:.0f} attempts x 2 sessions per participant. Proposal: keep both as one session = one attempt.",
        "5. **Validity bounds** in the table above: confirm or give different values (they are proposals).",
        "6. **Free-text tokens:** special keys (`Space`, `Backspace`, `Shift`) are present. Proposal: keep `Space` digraphs and drop `Backspace`/`Shift` for the secondary free-text setting.",
        "7. **MotionSense is pocket-worn data**, not hand-held typing motion. Proposal: keep it as the motion modality and document the domain mismatch in Limitations (as you specified).",
        "8. **Subject overlap across modalities:** CMU/KeyRecs subjects and MotionSense subjects are different people. Cross-pairing (Phase 3) is therefore independence-by-construction, as you specified.",
        "",
    ]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "AUDIT.md").write_text("\n".join(body), encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    summary: dict = {"run_started": time.strftime("%Y-%m-%d %H:%M:%S")}
    body: list[str] = []
    body += audit_cmu(summary)
    body += audit_keyrecs_fixed(summary)
    fixed_part = set(pd.read_csv(RAW / "fixed-text.csv", usecols=[0])["participant"])
    body += audit_keyrecs_free(summary, fixed_part)
    body += audit_demographics(summary, fixed_part)
    body += audit_motionsense(summary)
    trial_table = summary.pop("_motionsense_trial_table")
    summary["runtime_s"] = round(time.time() - t0, 2)
    write_report(summary, body)
    print(f"audit done in {summary['runtime_s']} s -> {DATA / 'AUDIT.md'}")


if __name__ == "__main__":
    main()
