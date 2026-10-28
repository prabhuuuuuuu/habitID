"""Phase 8: paper assets. Reads ONLY results/ and writes paper_assets/ (LaTeX table fragments, figures,
and numbers.tex with \\newcommand macros used in the text). Nothing is typed in by hand."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from habitid.utils.config import PAPER_ASSETS, RESULTS, load_cfg  # noqa: E402
from habitid.utils.runtime import phase_timer  # noqa: E402

TAB = PAPER_ASSETS / "tables"
FIG = PAPER_ASSETS / "figures"
NUM = {}


def ms(d, nd=3):
    if d is None:
        return "--"
    return f"{d['mean']:.{nd}f} $\\pm$ {d['std']:.{nd}f}"


def single(d, nd=3):
    if d is None:
        return "--"
    return f"{d['mean']:.{nd}f}"


def macro(name, value):
    """LaTeX macro names may contain letters only."""
    NUM[re.sub(r"[^A-Za-z]", "", name)] = value


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tabular(cols: str, header: list[str], rows: list[list[str]], caption: str, label: str, note: str = "") -> str:
    body = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    body += [" & ".join(r) + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    out = ["\\begin{table}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", "\\small"] + body
    if note:
        out.append(f"\\\\[2pt]\\footnotesize {note}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def main_table(main):
    names = [
        ("habitid_v0", "habitID-v0 (additive score)"), ("gate_only", "Validity gate only"),
        ("lr_handcrafted", "Logistic regression (hand-crafted)"), ("gbt_handcrafted", "GBDT (hand-crafted)"),
        ("isolation_forest", "Isolation Forest (typing)"), ("one_class_svm", "One-class SVM (typing)"),
        ("scaled_manhattan", "Scaled Manhattan (typing)"), ("noprior", "Neural, no prior ($\\lambda{=}0$)"),
        ("noadv", "Neural, no adversarial training"), ("typing_only", "Typing head only"),
        ("motion_only", "Motion head only"), ("otp_only", "OTP term only"), ("habitid_net", "\\textbf{habitID-Net}"),
    ]
    rows = []
    for key, label in names:
        r = main[key]
        rows.append([label, ms(r["AUC"]), ms(r["EER"]), ms(r["TPR@1%FPR"]), ms(r["PR-AUC@1"]), ms(r["PR-AUC@0.1"]),
                     ms(r["ECE"]) if r["ECE"]["n"] else "--"])
        k = key.replace("-", "_")
        macro(f"Auc{k}", single(r["AUC"])); macro(f"AucSd{k}", single({"mean": r["AUC"]["std"]}))
        macro(f"Tpr{k}", single(r["TPR@1%FPR"]))
        macro(f"PrAuc{k}", single(r["PR-AUC@1"]))
    return tabular("lcccccc", ["Method", "AUC", "EER", "TPR@1\\%FPR", "PR-AUC (1\\%)", "PR-AUC (0.1\\%)", "ECE"],
                   rows, "Held-out detection performance (5 folds $\\times$ seeds; mean $\\pm$ std over fold-seed runs).",
                   "tab:main",
                   "Validity-gate failures are rejected before scoring for every method. PR-AUC is reweighted to the stated base rate. ECE uses temperature-scaled probabilities; n/a methods do not output probabilities.")


def bootstrap_table(boot):
    rows = []
    for name, d in boot["paired_differences"].items():
        rows.append([name.replace("-", "$-$"), f"{d['delta_AUC']:+.3f}", f"[{d['CI95'][0]:+.3f}, {d['CI95'][1]:+.3f}]",
                     f"{d['p_raw']:.4f}", f"{d['p_holm']:.4f}"])
        key = name.split(" - ")[1].replace(" ", "").replace("(", "").replace(")", "").replace("_", "").replace("-", "")
        macro(f"dAuc{key.capitalize()}", f"{d['delta_AUC']:+.3f}")
    return tabular("lcccc", ["Comparison", "$\\Delta$AUC", "95\\% cluster CI", "$p$ (raw)", "$p$ (Holm)"], rows,
                   "Paired cluster-bootstrap AUC differences (seed 0, pooled test folds).", "tab:boot",
                   f"{boot['n_resamples']} subject-cluster resamples; human clusters = typing subjects, bot clusters = generator instances.")


def lofo_table(lofo):
    rows = []
    for variant, label in (("lofo_full", "habitID-Net"), ("lofo_noadv", "No adversarial training")):
        per = lofo[variant]["per_family"]
        cells = [f"{per[L]['AUC']:.3f} / {per[L]['TPR@1%FPR']:.3f}" for L in "ABCD"]
        worst = lofo[variant]
        rows.append([label] + cells + [f"{worst['worst_AUC']:.3f} / {worst['worst_TPR@1%FPR']:.3f} ({worst['worst_family']})"])
        macro(f"Lofo{variant.split('_')[1]}Worst", f"{worst['worst_AUC']:.3f}")
        macro(f"Lofo{variant.split('_')[1]}WorstFam", worst["worst_family"])
    return tabular("lcccccc", ["Model (family held out)", "A", "B", "C", "D", "Worst family"], rows,
                   "Leave-one-attacker-family-out: AUC / TPR@1\\%FPR on the held-out family (seed 0, pooled folds).",
                   "tab:lofo")


def conformal_table(conf, conf_lofo):
    rows = []
    for eps, d in conf.items():
        rows.append([f"{eps}", single(d["coverage_human"]), single(d["coverage_bot"]), single(d["abstention"]),
                     single(d["coverage_bot_by_letter"].get("A")), single(d["coverage_bot_by_letter"].get("B")),
                     single(d["coverage_bot_by_letter"].get("C")), single(d["coverage_bot_by_letter"].get("D")),
                     single(d["coverage_bot_by_letter"].get("O"))])
    t1 = tabular("lccccccccc", ["$\\epsilon$", "cov.\\ human", "cov.\\ bot", "abst.", "A", "B", "C", "D", "O"], rows,
                 "In-distribution conformal coverage and abstention (habitID-Net, 3 seeds $\\times$ 5 folds).",
                 "tab:conf")
    rows2 = []
    for L in "ABCD":
        d = conf_lofo[L]["0.1"]
        rows2.append([L, single({"mean": d["coverage_held_out_family_bots"]}), single({"mean": d["coverage_human"]}),
                      single({"mean": d["abstention"]})])
        macro(f"ConfLofo{L}Cov", f"{d['coverage_held_out_family_bots']:.3f}")
    t2 = tabular("lccc", ["Held-out family", "cov.\\ held-out bots ($\\epsilon{=}0.1$)", "cov.\\ human", "abst."], rows2,
                 "Conformal coverage under attacker shift: thresholds calibrated without the held-out family.",
                 "tab:conflofo")
    return t1 + "\n" + t2


def weights_table(w):
    rows = []
    for name in ["full", "noprior", "lam10", "noadv", "typing_only", "motion_only", "otp_only"]:
        if name not in w:
            continue
        r = w[name]
        a = r["alpha_mean"] + [None] * 3
        act = r["active"]
        cells = []
        for sig in ("typing", "motion", "otp"):
            if sig in act:
                cells.append(f"{r['alpha_mean'][act.index(sig)]:.3f}")
            else:
                cells.append("--")
        a0 = {"typing": 25 / 60, "motion": 20 / 60, "otp": 15 / 60}
        cells0 = [f"{a0[s]:.3f}" if s in act else "--" for s in ("typing", "motion", "otp")]
        rows.append([name.replace("_", " "), f"{r['lam_prior']:g}"] + cells + [f"{r['prior_distance_mean']:.3f}"] + [f"{r['n']}"])
        if name == "full":
            macro("AlphaTyping", cells[0]); macro("AlphaMotion", cells[1]); macro("AlphaOtp", cells[2])
            macro("PriorDist", f"{r['prior_distance_mean']:.3f}")
            macro("AlphaZeroTyping", cells0[0]); macro("AlphaZeroMotion", cells0[1]); macro("AlphaZeroOtp", cells0[2])
    t = tabular("lcccccc", ["Variant", "$\\lambda_{\\text{prior}}$", "$\\alpha_{\\text{typ}}$", "$\\alpha_{\\text{mot}}$",
                             "$\\alpha_{\\text{otp}}$", "$\\|\\alpha-\\alpha^0\\|$", "runs"], rows,
                "Learned fusion weights (mean over folds and seeds). $\\alpha^0 = (0.417, 0.333, 0.250)$.", "tab:alpha")
    return t


def ablation_table(abl, dstar):
    rows = []
    ind = abl["in_distribution_worst_letter"]
    for name, d in ind.items():
        rows.append([name, f"{d['worst_letter_AUC_in_dist']:.3f} ({d['worst_letter_in_dist']})"])
    rows.append(["LOFO worst case: habitID-Net", f"{abl['lofo_worst_case']['full']:.3f} ({abl['lofo_worst_case']['full_worst_family']})"])
    rows.append(["LOFO worst case: no adversarial training", f"{abl['lofo_worst_case']['noadv']:.3f} ({abl['lofo_worst_case']['noadv_worst_family']})"])
    g = abl["no_generatorD_augmentation"]
    rows.append(["No generator-D augmentation (worst letter, D held out)", f"{g['worst_letter_AUC']:.3f} ({g['worst_letter']})"])
    fx = abl["fixed_threshold_verdict_p0.5"]
    rows.append(["Fixed threshold $p\\geq0.5$ (no conformal): bot flag rate / human flag rate",
                 f"{fx['bot_detection_rate']:.3f} / {fx['human_false_flag_rate']:.3f}"])
    return tabular("lc", ["Variant", "Worst-case AUC (family)"], rows,
                   "Ablations. In-distribution rows: minimum per-letter AUC of the seed-0 model. LOFO rows: true held-out worst case.",
                   "tab:abl",
                   "Calibration does not change AUC (temperature scaling is monotone); its effect is in ECE (Table~\\ref{tab:calib}).")


def dstar_table(dstar):
    """Adaptive stress test: reported separately from the main comparison."""
    rows = []
    for name, label in (("full", "habitID-Net (trained with all families)"), ("lofo_full_D", "habitID-Net, family D held out")):
        if name not in dstar:
            continue
        a = dstar[name]["AUC_humans_vs_Dstar"]
        f = dstar[name]["Dstar_flag_rate_p0.5"]
        rows.append([label, ms(a, 3), single(f)])
        if name == "full":
            macro("DstarAuc", f"{a['mean']:.3f}"); macro("DstarFlag", f"{f['mean']:.3f}")
    return tabular("lcc", ["Model", "AUC, humans vs adaptive D* (best-of-8)", "D* flagged as human (p$\\geq$0.5)"], rows,
                   "Adaptive stress test (not part of the main comparison). D* selects, among 8 typing candidates from family D, the one the frozen detector scores most human-like.",
                   "tab:dstar")


def baseline_table(bs):
    rows = []
    names = {"habitid_v0": "habitID-v0", "lr_handcrafted": "Logistic regression", "gbt_handcrafted": "GBDT",
             "isolation_forest": "Isolation Forest", "one_class_svm": "One-class SVM", "scaled_manhattan": "Scaled Manhattan"}
    for k, label in names.items():
        r = bs[k]
        pf = r["per_family_AUC_vs_human"]
        rows.append([label, f"{r['AUC_pooled_5_folds']:.3f} [{r['AUC_CI95'][0]:.3f}, {r['AUC_CI95'][1]:.3f}]"]
                    + [f"{pf[L]:.3f}" for L in "ABCDO"])
    return tabular("lcccccc", ["Method", "AUC pooled (95\\% CI)", "A", "B", "C", "D", "O"], rows,
                   "Baselines: pooled five-fold AUC and AUC per attacker family (family vs human). Score direction fixed a priori (higher = more bot).",
                   "tab:baseline_families",
                   "Family A = naive fixed intervals; B = humanized; C = replay; D = learned; O = OTP-only.")


def typing_diag_table(td, ts=None):
    rows = []
    names = [("a_fused_habitID_net_with_OTP", "habitID-Net (fused, with OTP)"), ("b_typing_head_only", "typing head only"),
             ("c_gbdt_typing_features_only", "GBDT, typing features only"), ("d_lr_typing_features_only", "LR, typing features only")]
    for key, label in names:
        rows.append([label] + [f"{td[key][L]:.3f}" for L in "ABCD"])
    dist = td["typing_evidence_contribution_fused"]
    rows.append(["typing evidence $\\alpha_{\\text{typ}}g_{\\text{typ}}$, median: humans / typing-only bots",
                 f"{dist['human']['median']:.2f} / {dist['typing_only_bot']['median']:.2f}", "", "", ""])
    if ts is not None:
        # Post-hoc exploratory row group (not the main model; seed 0; see Discussion).
        pf = ts["typing_summary_full"]["per_family_typing_only_AUC"]
        rows.append(["\\emph{Exploratory, post hoc:} habitID-Net with typing summary statistics"] + [f"{pf[L]:.3f}" for L in "ABCD"])
        e = ts["typing_summary_full"]
        rows.append(["\\emph{Exploratory:} typing evidence median, humans / typing-only bots",
                     f"{e['typing_evidence_median_human']:.2f} / {e['typing_evidence_median_typing_only_bot']:.2f}", "", "", ""])
    macro("AlphaTypingFull", f"{td['alpha_typing_full']:.3f}")
    return tabular("lcccc", ["Model (typing-only bots of one family vs humans)", "A", "B", "C", "D"], rows,
                   "Typing-only bots: AUC per attacker family (seed 0). Only typing is automated; motion and OTP are human.",
                   "tab:typing_diag",
                   "GBDT and LR are fitted per fold on the training split using typing summary features only. The row marked exploratory is a post-hoc variant, not the main model; without-OTP version not trained.")


def headline_macros(tables, diag):
    """Numbers quoted in the abstract and prose, generated from results/ (no hand-typed values)."""
    bs = tables["bootstrap"]["AUC_seed0_pooled"]
    for key, name in (("habitid_net", "Hab"), ("gbt_handcrafted", "Gbt"), ("noadv", "NoAdv"), ("habitid_v0", "V0")):
        macro(f"Pool{name}Auc", f"{bs[key]['AUC']:.3f}")
        macro(f"Pool{name}Lo", f"{bs[key]['AUC_CI95'][0]:.3f}")
        macro(f"Pool{name}Hi", f"{bs[key]['AUC_CI95'][1]:.3f}")
    fl = tables["framing_lofo"]
    macro("LofoHabFamA", f"{fl['habitID_full']['A']:.3f}")
    macro("LofoGbtFamA", f"{fl['gbdt']['A']:.3f}")
    macro("LofoGbtFamB", f"{fl['gbdt']['B']:.3f}")
    macro("LofoHabFamB", f"{fl['habitID_full']['B']:.3f}")
    conf = tables["conformal_in_dist_full"]
    macro("AbstainLo", f"{100 * conf['0.05']['abstention']['mean']:.0f}")
    macro("AbstainHi", f"{100 * conf['0.01']['abstention']['mean']:.0f}")
    macro("CovHumanFive", f"{conf['0.05']['coverage_human']['mean']:.3f}")
    macro("CovBotFive", f"{conf['0.05']['coverage_bot']['mean']:.3f}")
    lc = tables["conformal_lofo"]["A"]
    macro("LofoCovTen", f"{lc['0.1']['coverage_held_out_family_bots']:.3f}")
    macro("LofoCovFive", f"{lc['0.05']['coverage_held_out_family_bots']:.3f}")
    cal = tables["calibration"]["habitID-Net"]
    macro("EceRawFull", f"{cal['ECE_uncal']['mean']:.3f}"); macro("EceTsFull", f"{cal['ECE_TS']['mean']:.3f}")
    gbt_ece = tables["main"]["gbt_handcrafted"]["ECE"]["mean"]
    macro("EceGbt", f"{gbt_ece:.3f}")
    for name, key in (("CmuToKr", "CMU_to_KeyRecs"), ("KrToCmu", "KeyRecs_to_CMU")):
        for mode, suffix in (("on", "On"), ("off", "Off"), ("pds", "Pds")):
            d = diag[f"{key}_{mode}"]
            macro(f"Xfer{name}{suffix}", f"{d['cross']['AUC']:.3f}")
        macro(f"XferIn{name}", f"{diag[f'{key}_on']['in_domain']['AUC']:.3f}")


def typing_summary_table(ts, td):
    """Post-hoc exploratory row group (not the main model). Seed 0, five folds."""
    rows = []
    def row(label, d, extra_typing=None, ece_val=None):
        pf = d["per_family_typing_only_AUC"] if "per_family_typing_only_AUC" in d else extra_typing
        fam = [f"{pf[L]:.3f}" for L in "ABCD"]
        pooled = f"{d['pooled_AUC_with_OTP']:.3f} [{d['pooled_AUC_CI95'][0]:.3f}, {d['pooled_AUC_CI95'][1]:.3f}]" if "pooled_AUC_with_OTP" in d else "--"
        e = ece_val if ece_val is not None else d.get("ECE_temperature_scaled")
        ev = f"{d['typing_evidence_median_human']:.2f} / {d['typing_evidence_median_typing_only_bot']:.2f}" if "typing_evidence_median_human" in d else "--"
        rows.append([label] + fam + [pooled, f"{e:.3f}" if e is not None else "--", ev])
    row("habitID-Net (main, full)", ts["original_full"])
    row("\\emph{Exploratory: typing summary statistics}", ts["typing_summary_full"])
    row("GBDT, all features", ts["gbdt_all_features"], ece_val=ts["gbdt_all_features"].get("ECE_probabilities_as_output"))
    row("GBDT, typing features only", {}, extra_typing=td["c_gbdt_typing_features_only"], ece_val=None)
    macro("TsTypingA", f"{ts['typing_summary_full']['per_family_typing_only_AUC']['A']:.3f}")
    macro("TsPooled", f"{ts['typing_summary_full']['pooled_AUC_with_OTP']:.3f}")
    macro("TsEce", f"{ts['typing_summary_full']['ECE_temperature_scaled']:.3f}")
    return tabular("lccccccc", ["Model", "A", "B", "C", "D", "pooled AUC with OTP (95\\% CI)", "ECE", "typing evidence median (human / typing-only bot)"],
                   rows, "Post-hoc exploratory typing-encoder variant (seed 0, five folds), next to the original model and GBDT. Typing-only-bot AUC per family.",
                   "tab:typsum",
                   "Post-hoc exploratory, not the main model and not tuned on test metrics. Without-OTP version of the variant not trained. GBDT ECE uses its probabilities as output.")


def exploratory_macros(ts):
    """Numbers for the post-hoc exploratory paragraph (Discussion). Labelled post hoc in the paper."""
    e, o, g = ts["typing_summary_full"], ts["original_full"], ts["gbdt_all_features"]
    for L in "ABCD":
        macro(f"TsFam{L}", f"{e['per_family_typing_only_AUC'][L]:.3f}")
        macro(f"OrigFam{L}", f"{o['per_family_typing_only_AUC'][L]:.3f}")
    macro("TsPooled", f"{e['pooled_AUC_with_OTP']:.3f}")
    macro("TsEce", f"{e['ECE_temperature_scaled']:.3f}")
    macro("OrigPooledEce", f"{o['ECE_temperature_scaled']:.3f}")
    macro("GbtPooledEce", f"{g['ECE_probabilities_as_output']:.3f}")
    macro("GbtPooledAucP", f"{g['pooled_AUC_with_OTP']:.3f}")
    macro("OrigPooledAucP", f"{o['pooled_AUC_with_OTP']:.3f}")


def per_type_table(pt):
    rows = []
    for key, label in (("full", "habitID-Net (with OTP)"), ("noprior", "no prior"), ("noadv", "no adversarial training"),
                       ("typing_only", "typing head only"), ("motion_only", "motion head only"), ("otp_only", "OTP term only")):
        r = pt[key]
        rows.append([label, ms(r.get("fully_automated")), ms(r.get("typing_bot")), ms(r.get("motion_bot")),
                     ms(r.get("otp_bot_by_construction"))])
        if key == "full":
            macro("PerTypeFullAll", single(r.get("fully_automated"))); macro("PerTypeFullTyping", single(r.get("typing_bot")))
            macro("PerTypeFullMotion", single(r.get("motion_bot"))); macro("PerTypeFullOtp", single(r.get("otp_bot_by_construction")))
            macro("ShareSingleSignal", f"{100 * r.get('share_single_signal_bots')['mean']:.0f}")
    return tabular("lcccc", ["Model", "fully automated", "typing-only bots", "motion-only bots", "OTP-only bots (by construction)"],
                   rows, "AUC of humans versus each automated type (mean $\\pm$ std over three seeds; OTP-only by construction).",
                   "tab:perType",
                   "Single-signal bots are about 60\\% of automated sessions, which caps pooled AUC.")


def no_otp_table(no):
    rows = []
    for key, label in (("neural:no_otp", "habitID-Net, typing + motion"), ("neural:motion_only", "motion head only"),
                       ("neural:typing_only", "typing head only"), ("base:lr_no_otp", "logistic regression"),
                       ("base:gbt_no_otp", "GBDT"), ("base:habitid_v0_no_otp", "habitID-v0 without OTP term")):
        kind, name = key.split(":")
        r = no[name]
        rows.append([label, ms(r["AUC"]), ms(r["EER"]), ms(r["TPR@1%FPR"]), ms(r["ECE"]) if r["ECE"]["n"] else "--", str(r["n_runs"])])
        if name == "no_otp":
            macro("NoOtpAuc", single(r["AUC"])); macro("NoOtpAucSd", f"{r['AUC']['std']:.3f}")
        if name == "gbt_no_otp":
            macro("GbtNoOtpAuc", single(r["AUC"])); macro("GbtNoOtpAucSd", f"{r['AUC']['std']:.3f}")
        if name == "lr_no_otp":
            macro("LrNoOtpAuc", single(r["AUC"]))
    return tabular("lccccc", ["Method (no OTP term)", "AUC", "EER", "TPR@1\\%FPR", "ECE", "runs"], rows,
                   "Primary comparison without the assumed OTP term. habitID-Net is seed 0 only (five folds); baselines as listed.",
                   "tab:noOtp")


def cap_table(ver):
    rows = []
    for v, d in ver["per_variant"].items():
        n = d["runs"]
        rows.append([v.replace("_", " "), str(n), str(n - d["hit_epoch_cap_at_best"]), str(d["hit_epoch_cap_at_best"]),
                     f"{d['hit_epoch_cap_at_best'] / n:.2f}"])
    return tabular("lcccc", ["Variant", "runs", "stopped by patience", "hit 24-epoch cap", "fraction capped"], rows,
                   "Epoch-cap status of finished runs. Capped runs were still improving; see the limitations.", "tab:cap")


def audit_table(audit):
    rows = [["CMU DSL-StrongPasswordData", str(audit["cmu"]["rows"]), f"{audit['cmu']['identity_DD_minus_UD_eq_H1_frac']:.4f}",
             f"{audit['cmu']['flight_negative_frac']:.4f}"],
            ["KeyRecs fixed-text", str(audit["keyrecs_fixed"]["rows"]),
             f"{audit['keyrecs_fixed']['identity_std_UD_eq_DD_minus_H1_frac']:.4f} (standard; stored = 0)",
             f"{audit['keyrecs_fixed']['flight_reconstructed_negative_frac']:.4f} (DD$-$H(k1))"],
            ["KeyRecs free-text", str(audit["keyrecs_free"]["valid_digraph_rows"]),
             f"{audit['keyrecs_free']['identity_std_UD_eq_DD_minus_H1_frac']:.4f}",
             f"{audit['keyrecs_free']['flight_reconstructed_negative_frac']:.4f}"]]
    return tabular("lccc", ["Dataset", "rows", "fraction with standard $\\mathrm{UD}=\\mathrm{DD}-H(k_1)$", "flight $<0$ (fraction)"],
                   rows, "Audit: digraph convention check (Appendix~A of the paper, from the Phase 0 audit).", "tab:audit")


def transfer_diag_table(diag):
    rows = []
    for key, d in diag.items():
        src, _, tgt, mode = key.split("_")[0], None, key.split("_")[2], key.split("_")[3]
        rows.append([f"{src} $\\to$ {tgt}", {"on": "flight on, source stats", "off": "flight off",
                                             "pds": "per-dataset stats"}[mode],
                     f"{d['in_domain']['AUC']:.3f}", f"{d['cross']['AUC']:.3f}", f"{d['cross']['TPR@1%FPR']:.3f}",
                     f"{d['cross']['human_logit_median']:.2f}", f"{d['cross']['bot_logit_median']:.2f}"])
        macro(f"Xfer{src}{tgt}{mode}Cross", f"{d['cross']['AUC']:.3f}")
        macro(f"Xfer{src}{tgt}{mode}In", f"{d['in_domain']['AUC']:.3f}")
    return tabular("llccccc", ["direction", "setting", "in-domain AUC (control)", "cross-dataset AUC", "cross TPR@1\\%FPR",
                               "median logit, target humans", "median logit, attackers"], rows,
                   "Cross-dataset typing transfer diagnostics (typing-only detector, seed 0 only).", "tab:transfer",
                   "Positive class = human; higher logit = more human. Attackers are fitted on the source training split. Per-dataset stats are computed from unlabelled target sessions.")


def transfer_figure(diag):
    import numpy as np
    z = np.load(RESULTS / "transfer_diag" / "CMU_to_KeyRecs_on.npz")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bins = np.linspace(-18, 4, 60)
    ax.hist(z["in_logit"][z["in_label"] == 1], bins=bins, density=True, alpha=0.5, label="CMU humans (held-out subj.)")
    ax.hist(z["cross_logit"][z["cross_label"] == 1], bins=bins, density=True, alpha=0.5, label="KeyRecs humans")
    ax.hist(z["cross_logit"][z["cross_label"] == 0], bins=bins, density=True, alpha=0.5, label="attackers A--D (on KeyRecs)")
    ax.set_xlabel("logit (higher = more human)", fontsize=8); ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=6); fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "transfer_scores.pdf"); fig.savefig(FIG / "transfer_scores.png", dpi=200); plt.close(fig)


def transfer_table(tr):
    rows = []
    for key, d in tr["transfer"].items():
        rows.append([key.replace("->", " $\\to$ ").replace("|", "; "),
                     ms(d["in_domain"]["AUC"], 3), ms(d["cross_domain"]["AUC"], 3), ms(d["cross_domain"]["TPR@1%FPR"], 3),
                     ms(d["cross_domain"]["ECE"], 3)])
        if "flight=off" in key:
            macro("XferFlightOffCross", f"{d['cross_domain']['AUC']['mean']:.3f}")
        else:
            macro("XferFlightOnCross", f"{d['cross_domain']['AUC']['mean']:.3f}")
    return tabular("lcccc", ["Direction; flight features", "in-domain AUC", "cross-domain AUC", "cross TPR@1\\%FPR", "cross ECE"],
                   rows, "Cross-dataset typing transfer (typing-only detector; 3 seeds).", "tab:transfer",
                   "Attackers are fitted on source-train humans only; cross-domain humans are the target dataset.")


def otp_table(tr):
    o = tr["otp_sensitivity"]
    rows = []
    for cell, d in o["grid"].items():
        rows.append([cell.replace("|", "; ").replace("_", " "), single(d["AUC"]), single(d["TPR@1%FPR"])])
    s = o["summary"]
    macro("OtpAucLo", f"{s['AUC_range'][0]:.3f}"); macro("OtpAucHi", f"{s['AUC_range'][1]:.3f}")
    macro("OtpTprLo", f"{s['TPR@1%FPR_range'][0]:.3f}"); macro("OtpTprHi", f"{s['TPR@1%FPR_range'][1]:.3f}")
    return tabular("lcc", ["OTP grid cell (true world)", "AUC", "TPR@1\\%FPR"], rows,
                   "OTP-model sensitivity: true-world OTP parameters perturbed; detector keeps nominal parameters (pooled folds, seed 0).",
                   "tab:otp")


def ondevice_table(od):
    s = od["summary"]
    lf, l8 = od["latency"]["fp32"], od["latency"]["int8"]
    rows = [["fp32 ONNX", f"{s['size_kb_fp32']:.1f}", f"{lf['median_ms']:.3f}", f"{lf['p95_ms']:.3f}", "--", "--"],
            ["int8 dynamic", f"{s['size_kb_int8']:.1f}", f"{l8['median_ms']:.3f}", f"{l8['p95_ms']:.3f}",
             f"{s['AUC_change_int8_minus_fp32']:+.4f}", f"{s['decision_agreement_mean']:.4f}"]]
    macro("OnnxKbFp", f"{s['size_kb_fp32']:.0f}"); macro("OnnxKbInt", f"{s['size_kb_int8']:.0f}")
    macro("OnnxMedFp", f"{lf['median_ms']:.2f}"); macro("OnnxP95Fp", f"{lf['p95_ms']:.2f}")
    macro("OnnxMedInt", f"{l8['median_ms']:.2f}"); macro("OnnxP95Int", f"{l8['p95_ms']:.2f}")
    macro("OnnxAucDelta", f"{s['AUC_change_int8_minus_fp32']:+.4f}")
    return tabular("lccccc", ["Format", "size (KB)", "median (ms)", "p95 (ms)", "$\\Delta$AUC vs fp32", "decision agreement"],
                   rows, "On-device cost on this CPU (onnxruntime, 1 thread, single session, N=1000). Not a mobile measurement.",
                   "tab:ondevice")


def cv_table(cv):
    rows = []
    for r in cv:
        rows.append([r["dataset"], str(r["n_keys"]), f"{r['median_cv_human']:.3f}", f"{r['median_boot_se_cv']:.3f}",
                     f"{r['share_human_cv_in_0.2_1.0']:.3f}",
                     f"{r['auc_cv_human_vs_A']:.3f}",
                     f"{r['auc_cv_human_vs_B']:.3f} [{r['auc_ci95_B'][0]:.3f}, {r['auc_ci95_B'][1]:.3f}]"])
    return tabular("llccccc", ["dataset", "keys $n$", "median CV (human)", "median boot.\\ SE", "share CV in [0.2,1]",
                               "AUC(CV), human vs A", "AUC(CV), human vs B (95\\% CI)"], rows,
                   "Flight-time CV reliability as a function of the number of keys. Family A is naive (fixed intervals); "
                   "family B is fitted to human marginals, so its near-0.5 AUC is partly by construction.", "tab:cv",
                   "AUC direction fixed a priori (higher CV = human). Values below 0.5 mean the reverse. Descriptive, not a system-level estimate.")


def counts_table(typ_counts, mot_counts, meta):
    rows = []
    for ds in ("CMU", "KeyRecs-fixed"):
        c = typ_counts[ds]
        rows.append([ds, f"{c['rows_raw']:,}", f"{c['corrupt_dropped']}", f"{c['out_of_bounds_dropped']:,}",
                     f"{c['kept']:,}", str(c["subjects_kept"])])
    macro("TypKept", f"{typ_counts['total_kept']:,}")
    macro("MotWindows", f"{mot_counts['windows']:,}")
    t1 = tabular("lccccc", ["dataset", "sessions (raw)", "corrupt dropped", "out-of-bounds dropped", "kept", "subjects"],
                 rows, "Typing sessions after validity filtering (10-key canonical representation).", "tab:counts")
    rows2 = []
    for m in meta:
        r = m["roles"]
        rows2.append([str(m["fold"]), f"{r['train']['humans']}/{r['train']['bots']}", f"{r['cal']['humans']}/{r['cal']['bots']}",
                      f"{r['test']['humans']}/{r['test']['bots']}", str(r["test"]["typing_pool_subjects"]),
                      str(r["test"]["motion_subjects"])])
    t2 = tabular("lccccc", ["fold", "train h/b", "cal h/b", "test h/b", "test typing subjects", "test motion subjects"],
                 rows2, "Per-fold session counts (humans / automated).", "tab:folds")
    return t1 + "\n" + t2


def figures(main_w, cal, conf, conf_lofo, cv, w):
    FIG.mkdir(parents=True, exist_ok=True)
    rel = cal["reliability_full_seed0"]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for a, key, title in ((ax[0], "uncalibrated", "before temperature scaling"),
                          (ax[1], "temperature_scaled", "after temperature scaling")):
        conf_v, acc, cnt = (rel[key]["conf"], rel[key]["acc"], rel[key]["count"])
        xs = [i / 15 + 1 / 30 for i in range(15)]
        a.plot([0, 1], [0, 1], "k--", lw=0.8)
        a.bar(xs, [0 if c != c else c for c in acc], width=1 / 15, alpha=0.6, edgecolor="k", lw=0.3)
        a.set_title(title, fontsize=8); a.set_xlabel("predicted P(human)", fontsize=8)
    ax[0].set_ylabel("observed human fraction", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.pdf"); fig.savefig(FIG / "reliability.png", dpi=200); plt.close(fig)

    eps = list(conf.keys())
    fig, a = plt.subplots(figsize=(3.4, 2.6))
    a.plot([float(e) for e in eps], [conf[e]["coverage_bot"]["mean"] for e in eps], "o-", label="bot coverage")
    a.plot([float(e) for e in eps], [conf[e]["coverage_human"]["mean"] for e in eps], "s-", label="human coverage")
    a.plot([float(e) for e in eps], [1 - float(e) for e in eps], "k:", label="nominal $1-\\epsilon$")
    a.set_xlabel("$\\epsilon$", fontsize=8); a.set_ylabel("empirical coverage", fontsize=8); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "conformal_coverage.pdf"); plt.close(fig)

    fig, a = plt.subplots(figsize=(3.4, 2.6))
    names = [n for n in ["full", "noprior", "noadv"] if n in w]
    labels = {"full": "habitID-Net", "noprior": "no prior", "noadv": "no adv."}
    xs = [0, 1, 2]
    for i, n in enumerate(names):
        a.bar([x + 0.25 * i for x in xs], w[n]["alpha_mean"][:3] + [0] * (3 - len(w[n]["alpha_mean"][:3])),
              width=0.25, label=labels[n])
    a.set_xticks([x + 0.25 for x in xs]); a.set_xticklabels(["typing", "motion", "OTP"], fontsize=8)
    a.set_ylabel("learned $\\alpha$", fontsize=8); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "alpha_weights.pdf"); plt.close(fig)

    fig, a = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for dname in sorted({r["dataset"] for r in cv}):
        rr = [r for r in cv if r["dataset"] == dname]
        a[0].plot([r["n_keys"] for r in rr], [r["median_boot_se_cv"] for r in rr], "o-", label=dname)
        a[1].plot([r["n_keys"] for r in rr], [r["auc_cv_human_vs_B"] for r in rr], "o-", label=dname)
    a[0].set_ylabel("median bootstrap SE of CV", fontsize=8); a[1].set_ylabel("AUC(CV), human vs B", fontsize=8)
    for x in a:
        x.set_xlabel("keys $n$", fontsize=8); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "cv_vs_keys.pdf"); plt.close(fig)


def main():
    cfg = load_cfg()
    with phase_timer("phase8_paper_assets"):
        tables = json.loads((RESULTS / "tables" / "main_eval.json").read_text(encoding="utf-8"))
        diag = json.loads((RESULTS / "tables" / "transfer_diag.json").read_text(encoding="utf-8"))
        tr = {}
        if (RESULTS / "tables" / "otp_sensitivity.json").exists():
            tr.update(json.loads((RESULTS / "tables" / "otp_sensitivity.json").read_text(encoding="utf-8")))
        audit = json.loads((RESULTS / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
        ver = json.loads((RESULTS / "tables" / "verification.json").read_text(encoding="utf-8"))
        cv = json.loads((RESULTS / "cv_reliability" / "cv_reliability.json").read_text(encoding="utf-8"))
        od = json.loads((RESULTS / "ondevice" / "ondevice.json").read_text(encoding="utf-8"))
        typ_counts = json.loads((RESULTS / "phase1" / "typing_counts.json").read_text(encoding="utf-8"))
        mot_counts = json.loads((RESULTS / "phase1" / "motion_counts.json").read_text(encoding="utf-8"))
        meta = json.loads((RESULTS / "phase2" / "fold_meta.json").read_text(encoding="utf-8"))
        probe = json.loads((RESULTS / "timing" / "probe.json").read_text(encoding="utf-8"))

        write(TAB / "main.tex", main_table(tables["main"]))
        write(TAB / "bootstrap.tex", bootstrap_table(tables["bootstrap"]))
        write(TAB / "lofo.tex", lofo_table(tables["lofo"]))
        write(TAB / "conformal.tex", conformal_table(tables["conformal_in_dist_full"], tables["conformal_lofo"]))
        write(TAB / "weights.tex", weights_table(tables["weights"]))
        write(TAB / "ablations.tex", ablation_table(tables["ablations"], tables["dstar"]))
        write(TAB / "dstar.tex", dstar_table(tables["dstar"]))
        write(TAB / "baseline_families.tex", baseline_table(tables["baseline_surprises"]))
        write(TAB / "cap.tex", cap_table(ver))
        write(TAB / "typing_diag.tex", typing_diag_table(
            json.loads((RESULTS / "tables" / "typing_only_diagnosis.json").read_text(encoding="utf-8")),
            json.loads((RESULTS / "tables" / "typing_summary_exploratory.json").read_text(encoding="utf-8"))))
        write(TAB / "per_type.tex", per_type_table(tables["per_type"]))
        exploratory_macros(json.loads((RESULTS / "tables" / "typing_summary_exploratory.json").read_text(encoding="utf-8")))

        headline_macros(tables, diag)
        write(TAB / "no_otp.tex", no_otp_table(tables["no_otp"]))
        write(TAB / "audit.tex", audit_table(audit))
        write(TAB / "transfer.tex", transfer_diag_table(diag))
        transfer_figure(diag)
        if "otp_sensitivity" in tr:
            write(TAB / "otp.tex", otp_table(tr))
        write(TAB / "ondevice.tex", ondevice_table(od))
        write(TAB / "cv.tex", cv_table(cv))
        write(TAB / "counts.tex", counts_table(typ_counts, mot_counts, meta))

        cal = tables["calibration"]
        macro("EceUncalFull", f"{cal['habitID-Net']['ECE_uncal']['mean']:.3f}")
        macro("EceTsFull", f"{cal['habitID-Net']['ECE_TS']['mean']:.3f}")
        macro("TempFull", f"{cal['habitID-Net']['T']['mean']:.2f}")
        write(TAB / "calibration.tex", tabular(
            "lcccc", ["Variant", "ECE (raw)", "ECE (temp.\\ scaled)", "$T$"],
            [[k, single(cal[k]["ECE_uncal"]), single(cal[k]["ECE_TS"]), single(cal[k]["T"], 2)]
             for k in ("habitID-Net", "no prior", "no adv.")],
            "Calibration (ECE, 15 bins) before and after temperature scaling (mean over folds and seeds).", "tab:calib"))
        write(TAB / "verdicts.tex", tabular(
            "lccc", ["Ground truth", "VERIFIED", "REVIEW", "SUSPICIOUS"],
            [["human"] + [ms(tables["verdicts_full_eps0.1"]["0.1"]["human"][v], 3) for v in ("VERIFIED", "REVIEW", "SUSPICIOUS")],
             ["automated"] + [ms(tables["verdicts_full_eps0.1"]["0.1"]["bot"][v], 3) for v in ("VERIFIED", "REVIEW", "SUSPICIOUS")]],
            "Verdict distribution of habitID-Net at $\\epsilon{=}0.1$ (in-distribution test, gate-passing sessions).", "tab:verdicts"))
        v0 = tables["v0_verdicts"]
        write(TAB / "v0_verdicts.tex", tabular(
            "lccc", ["Ground truth", "VERIFIED", "REVIEW", "SUSPICIOUS"],
            [["human"] + [ms(v0["human"][v], 3) for v in ("VERIFIED", "REVIEW", "SUSPICIOUS")],
             ["automated"] + [ms(v0["bot"][v], 3) for v in ("VERIFIED", "REVIEW", "SUSPICIOUS")]],
            "habitID-v0 verdict distribution (fixed thresholds 70/45).", "tab:v0verdicts"))
        write(TAB / "probe.tex", tabular("lcc", ["quantity", "value", "unit"],
                                         [["projected total runtime", f"{probe['projected_total_h']:.2f}", "h"],
                                          ["budget", f"{probe['budget_h']:.1f}", "h"]],
                                         "Runtime projection before the full runs.", "tab:probe"))
        macro("ProbeHours", f"{probe['projected_total_h']:.2f}")

        figures(tables["weights"], tables["calibration"], tables["conformal_in_dist_full"], tables["conformal_lofo"],
                cv, tables["weights"])

        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in NUM.items() if v != ""]
        write(PAPER_ASSETS / "numbers.tex", "% generated by scripts/10_make_paper_assets.py from results/ only\n" + "\n".join(lines) + "\n")
        write(PAPER_ASSETS / "README.md", "Generated by research/scripts/10_make_paper_assets.py. Do not edit by hand.\n")
    print(f"paper assets written to {PAPER_ASSETS}")


if __name__ == "__main__":
    main()
