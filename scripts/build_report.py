"""
Builds the one-page Word report from report/sim_summary.json + sim_results.csv.
Run:  python scripts/build_report.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
summary = json.loads((REP / "sim_summary.json").read_text())
rows = list(csv.DictReader((REP / "sim_results.csv").open()))

# ── Chart ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.15), dpi=200)

gen = [int(r["total"]) for r in rows if r["label"] == "genuine"]
fra = [int(r["total"]) for r in rows if r["label"] == "fraud"]
bins = range(0, 101, 5)
ax1.hist(gen, bins=bins, alpha=0.75, color="#2e7d32", label="Genuine (n=1000)")
ax1.hist(fra, bins=bins, alpha=0.75, color="#c62828", label="Fraud (n=1000)")
ax1.axvline(70, color="k", ls="--", lw=0.8)
ax1.axvline(45, color="k", ls=":", lw=0.8)
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)
ax1.text(70.5, ax1.get_ylim()[1] * 0.95, "VERIFIED ≥70", fontsize=5.5, va="top")
ax1.text(45.5, ax1.get_ylim()[1] * 0.95, "REVIEW ≥45", fontsize=5.5, va="top")
ax1.set_xlabel("Trust score (0–100)", fontsize=7)
ax1.set_ylabel("Sessions", fontsize=7)
ax1.set_title("Score distribution by class", fontsize=8)
ax1.tick_params(labelsize=6)
ax1.legend(fontsize=6, frameon=False)

arch = summary["by_archetype"]
names = list(arch.keys())
short = ["Genuine", "Bot", "SIM-swap", "OTP farm", "Fake ID"]
ver = [arch[n]["verified"] / arch[n]["n"] * 100 for n in names]
rev = [arch[n]["review"] / arch[n]["n"] * 100 for n in names]
sus = [arch[n]["suspicious"] / arch[n]["n"] * 100 for n in names]
ax2.bar(short, ver, color="#2e7d32", label="VERIFIED")
ax2.bar(short, rev, bottom=ver, color="#f9a825", label="REVIEW")
ax2.bar(short, sus, bottom=[v + r for v, r in zip(ver, rev)], color="#c62828", label="SUSPICIOUS")
ax2.set_ylabel("% of sessions", fontsize=7)
ax2.set_title("Verdict by session archetype", fontsize=8)
ax2.tick_params(labelsize=6)
ax2.set_ylim(0, 118)
ax2.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2)
plt.tight_layout(pad=0.4)
chart = REP / "results_chart.png"
fig.savefig(chart, bbox_inches="tight")

# ── Document helpers ──────────────────────────────────────────────────────
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)  # A4
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Inches(0.6))
sec.top_margin = Inches(0.45)
sec.bottom_margin = Inches(0.45)

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(9)
style.paragraph_format.space_after = Pt(1.5)
style.paragraph_format.space_before = Pt(0)
style.paragraph_format.line_spacing = 1.0

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = ACCENT
    # bottom border
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "1F3A5F")):
        b.set(qn(k), v)
    pbdr.append(b)
    pPr.append(pbdr)
    return p


def para(parts, bullet=False):
    """parts: list of (text, bold) tuples or a plain string."""
    p = doc.add_paragraph(style="List Bullet" if bullet else None)
    if bullet:
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.space_after = Pt(0.5)
    if isinstance(parts, str):
        parts = [(parts, False)]
    for text, bold in parts:
        r = p.add_run(text)
        r.bold = bold
    return p


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(data, col_widths, header=True, font=8):
    t = doc.add_table(rows=len(data), cols=len(data[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(col_widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            r.font.size = Pt(font)
            if header and i == 0:
                r.bold = True
                shade(c, "DCE6F1")
    return t


# ── Title ─────────────────────────────────────────────────────────────────
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.paragraph_format.space_after = Pt(0)
r = t.add_run("TrustScore AI — Passive Behavioural Trust Scoring for Indian Digital KYC")
r.bold = True
r.font.size = Pt(14)
r.font.color.rgb = ACCENT
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.paragraph_format.space_after = Pt(2)
r = s.add_run("One-page project report  |  React 19 + TypeScript, fully on-device, zero backend")
r.italic = True
r.font.size = Pt(8.5)
a = doc.add_paragraph()
a.alignment = WD_ALIGN_PARAGRAPH.CENTER
a.paragraph_format.space_after = Pt(2)
r = a.add_run("Name: ")
r.bold = True
r.font.size = Pt(9)
r = a.add_run("Pranav Prashant Shewale")
r.font.size = Pt(9)
r = a.add_run("     |     Registration Number: ")
r.bold = True
r.font.size = Pt(9)
r = a.add_run("24BCE1226")
r.font.size = Pt(9)

# ── 1. Dataset ────────────────────────────────────────────────────────────
heading("1. Dataset Description")
para([
    ("Source. ", True),
    ("No public dataset exists for Indian KYC behavioural biometrics, and the system is privacy-by-design "
     "(no data leaves the device), so the evaluation set is a ", False),
    ("synthetic, labelled corpus of 2,000 KYC sessions", True),
    (" generated by scripts/simulate.ts with a fixed seed and scored by the production engine "
     "(src/services/scoringEngine.ts) without modification.", False),
])
para([
    ("Size & classes. ", True),
    ("1,000 genuine sessions + 1,000 fraud sessions across four archetypes (250 each): scripted bot, "
     "SIM-swap mule, OTP-farm/paste-in, and fake identity.", False),
])
para([
    ("Features (per session). ", True),
    ("Keystroke events (down/up timestamps, hold time, inter-key flight time); pointer/accelerometer motion "
     "samples (velocity magnitude); OTP start/completion timestamps; phone MSISDN; 12-digit Aadhaar number.", False),
])
para([
    ("Preprocessing. ", True),
    ("Flight-time outlier filter (keep 0 < f < 2000 ms); minimum-sample gates (<4 keystrokes or <5 motion samples "
     "fall back to a neutral prior); mouse events throttled to ~60 Hz; phone stripped of non-digits and the +91 prefix; "
     "Aadhaar stripped of spaces/dashes. Derived features: coefficient of variation (CV = σ/μ) of flight times, "
     "words-per-minute, motion-velocity CV, OTP latency in seconds, 4-digit MSISDN prefix. No scaling or encoding "
     "is needed because the model is rule-based.", False),
])

# ── 2. Problem ────────────────────────────────────────────────────────────
heading("2. Problem Statement")
para(
    "SMS-OTP verification proves that a device received a message, not that a real, legitimate person is completing "
    "the flow. SIM-swap fraud, OTP farming and scripted account creation all defeat it. The goal is a zero-friction, "
    "on-device trust layer that scores how a user behaves during a standard 4-screen KYC flow "
    "(Name/Phone → Aadhaar → OTP → Score) and outputs a 0–100 score with a VERIFIED / REVIEW / SUSPICIOUS verdict, "
    "without any server call or data egress (DPDP Act 2023 aligned)."
)

# ── 3. Methodology ────────────────────────────────────────────────────────
heading("3. Methodology")
para([
    ("Model. ", True),
    ("A deterministic, expert-weighted additive fusion of six independent signals (weights sum to 100). "
     "Each signal is explained by a per-signal score, risk tier (low/medium/high) and a human-readable reason.", False),
])
table([
    ["Signal (weight)", "Feature", "Scoring rule (from source)"],
    ["Typing rhythm (25)", "CV of inter-key flight time × WPM gate",
     "CV 0.2–1.0 → 1.0; CV<0.05 → 0.1 (bot); CV>2 → 0.2; linear taper otherwise. WPM>200 → ×0.4"],
    ["Tilt / motion (20)", "CV of pointer or accelerometer velocity", "ratio = max(0, 1 − CV/3); accelerometer on mobile, mouse proxy on desktop"],
    ["Aadhaar check (20)", "Verhoeff check digit (D5 dihedral group)", "Binary: 20 if 12 digits, not starting 0/1, and check digit valid"],
    ["OTP response (15)", "Seconds from OTP shown to correct entry", "<2 s → 2 (automation); 2–5 s → 8; 5–30 s → 15→8 linear; 30–60 s → 7; >60 s → 5"],
    ["SIM age (10)", "4-digit MSISDN prefix lookup (52 blocks)", "vintage 10 · mature 7 · moderate 5 · recent 2 · unknown 4"],
    ["Phone format (10)", "Regex ^[6-9]\\d{9}$", "Binary 10 / 0"],
], [1.25, 2.15, 3.65], font=7.5)
para([
    ("Decision. ", True),
    ("total = Σ signal scores, clamped 0–100; verdict = VERIFIED (≥70), REVIEW (45–69), SUSPICIOUS (<45). "
     "For binary metrics below, VERIFIED = accept, anything else = flag.", False),
])

# ── 4. Results ────────────────────────────────────────────────────────────
heading("4. Results & Insights")
c = summary["confusion"]
table([
    ["Accuracy", "Precision (fraud)", "Recall (fraud)", "F1", "ROC-AUC", "FP / FN", "Mean score G / F", "Latency"],
    [f"{summary['accuracy']*100:.1f}%", f"{summary['precision']*100:.1f}%", f"{summary['recall']*100:.1f}%",
     f"{summary['f1']:.3f}", f"{summary['roc_auc']:.3f}", f"{c['fp']} / {c['fn']}",
     f"{summary['mean_score_genuine']} / {summary['mean_score_fraud']}", f"<1 ms"],
], [0.8, 1.0, 0.9, 0.6, 0.7, 0.7, 1.15, 0.7], font=7.5)
doc.add_picture(str(chart), width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.paragraphs[-1].paragraph_format.space_after = Pt(1)

a = summary["by_archetype"]
sm = summary["signal_means"]
para([("Zero false positives: ", True),
      ("all 1,000 genuine sessions were VERIFIED (mean 88.2), so the layer adds no friction for legitimate users.", False)], bullet=True)
para([("Bots and OTP farms are caught outright: ", True),
      (f"0% of scripted bots or paste-in sessions reached VERIFIED; {a['Scripted bot']['suspicious']/2.5:.0f}% of bots were SUSPICIOUS. "
       "Near-zero flight-time variance and sub-2 s OTP entry are the dominant tells.", False)], bullet=True)
para([("SIM-swap mules are the weak spot: ", True),
      (f"{a['SIM-swap mule']['verified']/2.5:.0f}% were VERIFIED because their typing and motion are genuinely human; "
       "only SIM-age and OTP latency penalise them. This archetype drives all 129 misses and caps recall at 87.1%.", False)], bullet=True)
para([("Signal informativeness: ", True),
      (f"Aadhaar ({sm['genuine']['aadhaar']} vs {sm['fraud']['aadhaar']}), typing ({sm['genuine']['typing']} vs {sm['fraud']['typing']}) "
       f"and SIM age ({sm['genuine']['sim']} vs {sm['fraud']['sim']}) separate the classes; motion ({sm['genuine']['motion']} vs "
       f"{sm['fraud']['motion']}) is nearly uninformative and its 20-pt weight should be reduced or re-tuned. ~10% of random 12-digit "
       "numbers pass Verhoeff by chance, which explains the 22 fake-identity sessions that slipped through.", False)], bullet=True)

# ── 5. Novelty ────────────────────────────────────────────────────────────
heading("5. Novelty")
para([("Fully passive & fully on-device: ", True),
      ("every signal is a side-effect of steps the user already performs; nothing is enrolled, uploaded or stored — "
       "compliance is achieved by construction rather than by policy.", False)], bullet=True)
para([("OTP response latency as a fraud signal: ", True),
      ("a piecewise human-reaction curve (5–30 s sweet spot) turns the OTP step itself into a behavioural probe, "
       "penalising both automation (<2 s) and relayed/assisted entry (>60 s).", False)], bullet=True)
para([("MSISDN prefix → subscriber-tenure heuristic: ", True),
      ("an open, offline approximation of operator SIM-age data, directly targeting SIM-swap risk.", False)], bullet=True)
para([("Real Verhoeff validation + explainability by design: ", True),
      ("true UIDAI check-digit maths rather than a length check, and a transparent additive score where every point "
       "is attributable to a named signal with a plain-language reason — no black-box model, no SHAP needed.", False)], bullet=True)

out = REP / "TrustScore_AI_Report.docx"
doc.save(out)
print("saved", out)
