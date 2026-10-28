# CLAIMS CHECK

Every quantitative or comparative claim in `paper/main.tex` (and the summary) mapped to the result file it comes from. Evidence strength is a judgement about the claim, not about the code. Scope throughout: human-versus-automated session discrimination on public human data with SYNTHETIC automated sessions. Not identity, not real fraud.

Evidence-strength key:
- **Strong**: direct from the saved outputs, multi-seed or multi-fold, with CI or Holm correction where a comparison is made.
- **Moderate**: direct, but single seed, single fold, or partial (scope cut).
- **Weak / by construction**: follows from an assumption we made (OTP parameters, synthetic attacker design) rather than from data about real behaviour.
- **Not claimed**: stated only as a limitation or a caution, not as a result.

Result files are under `research/results/`. "tables/main_eval.json" is written by `scripts/07_eval_main.py`; "tables/verification.json" by `scripts/06_verify_runs.py`.

## Data and setup

| claim | result file | evidence |
|---|---|---|
| CMU: 51 subjects, 400 repetitions each (20,400 sessions); 10-key canonical view | `results/phase1/typing_counts.json`; `research/data/AUDIT.md` | Strong (counts) |
| KeyRecs-fixed: 99 participants; 19,772 attempts | `research/data/AUDIT.md` sec. 2 | Strong |
| 38,827 typing sessions kept after validity filtering (CMU 20,098; KeyRecs 18,729) | `results/phase1/typing_counts.json` | Strong |
| MotionSense sit + standing: 4,254 windows, 24 subjects; 50 Hz as documented, unverified | `results/phase1/motion_counts.json` | Strong for counts; **rate not verified** |
| Subject-grouped folds; no overlap in any split, both modalities | `tests/test_leakage_and_shapes.py` (48 tests passing) | Strong |
| Validity gate passes 98% of synthetic numbers; invalidity independent of class | `results/phase2/fold_meta.json` | Strong (by design) |

## Main comparison (pooled five-fold, seed 0 bootstrap; mean over fold x seed)

| claim | result file | evidence |
|---|---|---|
| habitID-Net (with OTP term) pooled AUC 0.847, CI [0.836, 0.858] | `tables/main_eval.json` -> bootstrap.AUC_seed0_pooled | Strong (but OTP term is by construction, see below) |
| GBDT on hand-crafted features beats habitID-Net on pooled AUC: 0.891 [0.884, 0.897] vs 0.847; delta -0.044, CI [-0.056, -0.032], Holm p < 0.001 | `tables/main_eval.json` -> bootstrap.paired_differences | Strong |
| habitID-Net beats habitID-v0 by 0.091 AUC (Holm p < 0.001) | same | Strong, but v0 is handicapped (see v0 verdicts) |
| habitID-Net vs no-prior: delta -0.0006, not significant (Holm 0.91) | same | Strong: the prior term does not change AUC |
| No-adversarial variant beats full model: delta -0.019 (Holm 0.006) | same | Strong: adversarial training (lambda_adv 0.2) lowers pooled AUC |
| Per-fold mean +- std: full 0.833+-0.035; noprior 0.836+-0.035; noadv 0.872+-0.015; GBDT 0.891+-0.007; LR 0.786+-0.014; v0 0.756+-0.009 | `tables/main_eval.json` -> main | Strong (15 fold-seed runs for neural and GBDT; 5 for v0, LR) |
| habitID-Net calibration ECE 0.041 raw -> 0.037 after temperature scaling; no-adv 0.029 -> 0.028 | `tables/main_eval.json` -> calibration | Moderate: temperature scaling helps only modestly |
| GBDT has lower ECE than habitID-Net (0.023 vs 0.041) | `tables/main_eval.json` -> main | Strong; GBDT is better calibrated here |

## Per-attacker-type and no-OTP (primary scientific comparison)

| claim | result file | evidence |
|---|---|---|
| Pooled AUC is capped: 60.2% of bot sessions are single-signal (typing-only, motion-only, OTP-only) | `tables/main_eval.json` -> per_type.share_single_signal_bots | Strong (design) |
| Full model: fully automated 0.969; motion-bot 0.834; OTP-bot 0.904 (BY CONSTRUCTION); typing-bot 0.502 (chance) | `tables/main_eval.json` -> per_type.full | Strong for the numbers; the typing-bot chance result means the fused model does not use typing evidence when motion/OTP look human |
| Without OTP (typing + motion), habitID-Net AUC 0.715 (5 folds, seed 0); GBDT without OTP 0.783; LR 0.685 | `tables/main_eval.json` -> no_otp, and no_otp | Moderate (single seed for habitID-Net); this is the primary result and it shows habitID-Net is below GBDT |
| Without OTP, replay-family bots (C) are not separable: worst-letter AUC 0.479 | `tables/main_eval.json` -> ablations.in_distribution_worst_letter ("no OTP term") | Moderate (seed 0) |
| OTP-based separation of automated sessions is an artefact of the assumed OTP mixture | `configs/otp_assumptions.yaml`; `tables/transfer_otp.json` -> otp_sensitivity | **Weak / by construction**: never claim detection of OTP automation in real traffic |
| OTP sensitivity grid (3x3): AUC range across grid cells | `tables/otp_sensitivity.json` -> summary | Moderate: post-hoc on the nominal-trained detector |

## Leave-one-family-out (LOFO), seed 0

| claim | result file | evidence |
|---|---|---|
| habitID-Net LOFO per family: A 0.647, B 0.710, C 0.702, D 0.704; worst family A | `tables/main_eval.json` -> lofo.lofo_full | Moderate (seed 0 only; the LOFO-noadv arm for B and C was kept) |
| LOFO-noadv: A 0.614 (worst), B 0.718, C 0.687, D 0.703 | `tables/main_eval.json` -> lofo.lofo_noadv | Moderate |
| GBDT refit without family: A 0.490 (worst, chance), B 0.900, C 0.698, D 0.765. On the held-out-family WORST case habitID-Net is better (0.647 vs 0.490); GBDT is better on B and D | `tables/main_eval.json` -> framing_lofo | Moderate |
| Without family D from training, the worst remaining letter is C at 0.698 | `tables/main_eval.json` -> ablations.no_generatorD_augmentation | Moderate |

## Conformal verdicts and coverage

| claim | result file | evidence |
|---|---|---|
| In-distribution coverage close to nominal at eps 0.05: human 0.932, bot 0.954 | `tables/main_eval.json` -> conformal_in_dist_full | Strong for the numbers, but human coverage is slightly below nominal |
| Abstention (REVIEW) rate is large: 65% at eps 0.01, 53% at eps 0.05 | same | Strong: the REVIEW band is wide |
| Under held-out family A, bot coverage at eps 0.10 is 0.764 (nominal 0.90), at eps 0.05 is 0.884 (nominal 0.95) | `tables/main_eval.json` -> conformal_lofo.A | Strong numbers; **coverage degrades under attacker shift** (the guarantee does not transfer) |
| Verdicts at eps 0.10: humans 31% VERIFIED, 56% REVIEW, 13% SUSPICIOUS; bots 9% VERIFIED, 21% REVIEW, 70% SUSPICIOUS | `tables/main_eval.json` -> verdicts_full_eps0.1 | Strong (in-distribution) |

## habitID-v0 and the validity gate

| claim | result file | evidence |
|---|---|---|
| habitID-v0 pooled AUC 0.756 [0.740, 0.771] | `tables/main_eval.json` -> bootstrap, main | Strong |
| habitID-v0 verdicts are almost all VERIFIED: humans 99.8%, bots 92.0% | `tables/main_eval.json` -> v0_verdicts | Strong. Cause: the aadhaar (20), phone (10) and SIM (10) terms are constants in this offline setting, which adds 35 points to every score (D18). The 70/45 thresholds are therefore not meaningful here. Do not read this as a finding about the prototype's design. |
| Validity gate alone: AUC 0.500 (exactly chance) | `tables/main_eval.json` -> main.gate_only | Strong (by design: invalidity is independent of class) |

## Transfer and distribution shift (single seed, seed 0)

| claim | result file | evidence |
|---|---|---|
| Controls pass: in-domain AUC CMU->CMU held-out subjects 0.775; KeyRecs->KeyRecs 0.807 | `tables/transfer_diag.json` (in_domain) | Moderate (single seed) |
| CMU->KeyRecs cross-dataset AUC 0.252 (flight on), 0.257 (flight off), 0.289 (per-dataset normalization) | `tables/transfer_diag.json` (cross) | Moderate: a documented distribution shift, not a sign or pipeline error (controls pass; labels checked; same code path) |
| KeyRecs->CMU cross-dataset AUC 0.456 (on), 0.441 (off), 0.642 (per-dataset normalization) | same | Moderate; per-dataset normalization changes this direction above chance |
| KeyRecs DD and flight z-scores have SD 1.7-1.8 under CMU statistics (CMU: 1.0); 10% of values beyond |z| > 3 (CMU: 1%) | computed in the Phase-2 diagnostic, see `research/PROGRESS.md` | Moderate (descriptive) |
| Cross-dataset transfer of the typing detector is NOT established | - | Stated as a limitation |

## Epoch cap and adversarial training

| claim | result file | evidence |
|---|---|---|
| Fraction of finished runs that reached the 24-epoch cap: noadv 7/15 (47%), noprior 4/15 (27%), full 3/15 (20%) | `tables/verification.json` -> per_variant (cap column uses epochs_run >= 24) | Strong (counts, final) |
| Rerunning the 7 capped noadv runs at a 48-epoch cap changes AUC by +0.004 on average; all 7 stopped by patience before 48 | `tables/cap_sensitivity.json` | Moderate: the cap bound slightly for noadv; **full and noprior capped runs were NOT rerun** (projected cost > 30 min) |
| Adversarial weight 1.0 blocked learning in pilot runs (AUC 0.74 at the first epoch); 0.2 learns | `research/PROGRESS.md` pilot table; `results/pilot_cap12/` | Moderate (pilot, fold 0) |

## Learned weights

| claim | result file | evidence |
|---|---|---|
| Full model alpha = (0.343, 0.310, 0.395) vs alpha0 = (0.417, 0.333, 0.250); distance 0.165 | `tables/main_eval.json` -> weights.full | Strong numbers. The OTP weight is well above its prior, which is driven by the assumed OTP model: **weak / by construction** |
| lambda_prior = 10 pulls alpha to alpha0 (distance 0.009) | `tables/main_eval.json` -> weights.lam10 | Strong (mechanical) |

## Adaptive stress test (D\*)

| claim | result file | evidence |
|---|---|---|
| Best-of-8 adaptive typing against the full model is still flagged: AUC 0.982, 95% flagged at p >= 0.5 | `tables/main_eval.json` -> dstar.full | Moderate. Caveat: motion and OTP of these sessions stay automated, so this is a typing-only adaptation, not a full adaptive attack. Reported as a separate stress test. |
| With family D held out, D* AUC is 0.838; flagged 75% | `tables/main_eval.json` -> dstar.lofo_full_D | Moderate |

## On-device cost

| claim | result file | evidence |
|---|---|---|
| ONNX export and CPU latency on this machine only | `tables/ondevice.json` (written by `scripts/09_onnx_ondevice.py`) | **Not a mobile measurement**; see the paper caveat |

## Not claimed

- Real-world fraud detection, identity verification, human fraudsters, SIM swap, or client tampering (out of scope).
- Detection of OTP-based automation in real traffic.
- Mobile or browser latency.
- Superiority of habitID-Net over GBDT on pooled AUC (the data show the opposite).
- Calibration superiority of habitID-Net over GBDT (the data show the opposite).
- The literature claims in the related-work section (e.g. the number of detectors in Killourhy and Maxion 2009) are cited from the literature and were NOT re-verified in this run. Reference details must be checked before submission.


## Claim audit (abstract, contributions, results, discussion, conclusion)

Status: **supported** = matches the cited result file as written; **weakened** = sentence rewritten to the evidence level; **removed** = wording taken out. Every number in `main.tex` is now either a generated macro (`paper_assets/numbers.tex`) or from a generated table; `scripts/12_latex_checks.py` finds no un-generated decimals.

### Abstract

| # | sentence (summary) | source | status |
|---|---|---|---|
| A1 | GBDT has higher AUC and lower calibration error than habitID-Net, with OTP (0.891 vs 0.847; ECE 0.023 vs 0.037 after scaling) and without OTP (0.783 vs 0.715, five folds, seed 0) | `tables/main_eval.json` (bootstrap, main, no_otp) | supported (ECE compared with the scaled value; raw 0.041) |
| A2 | OTP-based numbers come from an assumed latency model and do not measure real OTP behaviour | `configs/otp_assumptions.yaml` | supported (by construction) |
| A3 | habitID-Net's higher number is confined to the worst held-out family under LOFO (0.647 vs 0.490); single seed, no CI | `tables/main_eval.json` (lofo, framing_lofo) | supported; seed count stated |
| A4 | conformal verdicts "close to nominal" in distribution, abstain 53-65%, lose coverage on held-out families | `tables/main_eval.json` (conformal) | **weakened**: human coverage 0.93 at nominal 0.95; sentence now says "close to nominal" with the abstention range |
| A5 | cross-dataset transfer failed one way (AUC 0.25-0.29) and was above chance the other way only after per-dataset normalization (0.64; 0.46 without) | `tables/transfer_diag.json` | supported; single seed (stated in the Results) |
| A6 | attackers are synthetic; results are against the modeled generators | design | supported |

### Introduction, contributions

| # | sentence (summary) | status |
|---|---|---|
| I1 | "previously unevaluated design" | **removed** |
| I2 | "rarely evaluated against adaptive adversaries" | **weakened** to "we found no evaluation of the prototype" (a statement about our search, not the literature) |
| I3 | contribution 1: prior term; "did not change pooled AUC" | supported (delta -0.0006, Holm p = 0.91) |
| I4 | contribution 2: conformal, coverage degradation under shift | supported; single seed stated |
| I5 | contribution 3 (was "worst-case rather than average robustness") | **rewritten**: LOFO reporting worst family; no-adversarial variant was higher (Holm p = 0.006) |
| I6 | contribution 4 (was "in-browser deployment study") | **rewritten**: CPU-only ONNX measurement; no browser or mobile numbers |
| I7 | contribution 5 (new): GBDT better on pooled AUC and ECE, transfer failed | supported |
| I8 | related work: "fourteen anomaly detectors" | **removed** (unverified literature number) |
| I9 | related work: "our contribution is the calibrated, prior-regularized fusion" | **rewritten** |

### Method (sentences with claims)

| # | sentence | status |
|---|---|---|
| M1 | output "a calibrated probability" | **weakened** to "a probability ... temperature-scaled on held-out data" (ECE 0.037 is above zero; GBDT is better calibrated) |
| M2 | conformal guarantee under exchangeability (literature) | kept as a standard statement about the method, not a result; flagged for literature check |

### Results

| # | sentence (summary) | source | status |
|---|---|---|---|
| R1 | pooled AUC 0.847 [0.836, 0.858] | `tables/main_eval.json` bootstrap | supported |
| R2 | GBDT 0.891 [0.884, 0.897]; difference -0.044 [-0.056, -0.032], Holm p < 0.001 | same | supported |
| R3 | no-adversarial 0.866; above full by 0.019 (Holm p = 0.006) | same | supported |
| R4 | prior term changes AUC by less than 0.001 (p = 0.91) | same | supported |
| R5 | habitID-v0 0.756 [0.740, 0.771] | same | supported |
| R6 | v0 verdicts: humans 99.8%, bots 92% VERIFIED, caused by the constant 35 points | `tables/main_eval.json` v0_verdicts; D18 | supported (cause follows from the formula) |
| R7 | validity gate alone at chance (0.500) | main.gate_only | supported |
| R8 | no-OTP: 0.715 +- 0.021 (seed 0); GBDT 0.783; LR 0.685 | no_otp | supported; seed stated |
| R9 | replay family near chance without OTP (worst letter 0.479) | ablations (no OTP term) | supported; seed 0 |
| R10 | per-type table (typing-only 0.502, motion 0.834, full 0.969, OTP 0.904 by construction; 60% single-signal) | `tables/main_eval.json` per_type | supported |
| R11 | typing-only diagnosis: GBDT detects A (0.9996) and B (0.837); typing head does not (A 0.623, B 0.567) | `tables/typing_only_diagnosis.json` | supported (seed 0) |
| R12 | cause of the typing-head failure | diagnosis | **weakened**: "likely cause" only; the LSTM was not tested with explicit CV input |
| R13 | LOFO per-family numbers (habitID A 0.647 ... GBDT A 0.490) | `framing_lofo` | supported; seed 0, no CI |
| R14 | without family D, worst remaining letter C at 0.698 | ablations | supported; seed 0 |
| R15 | ECE 0.041 to 0.037; GBDT 0.023; no claim of calibration superiority | calibration, main | supported |
| R16 | in-distribution coverage 0.932 (human, nominal 0.95) and 0.954 (bot) at eps 0.05; abstention 53% (0.05), 65% (0.01) | conformal_in_dist_full | supported |
| R17 | held-out family A bot coverage 0.764 at eps 0.10 and 0.884 at eps 0.05 | conformal_lofo | supported; seed 0 |
| R18 | learned alpha (0.343, 0.310, 0.395) vs prior; OTP weight moves toward the assumed model | weights.full | supported numbers; **weakened** interpretation (assumption-driven) |
| R19 | lambda = 10 pulls alpha to the prior (distance 0.009) | weights.lam10 | supported |
| R20 | adaptive D* flagged (AUC 0.982; 95% flagged) and caveat: typing-only adaptation | dstar | supported; caveat stated |
| R21 | epoch cap: +0.004 for the seven rerun noadv runs; all stopped by patience | `tables/cap_sensitivity.json` | supported |
| R22 | transfer diagnostics numbers and controls (0.775, 0.807) | `tables/transfer_diag.json` | supported; seed 0 |
| R23 | target DD and flight z-score SD 1.7-1.8 vs 1.0 | diagnostic (`research/PROGRESS.md`) | supported (descriptive) |
| R24 | "the cause is likely dispersion and key identity; not isolated" | diagnostic | supported as hedged |
| R25 | ONNX: 0.28 ms fp32, 0.30 ms int8, 268/237 KB, AUC change +0.0009, agreement in Table | `tables/ondevice.json` | supported; CPU only |

### Discussion and limitations

| # | sentence | status |
|---|---|---|
| D1 | "Results quantify robustness to the modeled generators" | **rewritten** to "performance against" |
| D2 | "Simple baselines are strong here ... GBDT has higher pooled AUC than the no-adversarial variant" | **rewritten** from "outperforms"; supported |
| D3 | "Cross-dataset transfer failed" | supported; single seed stated |
| D4 | "Adversarial weight sensitivity" (pilot runs) | supported as a pilot observation |
| D5 | "Incomplete runs and reduced scope" (new) | supported; lists every not-done item |

### Conclusion

| # | sentence | status |
|---|---|---|
| C1 | prior-regularized, temperature-scaled, conformally thresholded model | supported (no "calibrated" claim) |
| C2 | boosted tree has higher pooled AUC and lower ECE, with and without OTP | supported |
| C3 | fused advantage limited to the LOFO worst case, one seed, no CI | supported |
| C4 | "robustness" wording | **removed** |

### Words searched for and their disposition

"robust" (all instances rewritten or qualified as seed 0, no CI); "outperform" (removed); "better" (each use checked against a result file); "calibrated" (rewritten in M1 and C1; remaining uses refer to the TS procedure or to the GBDT comparison); "transfer" (only "cross-dataset typing transfer failed" and "we do not claim cross-population transfer"); "principled" (removed from abstract); "real-world" (only as a limitation); "in-browser" (removed).

### Remaining rough edges (not claims)

- Killourhy and Maxion (2009), Dias et al. (2023), MotionSense and all other references: need manual verification (`REFERENCES_TO_VERIFY.md`).
- The bibliography entry for MotionSense includes a note "(MotionSense dataset.)" in the paper text; remove when verified.


## Post-hoc exploratory variant: typing summary statistics (follow-up)

Label used everywhere: **post-hoc exploratory variant, not the main model**. It is not in the main comparison table. Seed 0, five folds, full variant only. Source: `results/tables/typing_summary_exploratory.json` (written by `scripts/14_typing_summary_eval.py` from `results/runs/typing_summary_full`). Deviation D41 (CV proxy: SD of the asinh-flight channel instead of raw flight CV). Selection caveat: the variant was motivated by the typing-only diagnosis, which was computed on test-fold predictions, so every gain below is optimistic. Nothing was tuned on test metrics; no hyperparameter was changed.

| claim | source | evidence |
|---|---|---|
| Typing-only-bot AUC, family A: 0.998 (main model 0.557; GBDT all features 0.9997) | `typing_summary_exploratory.json` per_family_typing_only_AUC | exploratory; seed 0 |
| Typing-only-bot AUC, family B: 0.781 (main 0.542; GBDT all features 0.783) | same | exploratory; seed 0 |
| Families C and D: no gain (0.508 and 0.517; main 0.484 and 0.487) | same | exploratory; seed 0 |
| Pooled AUC with OTP term: 0.884 [0.875, 0.893] vs main 0.847 [0.835, 0.858] and GBDT 0.890 [0.884, 0.897]; intervals overlap | same | exploratory; seed 0 bootstrap |
| ECE (temperature-scaled): 0.028 vs main 0.034 and GBDT 0.022 (GBDT probabilities as output) | same | exploratory; seed 0; pooled values, not the fold means used in the main table |
| Typing-evidence medians separate humans (-0.41) from typing-only bots (-0.76) in the variant; the main model's are 1.73 and 1.76 (no separation) | same | exploratory; seed 0 |
| Without-OTP version of the variant: not run | - | stated as not run in the paper and table |
| Summary statistics "help" typing-only bots in general | - | **weakened**: help only for families A and B; no effect on C or D; not a general improvement |
| The variant is better than GBDT on pooled AUC | - | **not claimed** (point estimate 0.884 vs 0.890, overlapping intervals) |
| Result is robust or pre-registered | - | **not claimed**: post hoc, seed 0, chosen after looking at test-fold diagnostics |

Paper locations: Results, typing-only subsection (one exploratory row group in the typing table); Discussion paragraph "Post-hoc exploratory typing variant".
