# PROGRESS

Working directory: `research/` (project root `habitID/`). The existing web app is untouched. `raw dataset/` is read-only and is now in `.gitignore` along with `research/data/` and `research/results/` (Gate A decision 5).

## Environment

- Windows 11, Python 3.13.12, numpy 2.4.4, pandas 3.0.2, torch 2.11.0 (CPU), scipy 1.17.1, scikit-learn 1.8.0, PyYAML 6.0.3, matplotlib 3.10.8, pytest 8.2.0, onnx 1.23.2, onnxscript 0.7.2, onnxruntime 1.31.0. Pinned in `requirements.txt`.
- 18 logical CPUs. torch.cuda.is_available() = False. All training on CPU.
- TeX: none installed, by decision (Gate A, item 4). The paper is NOT compiled.
- Other processes on this machine (user's Jupyter kernels and `app.py`) were left untouched. They share the CPU.

## Gate A decisions (applied)

1. Fixed-text flight = DD - H(k1). Logged as D2. Transfer runs include a flight-off sensitivity setting.
2. Validity bounds accepted. Out-of-bounds sessions are dropped, and counts are reported per dataset (below).
3. MotionSense 50 Hz accepted and flagged as unverified (D7).
4. TeX install skipped (D12).
5. `.gitignore` updated.

## Speed-mode scope (what was cut, D21)

Fixed-text typing only (free-text dropped). Motion = sit + std (walking dropped). Main comparison = 5 folds x 3 seeds for 6 variants. LOFO (full and no-adversarial) and all ablations = seed 0. Lambda sweep {0, 0.1, 10} on seed 0. OTP 3 x 3 grid, post-hoc. ONNX on CPU only.

---

## Phase 0: Data audit  [Gate A, approved]

`research/scripts/00_audit_data.py` -> `research/data/AUDIT.md`, `research/results/audit/audit_summary.json`. Runtime 18.5 s. (Re-run after the Gate A discussion; numbers unchanged except where the sign-convention check was corrected.)

## Phase 1: Canonical datasets  [timing: 9.1 s]

`scripts/01_build_canonical.py`. Typing: 10-key canonical representation (CMU Return dropped).

| dataset | raw sessions | corrupt dropped | out-of-bounds dropped | kept | subjects |
|---|---|---|---|---|---|
| CMU | 20,400 | 0 | 302 | 20,098 | 51 |
| KeyRecs-fixed | 19,772 | 5 | 1,038 | 18,729 | 99 |

Total kept: 38,827. Motion: 4,254 non-overlapping 3-s windows at 50 Hz (sit 2,233; std 2,021), 24 subjects.

## Phase 2: Folds and attacker corpora  [timing: 286 s; the first build took 416 s before a key-collision fix]

`scripts/02_build_folds_bots.py`. Subject folds (5, stratified by dataset for typing; independent for motion). Each fold: train / cal / test by subject; generators fitted on train only.

Per fold: 8,000 train / 2,000 cal / 4,000 test humans, with an equal number of bot sessions (bot mix: 40% full, 20% typing-only, 20% motion-only, 20% OTP-only). Letters A/B/C/D (joint typing + motion family) and O (OTP-only). Gate-failing synthetic Aadhaar numbers: about 2% per split, independent of class.

Two bugs were caught and fixed here: (a) the motion-subject key collided with the role-prefixed key (the leakage test caught it; folds rebuilt); (b) the motion-D generator diverged (fixed by a stable linear AR; D14).

## Phase 3: Session composition

Implemented inside Phase 2 (`src/habitid/bots/policy.py`). Human sessions are cross-paired within role and fold. OTP is drawn from the assumed human mixture. Bot OTP is drawn from the automated mixture for full and OTP-only types.

## Tests  [48 tests]

`pytest -q` passes: Verhoeff (known example; every single-digit error and every adjacent transposition detected), v0 hand-computed examples, class-conditional conformal coverage on exchangeable synthetic data, leakage (grouped folds partition all subjects; every composed fold has no subject overlap for typing or motion), validity bounds on every fold, generator shapes and positivity. Two test expectations were corrected after checking the arithmetic (Verhoeff check digit for "236" is 3; OTP LLR at 0.2 s is negative).

## Phase 4: Baselines  [running; see results/runs/baselines]

habitID-v0 (thresholds in `configs/v0_weights.yaml`), validity gate only, logistic regression and GBDT on hand-crafted features, Isolation Forest, one-class SVM and scaled Manhattan fit on human training typing only. CV reliability experiment: `scripts/04_cv_reliability.py`.

## Phase 5: habitID-Net training

### Pilots (fold 0, seed 0), all logged

Each row is one configuration on fold 0. AUC is over all test bots. "Stalled" means the model stopped at epoch 0 or 1.

| pilot | config | AUC (all bots) | per-letter AUC (A / B / C / D / O) | outcome |
|---|---|---|---|---|
| p1 | original (lr 2e-3, batch 256, 8 epochs, scale = 1/std) | 0.821 | - | learned, but the typing encoder had output std 0.002 (typing-only AUC 0.60) |
| p2 | typing-only, same | 0.585 | 0.78 / 0.64 / 0.50 / 0.51 / 0.49 | typing branch under-trained |
| p3 | typing-only, no adversarial, lr 2e-3 | 0.646 / 0.611 / 0.463 / 0.454 / 0.501 | - | stalled (epoch 0) |
| p4 | typing-only, no adv., lr 5e-3, 20 epochs | - | 0.88 / 0.78 / 0.52 / 0.52 / 0.51 | learns after about 8 epochs |
| p5 | full, batch 512, lr 5e-3, lambda_adv 1.0 | 0.740 | 0.72 / 0.73 / 0.69 / 0.69 / 0.88 | stalled (epoch 0) |
| p6 | full, no adversarial (lambda_adv 0) | 0.874 | 0.93 / 0.91 / 0.73 / 0.89 / 0.89 | learns; 5.3 s/epoch |
| p7 | full, lambda_adv 1.0, eps reduced | 0.749 | 0.72 / 0.74 / 0.70 / 0.70 / 0.88 | stalled (epoch 0) |
| p8 | full, lambda_adv 0.2, buggy PGD projection | 0.861 | 0.91 / 0.91 / 0.71 / 0.87 / 0.89 | learns slowly; best epoch 16 |
| p9 | full, lambda_adv 1.0, corrected projection | 0.741 | 0.71 / 0.73 / 0.69 / 0.69 / 0.88 | stalled (epoch 0) |
| p10 | full, lambda_adv 0.2, corrected projection | 0.874 | 0.93 / 0.92 / 0.73 / 0.88 / 0.89 | learns; adversarial training on; chosen |

Findings from the pilots:

- Adversarial weight 1.0 blocks learning in this setup, at any PGD radius tried. Weight 0.2 learns and matches the no-adversarial model. This is reported as a finding, and the default is lambda_adv = 0.2 (D17).
- The PGD projection bug (clipping bot sessions onto the human box) was real and was fixed (D16). It did not cause the lambda_adv = 1.0 collapse, which persisted after the fix.
- The scale division (D15) was the cause of the typing encoder's collapse in the original pilot.

### Chosen configuration

batch 512, lr 0.005, 12-epoch cap, patience 4 (early stopping on calibration BCE), lambda_prior 0.1, lambda_adv 0.2, PGD radii (typing 0.1, motion 0.02, OTP 0.05 in log units), 3 steps, generator-D refresh every 2 epochs by best-of-8.

### Timing probe (required before the full run)

`scripts/05b_timing_probe.py` -> `results/timing/probe.json`. Fold 0, seed 0 per variant (these runs are reused by the plan).

| variant | fold wall (s) | epochs | mean s/epoch |
|---|---|---|---|
| full | 179.4 | 12 | 14.1 |
| noprior | 179.6 | 12 | 14.2 |
| noadv | 72.6 | 12 | 5.3 |
| typing_only | 47.1 | 8 | 4.9 |
| motion_only | 112.5 | 12 | 8.7 |
| otp_only | 11.0 | 5 | 0.8 |
| lam10 | 178.0 | 12 | 14.1 |
| no_typing | 115.6 | 12 | - |
| no_motion | 55.7 | 7 | - |
| no_otp | 177.2 | 12 | 14.0 |
| lofo_full_D | 206.1 | 11 | 17.9 |

**Serial projection: 4.8 hours. This exceeds the 3-hour budget.** The plan was therefore run with seven parallel workers (three torch threads each, `research/run_train_parallel.ps1`). The wall-clock projection is measured after the first few minutes of the run and is recorded below. If it exceeds 3 hours, the grid is cut further (the next candidates, in order: drop seed 2 for the lambda sweep arm, then shorten the LOFO-noadv arm).

(Parallel-run throughput: see the section "Phase 5 execution" below, filled in when the workers finish.)

### Update: longer training (user request, 3-hour budget lifted)

The user asked for longer training, since the 12-epoch cap was reached before convergence by many adversarial and LOFO runs (best epochs were 9 to 11). The 3-hour budget no longer applies.

- The 12-epoch runs (20 completed run files: fold 0 and some folds of the main/ablation/LOFO plan) are kept as a documented pilot in `results/pilot_cap12/`. They are NOT used in any result table.
- New configuration: epoch cap 24, patience 5 (`configs/default.yaml`). Everything else unchanged.
- Workers: nine processes, two torch threads each (`run_train_parallel.ps1`), so that 18 threads share 18 logical CPUs. The worker plan balances the expected serial cost across workers.
- Baselines for folds 3 and 4 were re-run after an earlier background job timed out during fold 3. Folds 0 to 2 completed in that job and were not re-run.
- Expected wall-clock is a measured estimate, not a probe: it is recorded below once the workers have run for a while.

### Run ownership and hygiene (checked)

Worker logs claim 83 completed runs, each exactly once; epoch counts in the logs match the saved JSON. No run file is claimed by two workers. Original workers w1, w3, w7 never started (launch bug, D33); their work moved to w1b, w3b, w7b, which had no completed runs at the check.

| worker | owns (run IDs) | status at check |
|---|---|---|
| w1b | full seeds 0,1, folds 0-4 (10 runs; extra process x2 takes fold 4) | running |
| w2 | full seed 2 (done, 5); typing_only (15, 9 done); otp_only (15, 9 done) | running |
| w3b | noprior seeds 0,1, folds 0-4 (10; extra process x1 takes fold 4) | running |
| w4 | noprior seed 2 (done, 5); noadv (15, 10 done) | running |
| w5 | motion_only (15, done); no_motion (5, 4 done) | running |
| w6 | lofo_full_A (done), lofo_noadv_A (done); no_typing (5, 4 done) | running |
| w7b | lofo_full_B, lofo_noadv_B, lofo_noadv_C, folds 0-4 (15; extra process takes fold 4) | running |
| w8 | lofo_full_C, lofo_noadv_D (done); lam10 (5, 3 done) | running |
| w9 | lofo_full_D (done); no_otp (5, 4 done) | running |
| x1, x2, x3 | tail extras: noprior fold 4 (seeds 1,0); full fold 4 (seeds 1,0); LOFO-B fold 4 | running |

### ETA (measured from completed runs)

Mean wall-clock per completed run under current contention: full 1020 s, noprior 1269 s, noadv 435 s, typing-only 559 s, OTP-only 36 s, motion-only 475 s, LOFO-full 968-1217 s, LOFO-noadv 248-293 s, lam10 1046 s, no_typing 995 s, no_motion 744 s, no_otp 1002 s.

Runs still missing at the check: 67 (of 150). Serial remaining work is about 39,000 s. The three longest chains after tail offloading: noprior seeds 0-1 (8 runs, about 10,150 s at current contention), full seeds 0-1 (8 runs, about 8,160 s), LOFO-B (12 runs, about 7,200 s). Contention should fall as the short workers finish.

**Projected finish: about 2.3 h from the check (range 1.8 to 2.8 h).** This exceeds 2 hours.

**Cut candidates (not applied; waiting for your OK; none of these is in the main comparison):**
1. lam10 (lambda sweep): reduce from 5 folds to 3 folds, saving about 2,100 s.
2. no_typing and no_otp (leave-one-signal-out ablations): reduce to 3 folds, saving about 1,900 s.
3. LOFO-noadv for families B and C: reduce to A and D only, saving about 1,500 s (the brief asks for LOFO for both full and noadv; this would weaken that).

Not proposed for cutting: the main comparison (6 variants x 5 folds x 3 seeds), the LOFO-full arm, and the single-signal heads.

## Phases 6-8 (status, final)

All main training finished except lam10 fold 4 (cut before start; lam10 has folds 0-3). Evaluation (`07_eval_main.py`), OTP sensitivity (`08 otp`), transfer diagnostics (`08b`), epoch-cap sensitivity (`07b`), ONNX (`09`) and paper assets (`10`) have run. Outputs: `results/tables/*.json`, `paper_assets/`. The 48-epoch rerun covered only the seven capped noadv runs (mean AUC change +0.004); capped full and noprior runs were not rerun.

Key numbers (see `CLAIMS_CHECK.md` for the claim-by-claim mapping):
- Pooled AUC with OTP term: habitID-Net 0.847 [0.836, 0.858], GBDT 0.891 [0.884, 0.897], v0 0.756.
- No OTP term (seed 0 for habitID-Net): habitID-Net 0.715, GBDT 0.783, LR 0.685.
- LOFO worst case: habitID-Net 0.647 (family A, seed 0), GBDT 0.490.
- ECE: habitID-Net 0.041 (0.037 after temperature scaling), GBDT 0.023.
- Transfer (seed 0): CMU to KeyRecs 0.252 / 0.257 / 0.289; KeyRecs to CMU 0.456 / 0.441 / 0.642; in-domain controls 0.775 and 0.807. Documented distribution shift.
- ONNX (CPU, one thread): fp32 median 0.28 ms, 268 KB; int8 median 0.30 ms, 237 KB; AUC change +0.0009.

Deliverables present: `CLAIMS_CHECK.md`, `FACULTY_SUMMARY.md`, `paper/main.tex` (with `paper/results_section.tex` merged in), `paper_assets/`.

### QA pass (current task, no new training sweeps)

Planned, in order: (1) typing-only-bot diagnosis per family for fused, typing head, GBDT and LR on typing features; (2) LOFO seed cost estimate for seeds 1-2 (decision recorded below); (3) claim audit of abstract, contributions, results, discussion and conclusion; (4) automated LaTeX consistency checks; (5) limitations section completed from the "not done" list; (6) `submission/` package for Overleaf plus README files; (7) bibliography list with manual-verification flags.

Status (complete):

1. Typing-only-bot diagnosis: done (`scripts/11_typing_only_diagnosis.py`, `results/tables/typing_only_diagnosis.json`, paper Section "Typing-only bots"). Seed 0. Fused model: A 0.557, B 0.542, C 0.484, D 0.487. Typing head: A 0.623, B 0.567, C 0.464, D 0.489. GBDT on typing features: A 0.9996, B 0.837, C 0.512, D 0.530. LR on typing features: A 0.976, B 0.513, C 0.497, D 0.565. Learned typing alpha 0.329; typing contribution median 1.73 for humans and 1.76 for typing-only bots. GBDT detects naive and humanized typing-only bots and the typing head does not. The cause is not isolated (an LSTM with explicit CV input was not tested).
2. LOFO seeds 1-2: NOT run. Cost estimate 15 h serial, about 5 h parallel, against a 45-minute limit (D38). All robustness claims are single-seed with no CI.
3. Claim audit: done. Appended to `CLAIMS_CHECK.md`. Overclaims removed or weakened (see that file).
4. Consistency checks: `scripts/12_latex_checks.py`. Result: no missing inputs; no unresolved references (after fixing `sec:limits`); all citations have bibitems and vice versa; environments balanced; the only `\TBD` is the acknowledgment; no un-generated decimal numbers in the text (the one exception is the arXiv identifier in the bibliography). Hand-typed values replaced by generated macros and tables (per-type, no-OTP, headline numbers).
5. Limitations: "Incomplete runs and reduced scope" paragraph added with every not-done item.
6. Package: `scripts/13_build_submission.py` writes `submission/` (main.tex with relative paths, paper_assets, README with compile order) and the top-level `README.md` (reproduction, pinned requirements, dataset sources, licenses, SHA-256; `results/tables/input_hashes.json`).
7. Bibliography: `REFERENCES_TO_VERIFY.md` lists every entry with NEEDS MANUAL VERIFICATION.

Remaining TBDs: the acknowledgment only.

### Follow-up (bounded, then freeze): typing-encoder summary-statistics ablation

State before the follow-up: main results, paper, submission package and QA are complete as above; the main table is not changed by this follow-up.

Plan: a POST-HOC EXPLORATORY variant `typing_summary_full` (full variant, seed 0, folds 0-4, one worker per fold). It adds per-session summary statistics computed from the same z-normalized inputs the LSTM sees (means and SDs of the hold, DD and asinh-flight channels across positions; 6 features) concatenated to the LSTM's final hidden state before the typing head. Deviation from the request: the raw flight CV is not recoverable from the z-normalized inputs, so the SD of the asinh-flight channel is used as its proxy (logged in DEVIATIONS.md, D41). Everything else identical: folds, prior regularization, adversarial loop, early stopping on calibration BCE only. No hyperparameter changes. Without-OTP version of the variant is not trained (outside the requested scope); this is stated in the paper.

Reporting: per-family typing-only-bot AUC (A-D), pooled AUC with OTP, ECE, typing-evidence medians (humans vs typing-only bots), next to the original full model and GBDT. Result will be reported whether or not it helps. Not done: LOFO seeds 1-2; capped full and noprior reruns; multi-seed transfer; no-OTP habitID-Net seeds 1-2; lam10 fifth fold; TeX compile; mobile/browser latency; bibliography verification.

### Follow-up result (final)

Done. 	yping_summary_full (post-hoc exploratory; full variant; seed 0; folds 0-4 all completed in about 23 minutes, so no fold was dropped). Results (esults/tables/typing_summary_exploratory.json): typing-only AUC A 0.998, B 0.781, C 0.508, D 0.517 (main model A 0.557, B 0.542, C 0.484, D 0.487; GBDT all features A 0.9997, B 0.783). Pooled AUC with OTP 0.884 [0.875, 0.893] (main 0.847; GBDT 0.890, overlapping intervals). ECE 0.028 (main 0.034 pooled; GBDT 0.022). Typing-evidence medians: humans -0.41, typing-only bots -0.76 (main 1.73 vs 1.76). Without-OTP version not trained.

Changes made in this follow-up: models/habitid.py (TypingEncoder summary option, default off), 	rain/train.py (passes the flag), scripts/05_train_habitid.py (variant entry, group exploratory), scripts/14_typing_summary_eval.py (new), scripts/10_make_paper_assets.py (one row group in the typing table; exploratory macros), paper/main.tex (Discussion paragraph; typing subsection sentence), CLAIMS_CHECK.md (exploratory section), DEVIATIONS.md (D41). No change to the main table, no hyperparameter change, no change to other variants. Latex checks: clean apart from the acknowledgment TBD. Tests: 48 passed. Submission rebuilt.

Frozen.
