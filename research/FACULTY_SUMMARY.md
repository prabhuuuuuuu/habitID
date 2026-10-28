# FACULTY SUMMARY: habitID-Net (human vs. automated session discrimination)

**Task.** Score whether an onboarding-style session is human-operated or automated, from passive signals: typing rhythm, phone motion, and OTP-entry latency. This is NOT identity verification and does not detect human fraudsters, SIM swap or client tampering (all out of scope).

**Data.** Human data: CMU keystroke benchmark (51 subjects, 10-key view); KeyRecs fixed-text (99 participants); MotionSense pocket-worn sit and standing windows (24 subjects, 50 Hz as documented, not verified). Automated sessions are synthetic (attacker families A naive, B humanized, C replay, D learned, O OTP-only). OTP-latency parameters are ASSUMED (no public data). Aalto and HuMIdb were not used.

**Models.** Baseline habitID-v0 (fixed-weight additive score); validity gate; logistic regression and GBDT on hand-crafted features; Isolation Forest, one-class SVM and scaled Manhattan (typing, fit on humans only); habitID-Net: log-odds fusion of an LSTM (typing), a 1D-CNN (motion) and a fixed OTP log-likelihood ratio, with the weights regularized toward the hand-set prior, adversarial training with a learned generator, temperature scaling, and class-conditional conformal verdicts.

**Headline results** (five folds; pooled seed 0 with 95% subject-cluster bootstrap CIs)

| setting | habitID-Net | best baseline |
|---|---|---|
| Pooled AUC, with OTP term | 0.847 [0.836, 0.858] | GBDT 0.891 [0.884, 0.897] |
| Pooled AUC, no OTP term (primary) | 0.715 +- 0.021 (seed 0) | GBDT 0.783 +- 0.008 |
| Worst held-out family (LOFO) | 0.647 (family A) | GBDT 0.490 (family A) |
| Calibration (ECE) | 0.041 (0.037 after scaling) | GBDT 0.023 |
| Coverage at eps 0.05 (in-distribution) | human 0.932, bot 0.954 | - |
| Abstention at eps 0.05 | 53% | - |
| Typing-only bots (per type) | AUC 0.502 (chance) | - |

- The OTP-term AUC is driven by assumed OTP parameters. OTP-only bot AUC (0.904) is by construction; we do not claim OTP-based automation can be detected in real traffic.
- The validity gate alone is at chance (0.500) by design.
- habitID-v0 looks almost useless here (AUC 0.756; 92% of bots VERIFIED), but this is caused by constant offsets from the three absent signals (aadhaar, phone, SIM) that add 35 points in this offline setting. It is not a finding about the prototype's design.

**What failed or underperformed (reported as-is)**
1. GBDT on hand-crafted features beats habitID-Net on pooled AUC (0.891 vs 0.847) and without OTP (0.783 vs 0.715), and is better calibrated (ECE 0.023 vs 0.041).
2. The adversarial term at weight 1.0 blocked learning in pilots; the reported model uses 0.2. The no-adversarial variant is better than the full model on pooled AUC (0.866 vs 0.847, Holm p = 0.006).
3. Typing-only bots are at chance for the fused model. Replay bots are at chance without the OTP term.
4. Under held-out family A, conformal bot coverage falls to 0.76 at nominal 0.90. The guarantee does not survive attacker shift.
5. Cross-dataset typing transfer is below chance one way (CMU to KeyRecs, AUC 0.25 to 0.29). Controls pass; we interpret it as a distribution shift (dispersion and key identities), not a bug.
6. One-class typing detectors score naive bots as more human than humans (AUC 0.15 to 0.22 for family A). Their direction was not flipped.
7. Learned OTP weight moves well away from its prior (0.395 vs 0.250), which reflects the assumed OTP model.

**Limitations.** Synthetic attackers; attacker families are our design. Typing and motion come from different people, paired by construction, so within-person dependence is not modelled. Motion is pocket-worn data, not hand-held typing. The OTP model is assumed. The 24-epoch cap binds for 47% of no-adversarial runs and 20--27% of full and no-prior runs; the no-adversarial cap rerun changes AUC by +0.004, but the full and no-prior capped runs were not rerun. Cross-dataset results are single-seed. Single-seed LOFO and ablations. CPU-only latency, not a device measurement.

**Not done (listed honestly)**
- Full and no-prior capped runs were not rerun at a 48-epoch cap (projected cost above the limit). Only the seven capped no-adversarial runs were rerun.
- The no-OTP habitID-Net was run for seed 0 only (five folds), not three seeds.
- Cross-dataset transfer was run for seed 0 only; the earlier multi-seed transfer job was stopped and not completed.
- Lambda sweep: lam10 has four folds (fold 4 cut before start; fold 3 had already started and was kept). Ablations are seed 0.
- LOFO for habitID-Net was run for seed 0 only.
- No mobile, browser or real-traffic evaluation.
- Acknowledgment in the paper is left as \TBD (course, instructor and compute resources not supplied).
- Bibliography entries and related-work claims were not re-verified against the sources in this run.
- The paper was not compiled (no TeX toolchain; decision at Gate A).
- Per-phase elapsed times are taken from worker and step logs, because `results/timing/phases.json` keeps only the last run of each phase name.
