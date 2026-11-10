# Matched-context readout v064c (prereg v064-r2, readings v069, A_onpol confirmation: LoRA init seed 20261104)

Slots frozen before reading: M0 = v066_A0mc (sealed v066 generation: E4B QAT base without adapter, matched template), M1 = v064_A_onpol_c2 (A_onpol confirmation adapter, second training seed, matched template); every contrast is v064_A_onpol_c2 - v066_A0mc. 120 rollouts = 40 tasks x 3 decoding seeds, paired by task with the seeds grouped (never 120 independent tasks); known holdout; seed 1 (the sealed v064a report) is printed beside this second training seed and never pooled with it; the two v064 arms are never pooled.

Termination on primary40 three-seed means: **NON-COLLAPSED** ("did not meet the pre-defined collapse criteria") (loop 0.017, submit 0.992; NON-COLLAPSED: loop <= 0.30 and submit >= 0.80; COLLAPSED: loop > 0.40 or submit < 0.70; otherwise INTERMEDIATE).

Repair: v064_A_onpol_c2 pooled resolved 9/120 (by seed [3, 2, 4]); the pre-specified minimum repair observation (>= 4/120) is met.

v064_A_onpol_c2 - v066_A0mc on pooled resolved, paired task bootstrap (20000 draws, seed 20261103): -2.50 pp [-6.67, +0.00]: **uncertain relative to the base**.
A [0, 0] bootstrap is not evidence of equivalence.

Confirmation readings (readings addendum v069, primary40):

Termination sentence T1: "training on the base's own successful trajectories did not meet the pre-defined collapse criteria in two training seeds (loop 0.008 and 0.017, submit 0.958 and 0.992; resolved 10/120 and 9/120) on the known holdout"
Interval sentence: "uncertain relative to the base (9/120 vs 12/120; -2.50 pp [-6.67, +0.00])"
Prose flags: two-seed collapse-criteria prose allowed (needs the confirmation NON-COLLAPSED); "improved" prose not allowed (needs the confirmation's lower bound > 0, then always beside seed 1's interval (10/120 vs 12/120; -1.67 pp [-6.67, +2.50])).

| training seed | source | termination | loop | submit | pooled resolved (by seed) | arm - A0mc pp [95% CI] | CI wording |
|---|---|---|---|---|---|---|---|
| 1: v064_A_onpol | sealed v064a report 8f32f58b | NON-COLLAPSED | 0.008 | 0.958 | 10/120 [5, 3, 2] | -1.67 [-6.67, +2.50] | uncertain relative to the base |
| 2: v064_A_onpol_c2 | this readout | NON-COLLAPSED | 0.017 | 0.992 | 9/120 [3, 2, 4] | -2.50 [-6.67, +0.00] | uncertain relative to the base |

Seed 1 is printed from its sealed report and never pooled with seed 2; no pooled row exists.

| population | role | termination | loop | submit | v064_A_onpol_c2 pooled | v066_A0mc pooled | effect pp [95% CI] | CI wording |
|---|---|---|---|---|---|---|---|---|
| primary40 | reading | NON-COLLAPSED | 0.017 | 0.992 | 9/120 | 12/120 | -2.50 [-6.67, +0.00] | uncertain relative to the base |
| all48_sensitivity | sensitivity | NON-COLLAPSED | 0.035 | 0.993 | 9/144 | 12/144 | -2.08 [-5.56, +0.00] | uncertain relative to the base |
| control_clean36_sensitivity | sensitivity | NON-COLLAPSED | 0.019 | 0.991 | 9/108 | 12/108 | -2.78 [-7.41, +0.00] | uncertain relative to the base |

Descriptive rows, primary40 three-seed means (think, repro-first and M5 are descriptive; M5 undefined stays undefined):

| row | loop | submit | m5_error | think | repro-first | resolved by seed |
|---|---|---|---|---|---|---|
| v064_A_onpol_c2 (slot M1) | 0.017 | 0.992 | 0.027 | 0.915 | 0.000 | [3, 2, 4] |
| v066_A0mc (slot M0) | 0.008 | 1.000 | 0.014 | 0.899 | 0.000 | [4, 3, 5] |
| v055_M0qmc (cited from its sealed report 16399065, not a slot) | 0.758 | 0.508 | 0.456 | 0.000 | 0.000 | [0, 0, 0] |

| v042 paired effect, primary40 (descriptive) | v066_A0mc | v064_A_onpol_c2 | effect [95% CI] |
|---|---|---|---|
| loop | 0.008 | 0.017 | +0.008 [-0.017, +0.033] |
| submit | 1.000 | 0.992 | -0.008 [-0.025, +0.000] |
| m5_error | 0.014 | 0.027 | +0.016 [-0.004, +0.038] |
| think | 0.899 | 0.915 | +0.016 [+0.003, +0.032] |
| repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v066_A0mc_s1 | v066_A0mc | 652 | 32 | 24 | 1 |
| v066_A0mc_s2 | v066_A0mc | 595 | 32 | 26 | 0 |
| v066_A0mc_s3 | v066_A0mc | 553 | 33 | 27 | 0 |
| v064_A_onpol_c2_s1 | v064_A_onpol_c2 | 610 | 32 | 25 | 0 |
| v064_A_onpol_c2_s2 | v064_A_onpol_c2 | 659 | 39 | 30 | 0 |
| v064_A_onpol_c2_s3 | v064_A_onpol_c2 | 631 | 39 | 29 | 0 |

Training (v064_A_onpol_c2): U = 43 updates; 30 trajectories, 13 tasks, 342 sequences (duplication m = 9); adapter f9e06642; approved training contract f214de6f.
v055_M0qmc is cited from its sealed report only; no contrast is computed against it. Think rate is read from returned reasoning (trace thinking events); compaction and content-without-reasoning counts are descriptive (definitions in report.json).
