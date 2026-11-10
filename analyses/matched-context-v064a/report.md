# Matched-context readout v064a (prereg v064-r2, arm A_onpol)

Slots frozen before reading: M0 = v066_A0mc (sealed v066 generation: E4B QAT base without adapter, matched template), M1 = v064_A_onpol (A_onpol adapter, matched template); every contrast is v064_A_onpol - v066_A0mc. 120 rollouts = 40 tasks x 3 decoding seeds, paired by task with the seeds grouped (never 120 independent tasks); known holdout; one training seed; the two v064 arms are never pooled.

Termination on primary40 three-seed means: **NON-COLLAPSED** ("did not meet the pre-defined collapse criteria") (loop 0.008, submit 0.958; NON-COLLAPSED: loop <= 0.30 and submit >= 0.80; COLLAPSED: loop > 0.40 or submit < 0.70; otherwise INTERMEDIATE).

Repair: v064_A_onpol pooled resolved 10/120 (by seed [5, 3, 2]); the pre-specified minimum repair observation (>= 4/120) is met.

v064_A_onpol - v066_A0mc on pooled resolved, paired task bootstrap (20000 draws, seed 20261103): -1.67 pp [-6.67, +2.50]: **uncertain relative to the base**.
A [0, 0] bootstrap is not evidence of equivalence.

| population | role | termination | loop | submit | v064_A_onpol pooled | v066_A0mc pooled | effect pp [95% CI] | CI wording |
|---|---|---|---|---|---|---|---|---|
| primary40 | reading | NON-COLLAPSED | 0.008 | 0.958 | 10/120 | 12/120 | -1.67 [-6.67, +2.50] | uncertain relative to the base |
| all48_sensitivity | sensitivity | NON-COLLAPSED | 0.007 | 0.958 | 10/144 | 12/144 | -1.39 [-5.56, +2.08] | uncertain relative to the base |
| control_clean36_sensitivity | sensitivity | NON-COLLAPSED | 0.009 | 0.963 | 10/108 | 12/108 | -1.85 [-7.41, +2.78] | uncertain relative to the base |

Descriptive rows, primary40 three-seed means (think, repro-first and M5 are descriptive; M5 undefined stays undefined):

| row | loop | submit | m5_error | think | repro-first | resolved by seed |
|---|---|---|---|---|---|---|
| v064_A_onpol (slot M1) | 0.008 | 0.958 | 0.027 | 0.915 | 0.000 | [5, 3, 2] |
| v066_A0mc (slot M0) | 0.008 | 1.000 | 0.014 | 0.899 | 0.000 | [4, 3, 5] |
| v055_M0qmc (cited from its sealed report 16399065, not a slot) | 0.758 | 0.508 | 0.456 | 0.000 | 0.000 | [0, 0, 0] |

| v042 paired effect, primary40 (descriptive) | v066_A0mc | v064_A_onpol | effect [95% CI] |
|---|---|---|---|
| loop | 0.008 | 0.008 | +0.000 [-0.025, +0.025] |
| submit | 1.000 | 0.958 | -0.042 [-0.075, -0.008] |
| m5_error | 0.014 | 0.027 | +0.017 [-0.003, +0.040] |
| think | 0.899 | 0.915 | +0.017 [+0.004, +0.032] |
| repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v066_A0mc_s1 | v066_A0mc | 652 | 32 | 24 | 1 |
| v066_A0mc_s2 | v066_A0mc | 595 | 32 | 26 | 0 |
| v066_A0mc_s3 | v066_A0mc | 553 | 33 | 27 | 0 |
| v064_A_onpol_s1 | v064_A_onpol | 586 | 30 | 22 | 0 |
| v064_A_onpol_s2 | v064_A_onpol | 657 | 41 | 32 | 0 |
| v064_A_onpol_s3 | v064_A_onpol | 667 | 37 | 26 | 0 |

Training (v064_A_onpol): U = 43 updates; 30 trajectories, 13 tasks, 342 sequences (duplication m = 9); adapter ad0910f5; approved training contract 8e13817b.
v055_M0qmc is cited from its sealed report only; no contrast is computed against it. Think rate is read from returned reasoning (trace thinking events); compaction and content-without-reasoning counts are descriptive (definitions in report.json).
