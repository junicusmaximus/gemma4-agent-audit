# Matched-context readout v064t (prereg v064-r2, arm T_sub)

Slots frozen before reading: M0 = v066_A0mc (sealed v066 generation: E4B QAT base without adapter, matched template), M1 = v064_T_sub (T_sub adapter, matched template); every contrast is v064_T_sub - v066_A0mc. 120 rollouts = 40 tasks x 3 decoding seeds, paired by task with the seeds grouped (never 120 independent tasks); known holdout; one training seed; the two v064 arms are never pooled.

Termination on primary40 three-seed means: **COLLAPSED** (loop 0.683, submit 0.142; NON-COLLAPSED: loop <= 0.30 and submit >= 0.80; COLLAPSED: loop > 0.40 or submit < 0.70; otherwise INTERMEDIATE).

Repair: v064_T_sub pooled resolved 0/120 (by seed [0, 0, 0]); the pre-specified minimum repair observation (>= 4/120) is not met.

v064_T_sub - v066_A0mc on pooled resolved, paired task bootstrap (20000 draws, seed 20261103): -10.00 pp [-17.52, -3.33]: **worse than the base**.
A [0, 0] bootstrap is not evidence of equivalence.

| population | role | termination | loop | submit | v064_T_sub pooled | v066_A0mc pooled | effect pp [95% CI] | CI wording |
|---|---|---|---|---|---|---|---|---|
| primary40 | reading | COLLAPSED | 0.683 | 0.142 | 0/120 | 12/120 | -10.00 [-17.52, -3.33] | worse than the base |
| all48_sensitivity | sensitivity | COLLAPSED | 0.667 | 0.146 | 0/144 | 12/144 | -8.33 [-15.28, -2.78] | worse than the base |
| control_clean36_sensitivity | sensitivity | COLLAPSED | 0.694 | 0.120 | 0/108 | 12/108 | -11.11 [-19.44, -3.70] | worse than the base |

Descriptive rows, primary40 three-seed means (think, repro-first and M5 are descriptive; M5 undefined stays undefined):

| row | loop | submit | m5_error | think | repro-first | resolved by seed |
|---|---|---|---|---|---|---|
| v064_T_sub (slot M1) | 0.683 | 0.142 | 0.617 | 0.000 | 0.000 | [0, 0, 0] |
| v066_A0mc (slot M0) | 0.008 | 1.000 | 0.014 | 0.899 | 0.000 | [4, 3, 5] |
| v055_M0qmc (cited from its sealed report 16399065, not a slot) | 0.758 | 0.508 | 0.456 | 0.000 | 0.000 | [0, 0, 0] |

| v042 paired effect, primary40 (descriptive) | v066_A0mc | v064_T_sub | effect [95% CI] |
|---|---|---|---|
| loop | 0.008 | 0.683 | +0.675 [+0.575, +0.767] |
| submit | 1.000 | 0.142 | -0.858 [-0.917, -0.792] |
| m5_error | 0.014 | 0.617 | +0.613 [+0.529, +0.694] |
| think | 0.899 | 0.000 | -0.899 [-0.915, -0.879] |
| repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v066_A0mc_s1 | v066_A0mc | 652 | 32 | 24 | 1 |
| v066_A0mc_s2 | v066_A0mc | 595 | 32 | 26 | 0 |
| v066_A0mc_s3 | v066_A0mc | 553 | 33 | 27 | 0 |
| v064_T_sub_s1 | v064_T_sub | 3712 | 124 | 45 | 0 |
| v064_T_sub_s2 | v064_T_sub | 2936 | 171 | 40 | 20 |
| v064_T_sub_s3 | v064_T_sub | 2669 | 91 | 29 | 18 |

Training (v064_T_sub): U = 43 updates; 18 trajectories, 18 tasks, 342 sequences (duplication m = 9); adapter 7122a9d6; approved training contract 8e13817b.
v055_M0qmc is cited from its sealed report only; no contrast is computed against it. Think rate is read from returned reasoning (trace thinking events); compaction and content-without-reasoning counts are descriptive (definitions in report.json).
