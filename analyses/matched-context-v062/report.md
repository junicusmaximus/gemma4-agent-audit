# Matched-context readout v062 (M1q)

Slots frozen before reading: M0 = v055_M0qmc (matched template, M0q adapter), M1 = v062_M1qmc (matched template, M1q adapter); every contrast is v062_M1qmc - v055_M0qmc. Exploratory; known holdout; one adapter and training seed per arm.

Declared confounds: single training seed; one data source; the M1 mask removes about the empty-thought share of supervised targets (fewer supervised tokens than M0q at equal update count; the frozen mask-control-v035 denominators apply); no masking x history cell (Hq1+mask not run). The bf16-panel M1 (v043) and the QAT-panel M1q are reported as two panels, never pooled.

Pre-declared reading on primary40 three-seed means: **PK-ii** (PK-i False, PK-ii True; M5-error: read).

Think rate (primary40 three-seed mean, descriptive label only, threshold 0.50): 0.9054079430259293; label: thinking re-opened on the matched base.

| metric (primary40) | v055_M0qmc | v062_M1qmc | paired task-bootstrap effect [95% CI] |
|---|---|---|---|
| loop | 0.7583333333333333 | 0.775 | +0.017 [-0.092, +0.125] |
| submit | 0.5083333333333333 | 0.5750000000000001 | +0.067 [-0.050, +0.183] |
| m5_error | 0.4563850206869089 | 0.3928414952826232 | -0.068 [-0.165, +0.028] |
| think | 0.0 | 0.9054079430259293 | +0.905 [+0.855, +0.950] |
| repro | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |

| descriptive secondary (primary40) | paired effect [95% CI] |
|---|---|
| M4.opener.find_opener | -0.717 [-0.833, -0.583] |
| M4.opener.repro_script_opener | +0.000 [+0.000, +0.000] |
| M2.max_repeat | +0.508 [-5.817, +6.667] |
| M5.repeat_after.silent | +0.190 [+0.103, +0.272] |
| M5.repeat_after.normal | +0.003 [-0.042, +0.048] |

Resolution (v042 statistics, effect in pp):
- all48_sensitivity: 6.944 pp, 95% CI [1.3888888888888888, 13.888888888888888]; v055_M0qmc [0, 0, 0], v062_M1qmc [4, 2, 4].
- control_clean36_sensitivity: 9.259 pp, 95% CI [2.7777777777777777, 18.51851851851852]; v055_M0qmc [0, 0, 0], v062_M1qmc [4, 2, 4].
- primary40: 8.333 pp, 95% CI [1.6666666666666667, 16.666666666666664]; v055_M0qmc [0, 0, 0], v062_M1qmc [4, 2, 4].

Pooled v062_M1qmc resolved on primary40: 10/120 (notable: True; descriptive, no wording beyond observed before an independent-repo confirmation).

| row (primary40 three-seed means) | loop | submit | m5_error | think | repro | resolved by seed |
|---|---|---|---|---|---|---|
| v055_M0qmc (slot M0) | 0.7583333333333333 | 0.5083333333333333 | 0.4563850206869089 | 0.0 | 0.0 | [0, 0, 0] |
| v062_M1qmc (slot M1) | 0.775 | 0.5750000000000001 | 0.3928414952826232 | 0.9054079430259293 | 0.0 | [4, 2, 4] |
| v051_M0mc | 0.6583333333333333 | 0.6083333333333333 | 0.6522169630217212 | 0.0 | 0.7083333333333334 | [0, 0, 0] |
| v043_M0_sealed | 0.6166666666666667 | 0.6 | 0.4908857429666174 | 0.0 | 0.65 | [0, 0, 0] |
| v043_M1_sealed | 0.7000000000000001 | 0.13333333333333333 | 0.6541994342481312 | 0.7761616225867747 | 0.041666666666666664 | [0, 0, 0] |
| v052_A0cap | 0.008333333333333333 | 1.0 | 0.020014478764478764 | 0.908093180337559 | 0.0 | [5, 5, 5] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v055_M0qmc_s1 | v055_M0qmc | 3840 | 78 | 48 | 0 |
| v055_M0qmc_s2 | v055_M0qmc | 2050 | 44 | 22 | 2 |
| v055_M0qmc_s3 | v055_M0qmc | 3017 | 48 | 33 | 0 |
| v062_M1qmc_s1 | v062_M1qmc | 2817 | 95 | 46 | 7 |
| v062_M1qmc_s2 | v062_M1qmc | 3240 | 109 | 47 | 5 |
| v062_M1qmc_s3 | v062_M1qmc | 2826 | 98 | 44 | 3 |

Think-rate is read from returned reasoning (trace thinking events).
Content-without-reasoning is a trace proxy for responses with no <channel|> (definition in report.json). Compaction counts are reported for both slots (v055 beside v062); rows other than the two slots are descriptive only.
