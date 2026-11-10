# Matched-context readout v058 (Hq1)

Slots frozen before reading: M0 = v055_M0qmc (matched template, M0q adapter), M1 = v058_Hq1mc (matched template, Hq1 adapter); every contrast is v058_Hq1mc - v055_M0qmc. Exploratory; known holdout; one adapter and training seed per arm.

Declared confound: Hq1 training input context tokens are x1.24 (dry-pack figure) with the matching compute increase at equal update count; supervised targets are identical (4,391,649). Equal updates are not called equal compute. Total-effect contrast, no mitigation; stated here as in every table.

Pre-declared reading on primary40 three-seed means: **INTERMEDIATE** (PK-i False, PK-ii False; M5-error: read).

| metric (primary40) | v055_M0qmc | v058_Hq1mc | paired task-bootstrap effect [95% CI] |
|---|---|---|---|
| loop | 0.7583333333333333 | 0.9583333333333334 | +0.200 [+0.125, +0.275] |
| submit | 0.5083333333333333 | 0.09166666666666667 | -0.417 [-0.500, -0.333] |
| m5_error | 0.4563850206869089 | 0.5971847730551622 | +0.126 [+0.019, +0.234] |
| think | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |
| repro | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |

| descriptive secondary (primary40) | paired effect [95% CI] |
|---|---|
| M4.opener.find_opener | -0.058 [-0.142, +0.017] |
| M4.opener.repro_script_opener | +0.000 [+0.000, +0.000] |
| M2.max_repeat | +9.725 [+3.641, +15.700] |
| M5.repeat_after.silent | +0.024 [-0.052, +0.088] |
| M5.repeat_after.normal | +0.039 [-0.008, +0.089] |

Resolution (v042 statistics, effect in pp):
- all48_sensitivity: 0.000 pp, 95% CI [0.0, 0.0]; v055_M0qmc [0, 0, 0], v058_Hq1mc [0, 0, 0].
- control_clean36_sensitivity: 0.000 pp, 95% CI [0.0, 0.0]; v055_M0qmc [0, 0, 0], v058_Hq1mc [0, 0, 0].
- primary40: 0.000 pp, 95% CI [0.0, 0.0]; v055_M0qmc [0, 0, 0], v058_Hq1mc [0, 0, 0].

Pooled v058_Hq1mc resolved on primary40: 0/120 (notable: False; descriptive, no wording beyond observed before an independent-repo confirmation).

| row (primary40 three-seed means) | loop | submit | m5_error | think | repro | resolved by seed |
|---|---|---|---|---|---|---|
| v055_M0qmc (slot M0) | 0.7583333333333333 | 0.5083333333333333 | 0.4563850206869089 | 0.0 | 0.0 | [0, 0, 0] |
| v058_Hq1mc (slot M1) | 0.9583333333333334 | 0.09166666666666667 | 0.5971847730551622 | 0.0 | 0.0 | [0, 0, 0] |
| v051_M0mc | 0.6583333333333333 | 0.6083333333333333 | 0.6522169630217212 | 0.0 | 0.7083333333333334 | [0, 0, 0] |
| v043_M0_sealed | 0.6166666666666667 | 0.6 | 0.4908857429666174 | 0.0 | 0.65 | [0, 0, 0] |
| v043_M1_sealed | 0.7000000000000001 | 0.13333333333333333 | 0.6541994342481312 | 0.7761616225867747 | 0.041666666666666664 | [0, 0, 0] |
| v052_A0cap | 0.008333333333333333 | 1.0 | 0.020014478764478764 | 0.908093180337559 | 0.0 | [5, 5, 5] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v055_M0qmc_s1 | v055_M0qmc | 3840 | 78 | 48 | 0 |
| v055_M0qmc_s2 | v055_M0qmc | 2050 | 44 | 22 | 2 |
| v055_M0qmc_s3 | v055_M0qmc | 3017 | 48 | 33 | 0 |
| v058_Hq1mc_s1 | v058_Hq1mc | 3768 | 92 | 47 | 1 |
| v058_Hq1mc_s2 | v058_Hq1mc | 3776 | 85 | 48 | 1 |
| v058_Hq1mc_s3 | v058_Hq1mc | 3840 | 65 | 47 | 1 |

Think-rate is read from returned reasoning (trace thinking events).
Content-without-reasoning is a trace proxy for responses with no <channel|> (definition in report.json). Compaction counts are reported for both slots (v055 beside v058); rows other than the two slots are descriptive only.
