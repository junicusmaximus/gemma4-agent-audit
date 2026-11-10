# Matched-context readout v055 (M0q)

Slots frozen before reading: M0 = v054_M0q (served template), M1 = v055_M0qmc (matched template); every contrast is v055_M0qmc - v054_M0q. Exploratory; known holdout; one adapter and training seed.

Pre-declared reading on primary40 three-seed means: **MCq-ii** (MCq-i False, MCq-ii True).

| metric (primary40) | v054_M0q | v055_M0qmc | paired task-bootstrap effect [95% CI] |
|---|---|---|---|
| loop | 0.7916666666666666 | 0.7583333333333333 | -0.033 [-0.125, +0.058] |
| submit | 0.45 | 0.5083333333333333 | +0.058 [-0.042, +0.158] |
| m5_error | 0.5219489570111335 | 0.4563850206869089 | -0.064 [-0.162, +0.038] |
| think | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |
| repro | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |

| descriptive secondary (primary40) | paired effect [95% CI] |
|---|---|
| M4.opener.find_opener | +0.008 [-0.050, +0.075] |
| M4.opener.repro_script_opener | +0.000 [+0.000, +0.000] |
| M2.max_repeat | -1.475 [-6.142, +3.442] |
| M5.nonrecovery.silent | -0.043 [-0.095, +0.004] |
| M5.nonrecovery.normal | -0.037 [-0.093, +0.017] |

Resolution (v042 statistics, effect in pp):
- all48_sensitivity: 0.000 pp, 95% CI [0.0, 0.0]; v054_M0q [0, 0, 0], v055_M0qmc [0, 0, 0].
- control_clean36_sensitivity: 0.000 pp, 95% CI [0.0, 0.0]; v054_M0q [0, 0, 0], v055_M0qmc [0, 0, 0].
- primary40: 0.000 pp, 95% CI [0.0, 0.0]; v054_M0q [0, 0, 0], v055_M0qmc [0, 0, 0].

Pooled v055_M0qmc resolved on primary40: 0/120 (notable: False; never read as recovery).

| row (primary40 three-seed means) | loop | submit | m5_error | think | repro | resolved by seed |
|---|---|---|---|---|---|---|
| v054_M0q (slot M0) | 0.7916666666666666 | 0.45 | 0.5219489570111335 | 0.0 | 0.0 | [0, 0, 0] |
| v055_M0qmc (slot M1) | 0.7583333333333333 | 0.5083333333333333 | 0.4563850206869089 | 0.0 | 0.0 | [0, 0, 0] |
| v051_M0mc | 0.6583333333333333 | 0.6083333333333333 | 0.6522169630217212 | 0.0 | 0.7083333333333334 | [0, 0, 0] |
| v043_M0_sealed | 0.6166666666666667 | 0.6 | 0.4908857429666174 | 0.0 | 0.65 | [0, 0, 0] |
| v052_A0cap | 0.008333333333333333 | 1.0 | 0.020014478764478764 | 0.908093180337559 | 0.0 | [5, 5, 5] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v054_M0q_s1 | v054_M0q | 3712 | 83 | 47 | 2 |
| v054_M0q_s2 | v054_M0q | 1946 | 30 | 17 | 0 |
| v054_M0q_s3 | v054_M0q | 3213 | 59 | 35 | 0 |
| v055_M0qmc_s1 | v055_M0qmc | 3840 | 78 | 48 | 0 |
| v055_M0qmc_s2 | v055_M0qmc | 2050 | 44 | 22 | 2 |
| v055_M0qmc_s3 | v055_M0qmc | 3017 | 48 | 33 | 0 |

Think-rate is read from returned reasoning (trace thinking events); it is about 0 by construction in both slots.
Content-without-reasoning is a trace proxy for responses with no <channel|> (definition in report.json). Compaction counts are reported for both slots (v054 beside v055); rows other than the two slots are descriptive only.
