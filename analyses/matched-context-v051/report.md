# Matched-context readout v051

Slots frozen before reading: M0 = v043_M0 (served template), M1 = v051_M0mc (matched template); every contrast is v051_M0mc - v043_M0. Exploratory; known holdout; one adapter and training seed.

Pre-declared reading on primary40 three-seed means: **INTERMEDIATE** (MC-i False, MC-ii False).

| metric (primary40) | v043_M0 | v051_M0mc | paired task-bootstrap effect [95% CI] |
|---|---|---|---|
| loop | 0.6166666666666667 | 0.6583333333333333 | +0.042 [-0.092, +0.175] |
| submit | 0.6 | 0.6083333333333333 | +0.008 [-0.092, +0.108] |
| m5_error | 0.4908857429666174 | 0.6522169630217212 | +0.150 [-0.002, +0.305] |
| think | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |
| repro | 0.65 | 0.7083333333333334 | +0.058 [-0.017, +0.133] |

Resolution (v042 statistics, effect in pp):
- all48_sensitivity: 0.000 pp, 95% CI [0.0, 0.0]; v043_M0 [0, 0, 0], v051_M0mc [0, 0, 0].
- control_clean36_sensitivity: 0.000 pp, 95% CI [0.0, 0.0]; v043_M0 [0, 0, 0], v051_M0mc [0, 0, 0].
- primary40: 0.000 pp, 95% CI [0.0, 0.0]; v043_M0 [0, 0, 0], v051_M0mc [0, 0, 0].

Pooled v051_M0mc resolved on primary40: 0/120 (notable: False; never read as recovery).

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v043_M0_s1 | v043_M0 | 2158 | 40 | 26 | 7 |
| v043_M0_s2 | v043_M0 | 1388 | 43 | 24 | 9 |
| v043_M0_s3 | v043_M0 | 2566 | 66 | 35 | 5 |
| v051_M0mc_s1 | v051_M0mc | 1844 | 33 | 19 | 10 |
| v051_M0mc_s2 | v051_M0mc | 2206 | 57 | 33 | 7 |
| v051_M0mc_s3 | v051_M0mc | 2589 | 60 | 34 | 6 |

Think-rate is read from returned reasoning (trace thinking events) and is reported, not read, for M0.
Content-without-reasoning is a trace proxy (definition in report.json). Compaction fires earlier under the matched template because prompts are longer; counts are descriptive.
