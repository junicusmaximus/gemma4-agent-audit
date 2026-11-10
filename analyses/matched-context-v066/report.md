# Matched-context readout v066 (A0mc, descriptive)

A0mc = the E4B QAT w4a16 base with no adapter under the matched template 855bbbbb (v066 generation, 48 tasks x 3 decoding seeds, capped sandboxes). Descriptive base row: no PK reading, no contrast, no confidence interval. The v055_M0qmc rows are the sealed reference generation, reproduced row by row and shown for orientation only. Known holdout; decoding seeds are not training replicates.

## all48_sensitivity

| run | resolved | loop | submit | M5-error | think | repro-first |
|---|---|---|---|---|---|---|
| v066_A0mc_s1 | 4/48 | 0.021 | 1.000 | 0.013 | 0.905 | 0.000 |
| v066_A0mc_s2 | 3/48 | 0.021 | 1.000 | 0.028 | 0.898 | 0.000 |
| v066_A0mc_s3 | 5/48 | 0.000 | 1.000 | 0.023 | 0.900 | 0.000 |
| v066_A0mc three-seed mean | 12/144 pooled | 0.014 | 1.000 | 0.021 | 0.901 | 0.000 |
| v055_M0qmc three-seed mean (reference) | 0/144 pooled | 0.736 | 0.542 | 0.452 | 0.000 | 0.000 |

## control_clean36_sensitivity

| run | resolved | loop | submit | M5-error | think | repro-first |
|---|---|---|---|---|---|---|
| v066_A0mc_s1 | 4/36 | 0.028 | 1.000 | 0.013 | 0.902 | 0.000 |
| v066_A0mc_s2 | 3/36 | 0.000 | 1.000 | 0.019 | 0.890 | 0.000 |
| v066_A0mc_s3 | 5/36 | 0.000 | 1.000 | 0.000 | 0.899 | 0.000 |
| v066_A0mc three-seed mean | 12/108 pooled | 0.009 | 1.000 | 0.011 | 0.897 | 0.000 |
| v055_M0qmc three-seed mean (reference) | 0/108 pooled | 0.769 | 0.519 | 0.468 | 0.000 | 0.000 |

## primary40

| run | resolved | loop | submit | M5-error | think | repro-first |
|---|---|---|---|---|---|---|
| v066_A0mc_s1 | 4/40 | 0.025 | 1.000 | 0.015 | 0.904 | 0.000 |
| v066_A0mc_s2 | 3/40 | 0.000 | 1.000 | 0.023 | 0.892 | 0.000 |
| v066_A0mc_s3 | 5/40 | 0.000 | 1.000 | 0.005 | 0.900 | 0.000 |
| v066_A0mc three-seed mean | 12/120 pooled | 0.008 | 1.000 | 0.014 | 0.899 | 0.000 |
| v055_M0qmc three-seed mean (reference) | 0/120 pooled | 0.758 | 0.508 | 0.456 | 0.000 | 0.000 |

| descriptive row (primary40 three-seed means) | loop | submit | m5_error | think | repro | resolved by seed |
|---|---|---|---|---|---|---|
| v051_M0mc | 0.6583333333333333 | 0.6083333333333333 | 0.6522169630217212 | 0.0 | 0.7083333333333334 | [0, 0, 0] |
| v043_M0_sealed | 0.6166666666666667 | 0.6 | 0.4908857429666174 | 0.0 | 0.65 | [0, 0, 0] |
| v043_M1_sealed | 0.7000000000000001 | 0.13333333333333333 | 0.6541994342481312 | 0.7761616225867747 | 0.041666666666666664 | [0, 0, 0] |
| v052_A0cap | 0.008333333333333333 | 1.0 | 0.020014478764478764 | 0.908093180337559 | 0.0 | [5, 5, 5] |

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v055_M0qmc_s1 | v055_M0qmc | 3840 | 78 | 48 | 0 |
| v055_M0qmc_s2 | v055_M0qmc | 2050 | 44 | 22 | 2 |
| v055_M0qmc_s3 | v055_M0qmc | 3017 | 48 | 33 | 0 |
| v066_A0mc_s1 | v066_A0mc | 652 | 32 | 24 | 1 |
| v066_A0mc_s2 | v066_A0mc | 595 | 32 | 26 | 0 |
| v066_A0mc_s3 | v066_A0mc | 553 | 33 | 27 | 0 |

M5-error is undefined for a seed without error events and is never replaced by 0; a three-seed mean with an undefined seed is undefined.
Think-rate is read from returned reasoning (trace thinking events). v052 A0cap (default template) is cited as a descriptive row only; A0mc vs A0cap is a template difference across separate runs and is never a claim.
