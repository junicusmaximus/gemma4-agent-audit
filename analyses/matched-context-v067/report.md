# Matched-context readout v067 (M1q-s2, second training seed of the M1q recipe)

Slots frozen before reading: M0 = v055_M0qmc (matched template, M0q adapter), M1 = v067_M1qs2mc (matched template, M1q-s2 adapter, LoRA init seed 20261104). Readings of prereg v067 part B: (1) pooled resolved, (2) termination metrics and think rate vs M0qmc (descriptive, no label), (3) M1qs2mc - A0mc by the part (A) rule. M1q (v062, seed 1) and M1q-s2 are reported side by side, never pooled. Exploratory; known holdout; the independent-repository requirement of v062 is not lifted.

Pre-written part (B) sentence (pooled primary40 resolved 6/120): **notable**

> observed in two training seeds (10/120 and 6/120) on the known holdout

Reading (3), the pre-written part (A) sentence chosen by the primary40 CI of M1qs2mc - A0mc: **otherwise**

> relative to the same-template base the difference is uncertain (6/120 vs 12/120; -5.00 pp [-11.67, +0.00])

| population | v067_M1qs2mc resolved by seed | v066_A0mc resolved by seed | M1qs2mc - A0mc pp [95% CI] |
|---|---|---|---|
| primary40 | [3, 2, 1] | [4, 3, 5] | -5.000 [-11.667, +0.000] |
| all48_sensitivity | [3, 2, 1] | [4, 3, 5] | -4.167 [-9.722, +0.000] |
| control_clean36_sensitivity | [3, 2, 1] | [4, 3, 5] | -5.556 [-12.963, +0.000] |

Reading (1), resolution (v042 statistics, M1qs2mc - M0qmc in pp, descriptive):
- all48_sensitivity: 4.167 pp, 95% CI [0.6944444444444444, 9.027777777777777]; v055_M0qmc [0, 0, 0], v067_M1qs2mc [3, 2, 1].
- control_clean36_sensitivity: 5.556 pp, 95% CI [0.9259259259259258, 12.037037037037036]; v055_M0qmc [0, 0, 0], v067_M1qs2mc [3, 2, 1].
- primary40: 5.000 pp, 95% CI [0.8333333333333334, 10.0]; v055_M0qmc [0, 0, 0], v067_M1qs2mc [3, 2, 1].

Pooled v067_M1qs2mc resolved on primary40: 6/120 (notable: True); M1q (seed 1): 10/120; never pooled.

Reading (2), termination metrics and think rate on primary40 (three-seed means; descriptive, no label; M5-error: read):

| metric (primary40) | v055_M0qmc | v067_M1qs2mc | paired task-bootstrap effect [95% CI] |
|---|---|---|---|
| loop | 0.7583333333333333 | 0.8083333333333332 | +0.050 [-0.058, +0.150] |
| submit | 0.5083333333333333 | 0.4916666666666667 | -0.017 [-0.133, +0.108] |
| m5_error | 0.4563850206869089 | 0.5463701020220871 | +0.117 [-0.004, +0.238] |
| think | 0.0 | 0.8757449632538709 | +0.876 [+0.807, +0.933] |
| repro | 0.0 | 0.0 | +0.000 [+0.000, +0.000] |

Side by side (primary40 three-seed means; the two training seeds are never pooled; rows other than the slots are descriptive):

| row | loop | submit | m5_error | think | repro | resolved by seed |
|---|---|---|---|---|---|---|
| v055_M0qmc (slot M0) | 0.7583333333333333 | 0.5083333333333333 | 0.4563850206869089 | 0.0 | 0.0 | [0, 0, 0] |
| v067_M1qs2mc (slot M1, seed 2) | 0.8083333333333332 | 0.4916666666666667 | 0.5463701020220871 | 0.8757449632538709 | 0.0 | [3, 2, 1] |
| v051_M0mc | 0.6583333333333333 | 0.6083333333333333 | 0.6522169630217212 | 0.0 | 0.7083333333333334 | [0, 0, 0] |
| v043_M0_sealed | 0.6166666666666667 | 0.6 | 0.4908857429666174 | 0.0 | 0.65 | [0, 0, 0] |
| v043_M1_sealed | 0.7000000000000001 | 0.13333333333333333 | 0.6541994342481312 | 0.7761616225867747 | 0.041666666666666664 | [0, 0, 0] |
| v052_A0cap | 0.008333333333333333 | 1.0 | 0.020014478764478764 | 0.908093180337559 | 0.0 | [5, 5, 5] |
| v062_M1qmc | 0.775 | 0.5750000000000001 | 0.3928414952826232 | 0.9054079430259293 | 0.0 | [4, 2, 4] |
| v066_A0mc | 0.008333333333333333 | 1.0 | 0.014321119364979014 | 0.8987083981212338 | 0.0 | None |

Update 1 loss (descriptive): M1q 0.47084, M1q-s2 0.47084; equal at 5 decimals: True.
Reproduction check: the sealed v067-A contrast (M1q seed 1 - A0mc) was re-derived exactly in all three populations before reading (3).

| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |
|---|---|---|---|---|---|
| v055_M0qmc_s1 | v055_M0qmc | 3840 | 78 | 48 | 0 |
| v055_M0qmc_s2 | v055_M0qmc | 2050 | 44 | 22 | 2 |
| v055_M0qmc_s3 | v055_M0qmc | 3017 | 48 | 33 | 0 |
| v067_M1qs2mc_s1 | v067_M1qs2mc | 2879 | 107 | 46 | 4 |
| v067_M1qs2mc_s2 | v067_M1qs2mc | 3136 | 89 | 46 | 3 |
| v067_M1qs2mc_s3 | v067_M1qs2mc | 3040 | 110 | 45 | 4 |

Never written: "recovery", "partial recovery", "closed the gap", "masking restores / causes repair", "no-collapse".
Think-rate is read from returned reasoning (trace thinking events).
