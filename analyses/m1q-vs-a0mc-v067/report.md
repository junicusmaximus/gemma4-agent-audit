# M1q versus the matched-template base A0mc (prereg v067, part A)

Slots: M0 = v066_A0mc (E4B QAT base, no adapter, matched template), M1 = v062_M1qmc (M1q adapter, matched template). Contrast M1 - M0 on pooled resolved, paired by task with the three decoding seeds grouped; 20000 task-level bootstrap draws (seed 20261103); prereg v067 766c1e45. Exploratory; known holdout; one training seed.

Pre-written reading on primary40 (exactly one sentence applies, chosen by the CI): **otherwise**

> relative to the same-template base the difference is uncertain (10/120 vs 12/120; -1.67 pp [-8.33, +5.00])

| population | v062_M1qmc resolved by seed | v066_A0mc resolved by seed | effect pp [95% CI] |
|---|---|---|---|
| primary40 | [4, 2, 4] | [4, 3, 5] | -1.667 [-8.333, +5.000] |
| all48_sensitivity | [4, 2, 4] | [4, 3, 5] | -1.389 [-6.944, +4.167] |
| control_clean36_sensitivity | [4, 2, 4] | [4, 3, 5] | -1.852 [-9.259, +5.556] |

Termination metrics on primary40, three-seed means (descriptive only, copied from the sealed reports; no label):

| row | loop | submit | m5_error | think | repro |
|---|---|---|---|---|---|
| v062_M1qmc | 0.775 | 0.5750000000000001 | 0.3928414952826232 | 0.9054079430259293 | 0.0 |
| v066_A0mc | 0.008333333333333333 | 1.0 | 0.014321119364979014 | 0.8987083981212338 | 0.0 |

Reproduction check: the sealed v062 resolution statistics were re-derived exactly in all three populations before any v066 row was used.
Never written: "recovery", "partial recovery", "closed the gap", "masking restores / causes repair", "no-collapse".
