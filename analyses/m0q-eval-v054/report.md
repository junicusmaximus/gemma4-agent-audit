# M0q (v054) versus the sealed v043 M0 in the capped v043 environment

Exploratory known-holdout readout. M0q is M0 retrained on the dequantized QAT base; both are served LoRA-on with the v043 template.
Effects are M0q minus M0 (v042 slot M1 minus slot M0) with the 20,000-draw whole-task bootstrap; all intervals are descriptive.

| Reading, primary40 | M0q - M0 | 95% CI | Reading |
|---|---:|---|---|
| resolution | 0.0000 | [0.000, 0.000] | does_not_differ |
| loop | 17.5000 | [7.500, 26.667] | differs |
| submit | -15.0000 | [-25.833, -4.167] | differs |
| m5_error | 0.0970 | [-11.912, 12.175] | does_not_differ |
| repro_first | -65.0000 | [-72.500, -56.667] | differs |
| think | 0.0000 | [0.000, 0.000] | descriptive |

M0q pooled resolved (primary40): 0/120; notable and bounded (>= 4/120): False.

| Population | M0q resolved by seed | M0 resolved by seed | M0q - M0 (pp) | 95% CI (pp) |
|---|---|---|---:|---|
| all48_sensitivity | [0, 0, 0] | [0, 0, 0] | 0.000 | [0.000, 0.000] |
| control_clean36_sensitivity | [0, 0, 0] | [0, 0, 0] | 0.000 | [0.000, 0.000] |
| primary40 | [0, 0, 0] | [0, 0, 0] | 0.000 | [0.000, 0.000] |

| Population | Arm | think | loop | submit |
|---|---|---:|---:|---:|
| all48_sensitivity | M0 | 0.0000 | 0.5972 | 0.5764 |
| all48_sensitivity | M0q | 0.0000 | 0.7639 | 0.4236 |
| control_clean36_sensitivity | M0 | 0.0000 | 0.6204 | 0.5926 |
| control_clean36_sensitivity | M0q | 0.0000 | 0.8056 | 0.4630 |
| primary40 | M0 | 0.0000 | 0.6167 | 0.6000 |
| primary40 | M0q | 0.0000 | 0.7917 | 0.4500 |

| Secondary metric, primary40 | M0q - M0 | 95% CI | Units |
|---|---:|---|---|
| M1.observed_thinking_step_rate | 0.0000 | [0.000, 0.000] | percentage points |
| M2.loop_rate | 17.5000 | [7.500, 26.667] | percentage points |
| M2.max_repeat | 9.0583 | [3.533, 14.367] | count |
| M3.submit_rate | -15.0000 | [-25.833, -4.167] | percentage points |
| M4.opener.find_opener | 78.3333 | [66.667, 89.167] | percentage points |
| M4.opener.wrapper_opener | -1.6667 | [-4.167, 0.000] | percentage points |
| M4.opener.repro_script_opener | -65.0000 | [-72.500, -56.667] | percentage points |
| M4.opener.other | 6.6667 | [-5.000, 19.167] | percentage points |
| M4.opener.no_tool_call | -18.3333 | [-24.167, -12.500] | percentage points |
| M4.opener.any_target | 11.6667 | [-2.500, 25.833] | percentage points |
| M5.nonrecovery.error | 0.0970 | [-11.912, 12.175] | percentage points |
| M5.nonrecovery.silent | 8.3071 | [4.108, 13.029] | percentage points |
| M5.nonrecovery.normal | -29.0506 | [-42.696, -14.920] | percentage points |
| M2.distinct_ratio | 11.0678 | [3.112, 19.121] | percentage points |
| M6.thought_tokens_median | undefined | undefined | tokens |
| M6.thought_tokens_max | undefined | undefined | tokens |
| M6.thought_chars_median | undefined | undefined | characters |

Self-checks: recomputed M0 records equal the sealed v043 rows (144); producer CLI reproduces think/loop/submit for 3 runs and 120 per-task rows.

Limitations:
- Known-holdout exploratory intervention prompted by prior observations; not independent heldout confirmation.
- One paired training seed; three decoding seeds do not establish training-seed generality.
- Semantic deduplication against training data is unverified.
- Empty-thought/scaffold masking has prior art including ThinkPack; no method novelty claim.
- No claim that history dropping is harmless or that effects are specific to small models.
- Observed thought text is not direct evidence of internal reasoning; secondary mechanisms do not establish causal mediation.
- Historical A0/A1 comparisons remain descriptive; historical A1 initialization is not reconstructed.
- M0q and M0 differ in the training base weights (dequantized QAT versus bf16 unquantized); data, groups, hyper-parameters, seed and LoRA init are matched; loss trajectory and optimizer numerics unavoidably differ.
- The v054 server registers three M0q aliases (max_loras 3); the sealed v043 server registered six adapters. Template, engine build, flags and memory policy are otherwise the v043 ones.
- The comparator is the sealed v043 M0 (v043 template, capped environment) as set by the prereg re-freeze of 2026-11-05 18:3x UTC; the matched-context v051 M0 named in prereg section 3 is not part of this contrast.
- Readings are descriptive (prereg section 3): differs / does not differ beyond the paired interval; resolution >= 4/120 is notable and bounded.
