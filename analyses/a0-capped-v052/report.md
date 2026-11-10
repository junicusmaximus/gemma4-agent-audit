# Probe A (v052): A0cap versus M0 in the capped v043 environment

Exploratory known-holdout readout. A0cap is the untrained base on a LoRA-off server; M0 is the fresh v043 M0 adapter on a LoRA-on server.
Effects are A0cap minus M0 (v042 slot M1 minus slot M0) with the 20,000-draw whole-task bootstrap; all intervals are descriptive.

Probe A label (primary40): A-i. Pooled resolved 15/120, loop 0.0083; predicates {"pooled_ge_4": true, "loop_le_0.10": true, "pooled_le_1": false}.

Reading: environment inert: the cap does not drive the zero outcome of M0/M1; A0cap vs M0 is a within-environment, task-paired descriptive contrast (declared LoRA-off vs LoRA-on server-flag difference)

| Population | A0cap resolved by seed | M0 resolved by seed | A0cap - M0 (pp) | 95% CI (pp) |
|---|---|---|---:|---|
| all48_sensitivity | [5, 5, 5] | [0, 0, 0] | 10.417 | [3.472, 18.750] |
| control_clean36_sensitivity | [5, 5, 5] | [0, 0, 0] | 13.889 | [4.630, 25.000] |
| primary40 | [5, 5, 5] | [0, 0, 0] | 12.500 | [4.167, 22.500] |

| Population | Arm | think | loop | submit |
|---|---|---:|---:|---:|
| all48_sensitivity | M0 | 0.0000 | 0.5972 | 0.5764 |
| all48_sensitivity | A0cap | 0.9084 | 0.0139 | 1.0000 |
| control_clean36_sensitivity | M0 | 0.0000 | 0.6204 | 0.5926 |
| control_clean36_sensitivity | A0cap | 0.9063 | 0.0093 | 1.0000 |
| primary40 | M0 | 0.0000 | 0.6167 | 0.6000 |
| primary40 | A0cap | 0.9081 | 0.0083 | 1.0000 |

| Secondary metric, primary40 | A0cap - M0 | 95% CI | Units |
|---|---:|---|---|
| M1.observed_thinking_step_rate | 90.8093 | [89.853, 91.694] | percentage points |
| M2.loop_rate | -60.8333 | [-70.000, -51.667] | percentage points |
| M2.max_repeat | -24.2000 | [-28.467, -19.800] | count |
| M3.submit_rate | 40.0000 | [31.667, 49.167] | percentage points |
| M4.opener.find_opener | 0.0000 | [0.000, 0.000] | percentage points |
| M4.opener.wrapper_opener | -1.6667 | [-4.167, 0.000] | percentage points |
| M4.opener.repro_script_opener | -65.0000 | [-72.500, -56.667] | percentage points |
| M4.opener.other | 85.0000 | [79.167, 90.833] | percentage points |
| M4.opener.no_tool_call | -18.3333 | [-24.167, -12.500] | percentage points |
| M4.opener.any_target | -66.6667 | [-75.000, -58.333] | percentage points |
| M5.nonrecovery.error | -47.4288 | [-58.089, -36.583] | percentage points |
| M5.nonrecovery.silent | 0.2665 | [-1.297, 1.562] | percentage points |
| M5.nonrecovery.normal | -41.1418 | [-54.182, -28.336] | percentage points |
| M2.distinct_ratio | 64.4056 | [56.863, 71.438] | percentage points |
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
- A0cap and M0 share the v043 memory policy but differ in the server LoRA flag (A0cap LoRA-off as the sealed A0; M0 LoRA-on with six adapters); interchangeability of base outputs across the flag is untested.
- Historical A0 runs were uncapped; A0cap versus historical A0 is descriptive only.
- Probe A thresholds are exploration-informed (prereg cc043e03, thresholds unchanged since 3f9fc19f); readings are descriptive, not confirmatory.
