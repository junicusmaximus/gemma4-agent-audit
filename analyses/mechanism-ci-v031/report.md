# Exploration-informed mechanism intervals

exploration-informed descriptive secondary analysis; outcomes already known

A1-A0 contrasts below are descriptive. Other comparisons and coverage are in report.json.

| Metric | Paired tasks | A1-A0 effect | 95% task-bootstrap interval | Units |
|---|---:|---:|---|---|
| M1.observed_thinking_step_rate | 40 | -89.9955 | [-91.3892, -88.1373] | percentage points |
| M2.loop_rate | 40 | 54.1667 | [45.0000, 62.5000] | percentage points |
| M2.max_repeat | 40 | 19.8750 | [15.8500, 24.0002] | count |
| M3.submit_rate | 40 | -55.0000 | [-63.3333, -45.8333] | percentage points |
| M4.opener.find_opener | 40 | 0.0000 | [0.0000, 0.0000] | percentage points |
| M4.opener.wrapper_opener | 40 | 0.8333 | [0.0000, 2.5000] | percentage points |
| M4.opener.repro_script_opener | 40 | 65.0000 | [55.8333, 74.1667] | percentage points |
| M4.opener.other | 40 | -99.1667 | [-100.0000, -97.5000] | percentage points |
| M4.opener.no_tool_call | 40 | 33.3333 | [25.0000, 41.6667] | percentage points |
| M4.opener.any_target | 40 | 65.8333 | [56.6667, 74.1667] | percentage points |
| M5.nonrecovery.error | 37 | 48.9953 | [38.5192, 58.8226] | percentage points |
| M5.nonrecovery.silent | 37 | 1.3333 | [-0.3754, 3.8378] | percentage points |
| M5.nonrecovery.normal | 27 | 35.8336 | [21.7996, 50.5814] | percentage points |

- All metrics and comparisons are post-exploration, descriptive and unadjusted for multiplicity.
- One public repository and one training run per arm; three decoding seeds are not training replications.
- M1 is observed thinking-text presence per agent step, not reasoning quality or an LLM-call probability.
- M5 conditions on both arms having observed eligible transitions; support differs by comparison/stratum.
- M5 averages observed seed rates; missing values are not zero-filled. Complete-three-seed sensitivity is separate.
- No causal mechanism, equivalence or multiplicity-free significance claim follows from these intervals.
- M6 is excluded: producer thought length may be tokenizer tokens or whitespace words; provenance must establish units.
