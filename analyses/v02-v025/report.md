# Frozen v0.2 analysis of qualified v025 runs

Primary A1-A0 conclusion: supportive local evidence of a decrease.

| Population | Comparison | Effect (pp) | 95% paired task-bootstrap interval (pp) |
|---|---|---:|---|
| primary40 | A1-A0 | -5.833 | [-12.500, -0.833] |
| primary40 | A2-A1 | 0.000 | [0.000, 0.000] |
| primary40 | A2-A0 | -5.833 | [-12.500, -0.833] |
| sensitivity36 | A1-A0 | -6.481 | [-13.889, -0.926] |
| sensitivity36 | A2-A1 | 0.000 | [0.000, 0.000] |
| sensitivity36 | A2-A0 | -6.481 | [-13.889, -0.926] |
| secondary48 | A1-A0 | -4.861 | [-10.417, -0.694] |
| secondary48 | A2-A1 | 0.000 | [0.000, 0.000] |
| secondary48 | A2-A0 | -4.861 | [-10.417, -0.694] |

A1-A0 on primary40 is the primary comparison. All other rows are secondary/exploratory.
The bootstrap carries each task's complete paired three-seed arm vector together.
Best-of-three unions and majority/sign tests are descriptive; unions are not pass@1.

- Conditional on one public repository and fixed task population.
- One training run per trained arm; decoding seeds do not measure training-seed robustness.
- A1-A0 estimates the implemented distillation pipeline, without causal mechanism isolation.
- A2 comparisons and all secondary metrics are exploratory; no multiplicity-free claims.
- Team-wide outcome blinding is not independently established.

Per-task and per-seed outcomes, logged descriptive metrics, unavailable metrics, hashes,
software versions and RNG details are in report.json.
