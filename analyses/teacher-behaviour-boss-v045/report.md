# Teacher trajectory behaviour by arm

Data sha256 2ecfcb22d3fc; reference mechanism_metrics_v032 sha256 36042f6d; loop threshold 5. Details: README.md.

| metric | arm_ordinary_tokmatched | arm_short | all_resolved |
|---|---|---|---|
| n trajectories (well-formed) | 726 | 764 | 1670 |
| malformed (excluded) | 0 | 0 | 0 |
| resolved | all (arm definition) | all (arm definition) | all (arm definition) |
| tier share external_A | 0.017 | 0.013 | 0.008 |
| tier share external_B | 0.587 | 0.665 | 0.437 |
| tier share external_B2 | 0.346 | 0.257 | 0.493 |
| tier share native | 0.051 | 0.065 | 0.062 |
| mean n_steps / n_tool_calls | 34.71 / 34.63 | 34.14 / 34.05 | 33.95 / 33.86 |
| loop rate (max multiplicity >= 5) | 0.007 | 0.004 | 0.008 |
| max multiplicity mean / median | 2.35 / 2.0 | 2.28 / 2.0 | 2.31 / 2.0 |
| distinct-call ratio (mean) | 0.929 | 0.932 | 0.926 |
| submit_patch rate | 1.000 | 1.000 | 1.000 |
| first action: find_opener | 0.949 | 0.935 | 0.938 |
| first action: wrapper_opener | 0.000 | 0.000 | 0.000 |
| first action: repro_script_opener | 0.000 | 0.000 | 0.000 |
| first action: other | 0.051 | 0.065 | 0.062 |
| first action: no_tool_call | 0.000 | 0.000 | 0.000 |
| M5 repeat rate after error (mean; traj with >= 1 eligible, eligible transitions) | 0.002 (traj 609/726, elig 1617) | 0.002 (traj 642/764, elig 1627) | 0.002 (traj 1414/1670, elig 3641) |
| M5 repeat rate after silent (mean; traj with >= 1 eligible, eligible transitions) | 0.000 (traj 726/726, elig 9716) | 0.000 (traj 764/764, elig 10036) | 0.000 (traj 1670/1670, elig 21924) |
| M5 repeat rate after normal (mean; traj with >= 1 eligible, eligible transitions) | 0.000 (traj 725/726, elig 13067) | 0.000 (traj 763/764, elig 13576) | 0.000 (traj 1668/1670, elig 29280) |
| thought present, per-trajectory mean | 0.052 | 0.066 | 0.062 |
| thought present, pooled over model turns | 0.027 | 0.032 | 0.032 |
| trajectories whose first header has the think token | 726 / 726 | 764 / 764 | 1670 / 1670 |
