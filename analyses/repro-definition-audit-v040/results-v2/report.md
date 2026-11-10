# Reproduction-script definition harmonization v040

Post-exploration measurement harmonization; frozen decision thresholds unchanged

The raw and parsed predicates are preserved separately. Unknowns remain explicit.

| Arm | Raw trajectories anywhere | Common trajectories anywhere | Common first completion | Common first parsed action |
|---|---|---|---|---|
| arm_ordinary_tokmatched | 535/726 known; 0 unknown | 435/726 known; 0 unknown | 0/726 known; 0 unknown | 0/726 known; 0 unknown |
| arm_short | 554/764 known; 0 unknown | 456/762 known; 2 unknown | 0/764 known; 0 unknown | 0/764 known; 0 unknown |

Native captured-call validation: {"capture_disagreement_or_unmapped": 2, "same_parsed_calls_as_native_capture": 1100}

- Unknown mappings are not zero-filled.
- Training original trajectories and inference task episodes are different units.
- First supervised completion and first parsed action are separate scopes.
- This audit identifies a descriptive ordering difference, not its causal mechanism or a new paper finding.
- Existing inference labels are reused from v032; no new model execution or outcome readout.
