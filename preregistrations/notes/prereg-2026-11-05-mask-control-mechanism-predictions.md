---
date: 2026-11-05
type: decision
tags: [gemma4, paper-track, pre-registration, mask-control, mechanism]
related: [[finding-2026-11-04-sft-loop-collapse]], [[paper-experiment-contract-v0.3.1-mechanism-amendment-DRAFT]]
---
# Pre-registered mechanism predictions for mask-control v0.3.5 (M1 − M0) and the A1-hist packing arm

Author: Boss. Frozen at the sha256 / server mtime recorded in the session note, while M0 training was still in its first steps; no M0/M1 adapter, evaluation or analysis existed (`sft/runs/A1_{M0,M1}_v035` absent, `results/v035_*` absent). Momus review applied (three text fixes: producer pins, three-way readings, implementable packing rule). This note changes no Codex artifact; Codex's frozen readout (`contracts/mask-control-v035-readout.json`, sha 6319262b…) governs the outcome analysis and its claim limits apply here verbatim.

## A. Scope and exposure
M0 = fresh re-training of the unchanged A1 data (arm_ordinary_tokmatched, 726 trajectories, 1,661 chunks, 4,391,649 supervised targets); M1 = identical except that the 26,557 empty-thought target positions (23,833 `<channel|>` closes with the opener in the prompt + 681 × 4 opener-and-close runs; 0.60 % of targets) are masked, per-update loss normalised by M0's target count (208 update groups). Boss has seen the A0/A1/A2 outcomes and the training-example audit (`analysis/sft-audit-v033/full`), so the thresholds below are exploration-informed; the predictions are prospective for M0/M1.

## B. Metrics, producers and test (pinned to the readout)
- Producer: `tools/mechanism_metrics_v032.py` (sha 36042f6d…, the readout's `secondary_mechanisms.producer`), ids `analysis/mechanism-v032/primary40_ids.txt` (sha c7125743…), on `results/v035_{M0,M1}_s{1,2,3}`. Metrics: M1 think-rate (share of agent steps with non-empty thought text), M2 loop rate (identical call ≥ 5×) and median max-repeat, M3 submit rate, M4 first-action shares (`repro_script_opener`, `find_opener`, `wrapper_opener`, `no_tool_call`, `other`), M5 repeat rate after error / silent / normal observations, error-class counts.
- Arm-level M5 in this note = per-arm `nonrecovery.error.mean_repeat_rate` over the tasks with ≥ 1 eligible error transition in that arm; it is distinct from the readout's common-eligibility paired contrast, which is also reported.
- Test: paired M1 − M0 differences of 3-seed task means via `tools/analyze_mechanism_v031.py` (sha 821f2294…, `paired_task_bootstrap`, 20,000 draws, seed 20261103), as `analysis/mechanism-ci-v031` did for A1 − A0. Missing trace → metric undefined for that task-seed, reported, never zero-filled. Descriptive intervals, no multiplicity adjustment.
- Arm value = mean over the three seeds of the per-seed task means. A0 reference values = the sealed `analysis/v02-v025/report.json` (sha b2a2ebba…) and `analysis/mechanism-ci-v031/report.json`.

## C. Predictions (thresholds fixed now; every region has a label)
- **P-A control replication.** PASS = M0 think-rate < 0.05 AND loop rate ≥ 0.40 AND submit ≤ 0.60 AND resolved ≤ 1/40 in every seed. FAIL = M0 think-rate > 0.50 OR resolved ≥ 3/40 in ≥ 2 seeds → the A1 collapse did not replicate under re-training; every downstream claim is reported as training-seed-sensitive ("observed once"). Otherwise = PARTIAL replication: P-B/P-C are read but flagged replication-weak.
- **P-B channel restoration.** HOLDS = M1 think-rate ≥ 0.50 (A0 ≈ 0.90). FAILS = M1 think-rate < 0.10 → masking the direct empty-thought targets did not restore the channel; the "masked-think remedy transfers" claim fails at step one. Otherwise = INTERMEDIATE: P-C is not read.
- **P-C behaviour (read only when P-B HOLDS).** C1 recovery = loop ≤ 0.10 AND submit ≥ 0.90 AND M5-error ≤ 0.10 AND repro-script-first ≤ 0.20 AND pooled resolved(M1) ≥ 7/120 → the empty-thought target was the dominant cause; orphan contexts were tolerated. C2 partial = loop ≥ 0.30 OR repro-script-first ≥ 0.40 OR pooled resolved(M1) ≤ 2/120 → thinking is necessary, not sufficient; the remaining collapse is attributed to the other example properties (orphan contexts, wrapper/opener imitation) and a history arm becomes decisive. Otherwise = INTERMEDIATE, no reading. M5 support floor: the M5-error clause needs ≥ 10 M1 tasks with an eligible error transition in every seed; with fewer, the clause is undefined and C1 cannot be met (C2 is read on its own conditions). C1 and C2 are disjoint.
- **C3 negative replication** = P-B FAILS, regardless of loop rate (loops reported descriptively).
- **P-D exploratory H-pack signature.** Prediction: repro-script-first is unchanged by masking, |M1 − M0| ≤ 0.15, because it is hypothesised to come from history-dropped contexts (0 % first in the data, 63–68 % first under A1). A drop > 0.30 would instead implicate thought supervision in front-loading. Otherwise = no reading.

## D. Rules
Primary40 ids only; all48 descriptive. The readout's exposure and claim limits apply (one training seed; known holdout; masking has prior art — Reasoning-Trace Collapse 2605.21127, ThinkPack; observed thought text ≠ internal reasoning; no mediation claim). Nothing here is a confirmatory test. The history arm (A1-hist) is specified in a separate, unfrozen proposal note; its control design is pending an oracle consult and Codex's implementation review.
