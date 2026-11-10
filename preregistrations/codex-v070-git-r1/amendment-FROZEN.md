# v070 Git submission-path repair amendment — DRAFT for review
Date: 2026-11-09 UTC
Execution owner: Codex. Independent verifier: Claude.
User authorization: "그냥 코드 실행이랑 캐글에 제출까지 너가 진행해"; "이제 클로드가 검증할거야". Today's Kaggle submission already exists; no duplicate submission is part of this experiment.

## Status and provenance
This is an amendment after the sealed v068 failure was inspected, before any corrected v070 rollout. It is not an unchanged original preregistration. Original v068/v070 records, contracts, snapshots, installed packages, and artifacts remain immutable.
Original prereg SHA256 b7b4d4be49973aa01407f4b40e0e9e71bdb90f10b342f62837e0bee837a929cb.
Decision is C with mandatory fallback B. Momus reviewed the design conditions; this exact document must be confirmed and frozen before implementation or PIN.
New code and artifacts live only under PROJECT_ROOT/codex/experiments/v070-git-r1/. Existing shared GPU/evaluation locks remain authoritative.

## Mechanism and minimal scope
Installed setup_baseline_commit stages/commits/tags but does not initialize Git or check success. submit_patch requires a Git diff against _swegemma_baseline or HEAD. Original source archives lack Git metadata.
Implement an explicitly recorded v070-only startup wrapper, invoked within the existing v043 guarded evaluation path. It replaces only workspace baseline preparation in the three bound harness modules (container_setup, agent_runner, verification), before evaluation. It initializes a workspace-local Git repository where missing, configures repository-local commit identity, performs the existing baseline staging/commit/tag operation with checked return codes, and requires the workspace Git root, valid HEAD/tag and clean initial diff.
Preserve source file contents around Git initialization; retain the harness's existing setup/config preparation order. No installed package edits, snapshot edits, tool semantics changes, grader changes, hidden/gold data in production input, or global Git configuration.
Run and analysis clones may change only paths, generation names, explicit repair/amendment/control pins and comparator withholding needed for this generation. All frozen model, adapters, template, budget, seeds, arm order, producer and threshold rules remain fixed.
Existing test files must NOT be executed until inspected for real-path effects. New tests must sandbox every path and mock PIN/activation/locks; no test may trigger a live PIN or GPU service.

## Controls before PIN; no adaptive task filtering
Use the identical 12 eligible DEV tasks and order already fixed in v068. The original raw task sources are evaluator-only. For every task:
1. Production workspace setup plus repaired baseline succeeds; source tree unchanged by Git initialization and no gold edits in baseline.
2. Apply gold as an offline diagnostic edit, then call the actual submit_patch function through an official tool context. Require success response, correct nonempty captured patch, persisted patch artifact and hash.
3. Send THAT captured patch (not a bypass copy of task.patch) through fresh official verification with the same environment, runtime routing, resource policy and repaired setup; require resolved with expected target-test collection and no infrastructure errors.
4. Separately submit an empty patch through the same tool and verification path; require unresolved with expected target tests, not an infrastructure failure. Preserve evidence for expected harness empty-artifact behavior.
5. Fixture check for a tracked edit plus a nonignored new file: capture/filter behavior preserved.
All control workspaces are disposable and separate from production rollout workspaces. No gold edits/patch artifacts or gold history are available to inference; final fresh rollout starts are checked clean.
Require all 12 gold and all 12 empty controls to meet these criteria. A failed real control triggers B; do not replace/drop tasks or repeatedly tune environment against controls. Unit-test defects may be fixed before real controls. Record all attempted controls, including failed ones.
The controller must reject PIN without immutable passing controls, wrapper/code hashes and reviewer verification. No result-dependent success-only filtering.

## Readout and validity
Preserve the original E1/E2/E3 thresholds, complete-paired-panel requirement, all three arms and all three decoding seeds. Report repair descriptively and all unsuccessful runs.
Original producer submit_rate measures parsed submit_patch ATTEMPTS, not successful patch capture. Retain it for the frozen primary definition and label it accurately. Add descriptive successful-capture/artifact/error counters; these do not replace primary thresholds.
Any recurrence of the specific Git setup/submission defect is an infrastructure fault and stops the generation; it must not produce a scientific collapse label from that defect.
WITHHOLD the v068 A0mc comparator entirely (including reused rows, CIs and base-superiority prose). This supersedes the original comparator clause. No corrected-baseline inference is claimed.
Keep bytecode caches; existing r22-style regular __pycache__/*.pyc hash exclusion remains declared, with non-cache identity checks unchanged.
Matched names/sampling/template/versions/budgets remain checked against the frozen references; deliberate setup/wrapper and generation changes are explicit amendment differences rather than a false claim of complete environment identity.

## Deadlines and fallback
All code/tests, passing controls, code review and PIN must finish by 2026-11-09 22:00:00 UTC. Otherwise B: v070 not run due inability to validate the external submit path in time.
GPU alone after locks/preflight; maximum 3 hours from server-ready. No overlapping control job. Original sealed readout deadline 2026-11-10 09:00:00 UTC remains. Preserve original stop/cleanup rules plus Git infrastructure fault stop.
No automatic retry after a corrected scientific rollout faults. Partial results follow original incomplete/descriptive rules, never a complete E-cell claim.
B keeps K9 on the Rich holdout only and reports v070 not run; no inference from an unexecuted panel.

## v068 reporting
Retain the frozen z=0 output and raw counts in the audit trail. In scientific tables mark infrastructure-invalid, not model repair performance. All 144 rows had zero patch and no verification; 91 attempted submission and failed Git diff, while 53 never attempted.
Suggested text: "External validation was invalidated by a harness defect: workspaces lacked Git metadata required by submit_patch. No agent patch reached verification; recorded scores were 0/36 per arm. These records cannot assess repair or confirm the termination contrast."
Rich-holdout absence of this error is reported by Claude (2,253 traces); not independently rescanned in this amendment.
