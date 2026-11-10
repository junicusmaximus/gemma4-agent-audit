# v0.2.1 operational recovery amendment

Recorded prospectively on 2026-11-03 UTC, after the controller failure and the user's decision to resume the legacy chain, before this session has opened A0/A1/A2 outcome rows. This is a Codex-authored operational amendment, reviewed for executability by Momus; it is not represented as a user-authored protocol or as preregistration before training.

## Unchanged scientific specification

The frozen v0.2 contract remains authoritative for hypotheses, populations, eligibility, A0 baseline, training arms, decoding seeds, budgets, compaction/cache settings and statistical analysis. Do not change or reseal it.
- Contract SHA256: 680fa70fcf1bf5903121829690a9b533c5987031edff871279073c2aba526478
- Freeze manifest SHA256: a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355
- Original controller SHA256: e48419eef233c08cc0465b219157cdf2b6160e2a2b4080bf057860ce4fb6bf5a
- Original test SHA256: 5aedc5135d2c53552401d9fa6d3cc5efb7939512ffe6273ca7cb44d5cd572008

All 48 Rich tasks are evaluated, with the frozen 40-task primary population and 36-task sensitivity population. A1 minus A0 remains primary; A2 comparisons remain exploratory and confounded. Retain seeds 1, 2, 3, effective 4-minute/28-tool-call/80-turn/240-second budgets and concurrency 4. The 20,000 paired task-cluster bootstrap, seed 20261103, and all original caveats remain unchanged.

## Incident and responsibility

The v0.2 controller started at 10:50:38 UTC while the legacy orchestration shell was stopped and its A1 child continued training. At 14:46:22 UTC, the controller failed in wait_a1 -> proc -> Path.resolve('/proc/2798497/cwd'). A finished child of the stopped shell is a zombie; its stat entry retains exit information, but its cwd is unavailable and cmdline is empty. The implementation relied on the zombie's exit status while assuming live-process cwd semantics. The previous twelve mocked gate tests and code review missed this real Linux lifecycle case. This is a controller defect, not evidence that A1 training failed.

The external collaborator reports A1 exited successfully. Completed metadata and adapter structure were independently inspected without reading loss/outcome rows. A1's adapter SHA256 is 0e0523712dae1b96b4126b27bd4adaebb7994bef87b5ee893007016506bccdd9. Four completed A1 files were copied and hash-verified into canonical storage; receipt: migration/recovery-v021/a1-transfer-20261103T153217Z/transfer-receipt.json. The failed controller log is retained.

The user chose to resume the legacy shell at 15:24:59 UTC. Its A2 child PID 2804837 was independently confirmed running arm_short with the intended configuration. The recovery preserves that job and the user's resumed chain. It does not retrain either arm, signal the legacy processes, or relaunch the failed controller.

## Prospective replacement rule

Declare this rule before outcome inspection: the six new, qualified v0.2.1 runs supply the trained-arm inputs to the frozen primary analysis. The legacy trained-arm evaluations remain sealed diagnostic artifacts, regardless of their scores. Never choose between legacy and replacement runs using resolution rates, runtime results, or apparent quality.

A later routing canary establishes present routing only. It cannot reconstruct requests from earlier runs. The legacy probe prints a one-prompt log-probability difference; it does not implement the v0.2 effect-versus-drift acceptance gate, and its pipeline does not stop evaluation on failure. Therefore a successful post-hoc probe/canary does not retroactively qualify legacy runs.

## Recovery execution

After the exact resumed legacy shell has terminated, and no project trainer/evaluator is active:
1. Preserve and hash-verify completed A2 artifacts, submissions, logs and legacy evaluation outputs into PROJECT_ROOT/gemma4. Never overwrite differing files. Transfer may inspect metadata/structure; it must not disclose resolution outcomes.
2. Verify completion metadata, intended arm/configuration, checkpoint freshness, tensor integrity, frozen input hashes, and exact copied submission configuration. Missing evidence stops qualification; mere file existence is insufficient.
3. Reuse the existing serving instance only after recording and verifying its process identity/start time, launch arguments, model names and loaded adapter paths/hashes. Do not take over an unrelated server. A server change or artifact drift invalidates qualification.
4. Run and record adapter-effect-versus-base-repeat-drift checks for both arms. Run actual-driver, non-holdout canaries through an observing proxy for both exact submissions: requested adapter, successful HTTP 200 model action, no fallback/upstream error, correct task and error-free completion. Canary resolution is not a selection criterion.
5. Only after all gates pass, evaluate six fresh directories: results/v021_a1_e4b_s1, results/v021_a1_e4b_s2, results/v021_a1_e4b_s3, results/v021_a2_e4b_s1, results/v021_a2_e4b_s2, results/v021_a2_e4b_s3. Refuse existing destinations. Recheck freeze, process identity and artifact hashes before and after each run. Record completeness separately from resolution, without printing outcomes.
6. Stop on a failed gate, missing completion evidence, identity/hash drift or destination collision. Preserve partial runs and logs. Any further infrastructure retry gets a distinct manifest and output location; never silently restart or replace results.

Seal a separate operational manifest referencing this note, the original freeze, recovery code, checkpoint/submission hashes, gate timestamps, server identity and output locations. The queued runner's code/note hashes are pinned before it can proceed. Checkpoint hashes unavailable until A2 completes are recorded then, before gates/evaluation; this is declared prospective operational registration, not a claim those hashes were known earlier.

## Controller regression fix

A separate, inactive tools/overnight_v021.py captures authenticated live identity once, then polls stat without consulting zombie cwd/cmdline. Real Linux subprocess fixtures cover transition to a zombie under a stopped parent, nonzero exit and identity failures. It is for a future new run only: its CLI requires a live identity and cannot authenticate the already-finished A1 or resume the current A2 chain. The active recovery path is the separate post-legacy evaluation runner.

## Exposure and storage

No A0/A1/A2 outcome rows or resolved-count log lines have been read in this recovery session. Automated checks may validate task IDs, structural completion and infrastructure error fields without printing or selecting by resolution. Other collaborators' exposure is not certified. Keep original logs, including any summaries the legacy script writes, sealed until analysis readiness.

Canonical storage is PROJECT_ROOT/. The active legacy workspace remains until all jobs finish and completed artifacts are verified there. This amendment does not attempt previously policy-blocked local deletion. Separate execution receipts record actual tests, hashes, launch status and results availability.
