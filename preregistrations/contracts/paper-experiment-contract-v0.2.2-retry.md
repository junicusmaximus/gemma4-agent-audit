# v0.2.2 operational amendment: routing canary task-budget expiry

Recorded on 2026-11-04 UTC after observing the v0.2.1 non-holdout canary failure and before inspecting A0/A1/A2 holdout resolution outcomes. This is a Codex-authored operational correction, reviewed for executability by Momus. It is not represented as preregistration before training or before the canary failure.

## Preserved specification and chronology

The v0.2 scientific contract remains authoritative for hypotheses, data, populations, eligibility, A0 baseline, three decoding seeds, budgets, compaction/cache and statistical analysis. Its freeze manifest remains unchanged:
a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355.

Preserve v0.2.1 notes, code, logs, manifests and results:
- v0.2.1 retry note SHA256: a8b488f8e2974491746212189a13386ee07d168b6ef25339137c8a8df6165f32
- v0.2.1 runner SHA256: 22ecf8b92c3145ab72b509f9dcf2b45a90ee93b30c29d2caaf7f784b5cf8949d
- v0.2.1 A1 routing receipt SHA256: 2169b9db7469479957973ebf2f64f588fddaddc32efb28603fec32532c352312

The resumed legacy chain completed training and produced its six evaluation directories. The v0.2.1 recovery copied 866 files in 21 transfers; a separate read-only audit found no missing or mismatched canonical hashes. A2 completed at 2026-11-03 19:52:15 UTC. Canonical adapter hashes:
- A1: 0e0523712dae1b96b4126b27bd4adaebb7994bef87b5ee893007016506bccdd9
- A2: 601cb392d92ccc5bb96f66b169d9a674be60b58eb567e16f7ee41631b6c94abc

At 2026-11-03 23:53:39 UTC, v0.2.1 stopped after its A1 canary, before A2's canary and before any qualified holdout evaluations. For the predetermined non-holdout task fastapi_11194, its routing receipt records 12 observed requests and 12 successful actions, all A1, zero infrastructure/HTTP/request-timeout/empty-action faults, and drained=true. One downstream client disconnect was recorded. The task row reports exactly: Agent exceeded session timeout (4.0 min).

The strict HTTP/routing observer returned successfully; the subsequent blanket error-free task-row check rejected this ordinary session-budget expiry. Current A1 routing was demonstrated; this does not reconstruct the legacy evaluations' routing.

## Narrow gate correction

For new v0.2.2 canaries, keep the same predetermined task, submission/configuration, adapter hashes and effective four-minute task budget. Do not choose an easier task after failure.

Keep strict observer requirements unchanged: correct adapter exclusively, successful HTTP/model action, no disqualifying provider/HTTP/request-timeout errors, and complete request draining. An earlier successful response never excuses a later infrastructure fault.

After those checks pass and exactly the expected task row exists, accept either no task error or only the exact harness session-budget-expiry string for the configured limit. Reject any other error, incorrect limit, unexpected field type or conflicting error fields. Record accepted_budget_timeout and the exact permitted message in gate evidence; do not alter the original task result.

The exception applies only to the routing canary's task-row completion condition. Holdout task timeouts/errors continue to count as unsolved under the unchanged scientific contract. Infrastructure failures still prevent a run from being qualified.

## Prospective output rule

Keep legacy evaluations and the failed v0.2.1 canary sealed as diagnostic evidence. There are no v0.2.1 trained-arm primary runs to select or discard. The six new qualified v0.2.2 runs supply the trained-arm inputs to the frozen analysis:
- results/v022_a1_e4b_s1, results/v022_a1_e4b_s2, results/v022_a1_e4b_s3
- results/v022_a2_e4b_s1, results/v022_a2_e4b_s2, results/v022_a2_e4b_s3

Use distinct results/v022_canary_A1 and results/v022_canary_A2, logs and migration/recovery-v022/evaluation evidence. Refuse existing destinations. Run all six only after both arms pass every gate. Preserve failed or incomplete outputs; no silent overwrite or score-dependent retry/selection.

Seal the new note, runner/test hashes, original freeze, exact serving-process identity and checkpoint/submission hashes in the v0.2.2 operational evidence. Reuse the existing server only after ownership, identity and immutable-artifact checks. No retraining or server replacement is part of this correction.

Raw evaluation logs may contain resolution summaries and remain sealed. Status and routing receipts contain operational evidence only. No holdout resolution outcomes have been inspected in this correction.
