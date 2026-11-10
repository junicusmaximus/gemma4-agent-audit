# v0.2.3 operational amendment: complete configured budget category

Recorded on 2026-11-04 UTC after the v0.2.1 and v0.2.2 non-holdout canary failures, before inspecting any A0/A1/A2 holdout resolution outcomes. Codex authored this correction; Oracle reviewed the repeated failure and recommended the complete source-defined budget category. The prior v0.2.2 amendment remains preserved (SHA256 83698d22f206925bb71dfec95ee9694aabd173e960a776a432f7afdf08c10d45).

## Observations and cause

v0.2.1 A1 routing passed with 12 successful actions and no recorded infrastructure fault, then its canary task exceeded the four-minute session budget. v0.2.2 accepted only that session-expiry condition. Its new A1 canary instead reached the configured 80-turn limit after 84 successful, correctly routed actions; infrastructure/HTTP/read-timeout/empty-action faults were zero and all requests drained. The row reported exactly: Agent exceeded turns budget (80 turns).

Neither recovery created qualified trained-arm holdout runs. All failed canary artifacts, raw logs and original result rows are retained. The same predetermined non-holdout task fastapi_11194 was used throughout; no easier task or favorable outcome was selected.

The gate conflated ordinary task-budget exhaustion with failed routing or infrastructure. The first correction covered only one of the harness's three configured budget conditions. Installed swegemma/harness/agent_runner.py (SHA256 4d7eccabd203c26f7c39e882684d2fd851f383f6a5c76c4a75b0e559cd5d6a18) explicitly produces session, tool-call and turn-budget TaskResult errors. Raw ADK LLM-limit exceptions are normalized to the configured turn-budget error. Generic fallback and sandbox-execution errors are separate cases.

## Complete, narrow canary rule

Keep strict observe_run checks unchanged: correct adapter exclusively, successful model action, no disqualifying HTTP/provider/request-timeout fault, and complete request draining. A task-budget exception must never excuse an infrastructure failure.

After strict routing succeeds and exactly the expected task row exists, accept no task error or exactly one of these configured conditions:
- Agent exceeded session timeout (4.0 min)
- Agent exceeded tool call budget (28 calls)
- Agent exceeded turns budget (80 turns)

Generate the messages from the frozen effective limits rather than accepting generic substrings. Reject wrong limits, raw ADK/fallback messages, tool-response budget text, unknown/sandbox/worker errors, malformed field types and conflicting error fields, including two different otherwise-allowed conditions.

Record accepted_budget_exhaustion, its category, the exact message and the original task-result hash. Do not rewrite the result. Preserve the same canary task, submissions, adapter hashes and budgets. No successful patch or task resolution is required to demonstrate routing.

## Unchanged science and prospective outputs

The v0.2 scientific contract and freeze remain authoritative and immutable:
a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355.

Keep hypotheses, training/data/splits, frozen A0, 48-task evaluation/40-task primary/36-task sensitivity populations, three seeds, compaction/cache, effective budgets and statistical analysis unchanged. Holdout task errors/timeouts remain unsolved. Unknown or disqualifying infrastructure failures still prevent qualification.

The six newly qualified v0.2.3 runs prospectively supply trained-arm analysis inputs:
results/v023_a1_e4b_s1, results/v023_a1_e4b_s2, results/v023_a1_e4b_s3,
results/v023_a2_e4b_s1, results/v023_a2_e4b_s2, results/v023_a2_e4b_s3.

Use new v023 canary/log/evidence locations; refuse destination collisions. Both arms must pass every gate before the six holdout runs begin. Legacy and v0.2.1/v0.2.2 artifacts remain sealed diagnostics; no score-based selection, overwriting or silent retry is permitted.

Pin this note, new code/tests, harness taxonomy source, original freeze and serving/checkpoint/submission identities in operational evidence. Reuse the existing verified server; no retraining or server takeover is authorized by this amendment. Live gates, not unit tests alone, determine readiness.

Completion timestamps and non-holdout canary operational fields have been inspected for monitoring. No holdout resolution outcomes have been inspected.
