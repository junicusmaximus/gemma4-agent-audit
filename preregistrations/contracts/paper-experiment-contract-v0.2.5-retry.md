# v0.2.5 operational amendment: response validity and action counts

Recorded on 2026-11-04 UTC after the v0.2.4 A1 canary stopped and before inspecting holdout resolution outcomes. The original v0.2 scientific contract and freeze remain immutable:
a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355.

## Observed failure and evidence limit

v0.2.4's asynchronous transport passed independent unit and real-TCP cancellation tests. Its live A1 canary then recorded 83 HTTP200 responses, all routed to A1, including one response without a recognized action. The old successful counter was incremented before the action check, so the receipt supports 82 action-bearing responses, not 83. It recorded no HTTP/read-timeout faults and drained successfully; its sole infrastructure label was canary_has_no_model_action. The task reached the configured 80-turn limit.

The old receipt did not retain choices/message/finish-reason structure. Its loose envelope check also admitted some malformed shapes. Therefore the isolated response cannot retrospectively be declared schema-valid model behavior. v0.2.4 remains an unclassified diagnostic failure. It created no qualified trained-arm holdout runs. The unused v0.2.4 analysis seal and matching implementation are preserved; no analysis waiter was launched.

## Source-based correction

Installed vLLM and OpenAI message schemas permit content=None and no tool calls. A structurally valid empty, reasoning-only or refusal-only completion can therefore be model behavior without a supported action. Conversely, empty choices, a missing message, invalid field types or malformed tool-call objects are invalid provider responses.

The frozen harness uses non-streaming model calls. Validate that complete JSON message-envelope path; do not treat streaming deltas or usage-only chunks as complete non-streaming replies.

For new v0.2.5 observations:
- Validate the nonempty choices list, choice/message objects, assistant role and content/tool-call field types before action classification.
- Reject provider errors and malformed envelopes. Function argument strings may contain invalid model-generated JSON; do not turn that model behavior into a transport fault merely by parsing the argument string.
- Count only actual action-bearing responses in successful. Track structurally accepted responses and valid empty responses separately, with the invariant accepted responses = actions + empty responses.
- Permit structurally valid empty responses as diagnostic model behavior. A strict canary still requires at least one actual action, exclusively correct aliases, zero disqualifying infrastructure/HTTP/read-timeout faults, and completed draining.
- Persist safe structural evidence and response hashes for empty/invalid cases, without prompt or response text.

Keep the reviewed ASGI/httpx cancellation propagation, sibling isolation, backend-status preservation and retry-header behavior from v0.2.4. Preserve the three exact configured task-budget exceptions and reject unknown/conflicting task errors. The predetermined canary task, submissions, limits and adapters remain the same.

## Qualification and analysis

Repeat both live adapter-effect and routing gates. Only after both arms pass, run six fresh qualified outputs:
results/v025_a1_e4b_s1, results/v025_a1_e4b_s2, results/v025_a1_e4b_s3,
results/v025_a2_e4b_s1, results/v025_a2_e4b_s2, results/v025_a2_e4b_s3.

These new qualified runs prospectively supply trained-arm analysis inputs. Keep legacy and v021-v024 artifacts sealed as diagnostics; do not select runs by resolution rate or overwrite existing outputs. Preserve frozen A0, populations, seeds, budgets, compaction/cache and statistical methods.

The original unused analyzer and its v024 seal remain unchanged. A separate versioned analyzer updates only qualification acceptance: a valid empty-response count does not veto a canary with real actions, but malformed responses, an all-empty canary and inconsistent counters still fail. Statistical calculations remain unchanged. Seal the reviewed new implementation against the explicit v025 operational manifest before decoding any real outcomes.

Record code/test/source/amendment hashes and serving/checkpoint/submission identity in the new operational evidence. Existing model server and trained adapters are retained. Unit and real-TCP tests support implementation review; live gates remain required.

## Exposure

Only non-holdout canary error/operational metadata and completion timestamps have been inspected. No A0/A1/A2 holdout resolution outcomes were inspected when deciding this correction. Do not infer the classification of the unrecoverable v024 response from a later successful run.
