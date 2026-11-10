# v0.2.4 operational amendment: cancellation-preserving observation

Recorded on 2026-11-04 UTC after identifying a transport difference during v0.2.3 and before inspecting holdout resolution outcomes. This is an operational correction. The original v0.2 scientific contract and freeze remain unchanged:
a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355.

## Confirmed difference and disposition

Both v0.2.3 adapter gates passed at 02:53:08 UTC. Its first A1 holdout evaluation then started. A source audit found that the synchronous observer delayed upstream cancellation: it blocked while reading the model response and discovered downstream disconnection only after the response completed.

The actual direct driver uses asyncio.timeout through ADK/LiteLLM and AsyncOpenAI/httpx. Cancellation closes the unfinished HTTP request. The deployed stock vLLM chat endpoint uses with_cancellation; disconnection cancels its handler, and AsyncLLM.generate aborts the engine request. Keeping the observer's upstream connection alive can therefore keep cancelled generations running alongside the other concurrent tasks. The size or direction of any score/runtime effect has not been measured.

At 02:58:20 UTC, Codex stopped only its verified owned v0.2.3 process group 2861150 (two members, none live afterward). The existing serving process 2810091/starttime 441913368 remained intact. The exact signal/identity receipt is migration/recovery-v023/transport-disqualification-stop.json.

All v0.2.3 outputs, including its partial A1 run, are diagnostic and excluded from primary analysis. They are retained without opening resolution outcomes. Earlier v0.2.1/v0.2.2 artifacts remain unchanged. This disposition was made from transport evidence, not observed scores.

## Minimal transport correction

Use the existing ASGI/httpx asynchronous stack for observation. After reading the request body, race upstream forwarding against downstream http.disconnect. On a verified downstream disconnect, cancel and close only that upstream request promptly. Do not close the shared client or cancel sibling requests.

Preserve backend response status/body and coherent end-to-end headers, including retry behavior. Record an error status as soon as headers arrive so a later body failure or cancellation cannot erase it. Observe responses without manufacturing successful actions or suppressing provider faults.

Retain the reviewed routing/infrastructure classifications, pending-request accounting, bounded draining and strict canary requirements. Verified client cancellation is a separate event; it is not an upstream read timeout or provider failure. Preserve the complete three-case configured canary-budget classification from v0.2.3. Unknown or conflicting task errors still fail.

Required transport regression evidence uses real TCP connections and slow fake upstreams, not the production model: direct-client cancellation control; proxy cancellation before headers and during a response body; prompt upstream disconnect/work termination; an unaffected concurrent sibling request; retained backend-error status; and HTTP status/body/header passthrough. The previous observer must fail the cancellation regression and the replacement must pass. Unit tests alone do not substitute for live adapter gates.

## Prospective qualification and analysis

Repeat both adapter-effect and actual-driver routing canaries against the same verified server and predetermined non-holdout task fastapi_11194. All gates must pass before fresh holdout evaluations start.

The six qualified v0.2.4 runs prospectively supply trained-arm analysis inputs:
results/v024_a1_e4b_s1, results/v024_a1_e4b_s2, results/v024_a1_e4b_s3,
results/v024_a2_e4b_s1, results/v024_a2_e4b_s2, results/v024_a2_e4b_s3.

Use distinct v024 canary/log/evidence locations and refuse collisions. Preserve failures without overwrite or score-based replacement. Pin the reviewed runner/tests, transport-test evidence, amendment, original freeze, harness source, server identity and checkpoint/submission hashes in operational evidence.

Keep frozen A0, hypotheses, populations, seeds, budgets, compaction/cache and statistical methods unchanged. Do not modify A0 to use the defective transport merely to match it. Holdout task errors/timeouts remain unsolved; infrastructure/provenance failures prevent qualification.

A separate analysis implementation has been developed and tested only on synthetic data. It requires an explicit root seal selecting the qualified recovery generation and implementation/method hashes, plus all six completion receipts, before decoding any of the nine real outcome files. No automatic latest-run selection is permitted.

## Exposure and limits

Completion timestamps, non-holdout canary error fields and routing receipts were inspected for operational monitoring. No A0/A1/A2 holdout resolution outcomes were inspected when choosing this correction. Earlier artifacts remain sealed diagnostics. Transport tests demonstrate the specified cancellation/error behavior; they do not establish a performance gain or complete scorer equivalence.
