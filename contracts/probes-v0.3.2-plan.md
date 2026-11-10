# Serving-only probes v0.3.2

Prospective sensitivity experiments after the qualified v025 outcomes were known.
The frozen v0.2 primary analysis and all original adapters remain unchanged.

Run order: A1 alpha8, A2 alpha8, A1 temperature0.7, A0 temperature0.7.
Each run uses seed1 and all48 frozen tasks, with the original effective budget,
compaction, caching, prompts and tools. Report primary40 paired differences to
the corresponding qualified seed1 reference, and all48 descriptively. Alpha
comparisons use original A1/A2 seed1; temperature comparisons use A1/A0 seed1.
Differences and 20,000 whole-task bootstrap intervals (seed20261103) are
exploratory and do not identify a causal mechanism. Zero, negative and positive
differences are disjoint descriptive outcomes; no scale-versus-data decision rule.

Alpha8 changes only lora_alpha from16 to8 at r16, with identical weight bytes.
Fresh aliases are registered by controlled server restart. Each phase registers
exactly two adapters: scaled A1/A2, then original A1/A2. Fresh nonholdout BASE
and adapter canaries require correct aliases, real actions, drained requests and
zero infrastructure faults; only exact configured task-budget exhaustion is
accepted. Adapter effect must exceed repeated-base drift.

Task-local failures stay in the results. Infrastructure faults stop the chain;
there are no automatic retries or reused output paths. Original evidence and
failed diagnostics are preserved. Full BF16 probes and A1-clean are excluded.
M1/M6 must use the externally validated observable definitions; this protocol
does not claim completion_tokens is thought length or infer raw empty channels.

Activation requires the plan SHA and the exact current server PID/starttime.
No GPU operation occurs during preparation or the default runner inspection.
The last phase leaves a canonical original-adapter server running. Failures
preserve receipts and logs; they never silently restore/retry an evaluation.
