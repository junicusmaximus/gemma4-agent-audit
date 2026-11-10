---
date: 2026-11-03
type: experiment-contract
version: "0.2"
status: frozen prospective analysis amendment
---
# E4B distillation experiment contract v0.2

## Authority, timing and scope
This document replaces the conflicting analysis provisions of v0.1 and Addenda A-E. It is a prospective analysis amendment, NOT a registration before training. Recorded at 2026-11-03 10:46:18 UTC. A1 training began at approximately 2026-11-03 10:16:49 UTC; A0 had already run. Before this amendment, the reviewing Codex session saw data-selection statistics, gold/empty controls, dry-run throughput/loss, public community evidence, and the user-reported corrected development-panel results. It has NOT opened A0/A1/A2 outcome files. Other collaborators' exposure is not independently established and must be disclosed. No claim of complete team blinding is made.

The original overnight shell was held while its existing A1 trainer continued because training/probe failures did not gate evaluation. A replacement controller must enforce the conditions below. Execution permission follows the user's existing experiment authorization; this is Codex's technical review approval, not a claim that the user authored this contract.

## Question and arms
Primary question: does this implemented distillation pipeline change E4B's verified local repair rate under a fixed inference budget? A1 versus A0 cannot isolate a formatting mechanism or establish broad repository generalization.
- A0: supported E4B w4a16 base, no adapter.
- A1: existing arm_ordinary_tokmatched, 726 selected trajectories/tasks, approximately 19.32M input tokens and 4.39M loss tokens.
- A2: existing arm_short, 764 selected trajectories/tasks, approximately 19.30M input tokens and 4.43M loss tokens.
- Preserve the in-flight recipe: unquantized E4B, rank16, alpha16, learning rate0.0001, one epoch, sequence limit16384, bf16 and gradient checkpointing, assistant-only targets.
- Exact membership and training implementation are fixed by hashes in the machine-readable freeze manifest. No new training data or arm substitutions enter this version.
- A2 versus A1 is EXPLORATORY: coverage, teacher/tier mixture, attempt rank and loss-token count differ. It is not a clean causal test of trajectory efficiency. The small observed selection gap does not establish that its effect will be null.
- Disclose that about52% of sequences/61% of loss tokens use continuation chunks with truncated earlier history. Do not describe all training examples as context-faithful without a separate audit.
- One training run per trained arm; three decoding seeds do not establish training-seed robustness. Thoughts-in-targets and further data-selection variants require a separate prospective amendment.

## Fixed local evaluation configuration
Same existing prompt and seed-specific configuration in all arms; only adapter assignment varies. Decoding seeds1,2,3; temperature0.2, top_p0.95, max_output_tokens8192, thinking_budget4096 and retained thoughts.
Effective CLI budgets override the stored eight-minute YAML: max_time_minutes4, max_tool_calls28, max_turns80, timeout_seconds240, verification timeout300, subprocess sandbox, concurrency4.
Compaction: interval15, overlap2, threshold14336, retention5. Context cache: min_tokens2048, ttl_seconds1800, cache_intervals10. Freeze run_eval.py, serving implementation, model registry, trainer, control CSVs, split, data and submission files.
These are local single-5090 measurements. They do not demonstrate equivalence to the sequential four-L4 scorer or a safe full-run leaderboard budget.

## Eligibility fixed from controls before trained-arm outcomes
Primary population: the following40 Rich tasks with gold_resolved=True and empty_resolved=False in the hashed controls CSV:
rich_2725, rich_2943, rich_3006, rich_3043, rich_3052, rich_3061, rich_3063, rich_3067, rich_3105, rich_3180, rich_3278, rich_3454, rich_3468, rich_3470, rich_3471, rich_3472, rich_3480, rich_3506, rich_3518, rich_3521, rich_3535, rich_3675, rich_3676, rich_3718, rich_3772, rich_3777, rich_3882, rich_3894, rich_3905, rich_3930, rich_3934, rich_3935, rich_3938, rich_3942, rich_3944, rich_3953, rich_4006, rich_4075, rich_4076, rich_4077.

All48 original Rich tasks remain evaluated and reported as a secondary population. The eight primary exclusions are rich_3064, rich_3130, rich_3296, rich_3469, rich_3486, rich_3782, rich_4070, rich_4079. Seven were reported as dependency/colour mismatches; rich_3486 failed a required-test-pass condition. These diagnoses remain subject to inspection, but membership is fixed for this version.

The primary40 include three empty-run timeouts (rich_3480,rich_3772,rich_4006) and one empty collection-exit2 (rich_3930). Call this population CONTROL-QUALIFIED, not proof that every failure is an assertion-level bug. Report a predeclared sensitivity analysis on the other36 assertion-failure controls. Gold fixes may legitimately resolve hangs or import failures.
Development controls yield58 qualified tasks among81; seven empty-pass tasks cannot count as local repaired bugs. Compute membership from the joint gold/empty predicate, not subtraction without overlap verification.
No post-outcome silent reinclusion or exclusions. Dependency repairs or new control results create a new version with separately reported results.

## Primary estimand and analysis
Let y[a,i,s] be the harness resolution indicator for arm a, task i and decoding seed s. The primary effect is the mean over40 tasks of (mean_s y[A1,i,s] - mean_s y[A0,i,s]); report percentage points.
Report each seed and every task, not just the best run. Use20000 paired task-cluster bootstrap draws (analysis seed20261103) for a95% interval, carrying each task's complete three-seed arm vector together. This quantifies uncertainty conditional on this public repository; it does not model across-repository variation.
A positive point estimate alone, or wins in two of three seeds, is not sufficient for a superiority claim. Treat a95% interval wholly above zero as supportive local evidence only, with protocol validity and sensitivity results disclosed. If the interval spans zero, state inconclusive rather than proving no effect.
Majority-of-seeds McNemar/sign-test summaries and union-of-seeds solved sets are secondary/descriptive. The union is a multiple-attempt result, not single-run performance.
Secondary/exploratory: all48-task rates; the36-task control sensitivity; A2-A1 and A2-A0; tool calls, generated tokens, context errors, nonempty patch rate, time-to-first-edit and timing where actually logged. No multiplicity-free confirmatory claims for these.
Agent failures/timeouts count unresolved. Missing whole-run outputs or failed routing invalidate that run; preserve all artifacts and do not silently treat an incomplete run as a complete zero score. Any infrastructure retry must be labeled, use the same hashes/configuration, and retain its predecessor.

## Mandatory gates before trained-arm Rich evaluation
1. Both trainers must exit successfully; fresh checkpoint metadata and tensor bounds must match the intended arm, base and recipe. File existence alone is insufficient.
2. The hash-pinned contract and all listed immutable inputs must match before each stage. A changed manifest plus changed sidecar is not authorization to continue.
3. Repeated base probes establish numerical drift; each trained adapter must demonstrate a reproducible nonzero effect beyond that drift on fixed non-holdout prompts.
4. A non-Rich driver canary must exercise the actual run_eval.py path and produce successful, valid backend requests routed to the intended adapter. A recorded request name or a CLI exit0 alone is insufficient; transport/route failures must stop the chain.
5. The exact48 task IDs, three seeds, configuration and frozen40-task analysis population must be preserved. New outputs cannot overwrite existing outputs.
6. Keep trained-arm outcome logs sealed during gate validation; capture results without printing resolution summaries.

## Storage and records
Canonical destination: PROJECT_ROOT/. New work runs under its gemma4/ subdirectory; documents/evidence under briefing/. Preserve the original active trainer until completion; copy and verify its artifacts before retiring old paths. The user authorized deleting old copies only after successful verified transfer.
Machine-readable contract: gemma4/contracts/contract-v0.2.freeze.json and SHA256 sidecar. Copies of this text in briefing/research/gemma-4-2026/ and gemma4/contracts/ must match. Save adapter hashes, environment provenance, stage exit codes, gate evidence and submission IDs. No credentials in records.
