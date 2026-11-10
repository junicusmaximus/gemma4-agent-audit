---
date: 2026-11-06
type: decision
tags: [gemma4, paper-track, pre-registration, matched-context, m0q, v055, no-training]
related: [[prereg-2026-11-06-matched-context-eval]], [[finding-2026-11-06-m0q-eval-v054]], [[prereg-2026-11-06-qat-base-lora-fixture-and-M0q]], [[finding-2026-11-06-train-serve-serialization-audit]]
---
# Pre-registration: matched-context evaluation of M0q (v055) — weight-matched adapter, serving form matched; no training

Author: Boss, 2026-11-06 01:3x UTC, after reading v054 and before any v051 reading (v051 running; no v051 outcome read). Frozen at this file's sha256 (session note + 151 copy). Executor: Boss via an executor agent. Generation name **v055**. Nothing existing is modified; new names only.

## Why
v054 showed that the weight-matched adapter M0q starts like the teacher (find opener +78 pp, no token corruption) while resolution does not differ from the sealed M0 (0/120 both) and loop / submit differ in the other direction (+17.5 / −15.0 pp). v051 tests the serving-form channel on the sealed M0 (bf16-release-trained). v055 closes the grid cell "weights matched AND serving form matched": M0q served with the pinned matched template.

## Design (one change only)
- Arms: **v055_M0qmc_s{1,2,3}** = the M0q adapter (`sft/runs/A1_M0q_v053`, adapter_model 4c8f1f49…, meta 35259bc8…) served exactly as in v054 (QAT w4a16 base, LoRA-on server with the three M0q aliases, T 0.2, seeds 1/2/3, v043 budgets and memory guard, 48-task holdout, canaries) **except** `--chat-template` = `analysis/matched-context-boss-v049/chat_template.matched.jinja` sha 855bbbbbef10d51177c5d7eca19a3f3d39a11d9a80c142a9f19784029515a437 (the v051 template; nothing else).
- Comparator: **v054 M0q** (`results/v054_M0q_s{1,2,3}`), paired by task × seed — same adapter, environment, seeds; the only intended difference is the serving template. Descriptive rows beside it: v051 M0mc, sealed v043 M0, A0-capped v052.
- Launch byte check (stop rule, as v051): POST `/tokenize` for calls 1, 3 and 7 of the audited trace must equal the offline matched render; the server log must report the `openai` content format; any mismatch → `stopped.json`, no retry. Admission, canary, server-identity and cleanup failures stop likewise.
- Metrics: the pinned producers (`tools/mechanism_metrics_v032.py` 36042f6d…, `tools/analyze_mechanism_v031.py` 821f2294…) on primary40 (c7125743…), clean36, all48; v042 statistics; slots frozen before reading: slot M0 = v054 M0q, slot M1 = v055 M0qmc, effect M0qmc − M0q.

## Pre-declared readings (primary40 three-seed means; every region labelled)
- **MCq-i — the serving form accounts for most of the weight-matched adapter's collapse:** loop ≤ 0.30 AND submit ≥ 0.80 AND M5 repeat-after-error ≤ 0.20 (v054 M0q: 0.792 / 0.450 / 0.492).
- **MCq-ii — the serving form is behaviourally inert for M0q:** |Δ loop| ≤ 0.10 AND |Δ submit| ≤ 0.10 AND |Δ M5-error| ≤ 0.10.
- Otherwise intermediate: paired 95 % task-bootstrap intervals, no label.
- Resolution not predicted; pooled resolved(M0qmc) ≥ 4 / 120 reported as notable and bounded, never as recovery. Think-rate ≈ 0 by construction (reported). Find / repro-script openers, max repeat, M5 silent/normal, responses with no `<channel|>`, compaction counts (beside v054's) — descriptive.
- Limits: one adapter, one training seed, known holdout; MCq-i attributes behaviour to the serving form of THIS pipeline; it does not establish that matched serving makes the SFT beneficial; the matched template is a serving choice, not "the correct" form.

## Cost, order, files
≈ 2.3–2.6 h GPU (3 canaries + 3 runs). Order: after v051; before the H1 evaluation if H1 is approved (H1 training may precede or follow v055 — the GPU order is the user's decision, recorded in the session note). Runner `tools/run_matched_context_v055.py` (copy of the v051 runner: M0q aliases, results `results/v055_M0qmc_s{1,2,3}`, receipts `migration/matched-context-v055/`, contract `contracts/matched-context-v055-runtime-DRAFT.json` pinning template, adapter, runner diff and server-script diff); analysis `tools/analyze_matched_context_v055.py` (v042 statistics, slots as above), output `analysis/matched-context-v055/`. Leases, single-executor rule and the guarded-argv rule as for v051/v054.

## Exposure
Boss has read v054's readings and the v051 activation receipts (no v051 outcome). Thresholds copied from the v051 pre-registration (exploration-informed). Nothing here is confirmatory.
