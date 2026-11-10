---
date: 2026-11-05
type: decision
tags: [gemma4, paper-track, pre-registration, probes, capped-environment, mask-control]
related: [[finding-2026-11-05-mask-control-readout]], [[prereg-2026-11-05-mask-control-mechanism-predictions]]
---
# Pre-registered predictions for two no-training evaluations in the capped v043 environment

Author: Boss, 2026-11-05 ≈ 14:35 UTC (23:35 KST), before any such run exists (no `results/*A0cap*` or `results/*nothink*` directory, no contract). Frozen at this file's sha256, recorded in the session note and on the 151 copy. Execution, contract and gates are Codex's lane; the predictions bind to the design intent below, and any deviation from it re-freezes this note before the run. Nothing here is confirmatory.

## Design intent

- **Probe A — A0-capped.** The untuned base configuration exactly as the sealed A0 runs (`subs/e4b_budgetfit_s{1,2,3}`: temperature 0.2, decoding seeds 1/2/3, the same 48-task holdout, budgets 4 min / 28 calls / 80 turns, same prompts and tools), executed in the v043 memory-capped environment (per-sandbox 4 GiB, swap 0, guard sha 5ad9d40d…, same routing of agent / grading / setup commands). Metrics: resolution and the pinned v032 mechanism metrics on primary40 (ids c7125743…).
- **Probe B — M1-no-think.** The M1 adapter (`sft/runs/A1_M1_v035`, sha 34813c7a…) exactly as `v043_M1_s{1,2,3}`, except that the model's context at every action matches the training context: the thought channel is forced empty (`<|channel>thought` + newline + `<channel|>` prefilled, or thinking disabled so that the model itself emits the empty block — Codex chooses and records the implementation), and the training header's thinking-enable token is matched whichever way the training header had it (checked by the teacher-statistics task). Seeds 1/2/3.
- Readings use primary40 three-seed means from the pinned producer; descriptive, no new bootstrap required (if one is run, the unchanged v042 functions apply). all48 descriptive.

## Predictions (thresholds fixed now; every region has a label)

**Probe A.** Expected: pooled resolved ≈ the historical 7/120 (3/2/2), think ≥ 0.80, loop ≤ 0.10, submit ≥ 0.90.
- A-i *environment inert*: pooled resolved ≥ 4/120 AND loop ≤ 0.10 → the cap does not drive the zero outcome of M0/M1; A0-capped vs M0 becomes a within-environment, task-paired descriptive contrast and the paper's final-environment story is complete.
- A-ii *environment suppressive*: pooled resolved ≤ 1/120 → the capped environment itself suppresses resolution; the resolution endpoint of M0/M1 is uninformative and the paper rests on the behavioural rates.
- Otherwise intermediate: reported descriptively, no reading.

**Probe B.** Think-rate ≈ 0 by construction (not a finding). Informative metrics: repro-script first action, submit rate, loop rate, M5 repeat-after-error.
- B-i *context mismatch*: repro-script-first ≥ 0.45 AND submit ≥ 0.45 (M0: 0.65 / 0.60) → the M1 shift (−0.61 / −0.47) was the train/inference context mismatch; the masked targets' direct effect on the action policy is small; M1's extra deterioration is not evidence about masking per se.
- B-ii *policy change*: repro-script-first ≤ 0.20 AND submit ≤ 0.25 (M1: 0.04 / 0.13) → masking itself changed the action policy; the empty-thought positions carried the opener and submit behaviour.
- Otherwise intermediate: no reading. Resolution is not predicted (expected 0/120; ≥ 3/120 would be reported as unexpected). Loop and M5 are reported descriptively beside M0 and M1.
- Boss's prior, stated for the record and not a reading: B-i ≈ 60 %.

## Rules and exposure

Primary40 ids only; all48 descriptive. One training seed; known holdout; the historical A0 runs were uncapped, so Probe A vs historical A0 is descriptive. Boss had read the full M0/M1 readout and the sealed A0/A1 mechanism summaries before writing this; the thresholds are exploration-informed and prospective only for these two probes.

## Re-freeze 2026-11-05 16:1x UTC (before any Probe A run; original sha 14351776… superseded by this file's new sha)

- Executor: Boss (user decision 2026-11-06 00:5x KST, see [[2026-11-06-gpu-execution-owner-claude]]), not Codex.
- Server configuration for Probe A: **LoRA OFF**, exactly as the sealed A0 runs (`e4b_batch.sh:6`, `logs/vllm_server.log` enable_lora=False) — the only intended difference from the sealed A0 generation is the v043 memory policy (per-sandbox 4 GiB, swap 0, 48 GiB service, same guard shas). The M0/M1 generation used a LoRA-on server; whether LoRA-on and LoRA-off servers give interchangeable base outputs is untested, so the A0cap-vs-M0 within-environment contrast carries this declared server-flag difference. A single-seed LoRa-on A0 sensitivity may be added later as a separate, labelled run.
- Implementation: `tools/run_a0_capped_v048.py` (modified copy of the v043 runner; three runs `v048_A0cap_s{1,2,3}`, arm A0cap, model BASE, baselines `subs/e4b_budgetfit_s{1,2,3}`, seeds 1/2/3, T 0.2, no adapter; guard, `serve_vllm.py`, `run_eval.py`, budgets 4 min / 28 calls / 80 turns / concurrency 4, readout 6319262b and its 48 ids byte-identical), `tools/analyze_a0_capped_v048.py` (reuses the pinned v042 statistics by sha 23915fb0 and the producers 36042f6d / 821f2294 unchanged; M0 in one readout slot, A0cap in the other, slot and sign frozen before reading; populations primary40 / clean36 / all48). Receipts under `migration/a0-capped-v048/evaluation/`; results `results/v048_*`; contract `contracts/a0-capped-v048-runtime-DRAFT.json` + sha.
- Probe B (M1-no-think) is deprioritized (Codex review §6): it changes instruction/template/generation policy, not only budget; it runs, if at all, after the matched-context re-evaluation.
- Readings A-i / A-ii / intermediate unchanged. Stop rule: any canary, admission, server-identity or cleanup failure stops the generation with `stopped.json`, no automatic retry; no task reselection after seeing outcomes.

## Generation note 2026-11-05 16:5x UTC — v048 stopped, rerun as v052 (design unchanged)

Probe A generation v048 was activated 16:39:21 UTC (LoRA off, runtime R 8b62e5c2…, analysis contract A b41753cc… frozen pre-activation), qualified 3/3 canaries (68/61/83 s; server ready 56 s; 0 bounded OOMs; outer memory events 0) and STOPPED at 16:44:20 UTC before the first full run: the per-case idle check (`recover_eval_v025.assert_idle`, "Project trainer/evaluator remains active") matched the argv of another Boss agent's provenance-hashing command (`sha256sum …/run_eval.py …/sft/train_lora.py`), not a real evaluator. No retry by design; cleanup clean (server group empty, GPU 18 MiB, port 8000 free, locks released); no run started, no outcome read; v048 names consumed, receipts kept as evidence. The rerun is generation **v052** (`v052_A0cap_s{1,2,3}`, contracts `a0-capped-v052-*`, receipts `migration/a0-capped-v052/`), identical design, readings and stop rule; new coordination rule: during a generation no other agent may run a shell command naming `run_eval.py` or `train_lora.py` under the root (hash pinned files with python instead); activation only after Boss confirms the probe process tree is gone and the v051 preparer has finished hashing.
