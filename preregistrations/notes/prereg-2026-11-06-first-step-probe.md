---
date: 2026-11-06
type: decision
tags: [gemma4, paper-track, pre-registration, probe, train-serve-interface, first-step]
related: [[finding-2026-11-05-mask-control-readout]], [[prereg-2026-11-05-capped-A0-and-M1-nothink-probes]], [[proposal-2026-11-05-history-arm-A1-hist]]
---
# Pre-registered first-step decomposed probe (teacher-forced, no generation, no training)

Author: Boss, 2026-11-06 00:2x KST (2026-11-05 15:2x UTC), before any such probe exists. Frozen at this file's sha256 (session note + 151 copy). Design = Boss (data / experimental design lane); execution of the GPU steps = Codex (server execution lane) under its gates; the CPU step is part of the running serialization audit (`analysis/serialization-audit-boss-v046/`). Any deviation from the design intent re-freezes this note before the run.

## Why (observed facts)
- Teacher table (`analysis/teacher-behaviour-boss-v045/`, 726 training trajectories): first action is the `find /workspace -maxdepth …` opener in 94.9 %; loop 0.7 %; submit_patch 100 %; repeat-after-error 0.2 %.
- Student M0 (same data, fresh control, `analysis/mask-control-v043/`): first action `find` 0.0 % (repro-script 65 %, no tool call 18 %); loop 62 %; submit 60 %; repeat-after-error 49 %. Historical A1: `find` 0.0 % as well; A2 emitted a corrupted `find … -max-depth` opener in 60–83 % of tasks in two seeds.
- The first prompt is complete in training and at inference, so packing orphans and history cannot explain a step-1 failure. Something corrupts the policy before any loop can form: the LoRA never learned the opener (training defect), the served first prompt differs from the trained form (prompt-format mismatch), or the served stack (QAT w4a16 + LoRA) does not reproduce the trained distribution (serving artifact). This probe separates the three with teacher-forced log-probabilities only.

## Design intent
Target completion T = the teacher's modal exact first action, serialized as the model would emit it: `<|channel>thought` + newline + `<channel|>` + `<|tool_call>call:run_command{command:<|"|>find /workspace -maxdepth … <|"|>}<tool_call|>` (the modal exact string is computed from the 726 trajectories by the CPU step; if several variants exist, the most frequent one is used for every prompt and the variant list is recorded).
- **(a) training form, GPU, HF bf16 base `models/gemma-4-e4b-it` + M0 LoRA via PEFT (`sft/runs/A1_M0_v035`), and base alone:** three training trajectories whose first action is the opener; prompt = the stored header segment (`text_segments[0]`), completion = the stored first completion (ground truth, saved token ids). This is the training loss at those positions.
- **(b) evaluation form, same stack:** the 48 holdout first prompts rendered exactly as the harness renders them (the serialization audit's reconstruction of the step-1 request through the served chat template), completion = T.
- **(c) served stack, vLLM QAT w4a16 `models/gemma-4-e4b-it-qat-w4a16-ct` + M0 LoRA through the evaluation serving path, `prompt_logprobs` on prompt+T for the same 48 prompts; and base without adapter (c0).** Record the LoRA modules vLLM reports as loaded against `adapter_config.json` target_modules.
- Statistics per condition: m = mean per-token log-probability of the completion tokens; acc = share of completion tokens whose target is the argmax (teacher-forced accuracy); both also for the empty-thought close token and for the first ten tool-call tokens separately. No generation, no sampling, no task execution, no resolution outcome.

## Readings (fixed now; every region has a label)
- **R1 learned (training form):** m_a(M0) ≥ −0.5 AND acc_a(M0) ≥ 0.90 AND m_a(M0) − m_a(base) ≥ +1.0 → the adapter learned the opener in its training form. Otherwise → *training-defect branch* (loss mask, packer or first-turn supervision); H1 is moot until the defect is understood.
- **R2 prompt format (eval form vs training form, same stack):** m_b(M0) ≥ m_a(M0) − 1.0 → consistent. m_b(M0) < m_a(M0) − 2.0 → *prompt-format-mismatch branch* (the harness's first prompt differs materially from the trained header). Between → intermediate, reported, no reading.
- **R3 stack (served vs HF, same prompts):** |m_c(M0) − m_b(M0)| ≤ 0.5 → consistent. m_c(M0) < m_b(M0) − 1.0 → *serving-artifact branch*. Between → intermediate. Loaded modules ≠ target_modules → *partial-adapter branch* regardless of m.
- **R4 adapter active in serving:** m_c(M0) − m_c0(base) ≥ +1.0 on the opener → active. Otherwise → the served adapter does not change the opener distribution; report with R3.
- **Clean** = R1 learned ∧ R2 consistent ∧ R3 consistent ∧ R4 active. Only a clean probe leaves the step-1 anomaly unexplained by the interface, in which case it is reported as "learned and served, yet not emitted at T 0.2" and the generation-side check (first-step sampling under the served stack with logged top-k) is the next probe.

## Decision rule bound to this probe (H1 gate, from the oracle re-consult 11-05 15:1x UTC)
- Clean → the user may authorize H1 candidate-file generation through the proper path (Codex retries with the blocked write narrowed, or the user explicitly reassigns; never a bypass of the blocked write); H1 vs M0 then proceeds with its own pre-registration.
- Any non-clean branch → H1 is deferred; the replacement cell is a matched-path re-evaluation (M0 first, then A1 / M1) with no training, under the environment anchor Probe A.
- Ordering on the single GPU: this probe (≈ 1 h) → Probe A (A0-capped × 3) → branch → Probe B (M1-no-think) last.

## Cost and exposure
HF bf16 E4B + PEFT teacher-forcing on 51 sequences ≈ 10 min; vLLM start + 96 `prompt_logprobs` requests ≈ 15 min; total ≤ 1 h GPU. Boss has read M0/M1 and the teacher table; thresholds are exploration-informed and prospective for this probe only. Nothing here is confirmatory.
