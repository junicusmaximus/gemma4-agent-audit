---
date: 2026-11-07
type: prereg
status: FROZEN 2026-11-07 15:00Z — momus-v068 REJECT on 95c6bf92 (2 blocking: incomplete-panel denominators/pairing → complete task set T_c rule; M1qs2 inclusion iff gates_pass at --pin, repeatable pin only before any external rollout) fixed as proposed + 3 non-blocking (design pin named; NOT_RUN/INCOMPLETE rows not eligible; frozen before the gate launch); frozen before the extctl-v060-gate-r3 unit is launched and before any external E4B rollout exists; no further edits (errata → finding note)
tags: [gemma4, paper-track, v068, external-panel, dev14, independent-repositories, m1q, seed2, a0mc]
related: [[finding-2026-11-07-m1q-mask-on-qat-base-v062]], [[prereg-2026-11-07-m1q-seed2-and-a0mc-branches-v067]], [[2026-11-07-m1q-confirmations-seed2-dev14]], [[advice-2026-11-07-oracle-after-m1q-repair]], [[2026-11-07-gemma4-hq1-done-v058-v060]]
---
# Pre-registration — v068: external-repository panel for the M1q observation (DEV14 eligible tasks)

## Question
Is the repair that M1q (masked recipe on the dequantized QAT base, matched template) showed on the Rich holdout (10/120 vs 0/120 for the unmasked control M0qmc) also observed on tasks from independent repositories, relative to the same serving form's base and unmasked control?

## Panel
- Design pin: contracts/external-controls-v060-APPROVED-r3.json (ef147ef8…; DEV14 = mpmath-854, kayak__pypika-267, python-semver-264, attrs-888, docformatter-313, more-itertools-892, wtforms-694, loguru-1299, pipdeptree-274, jinja-2033, apiflask-659, python-api-core-197, werkzeug-2997, traitlets-727). Tasks = the DEV14 tasks whose v060 ledger status is ELIGIBLE (gold-pass ∧ empty-fail under the v060 rules), in the frozen DEV14 order; n_e = their count. CONFIRM14 is never read and never run. The eligible list, the ledger digest, snapshot / router / dependency-tree hashes are filled by the runner's --pin step from the COMPLETED, frozen v060 ledger before any E4B rollout. Ledger rows with status NOT_RUN or INCOMPLETE_<stage> (e.g. left by a pace-rule pause) count as not eligible and are listed; the time box never waits for them. If n_e < 4 the panel is not run (sentence: "the external panel had fewer than 4 eligible tasks (n_e)").
- After this read, DEV14 is relabelled as a main-track development set (declared exposure); CONFIRM14 stays sealed.

## Arms (one LoRA-enabled vLLM session, matched template 855bbbbb, rank 16)
- A0mc-ext: the E4B QAT w4a16 base (requests to the base name on the LoRA-enabled engine);
- M0q-ext: adapter sft/runs/A1_M0q_v053 (4c8f1f49…);
- M1q-ext: adapter sft/runs/A1_M1q_v062 (44ab8710…);
- M1qs2-ext: adapter sft/runs/A1_M1q_s2_v067 — included if and only if sft/runs/A1_M1q_s2_v067/meta.json shows gates_pass true at the moment --pin runs; --pin records its adapter sha or the --without-arm M1qs2 flag; a --pin may be repeated only while no external rollout exists, and the final pin's arm set is the one reported.
- Budgets identical to the Rich evaluations (time, tool calls, turns, timeout, from the v062 PINNED contract); decoding seeds 1, 2, 3; per seed the arm order A0mc, M0q, M1q, M1qs2; one official harness invocation per task (agent patch not skipped), 4 in flight, each in its own capped service with the v043 memory guard.

## Canaries and stops
One strict canary per arm on the first eligible DEV14 task, seed 1, run before the panel; canary rows are reported apart and never enter the readings. A canary with zero parsed tool calls or any routing fault stops the whole generation (as v062). Exclusion rule as v055/v058 (runner qualification failure or an error outside the normal-outcome markers); excluded rows are listed. Stop rules: ≥ 2 excluded rows in one arm, or any server restart, or a cap of 4 h from server-ready → stop, completed rows only, listed.

## Declared differences from the Rich evaluations
(a) External task directories carry no code-graph / embedding data, so those tools and their prompt section are absent; (b) no memory-guard admission receipt exists for external gold patches (the v060 controls ran unguarded) — only the replay receipt is required; (c) A0mc-ext is served by the LoRA-enabled engine, so it is not the same serving configuration as v066 A0mc; (d) canaries run on a panel task and are reported apart; (e) the canary stop rule above. The panel is a different task distribution from Rich (several repositories, gold/empty controls from v060).

## Readouts (all rows reported; per arm pooled resolved / (3·n_e); paired by task with the three seeds grouped)
- Primary cells (exactly one applies; pre-written; z = A0mc-ext pooled, x = M1q-ext, y = M0q-ext):
  - z = 0: "on the external panel the base resolved nothing (0/3n_e), so the panel cannot inform the trained arms (M1q x, M0q y)";
  - z > 0 ∧ x = 0: "M1q's repair was not observed on the external panel (0/3n_e; base z/3n_e; unmasked control y/3n_e)";
  - z > 0 ∧ x > 0 ∧ y = 0: "M1q's repair was also observed on an n_e-task external panel (x/3n_e vs 0/3n_e for the unmasked control; base z/3n_e)";
  - z > 0 ∧ x > 0 ∧ y > 0: "both trained arms resolved external tasks (M1q x/3n_e, unmasked control y/3n_e; base z/3n_e)".
- Incomplete panels: if any row is excluded or the panel is stopped, the readings use the complete task set T_c = tasks with a completed row for every present arm at all three seeds (listed; n_c = |T_c|; no replacement); all denominators become 3·n_c, the cells read x/3n_c etc. and state "stopped/excluded, n_c of n_e tasks"; the bootstrap pairs over T_c; if n_c < 4 the sentence is "fewer than 4 complete tasks (n_c of n_e)". All rows, including incomplete ones, are still reported.
- Paired task-bootstrap CIs (20,000 resamples, seed 20261103), descriptive, for M1q − M0q and M1q − A0mc: lower > 0 → "more than", upper < 0 → "fewer than", otherwise "uncertain"; with small n_e these are expected to be wide.
- M1qs2-ext, if present, is read with the same cells (x replaced by its count) and reported beside M1q-ext, never pooled with it.
- Descriptive: loop, submit, M5, think rate per arm (as the Rich analyses), canary rows apart.

## Time box
Launch only after the v060 ledger is complete and frozen and the runner's --pin succeeded; GPU alone (no CPU controls beside it); latest launch 2026-11-09 12:00 UTC; reading (sealed report + Boss reproduction) by 2026-11-10 09:00 UTC or "not run" in the paper's limitations.

## Not claimed
A panel statement only — never a hidden-set estimate; no generalisation beyond these repositories; no mechanism; no pooling across seeds or arms; no "recovery" / "restores" / "causes" wording; with small n_e an absence of repair is not evidence of absence.

## Freeze points
This file is frozen before the extctl-v060-gate-r3 unit is launched (so no v060 ledger row has been read); its sha is pinned by the runner's --pin step and in the analysis contract before the panel launches; nothing here is edited after any external E4B rollout exists (errata → finding note).
