---
date: 2026-11-08
type: prereg
status: FROZEN 2026-11-08 07:4xZ — momus-v068 REJECT on 5a13e333 (2 blocking: complete-set rule vs the incomplete sentence; comparator identity leaves named and engine slots excluded) fixed as proposed + 3 non-blocking applied (projection never changes launch/stop; one declared freeze alternative; 9·n_e); frozen before the extctl-v060-gate-r3 unit was launched (unit not-found at freeze) and before any v068 or v070 external E4B rollout exists; no further edits (errata → finding note)
tags: [gemma4, paper-track, v070, external-panel, dev14, k9, a_onpol, t_sub, contrast]
related: [[prereg-2026-11-07-external-panel-m1q-v068]], [[prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2]], [[prereg-2026-11-08-onpolicy-confirmation-readings-v069]], [[2026-11-08-paper-headline-k9-v070]], [[finding-2026-11-08-onpolicy-confirmation-v064c-v069]]
---
# Pre-registration — v070: external-repository panel for the K9 termination contrast (A_onpol seeds 1–2 and T_sub, DEV14 eligible tasks)

## Question
K9 rests on the known Rich holdout. The pre-written v064-r2 cell CONTRAST (one T_sub seed) and v069 T1 (two A_onpol seeds) both concern termination. Does the same termination pattern appear on tasks from independent repositories? The pattern: A_onpol seeds 1 and 2 do not meet the collapse criteria; T_sub does.

## Panel (identical to v068)
- **Design pin:** contracts/external-controls-v060-APPROVED-r3.json (ef147ef8…).
- **Tasks:**
  - the DEV14 tasks whose v060 ledger status is ELIGIBLE, in the frozen DEV14 order; n_e is their count;
  - rows NOT_RUN or INCOMPLETE_<stage> are not eligible;
  - CONFIRM14 is never read or run.
- **Pin:** the runner's --pin records the ledger digest and eligible list. They must equal the values v068's --pin recorded; any difference stops v070 before any rollout.
- **Threshold:** if n_e < 4, the panel is not run ("fewer than 4 eligible tasks (n_e)").
- **Overlap:** --pin also records each arm's training-repository overlap with the panel repositories. It is declared, not a stop rule.

## Arms
One LoRA-enabled vLLM session, identical to v068's engine configuration: serve script, matched template 855bbbbb, rank 16, base gemma-4-e4b-it-qat-w4a16-ct.
- A_onpol-s1-ext: sft/runs/A1_onpol_v064, adapter ad0910f5….
- A_onpol-s2-ext: sft/runs/A1_onpol_s2_v064, adapter f9e06642….
- T_sub-ext: sft/runs/A1_tsub_v064, adapter 7122a9d6….
- Budgets identical to v068 (from the v062 PINNED contract): time, tool calls, turns, timeout.
- Decoding seeds 1, 2, 3. Per seed the arm order is A_onpol-s1, A_onpol-s2, T_sub.
- One official harness invocation per task (agent patch not skipped), 4 in flight, each in its own capped service with the v043 memory guard.
- Training seeds and decoding seeds are kept distinct in every table.

## Canaries, exclusions, stops (as v068 and decision 97e9f6ee)
- **Canaries:** one strict canary per arm on the first eligible task, seed 1, reported apart, never in a reading.
  - A canary stops the generation only on zero parsed tool calls, a routing fault or an unqualifiable row.
  - Normal-marker endings pass, including budget exhaustion.
- **Exclusion:** a runner qualification failure, or an error outside the normal-outcome markers. Excluded rows are listed.
- **Stops:** ≥ 2 excluded rows in one arm, any server restart, or **3 h from server-ready** (user cap). On a stop, completed rows only, listed.
- **Feasibility:** after the canaries, the runner logs a projection of the full panel's wall time from startup and canary throughput. It is descriptive and never changes the launch or stop decision; the 3 h cap is not a completion guarantee.

## Readings (all rows always reported)
T_c = tasks with a completed row for every v070 arm at all three seeds; n_c = |T_c|; no replacement. **The planned paired set is complete iff n_c = n_e and no row was excluded. Only then does an E-cell apply as a reading.** Otherwise (a stop, an exclusion, or no sealed report by 11-10 09:00Z) the result is "incomplete (m of 9·n_e rollouts; n_c of n_e complete tasks)"; the T_c labels, loop / submit and repair are shown descriptively, and no cell sentence is a claim.

**Primary — termination labels per arm.** Pooled over 3·n_c rollouts, with the frozen v064-r2 thresholds: NON-COLLAPSED = loop ≤ 0.30 ∧ submit ≥ 0.80; COLLAPSED = loop > 0.40 ∨ submit < 0.70; otherwise INTERMEDIATE. Loop, submit and M5 use the producer mechanism_metrics_v032 (36042f6d…).

Exactly one cell applies:
- **E1** (A_onpol s1 NC ∧ s2 NC ∧ T_sub C): "the termination contrast was also observed on an n_c-task external panel (A_onpol training seeds 1 and 2: loop a1 and a2, submit b1 and b2; T_sub, one training seed: loop c, submit d)".
- **E2** (both A_onpol seeds NC ∧ T_sub not C): "no external contrast: the teacher subset did not meet the collapse criteria on the external panel (loop c, submit d; label L), while both A_onpol training seeds did not either (loop a1 and a2, submit b1 and b2)".
- **E3** (any A_onpol seed not NC): "the self-generated arm's non-collapse was not observed externally in training seed k (loop a, submit b; label L)", one clause per such seed, with T_sub's label and loop/submit beside it.

E1 is a statement about termination behaviour only. It never implies repair transfer.

**Repair — always reported, never a reading.**
- Per arm: pooled resolved / 3·n_c and per seed, including 0.
- Beside it, v068's A0mc-ext on T_c, for the same tasks and decoding seeds, with a descriptive paired task-bootstrap (20,000 draws, seed 20261103). It is shown only if the identity check holds. The check compares these leaves of contracts/external-e4b-v068-r21-runtime-PINNED.json (sha recorded at v070 --pin) with v070's: pin.ledger_digest + eligible list and order, decoding seeds, evaluation budgets, chat_template sha 855bbbbb, server_script_sha256, vLLM / swegemma / harness versions, base model id, LoRA rank. --max-loras and --lora-modules (adapter slots and names) and the engine session are excluded from the check and listed as declared difference (j). The comparator uses the tasks in T_c that also have a completed v068 A0mc-ext row at all three seeds (listed). The check result is written by --pin before any v070 rollout.
- If any compared leaf differs, no comparator is shown and the differing leaves are listed.
- No CI wording rule is applied; zeros are not evidence of absence.
- Descriptive per arm: loop, submit, M5 and think; canary rows apart.

## Time box and order (user decision 2026-11-08)
- **Order:** v068 → the protected main-track 31B window (2 h target, 4 h max) → v070. GPU alone, with no CPU job beside it.
- **Launch:** only after the v060 ledger is complete and frozen, v070's --pin has succeeded, and v068's run has ended.
- **Freeze:** if the planned paired set is not complete by 2026-11-10 09:00 UTC (sealed report and Boss reproduction), the paper reports "the external panel for the contrast was not completed before the result freeze (m of 9·n_e rollouts)". There is no generalisation from a partial, and incomplete records are kept.

## Declared differences
- v068's (a)–(e): no code-graph/embedding tools; no admission receipt for external gold; LoRA-enabled engine; canaries on a panel task; canary stop rule.
- (f) A separate engine session from v068, so the comparator is cross-session.
- (g) Training data are fastapi only (A_onpol) or 18 kit repositories (T_sub); the repository overlap is declared.
- (h) T_sub has one training seed.
- (i) The panel is a different task distribution from Rich.
- (j) Adapter slots and names (--max-loras, --lora-modules) and the engine session differ from v068 by construction.

## Not claimed
- This is a panel statement only.
- No hidden-set estimate, no generalisation beyond these repositories, no mechanism.
- No pooling across seeds or arms.
- None of these words: "prevents collapse", "on-policy learning", "improved", "equivalent", "the contrast replicated", "causes".
- No attribution to a single member of the bundle.
- With small n_c, an absence of contrast or repair is not evidence of absence.

## Freeze points
- Frozen after momus review.
- Frozen before any v068 or v070 external E4B rollout exists, and before any v068 model row is read.
- Frozen before the extctl-v060-gate-r3 unit was launched (no v060 ledger row existed at the freeze).
- Its sha is pinned in the v070 runtime and analysis contracts.
