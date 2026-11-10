---
date: 2026-11-08
type: prereg
status: FROZEN 2026-11-08 ≈ 02:0xZ — momus-v069 OKAY on 55043c7c (no blocking; one non-blocking wording note applied: "seed 1's interval"); written after the A_onpol (v064a) reading and before any confirmation training or evaluation exists; frozen before the confirmation training starts; its sha is pinned in the confirmation training contract and the v064c analysis contract; no further edits (errata → finding note)
tags: [gemma4, paper-track, v069, v064, on-policy, confirmation, seed2]
related: [[prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2]], [[prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065]], [[finding-2026-11-08-onpolicy-a-v064a]], [[2026-11-07-gemma4-hq1-done-v058-v060]]
---
# Pre-registration — v069: pre-written readings for the A_onpol confirmation (LoRA init seed 20261104)

## Context
A_onpol (v064a; finding 239b2c86) met the v064-r2 confirmation trigger: NON-COLLAPSED (loop 0.008, submit 0.958) with 10/120 on primary40 (≥ 4/120); A_onpol − A0mc −1.67 pp [−6.67, +2.50] ("uncertain relative to the base"). The frozen addendum (b602a5d3 §2) defines the confirmation: the A_onpol recipe with LoRA initialization seed 20261104, everything else identical (data, update groups and order, schedule, U = 43, gates), the init hash recorded instead of 9dfc7ccc, G-eq not applied; evaluated identically (48 × 3, matched template, comparator A0mc); both seeds reported, the original never replaced; "no-collapse" prose needs the confirmation to be NON-COLLAPSED; "improved" prose needs the confirmation's arm − A0mc CI lower bound > 0. This file adds only pre-written sentences; no rule of v064-r2 or the addendum changes.

## Arm and evaluation
- Training: the confirmation wrapper and contract (DRAFT → APPROVED beside) pin this file's sha; run dir sft/runs/A1_onpol_s2_v064; fresh init; no resume.
- Evaluation: generation v064c (results/v064_A_onpol_c2_s{1,2,3}), the v064a machinery re-pinned to the confirmation adapter; analyzer slots arm → M1, the sealed A0mc (v066) → M0; A_onpol seed 1 printed beside it from its sealed report, never pooled.

## Readings (primary40 three-seed means; all48 and clean36 as sensitivities; v064-r2 thresholds)
- Termination label (v064-r2): NON-COLLAPSED = loop ≤ 0.30 ∧ submit ≥ 0.80; COLLAPSED = loop > 0.40 ∨ submit < 0.70; otherwise INTERMEDIATE.
- Repair: pooled resolved /120; ≥ 4/120 = the minimum repair observation only.
- Exactly one termination sentence applies:
  - T1 (confirmation NON-COLLAPSED): "training on the base's own successful trajectories did not meet the pre-defined collapse criteria in two training seeds (loop a1 and a2, submit b1 and b2; resolved 10/120 and x/120) on the known holdout".
  - T2 (confirmation INTERMEDIATE): "the second training seed was intermediate (loop a2, submit b2; resolved x/120); no-collapse is not stated".
  - T3 (confirmation COLLAPSED): "seed-dependent: the second training seed met the collapse criteria (loop a2, submit b2; resolved x/120)".
- Exactly one interval sentence applies to the confirmation − A0mc paired task-bootstrap CI (20,000 draws, seed 20261103), as v064-r2 words it, with "(x/120 vs 12/120; d pp [l, u])" appended: lower bound > 0 → "improved over the matched-template base"; upper bound < 0 → "worse than the base"; otherwise → "uncertain relative to the base".
- "Improved" prose about the recipe is allowed only when the confirmation's lower bound > 0, and then always beside seed 1's interval (10/120 vs 12/120; −1.67 pp [−6.67, +2.50]).

## Not claimed
No pooling of the two seeds; no mechanism (the declared bundle: policy of origin, repository distribution, harness of origin); the training data come from one repository (fastapi/fastapi); Rich is the known holdout; nothing about the hidden competition set; the joint T_sub cell is read separately under v064-r2.

## Freeze points
Frozen before the confirmation training starts; its sha is pinned in the confirmation training contract and in the v064c analysis contract; no edits afterwards (errata → finding note).
