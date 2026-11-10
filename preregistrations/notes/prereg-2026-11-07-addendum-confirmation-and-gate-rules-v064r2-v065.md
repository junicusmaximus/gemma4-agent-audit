---
date: 2026-11-07
type: prereg-addendum
status: FROZEN 2026-11-07 10:46Z — momus-v064r2-v065 CONFIRM after one fix (G-eq not applied to a seed-20261104 confirmation), applied verbatim; written before A0mc, M1q's reading or any v064-r2 / v065 arm exists; its sha is pinned in every confirmation contract; no further edits (errata → finding notes)
tags: [gemma4, paper-track, v064, r2, v065, confirmation, tie-rule, pace-rule, addendum]
related: [[prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2]], [[prereg-2026-11-07-dose-lr-checkpoints-v065]], [[prereg-2026-11-07-m1q-mask-on-qat-base-v062]], [[2026-11-07-gemma4-v064-r2-v065-review]]
---
# Addendum to v064-r2 (0893728c) and v065 (0122bd91): confirmation selection, approximate-reproduction wording, CPU-job start rule

Source: Codex review record sessions/2026-11-07-gemma4-v064-r2-v065-review.md (3d884b49, local = 151), relayed by the user 2026-11-07 ≈ 10:45Z. The frozen v064-r2 and v065 files are not edited; this addendum only adds rules where they are silent. Points already covered by the frozen texts (no change): the pilot's ≥ 5 distinct successful tasks is a scale condition only, the training floor is ≥ 30 kept AND ≥ 10 distinct tasks (v064-r2 S1 floor); the 7 non-canary DEV8 tasks, if used as E4B training data, are never used for any E4B evaluation (v064-r2 Stage 0); ≥ 4/120 is the minimum repair observation that routes follow-up work, never evidence of improvement — improvement is read only from the arm − A0mc paired CI (lower bound > 0) under the same matched template.

## 1. Approximate-reproduction wording (v065 G-eq)
Passing G-eq means D1 reproduces the M1q loss trajectory within the stated tolerance (equal to the logged 5 decimals, or |Δloss| ≤ 1e-3). It does not mean byte-identical weights. D1's checkpoints are described only as "approximately on the M1q trajectory (loss rows 1–64 within tolerance)", never as "M1q's checkpoints".

## 2. Confirmation selection and tie rule (before any arm is read)
- Eligible for confirmation: any v065 checkpoint and the v064-r2 A_onpol arm that meet their frozen triggers (NON-COLLAPSED ∧ ≥ 4/120 pooled on primary40).
- Ranking among eligible candidates: (1) higher pooled primary40 repair; (2) higher lower bound of the arm − A0mc paired CI; (3) fewer optimizer updates; (4) lower peak learning rate; (5) v065 before v064-r2 (no new data needed). Only the top-ranked candidate is confirmed first; a second confirmation runs only if its training and evaluation can still launch inside the frozen time boxes (last evaluation launch 11-09 18:00 for v065, 11-09 20:00 for v064-r2); no third.
- If candidates are sealed at different times, the ranking is applied over the candidates sealed when the first confirmation would launch; a later-sealed candidate can take only the second slot.
- Confirmation = the candidate's recipe with LoRA initialization seed 20261104 (everything else identical: data, update groups and order, schedule, U, gates except the init-hash value, which is recorded instead of 9dfc7ccc…, and except G-eq, which is not applied to a confirmation — a confirmation never claims the M1q trajectory; its loss rows are reported descriptively), evaluated identically (48 × 3, matched template, comparator A0mc). Both seeds are reported; the original is never replaced. "No-collapse" prose needs the confirmation to be NON-COLLAPSED; "improved" prose needs the confirmation's arm − A0mc CI lower bound > 0.

## 3. CPU-job start rule during trainings (v060 gate and controls, v064 Stage 0)
A CPU job may overlap a training (M1q, v065 D1/D2, v064-r2 A_onpol/T_sub, any confirmation) only under the v062 pace rule, anchored on updates, never on clock time: either it is already running when the training launches (then the baseline = mean tokens/s of updates 2–20 includes it), or it starts after update 21 of that training has been logged; it is paused (STOP file / job pause) when the rolling mean of the last 3 updates drops below 0.95 × baseline and does not resume during that training. Each firing is reported with the same-step ratio to the reference run where one exists (M1q for D1), ≈ 1.00 = sequence-length effect, < 0.95 = contention. No CPU job runs beside a capped evaluation or the generation.

## Freeze
Frozen before the M1q evaluation is read; nothing here is edited afterwards (errata → finding notes).
