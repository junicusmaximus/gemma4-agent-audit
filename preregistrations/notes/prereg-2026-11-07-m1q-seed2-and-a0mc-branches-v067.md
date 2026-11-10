---
date: 2026-11-07
type: prereg
status: FROZEN 2026-11-07 14:04Z — momus-v067 REJECT on cd8434c0 (2 blocking: the seed-override wrapper needs the hyperparameter-equality/override exemptions + run-dir/pointer changes; the 'not a recovery' clause contradicted the ban) fixed as proposed + 3 non-blocking (contrast analyzer named and pinned before A0mc is read; 20,000 resamples; 'pooled ≥ 4'); written after the M1q reading and BEFORE the A0mc report is read; pinned in the M1q-s2 training contract and the v067 analysis/evaluation contracts; no further edits (errata → finding notes)
tags: [gemma4, paper-track, v067, m1q, seed2, a0mc, k8, replicate]
related: [[finding-2026-11-07-m1q-mask-on-qat-base-v062]], [[prereg-2026-11-07-m1q-mask-on-qat-base-v062]], [[prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2]], [[prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065]], [[advice-2026-11-07-oracle-after-m1q-repair]]
---
# Pre-registration — v067: (A) M1q versus the matched-template base A0mc, pre-written readings; (B) a second training seed of M1q

## Context
M1q (v062, finding 3b711578…) resolved 4/2/4 = 10/120 on primary40 versus 0/120 for the same-template unmasked control M0qmc (paired +8.33 pp [+1.67, +16.67]); label PK-ii; think 0.905; the frozen v062 wording requires an independent-repository confirmation before any wording beyond the observation. Its termination metrics meet the v064-r2 COLLAPSED criteria (loop 0.775 > 0.40; submit 0.575 < 0.70): repair appeared while termination did not move. The v062 "second training seed" clause (PK-i true) did not fire; this file is a new exploratory decision taken after the reading (declared as such).

## (A) M1q − A0mc reading (before the A0mc report is read)
- A0mc = generation v066 (PINNED runtime 0f5e04b2…, analysis a070f299…): the E4B QAT base, no adapter, matched template, 48 × 3. Its own report is descriptive.
- Contrast: M1qmc − A0mc, primary40, paired by task with the three decoding seeds grouped (task-level bootstrap, 20,000 resamples as the v058 machinery draws), on pooled resolved; all48 and clean36 as sensitivities. Computed without rerunning either evaluation by tools/analyze_m1q_vs_a0mc_v067.py = the v062 analyzer re-pinned with slot M0 = the sealed v066 A0mc rows and slot M1 = the sealed v062 M1qmc rows; its sha is recorded in the v067 analysis contract before the A0mc report is read (ANALYZE v066 waits for it).
- Pre-written sentences (exactly one applies; chosen by the CI, never after inspection):
  - lower bound > 0: "the masked arm on the QAT base resolved more tasks than the same-template base (x/120 vs y/120; +d pp [l, u]) — one training seed, known holdout";
  - upper bound < 0: "the masked arm resolved fewer tasks than the same-template base (x/120 vs y/120; d pp [l, u])";
  - otherwise: "relative to the same-template base the difference is uncertain (x/120 vs y/120; d pp [l, u])".
- Never: "recovery", "partial recovery", "closed the gap", "masking restores / causes repair", "no-collapse".

## (B) Second training seed of M1q (M1q-s2)
- Recipe identical to M1q (v062): M1 variant of mask-control-v035, dequantized QAT base (20f1c2f9…), rank 16, alpha 16, lr 1e-4, grad_accum 8, warmup 10 updates, weight decay 0, max_grad_norm 1.0, max_seq_len 16384, 208 updates, the frozen update groups and order — except the LoRA initialization seed **20261104**. The init-hash gate records the new initial_trainable_parameters_sha256 instead of requiring 9dfc7ccc…; all other v062 gates apply (names_match_probe, updates 208, M1 target count 4,365,092, frozen code shas). As a descriptive identity check, update 1's loss is compared with M1q's update 1 (0.47084; seed-independent because LoRA B starts at zero); a mismatch at 5 decimals is reported as a data/order/base difference.
- Training: new wrapper = a copy of sft/train_m1q_v062.py in which only these change: the seed (contract hyperparameters seed 20261104; the v035-hyperparameter-equality gate and the M0q/M1 override checks exempt `seed` only), the init-hash rule (record, not require 9dfc7ccc…), the run dirs sft/runs/A1_M1q_s2_v067{,_smoke}, the contract default and the prereg pointer (v067), and the unit name m1q-s2-v067-main; the full line diff is pinned in the DRAFT contract; DRAFT → APPROVED contract beside; fresh init; no resume. CPU-only v060 controls may overlap the training under the frozen addendum §3 (start before launch or after update 21; latch pause; no resume during the training).
- Evaluation: the v062 matched-context machinery re-pinned to the M1q-s2 adapter as generation v067 (results/v067_M1qs2mc_s{1,2,3}; 48 × 3; matched template; capped; receipts; canary rule; eligibility rule); evaluated alone on the GPU.
- Readings (primary40; all48 and clean36 sensitivities): (1) M1q-s2 pooled resolved / 120 and per seed; pooled ≥ 4 = notable (as v062); (2) termination metrics and think rate vs M0qmc as in v062 (descriptive); (3) M1q-s2 − A0mc with the rule of (A).
- Pre-written sentences: M1q-s2 ≥ 4/120 → "observed in two training seeds (10/120 and x/120) on the known holdout"; 1–3/120 → "below the notable threshold in the second seed (x/120)"; 0/120 → "seed-dependent: one of two training seeds (10/120 and 0/120)". Both seeds are always reported side by side; never pooled into one "M1q" number; the independent-repository requirement of v062 is not lifted by (B).

## Sequencing and time boxes (UTC)
(A) is read when the A0mc report is sealed. (B) training launch no later than 2026-11-08 18:00, after the A0mc evaluation and the v064 pilot (gap filler) as the GPU allows; evaluation right after the training; reading by 2026-11-10 09:00 or future work. The external-repository panel (DEV14, arms A0mc / M0q / M1q / M1q-s2 if sealed) is pre-registered separately before any E4B row on those tasks exists.

## Not claimed
No mechanism; no causal attribution to masking; no pooling across panels or seeds; no hidden-set statement; the bf16 M1 (v043, 0/120) and the QAT M1q rows are never read as a weight effect.

## Freeze points
This file's sha is pinned in the M1q-s2 training contract and the v067 evaluation contract; frozen before the A0mc report is read; nothing here is edited afterwards (errata → finding notes).
