---
date: 2026-11-07
type: contract-amendment
status: written before Stage 0 of v064-r2; supersedes paper-experiment-contract-v0.4-amendment-onpolicy-v064.md (bc12653c, which cites the superseded v064 text addd2519); the contract v0.1 itself is not edited
tags: [gemma4, paper-track, contract, amendment, v064, r2, on-policy]
related: [[paper-experiment-contract-v0.1]], [[paper-experiment-contract-v0.4-amendment-onpolicy-v064]], [[prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2]]
---
# Amendment v0.4b to paper-experiment-contract v0.1 — own E4B rollouts on the train split (v064-r2)

Lists, before the claim freeze, **own E4B base (A0, no adapter) rollouts on the competition train split** — generated only on the Stage-0 eligible tasks (current-environment gold resolved ∧ empty not resolved, among the 69 sound train tasks other than the canary fastapi_11194), kept iff resolved, per-task cap 3 — as the A_onpol training data of v064-r2, with the matched teacher subset T_sub drawn from the sealed arm_ordinary_tokmatched pool (726 trajectories). Design: prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md, FROZEN sha256 0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60. The 48 Rich holdout tasks are never generated on and never used for data selection. v065 (prereg-2026-11-07-dose-lr-checkpoints-v065.md, FROZEN sha256 0122bd91ae989c99a7ee471fca7ceea4e0c47619a1ef0229431fd932afde5707) trains only on the existing kit data and needs no amendment. Nothing else in contract v0.1 changes.
