---
date: 2026-11-09
type: decision
tags: [gemma4, v070, v068, attempt2, cache-rule, k-disclose, comparator]
related: [[v068-execution-change-record-2026-11-09]], [[prereg-2026-11-08-external-contrast-onpolicy-v070]], [[2026-11-09-gemma4-v068-attempt2]]
---
# v070 cache rule: k-disclose; the A0mc comparator is a technical reference only

**Recorded 2026-11-09 04:06Z.** This is before any v068 attempt-2 outcome exists: "ACTIVATE v068 r22 97882642…" was sent about 04:05Z, and at 04:06:11Z no v068a2 result or log directory existed. The attempt-2 evidence directory held only the 24 pin tree-evidence files.

## User decision (relayed in chat 2026-11-09, verbatim)
> k-disclose로 진행해. 기존에 동결한 모델·템플릿·예산·과제·시드 등 동일성 검사는 유지하고, 캐시 목록과 내용의 차이는 항목 (k)에 공개해.
>
> v068 attempt 2의 A0mc는 다른 실행에서 가져온 기술적 참고값으로만 제시해. 이 비교로 개선·회복·비열등성을 주장하지 마. 중심 판독은 v070 내부의 A_onpol–T_sub 대비로 유지하고, 실행 순서와 캐시 누적 영향도 한계로 기록해.
>
> 이 결정을 attempt 2 결과를 읽기 전에 기록해. 캐시는 삭제하지 말고, r22의 브리지 검사·정식 PIN·프리플라이트를 통과한 뒤 진행해.

Sources:
- Codex review on 151: codex/.briefing/sessions/2026-11-09-v070-cache-comparator-decision.md, sha 4e57b1c9….
- Codex's advisor (oracle) agreed, and confirmed that disclosing the cache difference does not prove the execution environments are equivalent.

## Operative rules
1. The rule is **(k-disclose)** from change record 6a0e5942 §7. The frozen v070 identity checks stay as they are: ledger digest and eligible list (r21 keys), seeds, budgets, template, server script, versions, base model, LoRA rank.
2. **(k) disclosure:** every difference between v070's starting cache list and v068 attempt 2's starting list (path, sha256) is listed. The pin evidence holds both lists.
3. **A0mc-ext comparator:**
   - It comes from v068 attempt 2 (generation v068a2) and is presented only as a technical reference from a different run.
   - No improvement, recovery or non-inferiority claim may rest on it.
   - It is never a reading.
4. **Central reading:** the within-v070 A_onpol vs T_sub contrast, through the pre-registered cells E1–E3.
5. **Limitations to record** in the v070 report and in the paper:
   - execution order: v070 runs after v068 attempts 1 and 2 on the same environment trees;
   - cache accumulation: bytecode caches written by the earlier attempts are present at v070's start.
6. **Preconditions,** both already met before activation:
   - caches are never deleted;
   - r22 passed the bridge (12/12), the formal PIN (97882642…, 04:03:33Z) and the preflight (9b7b1787…, ready, 04:04:16Z).
