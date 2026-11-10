---
date: 2026-11-09
type: change-record
status: FROZEN 2026-11-09 02:53Z — momus-v068 REJECT on 05fa7458 (3 blocking: bridge path list not synced or pinned; no bridge-refusal rule; disclosure understated) fixed as proposed, plus 3 non-blocking notes applied (two pre-written cache rules with a default for the user's A0mc-reuse question; the v070 pin records its own full hash; attempt 2 is not a server restart). Frozen before r22 is built, before any re-pin and before any v068 outcome (attempt 1 or 2) is read. No edits afterwards (errata → finding note).
tags: [gemma4, paper-track, v068, v070, execution-change, attempt2, bytecode-cache]
related: [[prereg-2026-11-07-external-panel-m1q-v068]], [[prereg-2026-11-08-external-contrast-onpolicy-v070]], [[2026-11-09-gemma4-v068-bytecode-retry-review]], [[v071-execution-change-record-2026-11-08]]
---
# v068 execution-change record — attempt 2 after the bytecode-cache integrity stop (runner r21 → r22)

## 1. Attempt 1: stopped, preserved, never a reading
- Pinned runtime contracts/external-e4b-v068-r21-runtime-PINNED.json `8450ed23…`; dev-ledger snapshot `79c26c72…`; runner r21 `e5902442…`. ACTIVATED 2026-11-08 12:57Z. The runner's integrity check stopped it after seed 1 (4 canaries + 48 seed-1 rows): migration/external-e4b-v068/evaluation/stopped.json `285b4ba4…` ("The eligible tasks or their build products changed"; elapsed 2,255.18 s; no automatic retry; cleanup group empty).
- Cause (established read-only; no outcome read): during the seed-1 rollouts Python wrote bytecode-cache files into `__pycache__` directories inside the hashed dependency trees (`results/extctl_v060/_rebuild/envs/<env>/site-packages`) of 5 of the 12 eligible environments. r21 recomputes the full v060 tree hash (sorted relative path + content sha256 of every file) after the canaries and after every seed and compares it with the pin (r21 l.419–430, 515–531, call site l.1368), so the additions stopped the run.
- Scope evidence:
  - Codex audit research/gemma-4-2026/artifacts/codex-v068-cache-audit-2026-11-09/audit.json `b4457509…` (script audit_remote.py `c59ca12f…`; candidate path list `a7bfafc5…`): 97 regular `.pyc` files directly inside `__pycache__` with mtime ≥ 12:57Z (pipdeptree 75, werkzeug 13, loguru 6, attrs 2, wtforms 1; 0 in the other 7 environments). Leaving exactly these out of the hash reproduces all 12 r21-pinned hashes. All 97 have their `.py` source, and every timestamp header's source mtime and size match the current source.
  - Boss re-measurement 2026-11-09 02:2xZ (read-only; the v060 module's own sha_file / sha_text / tree_hash under the swegemma Python 3.12.14):
    - 18,209 files in the 12 hashed trees;
    - 2,394 regular `.pyc` files directly inside `__pycache__`, of which 97 have mtime ≥ 12:57Z (the same 97, the same per-environment counts);
    - the full listing minus those 97 hashes to the r21-pinned value for 12/12 environments;
    - 0 symlinks; 0 `.pyc` outside a `__pycache__` directory; 0 other files inside one; 0 files nested below one;
    - the router directories hold one `.pyc` each, none with mtime ≥ 12:57Z. They are outside the tree hash; their sitecustomize.py is hashed separately.
- Preserved unchanged: everything attempt 1 wrote, the r21 contracts, the ledger snapshot, and the r21 runner, tests and analyzer. Manifests on 151, written 2026-11-09 by Boss with exclusive create:
  - migration/external-e4b-v068/attempt1-evidence-manifest.sha256 `d537ebab…`: 721 files (evaluation/, results/v068_*, subs/v068_*, logs/external-e4b-v068-controller.log);
  - migration/external-e4b-v068/attempt1-evidence-manifest-2.sha256 `f3b7d4ed…`: 109 files (logs/v068_* row logs and routing receipts, the 12:54Z preflight JSON, and three empty v060 uv lock files under _rebuild/tmp that are not attempt-1 artifacts; _rebuild/tmp/v068 holds directories only).

## 2. Decision
User, 2026-11-09: option B, with the eight conditions of the Codex review sessions/2026-11-09-gemma4-v068-bytecode-retry-review.md `c6aa8199…`. One fresh attempt 2 of the whole pre-registered panel, in a new generation, after a verification-only runner change (r22). Everything else is unchanged:
- the same 12 eligible tasks in the frozen order;
- the same four arms (A0mc, M0q, M1q, M1qs2) with the r21-pinned adapter shas;
- decoding seeds 1–3;
- the same canaries, budgets, template, engine configuration, stop rules and readings.

Attempt 2 is the sole v068 reading, whatever it shows. There is no attempt 3.

## 3. The change (r22): verification only
**(a) Exclusion rule.** This is the only change to any hash. A path below a hashed dependency tree is left out of the comparison hash if and only if:
- lstat reports a regular file (never a symlink or special file);
- its immediate parent directory is named `__pycache__`;
- its name ends in `.pyc`.

Everything else is hashed exactly as v060 tree_hash does: `.pyc` files anywhere else, every other file inside `__pycache__`, anything nested deeper below `__pycache__`, and symlinks.

**(b) Where it applies.** Only the dependency-tree comparison in verify_panel(tree=True), at the r21 call sites: preflight, activation, after the canaries and after every seed. Every other r21 check stays as it is: snapshot and router shas, ledger snapshot and digest, frozen order, eligibility and the not-eligible list, canary task, model inventory, adapter pins, template, server script, budgets and the remaining checks.

**(c) Hashes kept side by side.**
- The pinned eligible rows keep every r21 key and value. env_tree_sha256 stays the r21-pinned full (admission) hash.
- A new key, env_tree_filtered_sha256, holds the filtered hash. It is the attempt-2 baseline.
- The pin step writes these per-environment files to a new attempt-2 evidence directory and records their sha256 in the contract:
  - the full file list, as the exact lines tree_hash hashes (relative path and sha256);
  - the full hash at the re-pin;
  - the filtered hash;
  - the starting cache list (every excluded file: relative path, sha256, bytes, mtime_ns).
- v060 never stored the admission file list itself. It is preserved as the full list at the re-pin minus the 97 attempt-1 paths, which hashes to the admission hash by (d).

**(d) Bridge to admission.** The pin refuses unless both of these hold:
- For every environment, the full listing at the re-pin minus exactly the 97 attempt-1 candidate paths hashes to the r21-pinned env_tree_sha256. The paths are read only from migration/external-e4b-v068/attempt2-bridge-candidates.txt on 151: the 97 sorted lines `<environment_id>/<relative_path>` + LF, which is Codex's serialization, so the file's sha256 equals audit.json's candidate_pathlist_sha256 `a7bfafc5…`. Boss writes it in step 1 (exclusive create), and the r22 DRAFT pins its path and sha. This shows the non-cache content at the re-pin equals the admission content, which anchors the filtered baseline to admission.
- The eligible rows (r21 keys), the arm set (M1qs2 included, same adapter shas), the ledger digest, the not-eligible list and the canary task equal those of the r21 pin. The pre-registration lets --pin repeat only while no rollout exists; that clause guards the arm set, and the r22 pin cannot change it. This re-pin after attempt-1 rollouts is a declared deviation from that clause.

**Bridge refusal.** If the bridge refuses for any environment, the cause is recorded read-only and there is no attempt 2. The frozen prereg's stopped case then applies to attempt 1 (n_c = 0 → "fewer than 4 complete tasks (0 of 12)"). The 97-path list is never widened.

**(e) New generation and paths.**
- Generation `v068a2`: results/v068a2_*, subs/v068a2_*, logs/v068a2_*, _rebuild/tmp/v068a2 and a new evidence directory.
- Contracts: contracts/external-e4b-v068-r22-runtime-{DRAFT,PINNED}.json.
- rollout_exists() and every guard read the attempt-2 paths only.
- Nothing attempt 1 wrote is opened for writing, moved or deleted.

**(f) Tests, before any pin.** The r21 tests, adapted to r22, pass. The new filter tests check that:
- an added `.pyc` inside `__pycache__` passes;
- a changed, added or deleted non-cache file stops;
- a `.pyc` outside `__pycache__` stops;
- a non-`.pyc` file inside `__pycache__` stops;
- a file nested below `__pycache__` stops;
- a symlink named `x.pyc` inside `__pycache__` stops;
- the bridge refuses a listing that differs from admission by anything other than the 97 paths.

The full r21→r22 line diff is pinned in the r22 DRAFT contract, and Boss reviews it before the pin.

**(g) Not changed.** Tasks, eligibility, arms, adapters, seeds, budgets, template, sampling, helper, timeouts, 4 in flight, canary rules, exclusion and stop rules, readings, and the analyzer's rules (the analyzer reads attempt-2 paths only).

## 4. Cache policy: keep, and disclose
- **Starting state.** Nothing in the environments is deleted, moved or rewritten before or between attempts. Attempt 2 starts with the bytecode caches present at the re-pin. Measured now, the 12 hashed trees hold 2,394 regular `.pyc` files directly in `__pycache__`:
  - 2,297 present at admission (inside the admission hashes);
  - 97 written during attempt 1.

  Each router directory also holds one cache file. The re-pin lists the exact starting set; any difference from this measurement is reported, not corrected.
- **Disclosure.** Python reads `.pyc` files. Modules whose cache was written in attempt 1 may load from that cache in attempt 2 instead of being compiled from source.
  - The exclusion lets any change to such cache files pass the integrity check: added, modified or deleted, including the 2,297 present at admission. Every non-cache change still stops the run. The exclusion is not a claim that caches cannot affect execution.
  - No claim is made that attempt 2's execution, timing or outcomes equal what attempt 1 would have produced.
  - After attempt 2, Boss diffs the end state against the pinned starting cache list (path, sha256, bytes, mtime_ns), read-only, and reports added, changed and deleted cache files.
- **Declared difference (f) for v068.** Attempt 2 runs after attempt 1 was stopped by an integrity check (seed 1 only; never read). Runner r22 differs from r21 only in the dependency-tree comparison (bytecode-cache exclusion) and the generation paths. It starts with the bytecode caches written by attempt 1 (97 files in 5 environments).

## 5. Time box (absolute deadlines unchanged)
- Latest launch 2026-11-09 12:00Z (r21's LATEST_LAUNCH unchanged).
- Cap: 4 h from attempt 2's server-ready.
- Sealed report and Boss reproduction by 2026-11-10 09:00Z; otherwise "not run" in the paper's limitations.
- The new cap never extends the absolute deadlines.
- Attempt 2 is a new generation with its own server. Its launch is not a "server restart" under the frozen stop rule, which applies within attempt 2.
- If attempt 2 stops under any rule, its completed rows are read with the pre-registered incomplete-panel rule (T_c, attempt-2 rows only). There is no attempt 3.
- The GPU runs alone, with no CPU job beside it.

## 6. Reading; attempt 1
- Attempt 2 is the sole reading.
- Attempt-1 rows are never mixed in, and never used to fill, replace or choose tasks or seeds.
- Their outcomes are not read for the paper. The paper states only that attempt 1 was stopped by the integrity check after seed 1, with row counts.

## 7. v070: the same rule and policy, before its pin
- Before its pin, v070's runner adopts:
  - the exclusion rule of 3(a), with the tests of 3(f);
  - the keep policy (nothing deleted or moved);
  - a pin step that records, per environment, its starting cache list, its full file list and its own recomputed full hash (evidence files, as in 3(c)), and requires every environment's filtered hash to equal the r22-pinned env_tree_filtered_sha256. All other build-product checks are unchanged.
- v070's eligible rows carry env_tree_sha256 copied from the r22 pin (the admission values) and a recomputed env_tree_filtered_sha256. The copied value makes that key equal by construction; the anchoring comes from the filtered equality plus the r22 bridge (3(d)).
- Comparator rows are v068 attempt-2 A0mc-ext rows only.
- Identity check: v070's frozen check reads the named leaves of the r21 PINNED contract as written. The r22 pin must carry identical values for those leaves (Boss checks this at the re-pin), so the check is unaffected. The "eligible list" leaf compares the r21 keys of each row; env_tree_filtered_sha256 is not an r21 key.
- Cache state and the user's A0mc-reuse condition. v070 starts with the caches present after v068 attempt 2, so its starting cache list differs from attempt 2's whenever attempt 2 changed any cache file. Whether cache state counts as a condition under the user's rule "A0mc reuse only if conditions are identical" is the user's call. It is asked on 2026-11-09, and exactly one of two pre-written rules applies:
  - **(k-disclose)**, the default: cache state is not an identity leaf (the frozen v070 check names its leaves). The difference is disclosed as v070 declared difference (k), with the changed cache files listed, and the comparator follows the frozen leaves only.
  - **(k-strict)**: the comparator is shown only if v070's starting cache list (path, sha256) equals attempt 2's. Otherwise no comparator is shown, and the changed cache files are listed.
  - The user's choice is recorded before any attempt-2 outcome is read. If no choice is recorded by then, (k-disclose) applies.
- Order, as in the v071 change record `1a2cfa31…`: v068 → v070. v070 launches only after attempt 2 has ended, within its own ≤ 3 h box, by the 2026-11-10 09:00Z freeze.

## 8. Steps (Boss verifies each)
1. Freeze this record and sync it to 151. Its sha goes into the r22 DRAFT and PINNED contracts. In the same step, write migration/external-e4b-v068/attempt2-bridge-candidates.txt (3(d)) and check its sha256 against `a7bfafc5…`.
2. Build and test r22 off the GPU (the 3(a)–(f) diff only). Boss reviews the r21→r22 diff.
3. Install r22 beside r21 (r21 untouched). Pin on the literal "PIN v068 r22". Boss diffs the r22 PINNED leaves against r21's: only declared leaves differ.
4. Run the preflight. ACTIVATE on the literal "ACTIVATE v068 r22 <sha>" before 12:00Z.
5. Analyse attempt-2 paths only → sealed report → Boss reproduction, by 2026-11-10 09:00Z.
6. Apply section 7 to v070 before its pin.
