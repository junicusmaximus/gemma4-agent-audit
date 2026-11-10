#!/usr/bin/env python3
"""v064-r2 Stage 2, step 2: the training data of arms A_onpol and T_sub (boss v064-r2).

r2 of build_arms.py (065c1ec7; superseded, left unedited) for prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md
(FROZEN 0893728c...) and its addendum (FROZEN b602a5d3...). CPU only.
  A_onpol  rows of migration/onpolicy-v064/kept_trajectories.json (Stage-1 r2 schema onpolicy-kept-v064r2) whose task
           list equals the pinned Stage-0 eligible list (eligible.json 78b856a2..., 57 ids). Training set (Boss's
           pre-data rule, 11:3xZ): resolved and eligible and convertible by convert_traces_r2.py, then per task the 3
           lowest seeds; Unsupported traces are dropped before the cap and listed with the reason, and every difference
           from the runner's informational `selected` flag is listed. S1 floor on that final set: at least 30
           trajectories and at least 10 distinct tasks; otherwise neither arm is built, --write records the refusal
           in sft/data/onpolicy_v064/build.json and write.log, and both modes exit 3.
  m, U     S_A = A_onpol's packed sequences; m = ceil(320 / S_A); U = ceil(S_A * m / 8). Every trajectory of either
           arm is duplicated m times before grouping, so both arms train the same optimizer update count U.
  T_sub    the 726-trajectory arm_ordinary_tokmatched pool: the arms.json ids, equal to the distinct traj_ids of the
           mask_control_v035 identity ledger with as many ledger sequences as kit chunks (packed 1 to 5 per
           trajectory); its sequences are the M0q training sequences of sft/data/mask_control_v035/M0.jsonl (digests
           equal to the v035 ledger). Draw order (tier-stratified): per tier, in sorted tier order, the sorted
           traj_ids shuffled by one random.Random(20261103); draw j takes the tier with the largest deficit
           j * n_tier / 726 - drawn_tier (exact integers; ties: larger tier, then tier name). Trajectories are added
           in that order until ceil(S_T * m / 8) = U, i.e. S_T * m in (8(U - 1), 8U]. If a draw would skip the
           interval, the first later pool trajectory (1 to 5 sequences) that lands inside is taken instead
           (stratification dropped for that last draw); if none exists, T_sub is UNMATCHED and not built (A_onpol is
           read alone). The +-5 % (+-1 when 5 % < 1) sequence match S_T vs S_A is reported only.
  files    per arm, under sft/data/onpolicy_v064/<arm>/ as sft/data/mask_control_v035 for variant M0:
           trajectories.jsonl (kit rows, once), M0.jsonl (one row per duplicated trajectory, copy-major),
           identity_ledger.jsonl (per sequence; digests and shifted-target counts from the frozen prepare e35d3caf
           helpers; the M1 fields mirror M0: no mask), update_groups.json (its make_groups: seed 20261103, groups of 8,
           the last one partial), manifest.json (revision r2, counts, rules, selection record, draw record, DESC
           table, file shas); plus sft/data/onpolicy_v064/build.json and write.log.
  --dry-run   everything but writing; prints the counts Boss reproduces (exit 3 on an S1 refusal)
  --write     Boss's "BUILD DATA v064": the same, then writes into sft/data/onpolicy_v064 (must not exist)
  --pin       Boss's "PIN v064-train": contracts/onpolicy-training-v064r2-DRAFT.json with its PENDING_DATA leaves
              filled from the written data, written beside as contracts/onpolicy-training-v064r2-PINNED.json"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True

ROOT = "PROJECT_ROOT/gemma4"
SFT = ROOT + "/sft"
CONVERTER = SFT + "/onpolicy_v064/convert_traces_r2.py"
PREPARE = SFT + "/prepare_mask_control_v035.py"
PREPARE_SHA = "e35d3caf3749b1f2262bfc31a94fe21e71da7214a591e68a3527469ff8b7ffde"
M0_CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"
M0_CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
BRIEF = "PROJECT_ROOT/briefing/research/gemma-4-2026/"
PREREG = BRIEF + "prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md"
PREREG_SHA = "0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60"
ADDENDUM = BRIEF + "prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065.md"
ADDENDUM_SHA = "b602a5d366002bc5335fcc7a94ada72cba71535b3a63551f74d28bd9a8f39a56"
ELIGIBLE = ROOT + "/migration/onpolicy-v064/eligible.json"
ELIGIBLE_SHA = "78b856a2382068b86a1f5b3ea0624ef01b5962f4b1c98ecdf38f32edeb082625"
ELIGIBLE_LIST_SHA = "533c64a760716f8cb794de510931c36fbc0d6b737badffe409a217b1395c6ae1"
KEPT = ROOT + "/migration/onpolicy-v064/kept_trajectories.json"
KIT = SFT + "/data/trajectories.jsonl"
ARMS_FILE = SFT + "/data/arms.json"
V035 = SFT + "/data/mask_control_v035"
POOL_ARM, POOL_SIZE = "arm_ordinary_tokmatched", 726
OUT = SFT + "/data/onpolicy_v064"
ARMS = ("A_onpol", "T_sub")
DRAFT = ROOT + "/contracts/onpolicy-training-v064r2-DRAFT.json"
PINNED = ROOT + "/contracts/onpolicy-training-v064r2-PINNED.json"
PENDING_DATA = "PENDING_DATA"
SEED, GROUP, TARGET, CAP, MIN_KEPT, MIN_TASKS, MAX_LEN = 20261103, 8, 320, 3, 30, 10, 16384
CANARY = "fastapi_11194"
FILES = ("trajectories.jsonl", "M0.jsonl", "identity_ledger.jsonl", "update_groups.json", "manifest.json")
STATUS, REVISION, EXIT_REFUSED = "DRAFT_DATA_NO_TRAINING", "r2", 3


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def dump(value):
    return json.dumps(value, separators=(",", ":"))


def load_module(name, path, sha=None):
    if sha is not None and sha256_file(path) != sha:
        raise ValueError("frozen code changed: " + path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_eligible(path=ELIGIBLE, sha=ELIGIBLE_SHA, list_sha=ELIGIBLE_LIST_SHA):
    """The pinned Stage-0 eligible ids: file sha as pinned; schema, generation, the frozen r2 prereg, no stop; the count
    and the list sha (sha256 of the ids in candidate order, each followed by a newline) as pinned; no rich_ id and not
    the canary."""
    if sha256_file(path) != sha:
        raise ValueError("eligible list changed: " + path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    ids = doc.get("eligible_task_ids") or []
    digest = hashlib.sha256("".join(i + chr(10) for i in ids).encode()).hexdigest()
    if ((doc.get("schema"), doc.get("generation"), doc.get("stop"), (doc.get("prereg") or {}).get("sha256"))
            != ("onpolicy-v064-eligible-1", "v064", False, PREREG_SHA) or doc.get("eligible_count") != len(ids)
            or not ids or digest != list_sha or doc.get("eligible_list_sha256") != list_sha or len(set(ids)) != len(ids)
            or any(i.startswith("rich_") or i == CANARY for i in ids)):
        raise ValueError("eligible list content differs from its pin")
    return ids


def plan(s_a):
    """(m, U) of the prereg: m = ceil(320 / S_A), U = ceil(S_A * m / 8)."""
    if not isinstance(s_a, int) or s_a <= 0:
        raise ValueError("A_onpol has no packed sequence")
    m = -(-TARGET // s_a)
    return m, -(-s_a * m // GROUP)


def tier_order(meta, seed=SEED):
    """The T_sub draw order of the module doc over meta {traj_id: {"tier": ...}}."""
    rng, queues = random.Random(seed), {}
    for tier in sorted({v["tier"] for v in meta.values()}):
        ids = sorted(t for t, v in meta.items() if v["tier"] == tier)
        rng.shuffle(ids)
        queues[tier] = ids
    total, drawn, order = len(meta), collections.Counter(), []
    for j in range(1, total + 1):
        tier = max((t for t in queues if drawn[t] < len(queues[t])),
                   key=lambda t: (j * len(queues[t]) - drawn[t] * total, len(queues[t]), t))
        order.append(queues[tier][drawn[tier]])
        drawn[tier] += 1
    return order


def select_tsub(order, n_seq, m, updates):
    """(chosen traj_ids, how): the shortest prefix of the draw order with ceil(S_T * m / 8) = U; on a skip, the first
    later trajectory that lands S_T * m inside (8(U - 1), 8U]; (None, reason) when T_sub cannot be matched."""
    low, high = GROUP * (updates - 1), GROUP * updates
    chosen, s = [], 0
    for i, t in enumerate(order):
        if (s + n_seq[t]) * m > high:
            for later in order[i + 1:]:
                if low < (s + n_seq[later]) * m <= high:
                    return chosen + [later], "fallback: draw %d (%s) skipped the interval; %s taken without the " \
                        "stratification" % (i + 1, t, later)
            return None, "unmatched: draw %d skipped the interval and no later trajectory lands inside" % (i + 1)
        chosen.append(t)
        s += n_seq[t]
        if s * m > low:
            return chosen, "stratified: the first %d draws" % len(chosen)
    return None, "unmatched: the pool ran out"


def sequence_match(s_t, s_a):
    """The reported (never gating) sequence match: relative difference and the +-5 % (+-1 when 5 % < 1) band."""
    band = max(0.05 * s_a, 1.0)
    return {"S_T": s_t, "S_A": s_a, "relative_difference": round((s_t - s_a) / s_a, 4),
            "band_sequences": round(band, 2), "within_band": abs(s_t - s_a) <= band}


DESC_KEYS = ("tier", "task_id", "repository", "generating_model", "n_steps", "tool_calls", "total_tokens", "loss_tokens")


def check_inputs(m0_contract):
    """The kit file, arms.json and the v035 M0 rows and ledger equal the approved v0351 contract's hashes."""
    want = {KIT: m0_contract["source_hashes"][KIT], ARMS_FILE: m0_contract["source_hashes"][ARMS_FILE]}
    for name in ("M0.jsonl", "identity_ledger.jsonl"):
        want[V035 + "/" + name] = m0_contract["artifact_hashes"][V035 + "/" + name]
    bad = sorted(p for p, s in want.items() if sha256_file(p) != s)
    if bad:
        raise ValueError("pinned inputs changed: " + ", ".join(bad))
    return want


def read_pool():
    """{traj_id: DESC fields + n_chunks + kit_line (0-based)} of the 726 pool trajectories: the arms.json
    arm_ordinary_tokmatched ids, which must equal the distinct traj_ids of the mask_control_v035 identity ledger (the
    M0q training sequences), each with as many ledger sequences as kit chunks."""
    with open(ARMS_FILE, encoding="utf-8") as f:
        ids = json.load(f)[POOL_ARM]
    if len(ids) != POOL_SIZE or len(set(ids)) != POOL_SIZE:
        raise ValueError("the %s pool is not %d distinct trajectories" % (POOL_ARM, POOL_SIZE))
    with open(V035 + "/identity_ledger.jsonl", encoding="utf-8") as f:
        ledger = collections.Counter(json.loads(line)["traj_id"] for line in f)
    pool, meta = set(ids), {}
    with open(KIT, encoding="utf-8") as f:
        for n, line in enumerate(f):
            r = json.loads(line)
            if r["traj_id"] in pool:
                meta[r["traj_id"]] = dict({k: r[k] for k in DESC_KEYS}, n_chunks=len(r["chunks"]), kit_line=n)
    if set(meta) != pool or set(ledger) != pool or any(ledger[t] != meta[t]["n_chunks"] for t in pool):
        raise ValueError("pool ids or sequence counts differ between arms.json, the kit file and the v035 ledger")
    return meta


def read_tsub(chosen, meta, prepare):
    """(kit lines verbatim, v035 M0 rows) of the chosen trajectories in draw order. Each M0 row must name its kit line
    (source_line, 1-based), carry the kit row's chunks, and its sequences must have the v035 ledger's digests (the M0q
    training sequences)."""
    lines = {meta[t]["kit_line"]: t for t in chosen}
    kit = {}
    with open(KIT, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n in lines:
                kit[lines[n]] = line
    rows, want = {}, set(chosen)
    with open(V035 + "/M0.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["traj_id"] in want:
                rows[r["traj_id"]] = r
    ledger = {}
    with open(V035 + "/identity_ledger.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["traj_id"] in want:
                ledger[(e["traj_id"], e["chunk_index"])] = e
    for t in chosen:
        r, k = rows[t], json.loads(kit[t])
        same = [(c["input_ids"], c["labels"]) for c in r["chunks"]] == [(c["input_ids"], c["labels"]) for c in k["chunks"]]
        if r["source_line"] != meta[t]["kit_line"] + 1 or not same:
            raise ValueError(t + ": v035 M0 row differs from its kit row")
        for ci, c in enumerate(r["chunks"]):
            e = ledger[(t, ci)]
            if (prepare.digest(c["input_ids"]), prepare.digest(c["labels"])) != (e["input_sha256"], e["m0_labels_sha256"]):
                raise ValueError(t + ": sequence differs from the v035 identity ledger")
    return [kit[t] for t in chosen], [rows[t] for t in chosen]


def sequence_problems(rows):
    """Malformed sequences: ids / labels of unequal length or above 16,384, labels unequal to ids where supervised, or
    no target after the causal shift."""
    bad = []
    for r in rows:
        for ci, c in enumerate(r["chunks"]):
            ids, lab = c["input_ids"], c["labels"]
            if (not len(ids) == len(lab) <= MAX_LEN or any(y != -100 and y != x for x, y in zip(ids, lab))
                    or not any(y != -100 for y in lab[1:])):
                bad.append("%s chunk %d" % (r["traj_id"], ci))
    return bad


def duplicate(rows, m, sources):
    """M0.jsonl rows of the duplicated set, copy-major: copy 1 of every trajectory, then copy 2, ..."""
    return [{"traj_id": r["traj_id"], "copy": c, "source": sources[r["traj_id"]], "chunks": r["chunks"]}
            for c in range(1, m + 1) for r in rows]


def identity_ledger(m0_rows, prepare):
    """Per sequence: position, digests and shifted-target counts from the frozen prepare helpers (M1 fields = M0)."""
    out, memo = [], {}
    for row_no, r in enumerate(m0_rows):
        for ci, c in enumerate(r["chunks"]):
            key = (r["traj_id"], ci)
            if key not in memo:
                labels, targets = prepare.digest(c["labels"]), prepare.shifted(c["labels"])
                identity = prepare.target_identity(c["labels"])
                memo[key] = {"input_sha256": prepare.digest(c["input_ids"]), "m0_labels_sha256": labels,
                             "m1_labels_sha256": labels, "changed_original_positions": [],
                             "m0_shifted_targets": targets, "m1_shifted_targets": targets,
                             "m0_target_identity_sha256": identity, "m1_target_identity_sha256": identity,
                             "n_tokens": len(c["input_ids"])}
            out.append(dict({"sequence_index": len(out), "row": row_no, "traj_id": r["traj_id"], "copy": r["copy"],
                             "chunk_index": ci, "chain": c["chain"], "steps": c["steps"]}, **memo[key]))
    return out


def desc(items):
    """The prereg's DESC data table of one arm, before duplication."""
    mix = lambda key: dict(sorted(collections.Counter(str(i[key]) for i in items).items()))
    return {"trajectories": len(items), "tasks": len({i["task_id"] for i in items}), "repositories": mix("repository"),
            "repositories_distinct": len({str(i["repository"]).lower() for i in items}),
            "n_steps": sum(i["n_steps"] for i in items), "tool_calls": sum(i["tool_calls"] or 0 for i in items),
            "loss_tokens": sum(i["loss_tokens"] for i in items), "total_tokens": sum(i["total_tokens"] for i in items),
            "generating_model": mix("generating_model"), "tier": mix("tier"),
            "packed_sequences": sum(i["n_chunks"] for i in items)}


def arm_payload(arm, kit_lines, rows, items, m, updates, sources, prepare, extra):
    """One arm's file contents, checked: no malformed sequence; make_groups gives U updates whose denominators equal
    their M0 shifted targets."""
    bad = sequence_problems(rows)
    if bad:
        raise ValueError(arm + ": malformed sequences: " + ", ".join(bad[:5]))
    m0_rows = duplicate(rows, m, sources)
    ledger = identity_ledger(m0_rows, prepare)
    groups = prepare.make_groups(ledger)
    if len(groups) != updates or any(g["original_denominator"] != g["m1_remaining_targets"] for g in groups):
        raise ValueError(arm + ": update groups differ from the plan")
    counts = {"trajectories": len(rows), "tasks": len({i["task_id"] for i in items}), "duplication_m": m,
              "duplicated_rows": len(m0_rows), "packed_sequences": sum(len(r["chunks"]) for r in rows),
              "sequences": len(ledger), "updates": len(groups),
              "last_update_sequences": len(groups[-1]["sequence_indices"]),
              "shifted_targets": sum(e["m0_shifted_targets"] for e in ledger),
              "input_tokens": sum(e["n_tokens"] for e in ledger)}
    return {"arm": arm, "kit_lines": kit_lines, "m0_rows": m0_rows, "ledger": ledger, "groups": groups,
            "counts": counts, "extra": extra,
            "desc": dict(desc(items), duplication_m=m, updates=len(groups),
                         last_update_sequences=counts["last_update_sequences"])}


def emit(path, texts):
    """sha256 of the concatenated texts, also written to path (which must not exist) unless path is None."""
    h, f = hashlib.sha256(), (open(path, "x", encoding="utf-8", newline="\n") if path else None)
    try:
        for text in texts:
            h.update(text.encode("utf-8"))
            if f:
                f.write(text)
    finally:
        if f:
            f.close()
    return h.hexdigest()


def write_arm(p, folder, base):
    """(file shas, manifest) of one arm; files are written only when folder is given."""
    at = lambda name: folder + "/" + name if folder else None
    shas = {"trajectories.jsonl": emit(at("trajectories.jsonl"),
                                       (x if x.endswith("\n") else x + "\n" for x in p["kit_lines"])),
            "M0.jsonl": emit(at("M0.jsonl"), (dump(r) + "\n" for r in p["m0_rows"])),
            "identity_ledger.jsonl": emit(at("identity_ledger.jsonl"), (dump(e) + "\n" for e in p["ledger"])),
            "update_groups.json": emit(at("update_groups.json"), [json.dumps(p["groups"], indent=2) + "\n"])}
    manifest = dict(base, arm=p["arm"], counts=p["counts"], desc=p["desc"], files_sha256=dict(shas), **p["extra"])
    shas["manifest.json"] = emit(at("manifest.json"), [json.dumps(manifest, indent=1) + "\n"])
    return shas, manifest


RULES = {"selection": "resolved and eligible and convertible by convert_traces_r2.py, then per task the 3 lowest "
                      "seeds; Unsupported rows dropped before the cap and listed; the runner's selected flag is "
                      "informational and every difference is listed (Boss, pre-data operational decision 11:3xZ)",
         "s1": "after conversion exclusions and the cap: >= 30 trajectories and >= 10 distinct tasks, else neither arm "
               "is built and the refusal is recorded",
         "plan": "m = ceil(320 / S_A); U = ceil(S_A * m / 8); every trajectory duplicated m times, copy-major; both arms "
                 "train the same optimizer update count U",
         "draw": "T_sub: tier-stratified Random(20261103) order of the 726 pool; first prefix with ceil(S_T*m/8) = U; "
                 "on a skip the first later trajectory landing inside; else unmatched (build_arms_r2.py doc)",
         "groups": "prepare_mask_control_v035.make_groups on the identity ledger (seed 20261103, groups of 8)",
         "variant": "M0: kit labels unmasked; M1 ledger fields mirror M0"}


def select(kept, converted, excluded):
    """(rows, infos, differences): the convertible rows that are among the 3 lowest convertible seeds of their task, in
    manifest order; differences = every row whose selection here differs from the runner's selected flag, with why."""
    ok = {(e["task_id"], e["seed"]): (row, info) for e, row, info in converted}
    seeds = collections.defaultdict(list)
    for task, seed in ok:
        seeds[task].append(seed)
    chosen = {(task, s) for task, ss in seeds.items() for s in sorted(ss)[:CAP]}
    unsupported = {(x["task_id"], x["seed"]): x["reason"] for x in excluded}
    rows, infos, diffs = [], [], []
    for e in kept["kept"]:
        key = (e["task_id"], e["seed"])
        if key in chosen:
            rows.append(ok[key][0])
            infos.append(ok[key][1])
        if (key in chosen) != e["selected"]:
            why = ("Unsupported: " + unsupported[key] if key in unsupported else
                   "taken here: a lower-seed success of the task is Unsupported" if key in chosen else
                   "beyond the cap of 3 convertible rows")
            diffs.append({"traj_id": "v064_%s_s%d" % key, "runner_selected": e["selected"], "selected": key in chosen,
                          "why": why})
    return rows, infos, diffs


def build(kept_path=KEPT, converter=None, prepare=None, tools=None, eligible=(ELIGIBLE, ELIGIBLE_SHA, ELIGIBLE_LIST_SHA),
          min_kept=MIN_KEPT, min_tasks=MIN_TASKS):
    """Everything --dry-run and --write compute; writes nothing. Below the S1 floor the state carries the refusal and
    no payload."""
    for path, sha in ((PREREG, PREREG_SHA), (ADDENDUM, ADDENDUM_SHA), (M0_CONTRACT, M0_CONTRACT_SHA)):
        if sha256_file(path) != sha:
            raise ValueError("changed: " + path)
    with open(M0_CONTRACT, encoding="utf-8") as f:
        inputs = check_inputs(json.load(f))
    ids = read_eligible(*eligible)
    converter = load_module("convert_traces_v064r2", CONVERTER) if converter is None else converter
    prepare = load_module("prepare_mask_control_v035", PREPARE, PREPARE_SHA) if prepare is None else prepare
    tools = converter.load_tools() if tools is None else tools
    kept, converted, excluded = converter.convert_kept(kept_path, tools, ids)
    if kept.get("eligible_sha256") != eligible[1]:
        raise ValueError("the kept manifest names another eligible list")
    rows, infos, diffs = select(kept, converted, excluded)
    attempted = {t: ((kept.get("per_task") or {}).get(t) or {}).get("attempted") for t in ids}
    selection = {"rule": RULES["selection"], "uncapped": len(kept["kept"]), "runner_selected": kept["counts"]["selected"],
                 "convertible": len(converted), "unsupported": len(excluded), "capped": len(rows),
                 "capped_tasks": len({r["task_id"] for r in rows}), "k_attempted": attempted,
                 "k_attempted_distribution": {str(k): v for k, v in sorted(
                     collections.Counter(attempted.values()).items(), key=lambda kv: str(kv[0]))},
                 "differences_from_runner_selected": diffs}
    code = {"converter": sha256_file(CONVERTER), "build_arms": sha256_file(os.path.abspath(__file__)),
            "packer": converter.PINS[converter.BUILD_SFT], "prepare": PREPARE_SHA}
    base = {"status": STATUS, "generation": "v064", "revision": REVISION, "variant": "M0",
            "prereg": {"path": PREREG, "sha256": PREREG_SHA}, "addendum": {"path": ADDENDUM, "sha256": ADDENDUM_SHA},
            "eligible": {"path": eligible[0], "sha256": eligible[1], "list_sha256": eligible[2], "count": len(ids)},
            "rules": RULES, "inputs": inputs, "code_sha256": code,
            "kept_manifest": {"path": kept_path, "sha256": sha256_file(kept_path)}}
    state = {"base": base, "selection": selection, "excluded": excluded}
    if len(rows) < min_kept or selection["capped_tasks"] < min_tasks:
        return dict(state, refused="S1: %d trajectories over %d distinct tasks after conversion exclusions and the cap "
                    "of %d (floor >= %d and >= %d); A_onpol and T_sub are not built" % (
                        len(rows), selection["capped_tasks"], CAP, min_kept, min_tasks))
    s_a = sum(len(r["chunks"]) for r in rows)
    m, updates = plan(s_a)
    base["plan"] = {"S_A": s_a, "m": m, "U": updates}
    meta = read_pool()
    chosen, how = select_tsub(tier_order(meta), {t: v["n_chunks"] for t, v in meta.items()}, m, updates)
    sources = {"v064_%s_s%d" % (e["task_id"], e["seed"]): {"run": e.get("run"), "trace": e["trace_path"],
                                                         "trace_sha256": e["trace_sha256"],
                                                         "runner_selected": e["selected"]} for e in kept["kept"]}
    keep = ("traj_id", "calls", "newline", "no_newline", "token_offsets", "nudges", "merged_usage_steps", "history",
            "chunks", "dropped_steps", "loss_vs_server")
    a = arm_payload("A_onpol", [dump(r) + "\n" for r in rows], rows,
                    [dict({k: r[k] for k in DESC_KEYS}, n_chunks=len(r["chunks"])) for r in rows], m, updates,
                    {r["traj_id"]: sources[r["traj_id"]] for r in rows}, prepare,
                    {"selection": selection, "excluded": excluded, "conversion": [{k: i[k] for k in keep} for i in infos]})
    a["desc"].update({k: selection[k] for k in ("uncapped", "convertible", "capped", "k_attempted_distribution")})
    payloads, match = [a], None
    if chosen is not None:
        kit_lines, t_rows = read_tsub(chosen, meta, prepare)
        match = sequence_match(sum(meta[t]["n_chunks"] for t in chosen), s_a)
        draw = {"how": how, "draws": len(chosen), "traj_ids": chosen, "match": match,
                "pool_tiers": dict(sorted(collections.Counter(v["tier"] for v in meta.values()).items())),
                "pool_sequences_per_trajectory": {str(k): v for k, v in sorted(
                    collections.Counter(v["n_chunks"] for v in meta.values()).items())},
                "tiers_drawn": dict(sorted(collections.Counter(meta[t]["tier"] for t in chosen).items()))}
        payloads.append(arm_payload("T_sub", kit_lines, t_rows, [meta[t] for t in chosen], m, updates,
                                    {t: {"source_line": meta[t]["kit_line"] + 1} for t in chosen}, prepare,
                                    {"draw": draw}))
    return dict(state, payloads=payloads, how=how, match=match,
                summary={"uncapped": len(kept["kept"]), "convertible": len(converted), "excluded": len(excluded),
                         "capped": len(rows), "capped_tasks": selection["capped_tasks"], "S_A": s_a, "m": m,
                         "U": updates})


def report(state, results):
    """The lines Boss reproduces: the selection, S_A, m, U, T_sub's draw and S_T, counts and file shas per arm."""
    sel = state["selection"]
    lines = ["v064-r2 A_onpol selection: %d kept rows uncapped (runner selected %d); %d convertible, %d Unsupported "
             "(dropped before the cap); after the cap of %d per task: %d trajectories over %d tasks; S1 floor >= %d and "
             ">= %d: %s" % (sel["uncapped"], sel["runner_selected"], sel["convertible"], sel["unsupported"], CAP,
                            sel["capped"], sel["capped_tasks"], MIN_KEPT, MIN_TASKS,
                            "REFUSED" if state.get("refused") else "met")]
    lines += ["v064-r2 A_onpol Unsupported %s: %s" % (x["traj_id"], x["reason"]) for x in state["excluded"]]
    lines += ["v064-r2 A_onpol differs from the runner's selected flag: %s runner %s, here %s (%s)" % (
        d["traj_id"], d["runner_selected"], d["selected"], d["why"]) for d in sel["differences_from_runner_selected"]]
    if state.get("refused"):
        return lines + ["v064-r2 REFUSED " + state["refused"]]
    s = state["summary"]
    q = s["S_A"] * s["m"]
    lines.append("v064-r2 A_onpol: S_A = %d packed sequences; m = ceil(320 / %d) = %d; S_A * m = %d; U = ceil(%d / 8) "
                 "= %d (both arms: the same optimizer update count U)" % (s["S_A"], s["S_A"], s["m"], q, q, s["U"]))
    if state["match"] is None:
        lines.append("v064-r2 T_sub: %s; not built (A_onpol is read alone)" % state["how"])
    else:
        d, mt = results["T_sub"][1]["draw"], state["match"]
        lines.append("v064-r2 T_sub: %s; S_T = %d, S_T * m = %d in (%d, %d]; tiers drawn %s of pool %s (pool sequences "
                     "per trajectory %s); S_T vs S_A %+.1f %% (%s the +-%.2f-sequence band, reported only)" % (
                         d["how"], mt["S_T"], mt["S_T"] * s["m"], GROUP * (s["U"] - 1), GROUP * s["U"],
                         json.dumps(d["tiers_drawn"]), json.dumps(d["pool_tiers"]),
                         json.dumps(d["pool_sequences_per_trajectory"]), 100 * mt["relative_difference"],
                         "inside" if mt["within_band"] else "outside", mt["band_sequences"]))
    for arm, (shas, manifest) in results.items():
        lines.append("v064-r2 %s counts: %s" % (arm, json.dumps(manifest["counts"], sort_keys=True)))
        lines.append("v064-r2 %s files: %s" % (arm, " ".join("%s %s" % (n, shas[n]) for n in FILES)))
    return lines


def run(kept_path, write, eligible=(ELIGIBLE, ELIGIBLE_SHA, ELIGIBLE_LIST_SHA)):
    state = build(kept_path, eligible=eligible)
    if write:
        os.mkdir(OUT)  # refuses when it exists: never over an earlier build or refusal
    keys = ("status", "generation", "revision", "prereg", "addendum", "eligible", "kept_manifest", "code_sha256")
    if state.get("refused"):
        lines = report(state, {})
        if write:
            built = dict({k: state["base"][k] for k in keys}, status="S1_REFUSED", refused=state["refused"],
                         selection=state["selection"], excluded=state["excluded"], written_utc=utc())
            emit(OUT + "/build.json", [json.dumps(built, indent=1) + "\n"])
            emit(OUT + "/write.log", ["\n".join(lines) + "\n"])
        print("\n".join(lines), flush=True)
        return EXIT_REFUSED
    results = {}
    for p in state["payloads"]:
        folder = OUT + "/" + p["arm"] if write else None
        if folder:
            os.mkdir(folder)
        results[p["arm"]] = write_arm(p, folder, state["base"])
    lines = report(state, results)
    if write:
        arms = {arm: {"status": "built", "dir": OUT + "/" + arm, "files_sha256": shas, "counts": manifest["counts"]}
                for arm, (shas, manifest) in results.items()}
        arms.setdefault("T_sub", {"status": "unmatched", "how": state["how"]})
        built = dict({k: state["base"][k] for k in keys + ("plan",)}, summary=state["summary"],
                     selection=state["selection"], arms=arms, excluded=state["excluded"], written_utc=utc())
        emit(OUT + "/build.json", [json.dumps(built, indent=1) + "\n"])
        emit(OUT + "/write.log", ["\n".join(lines) + "\n"])
    print("\n".join(lines), flush=True)
    return 0


def pin(draft=DRAFT, pinned=PINNED, out=OUT):
    """PIN v064-train: the DRAFT contract with its PENDING_DATA leaves filled from the written data (file shas
    recomputed and equal to build.json; T_sub "UNMATCHED" when it was not built); written beside as PINNED."""
    with open(draft, encoding="utf-8") as f:
        contract = json.load(f)
    with open(out + "/build.json", encoding="utf-8") as f:
        built = json.load(f)
    if contract.get("status") != "DRAFT":
        raise ValueError("the training contract to pin is not a DRAFT")
    if built.get("status") != STATUS or built.get("revision") != REVISION:
        raise ValueError("the build is not an r2 data build (an S1 refusal or another revision); nothing to pin")
    for arm in ARMS:
        leaf, info = contract["arms"][arm], built["arms"][arm]
        if leaf["data"] != PENDING_DATA or leaf["stages"]["main"]["updates"] != PENDING_DATA:
            raise ValueError(arm + ": the DRAFT's data leaves are not PENDING_DATA")
        if info["status"] != "built":
            leaf["data"] = leaf["stages"]["main"]["updates"] = "UNMATCHED"
            continue
        shas = {n: sha256_file(out + "/" + arm + "/" + n) for n in FILES}
        if shas != info["files_sha256"]:
            raise ValueError(arm + ": data files differ from build.json")
        c = info["counts"]
        leaf["data"] = dict({k: c[k] for k in ("trajectories", "packed_sequences", "duplication_m", "sequences",
                                               "updates", "shifted_targets", "input_tokens")},
                            dir=out + "/" + arm, expected_sha256=shas)
        leaf["stages"]["main"]["updates"] = c["updates"]
    contract["data_build"] = {"build_json": {"path": out + "/build.json", "sha256": sha256_file(out + "/build.json")},
                              "draft": {"path": draft, "sha256": sha256_file(draft)}}
    sha = emit(pinned, [json.dumps(contract, indent=1) + "\n"])
    print("PINNED %s %s" % (pinned, sha), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="compute and print the counts; write nothing")
    mode.add_argument("--write", action="store_true", help='"BUILD DATA v064": write ' + OUT)
    mode.add_argument("--pin", action="store_true", help='"PIN v064-train": write ' + PINNED)
    ap.add_argument("--kept", default=KEPT, help="Stage-1 r2 kept manifest (--dry-run may name another)")
    ap.add_argument("--eligible", default=ELIGIBLE, help="Stage-0 eligible list (--dry-run may name another, checked "
                    "against its own recorded list sha)")
    opts = ap.parse_args(argv)
    if (opts.kept != KEPT or opts.eligible != ELIGIBLE) and not opts.dry_run:
        ap.error("only --dry-run reads another kept manifest or eligible list")
    if opts.pin:
        return pin()
    eligible = (ELIGIBLE, ELIGIBLE_SHA, ELIGIBLE_LIST_SHA)
    if opts.eligible != ELIGIBLE:
        with open(opts.eligible, encoding="utf-8") as f:
            eligible = (opts.eligible, sha256_file(opts.eligible), json.load(f).get("eligible_list_sha256"))
    return run(opts.kept, opts.write, eligible)


if __name__ == "__main__":
    sys.exit(main())
