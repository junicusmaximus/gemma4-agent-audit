#!/usr/bin/env python3
"""Hq1 history pack (boss v057): re-pack the M0 training chunks with rule S (superset-first). CPU only, one process.

Rules (RULES below): briefing/research/gemma-4-2026/proposal-2026-11-05-history-arm-A1-hist.md, sections 1, 2 and 6.
Input: the exact data of the M0q run (contracts/mask-control-v0351-APPROVED.json pins sft/data/mask_control_v035/M0.jsonl
and update_groups.json). An M0 chunk is header + X/completion runs. A chain's first chunk is copied unchanged. A later
chunk (parent, items k..m) becomes sub-chunks header + context + items: items i..j keep their M0 labels and are taken
greedily; the context (labels -100) is the 3-token thought opener, completion_{k-1}, the parent prefix X_k..completion_{i-1}
and, in the slack, earlier (completion_{a-1}, X_a) pairs while the sequence fits 16,384 tokens including the opener.
The opener at every modified sub-chunk start is a constitutive part of rule S, not a separate repair step.
If even opener + completion_{k-1} cannot fit, the sub-chunk keeps M0's orphan start (exception: no opener, M0 context);
exceptions stay unrepaired and are listed by sequence id and reason class in the parent ledger.
Every sub-chunk joins its parent's update group (parent order, then sub-chunk order); denominators stay M0's.
  --dry-run          statistics only: analysis/history-pack-boss-v057/dry-run-stats.json and dry-run.log
  --write --out DIR  H1.jsonl (M0.jsonl format), parent_ledger.jsonl, update_groups.json, write.log and manifest.json in a
                     fresh DIR; only after the user's go; refused under sealed directories; manifest.json only if all
                     six checks PASS and the files have the predicted sha256
Both modes run the six checks of check_history_pack_v057.py on the in-memory candidate. Deterministic."""
import argparse
import collections
import hashlib
import json
import os
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_history_pack_v057 as checks  # noqa: E402

# Every packing rule, from proposal-2026-11-05-history-arm-A1-hist.md:
RULES = {
    # section 1, constitutive part of rule S (not a separate repair step): <|channel>, thought, newline at every
    # modified sub-chunk start, labels -100
    "opener_ids": [100, 45518, 107],
    # section 1: the opener goes between the header's <|turn>model line and the first restored completion
    "header_end_ids": [105, 4368, 107],
    # section 1: every sequence fits 16,384 tokens, re-checked with the +3 opener tokens
    "max_len": 16384,
    # sections 2 and 6, rule S: the parent prefix is mandatory; where even opener + completion_{k-1} does not fit, the
    # sub-chunk keeps M0's orphan start and context (no opener); such exceptions stay unrepaired and are listed by
    # sequence id and reason class in the parent ledger
    "exception_policy": "orphan_start_kept",
    # section 1 (schedule): each sub-chunk joins its parent M0 chunk's update group; 208 groups and denominators unchanged
    "group_rule": "parent_group_order_preserved",
}

ROOT = "PROJECT_ROOT/gemma4"
REF_DIR = ROOT + "/sft/data/mask_control_v035"
M0, GROUPS = REF_DIR + "/M0.jsonl", REF_DIR + "/update_groups.json"
CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"
CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
TRAJECTORIES, ARMS, ARM = ROOT + "/sft/data/trajectories.jsonl", ROOT + "/sft/data/arms.json", "arm_ordinary_tokmatched"
TOKENIZER = ROOT + "/models/gemma-4-e4b-it-qat-w4a16-ct/tokenizer.json"
PROPOSAL = "PROJECT_ROOT/briefing/research/gemma-4-2026/proposal-2026-11-05-history-arm-A1-hist.md"
M0Q_META = ROOT + "/sft/runs/A1_M0q_v053/meta.json"  # read only: tokens / seconds of M0q for the time estimate
DRY_DIR = ROOT + "/analysis/history-pack-boss-v057"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = [os.path.join(HERE, n) for n in ("build_history_pack_v057.py", "check_history_pack_v057.py",
                                        "train_history_pack_v057.py", "tests/test_build_history_pack_v057.py",
                                        "tests/test_check_history_pack_v057.py", "tests/test_train_history_pack_v057.py")]
CODEX = {"virtual_chunks": 1696, "token_ratio": "1.24089", "exceptions": 35, "targets": 4391649, "groups": 208}
SEALED = [ROOT + "/" + p for p in ("sft/runs", "sft/data/mask_control_v035", "results", "subs", "contracts", "models",
                                   "analysis/mask-control-v043", "analysis/history-drypack",
                                   "analysis/history-pack-boss-v057")]
SEALED_NAMES = re.compile(r"v043|v035|v05[1-6]")


def dump(value):
    return json.dumps(value, separators=(",", ":"))


def verify_inputs(rules):
    """The approved M0 contract, the inputs it pins, and the opener / role ids of its tokenizer; returns shas."""
    shas = {CONTRACT: checks.sha256_file(CONTRACT)}
    if shas[CONTRACT] != CONTRACT_SHA:
        raise ValueError("approved contract changed")
    with open(CONTRACT) as f:
        contract = json.load(f)
    pins = {M0: contract["artifact_hashes"][M0], GROUPS: contract["artifact_hashes"][GROUPS],
            TRAJECTORIES: contract["source_hashes"][TRAJECTORIES], ARMS: contract["source_hashes"][ARMS],
            TOKENIZER: contract["source_hashes"][TOKENIZER]}
    for path, want in pins.items():
        shas[path] = checks.sha256_file(path)
        if shas[path] != want:
            raise ValueError("pinned input changed: " + path)
    shas[PROPOSAL] = checks.sha256_file(PROPOSAL)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    for text, key in (("<|channel>thought\n", "opener_ids"), ("<|turn>model\n", "header_end_ids")):
        if tok.encode(text, add_special_tokens=False).ids != rules[key]:
            raise ValueError("tokenizer ids differ from RULES[%r]" % key)
    return shas


def chain_item_counts(wanted):
    """Completions per (traj_id, chain) in build_sft's text segments; a dropped item would corrupt the history."""
    counts = {}
    with open(TRAJECTORIES, encoding="utf-8") as f:
        for line in f:
            p = line.index('"traj_id":"') + 11
            tid = line[p:line.index('"', p)]
            if tid in wanted:
                for s in json.loads(line)["text_segments"]:
                    if s["train"]:
                        counts[(tid, s["chain"])] = counts.get((tid, s["chain"]), 0) + 1
    return counts


def label_runs(labels, start):
    """Maximal runs of labels[start:] as (supervised, a, b)."""
    out, a = [], start
    while a < len(labels):
        sup, b = labels[a] != -100, a
        while b < len(labels) and (labels[b] != -100) == sup:
            b += 1
        out.append((sup, a, b))
        a = b
    return out


def segment_chain(chunks):
    """Header length, header, X_t (xs[0] is None: item 0's prompt is the header), completions, item steps and the item
    range (k, m) of every M0 chunk of one chain (chunks in chain order)."""
    lab0 = chunks[0]["labels"]
    h = next(n for n, y in enumerate(lab0) if y != -100)
    header = chunks[0]["input_ids"][:h]
    xs, comps, steps, spans = [None], [], [], []
    for n, c in enumerate(chunks):
        ids, lab = c["input_ids"], c["labels"]
        rr = label_runs(lab, h)
        if (len(ids) != len(lab) or ids[:h] != header or any(y != -100 for y in lab[:h]) or not rr
                or not rr[-1][0] or rr[0][0] != (n == 0)):
            raise ValueError("M0 chunk is not header + X/completion runs")
        k = len(comps)
        for sup, a, b in rr:
            if sup and lab[a:b] != ids[a:b]:
                raise ValueError("M0 labels differ from ids")
            (comps if sup else xs).append(ids[a:b])
        s0, s1 = c["steps"]
        if s1 - s0 != len(comps) - 1 - k or len(xs) != len(comps):
            raise ValueError("M0 steps do not match the completions of a chunk")
        steps += range(s0, s1 + 1)
        spans.append((k, len(comps) - 1))
    return h, header, xs, comps, steps, spans


def plan_parent(h, header, xs, comps, k, m, rules):
    """Rule S sub-chunks of the later chunk with items k..m, as (i, j, a): items i..j with the context opener +
    completion_a, X_{a+1}, ..., completion_{k-1} + parent prefix X_k..completion_{i-1}; a is None for an exception
    (M0's orphan start and context kept)."""
    limit, o = rules["max_len"], len(rules["opener_ids"])
    size = [h]  # size[t - k + 1] = header + X_k .. completion_t
    for t in range(k, m + 1):
        size.append(size[-1] + len(xs[t]) + len(comps[t]))

    def used(t):
        return size[t - k + 1]
    if used(m) > limit:
        raise ValueError("M0 chunk longer than max_len")
    base = o + len(comps[k - 1])  # mandatory de-orphan context: opener + completion_{k-1}
    plan, i = [], k
    while i <= m:
        if used(i) + base > limit:
            plan.append((i, m, None))
            break
        j = i
        while j < m and used(j + 1) + base <= limit:
            j += 1
        n, a = used(j) + base, k - 1
        while a >= 1 and n + len(comps[a - 1]) + len(xs[a]) <= limit:
            n, a = n + len(comps[a - 1]) + len(xs[a]), a - 1
        if a == 0:
            raise ValueError("restored history reaches item 0, where the opener rule is undefined (proposal section 1)")
        if (header[-len(rules["header_end_ids"]):] != rules["header_end_ids"] or xs[a][-o:] != rules["opener_ids"]
                or comps[a][:1] == rules["opener_ids"][:1]):
            raise ValueError("opener premise fails: header end, end of X_a or start of completion_a (proposal section 1)")
        plan.append((i, j, a))
        i = j + 1
    return plan


def build_sequence(header, xs, comps, k, i, j, a, rules):
    """input_ids, labels and restored history tokens of one sub-chunk (see plan_parent)."""
    ids, labels, n_hist = list(header), [-100] * len(header), 0
    if a is not None:
        ctx = comps[a][:]
        for t in range(a + 1, k):
            ctx += xs[t] + comps[t]
        n_hist = len(ctx)
        ctx = list(rules["opener_ids"]) + ctx
        ids += ctx
        labels += [-100] * len(ctx)
    for t in range(k, j + 1):
        ids += xs[t] + comps[t]
        labels += [-100] * len(xs[t]) + (comps[t] if t >= i else [-100] * len(comps[t]))
    return ids, labels, n_hist


def pack_record(rec, m0_base, h1_base, rules, item_counts=None):
    """Candidate record in M0.jsonl format and its parent-ledger rows for one M0 trajectory record; m0_base / h1_base
    are the global indices of the record's first M0 / candidate sequence."""
    order = []
    for c in rec["chunks"]:
        if not order or order[-1] != c["chain"]:
            if c["chain"] in order:
                raise ValueError("chunks of a chain are not consecutive")
            order.append(c["chain"])
    out, rows, pc, o = [], [], 0, len(rules["opener_ids"])
    for chain in order:
        chunks = [c for c in rec["chunks"] if c["chain"] == chain]
        h, header, xs, comps, steps, spans = segment_chain(chunks)
        if item_counts is not None and item_counts.get((rec["traj_id"], chain)) != len(comps):
            raise ValueError("%s chain %d: completions differ from the build_sft text segments" % (rec["traj_id"], chain))
        for c, (k, m) in zip(chunks, spans):
            plan = [(0, m, None)] if k == 0 else plan_parent(h, header, xs, comps, k, m, rules)
            for sub, (i, j, a) in enumerate(plan):
                if k == 0:
                    ids, labels, n_hist, kind = c["input_ids"], c["labels"], 0, "first_unchanged"
                else:
                    ids, labels, n_hist = build_sequence(header, xs, comps, k, i, j, a, rules)
                    kind = "modified" if a is not None else "exception_orphan_start_kept"
                out.append({"chain": chain, "steps": [steps[i], steps[j]], "n_tokens": len(ids),
                            "n_loss_tokens": sum(y != -100 for y in labels), "input_ids": ids, "labels": labels})
                rows.append({"sequence_index": h1_base + len(out) - 1, "traj_id": rec["traj_id"],
                             "source_line": rec["source_line"], "chunk_index": len(out) - 1, "chain": chain,
                             "steps": [steps[i], steps[j]], "parent_sequence_index": m0_base + pc,
                             "parent_chunk_index": pc, "sub_index": sub, "kind": kind, "items": [i, j],
                             "parent_items": [k, m], "restored_from_item": a,
                             "restored_steps": 0 if a is None else k - a, "restored_tokens": n_hist,
                             "opener_tokens": 0 if a is None else o,
                             "exception_reason": "no_room_for_completion_k_minus_1" if k and a is None else None,
                             "n_tokens": len(ids), "shifted_targets": sum(y != -100 for y in labels[1:]),
                             "input_sha256": checks.digest(ids), "labels_sha256": checks.digest(labels)})
            pc += 1
    return {"traj_id": rec["traj_id"], "source_line": rec["source_line"], "chunks": out}, rows


def h1_groups(m0_groups, children):
    """Each M0 update group with every parent replaced by its sub-chunks (parent order, then sub-chunk order)."""
    return [{"update": g["update"], "sequence_indices": [x for p in g["sequence_indices"] for x in children[p]],
             "parent_sequence_indices": list(g["sequence_indices"]), "original_denominator": g["original_denominator"],
             "m1_remaining_targets": g["m1_remaining_targets"]} for g in m0_groups]


def check_out(out):
    """--write target: a fresh directory outside every sealed location."""
    out = os.path.realpath(os.path.abspath(out))
    rel = os.path.relpath(out, os.path.realpath(ROOT))
    sealed = [os.path.realpath(s) for s in SEALED]
    if any(out.startswith(s) for s in sealed) or (not rel.startswith("..") and SEALED_NAMES.search(rel)):
        raise SystemExit("REFUSED: --out %s is under a sealed directory" % out)
    if os.path.lexists(out):
        raise SystemExit("REFUSED: --out %s exists; a fresh directory is required" % out)
    return out


def pack_all(rules, say, sinks=None):
    """Stream M0.jsonl once: pack, check and hash every record; sinks (write mode) get the candidate lines."""
    shas = verify_inputs(rules)
    say("inputs verified against contract %s: %s" % (CONTRACT_SHA[:8], ", ".join(
        "%s %s" % (os.path.basename(p), s[:8]) for p, s in shas.items())))
    with open(ARMS) as f:
        wanted = json.load(f)[ARM]
    counts = chain_item_counts(set(wanted))
    say("build_sft text segments: completion counts for %d chains" % len(counts))
    with open(GROUPS) as f:
        m0_groups = json.load(f)
    checker, children, seen, kinds = checks.Checker(rules), {}, [], collections.Counter()
    digests = {"H1.jsonl": hashlib.sha256(), "parent_ledger.jsonl": hashlib.sha256()}
    m0_seq = h1_seq = 0
    with open(M0, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cand, rows = pack_record(rec, m0_seq, h1_seq, rules, counts)
            checker.add(rec, cand, rows)
            text = {"H1.jsonl": dump(cand) + "\n", "parent_ledger.jsonl": "".join(dump(r) + "\n" for r in rows)}
            for name, t in text.items():
                digests[name].update(t.encode())
                if sinks:
                    sinks[name].write(t)
            for r in rows:
                children.setdefault(r["parent_sequence_index"], []).append(r["sequence_index"])
                kinds[r["kind"]] += 1
            seen.append(rec["traj_id"])
            m0_seq, h1_seq = m0_seq + len(rec["chunks"]), h1_seq + len(cand["chunks"])
            if len(seen) % 100 == 0:
                say("packed %d trajectories, %d M0 -> %d candidate sequences" % (len(seen), m0_seq, h1_seq))
    if sorted(seen) != sorted(wanted):
        raise ValueError("M0 trajectories differ from the frozen arm")
    groups = h1_groups(m0_groups, children)
    groups_text = json.dumps(groups, indent=2) + "\n"  # layout of M0's update_groups.json
    predicted = {n: d.hexdigest() for n, d in digests.items()}
    predicted["update_groups.json"] = hashlib.sha256(groups_text.encode()).hexdigest()
    report = checker.finish(m0_groups, groups)
    split = sum(1 for v in children.values() if len(v) > 1)
    return shas, report, groups_text, predicted, kinds, split, (len(seen), m0_seq, h1_seq)


def make_stats(mode, started, rules, shas, report, predicted, kinds, split, sizes):
    n_traj, m0_seq, h1_seq = sizes
    m0_tok, h1_tok = report["4"]["tokens_reference"], report["4"]["tokens_candidate"]
    opener_tok = report["5"]["opener_insertions"] * len(rules["opener_ids"])
    with open(M0Q_META) as f:
        meta = json.load(f)
    rate = meta["tokens"] / meta["seconds"]
    observed = {"virtual_chunks": h1_seq, "token_ratio": "%.6f" % (h1_tok / m0_tok),
                "exceptions": report["3"]["exceptions"], "targets": report["1"]["candidate"]["labelled"],
                "groups": report["2"]["updates"]}
    return {
        "mode": mode, "started_utc": started, "rules": rules, "proposal": PROPOSAL,
        "inputs_sha256": shas, "code_sha256": {p: checks.sha256_file(p) for p in CODE if os.path.exists(p)},
        "counts": {"trajectories": n_traj, "m0_sequences": m0_seq, "chains": kinds["first_unchanged"],
                   "first_chunks_unchanged": kinds["first_unchanged"], "later_parents": m0_seq - kinds["first_unchanged"],
                   "parents_split": split, "candidate_sequences": h1_seq, "modified": kinds["modified"],
                   "exceptions": kinds["exception_orphan_start_kept"], "opener_insertions": report["5"]["opener_insertions"],
                   "supervised_targets": report["1"]["candidate"]["labelled"], "update_groups": report["2"]["updates"]},
        "tokens": {"m0": m0_tok, "candidate": h1_tok, "ratio": "%.6f" % (h1_tok / m0_tok), "opener_tokens": opener_tok,
                   "ratio_without_opener": "%.6f" % ((h1_tok - opener_tok) / m0_tok),
                   "max_sequence_len": report["4"]["max_len"], "m0q_tokens_per_s": round(rate, 1),
                   "candidate_train_hours_at_m0q_rate": round(h1_tok / rate / 3600, 2)},
        "expected_codex_vs_observed": {k: {"codex": CODEX[k], "observed": observed[k]} for k in CODEX},
        "exceptions": report["3"]["exception_list"],
        "restored_history": {k: v for k, v in report["6"].items() if k.startswith(("restored", "histogram"))},
        "checks": {n: {k: v for k, v in r.items() if k != "exception_list" and not k.startswith(("restored", "histogram"))}
                   for n, r in report.items()},
        "predicted_output_sha256": predicted,
    }


def run(out, rules=RULES):
    t0, started = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mode = "write" if out else "dry-run"
    if out:
        os.makedirs(out)
    else:
        os.makedirs(DRY_DIR, exist_ok=True)
    log = open(os.path.join(out, "write.log") if out else os.path.join(DRY_DIR, "dry-run.log"), "w", encoding="utf-8")

    def say(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    say("start %s mode=%s pid=%d nice=%d cpus=%d" % (started, mode, os.getpid(), os.nice(0), os.cpu_count()))
    sinks = {n: open(os.path.join(out, n), "x", encoding="utf-8") for n in ("H1.jsonl", "parent_ledger.jsonl")} if out else None
    try:
        shas, report, groups_text, predicted, kinds, split, sizes = pack_all(rules, say, sinks)
        stats = make_stats(mode, started, rules, shas, report, predicted, kinds, split, sizes)
        passed = all(r["status"] == "PASS" for r in report.values())
        if out:
            for f in sinks.values():
                f.close()
            with open(os.path.join(out, "update_groups.json"), "x", encoding="utf-8") as f:
                f.write(groups_text)
            written = {n: checks.sha256_file(os.path.join(out, n)) for n in predicted}
            passed = passed and written == predicted
            stats["written_sha256"] = written
        stats.update(all_checks_pass=passed, wall_seconds=round(time.time() - t0, 1),
                     peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1), processes=1)
        checks.print_report(report)
        checks.print_report(report, out=log)
        if out and passed:
            with open(os.path.join(out, "manifest.json"), "x", encoding="utf-8") as f:
                json.dump(dict(stats, status="CANDIDATE_UNAPPROVED_NO_TRAINING"), f, indent=1)
        elif not out:
            with open(os.path.join(DRY_DIR, "dry-run-stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=1)
        say("done: all checks %s, %s -> %s sequences, ratio %s, wall %.1f s, peak RSS %.1f MiB" % (
            "PASS" if passed else "FAIL (no manifest.json)" if out else "FAIL", stats["counts"]["m0_sequences"],
            stats["counts"]["candidate_sequences"], stats["tokens"]["ratio"], stats["wall_seconds"],
            stats["peak_rss_mib"]))
        return 0 if passed else 1
    except Exception:
        say("ERROR\n" + traceback.format_exc())
        raise
    finally:
        log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="statistics only, no packed file")
    mode.add_argument("--write", action="store_true", help="candidate files into a fresh --out (after the user's go)")
    ap.add_argument("--out", help="fresh directory for --write")
    opts = ap.parse_args(argv)
    if opts.write and not opts.out:
        ap.error("--write needs --out")
    if opts.dry_run and opts.out:
        ap.error("--dry-run writes no packed file; --out belongs to --write")
    return run(check_out(opts.out) if opts.write else None)


if __name__ == "__main__":
    sys.exit(main())
