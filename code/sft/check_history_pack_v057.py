#!/usr/bin/env python3
"""Six data-contract checks of an Hq1 history pack (boss v057) against the M0 data that the M0q run trained on. CPU only.

Reference: sft/data/mask_control_v035/M0.jsonl + update_groups.json (pinned by contracts/mask-control-v0351-APPROVED.json).
Candidate: H1.jsonl + parent_ledger.jsonl + update_groups.json written by build_history_pack_v057.py --write, or the
in-memory candidate of its --dry-run. Every structure is re-derived from token ids and labels; the ledger is trusted
only for which M0 chunk a candidate sequence came from, and all its other fields are compared with re-derived values.
  1 supervised set    sha256 of the sorted [traj_id, step, token_id] list over all labelled positions (compact JSON) and
                      the count (4,391,649); each step's labelled ids in their order; the supervised ORDER of training
                      (updates, their sequences, steps and tokens, all in order); per update the targets and the
                      denominator, equal to M0's update_groups.json pinned by the approved M0 contract
  2 update groups     the same 208 updates and denominators; each M0 parent replaced by its sub-chunks (parent order, then
                      sub-chunk order); group targets equal the denominator; M0 groups equal the v035 seeded shuffle
  3 context superset  candidate = header + inserted + parent body prefix, inserted = nothing (unchanged first chunk or
                      exception) or opener + whole steps ending at completion_{k-1}; rule S compliance: an exception
                      only when opener + completion_{k-1} cannot fit, greedy items, maximal history
  4 length            every sequence <= 16,384 tokens including the +3 opener; ids and labels aligned
  5 opener            ids 100, 45518, 107 with labels -100 right after the header's <|turn>model line of every modified
                      sub-chunk; no inserted token other than the opener and M0 chain history
  6 ledger            parent mapping (sub-chunks partition each parent's items in order), restored steps and tokens,
                      per-sequence digests equal to the re-derived values; restored-history histogram
Usage: check_history_pack_v057.py --candidate DIR [--reference DIR] [--json-out FILE]  (exit 0 only if all six PASS)"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import random
import sys

import numpy as np

sys.dont_write_bytecode = True

ROOT = "PROJECT_ROOT/gemma4"
REFERENCE_DIR = ROOT + "/sft/data/mask_control_v035"
M0_CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"
M0_CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
M0_SEED, M0_GROUP_SIZE = 20261103, 8  # prepare_mask_control_v035.make_groups
EXPECTED_TARGETS, EXPECTED_GROUPS = 4391649, 208
NAMES = {1: "supervised set identical", 2: "update groups and schedule identical", 3: "context superset (rule S)",
         4: "length <= max_len incl. opener", 5: "opener at every modified start, nothing else new",
         6: "parent-mapping ledger and restored-history histogram"}
KEEP = 40  # failure messages kept per check
STEP_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, None)]
TOKEN_BINS = [(0, 0), (1, 999), (1000, 1999), (2000, 3999), (4000, 7999), (8000, None)]


def digest(value):
    """prepare_mask_control_v035.digest: sha256 of the compact JSON of value."""
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def json_list_sha256(parts):
    """sha256 of '[' + ','.join(parts) + ']' (parts = compact JSON of each element), i.e. digest(elements) streamed."""
    h = hashlib.sha256(b"[")
    for n, p in enumerate(parts):
        h.update(("," + p if n else p).encode())
    h.update(b"]")
    return h.hexdigest()


def supervised_digests(steps_by_traj):
    """(sorted-triple digest, ordered digest, count) of {traj_id: {step: labelled ids in order}}."""
    def triples():
        for tid in sorted(steps_by_traj):
            pre = "[" + json.dumps(tid) + ","
            for step in sorted(steps_by_traj[tid]):
                for tok in np.sort(steps_by_traj[tid][step]).tolist():
                    yield "%s%d,%d]" % (pre, step, tok)

    def ordered():
        for tid in sorted(steps_by_traj):
            pre = "[" + json.dumps(tid) + ","
            for step in sorted(steps_by_traj[tid]):
                yield "%s%d,[%s]]" % (pre, step, ",".join(map(str, steps_by_traj[tid][step].tolist())))
    count = sum(int(a.size) for s in steps_by_traj.values() for a in s.values())
    return json_list_sha256(triples()), json_list_sha256(ordered()), count


def training_order_sha256(groups, seqs):
    """sha256 of the supervised stream in training order: [traj_id, step, [ids]] for every step of every sequence of
    every update, in that order; seqs[x] = [(traj_id, step, ids), ...] of sequence x."""
    def items():
        for g in groups:
            for x in g.get("sequence_indices") or []:
                if isinstance(x, int) and 0 <= x < len(seqs):
                    for tid, step, ids in seqs[x]:
                        yield "[%s,%d,[%s]]" % (json.dumps(tid), step, ",".join(map(str, ids.tolist())))
    return json_list_sha256(items())


def runs(labels, start=0):
    """Maximal runs of labels[start:] as (supervised, a, b); supervised means labels != -100."""
    sup = labels[start:] != -100
    if not sup.size:
        return []
    edges = [0] + (np.flatnonzero(sup[1:] != sup[:-1]) + 1).tolist() + [int(sup.size)]
    return [(bool(sup[a]), start + a, start + b) for a, b in zip(edges[:-1], edges[1:])]


def canonical_groups(n):
    """Sequence indices of prepare_mask_control_v035.make_groups for n sequences."""
    order = list(range(n))
    random.Random(M0_SEED).shuffle(order)
    return [order[s:s + M0_GROUP_SIZE] for s in range(0, n, M0_GROUP_SIZE)]


def parse_reference(rec):
    """Chains of one M0 record (header, stream = completion_0, X_1, completion_1, ..., completion bounds in the stream,
    step -> item) and its chunks with their item ranges k..m."""
    chains, chunks = {}, []
    for c in rec["chunks"]:
        ids = np.asarray(c["input_ids"], dtype=np.int64)
        lab = np.asarray(c["labels"], dtype=np.int64)
        first = c["chain"] not in chains
        if first:
            h = int(np.argmax(lab != -100))
            chains[c["chain"]] = {"H": h, "header": ids[:h], "parts": [], "bounds": [], "step_item": {}, "pos": 0}
        ch = chains[c["chain"]]
        h, k = ch["H"], len(ch["bounds"])
        rr = runs(lab, h)
        if (len(ids) != len(lab) or not np.array_equal(ids[:h], ch["header"]) or (lab[:h] != -100).any()
                or not rr or not rr[-1][0] or rr[0][0] != first):
            raise ValueError("%s: reference chunk is not header + X/completion runs" % rec["traj_id"])
        for sup, a, b in rr:
            if sup and not np.array_equal(lab[a:b], ids[a:b]):
                raise ValueError("%s: reference labels differ from ids" % rec["traj_id"])
            ch["parts"].append(ids[a:b])
            if sup:
                ch["bounds"].append((ch["pos"], ch["pos"] + b - a))
            ch["pos"] += b - a
        m = len(ch["bounds"]) - 1
        s0, s1 = c["steps"]
        if s1 - s0 != m - k:
            raise ValueError("%s: reference steps do not match the completions of a chunk" % rec["traj_id"])
        ch["step_item"].update((s0 + t - k, t) for t in range(k, m + 1))
        chunks.append({"chain": c["chain"], "k": k, "m": m, "first": first, "ids": ids, "labels": lab,
                       "steps": c["steps"]})
    for ch in chains.values():
        ch["stream"] = np.concatenate(ch.pop("parts"))
        ch["start_item"] = {a: t for t, (a, _) in enumerate(ch["bounds"])}
    return chains, chunks


def summary(values):
    if not values:
        return {"n": 0}
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": int(v.size), "min": float(v.min()), "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "max": float(v.max()), "mean": round(float(v.mean()), 3)}


def histogram(values, bins):
    out = collections.OrderedDict()
    for lo, hi in bins:
        key = str(lo) if hi == lo else ("%d+" % lo if hi is None else "%d-%d" % (lo, hi))
        out[key] = sum(1 for v in values if v >= lo and (hi is None or v <= hi))
    return out


class Checker:
    """Accumulates the six checks over (reference record, candidate record, ledger rows), one trajectory at a time."""

    def __init__(self, rules, expected=(EXPECTED_TARGETS, EXPECTED_GROUPS)):
        self.rules, self.expected = rules, expected  # expected (targets, groups): the frozen M0 population
        self.examples = {n: [] for n in NAMES}
        self.failures = dict.fromkeys(NAMES, 0)
        self.ref_steps, self.cand_steps = {}, {}
        self.ref_seq, self.cand_seq = [], []  # per M0 / candidate sequence: [(traj_id, step, ids), ...] in order
        self.ref_targets, self.cand_targets = [], []  # shifted targets per M0 / candidate sequence
        self.children = {}  # M0 sequence index -> candidate sequence indices, in order
        self.kinds, self.reasons = collections.Counter(), collections.Counter()
        self.exceptions, self.openers, self.max_seen, self.tokens = [], 0, 0, [0, 0]
        self.seq_steps, self.seq_tokens, self.completion_steps = [], [], []

    def bad(self, n, message):
        self.failures[n] += 1
        if len(self.examples[n]) < KEEP:
            self.examples[n].append(message)

    def add(self, ref, cand, rows):
        tid, m0_base, h1_base = ref["traj_id"], len(self.ref_targets), len(self.cand_targets)
        if (cand.get("traj_id") != tid or cand.get("source_line") != ref["source_line"]
                or len(rows) != len(cand["chunks"]) or tid in self.ref_steps):
            raise ValueError("%s: candidate record or ledger rows do not align with the reference" % tid)
        chains, chunks = parse_reference(ref)
        steps = self.ref_steps[tid] = {}
        for p in chunks:
            ch, order = chains[p["chain"]], []
            for t in range(p["k"], p["m"] + 1):
                step = p["steps"][0] + t - p["k"]
                steps[step] = ch["stream"][slice(*ch["bounds"][t])].astype(np.int32)
                order.append((tid, step, steps[step]))
            self.ref_seq.append(order)
            self.ref_targets.append(int((p["labels"][1:] != -100).sum()))
            self.tokens[0] += len(p["ids"])
        csteps, parts, last = self.cand_steps.setdefault(tid, {}), {}, -1
        for idx, (c, row) in enumerate(zip(cand["chunks"], rows)):
            h = h1_base + idx
            ids = np.asarray(c["input_ids"], dtype=np.int64)
            lab = np.asarray(c["labels"], dtype=np.int64)
            self.cand_targets.append(int((lab[1:] != -100).sum()))
            self.tokens[1] += len(ids)
            self.max_seen = max(self.max_seen, len(ids))
            where = "%s seq %d" % (tid, h)
            s0 = (c.get("steps") or [0])[0]
            sup = [(tid, (s0 if isinstance(s0, int) else 0) + n, lab[a:b].astype(np.int32))
                   for n, (_, a, b) in enumerate(r for r in runs(lab) if r[0])]
            self.cand_seq.append(sup)
            if (not len(ids) == len(lab) <= self.rules["max_len"] or c.get("n_tokens") != len(ids)
                    or c.get("n_loss_tokens") != int((lab != -100).sum()) or not ((lab == -100) | (lab == ids)).all()):
                self.bad(4, where + ": length %d or id/label alignment" % len(ids))
            pc = row.get("parent_chunk_index")
            if not isinstance(pc, int) or not last <= pc < len(chunks):
                self.bad(6, where + ": parent chunk index %r" % (pc,))
                continue
            last = pc
            self.children.setdefault(m0_base + pc, []).append(h)
            got = self.derive(chains[chunks[pc]["chain"]], chunks[pc], c, ids, lab, where, csteps, sup)
            if got is None:
                continue
            parts.setdefault(pc, []).append(got["items"])
            got.update(sequence_index=h, traj_id=tid, source_line=ref["source_line"], chunk_index=idx,
                       chain=c.get("chain"), steps=c.get("steps"), parent_sequence_index=m0_base + pc,
                       parent_chunk_index=pc, sub_index=len(parts[pc]) - 1, n_tokens=len(ids),
                       shifted_targets=self.cand_targets[-1], input_sha256=digest(c["input_ids"]),
                       labels_sha256=digest(c["labels"]))
            diff = sorted(f for f in got if row.get(f) != got[f])
            if diff:
                self.bad(6, where + ": ledger fields differ from re-derived values: " + ",".join(diff))
            self.record(got)
        for pc, p in enumerate(chunks):
            seq = parts.get(pc)
            if not (seq and seq[0][0] == p["k"] and seq[-1][1] == p["m"]
                    and all(b[0] == a[1] + 1 for a, b in zip(seq, seq[1:]))):
                self.bad(6, "%s: candidate sequences do not partition the items of M0 chunk %d" % (tid, pc))

    def derive(self, ch, p, c, ids, lab, where, csteps, sup):
        """Decompose one candidate sequence against its parent (checks 1, 3, 5); returns the derived ledger values.
        sup = the sequence's own supervised runs as (traj_id, step, ids), steps counted from its steps field."""
        rules, limit = self.rules, self.rules["max_len"]
        opener, o = np.asarray(rules["opener_ids"]), len(rules["opener_ids"])
        h, k, m, bounds, stream = ch["H"], p["k"], p["m"], ch["bounds"], ch["stream"]
        s = c.get("steps") or [None, None]
        i, j = ch["step_item"].get(s[0]), ch["step_item"].get(s[1])
        if c.get("chain") != p["chain"] or i is None or j is None or not k <= i <= j <= m or s[1] - s[0] != j - i:
            self.bad(3, where + ": steps %r are not items of the parent chunk" % (c.get("steps"),))
            return None
        p0 = bounds[k - 1][1] if k else 0
        body = stream[p0:bounds[j][1]]
        ins = len(ids) - h - len(body)
        if ins < 0 or not np.array_equal(ids[:h], ch["header"]) or not np.array_equal(ids[h + ins:], body):
            self.bad(3, where + ": not header + inserted tokens + parent body prefix")
            self.bad(5, where + ": tokens that are not M0 chain content")
            return None
        expected, shift = np.full(len(ids), -100, dtype=np.int64), h + ins - p0
        for t in range(i, j + 1):
            a, b = bounds[t]
            expected[shift + a:shift + b] = stream[a:b]
        if not np.array_equal(lab, expected):
            self.bad(1, where + ": labels are not exactly completions %d..%d" % (i, j))
        if len(sup) != j - i + 1:
            self.bad(1, where + ": %d supervised runs for %d steps" % (len(sup), j - i + 1))
        for _, step, toks in sup:
            if step in csteps:
                self.bad(1, where + ": step %d supervised twice" % step)
            csteps[step] = toks
        got = {"items": [i, j], "parent_items": [k, m], "restored_from_item": None, "restored_steps": 0,
               "restored_tokens": 0, "opener_tokens": 0, "exception_reason": None}
        prev = bounds[k - 1][1] - bounds[k - 1][0] if k else 0
        if ins == 0:
            if p["first"]:
                got["kind"] = "first_unchanged"
                if (i, j) != (0, m):
                    self.bad(3, where + ": first chunk of a chain was split")
            else:
                need = h + o + prev + bounds[i][1] - p0
                got["kind"] = "exception_orphan_start_kept"
                got["exception_reason"] = "no_room_for_completion_k_minus_1" if need > limit else "unjustified"
                if need <= limit:
                    self.bad(3, where + ": exception although opener + completion_{k-1} fit (%d)" % need)
            return got
        got["kind"] = "modified"
        end = rules["header_end_ids"]
        if (p["first"] or ins <= o or not np.array_equal(ids[h:h + o], opener) or (lab[h:h + o] != -100).any()
                or ch["header"][-len(end):].tolist() != list(end)):
            self.bad(5, where + ": inserted tokens do not start with the opener after the <|turn>model line")
            return got
        n_hist = ins - o
        a = ch["start_item"].get(p0 - n_hist)
        if a is None or not 1 <= a <= k - 1 or not np.array_equal(ids[h + o:h + ins], stream[p0 - n_hist:p0]):
            self.bad(3, where + ": inserted history is not whole steps ending at completion_{k-1}")
            self.bad(5, where + ": inserted tokens other than the opener and M0 chain history")
            return got
        if len(ids) + bounds[a][0] - bounds[a - 1][0] <= limit:
            self.bad(3, where + ": history not maximal (completion %d and X_%d would fit)" % (a - 1, a))
        if j < m and h + o + prev + bounds[j + 1][1] - p0 <= limit:
            self.bad(3, where + ": items not greedy (item %d would fit)" % (j + 1))
        got.update(restored_from_item=a, restored_steps=k - a, restored_tokens=n_hist, opener_tokens=o)
        return got

    def record(self, got):
        self.kinds[got["kind"]] += 1
        self.openers += int(got["opener_tokens"] > 0)
        self.seq_steps.append(got["restored_steps"])
        self.seq_tokens.append(got["restored_tokens"])
        if got["kind"] == "modified":
            self.completion_steps += [got["restored_steps"]] * (got["items"][1] - got["items"][0] + 1)
        if got["exception_reason"]:
            self.reasons[got["exception_reason"]] += 1
            self.exceptions.append({f: got[f] for f in ("sequence_index", "traj_id", "parent_sequence_index",
                                                        "items", "parent_items", "exception_reason")})

    def finish(self, ref_groups, cand_groups):
        """Checks 1 and 2 over the whole pack and the report of all six checks."""
        ref, cand = supervised_digests(self.ref_steps), supervised_digests(self.cand_steps)
        shifted = [sum(self.ref_targets), sum(self.cand_targets)]
        if ref != cand or shifted[0] != shifted[1] or ref[2] != shifted[0] or ref[2] != self.expected[0]:
            self.bad(1, "supervised digests or counts differ (reference %s / %d, candidate %s / %d)"
                     % (ref[0][:12], ref[2], cand[0][:12], cand[2]))
        n_cand = len(self.cand_targets)
        order = [training_order_sha256(ref_groups, self.ref_seq), training_order_sha256(cand_groups, self.cand_seq)]
        if order[0] != order[1]:
            self.bad(1, "supervised order in training differs (reference %s, candidate %s)" % (order[0][:12], order[1][:12]))
        aggregates = [[[g.get("update"), sum(self.ref_targets[p] for p in g["sequence_indices"]),
                        g.get("original_denominator")] for g in ref_groups],
                      [[g.get("update"), sum(self.cand_targets[x] for x in g.get("sequence_indices") or []
                                             if isinstance(x, int) and 0 <= x < n_cand),
                        g.get("original_denominator")] for g in cand_groups]]
        if aggregates[0] != aggregates[1] or any(t != d for _, t, d in aggregates[0]):
            self.bad(1, "per-update targets or denominators differ from M0's")
        if [g.get("sequence_indices") for g in ref_groups] != canonical_groups(len(self.ref_targets)):
            self.bad(2, "reference groups are not the v035 seeded shuffle of the M0 sequences")
        if len(ref_groups) != self.expected[1] or len(cand_groups) != len(ref_groups):
            self.bad(2, "group counts: reference %d, candidate %d, expected %d"
                     % (len(ref_groups), len(cand_groups), self.expected[1]))
        used = []
        for g, cg in zip(ref_groups, cand_groups):
            want = [x for p in g["sequence_indices"] for x in self.children.get(p, [])]
            idx = cg.get("sequence_indices") or []
            same = all(cg.get(f) == g.get(f) for f in ("update", "original_denominator", "m1_remaining_targets"))
            if not same or cg.get("parent_sequence_indices") != g["sequence_indices"] or idx != want:
                self.bad(2, "update %s: membership, order or denominator differs" % g.get("update"))
            got = sum(self.cand_targets[x] for x in idx if isinstance(x, int) and 0 <= x < n_cand)
            if got != g["original_denominator"] or sum(self.ref_targets[p] for p in g["sequence_indices"]) != got:
                self.bad(2, "update %s: candidate targets %d, denominator %d" % (g.get("update"), got,
                                                                                g["original_denominator"]))
            used += idx
        if sorted(used) != list(range(n_cand)):
            self.bad(2, "candidate sequences are not used exactly once by the groups")
        if self.openers != self.kinds["modified"]:
            self.bad(5, "opener insertions %d != modified sequences %d" % (self.openers, self.kinds["modified"]))
        report = {str(n): {"name": NAMES[n], "status": "FAIL" if self.failures[n] else "PASS",
                           "failures": self.failures[n], "examples": self.examples[n]} for n in NAMES}
        report["1"].update(reference={"sha256_sorted_triples": ref[0], "sha256_ordered_steps": ref[1],
                                      "sha256_training_order": order[0], "sha256_update_aggregates": digest(aggregates[0]),
                                      "labelled": ref[2], "shifted": shifted[0]},
                           candidate={"sha256_sorted_triples": cand[0], "sha256_ordered_steps": cand[1],
                                      "sha256_training_order": order[1], "sha256_update_aggregates": digest(aggregates[1]),
                                      "labelled": cand[2], "shifted": shifted[1]}, expected=self.expected[0])
        report["2"].update(updates=len(cand_groups), reference_updates=len(ref_groups),
                           warmup_updates=max(1, round(0.05 * len(cand_groups))), candidate_sequences=n_cand,
                           reference_sequences=len(self.ref_targets))
        report["3"].update(kinds=dict(self.kinds), exceptions=len(self.exceptions), reason_classes=dict(self.reasons),
                           exception_list=self.exceptions)
        report["4"].update(max_len=self.max_seen, limit=self.rules["max_len"], tokens_reference=self.tokens[0],
                           tokens_candidate=self.tokens[1])
        report["5"].update(opener_insertions=self.openers, modified=self.kinds["modified"],
                           opener_ids=self.rules["opener_ids"])
        report["6"].update(
            restored_steps_per_modified_sequence=summary([v for v, t in zip(self.seq_steps, self.seq_tokens) if t]),
            restored_tokens_per_modified_sequence=summary([t for t in self.seq_tokens if t]),
            restored_steps_per_completion_in_modified=summary(self.completion_steps),
            histogram_restored_steps_all_sequences=histogram(self.seq_steps, STEP_BINS),
            histogram_restored_tokens_all_sequences=histogram(self.seq_tokens, TOKEN_BINS),
            histogram_restored_steps_per_completion=histogram(self.completion_steps, STEP_BINS))
        return report


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def print_report(report, out=sys.stdout):
    brief = {"1": (), "2": ("updates", "candidate_sequences"), "3": ("exceptions", "reason_classes"),
             "4": ("max_len", "limit"), "5": ("opener_insertions", "modified"), "6": ()}
    for n, r in report.items():
        extra = {k: r[k] for k in brief[n]}
        if n == "1":
            extra = {"labelled": r["candidate"]["labelled"], "sha256": r["candidate"]["sha256_sorted_triples"][:16]}
        print("CHECK %s %-52s %s failures=%d %s" % (n, r["name"], r["status"], r["failures"],
                                                    json.dumps(extra, sort_keys=True)), file=out, flush=True)


def check_streams(ref_records, cand_records, ledger_rows, ref_groups, cand_groups, rules, **kw):
    """All six checks over record iterables (reference and candidate in the same trajectory order)."""
    checker, ledger = Checker(rules, **kw), iter(ledger_rows)
    for ref, cand in itertools.zip_longest(ref_records, cand_records):
        if ref is None or cand is None:
            raise ValueError("reference and candidate record counts differ")
        checker.add(ref, cand, list(itertools.islice(ledger, len(cand["chunks"]))))
    if next(ledger, None) is not None:
        raise ValueError("ledger has more rows than candidate sequences")
    return checker.finish(ref_groups, cand_groups)


def check_dirs(reference, candidate, rules):
    with open(os.path.join(reference, "update_groups.json")) as f:
        ref_groups = json.load(f)
    with open(os.path.join(candidate, "update_groups.json")) as f:
        cand_groups = json.load(f)
    return check_streams(read_jsonl(os.path.join(reference, "M0.jsonl")), read_jsonl(os.path.join(candidate, "H1.jsonl")),
                         read_jsonl(os.path.join(candidate, "parent_ledger.jsonl")), ref_groups, cand_groups, rules)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="directory with H1.jsonl, parent_ledger.jsonl, update_groups.json")
    ap.add_argument("--reference", default=REFERENCE_DIR, help="directory with M0.jsonl and update_groups.json")
    ap.add_argument("--json-out", help="write the full report here (fresh file)")
    opts = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_history_pack_v057 import RULES
    report = check_dirs(opts.reference, opts.candidate, RULES)
    report["files_sha256"] = {os.path.join(d, n): sha256_file(os.path.join(d, n)) for d, names in (
        (opts.reference, ("M0.jsonl", "update_groups.json")),
        (opts.candidate, ("H1.jsonl", "parent_ledger.jsonl", "update_groups.json"))) for n in names}
    with open(M0_CONTRACT) as f:
        pins = json.load(f)["artifact_hashes"]
    pinned = sha256_file(M0_CONTRACT) == M0_CONTRACT_SHA and all(
        report["files_sha256"].get(os.path.join(opts.reference, n)) == pins.get(os.path.join(REFERENCE_DIR, n))
        for n in ("M0.jsonl", "update_groups.json"))
    report["1"]["reference_pinned_by_m0_contract"] = pinned
    if not pinned:
        report["1"].update(status="FAIL", failures=report["1"]["failures"] + 1)
        report["1"]["examples"].append("reference files differ from the approved M0 contract's artifact hashes")
    print_report({n: report[n] for n in map(str, NAMES)})
    if opts.json_out:
        with open(opts.json_out, "x") as f:
            json.dump(report, f, indent=1)
    return 0 if all(report[str(n)]["status"] == "PASS" for n in NAMES) else 1


if __name__ == "__main__":
    sys.exit(main())
