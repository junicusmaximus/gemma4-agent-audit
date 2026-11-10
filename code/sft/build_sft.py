#!/usr/bin/env python3
"""Pack Gluzdov LoRA-kit step records into per-trajectory SFT sequences for Gemma 4 E4B.

Own code. Only reads the kit's JSONL files and lora_index.csv; nothing shipped in the kit is run.
Outputs in --out: trajectories.jsonl, arms.json, stats.md.

One chain is one packed sequence: header (prefix of its first step, no loss) + completion_0 (loss)
+ X_1 (no loss) + completion_1 (loss) + ... where X_i is the part of prefix_i that follows the
re-rendered assistant turn i-1 (tool response + generation prompt). Sampled completions are kept
exactly as saved, so earlier turns inside a sequence appear in their sampled form, not in the
re-rendered form found in prefix_i (stats.md counts how often the two differ and how). A step whose
prefix does not extend any open chain (for example an auxiliary call) starts a new chain.
Native steps use the saved token ids; external steps tokenize header / X text with
add_special_tokens=False and keep the saved completion ids.
"""
import argparse
import collections
import csv
import json
import os
import random
import shutil
import time
from multiprocessing import Pool

KIT = os.path.expanduser("~/gemma4/external/gemma-4-serving-and-lora-verification-kit")
TOKENIZER_DIR = os.path.expanduser("~/gemma4/models/gemma-4-e4b-it-qat-w4a16-ct")
FILES = {
    "native": ["lora_native_resolved.jsonl"],
    "external_A": ["lora_external_A.jsonl"],
    "external_B": ["lora_external_B.part%03d.jsonl" % i for i in range(4)],
    "external_B2": ["lora_external_B2.part%03d.jsonl" % i for i in range(4)],
}
TOOL_RESPONSE, TURN_END, CHANNEL_CLOSE = "<|tool_response>", "<turn|>", "<channel|>"
GEN_PROMPT = "<|channel>thought\n"
SEED = 20261103
EXCLUDED_REPOS = ("textualize/rich",)
INDEX = {}


def lcp(a, b):
    """Length of the longest common prefix of two strs or two lists."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def find_end(seq, marker, start):
    """Index just after the first marker at or after start (str: substring, list: element); -1 if absent."""
    if isinstance(seq, str):
        k = seq.find(marker, start)
        return -1 if k < 0 else k + len(marker)
    try:
        return seq.index(marker, start) + 1
    except ValueError:
        return -1


def tail_after_last(seq, marker):
    """Part of seq after its last marker (the whole seq if the marker is absent)."""
    if isinstance(seq, str):
        k = seq.rfind(marker)
        return seq if k < 0 else seq[k + len(marker):]
    k = len(seq) - 1
    while k >= 0 and seq[k] != marker:
        k -= 1
    return seq if k < 0 else seq[k + 1:]


def join_end(p0, g0, p1, end_marker, close_marker):
    """Index in p1 where the re-rendered assistant turn g0 (sampled after p0) ends, or -1 if p1
    does not contain g0's tail (after its last close_marker) right before the first end_marker."""
    e = find_end(p1, end_marker, lcp(p0, p1))
    if e < 0:
        return -1
    tail = tail_after_last(g0, close_marker)
    seg = p1[max(0, e - len(tail)):e]
    # equal, or equal as a multiset: the chat template re-orders tool-call argument keys
    return e if seg == tail or collections.Counter(seg) == collections.Counter(tail) else -1


def loose_end(p0, g0, p1, end_marker):
    """Text-only fallback when the tool-call arguments were re-rendered, not just re-ordered: accept the end of the
    re-rendered turn if p1 extends p0 (minus a trailing generation prompt) and the turn has a similar length."""
    stable = p0[:-len(GEN_PROMPT)] if p0.endswith(GEN_PROMPT) and not p1.startswith(p0) else p0
    if not p1.startswith(stable):
        return -1
    e = find_end(p1, end_marker, len(stable))
    return e if e >= 0 and abs(e - len(stable) - len(g0)) <= max(64, len(g0) // 2) else -1


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def variant(p0, g0, p1, e):
    """How the re-rendered assistant turn in p1 relates to the sampled completion g0."""
    if p1.startswith(p0 + g0):
        return "exact"
    k = g0.rfind(CHANNEL_CLOSE)
    if k >= 0 and p1[:e] == p0 + g0[:k] + "\n" + g0[k:]:
        return "newline_before_channel_close"
    base = p0[:-len(GEN_PROMPT)] if p0.endswith(GEN_PROMPT) else p0
    if p1[:e] == base + tail_after_last(g0, CHANNEL_CLOSE):
        return "empty_thought_dropped"
    tail = tail_after_last(g0, CHANNEL_CLOSE)
    same = collections.Counter(p1[max(0, e - len(tail)):e]) == collections.Counter(tail)
    return "args_reordered" if same else "tool_call_rewritten"


def try_join(chain, p1, pi, tok, tid):
    """Extend chain by the step whose prefix is p1 (token ids pi or None).
    Returns (x_text, x_ids, variant, ids_check) or None when p1 does not continue the chain."""
    if not p1.startswith(chain["header"]):
        return None
    p0, g0, pi0, ci0 = chain["last"]
    marker = TOOL_RESPONSE if g0.endswith(TOOL_RESPONSE) else (TURN_END if g0.endswith(TURN_END) else None)
    if marker is None:
        return None
    e = join_end(p0, g0, p1, marker, CHANNEL_CLOSE)
    if e < 0:
        e = loose_end(p0, g0, p1, marker)
    if e < 0:
        return None
    x_text, x_ids, check = p1[e:], None, None
    if pi is not None and pi0 is not None:
        et = join_end(pi0, ci0, pi, tid[marker], tid[CHANNEL_CLOSE])
        if et >= 0:
            x_ids = pi[et:]
            check = encode(tok, x_text) == x_ids
    if x_ids is None:
        x_ids = encode(tok, x_text)
    return x_text, x_ids, variant(p0, g0, p1, e), check


def build_chains(exs, tok, tid):
    """Group the steps of one trajectory into chains; returns (chains, outcome counter)."""
    chains, out, prev = [], collections.Counter(), None
    for i, ex in enumerate(exs):
        p1, g1, pi = ex["prefix"], ex["completion"], ex.get("prompt_token_ids")
        ci = ex.get("completion_token_ids")
        if ci is None:
            ci = encode(tok, g1)
        hit, ch = None, None
        for cand in ([prev] if prev else []) + [c for c in reversed(chains) if c is not prev]:
            hit = try_join(cand, p1, pi, tok, tid)
            if hit:
                ch = cand
                break
        if hit is None:
            ch = {"header": p1, "items": [], "last": None}
            chains.append(ch)
            pre = (p1, pi if pi is not None else encode(tok, p1))
            out["start" if i == 0 else "break"] += 1
        else:
            pre = hit[:2]
            out["relink" if ch is not prev else hit[2]] += 1
            if hit[3] is not None:
                out["xcheck_ok" if hit[3] else "xcheck_bad"] += 1
        ch["items"].append({"step": i, "pre_text": pre[0], "pre_ids": pre[1], "comp_text": g1, "comp_ids": ci})
        ch["last"] = (p1, g1, pi, ci)
        prev = ch
    return chains, out


def chunk_chain(items, max_len):
    """Greedy split at step boundaries; every chunk keeps the chain header (prefix of its first step).
    A chunk that starts at step k > 0 is header + X_k + completion_k + ... Returns (chunks, dropped steps)."""
    header = items[0]["pre_ids"]
    chunks, dropped, segs, n, first, last = [], [], None, 0, 0, 0
    for k, it in enumerate(items):
        add = [(it["pre_ids"], False), (it["comp_ids"], True)]
        size = len(add[0][0]) + len(add[1][0])
        if segs is not None and n + size <= max_len:
            segs += add
            n += size
            last = k
            continue
        if segs is not None:
            chunks.append((first, last, segs))
            segs = None
        cand = ([(header, False)] if k else []) + add
        n_cand = sum(len(s) for s, _ in cand)
        if n_cand > max_len:
            dropped.append(k)
            continue
        segs, n, first, last = cand, n_cand, k, k
    if segs is not None:
        chunks.append((first, last, segs))
    return chunks, dropped


def make_chunk(ci, first, last, segs, items):
    ids, labels = [], []
    for s, train in segs:
        ids += s
        labels += s if train else [-100] * len(s)
    return {"chain": ci, "steps": [items[first]["step"], items[last]["step"]], "n_tokens": len(ids),
            "n_loss_tokens": sum(len(s) for s, t in segs if t), "input_ids": ids, "labels": labels}


def read_index():
    with open(os.path.join(KIT, "lora_index.csv"), encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _init(index):
    INDEX.update(index)


def pack_record(rec, tier, task, repo, tok, tid, max_len):
    exs = rec["sft_examples"]
    chains, outcomes = build_chains(exs, tok, tid)
    segs, chunks, c = [], [], collections.Counter()
    total = loss = 0
    for ci, ch in enumerate(chains):
        items = ch["items"]
        for it in items:
            segs.append({"chain": ci, "text": it["pre_text"], "train": False})
            segs.append({"chain": ci, "text": it["comp_text"], "train": True})
            total += len(it["pre_ids"]) + len(it["comp_ids"])
            loss += len(it["comp_ids"])
        got, dropped = chunk_chain(items, max_len)
        chunks += [make_chunk(ci, f, l, s, items) for f, l, s in got]
        c["dropped_items"] += len(dropped)
        c["dropped_loss"] += sum(len(items[k]["comp_ids"]) for k in dropped)
    row = INDEX.get(rec["id"]) or {}
    if tier == "native":
        c["hdr_ok" if encode(tok, exs[0]["prefix"]) == exs[0]["prompt_token_ids"] else "hdr_bad"] += 1
    if row:
        c["idx_rows"] += 1
        c["idx_loss_ok"] += int(row.get("supervised_tokens") == str(loss))
        c["idx_steps_ok"] += int(row.get("steps") == str(len(exs)))
    c.update({"trajectories": 1, "steps": len(exs), "pairs": max(len(exs) - 1, 0), "chains": len(chains),
              "extra_chains": len(chains) - 1, "chunks": len(chunks), "tokens": total, "loss_tokens": loss,
              "chunk_tokens": sum(k["n_tokens"] for k in chunks),
              "chunk_loss": sum(k["n_loss_tokens"] for k in chunks),
              "split_traj": int(len(chunks) > len(chains) or c["dropped_items"] > 0)})
    for k, v in outcomes.items():
        c["out_" + k] += v
    traj = {"traj_id": rec["id"], "tier": tier, "task_id": task, "repository": repo,
            "generating_model": row.get("generating_model") or rec.get("generating_model"),
            "n_steps": len(exs), "tool_calls": rec.get("tool_calls"), "total_tokens": total,
            "loss_tokens": loss, "n_chains": len(chains), "text_segments": segs, "chunks": chunks}
    meta = {k: traj[k] for k in ("traj_id", "tier", "task_id", "tool_calls", "total_tokens", "loss_tokens")}
    meta.update(n_chunks=len(chunks), chunk_tokens=c["chunk_tokens"], chunk_loss=c["chunk_loss"])
    return traj, meta, c


def process_file(job):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tier, fname, out, max_len, limit = job
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tid = {m: tok.convert_tokens_to_ids(m) for m in (TOOL_RESPONSE, TURN_END, CHANNEL_CLOSE)}
    metas, cnt, t0 = [], collections.Counter(), time.time()
    with open(os.path.join(KIT, fname), encoding="utf-8") as f:
        with open(os.path.join(out, "parts", fname), "w", encoding="utf-8") as w:
            for n, line in enumerate(f):
                if limit and n >= limit:
                    break
                rec = json.loads(line)
                if rec.get("resolved") is False:
                    cnt["skipped_unresolved"] += 1
                    continue
                task = rec.get("issue_id") or rec.get("instance") or rec.get("task_id")
                repo = rec.get("repository") or rec.get("repo")
                if (repo or "").lower() in EXCLUDED_REPOS or (task or "").lower().startswith("textualize__rich-"):
                    cnt["excluded_repo"] += 1
                    continue
                traj, meta, c = pack_record(rec, tier, task, repo, tok, tid, max_len)
                w.write(json.dumps(traj, separators=(",", ":")) + "\n")
                metas.append(meta)
                cnt.update(c)
    print("done %s: %d trajectories in %.0fs" % (fname, len(metas), time.time() - t0), flush=True)
    return tier, fname, metas, cnt


def summarize(metas):
    n = max(len(metas), 1)
    return {"trajectories": len(metas), "tasks": len({m["task_id"] for m in metas}),
            "sequences": sum(m["n_chunks"] for m in metas), "tokens": sum(m["total_tokens"] for m in metas),
            "sequence_tokens": sum(m["chunk_tokens"] for m in metas),
            "loss_tokens": sum(m["loss_tokens"] for m in metas),
            "mean_tokens_per_trajectory": round(sum(m["total_tokens"] for m in metas) / n, 1)}


def select_arms(metas):
    """arm_ordinary: random resolved trajectory per task; arm_short: fewest total tokens per task."""
    by_task = collections.defaultdict(list)
    for m in metas:
        by_task[m["task_id"]].append(m)
    rng, ordinary, short = random.Random(SEED), [], []
    for task in sorted(by_task):
        cands = sorted(by_task[task], key=lambda m: m["traj_id"])
        ordinary.append(rng.choice(cands))
        short.append(min(cands, key=lambda m: (m["total_tokens"],
                                               m["tool_calls"] if m["tool_calls"] is not None else 10 ** 9,
                                               m["traj_id"])))
    order = list(ordinary)
    random.Random(SEED + 1).shuffle(order)
    budget, matched, used = sum(m["total_tokens"] for m in short), [], 0
    for m in order:
        if used >= budget:
            break
        matched.append(m)
        used += m["total_tokens"]
    return ordinary, short, matched, by_task


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "n/a"


def main():
    ap = argparse.ArgumentParser(description="Build packed SFT sequences and experiment arms.")
    ap.add_argument("--tiers", default="native,external_A,external_B,external_B2")
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--out", default="~/gemma4/sft/data/")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N records of each file")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--multi-only", action="store_true",
                    help="add the multi-attempt arms to an existing --out; trajectories.jsonl is only read")
    args = ap.parse_args()
    tiers = args.tiers.split(",")
    for t in tiers:
        if t not in FILES:
            raise SystemExit("unknown tier %s; choose from %s" % (t, ",".join(FILES)))
    out = os.path.expanduser(args.out)
    if args.multi_only:
        add_multi_arms(out)
        return
    os.makedirs(os.path.join(out, "parts"), exist_ok=True)
    jobs = [(t, f, out, args.max_seq_len, args.limit) for t in tiers for f in FILES[t]]
    t0 = time.time()
    with Pool(min(args.procs, len(jobs)), initializer=_init, initargs=(read_index(),)) as pool:
        results = pool.map(process_file, jobs, chunksize=1)
    with open(os.path.join(out, "trajectories.jsonl"), "wb") as w:
        for _, fname, _, _ in results:
            part = os.path.join(out, "parts", fname)
            with open(part, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 20)
            os.remove(part)
    os.rmdir(os.path.join(out, "parts"))
    write_outputs(out, args, tiers, results, time.time() - t0)


def write_outputs(out, args, tiers, results, seconds):
    by_tier = collections.defaultdict(collections.Counter)
    tier_metas = collections.defaultdict(list)
    for t, _, ms, cnt in results:
        by_tier[t].update(cnt)
        tier_metas[t] += ms
    metas = [m for t in tiers for m in tier_metas[t]]
    ordinary, short, matched, by_task = select_arms(metas)
    arms = {"arm_ordinary": ordinary, "arm_short": short, "arm_ordinary_tokmatched": matched,
            "all_resolved": metas}
    summary = {k: summarize(v) for k, v in arms.items()}
    differ = sum(1 for a, b in zip(ordinary, short) if a["traj_id"] != b["traj_id"])
    choice = sum(1 for v in by_task.values() if len(v) > 1)
    doc = {"meta": {"seed": SEED, "max_seq_len": args.max_seq_len, "tiers": tiers, "tasks": len(by_task),
                    "tasks_with_more_than_one_candidate": choice,
                    "tasks_where_ordinary_differs_from_short": differ,
                    "arm_ordinary": "one resolved trajectory per task, uniformly at random "
                                    "(random.Random(seed) over sorted task ids, candidates sorted by traj_id)",
                    "arm_short": "per task the resolved trajectory with the fewest total tokens "
                                 "(tie: fewest tool calls, then traj_id)",
                    "arm_ordinary_tokmatched": "arm_ordinary subsampled (random.Random(seed+1) order) until "
                                               "its total tokens reach those of arm_short",
                    "all_resolved": "every kept trajectory of the selected tiers"}, "summary": summary}
    doc.update({k: [m["traj_id"] for m in v] for k, v in arms.items()})
    mlines = apply_multi(doc, by_task, ordinary, short)
    with open(os.path.join(out, "arms.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    tot = collections.Counter()
    for t in tiers:
        tot.update(by_tier[t])
    rows1, rows2 = [], []
    for t in tiers + ["all"]:
        c = tot if t == "all" else by_tier[t]
        ms = metas if t == "all" else tier_metas[t]
        tk = sorted(m["total_tokens"] for m in ms) or [0]
        p = c["pairs"]
        rows1.append([t, p, c["out_exact"], c["out_newline_before_channel_close"], c["out_empty_thought_dropped"],
                      c["out_args_reordered"], c["out_tool_call_rewritten"], c["out_relink"], c["out_break"], pct(c["out_exact"], p),
                      pct(p - c["out_break"], p)])
        rows2.append([t, c["trajectories"], len({m["task_id"] for m in ms}), c["steps"], c["chains"],
                      c["extra_chains"], c["chunks"], c["tokens"], c["chunk_tokens"], c["loss_tokens"],
                      "%d / %d" % (tk[len(tk) // 2], tk[-1]), c["split_traj"], c["dropped_items"], c["dropped_loss"]])
    s_o, s_s, s_m = summary["arm_ordinary"], summary["arm_short"], summary["arm_ordinary_tokmatched"]
    rows3 = [[k, v["trajectories"], v["tasks"], v["sequences"], v["tokens"], v["sequence_tokens"], v["loss_tokens"],
              v["mean_tokens_per_trajectory"]] for k, v in summary.items()]
    L = ["# build_sft stats", "",
         "Kit %s, tokenizer %s, max_seq_len %d, seed %d, %.0f s." % (KIT, TOKENIZER_DIR, args.max_seq_len, SEED, seconds),
         "", "## 1. Prefix-chain check (pairs of consecutive steps)", "",
         "Assumption: prefix_i = prefix_{i-1} + completion_{i-1} + rendered tool result + generation prompt. "
         "exact = holds literally. newline_before_channel_close = the re-rendered history has one extra newline "
         "before the last channel-close marker of the sampled turn. empty_thought_dropped = the re-rendered history "
         "omits the (empty) thought block of the sampled turn. args_reordered = tool-call arguments in another order (the template sorts argument keys). tool_call_rewritten = the tool call differs in content too (joined by a length check only). "
         "relink = continues an earlier chain after an auxiliary step. break = starts a new chain.", "",
         table(["tier", "pairs", "exact", "newline_before_channel_close", "empty_thought_dropped", "args_reordered", "tool_call_rewritten",
                "relink", "break", "exact rate", "joined rate"], rows1), "",
         "Tokenization checks (native, ids known): header encode == saved prompt ids: %d ok / %d bad; "
         "X-text encode == saved id slice: %d ok / %d bad." % (tot["hdr_ok"], tot["hdr_bad"],
                                                              tot["out_xcheck_ok"], tot["out_xcheck_bad"]),
         "", "## 2. Per tier", "",
         table(["tier", "trajectories", "tasks", "steps", "chains", "extra chains", "sequences", "tokens (unsplit)",
                "tokens in sequences", "loss tokens", "median / max tokens", "split trajectories", "dropped steps",
                "dropped loss tokens"], rows2), "",
         "## 3. Per arm", "",
         table(["arm", "trajectories", "tasks", "sequences", "tokens (unsplit)", "tokens in sequences",
                "loss tokens", "mean tokens/traj"], rows3), "",
         "Tasks: %d, with more than one candidate: %d, where arm_ordinary differs from arm_short: %d."
         % (len(by_task), choice, differ), "", "## 4. Token-budget equalization", "",
         "arm_short has %d total / %d loss tokens, arm_ordinary %d / %d (ratio %.2f / %.2f). Options: "
         "(a) arm_ordinary_tokmatched, a random subset of arm_ordinary with %d total / %d loss tokens over %d tasks "
         "(equal compute, fewer tasks); (b) keep the same tasks and give arm_short more optimizer steps or epochs "
         "until the tokens seen match (equal compute, repeated data). Total tokens approximate compute, loss tokens "
         "approximate gradient signal; fix which one is matched before training." % (
             s_s["tokens"], s_s["loss_tokens"], s_o["tokens"], s_o["loss_tokens"],
             s_o["tokens"] / max(s_s["tokens"], 1), s_o["loss_tokens"] / max(s_s["loss_tokens"], 1),
             s_m["tokens"], s_m["loss_tokens"], s_m["tasks"]),
         "", "## 5. Checks", "",
         "- excluded textualize/rich: %d; skipped unresolved: %d; duplicate trajectory ids: %d" % (
             tot["excluded_repo"], tot["skipped_unresolved"], len(metas) - len({m["traj_id"] for m in metas})),
         "- kit index cross-check: loss tokens equal supervised_tokens for %d / %d rows; steps equal for %d / %d"
         % (tot["idx_loss_ok"], tot["idx_rows"], tot["idx_steps_ok"], tot["idx_rows"])]
    L += mlines
    with open(os.path.join(out, "stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote trajectories.jsonl, arms.json, stats.md in %s (%d trajectories)" % (out, len(metas)))


MULTI_DESC = {
    "arm_ordinary_multi": "arm_ordinary restricted to tasks with at least two resolved trajectories (same random picks)",
    "arm_short_multi": "arm_short restricted to the same tasks",
    "arm_short_calls": "per multi-attempt task the trajectory with the fewest tool calls (tie: fewest total tokens, then traj_id)",
}


def load_metas(path):
    """Per-trajectory metadata of an existing trajectories.jsonl (read only)."""
    metas = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            metas.append({"traj_id": t["traj_id"], "tier": t["tier"], "task_id": t["task_id"],
                          "tool_calls": t["tool_calls"], "total_tokens": t["total_tokens"],
                          "loss_tokens": t["loss_tokens"], "n_chunks": len(t["chunks"]),
                          "chunk_tokens": sum(c["n_tokens"] for c in t["chunks"]),
                          "chunk_loss": sum(c["n_loss_tokens"] for c in t["chunks"])})
    return metas


def gap(base, other, field):
    """Per-task base[field] - other[field] over two task-aligned lists: mean over all tasks and over the differing ones."""
    d = [(b[field] or 0) - (o[field] or 0) for b, o in zip(base, other)]
    nz = [x for x in d if x]
    return {"mean_all_tasks": round(sum(d) / max(len(d), 1), 1), "tasks_differing": len(nz),
            "mean_differing_tasks": round(sum(nz) / max(len(nz), 1), 1), "total": sum(d)}


def add_multi_arms(out):
    """Add the multi-attempt arms to an existing arms.json / stats.md; trajectories.jsonl is only read.
    Refuses to touch arms.json unless the existing arms are reproduced exactly from the trajectories."""
    metas = load_metas(os.path.join(out, "trajectories.jsonl"))
    ordinary, short, matched, by_task = select_arms(metas)
    with open(os.path.join(out, "arms.json"), encoding="utf-8") as f:
        doc = json.load(f)
    for key, arm in (("arm_ordinary", ordinary), ("arm_short", short), ("arm_ordinary_tokmatched", matched),
                     ("all_resolved", metas)):
        if doc[key] != [m["traj_id"] for m in arm]:
            raise SystemExit("existing arm %s differs from its recomputation; arms.json left untouched" % key)
    lines = apply_multi(doc, by_task, ordinary, short)
    stats_path = os.path.join(out, "stats.md")
    with open(stats_path, encoding="utf-8") as f:
        text = f.read()
    cut = text.find("## 6. Multi-attempt arms")
    text = (text[:cut] if cut >= 0 else text).rstrip() + "\n" + "\n".join(lines) + "\n"
    for path, content in ((os.path.join(out, "arms.json"), json.dumps(doc, indent=1)), (stats_path, text)):
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(path + ".tmp", path)
    print("added multi-attempt arms to %s (%d tasks)" % (out, len(by_task)))


def apply_multi(doc, by_task, ordinary, short):
    """Add the multi-attempt arms to doc (arms.json content); returns the stats.md lines describing them."""
    multi = sorted(t for t, v in by_task.items() if len(v) > 1)
    keep = set(multi)
    arms = {"arm_ordinary_multi": [m for m in ordinary if m["task_id"] in keep],
            "arm_short_multi": [m for m in short if m["task_id"] in keep],
            "arm_short_calls": [min(by_task[t], key=lambda m: (m["tool_calls"] if m["tool_calls"] is not None else 10 ** 9,
                                                              m["total_tokens"], m["traj_id"])) for t in multi]}
    for ms in arms.values():
        assert [m["task_id"] for m in ms] == multi
    base, summ, rows = arms["arm_ordinary_multi"], {}, []
    for name, ms in arms.items():
        s = summarize(ms)
        s["mean_tool_calls_per_trajectory"] = round(sum(m["tool_calls"] or 0 for m in ms) / max(len(ms), 1), 1)
        s["tier_mix"] = dict(collections.Counter(m["tier"] for m in ms))
        summ[name] = s
        rows.append([name, s["trajectories"], s["sequences"], s["tokens"], s["sequence_tokens"], s["loss_tokens"],
                     s["mean_tokens_per_trajectory"], s["mean_tool_calls_per_trajectory"],
                     ", ".join("%s %d" % kv for kv in sorted(s["tier_mix"].items()))])
    contrast, crows = {}, []
    for name in ("arm_short_multi", "arm_short_calls"):
        c = {"pick_differs_from_ordinary": sum(1 for a, b in zip(base, arms[name]) if a["traj_id"] != b["traj_id"]),
             "total_tokens": gap(base, arms[name], "total_tokens"), "loss_tokens": gap(base, arms[name], "loss_tokens"),
             "tool_calls": gap(base, arms[name], "tool_calls"),
             "token_ratio_ordinary_over_arm": round(summ["arm_ordinary_multi"]["tokens"] / max(summ[name]["tokens"], 1), 3)}
        contrast[name] = c
        crows.append([name, c["pick_differs_from_ordinary"], c["total_tokens"]["mean_all_tasks"],
                      c["total_tokens"]["mean_differing_tasks"], c["token_ratio_ordinary_over_arm"],
                      c["loss_tokens"]["mean_all_tasks"], c["tool_calls"]["mean_all_tasks"],
                      c["tool_calls"]["mean_differing_tasks"]])
    present, within, ntraj, mix = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for t in multi:
        cnt = collections.Counter(m["tier"] for m in by_task[t])
        mix[" + ".join("%s x%d" % (k, cnt[k]) for k in sorted(cnt))] += 1
        for k, n in cnt.items():
            present[k] += 1
            ntraj[k] += n
            within[k] += int(n > 1)
    every = collections.Counter(k for v in by_task.values() for k in {m["tier"] for m in v})
    trows = [[k, every[k], present[k], within[k], ntraj[k]] for k in sorted(every)]
    doc.update({k: [m["traj_id"] for m in v] for k, v in arms.items()})
    doc.update({"multi_attempt_tasks": multi, "multi_contrast": contrast})
    doc["summary"] = {**doc["summary"], **summ}
    doc["meta"].update(MULTI_DESC)
    return ["", "## 6. Multi-attempt arms", "",
            "Tasks with at least two resolved trajectories: %d of %d. arm_ordinary_multi and arm_short_multi are "
            "arm_ordinary and arm_short restricted to these tasks (same random picks). arm_short_calls takes per task the "
            "trajectory with the fewest tool calls (tie: fewest total tokens, then traj_id)." % (len(multi), len(by_task)),
            "", table(["arm", "trajectories", "sequences", "tokens (unsplit)", "tokens in sequences", "loss tokens",
                       "mean tokens/traj", "mean tool calls/traj", "tier mix (trajectories)"], rows), "",
            "Per-task gap, arm_ordinary_multi minus the arm (positive = the arm is cheaper), averaged over all "
            "multi-attempt tasks or only over tasks where the pick differs.", "",
            table(["arm", "tasks with a different pick", "token gap, all tasks", "token gap, differing tasks",
                   "token ratio ordinary/arm", "loss-token gap, all tasks", "tool-call gap, all tasks",
                   "tool-call gap, differing tasks"], crows), "",
            "Tier composition of the multi-attempt task set.", "",
            table(["tier", "tasks in tier (all)", "multi tasks containing the tier",
                   "multi tasks with 2+ attempts inside the tier", "trajectories in multi set"], trows), "",
            "Attempt mixes (tasks): " + "; ".join("%s: %d" % kv for kv in mix.most_common(8)) + "."]


if __name__ == "__main__":
    main()
