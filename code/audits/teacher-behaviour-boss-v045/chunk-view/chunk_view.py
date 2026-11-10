#!/usr/bin/env python3
"""Chunk-level behaviour statistics of the supervised SFT targets (CPU only; the tokenizer only decodes ids).

Decodes every supervised completion (contiguous label run) of the packed chunks and parses it with the functions of
teacher_behaviour.py (imported read-only; it pins tools/mechanism_metrics_v032.py). See README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ anywhere on the server

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent  # analysis/teacher-behaviour-boss-v045: teacher_behaviour.py, report.json, per_trajectory.jsonl
GEMMA4_ROOT = HERE.parents[2]
sys.path.insert(0, str(PARENT))
import teacher_behaviour as tb  # noqa: E402  (read-only import; it verifies the sha256 of the reference module)

MM = tb.MM
DATA_DIR = GEMMA4_ROOT / "sft" / "data"
DEFAULT_CHUNKS = DATA_DIR / "mask_control_v035" / "M0.jsonl"
DEFAULT_SOURCE = DATA_DIR / "trajectories.jsonl"
DEFAULT_TOKENIZER = GEMMA4_ROOT / "models" / "gemma-4-e4b-it-qat-w4a16-ct"
RAW_ARM = "arm_ordinary_tokmatched"  # the arm that M0 packs (manifest.json), for the reconciliation with the raw table
ROLE_IDS = (105, 4368, 107)  # <|turn> model newline: the last three tokens of every chain header
MARKERS = {2: "<bos>", 105: "<|turn>", 106: "<turn|>", 100: "<|channel>", 101: "<channel|>",
           48: "<|tool_call>", 49: "<tool_call|>", 50: "<|tool_response>", 51: "<tool_response|>"}
STOP_IDS = (50, 106)  # <|tool_response> after a call, <turn|> after a final answer or a summary
GROUPS = ("first", "later", "all", "later_orphan", "later_chain_start")
NL = chr(10)
MAX_ERROR_SAMPLES = 5


def label_runs(labels):
    """Contiguous runs of supervised positions (label != -100) as half-open (start, end) pairs."""
    runs, start = [], None
    for index, label in enumerate(labels):
        if label != -100 and start is None:
            start = index
        elif label == -100 and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def header_end(ids):
    """Index just after the first <|turn> model newline (the end of the chain header), or -1 when it is absent."""
    for index in range(len(ids) - len(ROLE_IDS) + 1):
        if ids[index] == ROLE_IDS[0] and tuple(ids[index:index + len(ROLE_IDS)]) == ROLE_IDS:
            return index + len(ROLE_IDS)
    return -1


def chunk_type(steps):
    """The task's rule: 'first' when the chunk starts at step 0 of the trajectory, else 'later'."""
    return "first" if steps[0] == 0 else "later"


def start_kind(gap_length, gap_text):
    """What stands between the chain header and the first supervised token, from the decoded text."""
    if gap_length == 0:
        return "direct"
    if gap_text.startswith(tb.RESPONSE_OPEN):
        return "orphan_response"  # a tool response whose call and earlier history are not in the chunk
    if gap_text.startswith(tb.CHANNEL_OPEN):  # a continuation header (chain > 0) re-serializes recent history after the summary
        return "opener_only" if gap_text == tb.CHANNEL_OPEN else "replayed_history"
    return "other"


def completion_errors(index, ids, start, end, text, calls, call_errors, response_errors):
    """Format problems of one decoded completion (same checks as the raw-trajectory adapter, plus the stop token)."""
    errors = ["completion %d: %s" % (index, message) for message in call_errors + response_errors]
    if len(calls) > 1:
        errors.append("completion %d: %d tool calls (one expected)" % (index, len(calls)))
    if tb.CHANNEL_CLOSE not in text:
        errors.append("completion %d: no channel close marker" % index)
    if ids[end - 1] not in STOP_IDS or any(i in STOP_IDS for i in ids[start:end - 1]):
        errors.append("completion %d: stop token missing at the end or present inside" % index)
    return errors


def analyse_chunk(chunk, decode, expected=None):
    """Row of one packed chunk; 'expected' = text its ids must decode to. Problems go to row['errors'] (status malformed)."""
    ids, labels, steps, errors = chunk["input_ids"], chunk["labels"], chunk["steps"], []
    runs = label_runs(labels)
    if len(runs) != steps[1] - steps[0] + 1:
        errors.append("run_count: %d supervised runs for steps %s" % (len(runs), steps))
    if len(ids) != chunk["n_tokens"] or sum(b - a for a, b in runs) != chunk["n_loss_tokens"]:
        errors.append("token_count: ids or labels disagree with n_tokens / n_loss_tokens")
    if any(labels[a:b] != ids[a:b] for a, b in runs):
        errors.append("label_alignment: supervised labels differ from input_ids")
    first_start, head = (runs[0][0] if runs else len(ids)), header_end(ids)
    if not 0 < head <= first_start:
        errors.append("header: no chain header end before the first supervised token")
        head = first_start
    gap = decode(ids[head:first_start])
    kind = start_kind(first_start - head, gap)
    if kind == "orphan_response":
        try:
            tb.parse_response(gap)
        except ValueError as exc:
            errors.append("orphan response: %s" % exc)
    pieces, pairs, completions = [decode(ids[:head]), gap], [], []
    for index, (start, end) in enumerate(runs):
        text = decode(ids[start:end])
        stop = runs[index + 1][0] if index + 1 < len(runs) else len(ids)
        result = decode(ids[end:stop]) if stop > end else None  # the tool response after this completion, if in the chunk
        calls, call_errors = tb.parse_calls(text)
        observation, response_errors = tb.step_observation(calls, result)
        errors += completion_errors(index, ids, start, end, text, calls, call_errors, response_errors)
        pairs.extend((call, observation) for call in calls)
        completions.append({"n_tokens": end - start, "thought": bool(tb.thought_text(text).strip()), "calls": calls,
                            "controls": dict(sorted(Counter(MARKERS[i] for i in ids[start:end] if i in MARKERS).items()))})
        pieces += [text] if result is None else [text, result]
    text = "".join(pieces)
    diff = None if expected is None else first_difference(text, expected)
    if diff is not None:
        errors.append("roundtrip: decoded chunk differs from the stored segments at character %d" % diff)
    first_calls = completions[0]["calls"] if completions else []
    row = {
        "chain": chunk["chain"], "roundtrip_diff": diff, "steps": steps, "type": chunk_type(steps), "start_kind": kind,
        "n_tokens": len(ids), "n_loss_tokens": sum(b - a for a, b in runs), "n_completions": len(runs), "n_calls": len(pairs),
        "first_action": MM.classify_opener(first_calls[0] if first_calls else None),
        "first_call_tool": first_calls[0]["function_name"] if first_calls else None,
        **MM.repeat_stats([MM.tool_call_identity(call) for call, _ in pairs], MM.DEFAULT_LOOP_THRESHOLD),
        "submit_called": MM.submit_called(pairs), "nonrecovery": MM.nonrecovery(pairs),
        "n_thinking": sum(c["thought"] for c in completions),
        "history_free_actions": int(kind == "orphan_response" and bool(first_calls)),
        "completions": completions, "n_errors": len(errors), "errors": errors[:MAX_ERROR_SAMPLES],
        "status": "malformed" if errors else "ok"}
    return row


def source_offsets(path):
    """({traj_id: byte offset of its line}, sha256 of the file) from one raw pass over trajectories.jsonl."""
    pattern, offsets, digest = re.compile(rb'^[{]"traj_id":"([^"]+)"'), {}, hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            match = pattern.match(line)
            if match:
                offsets[match.group(1).decode()] = handle.tell() - len(line)
    return offsets, digest.hexdigest()


def reference_parts(record):
    """(chain header texts by chain number, [(chain number, completion, result or None)] in global step order)."""
    headers, steps = {}, []
    for chain in tb.group_chains(record["text_segments"]):
        headers[chain[0]["chain"]] = chain[0]["text"]
        steps.extend((chain[0]["chain"], completion, result) for completion, result in tb.chain_steps(chain))
    return headers, steps


def expected_chunk_text(chunk, headers, steps):
    """Text a chunk must decode to: header, the orphan tool response (chunk starts mid-chain), then its completion/response pairs."""
    first, last = chunk["steps"]
    text = headers[chunk["chain"]]
    if first > 0 and steps[first - 1][0] == chunk["chain"]:
        text += steps[first - 1][2] or ""
    for index in range(first, last + 1):
        text += steps[index][1] + ((steps[index][2] or "") if index < last else "")
    return text


def first_difference(left, right):
    """Offset of the first differing character (the shorter length when one string is a prefix); None when equal."""
    if left == right:
        return None
    for offset, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return offset
    return min(len(left), len(right))


def header_roundtrip(chunk, record, decode):
    """Required one-chunk check: decoded header ids against the stored header segment text_segments[0].text."""
    decoded, stored = decode(chunk["input_ids"][:header_end(chunk["input_ids"])]), record["text_segments"][0]["text"]
    return {"traj_id": record["traj_id"], "equal": decoded == stored, "first_diff_offset": first_difference(decoded, stored)}


def scan(chunks_path, source_path, offsets, decode):
    """Stream the chunk file, comparing every chunk with the stored segments: (rows, unreadable, missing, header check, sha256)."""
    rows, unreadable, missing, header_check, digest = [], [], [], None, hashlib.sha256()
    with chunks_path.open("rb") as handle, source_path.open("rb") as source:
        for number, line in enumerate(handle, start=1):
            digest.update(line)
            try:
                record = json.loads(line)
                traj_id, chunks = record["traj_id"], record["chunks"]
            except (ValueError, KeyError, TypeError) as exc:
                unreadable.append("line %d: %s" % (number, exc))
                continue
            if traj_id not in offsets:
                missing.append(traj_id)
                continue
            source.seek(offsets[traj_id])
            stored = json.loads(source.readline())
            headers, steps = reference_parts(stored)
            for index, chunk in enumerate(chunks):
                row = analyse_chunk(chunk, decode, expected_chunk_text(chunk, headers, steps))
                if header_check is None and row["type"] == "first":
                    header_check = header_roundtrip(chunk, stored, decode)
                rows.append({"traj_id": traj_id, "source_line": record.get("source_line"), "chunk_index": index, **row})
    return rows, unreadable, missing, header_check, digest.hexdigest()


def group_rows(rows):
    """Rows by reporting group: the task's two chunk types, all, and 'later' split by the decoded start of the chunk."""
    later = [r for r in rows if r["type"] == "later"]
    return {"first": [r for r in rows if r["type"] == "first"], "later": later, "all": rows,
            "later_orphan": [r for r in later if r["start_kind"] == "orphan_response"],
            "later_chain_start": [r for r in later if r["start_kind"] != "orphan_response"]}


def totals_of(rows):
    return {"chunks": len(rows), "completions": sum(r["n_completions"] for r in rows),
            "calls": sum(r["n_calls"] for r in rows), "loss_tokens": sum(r["n_loss_tokens"] for r in rows)}


def summarize(rows, totals):
    """Group summary (mirrors teacher_behaviour.summarize_arm; chunks replace trajectories). Denominators: chunks for first
    action / loop / submit, completions for thought, M5 transitions for M5, all calls of all chunks for history-free."""
    n, t = len(rows), totals_of(rows)

    def col(key):
        return [r[key] for r in rows]

    nonrecovery = {}
    for stratum in MM.STRATA:
        cells = [r["nonrecovery"][stratum] for r in rows]
        eligible, identical = sum(c["eligible"] for c in cells), sum(c["identical"] for c in cells)
        nonrecovery[stratum] = {"mean_repeat_rate": tb.stat(c["repeat_rate"] for c in cells),
                                "chunks_with_eligible": sum(c["eligible"] > 0 for c in cells), "eligible_transitions": eligible,
                                "identical_transitions": identical, "pooled_repeat_rate": tb.share(identical, eligible)}
    thinking, hf = sum(col("n_thinking")), sum(col("history_free_actions"))
    return {
        "n_chunks": n, "n_completions": t["completions"], "n_calls": t["calls"], "n_loss_tokens": t["loss_tokens"],
        "n_tokens": sum(col("n_tokens")), "mean_chunk_tokens": tb.stat(col("n_tokens")),
        "mean_completions_per_chunk": tb.share(t["completions"], n), "share_of_chunks": tb.share(n, totals["chunks"]),
        "share_of_completions": tb.share(t["completions"], totals["completions"]),
        "share_of_loss_tokens": tb.share(t["loss_tokens"], totals["loss_tokens"]),
        "first_action_counts": {o: col("first_action").count(o) for o in MM.OPENERS},
        "first_call_tool_counts": dict(sorted(Counter(r["first_call_tool"] or "none" for r in rows).items())),
        "loop_count": sum(col("loop")), "max_repeat_mean": tb.stat(col("max_repeat")),
        "max_repeat_median": tb.stat(col("max_repeat"), statistics.median), "distinct_ratio_mean": tb.stat(col("distinct_ratio")),
        "chunks_with_calls": sum(c > 0 for c in col("n_calls")), "submit_chunks": sum(col("submit_called")),
        "nonrecovery": nonrecovery, "unknown_excluded_transitions": sum(r["nonrecovery"]["unknown_excluded"] for r in rows),
        "thought": {"nonblank": thinking, "completions": t["completions"], "pooled_rate": tb.share(thinking, t["completions"]),
                    "per_chunk_mean": tb.stat(r["n_thinking"] / r["n_completions"] for r in rows if r["n_completions"])},
        "history_free_actions": hf, "history_free_share_of_all_calls": tb.share(hf, totals["calls"]),
        "history_free_share_of_group_calls": tb.share(hf, t["calls"]),
        "malformed_chunks": sum(r["status"] != "ok" for r in rows)}


def reconciliation(groups, rows, parent_dir):
    """Totals against the raw table (parent per_trajectory.jsonl) for the arm M0 packs. All but the M5 rows must agree;
    raw_minus_chunk of an M5 row = transitions that straddle two chunks (inside no chunk)."""
    with (parent_dir / "per_trajectory.jsonl").open(encoding="utf-8") as handle:
        raw = [r for r in map(json.loads, handle) if RAW_ARM in r["arms"]]
    whole, firsts = groups["all"], groups["first"]["first_action_counts"]
    pairs = {"trajectories": (len({r["traj_id"] for r in rows}), len(raw)),
             "completions": (whole["n_completions"], sum(r["n_steps"] for r in raw)),
             "tool_calls": (whole["n_calls"], sum(r["n_tool_calls"] for r in raw)),
             "nonblank_thoughts": (whole["thought"]["nonblank"], sum(r["n_thinking_steps"] for r in raw))}
    pairs.update({"first_action_" + o: (firsts[o], sum(r["opener"] == o for r in raw)) for o in MM.OPENERS})
    pairs.update({"m5_eligible_" + s: (whole["nonrecovery"][s]["eligible_transitions"],
                                       sum(r["nonrecovery"][s]["eligible"] for r in raw)) for s in MM.STRATA})
    return {name: {"chunk_view": c, "raw_table": r, "raw_minus_chunk": r - c} for name, (c, r) in pairs.items()}


def type_check(rows):
    """The task's chunk type (steps rule) against the decoded start of the chunk (tool response after the header?)."""
    odd = [r for r in rows if r["type"] == "later" and r["start_kind"] != "orphan_response"]
    return {"type_by_steps/start_kind": dict(sorted(Counter("%s/%s" % (r["type"], r["start_kind"]) for r in rows).items())),
            "later_by_steps_rule": sum(r["type"] == "later" for r in rows),
            "later_by_decoded_orphan_response": sum(r["start_kind"] == "orphan_response" for r in rows),
            "mismatch_first_but_orphan_response": sum(r["type"] == "first" and r["start_kind"] == "orphan_response" for r in rows),
            "mismatch_later_but_no_orphan_response": len(odd),
            "mismatch_later_by_chain_and_start": dict(sorted(Counter("chain%d/%s" % (r["chain"], r["start_kind"]) for r in odd).items()))}


def ratio(count, total):
    return "%s (%d/%d)" % (tb.fmt(tb.share(count, total)), count, total)


def render_markdown(report):
    """report.md: one table (rows = metrics, columns = chunk groups; every share carries its numerator/denominator)."""
    groups, fmt = report["groups"], tb.fmt

    def m5(g, s):
        c = g["nonrecovery"][s]
        return "%s (pooled %s; chunks %d/%d, elig %d)" % (fmt(c["mean_repeat_rate"]), fmt(c["pooled_repeat_rate"]),
                                                          c["chunks_with_eligible"], g["n_chunks"], c["eligible_transitions"])

    spec = [("chunks (share of all)", lambda g: "%d (%s)" % (g["n_chunks"], fmt(g["share_of_chunks"]))),
            ("supervised completions (mean per chunk)", lambda g: "%d (%s)" % (g["n_completions"], fmt(g["mean_completions_per_chunk"], 2))),
            ("supervised tool calls parsed", lambda g: str(g["n_calls"])),
            ("supervised tokens (share of all)", lambda g: "%d (%s)" % (g["n_loss_tokens"], fmt(g["share_of_loss_tokens"]))),
            ("mean chunk length, tokens", lambda g: fmt(g["mean_chunk_tokens"], 0))]
    spec += [("first supervised action: " + o, lambda g, o=o: ratio(g["first_action_counts"][o], g["n_chunks"])) for o in MM.OPENERS]
    spec += [("loop in chunk (max multiplicity >= %d)" % MM.DEFAULT_LOOP_THRESHOLD, lambda g: ratio(g["loop_count"], g["n_chunks"])),
             ("max multiplicity mean / median", lambda g: "%s / %s" % (fmt(g["max_repeat_mean"], 2), fmt(g["max_repeat_median"], 1))),
             ("distinct-call ratio, mean over chunks with a call", lambda g: "%s (n=%d)" % (fmt(g["distinct_ratio_mean"]), g["chunks_with_calls"])),
             ("chunk has a submit_patch call (extra row)", lambda g: ratio(g["submit_chunks"], g["n_chunks"]))]
    spec += [("M5 repeat after %s: mean (pooled; chunks with >= 1 eligible, eligible transitions)" % s, lambda g, s=s: m5(g, s)) for s in MM.STRATA]
    spec += [("thought present, pooled over completions", lambda g: ratio(g["thought"]["nonblank"], g["n_completions"])),
             ("thought present, per-chunk mean", lambda g: fmt(g["thought"]["per_chunk_mean"])),
             ("history-free actions (share of ALL supervised calls)", lambda g: "%d (%s)" % (g["history_free_actions"], fmt(g["history_free_share_of_all_calls"], 4))),
             ("history-free actions (share of the group's own calls)", lambda g: ratio(g["history_free_actions"], g["n_calls"]))]
    check, header, bad = report["type_check"], report["header_roundtrip"], report["malformed"]
    lines = ["# Chunk view of the supervised targets: %s" % Path(report["inputs"]["chunks_file"]).name, "",
             "Groups: first = steps[0] == 0; later = steps[0] > 0; later_orphan = later with a tool response right after the header; "
             "later_chain_start = later without it (chain > 0 summarizer or continuation chunk). Details: README.md.", "",
             "| metric | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    lines += ["| %s | %s |" % (label, " | ".join(cell(groups[name]) for name in GROUPS)) for label, cell in spec]
    lines += ["", "Type check: %s." % ", ".join("%s %d" % item for item in check["type_by_steps/start_kind"].items()),
              "Mismatches: first chunks with an orphan response %d; later chunks without one %d." % (
                  check["mismatch_first_but_orphan_response"], check["mismatch_later_but_no_orphan_response"]),
              "Round trip: header of %s equal %s (first differing offset %s); %d of %d chunks decode to the stored segments." % (
                  header["traj_id"], header["equal"], header["first_diff_offset"], report["roundtrip_equal_chunks"], report["totals"]["chunks"]),
              "Malformed chunks %d, unreadable lines %d, trajectories missing from the source %d." % (
                  bad["chunks"], bad["unreadable_lines"], bad["missing_trajectories"])]
    return NL.join(lines) + NL


def manifest_check(chunks, totals):
    """[this run, manifest.json next to the chunk file] for the chunk and supervised-token counts (None without a manifest)."""
    path, key = chunks.parent / "manifest.json", ("m0" if chunks.name.startswith("M0") else "m1") + "_shifted_targets"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return manifest and {"chunks": [totals["chunks"], manifest["chunks"]], "supervised_tokens": [totals["loss_tokens"], manifest[key]]}


def main(argv=None):
    """CLI entry point: 0 on success, 2 on input errors, 3 when malformed chunks or unreadable lines were found."""
    parser = argparse.ArgumentParser(description="Chunk-level behaviour metrics of the supervised SFT targets, by chunk type.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="trajectories.jsonl, for the round-trip check")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    try:
        import transformers  # imported late: the pure functions and their tests need no tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

        def decode(ids):
            return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        offsets, source_sha = source_offsets(args.source)
        rows, unreadable, missing, header_check, chunks_sha = scan(args.chunks, args.source, offsets, decode)
        tokenizer_sha = hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest()
        reference = hashlib.sha256(Path(tb.__file__).read_bytes()).hexdigest()
    except (OSError, ValueError, ImportError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    totals, bad = totals_of(rows), [r for r in rows if r["status"] != "ok"]
    groups = {name: summarize(members, totals) for name, members in group_rows(rows).items()}
    report = {
        "inputs": {"chunks_file": str(args.chunks), "chunks_sha256": chunks_sha, "source_file": str(args.source),
                   "source_sha256": source_sha, "tokenizer_dir": str(args.tokenizer), "tokenizer_json_sha256": tokenizer_sha,
                   "teacher_behaviour_sha256": reference, "reference_module_sha256": tb.REFERENCE_SHA256,
                   "loop_threshold": MM.DEFAULT_LOOP_THRESHOLD, "python": sys.version.split()[0],
                   "transformers": transformers.__version__},
        "totals": totals, "groups": groups, "type_check": type_check(rows),
        "header_roundtrip": header_check or {"traj_id": None, "equal": None, "first_diff_offset": None},
        "roundtrip_equal_chunks": sum(r["roundtrip_diff"] is None for r in rows),
        "reconciliation": reconciliation(groups, rows, PARENT), "manifest_check": manifest_check(args.chunks, totals),
        "completions_with_marker": dict(sorted(Counter(m for r in rows for c in r["completions"] for m in c["controls"]).items())),
        "malformed": {"chunks": len(bad), "unreadable_lines": len(unreadable), "missing_trajectories": len(missing),
                      "error_kinds_sampled": dict(sorted(Counter(e.split(":")[0] for r in bad for e in r["errors"]).items())),
                      "samples": [{"traj_id": r["traj_id"], "chunk_index": r["chunk_index"], "errors": r["errors"]} for r in bad[:MAX_ERROR_SAMPLES]]}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("per_chunk.jsonl", "".join(json.dumps(r, sort_keys=True) + NL for r in rows)),
                       ("report.json", json.dumps(report, indent=2, sort_keys=True) + NL), ("report.md", render_markdown(report))):
        (args.out_dir / name).write_bytes(text.encode("utf-8"))
    for name in GROUPS:
        g = groups[name]
        print("%s: chunks=%d completions=%d calls=%d loops=%d malformed=%d" % (
            name, g["n_chunks"], g["n_completions"], g["n_calls"], g["loop_count"], g["malformed_chunks"]))
    problems = len(bad) + len(unreadable) + len(missing)
    if problems:
        print("WARNING: %d malformed chunks / unreadable lines / missing trajectories, see report.json" % problems, file=sys.stderr)
    return 3 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
