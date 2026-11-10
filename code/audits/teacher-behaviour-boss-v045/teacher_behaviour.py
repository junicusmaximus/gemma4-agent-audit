#!/usr/bin/env python3
"""Behaviour statistics inside the SFT teacher trajectories (CPU only, standard library).

Computes the metrics of tools/mechanism_metrics_v032.py (loop, submit, opener, M5 non-recovery, thought presence)
on the trajectories of sft/data/trajectories.jsonl, by arm. The trajectory text is adapted to the trace format
the reference functions expect; the reference functions themselves are imported, not copied. See README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ anywhere on the server

GEMMA4_ROOT = Path(__file__).resolve().parents[2]  # .../gemma4 (this file lives in gemma4/analysis/<name>/)
TOOLS_DIR = GEMMA4_ROOT / "tools"
DEFAULT_DATA = GEMMA4_ROOT / "sft" / "data" / "trajectories.jsonl"
DEFAULT_ARMS = GEMMA4_ROOT / "sft" / "data" / "arms.json"
REFERENCE_MODULE = "mechanism_metrics_v032"
REFERENCE_SHA256 = "36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d"

NL = chr(10)
QUOTE = '<|"|>'
CALL_START, CALL_PREFIX, CALL_CLOSE = "<|tool_call>", "call:", "<tool_call|>"
STOP_TOKENS = ("<|tool_response>", "<turn|>")  # end an unclosed call; never in served output text
CHANNEL_OPEN = "<|channel>thought" + NL
CHANNEL_CLOSE = "<channel|>"
RESPONSE_OPEN = "response:"
RESPONSE_VALUE = "{value:" + QUOTE
RESPONSE_CLOSE = QUOTE + "}<tool_response|>"
THINK_TOKEN = "<|think|>"
SYSTEM_START = "<bos><|turn>system" + NL
SUMMARIZER_START = SYSTEM_START + THINK_TOKEN + NL + "<turn|>"  # compaction call: empty system prompt
REQUIRED_FIELDS = ("traj_id", "tier", "task_id", "generating_model", "n_steps", "tool_calls", "n_chains", "text_segments")
MAX_ERROR_SAMPLES = 5


def load_reference():
    """Import the pinned reference module; refuse to run when its bytes differ from REFERENCE_SHA256."""
    path = TOOLS_DIR / (REFERENCE_MODULE + ".py")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError("%s has sha256 %s, expected %s" % (path, digest, REFERENCE_SHA256))
    if str(TOOLS_DIR) not in sys.path:
        sys.path.append(str(TOOLS_DIR))
    return importlib.import_module(REFERENCE_MODULE)


MM = load_reference()
STRATA = MM.STRATA
OPENERS = MM.OPENERS


def nested_end(text, i):
    """(index after the bracket that closes text[i] ('{' or '['), closed?); string contents are skipped."""
    opener, closer = ("{", "}") if text[i] == "{" else ("[", "]")
    depth, i = 1, i + 1
    while i < len(text) and depth > 0:
        if text.startswith(QUOTE, i):
            end = text.find(QUOTE, i + len(QUOTE))
            i = len(text) if end < 0 else end + len(QUOTE)
            continue
        depth += (text[i] == opener) - (text[i] == closer)
        i += 1
    return i, depth == 0


def parse_items(text, keyed):
    """Port of vllm.parser.gemma4 _parse_gemma4_args (keyed: dict) and _parse_gemma4_array (list), non-streaming.
    Tolerant of malformed text like the original; bare values stay raw strings (the server coerces them by schema)."""
    result, i, n = ({} if keyed else []), 0, len(text)
    while True:
        while i < n and text[i] in " ,\n\t":
            i += 1
        if i >= n:
            return result
        key = None
        if keyed:
            colon = text.find(":", i)
            if colon < 0:
                return result
            key = text[i:colon].strip()
            if key.startswith(QUOTE) and key.endswith(QUOTE):
                key = key[len(QUOTE):-len(QUOTE)]
            i = colon + 1
            while i < n and text[i] in " \n\t":
                i += 1
            if i >= n:
                result[key] = ""
                return result
        if text.startswith(QUOTE, i):
            end = text.find(QUOTE, i + len(QUOTE))
            value, i = (text[i + len(QUOTE):], n) if end < 0 else (text[i + len(QUOTE):end], end + len(QUOTE))
        elif text[i] in "{[":
            start = i
            i, closed = nested_end(text, start)
            value = parse_items(text[start + 1:i - 1 if closed else i], text[start] == "{")
        else:
            j = i
            while j < n and text[j] not in (",}]" if keyed else ",]"):
                j += 1
            if j == i:
                return result
            value, i = text[i:j].strip(), j
        if keyed:
            result[key] = value
        else:
            result.append(value)


def parse_calls(completion):
    """Tool calls of one completion as the vLLM gemma4 parser extracts them: ([{function_name, arguments}], errors)."""
    calls, errors, i = [], [], 0
    while True:
        start = completion.find(CALL_START, i)
        if start < 0:
            return calls, errors
        end = completion.find(CALL_CLOSE, start)
        stop = next((s for s in STOP_TOKENS if completion.endswith(s)), "")
        body = completion[start + len(CALL_START):len(completion) - len(stop) if end < 0 else end]
        i = len(completion) if end < 0 else end + len(CALL_CLOSE)
        prefix = body.find(CALL_PREFIX)
        brace = body.find("{", prefix) if prefix >= 0 else -1
        name = body[prefix + len(CALL_PREFIX):brace].strip() if brace >= 0 else ""
        if not name:
            errors.append("tool call without call:NAME{ near %r" % body[:40])
            continue
        raw = body[brace + 1:].strip()
        calls.append({"function_name": name, "arguments": parse_items(raw[:-1] if raw.endswith("}") else raw, True)})


def parse_response(segment):
    """(function_name, content) of the tool response that opens a result segment; raises ValueError."""
    marker, end = segment.find(RESPONSE_VALUE), segment.rfind(RESPONSE_CLOSE)
    if not segment.startswith(RESPONSE_OPEN) or marker < 0 or end < marker + len(RESPONSE_VALUE):
        raise ValueError("result segment is not response:NAME{value:<q>...<q>}")
    return segment[len(RESPONSE_OPEN):marker].strip(), segment[marker + len(RESPONSE_VALUE):end]


def thought_text(completion):
    """Text from the optional channel opener to the first channel close; '' when there is no close marker."""
    body = completion[len(CHANNEL_OPEN):] if completion.startswith(CHANNEL_OPEN) else completion
    end = body.find(CHANNEL_CLOSE)
    return body[:end] if end >= 0 else ""


def group_chains(segments):
    """Segments grouped by chain index (chains in index order, segment order kept)."""
    chains = {}
    for segment in segments:
        chains.setdefault(segment["chain"], []).append(segment)
    return [chains[index] for index in sorted(chains)]


def chain_steps(chain):
    """(completion text, following result text or None) per step of one chain (header, then completion/result)."""
    if [s["train"] for s in chain] != [bool(k % 2) for k in range(len(chain))]:
        raise ValueError("segments do not alternate header / completion / result")
    return [(chain[k]["text"], chain[k + 1]["text"] if k + 1 < len(chain) else None)
            for k in range(1, len(chain), 2)]


def check_record(record):
    """Raise ValueError unless the record has the fields and segment shape this script reads."""
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    if not (isinstance(record["text_segments"], list) and record["text_segments"] and all(
            isinstance(s, dict) and {"chain", "text", "train"} <= set(s) for s in record["text_segments"])):
        raise ValueError("text_segments must be a non-empty list of {chain, text, train} objects")


def step_observation(calls, result):
    """(observation dict or None, errors) for the single call of a step and the result segment after it."""
    if not calls or result is None:
        return None, []  # no call, or chain-final call: its result is not in this chain (stratum unknown)
    try:
        name, content = parse_response(result)
    except ValueError as exc:
        return None, ["tool response of %r: %s" % (calls[0]["function_name"], exc)]
    if name != calls[0]["function_name"]:
        return None, ["tool response %r follows call %r" % (name, calls[0]["function_name"])]
    return {"content": content}, []


def analyse_trajectory(record):
    """Behaviour row of one record. Raises ValueError for an unusable record; format problems inside a usable
    record are listed in row['errors'] and make row['status'] 'malformed'."""
    check_record(record)
    chains = group_chains(record["text_segments"])
    errors, pairs, thoughts, n_summarizer, last_step = [], [], [], 0, ([], None)
    for chain in chains:
        try:
            steps = chain_steps(chain)
        except ValueError as exc:
            errors.append("chain %s: %s" % (chain[0]["chain"], exc))
            continue
        n_summarizer += len(steps) if chain[0]["text"].startswith(SUMMARIZER_START) else 0
        for completion, result in steps:
            calls, call_errors = parse_calls(completion)
            observation, response_errors = step_observation(calls, result)
            errors.extend(call_errors + response_errors)
            if len(calls) > 1:
                errors.append("completion with %d tool calls (one expected)" % len(calls))
            pairs.extend((call, observation) for call in calls)
            last_step = (calls, result)
            thoughts.append(thought_text(completion))
            if CHANNEL_CLOSE not in completion:
                errors.append("completion without a channel close marker")
    n_steps, n_thinking = len(thoughts), sum(bool(text.strip()) for text in thoughts)
    unanswered = int(bool(last_step[0]) and last_step[1] is None)  # a final call without tool response (external tiers)
    for label, parsed, recorded in (("chains", len(chains), record["n_chains"]), ("completions", n_steps, record["n_steps"]),
                                    ("answered tool calls", len(pairs) - unanswered, record["tool_calls"])):
        if parsed != recorded:
            errors.append("parsed %d %s, the record says %s" % (parsed, label, recorded))
    repeat = MM.repeat_stats([MM.tool_call_identity(call) for call, _ in pairs], MM.DEFAULT_LOOP_THRESHOLD)
    first_header = chains[0][0]["text"]
    return {
        "traj_id": record["traj_id"], "task_id": record["task_id"], "tier": record["tier"],
        "generating_model": record["generating_model"], "n_chains": record["n_chains"],
        "n_steps": n_steps, "n_tool_calls": len(pairs), "n_summarizer_steps": n_summarizer,
        **repeat,
        "submit_called": MM.submit_called(pairs),
        "opener": MM.classify_opener(pairs[0][0] if pairs else None),
        "nonrecovery": MM.nonrecovery(pairs),
        "n_thinking_steps": n_thinking,
        "think_rate": n_thinking / n_steps if n_steps else None,
        "first_header_has_think_token": THINK_TOKEN in first_header,
        "all_headers_have_think_token": all(THINK_TOKEN in c[0]["text"] for c in chains),
        "n_errors": len(errors), "errors": errors[:MAX_ERROR_SAMPLES],
        "status": "malformed" if errors else "ok",
    }


def stat(values, fn=statistics.fmean):
    """fn over the non-None values (None if there are none); fmean matches the reference summary's mean."""
    kept = [v for v in values if v is not None]
    return fn(kept) if kept else None


def share(count, total):
    return count / total if total else None


def summarize_arm(name, rows):
    """Arm summary over the well-formed rows (malformed rows are only counted); mirrors reference summarize_run."""
    ok = [r for r in rows if r["status"] == "ok"]
    n = len(ok)

    def col(key):
        return [r[key] for r in ok]

    nonrecovery = {}
    for stratum in STRATA:
        cells = [r["nonrecovery"][stratum] for r in ok]
        with_eligible = sum(c["eligible"] > 0 for c in cells)
        eligible, identical = sum(c["eligible"] for c in cells), sum(c["identical"] for c in cells)
        nonrecovery[stratum] = {
            "mean_repeat_rate": stat(c["repeat_rate"] for c in cells), "n_trajectories": with_eligible,
            "share_trajectories": share(with_eligible, n), "eligible_transitions": eligible,
            "identical_transitions": identical, "pooled_repeat_rate": share(identical, eligible)}
    turns, thinking, tiers = sum(col("n_steps")), sum(col("n_thinking_steps")), Counter(col("tier"))
    return {
        "arm": name, "n": n, "n_malformed": len(rows) - n,
        "tier_counts": dict(sorted(tiers.items())), "tier_shares": {t: share(c, n) for t, c in sorted(tiers.items())},
        "generating_model_counts": dict(sorted(Counter(col("generating_model")).items())),
        "n_steps_mean": stat(col("n_steps")), "n_steps_median": stat(col("n_steps"), statistics.median),
        "n_tool_calls_mean": stat(col("n_tool_calls")), "n_tool_calls_median": stat(col("n_tool_calls"), statistics.median),
        "n_summarizer_steps": sum(col("n_summarizer_steps")),
        "loop_count": sum(col("loop")), "loop_rate": share(sum(col("loop")), n),
        "max_repeat_mean": stat(col("max_repeat")), "max_repeat_median": stat(col("max_repeat"), statistics.median),
        "distinct_ratio_mean": stat(col("distinct_ratio")),
        "submit_count": sum(col("submit_called")), "submit_rate": share(sum(col("submit_called")), n),
        "opener_counts": {o: col("opener").count(o) for o in OPENERS},
        "opener_shares": {o: share(col("opener").count(o), n) for o in OPENERS},
        "nonrecovery": nonrecovery,
        "unknown_excluded_transitions": sum(r["nonrecovery"]["unknown_excluded"] for r in ok),
        "thought": {"model_turns": turns, "nonempty_turns": thinking, "pooled_rate": share(thinking, turns),
                    "per_trajectory_mean": stat(col("think_rate")),
                    "trajectories_with_any": sum(v > 0 for v in col("n_thinking_steps"))},
        "header_think_token": {"n": n, "first_header": sum(col("first_header_has_think_token")),
                               "all_chain_headers": sum(col("all_headers_have_think_token"))},
    }


def fmt(value, digits=3):
    return "n/a" if value is None else "%.*f" % (digits, value)


def render_markdown(report):
    """report.md: one table, rows = metrics, columns = arms."""
    arms = [report["arms"][name] for name in report["arms_requested"]]
    tiers = sorted({t for a in arms for t in a["tier_counts"]})

    def m5(a, s):
        c = a["nonrecovery"][s]
        return "%s (traj %d/%d, elig %d)" % (fmt(c["mean_repeat_rate"]), c["n_trajectories"], a["n"], c["eligible_transitions"])

    spec = [("n trajectories (well-formed)", lambda a: str(a["n"])), ("malformed (excluded)", lambda a: str(a["n_malformed"])),
            ("resolved", lambda a: "all (arm definition)")]
    spec += [("tier share " + t, lambda a, t=t: fmt(a["tier_shares"].get(t, 0.0))) for t in tiers]
    spec += [("mean n_steps / n_tool_calls", lambda a: "%s / %s" % (fmt(a["n_steps_mean"], 2), fmt(a["n_tool_calls_mean"], 2))),
             ("loop rate (max multiplicity >= %d)" % MM.DEFAULT_LOOP_THRESHOLD, lambda a: fmt(a["loop_rate"])),
             ("max multiplicity mean / median", lambda a: "%s / %s" % (fmt(a["max_repeat_mean"], 2), fmt(a["max_repeat_median"], 1))),
             ("distinct-call ratio (mean)", lambda a: fmt(a["distinct_ratio_mean"])),
             ("submit_patch rate", lambda a: fmt(a["submit_rate"]))]
    spec += [("first action: " + o, lambda a, o=o: fmt(a["opener_shares"][o])) for o in OPENERS]
    spec += [("M5 repeat rate after %s (mean; traj with >= 1 eligible, eligible transitions)" % s, lambda a, s=s: m5(a, s)) for s in STRATA]
    spec += [("thought present, per-trajectory mean", lambda a: fmt(a["thought"]["per_trajectory_mean"])),
             ("thought present, pooled over model turns", lambda a: fmt(a["thought"]["pooled_rate"])),
             ("trajectories whose first header has the think token", lambda a: "%d / %d" % (a["header_think_token"]["first_header"], a["n"]))]
    head = ["# Teacher trajectory behaviour by arm", "",
            "Data sha256 %s; reference %s sha256 %s; loop threshold %d. Details: README.md." % (
                report["inputs"]["data_sha256"][:12], REFERENCE_MODULE, REFERENCE_SHA256[:8], MM.DEFAULT_LOOP_THRESHOLD), "",
            "| metric | " + " | ".join(a["arm"] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    return NL.join(head + ["| %s | %s |" % (label, " | ".join(cell(a) for a in arms)) for label, cell in spec]) + NL


def scan(data_path, membership):
    """Stream the data file once: (rows in file order, each with its 'arms' list; unreadable lines; file sha256)."""
    wanted = frozenset().union(*membership.values())
    digest, rows, unreadable = hashlib.sha256(), [], []
    with data_path.open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            digest.update(line)
            try:
                record = json.loads(line)
                traj_id = record["traj_id"]
            except (ValueError, KeyError, TypeError) as exc:
                unreadable.append("line %d: %s" % (number, exc))
                continue
            if traj_id not in wanted:
                continue
            try:
                row = analyse_trajectory(record)
            except ValueError as exc:
                row = {"traj_id": traj_id, "status": "malformed", "n_errors": 1, "errors": [str(exc)]}
            rows.append({**row, "arms": [name for name, ids in membership.items() if traj_id in ids]})
    return rows, unreadable, digest.hexdigest()


def main(argv=None):
    """CLI entry point: 0 on success, 2 on input errors, 3 when malformed records or missing ids were found."""
    parser = argparse.ArgumentParser(description="Behaviour metrics inside the SFT teacher trajectories, per arm.")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME", help="arms.json key (repeatable)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--arms-file", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    try:
        arms = json.loads(args.arms_file.read_text(encoding="utf-8"))
        names = list(dict.fromkeys(args.arm))
        if not all(isinstance(arms.get(name), list) for name in names):
            raise ValueError("%s has no id list for every requested arm" % args.arms_file)
        membership = {name: frozenset(arms[name]) for name in names}
        rows, unreadable, data_digest = scan(args.data, membership)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    seen, bad = {r["traj_id"] for r in rows}, [r for r in rows if r["status"] != "ok"]
    report = {
        "inputs": {"data": str(args.data), "data_size_bytes": args.data.stat().st_size, "data_sha256": data_digest,
                   "arms_file": str(args.arms_file), "arms_file_sha256": hashlib.sha256(args.arms_file.read_bytes()).hexdigest(),
                   "reference_module": REFERENCE_MODULE + ".py", "reference_sha256": REFERENCE_SHA256,
                   "loop_threshold": MM.DEFAULT_LOOP_THRESHOLD},
        "arms_requested": names,
        "arm_ids_missing_from_data": {name: len(ids - seen) for name, ids in membership.items()},
        "unreadable_lines": len(unreadable), "unreadable_line_samples": unreadable[:MAX_ERROR_SAMPLES],
        "malformed_trajectories": len(bad),
        "malformed_samples": [{"traj_id": r["traj_id"], "errors": r["errors"]} for r in bad[:MAX_ERROR_SAMPLES]],
        "arms": {name: summarize_arm(name, [r for r in rows if name in r["arms"]]) for name in names}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in (("per_trajectory.jsonl", "".join(json.dumps(r, sort_keys=True) + NL for r in rows)),
                           ("report.json", json.dumps(report, indent=2, sort_keys=True) + NL),
                           ("report.md", render_markdown(report))):
        (args.out_dir / filename).write_bytes(text.encode("utf-8"))
    for name, arm in report["arms"].items():
        print("%s: n=%d malformed=%d loop_rate=%s submit_rate=%s" % (name, arm["n"], arm["n_malformed"], fmt(arm["loop_rate"]), fmt(arm["submit_rate"])))
    problems = len(bad) + len(unreadable) + sum(report["arm_ids_missing_from_data"].values())
    if problems:
        print("WARNING: %d malformed/unreadable/missing items, see report.json" % problems, file=sys.stderr)
    return 3 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
