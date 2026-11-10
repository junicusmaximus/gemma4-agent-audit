#!/usr/bin/env python3
"""Mechanism metrics M1-M6 and a failure taxonomy for swegemma agent-evaluation runs.

Input per run directory: task_results.jsonl and traces/trace_<instance_id>.json.
Outcome fields (resolved, test_exit_code) and final_metrics are never read.
Standard library only; --tokenizer imports transformers lazily. See README-mechanism-metrics-v032.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

STEP_LABELS = ("tool_call", "truncated", "unparseable", "other_text", "empty")
OPENERS = ("find_opener", "wrapper_opener", "repro_script_opener", "other", "no_tool_call")
STRATA = ("error", "silent", "normal")
TAXONOMY = ("loop", "no_parsed_tool_call", "submitted", "neither")
ERROR_MARKERS = {"timeout": "session timeout", "turn_cap": "turns budget",  # checked in order, first match wins
                 "context_overflow": "ContextWindowExceeded", "no_submit": "without explicit submit_patch"}
ERROR_CLASSES = (*ERROR_MARKERS, "other", "none")
RAW_TOOL_MARKER = "<|tool_call>"
FIND_OPENER = re.compile(r"^find /workspace -max(?:-?depth)? [12]\b")
WRAPPER_PREFIX = "__public_sft_output="
REPRO_SHEBANG = "#!/usr/bin/env python3"
REPRO_WINDOW = 200
PATH_SEPARATORS = ("/", chr(92))  # slash and backslash: ids containing either are never opened
DEFAULT_LOOP_THRESHOLD = 5
DEFAULT_MAX_OUTPUT_TOKENS = 8192
READ_ENCODING = "utf-8-sig"  # utf-8, tolerating a BOM written by Windows tools


def _is_number(value: Any) -> bool:
    """True for finite ints and floats (bool excluded)."""
    is_real = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_real and (isinstance(value, int) or math.isfinite(value))


def _as_dict(value: Any) -> dict:
    """The value if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    """The value if it is a str, else the empty string."""
    return value if isinstance(value, str) else ""


def _share(count: int, total: int) -> float | None:
    """count / total, or None when total is zero."""
    return count / total if total else None


def _mean(values: Iterable[Any]) -> float | None:
    """Mean of the numeric entries of values (None if there are none)."""
    kept = [v for v in values if _is_number(v)]
    return statistics.fmean(kept) if kept else None


def _median(values: Iterable[Any]) -> float | None:
    """Median of the numeric entries of values (None if there are none)."""
    kept = [v for v in values if _is_number(v)]
    return statistics.median(kept) if kept else None


def parsed_tool_calls(step: dict) -> list[dict]:
    """The dict entries of step['tool_calls'] (empty when absent or not a list)."""
    calls = step.get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def step_completion_tokens(step: dict) -> float | None:
    """metrics.completion_tokens, else the max tool_calls[*].extra.usage.completion_tokens, else None."""
    value = _as_dict(step.get("metrics")).get("completion_tokens")
    if _is_number(value):
        return value
    usages = (_as_dict(_as_dict(call.get("extra")).get("usage")) for call in parsed_tool_calls(step))
    found = [usage["completion_tokens"] for usage in usages if _is_number(usage.get("completion_tokens"))]
    return max(found) if found else None


def _event_type(step: dict) -> Any:
    """The step's extra.event_type (None when absent)."""
    return _as_dict(step.get("extra")).get("event_type")


def has_thinking(step: dict) -> bool:
    """True if the step carries thinking text: event_type 'thinking' and a non-blank message, with or without tool_calls."""
    return _event_type(step) == "thinking" and bool(_text(step.get("message")).strip())


def classify_step(step: dict, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
    """Action label of one agent step (first match wins): tool_call, truncated (completion tokens >= cap),
    unparseable (raw tool-call marker in the text), other_text, empty. Thinking is a separate dimension.
    """
    if parsed_tool_calls(step):
        return "tool_call"
    tokens = step_completion_tokens(step)
    if tokens is not None and tokens >= max_output_tokens:
        return "truncated"
    message = _text(step.get("message"))
    if RAW_TOOL_MARKER in message:
        return "unparseable"
    return "other_text" if message.strip() else "empty"


def tool_call_identity(call: dict) -> tuple[str, str]:
    """M2 identity of a parsed call: (function_name, canonical JSON of its arguments)."""
    arguments = json.dumps(call.get("arguments"), sort_keys=True, ensure_ascii=False)
    return str(call.get("function_name")), arguments


def flatten_calls(steps: list[dict]) -> list[tuple[dict, Any]]:
    """Parsed tool calls in trace order as (call, observation) pairs; a step's calls share its observation."""
    return [(call, step.get("observation")) for step in steps for call in parsed_tool_calls(step)]


def classify_opener(call: dict | None) -> str:
    """M4: category of the first parsed tool call (None means the task has no parsed call)."""
    if call is None:
        return "no_tool_call"
    arguments = _as_dict(call.get("arguments"))
    name = call.get("function_name")
    command, content = _text(arguments.get("command")), _text(arguments.get("content"))
    if name == "run_command" and FIND_OPENER.match(command):
        return "find_opener"
    if name == "run_command" and command.startswith(WRAPPER_PREFIX):
        return "wrapper_opener"
    if name == "write_file" and content.startswith(REPRO_SHEBANG):
        return "repro_script_opener" if "reproduce" in content[:REPRO_WINDOW].lower() else "other"
    return "other"


def observation_stratum(observation: Any) -> str:
    """M5 stratum; unsupported content/error-field types are unknown, not fabricated successes."""
    content = _as_dict(observation).get("content")
    try:
        body = json.loads(content) if isinstance(content, str) else None
    except (ValueError, RecursionError):
        body = None
    if not isinstance(body, dict):
        return "unknown"
    if "error" in body:
        if not isinstance(body["error"], str):
            return "unknown"
        if body["error"].strip():
            return "error"
    bad_status = "status" in body and body["status"] != "ok"
    if bad_status or body.get("exit_code") not in (0, None) or _text(body.get("stderr")).strip():
        return "error"
    return "normal" if _text(body.get("stdout")).strip() else "silent"


def nonrecovery(pairs: list[tuple[dict, Any]]) -> dict:
    """M5: per observation stratum, how often the next parsed call is identical to the current one."""
    identities = [tool_call_identity(call) for call, _ in pairs]
    strata = [observation_stratum(observation) for _, observation in pairs[:-1]]
    repeats = [now == nxt for now, nxt in zip(identities, identities[1:])]
    result = {}
    for stratum in STRATA:
        hits = [repeat for name, repeat in zip(strata, repeats) if name == stratum]
        result[stratum] = {"eligible": len(hits), "identical": sum(hits), "repeat_rate": _share(sum(hits), len(hits))}
    return {**result, "unknown_excluded": strata.count("unknown")}


def default_tokenize(text: str) -> int:
    """Default M6 token count: number of whitespace-separated words."""
    return len(text.split())


def load_tokenizer(path: str) -> Callable[[str], int]:
    """Token counter backed by a local Hugging Face tokenizer; transformers is imported lazily."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--tokenizer requires the 'transformers' package, which is not installed; "
                           "install it or omit --tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return lambda text: len(tokenizer(text, add_special_tokens=True)["input_ids"])


def file_sha256(path: Path) -> str:
    """Hash a file without loading model weights or other unrelated assets."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_provenance(path: str | None) -> dict:
    """Identify the exact local tokenizer assets and the unchanged text-tokenization policy."""
    if path is None:
        return {"identifier": "python_whitespace_split", "method": "len(text.split())",
                "unit": "whitespace_words", "diagnostic_only": True, "files_sha256": {}}
    folder = Path(path).expanduser().resolve()
    assets = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja")
    hashes = {name: file_sha256(folder / name) for name in assets if (folder / name).is_file()}
    if not any(name in hashes for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("--tokenizer needs verifiable local tokenizer assets")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"identifier": str(folder), "method": "len(tokenizer(text, add_special_tokens=True)['input_ids'])",
            "loader": "transformers.AutoTokenizer.from_pretrained", "transformers_version": importlib.metadata.version("transformers"),
            "unit": "tokenizer_tokens", "diagnostic_only": False, "add_special_tokens": True,
            "local_files_only": True, "trust_remote_code": False, "apply_chat_template": False,
            "files_sha256": hashes, "assets_sha256": fingerprint}


def generation_provenance(args: argparse.Namespace, tokenizer: dict, summaries: list[dict]) -> dict:
    """Sidecar only: preserve the existing summary/one-task-per-row output schemas."""
    method = "retokenized_retained_thinking_text_length" if args.tokenizer else "retained_thinking_text_word_length"
    return {
        "schema_version": 1, "input_generation": "mechanism-v032",
        "producer": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
        "base_producer_sha256": "4deb338fb6d915f4e0fa4aa7b1b43add193c3a7c18518f43a333312761f911d6",
        "parameters": {"loop_threshold": args.loop_threshold, "max_output_tokens": args.max_output_tokens},
        "task_ids": None if args.task_ids is None else {"path": str(args.task_ids.resolve()), "sha256": file_sha256(args.task_ids)},
        "run_inputs": [{"path": str(path.resolve()), "task_results_sha256": file_sha256(path / "task_results.jsonl")}
                       for path in args.run_dir],
        "tokenizer": tokenizer,
        "metric_definitions": {
            "M1": {"label": "step_based_observed_nonempty_thinking_text_presence_proxy", "unit": "share_of_recorded_agent_steps",
                   "numerator": "agent steps with extra.event_type == thinking and a nonblank retained message",
                   "denominator": "all recorded agent steps", "aggregation": "per-task step share; run mean of defined task shares",
                   "raw_thinking_empty_vs_missing_identifiable": False},
            "M5": {"error_field_rule": "nonblank string -> error; blank string -> original fallback; present nonstring -> unknown",
                   "missing_error_field": "original status/exit_code/stderr/stdout rules unchanged"},
            "M6": {"label": method, "unit": tokenizer["unit"], "provider_thinking_token_count": False,
                   "text_scope": "retained nonblank messages with extra.event_type == thinking",
                   "aggregation": "per-task median/max, then run median of task medians"}},
        "runs": [{"run": summary["run"], "n_tasks": summary["n_tasks"]} for summary in summaries],
        "outputs": {"summary": {"path": str(args.json_out.resolve()), "sha256": file_sha256(args.json_out)},
                    "per_task": None if args.per_task is None else {"path": str(args.per_task.resolve()), "sha256": file_sha256(args.per_task)}}}


def thought_stats(thoughts: list[str], tokenize: Callable[[str], int]) -> dict:
    """M6: token-length median and max, and character-length median, of the has_thinking messages (None if none)."""
    tokens = [tokenize(text) for text in thoughts]
    return {
        "thought_tokens_median": _median(tokens),
        "thought_tokens_max": max(tokens, default=None),
        "thought_chars_median": _median([len(text) for text in thoughts]),
    }


def classify_taxonomy(loop: bool, n_calls: int, submitted: bool) -> str:
    """Taxonomy: loop, else no_parsed_tool_call, else submitted, else neither."""
    if loop:
        return "loop"
    if n_calls == 0:
        return "no_parsed_tool_call"
    return "submitted" if submitted else "neither"



def think_rate(steps: list[dict]) -> float | None:
    """M1: share of agent steps with has_thinking (presence of thinking text); None without steps."""
    return _share(sum(has_thinking(step) for step in steps), len(steps))


def repeat_stats(identities: list[tuple[str, str]], loop_threshold: int) -> dict:
    """M2: max_repeat (largest multiplicity of one call), loop (max_repeat >= threshold), distinct_ratio."""
    counts = Counter(identities)
    max_repeat = max(counts.values(), default=0)
    ratio = _share(len(counts), len(identities))
    return {"max_repeat": max_repeat, "loop": max_repeat >= loop_threshold, "distinct_ratio": ratio}


def submit_called(pairs: list[tuple[dict, Any]]) -> bool:
    """M3: True if any parsed tool call is submit_patch."""
    return any(call.get("function_name") == "submit_patch" for call, _ in pairs)


def compute_task_metrics(trace: dict, loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
                         max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                         tokenize: Callable[[str], int] = default_tokenize) -> dict:
    """M1-M6 and the taxonomy label for one parsed trace (a dict with a 'steps' list); input is not modified."""
    steps = [s for s in trace["steps"] if isinstance(s, dict) and s.get("source") == "agent"]
    labels = [classify_step(step, max_output_tokens) for step in steps]
    pairs = flatten_calls(steps)
    repeat = repeat_stats([tool_call_identity(call) for call, _ in pairs], loop_threshold)
    submitted = submit_called(pairs)
    thoughts = [_text(s.get("message")) for s in steps if has_thinking(s)]
    blank = sum(_event_type(s) == "thinking" and not has_thinking(s) for s in steps)
    return {
        "n_agent_steps": len(steps),
        "n_thinking_steps": len(thoughts),
        "thinking_empty_steps": blank,
        "label_counts": {label: labels.count(label) for label in STEP_LABELS},
        "think_rate": think_rate(steps),
        "n_tool_calls": len(pairs),
        **repeat,
        "submit_called": submitted,
        "opener": classify_opener(pairs[0][0] if pairs else None),
        "nonrecovery": nonrecovery(pairs),
        **thought_stats(thoughts, tokenize),
        "taxonomy": classify_taxonomy(repeat["loop"], len(pairs), submitted),
    }


def read_task_rows(path: Path) -> list[dict]:
    """Parse task_results.jsonl (blank lines skipped); a malformed row raises ValueError."""
    rows = []
    with path.open(encoding=READ_ENCODING) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
                raise ValueError(f"{path}:{number}: expected an object with a string instance_id")
            rows.append(row)
    return rows


def read_task_ids(path: Path) -> frozenset[str]:
    """Instance ids from a text file, one per line (blank lines ignored)."""
    stripped = (line.strip() for line in path.read_text(encoding=READ_ENCODING).splitlines())
    return frozenset(line for line in stripped if line)


def load_trace(run_dir: Path, instance_id: str) -> tuple[dict | None, str | None]:
    """Load traces/trace_<instance_id>.json as (trace, None), or (None, reason) when it is unusable:
    unsafe_id (path separator in the id), not_found, unreadable (I/O or JSON error), bad_structure.
    """
    if any(separator in instance_id for separator in PATH_SEPARATORS):
        return None, "unsafe_id"
    path = run_dir / "traces" / f"trace_{instance_id}.json"
    if not path.is_file():
        return None, "not_found"
    try:
        trace = json.loads(path.read_text(encoding=READ_ENCODING))
    except (OSError, ValueError, RecursionError):
        return None, "unreadable"
    if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list):
        return None, "bad_structure"
    return trace, None


def build_task_record(run_dir: Path, instance_id: str, compute: Callable[[dict], dict]) -> dict:
    """Per-task record: compute(trace) metrics, or all-None metrics plus trace_error when the trace is unusable."""
    trace, reason = load_trace(run_dir, instance_id)
    metrics = dict.fromkeys(compute({"steps": []})) if trace is None else compute(trace)
    return {"instance_id": instance_id, "trace_missing": trace is None, "trace_error": reason, **metrics}


def classify_error(error: Any) -> str:
    """Class of a task_results 'error' value: timeout, turn_cap, context_overflow, no_submit, other or none."""
    if error is None:
        return "none"
    return next((name for name, marker in ERROR_MARKERS.items() if marker in _text(error)), "other")


def result_stats(rows: list[dict]) -> dict:
    """Run-level numbers from task_results.jsonl rows (outcome fields are not used)."""
    classes = [classify_error(row.get("error")) for row in rows]
    return {
        "error_class_counts": {name: classes.count(name) for name in ERROR_CLASSES},
        "empty_patch_count": sum(1 for row in rows if row.get("agent_patch_size") == 0),
        "mean_total_llm_calls": _mean(row.get("total_llm_calls") for row in rows),
        "mean_duration_seconds": _mean(row.get("duration_seconds") for row in rows),
    }


def summarize_run(name: str, rows: list[dict], records: list[dict]) -> dict:
    """Run summary: means and shares over records that have a trace, plus result-side counts from rows."""
    present = [r for r in records if not r["trace_missing"]]
    total = len(present)
    rates = {s: [r["nonrecovery"][s]["repeat_rate"] for r in present] for s in STRATA}
    return {
        "run": name,
        "n_tasks": len(records), "missing_traces": len(records) - total,
        "think_rate_mean": _mean(r["think_rate"] for r in present),
        "thinking_steps": sum(r["n_thinking_steps"] for r in present),
        "thinking_empty_steps": sum(r["thinking_empty_steps"] for r in present),
        "loop_rate": _share(sum(r["loop"] for r in present), total),
        "max_repeat_median": _median(r["max_repeat"] for r in present),
        "distinct_ratio_mean": _mean(r["distinct_ratio"] for r in present),
        "submit_rate": _share(sum(r["submit_called"] for r in present), total),
        "opener_shares": {o: _share(sum(r["opener"] == o for r in present), total) for o in OPENERS},
        "nonrecovery": {s: {"mean_repeat_rate": _mean(v), "n_tasks": sum(x is not None for x in v)}
                        for s, v in rates.items()},
        "thought_tokens_median": _median(r["thought_tokens_median"] for r in present),
        "thought_chars_median": _median(r["thought_chars_median"] for r in present),
        "thought_n_tasks": sum(r["thought_tokens_median"] is not None for r in present),
        "taxonomy_counts": {t: sum(r["taxonomy"] == t for r in present) for t in TAXONOMY},
        "label_totals": {k: sum(r["label_counts"][k] for r in present) for k in STEP_LABELS},
        **result_stats(rows),
    }


def analyze_run(run_dir: Path, task_ids: frozenset[str] | None = None,
                compute: Callable[[dict], dict] = compute_task_metrics) -> tuple[dict, list[dict]]:
    """Analyze one run directory; return (run summary, per-task records), optionally limited to task_ids."""
    rows = read_task_rows(run_dir / "task_results.jsonl")
    if task_ids is not None:
        rows = [row for row in rows if row["instance_id"] in task_ids]
    name = run_dir.resolve().name or str(run_dir)
    records = [build_task_record(run_dir, row["instance_id"], compute) for row in rows]
    return summarize_run(name, rows, records), [{"run": name, **record} for record in records]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line (see README.md)."""
    parser = argparse.ArgumentParser(description="Mechanism metrics M1-M6 and failure taxonomy per run.")
    parser.add_argument("--run-dir", action="append", required=True, type=Path, metavar="DIR")
    parser.add_argument("--task-ids", type=Path, metavar="FILE", help="restrict to these ids, one per line")
    parser.add_argument("--loop-threshold", type=int, default=DEFAULT_LOOP_THRESHOLD)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--tokenizer", metavar="PATH", help="local Hugging Face tokenizer dir (needs transformers)")
    parser.add_argument("--json", dest="json_out", required=True, type=Path, metavar="OUT.json")
    parser.add_argument("--per-task", type=Path, metavar="OUT.jsonl")
    parser.add_argument("--provenance", dest="provenance_out", type=Path, metavar="OUT.provenance.json",
                        help="provenance sidecar (default: summary filename with .provenance.json suffix)")
    args = parser.parse_args(argv)
    if min(args.loop_threshold, args.max_output_tokens) < 1:
        parser.error("--loop-threshold and --max-output-tokens must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit status (2 on input errors)."""
    args = parse_args(argv)
    digest_keys = ("n_tasks", "missing_traces", "think_rate_mean", "loop_rate", "submit_rate")
    try:
        tokenize = default_tokenize if args.tokenizer is None else load_tokenizer(args.tokenizer)
        provenance_path = args.provenance_out or args.json_out.with_suffix(".provenance.json")
        if provenance_path.resolve() in {args.json_out.resolve(), None if args.per_task is None else args.per_task.resolve()}:
            raise ValueError("provenance sidecar must differ from metric output paths")
        tokenizer_metadata = tokenizer_provenance(args.tokenizer)
        task_ids = None if args.task_ids is None else read_task_ids(args.task_ids)
        compute = partial(compute_task_metrics, loop_threshold=args.loop_threshold,
                          max_output_tokens=args.max_output_tokens, tokenize=tokenize)
        results = [analyze_run(run_dir, task_ids, compute) for run_dir in args.run_dir]
        names = [summary["run"] for summary, _ in results]
        if len(set(names)) != len(names):
            raise ValueError("run directory names must be unique: " + ", ".join(names))
        if tokenizer_provenance(args.tokenizer) != tokenizer_metadata:
            raise RuntimeError("Tokenizer assets changed during metric computation")
        summaries = json.dumps([summary for summary, _ in results], indent=2) + "\n"
        args.json_out.write_bytes(summaries.encode("utf-8"))
        if args.per_task is not None:
            lines = "".join(json.dumps(record) + "\n" for _, records in results for record in records)
            args.per_task.write_bytes(lines.encode("utf-8"))
        provenance = generation_provenance(args, tokenizer_metadata, [summary for summary, _ in results])
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for summary, _ in results:
        print(summary["run"], {key: summary[key] for key in digest_keys})
    return 0


if __name__ == "__main__":
    sys.exit(main())
