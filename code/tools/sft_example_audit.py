"""Read-only audit of how SWE-agent trajectories were packed into LoRA-SFT chunks.

Standard library only (Python 3.8+). trajectories.jsonl is streamed line by line. Only the four
files in --out are written: chunks.jsonl, summary.json, summary.md and inputs.sha256.

Input. One trajectory per line, with text_segments (per chain, in step order, a (pre_text,
train=False) and a (comp_text, train=True) pair per item) and chunks (greedy splits of each chain at
item boundaries; a chunk that starts at item k>0 is the chain header plus ONE later tool
response, the intervening history dropped; an item that does not fit is in no chunk).
labels[i] == input_ids[i] on supervised tokens and -100 elsewhere. Token ids come from
--marker-ids and are never hard-coded: the keys "<|turn>", "<turn|>", "<channel|>",
"<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>" and "channel_open_ids"
(the ids of the opener text "<|channel>thought" + newline).

History. chain_first_step = min(steps[0]) over the chunks of one chain; prior_items_absent =
steps[0] - chain_first_step; history_dropped = prior_items_absent > 0.

Runs. A loss run is a maximal run of labels != -100 (n_runs). alignment_uncertain = (n_runs !=
steps[1] - steps[0] + 1); it is also set when the chain's train-segment count differs from the
chain's total run count (an item was dropped, so the offset rule below would shift). Per run: start,
length, run_prefix_len (tokens before the first <channel|>, -1 if none), opener_in_run (the run
starts with exactly channel_open_ids), thought_tokens (run_prefix_len minus the opener length when
opener_in_run; 0 without <channel|>), empty_thought (run_prefix_len >= 0 and thought_tokens == 0)
and has_channel (run_prefix_len >= 0).

Categories. channel_markers = opener ids (when opener_in_run) + the first <channel|>; thought =
thought_tokens; tool_call = tokens inside every <|tool_call> ... <tool_call|> span, both markers
included (an unclosed span runs to the end of the run); tool_response_opener and turn_end = the
<|tool_response> and <turn|> ids outside such spans; other = the rest. A run whose six categories do
not sum to its length is counted in category_sum_violations.

Orphans. orphan_tool_response = history_dropped and, in the no-loss span after the LAST <|turn> id
before the first loss run (that id excluded; preamble_tokens = the span length), a <tool_response|>
id is present and no <|tool_response> id is present.

Text flags. The j-th loss run of a chunk is the (offset + j)-th train segment of its chain (offset =
runs in the chain's earlier chunks). wrapper: comp_text contains __public_sft_output=;
canonical_opener: comp_text contains call:run_command{command:<|"|> immediately followed by
find /workspace -maxdepth [12] -not -path; submit_patch: comp_text contains call:submit_patch. All
three are null for the runs of an alignment-uncertain chunk.

Aggregates. For every arm key and every arm x tier: counts and shares of history-dropped chunks and
loss tokens, orphan chunks (share of history-dropped chunks), empty-thought runs and tokens, thought
tokens, category shares, text-flag runs (share of runs with known flags), dropped items, and the
0-based chunk position of every supervised token (nearest-rank p50/p90 and max, token-weighted).
joint = the 8 cells history_dropped x empty_thought x (wrapper or canonical_opener), with runs and
loss tokens. Loss tokens are recounted from the labels; chunks whose recorded n_loss_tokens differs
are counted in loss_count_mismatch_chunks.

Served reference. Per --trace-dir (files DIR/traces/trace_*.json) and pooled: step counts, the
prompt_tokens distribution of served LLM steps (a metrics dict with a numeric prompt_tokens) and,
per arm, the share of those steps whose prompt_tokens exceeds the arm's loss_pos_p90.
"""
import argparse
import glob
import hashlib
import itertools
import json
import logging
import os
import re
import statistics
import sys
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

NL, TAB = chr(10), chr(9)
IGNORE_INDEX = -100
PROGRESS_EVERY = 200
HASH_BLOCK_BYTES = 1 << 20
P_MEDIAN, P_HIGH = 50, 90
OUTPUT_FILES: Tuple[str, ...] = ("chunks.jsonl", "summary.json", "summary.md", "inputs.sha256")
DEFAULT_ARM_KEYS: Tuple[str, ...] = (
    "arm_ordinary",
    "arm_short",
    "arm_ordinary_tokmatched",
    "all_resolved",
    "arm_ordinary_multi",
    "arm_short_multi",
    "arm_short_calls",
)
CATEGORIES: Tuple[str, ...] = (
    "channel_markers",
    "thought",
    "tool_call",
    "tool_response_opener",
    "turn_end",
    "other",
)
TEXT_FLAGS: Tuple[str, ...] = ("wrapper", "canonical_opener", "submit_patch")
NO_FLAGS: Mapping[str, Optional[bool]] = dict.fromkeys(TEXT_FLAGS)
TRAJECTORY_FIELDS: Tuple[str, ...] = (
    "n_train_segments",
    "n_runs_total",
    "dropped_items",
    "loss_tokens_match",
)
SERVED_COUNTS: Tuple[str, ...] = (
    "n_traces",
    "n_unreadable_traces",
    "n_steps_with_metrics",
    "n_llm_steps_without_metrics",
    "n_other_steps",
)
WRAPPER_TEXT = "__public_sft_output="
SUBMIT_PATCH_TEXT = "call:submit_patch"
CANONICAL_OPENER_RE = re.compile(re.escape('call:run_command{command:<|"|>')
                                 + r"find /workspace -maxdepth [12] -not -path")
MARKER_KEYS: Tuple[str, ...] = (
    "<|turn>",
    "<turn|>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
)


class Markers(NamedTuple):
    """Tokenizer ids read from markers.json."""

    turn_open: int
    turn_close: int
    channel_close: int
    call_open: int
    call_close: int
    resp_open: int
    resp_close: int
    channel_open_ids: Tuple[int, ...]


Json = Dict[str, Any]
Run = Tuple[int, int]
Members = Dict[str, FrozenSet[str]]
TextFlags = Mapping[str, Optional[bool]]
Column = Tuple[str, Callable[[Json], Any]]
LOG = logging.getLogger("sft_example_audit")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline=NL) as fh:
        fh.write(text)


def load_markers(path: str) -> Markers:
    """Read the tokenizer marker ids from markers.json (no id is hard-coded in this script)."""
    raw = load_json(path)
    wanted = MARKER_KEYS + ("channel_open_ids",)
    if not isinstance(raw, dict) or any(key not in raw for key in wanted):
        raise ValueError("{}: need a JSON object with the keys {}".format(path, list(wanted)))
    opener = raw["channel_open_ids"]
    ids = [raw[key] for key in MARKER_KEYS]
    opener_ok = isinstance(opener, list) and opener and all(is_int(i) for i in opener)
    if not (all(is_int(i) for i in ids) and opener_ok):
        raise ValueError(
            "{}: marker ids must be ints, channel_open_ids a non-empty int list".format(path)
        )
    return Markers(*ids, channel_open_ids=tuple(opener))


def load_arms(path: str, arm_keys: Sequence[str]) -> Tuple[Members, Json]:
    """Return ({arm: frozenset(traj_ids)}, report).

    Requested keys that are absent are reported as missing; keys not requested are ignored.
    """
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("{}: arms file must be a JSON object".format(path))
    members, missing, invalid = {}, [], []
    for key in arm_keys:
        value = raw.get(key)
        if key not in raw:
            missing.append(key)
        elif isinstance(value, list) and all(isinstance(i, str) for i in value):
            members[key] = frozenset(value)
        else:
            invalid.append(key)
    report = {"requested": list(arm_keys), "missing": missing, "invalid": invalid,
              "n_ids": {key: len(ids) for key, ids in members.items()}}
    return members, report


def loss_runs(labels: Sequence[int]) -> List[Run]:
    """Maximal runs of labels != -100 as (start, length) pairs."""
    runs, start = [], None
    for i, label in enumerate(labels):
        if label != IGNORE_INDEX:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(labels) - start))
    return runs


def run_stats(ids: List[int], markers: Markers) -> Json:
    """run_prefix_len, opener_in_run, thought_tokens, empty_thought, has_channel of one loss run."""
    opener = markers.channel_open_ids
    prefix = ids.index(markers.channel_close) if markers.channel_close in ids else -1
    opener_in_run = tuple(ids[:len(opener)]) == opener
    thought = prefix - (len(opener) if opener_in_run else 0) if prefix >= 0 else 0
    return {"run_prefix_len": prefix, "opener_in_run": opener_in_run, "thought_tokens": thought,
            "empty_thought": prefix >= 0 and thought == 0, "has_channel": prefix >= 0}


def categorize_run(ids: Sequence[int], stats: Json, markers: Markers) -> Dict[str, int]:
    """Token count per category; `other` is the remainder, floored at 0.

    Overlapping categories therefore show up as a sum that differs from the run length.
    """
    in_call, n_call, n_resp, n_turn = False, 0, 0, 0
    for tok in ids:
        if in_call:
            n_call += 1
            in_call = tok != markers.call_close
        elif tok == markers.call_open:
            n_call += 1
            in_call = True
        elif tok == markers.resp_open:
            n_resp += 1
        elif tok == markers.turn_close:
            n_turn += 1
    opener_ids = len(markers.channel_open_ids) if stats["opener_in_run"] else 0
    channel = opener_ids + int(stats["has_channel"])
    thought = stats["thought_tokens"]
    other = max(len(ids) - (channel + thought + n_call + n_resp + n_turn), 0)
    return {"channel_markers": channel, "thought": thought, "tool_call": n_call,
            "tool_response_opener": n_resp, "turn_end": n_turn, "other": other}


def run_record(run_ids: List[int], start: int, markers: Markers) -> Json:
    """Per-run record (without text flags) for the tokens of one loss run starting at `start`."""
    stats = run_stats(run_ids, markers)
    return dict(start=start, length=len(run_ids), **stats,
                cat=categorize_run(run_ids, stats, markers))


def orphan_check(
    input_ids: Sequence[int],
    first_start: Optional[int],
    history_dropped: bool,
    markers: Markers,
) -> Tuple[bool, Optional[int]]:
    """(orphan_tool_response, preamble_tokens) from the no-loss span after the last <|turn> id
    that precedes the first loss run (first_start). Both are falsy/None without a loss run."""
    if first_start is None:
        return False, None
    turn_at = -1
    for i in range(first_start - 1, -1, -1):
        if input_ids[i] == markers.turn_open:
            turn_at = i
            break
    span = input_ids[turn_at + 1:first_start]
    orphan = history_dropped and markers.resp_close in span and markers.resp_open not in span
    return orphan, len(span)


def text_flags(text: str) -> Dict[str, bool]:
    """The three comp_text flags of one train segment."""
    return {"wrapper": WRAPPER_TEXT in text,
            "canonical_opener": CANONICAL_OPENER_RE.search(text) is not None,
            "submit_patch": SUBMIT_PATCH_TEXT in text}


def train_texts_by_chain(segments: Iterable[Json]) -> Dict[int, List[str]]:
    """{chain: [comp_text, ...]} for the train=True segments, in file order."""
    texts = {}
    for seg in segments:
        if seg["train"]:
            texts.setdefault(seg["chain"], []).append(seg["text"])
    return texts


def align_text_flags(
    chain_texts: List[str], offset: int, n_runs: int, chain_run_total: int
) -> Optional[List[TextFlags]]:
    """Text flags of a chunk's runs; run j takes the (offset + j)-th train segment of its chain.

    Returns None when the chain's train-segment count differs from its run count, because a
    dropped item would shift the offsets.
    """
    if chain_run_total != len(chain_texts):
        return None
    texts = chain_texts[offset:offset + n_runs]
    return [text_flags(text) for text in texts] if len(texts) == n_runs else None


def require(obj: Json, keys: Iterable[str], what: str) -> None:
    """Fail fast with a readable message when a record lacks required keys."""
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError("{} lacks keys {}".format(what, missing))


def check_chunk(chunk: Json, traj_id: str) -> None:
    what = "trajectory {} chunk".format(traj_id)
    require(chunk, ("chain", "steps", "n_tokens", "n_loss_tokens", "input_ids", "labels"), what)
    if len(chunk["input_ids"]) != len(chunk["labels"]) or len(chunk["steps"]) != 2:
        raise ValueError("{}: input_ids/labels lengths differ or steps is not a pair".format(what))


def chunk_row(
    meta: Json,
    chunk: Json,
    runs: List[Run],
    chain_first_step: int,
    flags: Optional[List[TextFlags]],
    markers: Markers,
) -> Json:
    """One chunks.jsonl row. flags = per-run text flags of this chunk, or None when unalignable."""
    ids = chunk["input_ids"]
    first, last = chunk["steps"]
    records = [run_record(ids[s:s + n], s, markers) for s, n in runs]
    span_mismatch = len(runs) != last - first + 1
    uncertain = span_mismatch or flags is None
    run_flags = [NO_FLAGS] * len(runs) if span_mismatch or flags is None else flags
    merged = [{**record, **flag} for record, flag in zip(records, run_flags)]
    dropped = first - chain_first_step > 0
    orphan, preamble = orphan_check(ids, runs[0][0] if runs else None, dropped, markers)
    positions = [p for s, n in runs for p in range(s, s + n)]
    return {
        "traj_id": meta["traj_id"], "tier": meta["tier"], "task_id": meta["task_id"],
        "arms": meta["arms"], "chain": chunk["chain"], "steps": [first, last],
        "n_tokens": chunk["n_tokens"], "n_loss_tokens": chunk["n_loss_tokens"],
        "loss_tokens_counted": len(positions), "chain_first_step": chain_first_step,
        "prior_items_absent": first - chain_first_step, "history_dropped": dropped,
        "n_runs": len(runs), "alignment_uncertain": uncertain,
        "orphan_tool_response": orphan, "preamble_tokens": preamble,
        "loss_pos_median": statistics.median(positions) if positions else None,
        "loss_pos_max": positions[-1] if positions else None,
        "category_sum_violations": sum(1 for r in merged if sum(r["cat"].values()) != r["length"]),
        "runs": merged,
    }


def trajectory_stats(
    traj: Json,
    meta: Json,
    texts: Dict[int, List[str]],
    run_total: Dict[int, int],
    chunks: List[Json],
) -> Json:
    """Per-trajectory counts; loss_tokens_match is None when an item was dropped."""
    n_train = sum(len(chain_texts) for chain_texts in texts.values())
    n_runs = sum(run_total.values())
    dropped = n_train - n_runs
    match = None
    if dropped == 0:
        match = sum(c["n_loss_tokens"] for c in chunks) == traj["loss_tokens"]
    return {"traj_id": meta["traj_id"], "tier": meta["tier"], "arms": meta["arms"],
            "n_train_segments": n_train, "n_runs_total": n_runs, "dropped_items": dropped,
            "loss_tokens_match": match}


def audit_trajectory(traj: Json, members: Members, markers: Markers) -> Tuple[Json, List[Json]]:
    """(traj_stats, chunk rows) of one parsed trajectory; the input record is not modified."""
    required = ("traj_id", "tier", "task_id", "loss_tokens", "text_segments", "chunks")
    require(traj, required, "trajectory")
    traj_id, chunks = traj["traj_id"], traj["chunks"]
    for chunk in chunks:
        check_chunk(chunk, traj_id)
    runs_per_chunk = [loss_runs(chunk["labels"]) for chunk in chunks]
    chain_first, run_total = {}, {}
    for chunk, runs in zip(chunks, runs_per_chunk):
        chain, first = chunk["chain"], chunk["steps"][0]
        chain_first[chain] = min(chain_first.get(chain, first), first)
        run_total[chain] = run_total.get(chain, 0) + len(runs)
    texts = train_texts_by_chain(traj["text_segments"])
    meta = {"traj_id": traj_id, "tier": traj["tier"], "task_id": traj["task_id"],
            "arms": sorted(arm for arm, ids in members.items() if traj_id in ids)}
    rows, offset = [], {}
    for chunk, runs in zip(chunks, runs_per_chunk):
        chain = chunk["chain"]
        chain_texts = texts.get(chain, [])
        flags = align_text_flags(chain_texts, offset.get(chain, 0), len(runs), run_total[chain])
        rows.append(chunk_row(meta, chunk, runs, chain_first[chain], flags, markers))
        offset[chain] = offset.get(chain, 0) + len(runs)
    return trajectory_stats(traj, meta, texts, run_total, chunks), rows


def rank_of(pct: int, n: int) -> int:
    """1-based nearest rank of the pct-th percentile among n values (integer arithmetic)."""
    return (pct * n + 99) // 100


def percentile(sorted_values: Sequence[float], pct: int) -> Optional[float]:
    """Nearest-rank percentile of an ascending list; None when empty."""
    if not sorted_values:
        return None
    return sorted_values[rank_of(pct, len(sorted_values)) - 1]


def value_at_rank(segments: List[Tuple[int, int, int]], rank: int) -> Optional[int]:
    """Position of the rank-th supervised token; segments are (start, end, tokens_per_position)."""
    seen = 0
    for start, end, count in segments:
        size = (end - start) * count
        if seen + size >= rank:
            return start + (rank - seen - 1) // count
        seen += size
    return None


def position_quantiles(diff: Dict[int, int]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Token-weighted (p50, p90, max) positions from a {position: delta} difference array."""
    segments, running, prev = [], 0, None
    for pos in sorted(diff):
        if running > 0:
            segments.append((prev, pos, running))
        running += diff[pos]
        prev = pos
    total = sum((end - start) * count for start, end, count in segments)
    if total == 0:
        return None, None, None
    return (value_at_rank(segments, rank_of(P_MEDIAN, total)),
            value_at_rank(segments, rank_of(P_HIGH, total)), segments[-1][1] - 1)


def share(count: int, denominator: int) -> Optional[float]:
    return count / denominator if denominator else None


def count_share(count: int, denominator: int) -> Json:
    return {"count": count, "share": share(count, denominator)}


def new_acc() -> Json:
    """Mutable per-group counters; they live only inside aggregate()."""
    return {"n_trajectories": 0, "n_chunks": 0, "n_runs": 0, "n_runs_text_known": 0,
            "loss_tokens": 0, "hd_chunks": 0, "hd_loss_tokens": 0, "orphan_chunks": 0,
            "empty_runs": 0, "empty_loss_tokens": 0, "thought_tokens": 0,
            "cat": dict.fromkeys(CATEGORIES, 0), "wrapper": 0, "canonical": 0, "submit": 0,
            "dropped_items": 0, "uncertain_chunks": 0, "sum_violations": 0,
            "loss_count_mismatch": 0, "joint": {}, "joint_unknown": [0, 0], "pos_diff": {}}


def acc_add_run(acc: Json, run: Json, history_dropped: bool) -> None:
    """Add one run record to a group's counters."""
    length, start = run["length"], run["start"]
    acc["thought_tokens"] += run["thought_tokens"]
    for name in CATEGORIES:
        acc["cat"][name] += run["cat"][name]
    if run["empty_thought"]:
        acc["empty_runs"] += 1
        acc["empty_loss_tokens"] += length
    diff = acc["pos_diff"]
    diff[start] = diff.get(start, 0) + 1
    diff[start + length] = diff.get(start + length, 0) - 1
    if run["wrapper"] is None:
        acc["joint_unknown"][0] += 1
        acc["joint_unknown"][1] += length
        return
    acc["n_runs_text_known"] += 1
    acc["wrapper"] += int(run["wrapper"])
    acc["canonical"] += int(run["canonical_opener"])
    acc["submit"] += int(run["submit_patch"])
    key = (history_dropped, run["empty_thought"], run["wrapper"] or run["canonical_opener"])
    cell = acc["joint"].setdefault(key, [0, 0])
    cell[0] += 1
    cell[1] += length


def acc_add_chunk(acc: Json, row: Json) -> None:
    """Add one chunk row (and its runs) to a group's counters."""
    n_counted = row["loss_tokens_counted"]
    acc["n_chunks"] += 1
    acc["n_runs"] += row["n_runs"]
    acc["loss_tokens"] += n_counted
    acc["uncertain_chunks"] += int(row["alignment_uncertain"])
    acc["sum_violations"] += row["category_sum_violations"]
    acc["loss_count_mismatch"] += int(n_counted != row["n_loss_tokens"])
    if row["history_dropped"]:
        acc["hd_chunks"] += 1
        acc["hd_loss_tokens"] += n_counted
        acc["orphan_chunks"] += int(row["orphan_tool_response"])
    for run in row["runs"]:
        acc_add_run(acc, run, row["history_dropped"])


def joint_table(acc: Json) -> Json:
    """The 8 joint cells (history_dropped x empty_thought x text flag) plus unknown-text runs."""
    cells = []
    for dropped, empty, wrapped in itertools.product((False, True), repeat=3):
        runs, tokens = acc["joint"].get((dropped, empty, wrapped), (0, 0))
        cells.append({"history_dropped": dropped, "empty_thought": empty,
                      "wrapper_or_canonical_opener": wrapped, "runs": runs, "loss_tokens": tokens})
    unknown = {"runs": acc["joint_unknown"][0], "loss_tokens": acc["joint_unknown"][1]}
    return {"cells": cells, "text_unknown": unknown}


def finalize_group(acc: Json) -> Json:
    """JSON-ready metrics of one group (counts, shares, loss positions, joint table)."""
    tokens, n_runs, known = acc["loss_tokens"], acc["n_runs"], acc["n_runs_text_known"]
    p50, p90, p_max = position_quantiles(acc["pos_diff"])
    return {
        "n_trajectories": acc["n_trajectories"], "n_chunks": acc["n_chunks"], "n_runs": n_runs,
        "loss_tokens": tokens,
        "history_dropped_chunks": count_share(acc["hd_chunks"], acc["n_chunks"]),
        "history_dropped_loss_tokens": count_share(acc["hd_loss_tokens"], tokens),
        "orphan_chunks": count_share(acc["orphan_chunks"], acc["hd_chunks"]),
        "empty_thought_runs": count_share(acc["empty_runs"], n_runs),
        "empty_thought_loss_tokens": count_share(acc["empty_loss_tokens"], tokens),
        "thought_tokens_total": acc["thought_tokens"],
        "category_tokens": dict(acc["cat"]),
        "category_shares": {name: share(value, tokens) for name, value in acc["cat"].items()},
        "n_runs_text_known": known,
        "wrapper_runs": count_share(acc["wrapper"], known),
        "canonical_opener_runs": count_share(acc["canonical"], known),
        "submit_patch_runs": count_share(acc["submit"], known),
        "dropped_items_total": acc["dropped_items"],
        "alignment_uncertain_chunks": acc["uncertain_chunks"],
        "category_sum_violations": acc["sum_violations"],
        "loss_count_mismatch_chunks": acc["loss_count_mismatch"],
        "loss_pos_p50": p50, "loss_pos_p90": p90, "loss_pos_max": p_max,
        "joint": joint_table(acc),
    }


def aggregate(results: Iterable[Tuple[Json, List[Json]]], arm_keys: Sequence[str]) -> Json:
    """Consume (traj_stats, rows) pairs once; return aggregates per arm and arm x tier."""
    accs = {(arm, None): new_acc() for arm in arm_keys}
    trajectories, n_traj, duplicates = {}, 0, 0
    for stats, rows in results:
        n_traj += 1
        duplicates += int(stats["traj_id"] in trajectories)
        trajectories[stats["traj_id"]] = {key: stats[key] for key in TRAJECTORY_FIELDS}
        for arm in stats["arms"]:
            for key in ((arm, None), (arm, stats["tier"])):
                acc = accs.setdefault(key, new_acc())
                acc["n_trajectories"] += 1
                acc["dropped_items"] += stats["dropped_items"]
                for row in rows:
                    acc_add_chunk(acc, row)
    by_arm = {}
    for arm in arm_keys:
        tiers = sorted(tier for (name, tier) in accs if name == arm and tier is not None)
        by_arm[arm] = {"overall": finalize_group(accs[(arm, None)]),
                       "tiers": {tier: finalize_group(accs[(arm, tier)]) for tier in tiers}}
    return {"n_trajectories": n_traj, "duplicate_traj_ids": duplicates,
            "aggregates": by_arm, "trajectories": trajectories}


def list_trace_files(trace_dir: str) -> List[str]:
    if not os.path.isdir(trace_dir):
        raise ValueError("trace dir not found: {}".format(trace_dir))
    pattern = os.path.join(glob.escape(trace_dir), "traces", "trace_*.json")
    return sorted(glob.glob(pattern))


def classify_step(step: Any) -> Tuple[str, Optional[float]]:
    """('metrics', prompt_tokens) for a served LLM step, ('llm_no_metrics', None) for a step with
    llm_call_count but no metrics dict, else ('other', None)."""
    if not isinstance(step, dict):
        return "other", None
    metrics = step.get("metrics")
    if isinstance(metrics, dict):
        tokens = metrics.get("prompt_tokens")
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            return "metrics", tokens
        return "other", None
    if "llm_call_count" in step:
        return "llm_no_metrics", None
    return "other", None


def scan_trace_dir(trace_dir: str) -> Tuple[Dict[str, int], List[float]]:
    """(step counts, prompt_tokens list) over the trace_*.json files of one trace dir."""
    files = list_trace_files(trace_dir)
    counts = dict.fromkeys(SERVED_COUNTS, 0)
    counts["n_traces"] = len(files)
    count_key = {"metrics": "n_steps_with_metrics", "llm_no_metrics": "n_llm_steps_without_metrics",
                 "other": "n_other_steps"}
    prompts = []
    for path in files:
        try:
            steps = load_json(path)["steps"]
        except (OSError, ValueError, KeyError, TypeError):
            steps = None
        if not isinstance(steps, list):
            counts["n_unreadable_traces"] += 1
            continue
        for step in steps:
            kind, tokens = classify_step(step)
            counts[count_key[kind]] += 1
            if kind == "metrics":
                prompts.append(tokens)
    return counts, prompts


def prompt_summary(prompts: Sequence[float]) -> Json:
    ordered = sorted(prompts)
    return {"n": len(ordered), "p50": percentile(ordered, P_MEDIAN),
            "p90": percentile(ordered, P_HIGH), "max": ordered[-1] if ordered else None}


def served_entry(
    label: str,
    counts: Dict[str, int],
    prompts: Sequence[float],
    arm_p90: Dict[str, Optional[int]],
) -> Json:
    """Step counts, prompt_tokens distribution and per-arm share of steps above the train p90."""
    shares: Dict[str, Optional[float]] = {}
    for arm, p90 in arm_p90.items():
        above = sum(1 for tokens in prompts if p90 is not None and tokens > p90)
        shares[arm] = share(above, len(prompts)) if p90 is not None else None
    return {"dir": label, **counts, "prompt_tokens": prompt_summary(prompts),
            "share_served_steps_above_train_p90": shares}


def served_stats(trace_dirs: Sequence[str], arm_p90: Dict[str, Optional[int]]) -> Json:
    """Served reference per trace dir and pooled; arm_p90 = {arm: arm loss_pos_p90 or None}."""
    per_dir, pooled_counts, pooled_prompts = [], dict.fromkeys(SERVED_COUNTS, 0), []
    for trace_dir in trace_dirs:
        counts, prompts = scan_trace_dir(trace_dir)
        per_dir.append(served_entry(trace_dir, counts, prompts, arm_p90))
        pooled_counts = {key: pooled_counts[key] + counts[key] for key in SERVED_COUNTS}
        pooled_prompts.extend(prompts)
    return {"arm_train_p90": dict(arm_p90), "per_dir": per_dir,
            "pooled": served_entry("pooled", pooled_counts, pooled_prompts, arm_p90)}


def fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.2f}%".format(100.0 * value)


def fmt_entry(entry: Json) -> str:
    return "{} ({})".format(entry["count"], fmt_pct(entry["share"]))


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    """Markdown table as a list of lines."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def plain_col(title: str, key: str) -> Column:
    return title, lambda group: group[key]


def entry_col(title: str, key: str) -> Column:
    return title, lambda group: fmt_entry(group[key])


def share_col(name: str) -> Column:
    return name, lambda group: fmt_pct(group["category_shares"][name])


def position_col(group: Json) -> str:
    p50, p90, p_max = group["loss_pos_p50"], group["loss_pos_p90"], group["loss_pos_max"]
    return "{} / {} / {}".format(p50, p90, p_max)


GROUP_TABLES = (
    ("Population", (plain_col("trajectories", "n_trajectories"), plain_col("chunks", "n_chunks"),
                    plain_col("runs", "n_runs"), plain_col("loss tokens", "loss_tokens"))),
    ("History-dropped chunks and orphan tool responses",
     (entry_col("history-dropped chunks", "history_dropped_chunks"),
      entry_col("history-dropped loss tokens", "history_dropped_loss_tokens"),
      entry_col("orphan chunks (of history-dropped)", "orphan_chunks"))),
    ("Empty thought blocks",
     (entry_col("empty-thought runs", "empty_thought_runs"),
      entry_col("empty-thought loss tokens", "empty_thought_loss_tokens"),
      plain_col("thought tokens", "thought_tokens_total"))),
    ("Loss-token categories (share of loss tokens)", tuple(share_col(name) for name in CATEGORIES)),
    ("Text flags (share of runs with known flags)",
     (entry_col("wrapper", "wrapper_runs"), entry_col("canonical opener", "canonical_opener_runs"),
      entry_col("submit_patch", "submit_patch_runs"),
      plain_col("runs with known flags", "n_runs_text_known"))),
    ("Loss position and integrity counters",
     (("loss pos p50 / p90 / max", position_col), plain_col("dropped items", "dropped_items_total"),
      plain_col("alignment-uncertain chunks", "alignment_uncertain_chunks"),
      plain_col("category-sum violations", "category_sum_violations"),
      plain_col("loss-count mismatches", "loss_count_mismatch_chunks"))),
)


def group_rows(summary: Json) -> Iterator[Tuple[str, Json]]:
    """(label, group) for every arm and every arm x tier, in summary order."""
    for arm, sections in summary["aggregates"].items():
        yield arm, sections["overall"]
        for tier, group in sections["tiers"].items():
            yield "{} / {}".format(arm, tier), group


def arm_status(key: str, arms: Json) -> str:
    if key in arms["missing"]:
        return "missing"
    return "invalid" if key in arms["invalid"] else "ok"


def arms_section(summary: Json) -> List[str]:
    arms, rows = summary["arms"], []
    for key in arms["requested"]:
        status = arm_status(key, arms)
        matched = summary["aggregates"][key]["overall"]["n_trajectories"] if status == "ok" else ""
        rows.append([key, status, arms["n_ids"].get(key, ""), matched])
    headers = ["arm key", "status", "ids listed", "trajectories matched"]
    return ["## Arms", ""] + md_table(headers, rows) + [""]


def joint_section(summary: Json) -> List[str]:
    rows = []
    for label, group in group_rows(summary):
        for cell in group["joint"]["cells"]:
            if cell["runs"]:
                rows.append([label, cell["history_dropped"], cell["empty_thought"],
                             cell["wrapper_or_canonical_opener"], cell["runs"],
                             cell["loss_tokens"]])
        unknown = group["joint"]["text_unknown"]
        if unknown["runs"]:
            rows.append([label, "?", "?", "? (text flags null)", unknown["runs"],
                         unknown["loss_tokens"]])
    headers = ["group", "history_dropped", "empty_thought", "wrapper or canonical_opener",
               "runs", "loss tokens"]
    note = "Cells without runs are omitted here; summary.json lists all eight per group."
    return ["## Joint table", "", note, ""] + md_table(headers, rows) + [""]


def served_section(served: Optional[Json]) -> List[str]:
    if served is None:
        return []
    arms = list(served["arm_train_p90"])
    headers = (["trace dir", "traces", "unreadable", "steps with metrics",
                "LLM steps without metrics", "other steps", "n", "prompt p50", "prompt p90",
                "prompt max"]
               + ["above train p90: " + arm for arm in arms])
    rows = []
    for entry in served["per_dir"] + [served["pooled"]]:
        prompt = entry["prompt_tokens"]
        shares = [fmt_pct(entry["share_served_steps_above_train_p90"][arm]) for arm in arms]
        rows.append([entry["dir"], entry["n_traces"], entry["n_unreadable_traces"],
                     entry["n_steps_with_metrics"], entry["n_llm_steps_without_metrics"],
                     entry["n_other_steps"], prompt["n"], prompt["p50"], prompt["p90"],
                     prompt["max"]] + shares)
    p90_rows = [[arm, value] for arm, value in served["arm_train_p90"].items()]
    return (["## Served reference", ""] + md_table(headers, rows)
            + ["", "Train loss_pos_p90 per arm:", ""]
            + md_table(["arm", "loss_pos_p90"], p90_rows) + [""])


def write_markdown(summary: Json, path: str) -> None:
    """summary.md: one table per metric family (arm and arm x tier rows), then joint, served."""
    intro = ("Trajectories audited: {} (limit: {}). Definitions: module docstring of "
             "sft_example_audit.py. Input hashes: inputs.sha256.")
    n_audited, limit = summary["n_trajectories_audited"], summary["limit"]
    lines = ["# SFT example audit", "", intro.format(n_audited, limit), ""]
    lines += arms_section(summary)
    for title, columns in GROUP_TABLES:
        rows = [[label] + [fn(group) for _, fn in columns] for label, group in group_rows(summary)]
        headers = ["group"] + [name for name, _ in columns]
        lines += ["## " + title, ""] + md_table(headers, rows) + [""]
    lines += joint_section(summary) + served_section(summary["served"])
    write_text(path, NL.join(lines) + NL)


def read_trajectories(path: str, limit: Optional[int] = None) -> Iterator[Json]:
    """Yield parsed trajectories line by line (blank lines skipped), at most `limit` of them."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if limit is not None and count >= limit:
                return
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError("{}: line {}: {}".format(path, lineno, exc)) from exc
            count += 1
            yield record


def audit_stream(
    data_path: str,
    limit: Optional[int],
    members: Members,
    markers: Markers,
    sink: TextIO,
) -> Iterator[Tuple[Json, List[Json]]]:
    """Yield (traj_stats, rows) per trajectory; write each chunk row to `sink` as a JSON line."""
    n_chunks = 0
    for n, traj in enumerate(read_trajectories(data_path, limit), 1):
        stats, rows = audit_trajectory(traj, members, markers)
        for row in rows:
            sink.write(json.dumps(row, separators=(",", ":")) + NL)
        n_chunks += len(rows)
        if n % PROGRESS_EVERY == 0:
            LOG.info("%d trajectories, %d chunks", n, n_chunks)
        yield stats, rows


def file_digest(path: str) -> Tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def describe_inputs(
    files: Sequence[Tuple[str, str]], trace_dirs: Sequence[str]
) -> Tuple[List[Json], List[Json]]:
    """files = [(role, path)]; returns (file entries, trace-dir entries) for the input records."""
    entries = []
    for role, path in files:
        sha, size = file_digest(path)
        entries.append({"role": role, "path": os.path.abspath(path), "sha256": sha,
                        "size_bytes": size})
    dirs = [{"path": os.path.abspath(d), "n_trace_files": len(list_trace_files(d))}
            for d in trace_dirs]
    return entries, dirs


def inputs_text(entries: Sequence[Json], dirs: Sequence[Json]) -> str:
    lines = [TAB.join(("# sha256", "size_bytes", "role", "path"))]
    lines += [TAB.join((e["sha256"], str(e["size_bytes"]), e["role"], e["path"])) for e in entries]
    lines += [TAB.join(("-", "n_trace_files=" + str(d["n_trace_files"]), "trace_dir", d["path"]))
              for d in dirs]
    return NL.join(lines) + NL


def guard_outputs(out_dir: str, input_paths: Iterable[str]) -> None:
    """Refuse to run when one of the four output files would overwrite an input file."""
    inputs = {os.path.normcase(os.path.realpath(p)) for p in input_paths}
    for name in OUTPUT_FILES:
        target = os.path.join(out_dir, name)
        if os.path.normcase(os.path.realpath(target)) in inputs:
            raise ValueError("refusing to overwrite the input file {}".format(target))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit of packed SFT chunks.",
                                     allow_abbrev=False)
    parser.add_argument("--data", required=True, help="trajectories.jsonl (streamed)")
    parser.add_argument("--arms", required=True, help="arms.json")
    parser.add_argument("--marker-ids", required=True, help="markers.json (tokenizer ids)")
    parser.add_argument("--arm-keys", default=",".join(DEFAULT_ARM_KEYS),
                        help="comma list of arm keys")
    parser.add_argument("--trace-dir", action="append", default=[],
                        help="directory with traces/trace_*.json (repeatable)")
    parser.add_argument("--limit", type=int, default=None,
                        help="audit only the first N trajectories")
    parser.add_argument("--out", required=True, help="output directory")
    return parser


def run(args: argparse.Namespace) -> int:
    markers = load_markers(args.marker_ids)
    arm_keys = list(dict.fromkeys(key.strip() for key in args.arm_keys.split(",") if key.strip()))
    members, arm_report = load_arms(args.arms, arm_keys)
    guard_outputs(args.out, [args.data, args.arms, args.marker_ids])
    os.makedirs(args.out, exist_ok=True)
    files = [("data", args.data), ("arms", args.arms), ("markers", args.marker_ids)]
    entries, dirs = describe_inputs(files, args.trace_dir)
    write_text(os.path.join(args.out, "inputs.sha256"), inputs_text(entries, dirs))
    with open(os.path.join(args.out, "chunks.jsonl"), "w", encoding="utf-8", newline=NL) as sink:
        stream = audit_stream(args.data, args.limit, members, markers, sink)
        result = aggregate(stream, list(members))
    groups = result["aggregates"]
    arm_p90 = {arm: sections["overall"]["loss_pos_p90"] for arm, sections in groups.items()}
    summary = {
        "limit": args.limit, "n_trajectories_audited": result["n_trajectories"],
        "duplicate_traj_ids": result["duplicate_traj_ids"],
        "inputs": {"files": entries, "trace_dirs": dirs}, "arms": arm_report,
        "aggregates": result["aggregates"], "trajectories": result["trajectories"],
        "served": served_stats(args.trace_dir, arm_p90) if args.trace_dir else None,
    }
    write_text(os.path.join(args.out, "summary.json"), json.dumps(summary, indent=2) + NL)
    write_markdown(summary, os.path.join(args.out, "summary.md"))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    try:
        return run(args)
    finally:
        LOG.removeHandler(handler)


if __name__ == "__main__":
    sys.exit(main())
