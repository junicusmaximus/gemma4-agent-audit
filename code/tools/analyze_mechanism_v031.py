#!/usr/bin/env python3
"""Exploration-informed, task-paired mechanism intervals; no primary-outcome reanalysis."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

ARMS = ("A0", "A1", "A2")
SEEDS = (1, 2, 3)
RUNS = {f"{'v025_' if arm != 'A0' else ''}{arm.lower()}_e4b_s{seed}": (arm, seed)
        for arm in ARMS for seed in SEEDS}
OPENERS = ("find_opener", "wrapper_opener", "repro_script_opener", "other", "no_tool_call")
STRATA = ("error", "silent", "normal")
DRAWS, SEED = 20000, 20261103
PRODUCER_SHA = "36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d"
REPORT_SHA = "b2a2ebba991f8fb2c61d548beaf054c03d4163ce0c4e61f0f12b77b0083d30a4"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def count(value, label):
    require(type(value) is int and value >= 0, f"Invalid nonnegative integer: {label}")
    return value


def rate(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"Invalid proportion: {label}")
    return float(value)


def validate_row(row):
    require(isinstance(row, dict), "Expected a task record object")
    require(type(row.get("trace_missing")) is bool, "trace_missing must be boolean")
    fields = ("think_rate", "loop", "max_repeat", "submit_called", "opener", "nonrecovery")
    if row["trace_missing"]:
        require(all(row.get(key) is None for key in fields), "Missing trace has defined metrics")
        return
    steps = count(row.get("n_agent_steps"), "n_agent_steps")
    thoughts = count(row.get("n_thinking_steps"), "n_thinking_steps")
    require(thoughts <= steps, "Thinking step count exceeds agent steps")
    if steps:
        require(math.isclose(rate(row.get("think_rate"), "think_rate"), thoughts / steps,
                             rel_tol=1e-12, abs_tol=1e-12), "think_rate/count mismatch")
    else:
        require(row.get("think_rate") is None, "Zero agent steps require undefined think_rate")
    calls = count(row.get("n_tool_calls"), "n_tool_calls")
    repeat = count(row.get("max_repeat"), "max_repeat")
    require(repeat <= calls and (calls == 0 or repeat >= 1), "max_repeat inconsistent with call count")
    require(type(row.get("loop")) is bool and row["loop"] == (repeat >= 5),
            "loop must match fixed max_repeat >= 5 rule")
    require(type(row.get("submit_called")) is bool, "submit_called must be boolean")
    require(row.get("opener") in OPENERS, "Unknown opener category")
    require((row["opener"] == "no_tool_call") == (calls == 0), "Opener/call-count mismatch")
    require(calls > 0 or (repeat == 0 and not row["submit_called"]), "Action without parsed calls")
    recovery = row.get("nonrecovery")
    require(isinstance(recovery, dict), "Expected nonrecovery object")
    eligible_total = count(recovery.get("unknown_excluded"), "unknown_excluded")
    for stratum in STRATA:
        item = recovery.get(stratum)
        require(isinstance(item, dict), f"Missing nonrecovery stratum: {stratum}")
        eligible = count(item.get("eligible"), f"{stratum}.eligible")
        identical = count(item.get("identical"), f"{stratum}.identical")
        require(identical <= eligible, f"{stratum}: identical exceeds eligible")
        if eligible:
            require(math.isclose(rate(item.get("repeat_rate"), stratum), identical / eligible,
                                 rel_tol=1e-12, abs_tol=1e-12), f"{stratum}: rate/count mismatch")
        else:
            require(item.get("repeat_rate") is None, f"{stratum}: zero eligibility requires null")
        eligible_total += eligible
    require(eligible_total == max(calls - 1, 0), "Transition strata do not account for all call pairs")


def load_records(data, expected_ids):
    require(len(expected_ids) == len(set(expected_ids)) > 0, "Task IDs must be unique and nonempty")
    expected = set(expected_ids)
    records = {}
    for line_number, line in enumerate(data.splitlines(), 1):
        require(bool(line.strip()), f"Blank record at line {line_number}")
        row = strict_json(line)
        require(isinstance(row, dict), f"Expected object at line {line_number}")
        run, task = row.get("run"), row.get("instance_id")
        require(type(run) is str and run in RUNS, f"Unexpected run at line {line_number}")
        require(type(task) is str and task in expected, f"Unexpected task at line {line_number}")
        key = (*RUNS[run], task)
        require(key not in records, f"Duplicate task/run record: {run}/{task}")
        validate_row(row)
        records[key] = row
    wanted = {(arm, seed, task) for arm in ARMS for seed in SEEDS for task in expected}
    require(set(records) == wanted, f"Incomplete run/task grid: expected {len(wanted)}, got {len(records)}")
    return records


def metric_specs():
    result = {
        "M1.observed_thinking_step_rate": ("proportion", lambda r: r["think_rate"]),
        "M2.loop_rate": ("proportion", lambda r: float(r["loop"])),
        "M2.max_repeat": ("count", lambda r: float(r["max_repeat"])),
        "M3.submit_rate": ("proportion", lambda r: float(r["submit_called"])),
    }
    for opener in OPENERS:
        result[f"M4.opener.{opener}"] = ("proportion", lambda r, o=opener: float(r["opener"] == o))
    result["M4.opener.any_target"] = ("proportion", lambda r: float(r["opener"] in OPENERS[:3]))
    for stratum in STRATA:
        result[f"M5.nonrecovery.{stratum}"] = (
            "proportion", lambda r, s=stratum: r["nonrecovery"][s]["repeat_rate"])
    return result


def task_values(records, ids, arm, extract):
    values = {}
    for task in ids:
        values[task] = []
        for seed in SEEDS:
            row = records[(arm, seed, task)]
            value = None if row["trace_missing"] else extract(row)
            values[task].append(None if value is None else float(value))
    return values


def paired_task_bootstrap(treated, baseline, draws=DRAWS, seed=SEED):
    require(type(draws) is int and draws > 0, "draws must be a positive integer")
    require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    t, b = np.asarray(treated, dtype=float), np.asarray(baseline, dtype=float)
    require(t.ndim == b.ndim == 1 and t.shape == b.shape, "Expected paired task means")
    require(np.isfinite(t).all() and np.isfinite(b).all(), "Nonfinite paired task means")
    n = len(t)
    effect = float((t - b).mean()) if n else None
    result = {"n_paired_tasks": n, "treated_mean": float(t.mean()) if n else None,
              "baseline_mean": float(b.mean()) if n else None, "effect": effect,
              "ci95": None, "ci_undefined_reason": None}
    if n < 2:
        result["ci_undefined_reason"] = "no_paired_tasks" if not n else "fewer_than_two_paired_tasks"
        return result
    delta = t - b
    rng = np.random.Generator(np.random.PCG64(seed))
    indexes = rng.integers(0, n, size=(draws, n))
    result["ci95"] = np.quantile(delta[indexes].mean(axis=1), [.025, .975], method="linear").tolist()
    return result


def compare(treated, baseline, units, draws=DRAWS, seed=SEED, complete_only=False):
    require(set(treated) == set(baseline), "Paired arms must have the same task-ID grid")
    coverage = {}
    for task in sorted(treated):
        require(len(treated[task]) == len(baseline[task]) == 3, "Expected three seed positions")
        coverage[task] = {
            "treated": sum(v is not None for v in treated[task]),
            "baseline": sum(v is not None for v in baseline[task]),
        }
    minimum = 3 if complete_only else 1
    paired = [task for task in sorted(treated)
              if coverage[task]["treated"] >= minimum and coverage[task]["baseline"] >= minimum]
    def mean(values):
        return float(np.mean([x for x in values if x is not None]))
    result = paired_task_bootstrap([mean(treated[x]) for x in paired],
                                   [mean(baseline[x]) for x in paired], draws, seed)
    result.update(units=units, paired_task_ids=paired, seed_coverage_by_task=coverage,
                  treated_eligible_tasks=sum(c["treated"] >= minimum for c in coverage.values()),
                  baseline_eligible_tasks=sum(c["baseline"] >= minimum for c in coverage.values()),
                  support_rule="all_three_seeds_each_arm" if complete_only else "at_least_one_seed_each_arm",
                  seed_aggregation="unweighted mean of each arm's observed seed rates within a task",
                  per_task=[{"instance_id": task, "treated_seed_values": treated[task],
                             "baseline_seed_values": baseline[task],
                             "treated_task_mean": mean(treated[task]),
                             "baseline_task_mean": mean(baseline[task])} for task in paired])
    if units == "proportion":
        result["effect_pp"] = None if result["effect"] is None else result["effect"] * 100
        result["ci95_pp"] = None if result["ci95"] is None else [x * 100 for x in result["ci95"]]
    return result


def make_report(records, ids, provenance, draws=DRAWS, seed=SEED):
    results = {}
    for name, (units, extract) in metric_specs().items():
        arms = {arm: task_values(records, ids, arm, extract) for arm in ARMS}
        comparisons = {}
        for treated, baseline in (("A1", "A0"), ("A2", "A0"), ("A2", "A1")):
            item = compare(arms[treated], arms[baseline], units, draws, seed)
            item["complete_three_seed_sensitivity"] = compare(
                arms[treated], arms[baseline], units, draws, seed, complete_only=True)
            if name.startswith("M5."):
                stratum = name.rsplit(".", 1)[1]
                item["transition_counts_by_task"] = {
                    task: {arm: {str(seed): (None if records[(arm, seed, task)]["trace_missing"] else {
                        key: records[(arm, seed, task)]["nonrecovery"][stratum][key]
                        for key in ("eligible", "identical")})
                        for seed in SEEDS} for arm in (treated, baseline)}
                    for task in sorted(ids)}
            comparisons[f"{treated}-{baseline}"] = item
        results[name] = comparisons
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_status": "exploration-informed descriptive secondary analysis; outcomes already known",
        "task_ids": sorted(ids), "task_count": len(ids), "metrics": results, "provenance": provenance,
        "method": {"draws": draws, "seed": seed, "rng": "NumPy Generator(PCG64)",
                   "quantile_method": "linear", "bootstrap": "paired whole-task cluster percentile",
                   "python": platform.python_version(), "numpy": np.__version__},
        "limitations": [
            "All metrics and comparisons are post-exploration, descriptive and unadjusted for multiplicity.",
            "One public repository and one training run per arm; three decoding seeds are not training replications.",
            "M1 is observed thinking-text presence per agent step, not reasoning quality or an LLM-call probability.",
            "M5 conditions on both arms having observed eligible transitions; support differs by comparison/stratum.",
            "M5 averages observed seed rates; missing values are not zero-filled. Complete-three-seed sensitivity is separate.",
            "No causal mechanism, equivalence or multiplicity-free significance claim follows from these intervals.",
            "M6 is excluded: producer thought length may be tokenizer tokens or whitespace words; provenance must establish units.",
        ],
    }


def load_inputs(root, manifest_path, manifest_sha256):
    root = root.resolve()
    require(manifest_path.resolve().is_relative_to(root), "Manifest path escapes project root")
    manifest_bytes = manifest_path.read_bytes()
    require(digest(manifest_bytes) == manifest_sha256, "Input manifest hash mismatch")
    manifest = strict_json(manifest_bytes)
    require(isinstance(manifest, dict), "Expected input-manifest object")
    require(manifest.get("schema_version") == 1 and type(manifest["schema_version"]) is int,
            "Expected input-manifest schema version 1")
    require(manifest.get("input_generation") == "mechanism-v032", "Expected corrected mechanism-v032 inputs")
    require(manifest.get("sealed_report_sha256") == REPORT_SHA, "Wrong sealed-report pin")
    require(manifest.get("producer_sha256") == PRODUCER_SHA, "Unapproved metric producer SHA")
    paths = {"sealed_report": root / "analysis/v02-v025/report.json"}
    for name in ("metrics", "task_ids", "producer"):
        value = manifest.get(f"{name}_path")
        require(type(value) is str and value, f"Missing {name} path")
        path = (root / value).resolve()
        require(path.is_relative_to(root), f"{name} path escapes project root")
        paths[name] = path
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    require(digest(blobs["sealed_report"]) == REPORT_SHA, "Sealed analysis report hash mismatch")
    for name in ("metrics", "task_ids", "producer"):
        require(digest(blobs[name]) == manifest.get(f"{name}_sha256"), f"{name} hash mismatch")
    report = strict_json(blobs["sealed_report"])
    ids = blobs["task_ids"].decode().splitlines()
    require(len(ids) == len(set(ids)) == 40, "Expected exactly 40 unique primary task IDs")
    sealed_ids = report["populations"]["primary40"]["task_ids"]
    require(set(ids) == set(sealed_ids) and len(sealed_ids) == 40, "Primary population mismatch")
    records = load_records(blobs["metrics"], ids)
    hashes = {str(paths[name]): digest(blob) for name, blob in blobs.items()}
    hashes[str(manifest_path.resolve())] = digest(manifest_bytes)
    provenance = {"input_hashes": hashes, "analysis_code_sha256": digest(Path(__file__).read_bytes()),
                  "input_generation": manifest["input_generation"],
                  "metric_producer_sha256": digest(blobs["producer"]), "sealed_report_sha256": REPORT_SHA,
                  "expected_runs": RUNS,
                  "exposure": "Legacy behavior and outcome proxies informed metrics; v025 primary outcomes known before this analysis."}
    return records, ids, provenance


def markdown(report):
    lines = ["# Exploration-informed mechanism intervals", "", report["analysis_status"], "",
             "A1-A0 contrasts below are descriptive. Other comparisons and coverage are in report.json.", "",
             "| Metric | Paired tasks | A1-A0 effect | 95% task-bootstrap interval | Units |",
             "|---|---:|---:|---|---|"]
    for name, comparisons in report["metrics"].items():
        item = comparisons["A1-A0"]
        effect = item.get("effect_pp", item["effect"])
        interval = item.get("ci95_pp", item["ci95"])
        effect_text = "undefined" if effect is None else f"{effect:.4f}"
        interval_text = item["ci_undefined_reason"] if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
        lines.append(f"| {name} | {item['n_paired_tasks']} | {effect_text} | {interval_text} | "
                     f"{'percentage points' if item['units'] == 'proportion' else item['units']} |")
    lines.extend(["", *[f"- {x}" for x in report["limitations"]], ""])
    return "\n".join(lines)


def analyze(root, output, manifest_path, manifest_sha256):
    require(output.resolve().is_relative_to(root.resolve()), "Output path escapes project root")
    require(not output.exists(), "Refuse existing analysis output directory")
    records, ids, provenance = load_inputs(root, manifest_path, manifest_sha256)
    report = make_report(records, ids, provenance)
    for path, expected in provenance["input_hashes"].items():
        require(digest(Path(path).read_bytes()) == expected, "Input changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output / "report.md").open("x") as stream:
        stream.write(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output_dir.resolve(),
            args.input_manifest.resolve(), args.manifest_sha256)
    print(f"Exploration-informed mechanism analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
