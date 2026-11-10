#!/usr/bin/env python3
"""Analyze all registered v032 serving probes only after the whole batch qualifies."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

ANALYSIS_SHA = "f63610c9babf5754183135b84b0a9a09a2d31a5d3e793931394650d7e44df9f9"
EXECUTION_SHA = "d1b13d9873f44f967114c566fff81880c31357c2f6db3d1b811687ac792f2630"
PRODUCER_SHA = "36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d"
CI_SHA = "821f2294661bf8928979d3df7da6c94fd8f662ee2ed17621a8c8dfaa06207478"
FROZEN_SHA = "20208c74cc921d9dd4975b1b6a7129b6a4b55825f17930429a4fa624935e229c"
BASELINE_MANIFEST_SHA = "f6a1138406313e6e27a41d59fc442ca51b0b727230b19fdc3c5568945034c262"
BASE = "gemma-4-e4b-it-qat-w4a16-ct"
CONDITIONS = {"A0": "results/a0_e4b_s1", "A1": "results/v025_a1_e4b_s1",
              "A2": "results/v025_a2_e4b_s1", "A1a8": "results/v032_A1a8",
              "A2a8": "results/v032_A2a8", "A1_t07": "results/v032_A1_t07",
              "A0_t07": "results/v032_A0_t07"}
CONTRASTS = [["A1a8", "A1"], ["A2a8", "A2"], ["A1_t07", "A1"], ["A0_t07", "A0"],
             ["A1_t07", "A0_t07"], ["A1a8", "A0"], ["A2a8", "A0"]]
VARIANTS = [("A1a8", "alpha8", "A1_alpha8", "e4b_A1a8_s1", "A1", .2),
            ("A2a8", "alpha8", "A2_alpha8", "e4b_A2a8_s1", "A2", .2),
            ("A1_t07", "temperature", "A1", "e4b_A1_t07_s1", "A1", .7),
            ("A0_t07", "temperature", BASE, "e4b_A0_t07_s1", None, .7)]
TOKENIZER_FILES = {"chat_template.jinja": "55572b8d3c8342044e25874c73fe5234b661fa0a57a57f6ef75b58e03d7d959a",
                   "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
                   "tokenizer_config.json": "97af7f9a43d9c26b4ae9558b39c2d6ffed60b9869f2cf2b9a5282b05c6dac67c"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def path_in(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes project root")
    return path


def pin(root, relative, hashes, expected=None):
    path = path_in(root, relative)
    actual = hashes.get(str(path)) or sha(path)
    require(expected is None or actual == expected, "Hash mismatch: " + str(relative))
    hashes[str(path)] = actual
    return path


def document(root, relative, hashes, expected=None):
    path = pin(root, relative, hashes, expected)
    value = strict_json(path.read_bytes())
    require(isinstance(value, dict), "Expected document object: " + str(relative))
    return value


def timestamp(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None, "Timestamp requires timezone")
    return result


def checked_module(root, name, expected, hashes):
    path = pin(root, "tools/" + name + ".py", hashes, expected)
    spec = importlib.util.spec_from_file_location("probe_analysis_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dependencies(root, hashes):
    return (checked_module(root, "analyze_mechanism_v031", CI_SHA, hashes),
            checked_module(root, "analyze_contract_v02_v025", FROZEN_SHA, hashes),
            checked_module(root, "mechanism_metrics_v032", PRODUCER_SHA, hashes))


def verify_plans(root, analysis_sha, execution_sha, hashes):
    require(analysis_sha == ANALYSIS_SHA and execution_sha == EXECUTION_SHA, "Unregistered plan pins")
    plans = []
    for name, expected in (("probes-v0.3.2-analysis-plan", analysis_sha),
                           ("probes-v0.3.2-plan", execution_sha)):
        relative = "contracts/" + name + ".json"
        plan = document(root, relative, hashes, expected)
        sidecar = pin(root, relative + ".sha256", hashes).read_text().split()
        require(sidecar and sidecar[0] == expected, "Plan sidecar mismatch")
        plans.append(plan)
    analysis, execution = plans
    require(analysis["execution_plan_sha256"] == execution_sha and execution["root"] == str(root),
            "Analysis/execution plan linkage mismatch")
    require(analysis["conditions"] == CONDITIONS and analysis["contrasts"] == CONTRASTS,
            "Registered conditions/contrasts changed")
    require(analysis["metrics_producer_sha256"] == PRODUCER_SHA and analysis["seed_per_condition"] == 1,
            "Metric producer or decoding seed changed")
    require(analysis["bootstrap"] == {"draws": 20000, "interval": "95% percentile", "quantile_method": "linear",
                                     "rng": "NumPy PCG64", "seed": 20261103, "unit": "paired whole task"},
            "Registered bootstrap changed")
    expected_runs = [dict(zip(("name", "phase", "alias", "submission", "original", "temperature"), row), seed=1)
                     for row in VARIANTS]
    require(execution["runs"] == expected_runs and execution["version"] == "v0.3.2", "Variant matrix changed")
    require(execution["effective_budget"] == {"concurrency": 4, "max_time_minutes": 4, "max_tool_calls": 28,
                                             "max_turns": 80, "timeout_seconds": 240}, "Effective budget changed")
    for relative, expected in execution["pins"].items():
        pin(root, relative, hashes, expected)
    for folder, inventory in execution["directories"].items():
        path = path_in(root, folder)
        names = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        require(names == set(inventory), "Directory inventory changed: " + folder)
        for name, expected in inventory.items():
            pin(root, str(Path(folder) / name), hashes, expected)
    return analysis, execution


def verify_route(root, log_name, summary, alias, strict, hashes, frozen):
    raw = document(root, "logs/v032_" + log_name + ".log.routing.json", hashes)
    for key in ("request_timeouts", "client_disconnects", "empty_actions", "cancelled_upstream_requests",
                "schema_accepted_responses"):
        require(type(raw[key]) is int and raw[key] >= 0, "Malformed route counter: " + key)
    require(type(summary["errors"]) is int and summary["errors"] == 0, "Route summary has errors")
    require(isinstance(raw["infrastructure_errors"], list) and not raw["infrastructure_errors"]
            and isinstance(raw["invalid_response_metadata"], list) and not raw["invalid_response_metadata"],
            "Infrastructure or invalid-response fault")
    require(isinstance(raw["seen"], list) and isinstance(raw["successful"], list), "Malformed alias evidence")
    frozen.verify_observer(raw, summary, alias, strict=strict)
    require(summary["cancelled_upstream_requests"] == raw["cancelled_upstream_requests"],
            "Cancellation summary mismatch")
    return timestamp(raw["observed_at"])


def verify_phase(root, phase, activation_at, hashes, frozen, execution):
    evidence = "migration/probes-v032/evaluation/"
    name = phase["name"]
    launched = document(root, evidence + name + "-server-launched.json", hashes)
    ready = document(root, evidence + name + "-server-ready.json", hashes)
    gates = document(root, evidence + name + "-gates.json", hashes)
    server = gates["server"]
    require(gates["phase"] == name and server == {k: v for k, v in ready.items()
                                                if k not in ("ready_at", "registered_models")},
            "Phase server identity mismatch")
    require(server["pid"] == launched["pid"] and server["start"] == launched["start"]
            and Path(server["manifest"]).resolve() == path_in(root, phase["manifest"]), "Server launch linkage mismatch")
    require(launched["argv"][1:] == [str(root / "serve_vllm.py"), str(root / "models" / BASE), BASE,
                                     str(root / phase["manifest"])], "Server argv mismatch")
    require(ready["registered_models"] == sorted([BASE, *phase["aliases"]]), "Registered aliases mismatch")
    passed = timestamp(gates["passed_at"])
    require(activation_at <= timestamp(launched["started_at"]) <= timestamp(ready["ready_at"]) <= passed,
            "Phase chronology mismatch")
    cases = {"BASE": BASE, "A1_alpha8": "A1_alpha8", "A2_alpha8": "A2_alpha8"} if name == "alpha8" else {
        "BASE_t07": BASE, "A1_t07": "A1"}
    require(set(gates["canaries"]) == set(cases), "Incomplete phase canaries")
    allowed = {"Agent exceeded session timeout (4.0 min)": "session_timeout",
               "Agent exceeded tool call budget (28 calls)": "tool_calls",
               "Agent exceeded turns budget (80 turns)": "turns"}
    for label, alias in cases.items():
        item = gates["canaries"][label]
        log_name = "canary_" + name + "_" + label
        result = "results/v032_" + log_name
        require(type(item["task_rows"]) is int and item["task_rows"] == 1
                and Path(item["results_dir"]).resolve() == path_in(root, result), "Invalid canary receipt")
        pin(root, result + "/task_results.jsonl", hashes, item["sha256"])
        error = item["canary_episode_error"]
        require(type(item["accepted_budget_exhaustion"]) is bool, "Malformed canary budget status")
        require((error is None and item["accepted_budget_exhaustion"] is False and item["budget_exhaustion_category"] is None)
                or (type(error) is str and error in allowed and item["accepted_budget_exhaustion"] is True
                    and item["budget_exhaustion_category"] == allowed[error]), "Unqualified canary error")
        require(item["original_row_sha256"] == item["sha256"], "Canary original-row hash mismatch")
        require(timestamp(ready["ready_at"]) <= verify_route(root, log_name, item["routing"], alias, True,
                                                             hashes, frozen) <= passed, "Canary chronology mismatch")
    return gates


def verify_baseline_traces(root, hashes, ids):
    manifest = document(root, "analysis/mechanism-v032/source-trace-manifest.json", hashes, BASELINE_MANIFEST_SHA)
    require(manifest["unchanged_before_after"] is True and len(manifest["files"]) == 441,
            "Historical source-trace manifest is incomplete")
    for name in ("A0", "A1", "A2"):
        relatives = [CONDITIONS[name] + "/task_results.jsonl", *[
            CONDITIONS[name] + "/traces/trace_" + task + ".json" for task in ids]]
        for relative in relatives:
            require(relative in manifest["files"], "Historical trace lacks a registered source hash")
            pin(root, relative, hashes, manifest["files"][relative])


def qualify(root, analysis_sha, execution_sha):
    root = root.resolve()
    evidence = "migration/probes-v032/evaluation/"
    required = ["completed.json", "activation.json", *[x[0] + "-complete.json" for x in VARIANTS],
                *[phase + suffix for phase in ("alpha8", "temperature")
                  for suffix in ("-gates.json", "-server-ready.json", "-server-launched.json")]]
    require(all(path_in(root, evidence + name).is_file() for name in required), "Whole probe batch is incomplete")
    require(not path_in(root, evidence + "stopped.json").exists(), "Stopped probe generation is not qualified")
    hashes = {}
    analysis, execution = verify_plans(root, analysis_sha, execution_sha, hashes)
    ci, frozen, producer = dependencies(root, hashes)
    complete = document(root, evidence + "completed.json", hashes)
    activation = document(root, evidence + "activation.json", hashes)
    require(complete["plan_sha256"] == activation["plan_sha256"] == execution_sha, "Execution receipt plan mismatch")
    started, finished = timestamp(activation["activated_at"]), timestamp(complete["completed_at"])
    require(timestamp(execution["registered_at"]) <= timestamp(analysis["recorded_at_utc"]) <= started <= finished,
            "Registration/execution chronology mismatch")
    ids = execution["holdout_ids"]
    require(len(ids) == len(set(ids)) == 48 and all(type(x) is str and "/" not in x and "\\" not in x for x in ids),
            "Expected 48 unique safe task IDs")
    primary = pin(root, analysis["primary_population_ids_path"], hashes,
                  analysis["primary_population_ids_sha256"]).read_text().splitlines()
    require(len(primary) == len(set(primary)) == 40 and set(primary) <= set(ids), "Primary population mismatch")
    previous = started
    for phase in execution["phases"]:
        gates = verify_phase(root, phase, previous, hashes, frozen, execution)
        for run in (r for r in execution["runs"] if r["phase"] == phase["name"]):
            done = document(root, evidence + run["name"] + "-complete.json", hashes)
            require(done["plan_sha256"] == execution_sha and type(done["task_rows"]) is int
                    and done["task_rows"] == 48, "Unqualified task count or execution plan")
            require(done["gates_sha256"] == hashes[str(path_in(root, evidence + phase["name"] + "-gates.json"))],
                    "Run/phase gate hash mismatch")
            run_started, run_done = timestamp(done["started_at"]), timestamp(done["completed_at"])
            require(max(previous, timestamp(gates["passed_at"])) <= run_started <= run_done <= finished,
                    "Probe run chronology mismatch")
            result = CONDITIONS[run["name"]]
            require(Path(done["results_dir"]).resolve() == path_in(root, result), "Run result directory mismatch")
            pin(root, result + "/task_results.jsonl", hashes, done["sha256"])
            require(run_started <= verify_route(root, run["name"], done["routing"], run["alias"], False,
                                                hashes, frozen) <= run_done, "Run routing chronology mismatch")
            previous = run_done
    require(complete["final_server"] == gates["server"], "Final server identity mismatch")
    for name, expected in analysis["baseline_result_hashes"].items():
        pin(root, CONDITIONS[name] + "/task_results.jsonl", hashes, expected)
    verify_baseline_traces(root, hashes, ids)
    missing_files = []
    for name, *_ in VARIANTS:
        for task in ids:
            trace = path_in(root, CONDITIONS[name] + "/traces/trace_" + task + ".json")
            if trace.is_file():
                pin(root, str(trace), hashes)
            else:
                missing_files.append(str(trace))
    tokenizer_dir = root / "models/gemma-4-e4b-it"
    for name, expected in TOKENIZER_FILES.items():
        pin(root, str(tokenizer_dir / name), hashes, expected)
    pin(root, str(Path(__file__).resolve()), hashes)
    pin(root, "tools/test_analyze_probes_v032.py", hashes)
    return {"analysis": analysis, "execution": execution, "hashes": hashes, "all_ids": ids,
            "primary_ids": primary, "missing_files": missing_files, "tokenizer_dir": tokenizer_dir}, (ci, frozen, producer)


def decode_conditions(root, context, helpers):
    ci, frozen, producer = helpers
    tokenizer_info = producer.tokenizer_provenance(str(context["tokenizer_dir"]))
    require(tokenizer_info["files_sha256"] == TOKENIZER_FILES and tokenizer_info["add_special_tokens"] is True,
            "Tokenizer assets or text-tokenization policy changed")
    tokenize = producer.load_tokenizer(str(context["tokenizer_dir"]))
    rows, metrics = {}, {}
    for name, relative in CONDITIONS.items():
        folder = path_in(root, relative)
        rows[name] = frozen.load_run(folder / "task_results.jsonl", context["all_ids"])
        metrics[name] = {}
        for task in context["all_ids"]:
            record = producer.build_task_record(folder, task, lambda value: producer.compute_task_metrics(
                value, loop_threshold=5, max_output_tokens=8192, tokenize=tokenize))
            ci.validate_row(record)
            metrics[name][task] = record
    return rows, metrics, tokenizer_info


def paired(values, baseline, units, ci, draws, seed):
    tasks = sorted(task for task in values if values[task] is not None and baseline[task] is not None)
    result = ci.paired_task_bootstrap([values[x] for x in tasks], [baseline[x] for x in tasks], draws, seed)
    result.update(units=units, paired_task_ids=tasks,
                  treated_eligible_tasks=sum(x is not None for x in values.values()),
                  baseline_eligible_tasks=sum(x is not None for x in baseline.values()),
                  per_task=[{"instance_id": x, "treated": values[x], "baseline": baseline[x]} for x in tasks])
    if units == "proportion":
        result["effect_pp"] = None if result["effect"] is None else 100 * result["effect"]
        result["ci95_pp"] = None if result["ci95"] is None else [100 * x for x in result["ci95"]]
    return result


def make_report(context, rows, metrics, tokenizer, helpers, draws=20000, seed=20261103):
    ci, _, producer = helpers
    populations = {}
    for label, ids in (("primary40", context["primary_ids"]), ("all48", context["all_ids"])):
        condition_summaries = {}
        for name in CONDITIONS:
            records = [metrics[name][task] for task in ids]
            summary = producer.summarize_run(name, [rows[name][task] for task in ids], records)
            summary.update(resolved_count=sum(rows[name][task]["resolved"] for task in ids),
                           resolution_rate=sum(rows[name][task]["resolved"] for task in ids) / len(ids),
                           missing_trace_reasons=dict(Counter(r["trace_error"] for r in records if r["trace_missing"])),
                           unknown_observation_transitions=sum(r["nonrecovery"]["unknown_excluded"] for r in records
                                                               if not r["trace_missing"]))
            condition_summaries[name] = summary
        specs = {"resolved": ("proportion", None), **ci.metric_specs()}
        comparisons = {}
        for treated, baseline in CONTRASTS:
            values = {}
            for metric, (units, extract) in specs.items():
                arms = {}
                for arm in (treated, baseline):
                    arms[arm] = {task: (float(rows[arm][task]["resolved"]) if extract is None else
                                       None if metrics[arm][task]["trace_missing"] else extract(metrics[arm][task]))
                                 for task in ids}
                item = paired(arms[treated], arms[baseline], units, ci, draws, seed)
                if metric.startswith("M5."):
                    stratum = metric.rsplit(".", 1)[1]
                    item["transition_counts_by_task"] = {task: {arm: (None if metrics[arm][task]["trace_missing"] else {
                        key: metrics[arm][task]["nonrecovery"][stratum][key] for key in ("eligible", "identical")})
                        for arm in (treated, baseline)} for task in ids}
                values[metric] = item
            comparisons[treated + "-" + baseline] = values
        populations[label] = {"task_ids": ids, "conditions": condition_summaries, "contrasts": comparisons}
    return {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "label": context["analysis"]["label"],
            "method": dict(context["analysis"]["bootstrap"], draws=draws, seed=seed),
            "limitations": context["analysis"]["claim_limits"], "populations": populations,
            "metric_definitions": {"M1": "observed nonempty thinking text per recorded agent step",
                                   "M5": context["analysis"]["m5"], "M6": context["analysis"]["m6"],
                                   "M6_aggregation": "median across tasks of within-task retained-text length medians; absent text is undefined"},
            "per_task": {name: {task: {"resolved": rows[name][task]["resolved"], "mechanism": metrics[name][task]}
                                for task in context["all_ids"]} for name in CONDITIONS},
            "provenance": {"input_hashes": context["hashes"], "missing_trace_files": context["missing_files"],
                           "tokenizer": tokenizer, "python": platform.python_version(), "numpy": np.__version__}}


def markdown(report):
    lines = ["# Registered serving-probe analysis v0.3.2", "", report["label"], "",
             "All seven registered contrasts are reported. Effects are percentage points.", "",
             "| Population | Contrast | Paired tasks | Resolution difference | 95% task-bootstrap interval |",
             "|---|---|---:|---:|---|"]
    for population, result in report["populations"].items():
        for name, metrics in result["contrasts"].items():
            item = metrics["resolved"]
            lines.append(f"| {population} | {name} | {item['n_paired_tasks']} | {item['effect_pp']:.3f} | "
                         f"[{item['ci95_pp'][0]:.3f}, {item['ci95_pp'][1]:.3f}] |")
    lines += ["", "Mechanism contrasts, conditional M5 support, M6 retained-text proxies and per-task data are in report.json.",
              "", *["- " + x for x in report["limitations"]], ""]
    return "\n".join(lines)


def analyze(root, analysis_sha, execution_sha, output):
    root, output = root.resolve(), output.resolve()
    require(output.is_relative_to(root) and not output.exists(), "Output must be fresh and inside project root")
    context, helpers = qualify(root, analysis_sha, execution_sha)
    rows, metrics, tokenizer = decode_conditions(root, context, helpers)
    report = make_report(context, rows, metrics, tokenizer, helpers)
    for path, expected in context["hashes"].items():
        require(sha(path) == expected, "Input changed during analysis: " + path)
    require(all(not Path(path).exists() for path in context["missing_files"]), "Missing trace appeared during analysis")
    output.mkdir(parents=True, exist_ok=False)
    for name, text in (("report.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                       ("report.md", markdown(report))):
        with (output / name).open("x") as stream:
            stream.write(text)
    manifest = {"analysis_plan_sha256": analysis_sha, "execution_plan_sha256": execution_sha,
                "outputs": {name: sha(output / name) for name in ("report.json", "report.md")}}
    with (output / "output-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-plan-sha", required=True)
    parser.add_argument("--execution-plan-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.analysis_plan_sha, args.execution_plan_sha, args.output_dir)
    print("Qualified serving-probe analysis written to " + str(args.output_dir))


if __name__ == "__main__":
    main()
