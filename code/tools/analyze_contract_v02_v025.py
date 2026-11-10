#!/usr/bin/env python3
"""Frozen Gemma v0.2 analysis; only the recovery generation declared in the seal.

NumPy APIs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
and https://numpy.org/doc/stable/reference/generated/numpy.quantile.html .
No outcome JSONL is decoded until all qualification and registration checks pass.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import numpy as np

FREEZE_SHA = "a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355"
SENSITIVITY_EXCLUSIONS = {"rich_3480", "rich_3772", "rich_4006", "rich_3930"}
ARMS, SEEDS = ("A0", "A1", "A2"), (1, 2, 3)
UNAVAILABLE = {
    "generated_tokens": "Not exported by the pinned task-result serializer.",
    "time_to_first_edit": "Not exported; traces are not interpreted by this analysis.",
    "typed_context_error_count": "No typed context-error field is exported; no text heuristic is used.",
}
ROW_FIELDS = {
    "instance_id", "repo", "resolved", "agent_patch_size", "test_exit_code",
    "duration_seconds", "error", "tool_calls", "total_llm_calls", "task_index",
}
ROW_REQUIRED = {
    "instance_id", "resolved", "error", "agent_patch_size", "duration_seconds",
    "tool_calls", "total_llm_calls",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    require(isinstance(value, str), "Missing timestamp")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Timestamp has no timezone")
    return parsed


def checked_hash(path, expected, label):
    require(isinstance(expected, str) and len(expected) == 64, "Missing hash: " + label)
    require(Path(path).is_file() and sha(path) == expected, "Hash mismatch: " + label)


def method_spec():
    return {"bootstrap": "paired whole-task cluster percentile", "draws": 20000,
            "seed": 20261103, "rng": "NumPy Generator(PCG64)",
            "quantile_method": "linear", "python": platform.python_version(),
            "numpy": np.__version__}


def populations(doc):
    all_ids, primary = doc["holdout_ids"], doc["primary"]["eligible_task_ids"]
    require(len(all_ids) == len(set(all_ids)) == 48, "Expected 48 distinct holdout IDs")
    require(len(primary) == len(set(primary)) == 40 and set(primary) <= set(all_ids),
            "Expected 40 fixed primary IDs")
    require(set(doc["primary"]["excluded_task_ids"]) == set(all_ids) - set(primary),
            "Primary exclusions disagree")
    require(SENSITIVITY_EXCLUSIONS <= set(primary), "Sensitivity IDs absent from primary")
    sensitivity = [task for task in primary if task not in SENSITIVITY_EXCLUSIONS]
    require(len(sensitivity) == 36, "Expected 36 sensitivity IDs")
    return {"primary40": sorted(primary), "sensitivity36": sorted(sensitivity),
            "secondary48": sorted(all_ids)}


def registration(root, seal_path, expected_freeze=FREEZE_SHA):
    seal = read_json(seal_path)
    version = seal.get("recovery_version")
    require(isinstance(version, str) and re.fullmatch(r"v0[0-9]{2}", version),
            "Invalid sealed recovery generation")
    require(seal.get("method") == method_spec(), "Sealed method/software differs from runtime")
    code = Path(__file__).resolve()
    checked_hash(code, seal.get("analysis_code_sha256"), "analysis implementation")
    checked_hash(code.with_name("test_analyze_contract_v02_v025.py"),
                 seal.get("analysis_test_sha256"), "analysis tests")
    require(seal.get("freeze_sha256") == expected_freeze, "Seal uses a different freeze")
    freeze = root / "contracts/contract-v0.2.freeze.json"
    checked_hash(freeze, expected_freeze, "original freeze")
    require(freeze.with_suffix(".json.sha256").read_text().split()[0] == expected_freeze,
            "Freeze sidecar disagrees")
    doc = read_json(freeze)
    require(doc["approved"] is True and doc["seeds"] == list(SEEDS), "Unexpected protocol")
    require(doc["primary"]["bootstrap_draws"] == 20000 and
            doc["primary"]["analysis_seed"] == 20261103, "Unexpected bootstrap specification")
    require(doc["effective_budget"] == {"concurrency": 4, "max_time_minutes": 4,
            "max_tool_calls": 28, "max_turns": 80, "timeout_seconds": 240},
            "Unexpected effective budget")
    populations(doc)
    evidence = root / f"migration/recovery-{version}/evaluation"
    manifest = evidence / "operational-manifest.json"
    checked_hash(manifest, seal.get("recovery_manifest_sha256"), "recovery manifest")
    for name, expected in doc["files"].items():
        checked_hash(root / name, expected, "frozen " + name)
    with (root / "results/controls_rich/controls.csv").open() as stream:
        qualified = {row["instance_id"] for row in csv.DictReader(stream)
                     if row["gold_resolved"] == "True" and row["empty_resolved"] == "False"}
    require(qualified == set(doc["primary"]["eligible_task_ids"]), "Controls membership changed")
    return doc, seal, evidence


def verify_observer(raw, summary, arm, strict=False):
    require(raw["drained"] is True and raw["seen"] and
            all(value == arm for value in raw["seen"]), "Routing missing or wrong alias")
    require(not raw["infrastructure_errors"], "Recorded infrastructure failure")
    require(all(value == arm for value in raw["successful"]), "Wrong successful alias")
    require(type(raw["schema_accepted_responses"]) is int and
            type(raw["empty_actions"]) is int and raw["empty_actions"] >= 0,
            "Malformed response counters")
    require(raw["schema_accepted_responses"] == len(raw["successful"]) + raw["empty_actions"]
            and raw["schema_accepted_responses"] <= len(raw["seen"]), "Response counter mismatch")
    metadata = raw["empty_response_metadata"]
    require(isinstance(metadata, list) and len(metadata) == raw["empty_actions"]
            and all(isinstance(value, dict) for value in metadata),
            "Empty-response envelope evidence mismatch")
    for key in ("successful_requests", "schema_accepted_responses", "empty_actions"):
        require(type(summary[key]) is int and summary[key] >= 0,
                "Malformed summary response counter: " + key)
    require(summary["model"] == arm and summary["errors"] == 0, "Routing summary failed")
    require(summary["successful_requests"] == len(raw["successful"]), "Request count mismatch")
    for key in ("task_http_errors", "request_timeouts", "client_disconnects", "empty_actions",
                "schema_accepted_responses", "empty_response_metadata"):
        require(summary[key] == raw[key], "Routing summary mismatch: " + key)
    if strict:
        require(raw["successful"] and not raw["task_http_errors"] and
                raw["request_timeouts"] == 0,
                "Canary transport/model-action qualification failed")


def verify_gate(root, gates, arm, version):
    gate = gates["arms"][arm]
    require(gate["effect_exceeds_base_repeat_drift"] is True and
            timestamp(gate["passed_at"]) <= timestamp(gates["all_passed_at"]),
            "Adapter effect gate missing")
    canary = root / f"logs/{version}_canary_{arm}.log.routing.json"
    verify_observer(read_json(canary), gate["routing"], arm, strict=True)
    checked_hash(root / f"results/{version}_canary_{arm}/task_results.jsonl",
                 gate["routing"]["sha256"], "canary result")
    require(gate["routing"]["task_rows"] == 1, "Canary row count invalid")
    return {str(canary): sha(canary)}


def verify_completed_run(root, evidence, arm, seed, version, finished):
    receipt = evidence / f"{arm}_s{seed}-complete.json"
    done, gates = read_json(receipt), evidence / "gates.json"
    require(done["task_rows"] == 48 and done["gates_sha256"] == sha(gates),
            "Completion receipt does not qualify all tasks")
    require(timestamp(read_json(gates)["all_passed_at"]) <= timestamp(done["started_at"]) <=
            timestamp(done["completed_at"]) <= finished, "Run chronology invalid")
    route = root / f"logs/{version}_{arm}_s{seed}.log.routing.json"
    verify_observer(read_json(route), done["routing"], arm)
    path = root / f"results/{version}_{arm.lower()}_e4b_s{seed}/task_results.jsonl"
    checked_hash(path, done["sha256"], "qualified " + arm + str(seed))
    return {str(path): done["sha256"]}, {str(receipt): sha(receipt), str(route): sha(route)}


def qualification(root, doc, seal, evidence):
    version = seal["recovery_version"]
    completed = read_json(evidence / "completed.json")
    require(completed["outcomes_opened"] is False, "Controller completion seal is missing")
    finished = timestamp(completed["completed_at"])
    manifest, gates = read_json(evidence / "operational-manifest.json"), read_json(evidence / "gates.json")
    require(manifest["freeze_sha256"] == seal["freeze_sha256"], "Recovery freeze mismatch")
    expected_outputs = {str(root / f"results/{version}_canary_{arm}") for arm in ("A1", "A2")}
    expected_outputs |= {str(root / f"results/{version}_{arm.lower()}_e4b_s{s}")
                         for arm in ("A1", "A2") for s in SEEDS}
    require(len(manifest["outputs"]) == 8 and set(manifest["outputs"]) == expected_outputs,
            "Recovery manifest outputs do not match the sealed generation")
    for key, filename in (("runner_sha256", f"recover_eval_{version}.py"),
                          ("tests_sha256", f"test_recover_eval_{version}.py"),
                          ("helper_sha256", "overnight_v02.py")):
        checked_hash(root / "tools" / filename, manifest[key], filename)
    checked_hash(manifest["retry_note"], manifest["retry_note_sha256"], "retry amendment")
    checked_hash(manifest["budget_policy_source"]["path"],
                 manifest["budget_policy_source"]["sha256"], "harness budget taxonomy")
    checked_hash(root / "data/competition/tasks.jsonl", manifest["task_dataset_sha256"], "task data")
    require(gates["task_dataset_sha256"] == manifest["task_dataset_sha256"], "Gate task data mismatch")
    gate_time = timestamp(gates["all_passed_at"])
    require(timestamp(manifest["registered_at"]) <= gate_time <= finished, "Gate chronology invalid")
    folders = {root / "sft/runs" / arm for arm in ("A1", "A2")}
    folders |= {root / f"subs/e4b_{arm}_s{s}" for arm in ("A1", "A2") for s in SEEDS}
    for folder in folders:
        actual = {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
        require(actual == gates["watched_files"][str(folder)],
                "Canonical adapter/submission changed: " + str(folder))
    result_hashes, receipt_hashes = {}, {}
    for arm in ("A1", "A2"):
        receipt_hashes.update(verify_gate(root, gates, arm, version))
        for seed in SEEDS:
            results, receipts = verify_completed_run(root, evidence, arm, seed, version, finished)
            result_hashes.update(results)
            receipt_hashes.update(receipts)
    return result_hashes, receipt_hashes

def preflight(root, seal_path, expected_freeze=FREEZE_SHA):
    """Read metadata and hash bytes only; never decode task outcomes here."""
    doc, seal, evidence = registration(root, seal_path, expected_freeze)
    version = seal["recovery_version"]
    result_hashes, receipts = qualification(root, doc, seal, evidence)
    paths = {}
    for arm in ARMS:
        for seed in SEEDS:
            relative = (f"results/a0_e4b_s{seed}/task_results.jsonl" if arm == "A0" else
                        f"results/{version}_{arm.lower()}_e4b_s{seed}/task_results.jsonl")
            path = root / relative
            expected = doc["files"][relative] if arm == "A0" else result_hashes[str(path)]
            checked_hash(path, expected, arm + str(seed))
            result_hashes[str(path)] = expected
            paths[(arm, seed)] = path
    metadata = [seal_path, evidence / "operational-manifest.json", evidence / "gates.json",
                evidence / "completed.json"]
    return doc, paths, {"seal": seal, "inputs": result_hashes,
                       "receipts": receipts, "metadata": {str(p): sha(p) for p in metadata}}


def load_run(path, expected_ids):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict) and ROW_REQUIRED <= row.keys() and
                    row.keys() <= ROW_FIELDS, "Unknown task-result schema")
            task = row["instance_id"]
            require(isinstance(task, str) and task in expected_ids and task not in rows,
                    "Wrong or duplicate task ID")
            require(type(row["resolved"]) is bool, "Resolution must be a literal boolean")
            require(row["error"] is None or isinstance(row["error"], str), "Malformed task error")
            require(not (row["resolved"] and row["error"]), "Resolved task has conflicting error")
            for field in ("tool_calls", "total_llm_calls", "agent_patch_size"):
                require(type(row[field]) is int and row[field] >= 0, "Invalid " + field)
            duration = row["duration_seconds"]
            require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                    "Invalid duration")
            rows[task] = row
    require(set(rows) == set(expected_ids), "Missing task outcomes")
    return rows


def paired_bootstrap(treated, baseline, draws=20000, seed=20261103):
    treated, baseline = np.asarray(treated), np.asarray(baseline)
    require(treated.shape == baseline.shape and treated.ndim == 2 and
            treated.shape[1] == 3 and treated.shape[0] > 0, "Expected paired task-by-three-seed arrays")
    delta = (treated.astype(float) - baseline.astype(float)).mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    indexes = rng.integers(0, len(delta), size=(draws, len(delta)))
    interval = np.quantile(delta[indexes].mean(axis=1), [0.025, 0.975], method="linear")
    return {"effect_pp": float(100 * delta.mean()), "ci95_pp": (100 * interval).tolist(),
            "seed_effects_pp": (100 * (treated.mean(axis=0) - baseline.mean(axis=0))).tolist()}


def exact_sign(treated, baseline):
    wins, losses = int(np.sum(treated & ~baseline)), int(np.sum(baseline & ~treated))
    discordant = wins + losses
    p = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
             / 2 ** discordant) if discordant else 1.)
    return {"treated_only": wins, "baseline_only": losses, "exact_two_sided_p": p,
            "interpretation": "Descriptive majority-of-seeds McNemar/sign test; not primary inference."}


def summarize_population(ids, runs, draws, seed):
    arrays = {arm: np.array([[runs[(arm, s)][task]["resolved"] for s in SEEDS] for task in ids])
              for arm in ARMS}
    arms = {}
    for arm, values in arrays.items():
        rows = [runs[(arm, s)][task] for task in ids for s in SEEDS]
        metrics = {name: {"mean": float(np.mean([r[name] for r in rows])),
                          "median": float(np.median([r[name] for r in rows])),
                          "p90": float(np.quantile([r[name] for r in rows], .9, method="linear"))}
                   for name in ("tool_calls", "total_llm_calls", "duration_seconds")}
        arms[arm] = {"mean_resolution_rate": float(values.mean()),
                     "seed_resolution_rates": dict(zip(map(str, SEEDS), values.mean(axis=0).tolist())),
                     "majority_solved": [task for task, ok in zip(ids, values.sum(axis=1) >= 2) if ok],
                     "best_of_three_union_solved": [task for task, ok in zip(ids, values.any(axis=1)) if ok],
                     "nonempty_patch_rate": sum(r["agent_patch_size"] > 0 for r in rows) / len(rows),
                     "task_error_count": sum(bool(r["error"]) for r in rows),
                     "metrics_all_task_seed_rows": metrics}
    comparisons = {}
    for treated, baseline in (("A1", "A0"), ("A2", "A1"), ("A2", "A0")):
        effect = paired_bootstrap(arrays[treated], arrays[baseline], draws, seed)
        effect["majority_sign_test"] = exact_sign(arrays[treated].sum(axis=1) >= 2,
                                                 arrays[baseline].sum(axis=1) >= 2)
        effect["role"] = "primary only in primary40" if treated == "A1" else "exploratory"
        comparisons[treated + "-" + baseline] = effect
    return {"task_count": len(ids), "task_ids": ids, "arms": arms, "comparisons": comparisons}


def make_report(doc, runs, provenance):
    groups = populations(doc)
    results = {name: summarize_population(ids, runs, doc["primary"]["bootstrap_draws"],
                                          doc["primary"]["analysis_seed"])
               for name, ids in groups.items()}
    interval = results["primary40"]["comparisons"]["A1-A0"]["ci95_pp"]
    conclusion = ("supportive local evidence" if interval[0] > 0 else
                  "supportive local evidence of a decrease" if interval[1] < 0 else "inconclusive")
    per_task = {task: {arm: {str(s): {key: runs[(arm, s)][task][key] for key in
                    ("resolved", "error", "tool_calls", "total_llm_calls",
                     "duration_seconds", "agent_patch_size")} for s in SEEDS} for arm in ARMS}
                for task in sorted(doc["holdout_ids"])}
    return {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "contract_id": doc["contract_id"], "recovery_version": provenance["seal"]["recovery_version"],
            "method": dict(method_spec(), error_rule="False+error is unresolved; True+error is rejected",
                           union="Best-of-three union is descriptive; it is not pass@1."),
            "limitations": ["Conditional on one public repository and fixed task population.",
                           "One training run per trained arm; decoding seeds do not measure training-seed robustness.",
                           "A1-A0 estimates the implemented distillation pipeline, without causal mechanism isolation.",
                           "A2 comparisons and all secondary metrics are exploratory; no multiplicity-free claims.",
                           "Team-wide outcome blinding is not independently established."],
            "unavailable_metrics": UNAVAILABLE, "primary_conclusion": conclusion,
            "populations": results, "per_task": per_task, "provenance": provenance}


def markdown(report):
    lines = ["# Frozen v0.2 analysis of qualified " + report["recovery_version"] + " runs", "",
             "Primary A1-A0 conclusion: " + report["primary_conclusion"] + ".", "",
             "| Population | Comparison | Effect (pp) | 95% paired task-bootstrap interval (pp) |",
             "|---|---|---:|---|"]
    for name, population in report["populations"].items():
        for pair, result in population["comparisons"].items():
            lo, hi = result["ci95_pp"]
            lines.append(f"| {name} | {pair} | {result['effect_pp']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "A1-A0 on primary40 is the primary comparison. All other rows are secondary/exploratory.",
              "The bootstrap carries each task's complete paired three-seed arm vector together.",
              "Best-of-three unions and majority/sign tests are descriptive; unions are not pass@1.",
              "", *["- " + text for text in report["limitations"]], "",
              "Per-task and per-seed outcomes, logged descriptive metrics, unavailable metrics, hashes,",
              "software versions and RNG details are in report.json.", ""]
    return "\n".join(lines)


def analyze(root, seal_path, output, expected_freeze=FREEZE_SHA):
    require(not output.exists(), "Refuse existing analysis output directory")
    doc, paths, provenance = preflight(root, seal_path, expected_freeze)
    output.mkdir(parents=True, exist_ok=False)
    runs = {key: load_run(path, doc["holdout_ids"]) for key, path in paths.items()}
    # Reject inputs changed between metadata validation and outcome decoding.
    for path, expected in provenance["inputs"].items():
        checked_hash(path, expected, "analysis input")
    report = make_report(doc, runs, provenance)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    (output / "report.md").write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("PROJECT_ROOT/gemma4"))
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.seal.resolve(), args.output_dir.resolve())
    print("Qualified analysis written to " + str(args.output_dir))


if __name__ == "__main__":
    main()
