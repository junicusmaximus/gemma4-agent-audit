#!/usr/bin/env python
"""Prepare and execute one immutable, serial all-task submit/verify control generation."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path("PROJECT_ROOT/gemma4")
WORK = Path("PROJECT_ROOT/codex/experiments/v070-git-r2")
EVIDENCE = WORK / "controls/evidence"
PLAN = WORK / "controls/control-plan.json"
REPORT = EVIDENCE / "control-report.json"
TEMP = ROOT / "results/extctl_v060/_rebuild/tmp/v070git2-controls"
AMENDMENT_SHA = "dff250b2870b185069ec5a86b9b7a2e811ed9fc0b65a728049f93049e8d57a99"
CLARIFICATION_SHA = "95742adc0c8ea8fc863b5e53937f0754dc0cdabdd6e84f083f93477b0472626d"
PARENT = ROOT / "contracts/external-e4b-v068-r22-runtime-PINNED.json"
PYTHON = Path("HOME_DIR/miniconda3/envs/swegemma/bin/python")
SOURCES = ["wrapper/git_baseline_v070.py", "wrapper/run_guarded_git_v070.py",
           "controls/control_case_v070.py", "controls/run_controls_v070.py"]
LOCKS = [ROOT / "migration/recovery-screen-v034/evaluation.lock",
         ROOT / "migration/mask-control-v0351/training.lock"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, sort_keys=True, indent=2)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def source_pins():
    return {str(WORK / name): sha(WORK / name) for name in SOURCES}


def git_fixture_receipt():
    path = WORK / "wrapper/unit-test-report.json"
    doc = read(path)
    if doc.get("schema") != "v070-git-isolated-unit-tests-v1" or doc.get("exit_code") != 0 or doc.get("tests", 0) < 8:
        raise RuntimeError("The isolated Git fixture suite did not pass")
    expected = {str(WORK / "wrapper" / name) for name in (
        "git_baseline_v070.py", "run_guarded_git_v070.py", "test_git_baseline_v070.py")}
    if set(doc.get("source_pins", {})) != expected:
        raise RuntimeError("Git fixture receipt source coverage differs")
    for name, digest in doc["source_pins"].items():
        if sha(name) != digest:
            raise RuntimeError("Git fixture receipt is stale: " + name)
    return {"path": str(path), "sha256": sha(path), "tests": doc["tests"]}


def load_runner():
    sys.path.insert(0, str(ROOT / "tools"))
    path = ROOT / "tools/run_external_e4b_v070.py"
    spec = importlib.util.spec_from_file_location("v070_control_service_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    if PLAN.exists():
        raise RuntimeError("Control plan already exists; refuse overwrite")
    if sha(WORK / "amendment-FROZEN.md") != AMENDMENT_SHA:
        raise RuntimeError("Frozen amendment changed")
    if sha(WORK / "operational-temp-clarification.json") != CLARIFICATION_SHA:
        raise RuntimeError("Operational temporary-root clarification changed")
    parent = read(PARENT)
    tasks = parent["panel"]["eligible"]
    if len(tasks) != 12:
        raise RuntimeError("Expected exactly the original 12-task panel")
    pins = source_pins()
    fixture = git_fixture_receipt()
    budget = parent["evaluation"]["effective_budget"]
    cases = []
    for task in tasks:
        for kind in ("gold", "empty"):
            name = task["instance_id"] + "--" + kind
            home = TEMP / name
            case = dict(schema="v070-git-control-case-contract-v1", task=task, kind=kind,
                output=str(EVIDENCE / "cases" / name), home=str(home),
                guard_receipt=str(EVIDENCE / "cases" / (name + ".memory-guard.json")),
                budget=budget, amendment_sha256=AMENDMENT_SHA, source_pins=pins)
            path = WORK / "controls/cases" / (name + ".json")
            write(path, case)
            cases.append(dict(name=name, task=task["instance_id"], kind=kind,
                              path=str(path), sha256=sha(path)))
    doc = dict(schema="v070-git-control-plan-v1", prepared_utc=utc(),
        amendment_sha256=AMENDMENT_SHA, operational_temp_clarification_sha256=CLARIFICATION_SHA,
        parent=dict(path=str(PARENT), sha256=sha(PARENT)),
        runner_reference=dict(path=str(ROOT / "tools/run_external_e4b_v070.py"),
                              sha256=sha(ROOT / "tools/run_external_e4b_v070.py")),
        source_pins=pins, git_fixture_receipt=fixture, tasks=[task["instance_id"] for task in tasks], cases=cases,
        stop="first failed real control; no retry and no filtering", gpu=False)
    write(PLAN, doc)
    return dict(path=str(PLAN), sha256=sha(PLAN), tasks=len(tasks), cases=len(cases), executed=False)


def verify_plan(expected):
    if sha(PLAN) != expected:
        raise RuntimeError("Control plan SHA mismatch")
    plan = read(PLAN)
    if sha(WORK / "amendment-FROZEN.md") != plan["amendment_sha256"]:
        raise RuntimeError("Amendment changed")
    if sha(WORK / "operational-temp-clarification.json") != plan["operational_temp_clarification_sha256"]:
        raise RuntimeError("Operational temporary-root clarification changed")
    for name, digest in plan["source_pins"].items():
        if sha(name) != digest:
            raise RuntimeError("Control source changed: " + name)
    fixture = git_fixture_receipt()
    if fixture != plan["git_fixture_receipt"]:
        raise RuntimeError("Pinned Git fixture receipt changed")
    for item in (plan["parent"], plan["runner_reference"], *plan["cases"]):
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError("Control input changed: " + item["path"])
    return plan


def execute(plan_sha):
    plan = verify_plan(plan_sha)
    verified_fixture = git_fixture_receipt()
    if REPORT.exists() or EVIDENCE.exists():
        raise RuntimeError("Control evidence already exists; no real-control retry")
    runner = load_runner()
    module = runner.v060()
    contract, _ = runner.panel_plan(module)
    task_map = {row["instance_id"]: row for row in read(PARENT)["panel"]["eligible"]}
    absolute_stop = dt.datetime(2026, 11, 9, 21, 50, tzinfo=dt.timezone.utc)
    seconds = min(5 * 3600, (absolute_stop - dt.datetime.now(dt.timezone.utc)).total_seconds())
    if seconds <= 60:
        raise RuntimeError("No time remains for validated controls and PIN")
    deadline = time.monotonic() + seconds
    doc = dict(schema="v070-git-controls-v1", status="FAIL", completed=False, all_passed=False,
        started_utc=utc(), amendment_sha256=AMENDMENT_SHA, tasks=plan["tasks"], rows=[],
        source_pins=plan["source_pins"], evidence_pins={}, control_plan_sha256=plan_sha,
        git_fixture_passed=(verified_fixture == plan["git_fixture_receipt"]), git_fixture_receipt=verified_fixture,
        wrapper_sha256=sha(WORK / "wrapper/run_guarded_git_v070.py"),
        control_runner_sha256=sha(Path(__file__)), gpu=False)
    with ExitStack() as stack:
        for path in LOCKS:
            handle = stack.enter_context(path.open("a+"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        EVIDENCE.mkdir(parents=True, exist_ok=False)
        service_dir = EVIDENCE / "services"
        service_dir.mkdir()
        cursor = runner.kernel_cursor()
        by_task = {}
        try:
            for item in plan["cases"]:
                if time.monotonic() >= deadline - runner.RESERVE:
                    raise RuntimeError("Control deadline reached before all cases")
                case = read(item["path"])
                task = task_map[item["task"]]
                for key in ("taskfile", "snapshot", "router"):
                    path = Path(task[key]) / "sitecustomize.py" if key == "router" else Path(task[key])
                    if sha(path) != task[key + "_sha256"]:
                        raise RuntimeError("Frozen task product changed: " + str(path))
                home = Path(case["home"])
                if not home.resolve().is_relative_to(TEMP.resolve()) or home == TEMP or home.exists():
                    raise RuntimeError("Temporary workspace is unsafe or already exists: " + str(home))
                home.mkdir(parents=True, exist_ok=False)
                env = {**os.environ, **module.child_env(contract, home, Path(task["router"]))}
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                log = EVIDENCE / (item["name"] + ".log")
                bound = runner.MemoryBoundedService(
                    min(deadline, time.monotonic() + 1200 + runner.RESERVE), service_dir)
                try:
                    bound([str(PYTHON), "-B", str(WORK / "controls/control_case_v070.py"),
                           "--case", item["path"], "--case-sha256", item["sha256"]],
                          log, env, ROOT / "data/competition")
                    result = read(Path(case["output"]) / "case-report.json")
                    result["memory_guard"] = runner.guard_qualification(case["guard_receipt"])
                    setup = read(case["guard_receipt"] + ".git-setup.json")
                    result["git_setup"] = dict(path=case["guard_receipt"] + ".git-setup.json",
                                              sha256=sha(case["guard_receipt"] + ".git-setup.json"))
                    if not setup["completed"] or len(setup["entries"]) != 2 or not all(
                            row.get("completed") and row.get("clean") and
                            row["source_before_sha256"] == row["source_after_sha256"]
                            for row in setup["entries"]):
                        raise RuntimeError("Agent/fresh-verifier Git setup evidence failed")
                    if not result["passed"]:
                        raise RuntimeError("Real control failed: " + item["name"])
                    row = by_task.setdefault(item["task"], {"task": item["task"]})
                    row[item["kind"]] = result
                    doc["rows"] = list(by_task.values())
                    print(json.dumps(dict(case=item["name"], passed=True, resolved=result["resolved"],
                                          test_exit_code=result["test_exit_code"])), flush=True)
                finally:
                    end_receipt = service_dir / (log.name + ".service-end.json")
                    service_empty = end_receipt.is_file() and read(end_receipt).get("cgroup_empty") is True
                    if service_empty and home.exists() and home.resolve().is_relative_to(TEMP.resolve()) and home != TEMP:
                        shutil.rmtree(home)
                    elif home.exists():
                        doc.setdefault("preserved_control_workspaces", []).append(str(home))
            doc["global_oom_check"] = runner.global_oom_check(cursor)
            if len(doc["rows"]) != 12 or any(set(row) != {"task", "gold", "empty"} for row in doc["rows"]):
                raise RuntimeError("Control panel is incomplete")
            doc.update(status="PASS", completed=True, all_passed=True)
        except BaseException as exc:
            doc.update(error=f"{type(exc).__name__}: {exc}", completed=True, all_passed=False)
            # Preserve any failing case report rather than keeping only successful cases.
            doc["attempted_cases"] = [str(path) for path in (EVIDENCE / "cases").glob("*/case-report.json")]
            raise
        finally:
            doc["finished_utc"] = utc()
            doc["evidence_pins"] = {str(path): sha(path) for path in sorted(EVIDENCE.rglob("*"))
                                    if path.is_file() and path != REPORT}
            doc["evidence_pins"][plan["git_fixture_receipt"]["path"]] = plan["git_fixture_receipt"]["sha256"]
            write(REPORT, doc)
    return dict(report=str(REPORT), sha256=sha(REPORT), status=doc["status"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    if args.execute and not args.plan_sha256:
        parser.error("--execute requires --plan-sha256")
    result = prepare() if args.prepare else execute(args.plan_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
