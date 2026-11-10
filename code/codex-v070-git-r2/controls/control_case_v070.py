#!/usr/bin/env python
"""Offline scripted agent: real submit_patch, persisted capture, then fresh verification."""
from __future__ import annotations
import argparse
import asyncio
import base64
import functools
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path("PROJECT_ROOT/gemma4")
WORK = Path("PROJECT_ROOT/codex/experiments/v070-git-r2")
BASE = "gemma-4-e4b-it-qat-w4a16-ct"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _required_identity(node):
    """Map one full pytest node to JUnit's deterministic module/class representation."""
    prefix, bracket, parameters = node.partition("[")
    parts = prefix.split("::")
    if len(parts) < 2 or not parts[-1]:
        return None
    file = parts[0].replace("\\", "/")
    while file.startswith("./"):
        file = file[2:]
    if not file.endswith(".py") or any(part in ("", "..") for part in file.split("/")):
        return None
    module = file[:-3].replace("/", ".")
    ancestry = parts[1:-1]
    if any(not name for name in ancestry):
        return None
    classname = ".".join([module, *ancestry])
    return file, classname, parts[-1] + (bracket + parameters if bracket else "")


def _junit_classname(value):
    """Only normalize path separators and explicit .py representation, never names."""
    normalized = value.replace("\\", "/")
    if "/" in normalized:
        normalized = normalized.replace(".py.", ".").removesuffix(".py").replace("/", ".")
    return normalized


def junit_evidence(xml_text, task):
    root = ET.fromstring(xml_text)
    cases = list(root.iter("testcase"))
    required = list(task.FAIL_TO_PASS or [])
    identities = {node: _required_identity(node) for node in required}
    required_keys = {}
    for node, identity in identities.items():
        if identity is not None:
            required_keys.setdefault(identity[1:], []).append(node)
    indexed = {}
    for case in cases:
        key = (_junit_classname(case.get("classname", "")), case.get("name", ""))
        indexed.setdefault(key, []).append(case)
    classified = {name: [] for name in ("passed", "failure", "error", "skipped")}
    observed, missing, ambiguous = [], [], []
    for node in required:
        identity = identities[node]
        if identity is None:
            missing.append(node)
            continue
        file, classname, name = identity
        candidates = indexed.get((classname, name), [])
        candidates = [case for case in candidates if not case.get("file") or
                      case.get("file").replace("\\", "/").removeprefix("./") == file]
        if not candidates:
            missing.append(node)
            continue
        observed.append(node)
        if len(candidates) != 1 or len(required_keys[(classname, name)]) != 1:
            ambiguous.append(node)
            continue
        case = candidates[0]
        state = next((key for key in ("error", "skipped", "failure")
                      if case.find(key) is not None), "passed")
        classified[state].append(node)
    errors = len(list(root.iter("error")))
    targets_ok = bool(required) and not missing and not ambiguous and len(set(required)) == len(required)
    return dict(collected=len(cases), required=required, observed=observed,
                failing=classified["failure"], passed=classified["passed"],
                errors=classified["error"], skipped=classified["skipped"],
                missing=missing, ambiguous=ambiguous, junit_error_count=errors,
                skipped_testcases=sum(case.find("skipped") is not None for case in cases),
                target_tests_verified=targets_ok and errors == 0 and not classified["skipped"])


def control_passed(kind, counts, resolved, test_exit_code, artifact_equal, error):
    """The unchanged admission predicate, exposed for regression of exact identities."""
    return bool(
        counts["target_tests_verified"] and artifact_equal and not error
        and ((kind == "gold" and resolved and test_exit_code == 0
              and len(counts["passed"]) == len(counts["required"]) and not counts["failing"])
             or (kind == "empty" and not resolved and test_exit_code == 1
                 and len(counts["failing"]) == len(counts["required"]) and not counts["passed"])))


def observe_submit_tool(original, responses):
    @functools.wraps(original)
    def recorded_submit():
        value = original()
        responses.append(value if isinstance(value, dict) else json.loads(value))
        return value
    return recorded_submit


async def run_case(case, driver, repair):
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks
    from swegemma.context import SwegemmaContext
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness import verification
    from swegemma.results import append_task_result
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    out = Path(case["output"])
    out.mkdir(parents=True, exist_ok=False)
    taskrow, kind = case["task"], case["kind"]
    budget = case["budget"]
    argv = ["--tasks", taskrow["taskfile"], "--snapshots-dir", str(Path(taskrow["snapshot"]).parent),
            "--submission-dir", str(ROOT / "subs/e4b_budgetfit_s1"), "--results-dir", str(out / "results"),
            "--models-yaml", str(ROOT / "models.yaml"), "--sandbox", "subprocess", "--display", "quiet",
            "--task-id", taskrow["instance_id"], "--concurrency", "1",
            "--max-time-minutes", str(budget["max_time_minutes"]), "--max-tool-calls", str(budget["max_tool_calls"]),
            "--max-turns", str(budget["max_turns"]), "--timeout-seconds", str(budget["timeout_seconds"])]
    config = driver["build_eval_config"](driver["parse_args"](argv))
    evaluator = Evaluator(config)
    tasks = [task for task in load_tasks(config.tasks_path) if task.instance_id == taskrow["instance_id"]]
    if len(tasks) != 1:
        raise RuntimeError("Control task is absent or duplicated")
    task = evaluator._hydrate_task_from_secret(tasks[0])
    snapshot, base, patch = resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)
    if digest(snapshot) != taskrow["snapshot_sha256"]:
        raise RuntimeError("Snapshot identity changed")
    context = SwegemmaContext(docker_manager=evaluator.docker, container_id="", task=task, repo=task.repo,
        problem_statement=task.problem_statement, hints_text=task.hints_text or "", graph_dir=config.graph_dir,
        embeddings_dir=config.embeddings_dir, budget=config.budget, harness=config.harness)
    record = {"submit_responses": [], "scripted_calls": [], "gold_patch_sha256": hashlib.sha256(task.patch.encode()).hexdigest()}
    original_tools = context.create_tools
    def recorded_tools():
        tools = original_tools()
        original_submit = tools["submit_patch"]
        tools["submit_patch"] = observe_submit_tool(original_submit, record["submit_responses"])
        return tools
    context.create_tools = recorded_tools
    actions = []
    if kind == "gold":
        encoded = base64.b64encode(task.patch.encode()).decode()
        command = "printf %s " + encoded + " | base64 -d > /tmp/v070-offline-gold.patch && cd /workspace && git apply --binary /tmp/v070-offline-gold.patch"
        actions.append(("run_command", {"command": command}))
    actions.append(("submit_patch", {}))
    cursor = [0]
    class ScriptedLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            index = cursor[0]
            cursor[0] += 1
            if index >= len(actions):
                parts = [types.Part(text="Offline control complete.")]
            else:
                name, arguments = actions[index]
                record["scripted_calls"].append(name)
                parts = [types.Part(function_call=types.FunctionCall(name=name, args=arguments))]
            yield LlmResponse(content=types.Content(role="model", parts=parts))
    config.models.register(BASE, ScriptedLlm(model="v070-offline-script"))
    began = time.perf_counter()
    original_exec = verification.sandbox_exec
    junit_records = []
    async def observed_exec(docker, container_id, command, **kwargs):
        result = await original_exec(docker, container_id, command, **kwargs)
        junit_path = command.split("--junitxml=", 1)[1].split()[0] if "-m pytest " in command and "--junitxml=" in command else None
        if junit_path:
            xml_result = await original_exec(docker, container_id, "cat " + junit_path)
            if xml_result.exit_code != 0:
                raise RuntimeError("Control could not read actual verifier JUnit")
            junit_records.append(xml_result.stdout)
        return result
    verification.sandbox_exec = observed_exec
    result = None
    try:
        captured, agent_error, trace = await evaluator._run_agent_sandbox(
            task, snapshot, 1, 1, context=context, base_snapshot_path=base, patch_path=patch)
        record["agent_error"] = agent_error
        record["patch_submitted"] = context.patch_submitted
        if agent_error or not context.patch_submitted or len(record["submit_responses"]) != 1:
            raise RuntimeError("Actual agent submission path did not complete successfully")
        if record["submit_responses"][0].get("status") != "ok":
            raise RuntimeError("Actual submit_patch returned an error")
        if (kind == "gold" and not captured.strip()) or (kind == "empty" and captured != ""):
            raise RuntimeError("Captured patch emptiness contradicts control kind")
        captured_file = out / "captured.patch"
        captured_file.write_text(captured, encoding="utf-8")
        if captured_file.read_text() != captured:
            raise RuntimeError("Captured artifact differs from submitted patch")
        # The fresh verifier receives only the persisted actual submission, never task.patch.
        result = await verification.verify_task(evaluator.docker, config, task, snapshot,
            agent_patch=captured_file.read_text(), agent_error=None, trace=trace, start_time=began,
            base_snapshot_path=base, patch_path=patch)
        append_task_result(result, config.results_dir, [result])
        if len(junit_records) != 1:
            raise RuntimeError("Expected exactly one fresh verifier pytest execution")
        xml_path = out / "verification-junit.xml"
        xml_path.write_text(junit_records[0], encoding="utf-8")
        counts = junit_evidence(junit_records[0], task)
        official_artifact = config.results_dir / "patches" / (task.instance_id.replace("/", "__") + ".patch")
        artifact_equal = official_artifact.is_file() and official_artifact.read_text() == captured
        if kind == "empty":
            artifact_equal = not official_artifact.exists()
        passed = control_passed(kind, counts, result.resolved, result.test_exit_code, artifact_equal, result.error)
        record.update(schema="v070-git-control-case-v1", task=task.instance_id, kind=kind,
            passed=passed, submit_ok=True, artifact_nonempty=bool(captured), resolved=result.resolved,
            test_exit_code=result.test_exit_code, target_tests_verified=counts["target_tests_verified"],
            junit=counts, infrastructure_errors=(counts["junit_error_count"] + int(result.error is not None or result.test_exit_code not in (0, 1))),
            captured_patch_sha256=digest(captured_file), captured_patch_bytes=len(captured.encode()),
            artifact_sha256=digest(official_artifact) if official_artifact.exists() else None,
            official_artifact_expected=kind == "gold", artifact_equal=artifact_equal,
            duration_seconds=time.perf_counter() - began)
        write_json(out / "case-report.json", record)
        return record
    finally:
        verification.sandbox_exec = original_exec
        evaluator.docker.cleanup_all()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--case-sha256", required=True)
    args = parser.parse_args(argv)
    if digest(args.case) != args.case_sha256:
        raise SystemExit("Control case contract changed")
    case = json.loads(args.case.read_text())
    sys.path.insert(0, str(ROOT))
    driver = runpy.run_path(str(ROOT / "run_eval.py"), run_name="v070_controls_environment")
    sys.path.insert(0, str(WORK / "wrapper"))
    import git_baseline_v070 as repair
    repair.install()
    sys.path.insert(0, str(ROOT / "tools"))
    from sandbox_memory_guard_v043 import MemoryGuard
    guard_path = Path(case["guard_receipt"])
    guard = MemoryGuard(guard_path)
    guard.install()
    completed, error, record = False, None, None
    try:
        record = asyncio.run(run_case(case, driver, repair))
        completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        out = Path(case["output"])
        out.mkdir(parents=True, exist_ok=True)
        failure = out / "case-report.json"
        if not failure.exists():
            write_json(failure, dict(schema="v070-git-control-case-v1", task=case["task"]["instance_id"],
                kind=case["kind"], passed=False, infrastructure_errors=1, error=error))
        raise
    finally:
        payload = guard.close(completed=completed, error=error)
        repair.write_receipt(str(guard_path) + ".git-setup.json", completed=completed,
                             error=error, entry_wrapper=WORK / "wrapper/run_guarded_git_v070.py")
        if completed and not payload["completed"]:
            raise RuntimeError("Memory guard cleanup failed")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
