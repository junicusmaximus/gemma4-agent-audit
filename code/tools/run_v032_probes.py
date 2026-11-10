"""Inspect a prepared serving probe plan; --activate explicitly starts execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from prepare_v032_probes import BASE, PLAN, FREEZE_SHA, VARIANTS, inventory, load_recovery, require, sha, write_json


def verify_plan(root, expected_sha):
    path = root / PLAN
    require(sha(path) == expected_sha, "Plan hash mismatch")
    require(path.with_suffix(".json.sha256").read_text().split()[0] == expected_sha,
            "Plan sidecar mismatch")
    plan = json.loads(path.read_text())
    require(plan["root"] == str(root.resolve()), "Plan root mismatch")
    require(plan["version"] == "v0.3.2", "Unexpected protocol version")
    require(len(plan["holdout_ids"]) == len(set(plan["holdout_ids"])) == 48,
            "Expected 48 unique holdout IDs")
    expected_runs = [dict(zip(("name", "phase", "alias", "submission", "original", "temperature"), item), seed=1)
                     for item in VARIANTS]
    require(plan["runs"] == expected_runs, "Run order or intervention changed")
    expected_phases = [
        {"name": "alpha8", "manifest": "subs/e4b_v032_alpha8_manifest", "aliases": ["A1_alpha8", "A2_alpha8"]},
        {"name": "temperature", "manifest": "subs/e4b_v032_original_manifest", "aliases": ["A1", "A2"]},
    ]
    require(plan["phases"] == expected_phases, "Unexpected adapter capacity, aliases or phase manifest")
    snapshots = root / "data/competition/snapshots"
    namespace = sorted(str(p.relative_to(snapshots)) for p in snapshots.rglob("*") if p.is_file())
    require(namespace == plan["snapshot_namespace"], "Snapshot resolver namespace changed")
    for name, expected in plan["pins"].items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and sha(path) == expected, "Pinned input changed: " + name)
    for name, expected in plan["directories"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Invalid inventory directory")
        folder = root / name
        actual_names = {str(p.relative_to(folder)) for p in folder.rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts}
        require(actual_names == set(expected), "Directory inventory changed: " + name)
        for relative, digest in expected.items():
            pin_name = str(Path(name) / relative)
            actual = plan["pins"].get(pin_name)
            if actual is None:
                actual = sha(folder / relative)
            require(actual == digest, "Directory inventory hash changed: " + name)
    recovery = load_recovery(root)
    helper = recovery.load_helper(root)
    original = helper.freeze(root / "contracts/contract-v0.2.freeze.json", FREEZE_SHA)
    require(plan["holdout_ids"] == original["holdout_ids"], "Task ordering changed")
    require(plan["effective_budget"] == original["effective_budget"], "Effective budgets changed")
    return plan, recovery, helper


def group_members(recovery, group):
    members = {}
    for pid, value in recovery.processes().items():
        try:
            if os.getpgid(pid) != group:
                continue
            require(recovery.stat(pid)["start"] == value["start"], "PID reused during group read")
            members[pid] = {key: value[key] for key in ("pid", "parent", "start", "argv")}
        except (ProcessLookupError, FileNotFoundError):
            continue
    return members


def server_identity(recovery, pid, start, manifest):
    rows = recovery.processes()
    require(pid in rows and rows[pid]["start"] == str(start), "Server PID identity changed")
    row = rows[pid]
    args = row["argv"]
    require(len(args) == 5 and Path(args[1]).name == "serve_vllm.py"
            and args[3] == BASE and Path(args[4]).resolve() == Path(manifest).resolve(),
            "Server arguments or manifest changed")
    group = os.getpgid(pid)
    require(group != os.getpgrp(), "Refuse to signal controller process group")
    members = group_members(recovery, group)
    descendants = {pid}
    for _ in rows:
        expanded = descendants | {p for p, v in rows.items() if v["parent"] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    require(set(members) <= descendants and pid in members, "Server group has unrelated processes")
    inodes = {line.split()[9] for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
              if line.split()[1] == "0100007F:1F40" and line.split()[3] == "0A"}
    require(len(inodes) == 1, "Expected one localhost:8000 listener")
    inode = next(iter(inodes))
    owners = []
    for member in members:
        try:
            if any(os.readlink(fd) == "socket:[" + inode + "]"
                   for fd in Path(f"/proc/{member}/fd").iterdir()):
                owners.append(member)
        except (FileNotFoundError, PermissionError):
            continue
    require(bool(owners), "Listener is not in the verified server group")
    return {"pid": pid, "start": str(start), "manifest": str(manifest), "group": group,
            "members": members, "listener_inode": inode, "listener_owners": owners}


def same_members(expected, actual):
    return expected == actual


def same_process(expected, actual):
    return all(expected.get(key) == actual.get(key) for key in ("pid", "start", "argv"))


def stop_server(recovery, identity, grace_seconds=60):
    current = server_identity(recovery, identity["pid"], identity["start"], identity["manifest"])
    require(current == identity, "Server group changed before shutdown")
    require(same_members(identity["members"], group_members(recovery, identity["group"])),
            "Server membership changed immediately before shutdown")
    os.killpg(identity["group"], signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        remaining = group_members(recovery, identity["group"])
        require(all(pid in identity["members"] and same_process(identity["members"][pid], row)
                    for pid, row in remaining.items()), "Unexpected process appeared during shutdown")
        if not remaining:
            return
        time.sleep(0.5)
    require(False, "Verified server group did not exit after SIGTERM; no escalation attempted")


def environment(plan):
    env = dict(os.environ)
    env.pop("MODEL_PROXY_URL", None)
    for key in list(env):
        if key.startswith(("VLLM_", "CUDA_", "PYTORCH_", "NCCL_")) or key in ("OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
            env.pop(key)
    env.update(plan["server_environment"])
    env.update(OPENAI_BASE_URL="http://127.0.0.1:8000/v1",
               OPENAI_API_BASE="http://127.0.0.1:8000/v1", OPENAI_API_KEY="EMPTY",
               LOCAL_API_KEY="EMPTY", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OTEL_SDK_DISABLED="true", LITELLM_TELEMETRY="False",
               LITELLM_LOCAL_MODEL_COST_MAP="True")
    return env


def start_server(root, phase, plan, helper, recovery, env, evidence):
    manifest = root / phase["manifest"]
    log = root / ("logs/v032_server_" + phase["name"] + ".log")
    require(not log.exists(), "Server log destination collision")
    argv = [str(helper.PY), str(root / "serve_vllm.py"), str(root / "models" / BASE), BASE, str(manifest)]
    with log.open("x") as stream:
        child = subprocess.Popen(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    start = recovery.stat(child.pid)["start"]
    write_json(evidence / (phase["name"] + "-server-launched.json"),
               {"pid": child.pid, "start": start, "argv": argv, "started_at": recovery.utc()})
    deadline = time.monotonic() + 1200
    while True:
        require(child.poll() is None, "New server exited before readiness")
        require(time.monotonic() < deadline, "New server startup deadline exceeded")
        try:
            models = {item["id"] for item in helper.api("models")["data"]}
            if BASE in models:
                require(models == {BASE, *phase["aliases"]}, "Unexpected registered model/adapter set")
                identity = server_identity(recovery, child.pid, start, manifest)
                write_json(evidence / (phase["name"] + "-server-ready.json"),
                           dict(identity, ready_at=recovery.utc(), registered_models=sorted(models)))
                return identity
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)


def planned_paths(root, plan):
    names = ["canary_" + phase["name"] + "_" + name
             for phase in plan["phases"]
             for name in (["BASE"] + phase["aliases"] if phase["name"] == "alpha8" else ["BASE_t07", "A1_t07"])]
    names += [run["name"] for run in plan["runs"]]
    paths = [root / "results" / ("v032_" + name) for name in names]
    paths += [root / "logs" / ("v032_" + name + ".log") for name in names]
    paths += [root / "logs" / ("v032_" + name + ".log.routing.json") for name in names]
    paths += [root / "logs" / ("v032_server_" + phase["name"] + ".log") for phase in plan["phases"]]
    paths += [root / "migration/probes-v032/evaluation"]
    return paths


def canary_cases(root, phase):
    if phase["name"] == "alpha8":
        return [("BASE", BASE, root / "subs/e4b_budgetfit_s1"),
                ("A1_alpha8", "A1_alpha8", root / "subs/e4b_A1a8_s1"),
                ("A2_alpha8", "A2_alpha8", root / "subs/e4b_A2a8_s1")]
    return [("BASE_t07", BASE, root / "subs/e4b_A0_t07_s1"),
            ("A1_t07", "A1", root / "subs/e4b_A1_t07_s1")]


def run_observed(root, plan, recovery, helper, env, name, alias, sub, ids, strict):
    result = root / "results" / ("v032_" + name)
    log = root / "logs" / ("v032_" + name + ".log")
    require(not result.exists() and not log.exists(), "Run destination collision")
    command = helper.eval_command(sub, result, ids, plan["effective_budget"], 1 if strict else None)
    routing = recovery.observe_run(helper, alias, command, log, env, root / "data/competition", strict=strict)
    rows = (recovery.validate_canary_result(result / "task_results.jsonl", ids[0], plan["effective_budget"])
            if strict else recovery.validate_rows(result / "task_results.jsonl", ids))
    return dict(rows, routing=routing, results_dir=str(result))


def activate(root, plan_sha, existing_pid, existing_start):
    plan, recovery, helper = verify_plan(root, plan_sha)
    require(plan["existing_server"] == {"pid": existing_pid, "start": str(existing_start)},
            "Activation identity differs from prepared identity")
    require(all(not path.exists() for path in planned_paths(root, plan)), "Execution destination collision")
    legacy = Path("HOME_DIR/gemma4")
    recovery.assert_idle((root, legacy))
    identity = server_identity(recovery, existing_pid, existing_start, plan["existing_manifest"])
    evidence = root / "migration/probes-v032/evaluation"
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / "activation.json", {"activated_at": recovery.utc(), "plan_sha256": plan_sha,
                                            "existing_server": identity, "controller_pid": os.getpid()})
    env = environment(plan)
    try:
        for phase in plan["phases"]:
            verify_plan(root, plan_sha)
            recovery.assert_idle((root, legacy))
            stop_server(recovery, identity)
            identity = start_server(root, phase, plan, helper, recovery, env, evidence)
            gates = {"server": identity, "phase": phase["name"], "canaries": {}}
            for name, alias, sub in canary_cases(root, phase):
                verify_plan(root, plan_sha)
                require(server_identity(recovery, identity["pid"], identity["start"], identity["manifest"]) == identity,
                        "Server changed before canary")
                if alias != BASE:
                    helper.effect(alias)
                gates["canaries"][name] = run_observed(root, plan, recovery, helper, env,
                    "canary_" + phase["name"] + "_" + name, alias, sub, [plan["canary_task"]], True)
                verify_plan(root, plan_sha)
                require(server_identity(recovery, identity["pid"], identity["start"], identity["manifest"]) == identity,
                        "Server changed during canary")
            write_json(evidence / (phase["name"] + "-gates.json"), dict(gates, passed_at=recovery.utc()))
            for run in (run for run in plan["runs"] if run["phase"] == phase["name"]):
                verify_plan(root, plan_sha)
                recovery.assert_idle((root, legacy))
                require(server_identity(recovery, identity["pid"], identity["start"], identity["manifest"]) == identity,
                        "Server changed before evaluation")
                started = recovery.utc()
                result = run_observed(root, plan, recovery, helper, env, run["name"], run["alias"],
                                      root / "subs" / run["submission"], plan["holdout_ids"], False)
                verify_plan(root, plan_sha)
                require(server_identity(recovery, identity["pid"], identity["start"], identity["manifest"]) == identity,
                        "Server changed during evaluation")
                write_json(evidence / (run["name"] + "-complete.json"),
                           dict(result, started_at=started, completed_at=recovery.utc(),
                                plan_sha256=plan_sha, gates_sha256=sha(evidence / (phase["name"] + "-gates.json"))))
                print(recovery.utc(), run["name"], "qualified evaluation completed", flush=True)
        write_json(evidence / "completed.json", {"completed_at": recovery.utc(), "plan_sha256": plan_sha,
                                               "final_server": identity})
    except BaseException as error:
        write_json(evidence / "stopped.json", {"stopped_at": recovery.utc(), "type": type(error).__name__,
                                              "reason": str(error), "last_server": identity})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--existing-server-pid", type=int)
    parser.add_argument("--existing-server-start")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.activate:
        plan, _, _ = verify_plan(root, args.plan_sha)
        print(json.dumps({"status": "prepared_only", "runs": plan["runs"], "gpu_operations": False}))
        return
    require(args.existing_server_pid is not None and args.existing_server_start is not None,
            "Activation requires exact existing server PID and starttime")
    activate(root, args.plan_sha, args.existing_server_pid, args.existing_server_start)


if __name__ == "__main__":
    main()
