"""Prospective v0.2.5 evaluation retry; never signals or starts a model server."""
from __future__ import annotations

import asyncio
import socket
import contextlib
import httpx
from starlette.applications import Starlette
from starlette.requests import Request, ClientDisconnect
from starlette.routing import Route
import uvicorn
import argparse
import csv
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

HELPER_SHA = "e48419eef233c08cc0465b219157cdf2b6160e2a2b4080bf057860ce4fb6bf5a"
FREEZE_SHA = "a7a301b7ec6a8ebfe123767b33ed7a95f0f1192a3425e1e2f8c9efe2ecdb8355"
BUDGET_POLICY_SOURCE = Path("HOME_DIR/miniconda3/envs/swegemma/lib/python3.12/site-packages/swegemma/harness/agent_runner.py")
BUDGET_POLICY_SHA = "4d7eccabd203c26f7c39e882684d2fd851f383f6a5c76c4a75b0e559cd5d6a18"
BASE = "gemma-4-e4b-it-qat-w4a16-ct"
ARMS = {"A1": "arm_ordinary_tokmatched", "A2": "arm_short"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def stat(pid):
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": fields[0], "parent": int(fields[1]), "start": fields[19]}


def wait_parent(pid, start, timeout=43200, interval=15):
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = stat(pid)
        except FileNotFoundError:
            return
        require(value["start"] == str(start), "Legacy parent PID reused")
        if value["state"] == "Z":
            return
        require(time.monotonic() < deadline, "Legacy chain completion deadline exceeded")
        time.sleep(interval)


def processes():
    result = {}
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            value = stat(int(folder.name))
            if value["state"] == "Z":
                continue
            value["argv"] = [os.fsdecode(v) for v in (folder / "cmdline").read_bytes().split(b"\0") if v]
            value["cwd"] = str((folder / "cwd").resolve(strict=True))
            require(stat(value["pid"])["start"] == value["start"], "Process changed during identity read")
            result[value["pid"]] = value
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def assert_idle(roots):
    roots = {str(p) for p in roots}
    for value in processes().values():
        scripts = {Path(v).name for v in value["argv"]}
        if scripts & {"train_lora.py", "run_eval.py"}:
            tied = value["cwd"] in roots or any(any(v.startswith(r + "/") for r in roots) for v in value["argv"])
            require(not tied, "Project trainer/evaluator remains active")


def server_identity(legacy):
    rows = processes()
    candidates = [p for p in rows.values() if len(p["argv"]) >= 5
                  and Path(p["argv"][1]).name == "serve_vllm.py"
                  and p["argv"][3] == BASE
                  and Path(p["argv"][4]).resolve() == legacy / "subs/e4b_arms_manifest"]
    require(len(candidates) == 1, "Expected exactly one legacy adapter server")
    server = candidates[0]
    children = {server["pid"]}
    for _ in rows:
        added = {pid for pid, p in rows.items() if p["parent"] in children}
        if added <= children:
            break
        children |= added
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == "0100007F:1F40" and fields[3] == "0A":
            inodes.add(fields[9])
    require(len(inodes) == 1, "Expected one localhost:8000 listening socket")
    inode = next(iter(inodes))
    owners = []
    for pid in sorted(children):
        try:
            owns = any(os.readlink(fd) == "socket:[" + inode + "]" for fd in (Path("/proc") / str(pid) / "fd").iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        if owns:
            owners.append({"pid": pid, "start": rows[pid]["start"]})
    require(bool(owners), "Listening socket is not owned by the legacy server tree")
    return {"server_pid": server["pid"], "server_start": server["start"], "argv": server["argv"],
            "listener_inode": inode, "listener_owners": owners}


def inventory(folder):
    require(folder.is_dir(), "Missing source directory: " + str(folder))
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def copy_verified(source, destination, receipts):
    """Copy completed data only; existing identical content may be retained."""
    before = inventory(source) if source.is_dir() else {"": sha(source)}
    if source.is_dir():
        if destination.exists():
            require(inventory(destination) == before, "Different canonical destination: " + str(destination))
        else:
            shutil.copytree(source, destination)
        after = inventory(destination)
        stable = inventory(source)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha(destination) == before[""], "Different canonical destination: " + str(destination))
        else:
            with source.open("rb") as inp, destination.open("xb") as out:
                shutil.copyfileobj(inp, out)
            shutil.copystat(source, destination)
        after, stable = {"": sha(destination)}, {"": sha(source)}
    require(before == after == stable, "Copy/source integrity changed")
    receipts.append({"source": str(source), "destination": str(destination), "files": before})


def copy_diagnostic(source, destination, receipts):
    if source.is_dir():
        copy_verified(source, destination, receipts)
    else:
        receipts.append({"source": str(source), "status": "missing_legacy_diagnostic"})


def load_helper(root):
    path = root / "tools/overnight_v02.py"
    require(sha(path) == HELPER_SHA, "Frozen helper changed")
    spec = importlib.util.spec_from_file_location("frozen_overnight_v02", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = root
    return helper


def validate_training(helper, legacy, arm, started_at):
    helper.checkpoint(arm, started_at, legacy)
    metadata = json.loads((legacy / ("sft/runs/" + arm + ".meta.json")).read_text())
    args = metadata["args"]
    expected = {"arm": ARMS[arm], "rank": 16, "alpha": 16, "lr": 1e-4, "epochs": 1,
                "max_seq_len": 16384, "grad_accum": 8, "seed": 20261103,
                "dry_run": False, "crop": 0, "device": "cuda"}
    require(all(args.get(k) == v for k, v in expected.items()), "Trainer arguments mismatch: " + arm)
    require(Path(args["data"]).resolve() == legacy / "sft/data/trajectories.jsonl", "Wrong training data")
    require(Path(args["base"]).resolve() == legacy / "models/gemma-4-e4b-it", "Wrong training base")
    require(Path(args["out"]).resolve() == legacy / "sft/runs" / arm, "Wrong adapter destination")
    summary = metadata["summary"]
    require(summary["arm"] == ARMS[arm] and summary["steps"] == 208
            and summary["names_match_probe"] is True, "Incomplete training metadata: " + arm)


def validate_submission(root, arm, seed):
    sub = root / ("subs/e4b_" + arm + "_s" + str(seed))
    base = root / ("subs/e4b_budgetfit_s" + str(seed))
    expected = (base / "agent.yaml").read_text()
    lines = (sub / "agent.yaml").read_text().splitlines(keepends=True)
    require(sum(line.strip() == "adapter: " + arm for line in lines) == 1, "Missing/duplicate adapter alias")
    require("".join(line for line in lines if not line.startswith("adapter:")) == expected,
            "Submission agent configuration differs from base")
    base_files = inventory(base)
    expected_files = set(base_files) | {"adapters/" + arm + "/adapter_config.json", "adapters/" + arm + "/adapter_model.safetensors"}
    require(set(inventory(sub)) == expected_files, "Unexpected submission files")
    for name, digest in base_files.items():
        if name != "agent.yaml":
            require((sub / name).is_file() and sha(sub / name) == digest, "Submission base file changed: " + name)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        require(sha(sub / "adapters" / arm / name) == sha(root / "sft/runs" / arm / name),
                "Submission adapter differs from completed checkpoint")


def validate_rows(path, ids, error_free=False):
    seen = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            seen.append(row.get("instance_id", row.get("task_id")))
            if error_free:
                require(not row.get("error_message", row.get("error")), "Canary harness error")
    require(len(seen) == len(ids) and len(set(seen)) == len(ids) and set(seen) == set(ids),
            "Missing/duplicate/unexpected task results")
    return {"task_rows": len(seen), "sha256": sha(path)}


def validate_canary_result(path, task, budget):
    rows = validate_rows(path, [task])
    row = next(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    values = [row[key] for key in ("error", "error_message") if key in row]
    require(all(value is None or isinstance(value, str) for value in values), "Malformed canary error field")
    errors = [value for value in values if value]
    allowed = {}
    if budget.get("max_time_minutes") is not None:
        allowed[f"Agent exceeded session timeout ({float(budget['max_time_minutes'])} min)"] = "session_timeout"
    if budget.get("max_tool_calls") is not None:
        allowed[f"Agent exceeded tool call budget ({budget['max_tool_calls']} calls)"] = "tool_calls"
    if budget.get("max_turns") is not None:
        allowed[f"Agent exceeded turns budget ({budget['max_turns']} turns)"] = "turns"
    accepted = bool(errors) and len(set(errors)) == 1 and errors[0] in allowed
    require(not errors or accepted, "Canary harness error is not one unambiguous configured budget exhaustion")
    return dict(rows, accepted_budget_exhaustion=accepted,
                budget_exhaustion_category=allowed[errors[0]] if accepted else None,
                canary_episode_error=errors[0] if accepted else None, original_row_sha256=rows["sha256"])


def backend_frames(content):
    frame = json.loads(content)
    require(isinstance(frame, dict) and frame.get("error") is None, "Invalid/error provider envelope")
    choices = frame.get("choices")
    require(isinstance(choices, list) and bool(choices), "Provider choices missing or empty")
    for choice in choices:
        require(isinstance(choice, dict), "Provider choice is not an object")
        message = choice.get("message")
        require(isinstance(message, dict) and message.get("role") == "assistant", "Invalid/missing assistant message")
        require(message.get("content") is None or isinstance(message["content"], str), "Invalid message content type")
        require(choice.get("finish_reason") is None or isinstance(choice["finish_reason"], str), "Invalid finish reason type")
        for key in ("reasoning", "reasoning_content", "refusal"):
            require(message.get(key) is None or isinstance(message[key], str), "Invalid optional message text type")
        calls = message.get("tool_calls")
        require(calls is None or isinstance(calls, list), "Invalid tool calls type")
        for call in calls or []:
            require(isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("type") == "function",
                    "Invalid function tool call")
            function = call.get("function")
            require(isinstance(function, dict) and isinstance(function.get("name"), str)
                    and isinstance(function.get("arguments"), str), "Invalid function tool call payload")
    return [frame]


def empty_response_metadata(frames, payload):
    choices = []
    for frame in frames:
        for choice in frame["choices"]:
            message = choice["message"]
            content, calls = message.get("content"), message.get("tool_calls")
            choices.append({"role": message["role"], "content_type": type(content).__name__,
                            "content_length": len(content) if isinstance(content, str) else 0,
                            "tool_calls_type": type(calls).__name__, "tool_calls_count": len(calls) if calls else 0,
                            "finish_reason": choice.get("finish_reason"),
                            "reasoning_present": bool(message.get("reasoning") or message.get("reasoning_content"))})
    return {"sha256": hashlib.sha256(payload).hexdigest(), "choices_count": len(choices), "choices": choices}


def invalid_response_metadata(content):
    metadata = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    try:
        frame = json.loads(content)
    except Exception:
        return dict(metadata, json_type="unparseable")
    metadata["json_type"] = type(frame).__name__
    if not isinstance(frame, dict):
        return metadata
    metadata["error_type"] = type(frame.get("error")).__name__
    choices = frame.get("choices")
    metadata["choices_type"] = type(choices).__name__
    if isinstance(choices, list):
        metadata["choices_count"] = len(choices)
        shapes = []
        for choice in choices[:4]:
            shape = {"choice_type": type(choice).__name__}
            if isinstance(choice, dict):
                message = choice.get("message")
                shape.update(message_type=type(message).__name__, finish_reason_type=type(choice.get("finish_reason")).__name__)
                if isinstance(message, dict):
                    shape.update({key + "_type": type(message.get(key)).__name__ for key in ("role", "content", "tool_calls", "reasoning")})
            shapes.append(shape)
        metadata.update(choice_shapes=shapes, truncated_choices=max(0, len(choices) - len(shapes)))
    return metadata


UPSTREAM_BASE = "http://127.0.0.1:8000"


def create_observer_app(helper, arm, evidence, strict=False, upstream_base=UPSTREAM_BASE):
    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
            app.state.client = client
            yield

    class DownstreamDisconnect(Exception):
        pass

    class Endpoint:
        def disconnected(self, state):
            if not state["disconnected"]:
                state["disconnected"] = True
                evidence["client_disconnects"] += 1

        async def send(self, send, message, state):
            try:
                await send(message)
            except OSError as error:
                self.disconnected(state)
                raise DownstreamDisconnect() from error

        async def disconnect(self, request):
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    return

        async def forward(self, request, body, send, state):
            started = False
            chat = request.url.path.endswith("/chat/completions")
            excluded = {b"host", b"connection", b"content-length", b"transfer-encoding"}
            headers = [(k, v) for k, v in request.scope["headers"] if k.lower() not in excluded]
            url = upstream_base.rstrip("/") + request.url.path
            if request.url.query:
                url += "?" + request.url.query
            try:
                async with request.app.state.client.stream(request.method, url, content=body, headers=headers) as response:
                    status = response.status_code
                    if status != 200:
                        target = "task_http_errors" if status in (400, 422) else "infrastructure_errors"
                        evidence[target].append("http_" + str(status))
                    omitted = {b"connection", b"keep-alive", b"transfer-encoding", b"content-length", b"content-encoding"}
                    outgoing = [(k, v) for k, v in response.headers.raw if k.lower() not in omitted]
                    await self.send(send, {"type": "http.response.start", "status": status, "headers": outgoing}, state)
                    started = True
                    chunks = []
                    async for chunk in response.aiter_bytes():
                        if chat and status == 200:
                            chunks.append(chunk)
                        await self.send(send, {"type": "http.response.body", "body": chunk, "more_body": True}, state)
                    if chat and status == 200:
                        content = b"".join(chunks)
                        try:
                            frames = backend_frames(content)
                        except Exception:
                            evidence["invalid_response_metadata"].append(invalid_response_metadata(content))
                            raise
                        evidence["schema_accepted_responses"] += 1
                        action = helper.has_model_action(content)
                        if action:
                            evidence["successful"].append(arm)
                        else:
                            evidence["empty_actions"] += 1
                            evidence["empty_response_metadata"].append(empty_response_metadata(frames, content))
                    await self.send(send, {"type": "http.response.body", "body": b"", "more_body": False}, state)
            except asyncio.CancelledError:
                raise
            except DownstreamDisconnect:
                return
            except (httpx.ReadTimeout, TimeoutError):
                evidence["request_timeouts"] += 1
                await self.finish_failure(send, started, state)
            except Exception:
                evidence["infrastructure_errors"].append("invalid_provider_response_or_connection_failure")
                await self.finish_failure(send, started, state)

        async def finish_failure(self, send, started, state):
            try:
                if not started:
                    await self.send(send, {"type": "http.response.start", "status": 502,
                                         "headers": [(b"content-type", b"application/json")]}, state)
                await self.send(send, {"type": "http.response.body", "body": b"", "more_body": False}, state)
            except DownstreamDisconnect:
                pass

        async def __call__(self, scope, receive, send):
            request = Request(scope, receive=receive)
            state = {"disconnected": False}
            app = request.app
            app.state.active += 1
            forwarding = watching = None
            try:
                try:
                    body = await request.body()
                except ClientDisconnect:
                    self.disconnected(state)
                    return
                chat = request.url.path.endswith("/chat/completions")
                if chat:
                    try:
                        model = json.loads(body)["model"]
                    except Exception:
                        evidence["infrastructure_errors"].append("wrong_alias_or_invalid_request")
                        await self.finish_failure(send, False, state)
                        return
                    evidence["seen"].append(model)
                    if model != arm:
                        evidence["infrastructure_errors"].append("wrong_alias_or_invalid_request")
                        await self.finish_failure(send, False, state)
                        return
                forwarding = asyncio.create_task(self.forward(request, body, send, state))
                watching = asyncio.create_task(self.disconnect(request))
                done, _ = await asyncio.wait((forwarding, watching), return_when=asyncio.FIRST_COMPLETED)
                if watching in done:
                    await watching
                if forwarding in done:
                    await forwarding
                else:
                    self.disconnected(state)
                    evidence["cancelled_upstream_requests"] += 1
                    forwarding.cancel()
                    await asyncio.gather(forwarding, return_exceptions=True)
            except asyncio.CancelledError:
                evidence["infrastructure_errors"].append("observer_handler_cancelled_before_clean_completion")
                raise
            except Exception:
                evidence["infrastructure_errors"].append("observer_handler_failure")
                raise
            finally:
                pending = [task for task in (forwarding, watching) if task is not None and not task.done()]
                for task in pending:
                    if not task.cancelling():
                        task.cancel()
                tasks = [task for task in (forwarding, watching) if task is not None]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                app.state.active -= 1

    app = Starlette(routes=[Route("/{path:path}", Endpoint(), methods=["GET", "POST"])], lifespan=lifespan)
    app.state.active = 0
    return app


def observe_run(helper, arm, command, log, env, cwd, strict=False, *, upstream_base=UPSTREAM_BASE):
    evidence = {"seen": [], "successful": [], "infrastructure_errors": [], "task_http_errors": [],
                "request_timeouts": 0, "client_disconnects": 0, "empty_actions": 0, "cancelled_upstream_requests": 0,
                "schema_accepted_responses": 0, "empty_response_metadata": [], "invalid_response_metadata": []}
    app = create_observer_app(helper, arm, evidence, strict, upstream_base)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, loop="asyncio", http="h11", lifespan="on",
                            log_level="critical", access_log=False, log_config=None, timeout_graceful_shutdown=310)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    child_env = dict(env, MODEL_PROXY_URL="http://127.0.0.1:" + str(port) + "/v1")
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            require(thread.is_alive() and time.monotonic() < deadline, "ASGI observer failed to start")
            time.sleep(0.01)
        helper.run(command, log, child_env, cwd)
    finally:
        server.should_exit = True
        thread.join(timeout=315)
        drained = not thread.is_alive() and app.state.active == 0
        listener.close()
        write_json(Path(str(log) + ".routing.json"), dict(evidence, observed_at=utc(), drained=drained))
        require(drained, "Backend requests did not drain; qualification uncertain")
    faults = list(evidence["infrastructure_errors"])
    if strict:
        faults += evidence["task_http_errors"] + (["request_timeout"] if evidence["request_timeouts"] else [])
        helper.verify_routing(arm, evidence["successful"], faults)
    else:
        require(not faults, "Holdout backend/provenance infrastructure fault")
    require(evidence["seen"] and all(model == arm for model in evidence["seen"]), "Unexpected adapter routing")
    return {"model": arm, "successful_requests": len(evidence["successful"]), "errors": len(faults),
            "task_http_errors": evidence["task_http_errors"], "request_timeouts": evidence["request_timeouts"],
            "client_disconnects": evidence["client_disconnects"], "empty_actions": evidence["empty_actions"],
            "cancelled_upstream_requests": evidence["cancelled_upstream_requests"],
            "schema_accepted_responses": evidence["schema_accepted_responses"],
            "empty_response_metadata": evidence["empty_response_metadata"],
            "invalid_response_metadata": evidence["invalid_response_metadata"]}


def canary(helper, root, arm, budget, env):
    with (root / "results/controls_train/controls.csv").open() as stream:
        tasks = sorted(row["instance_id"] for row in csv.DictReader(stream)
                       if row["gold_resolved"] == "True" and row["empty_resolved"] == "False"
                       and row["empty_exit_code"] == "1")
    require(bool(tasks), "No eligible non-holdout canary")
    result = root / ("results/v025_canary_" + arm)
    routing = observe_run(helper, arm,
                          helper.eval_command(root / ("subs/e4b_" + arm + "_s1"), result, [tasks[0]], budget, 1),
                          root / ("logs/v025_canary_" + arm + ".log"), env, root / "data/competition", strict=True)
    rows = validate_canary_result(result / "task_results.jsonl", tasks[0], budget)
    return dict(rows, **routing, task=tasks[0])


def planned_outputs(root):
    return [root / ("results/v025_canary_" + arm) for arm in ARMS] + [
        root / ("results/v025_" + arm.lower() + "_e4b_s" + str(seed))
        for arm in ARMS for seed in (1, 2, 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--parent-start", required=True)
    parser.add_argument("--a1-start", type=float, required=True)
    parser.add_argument("--a2-start", type=float, required=True)
    parser.add_argument("--retry-note", type=Path, required=True)
    parser.add_argument("--retry-note-sha", required=True)
    parser.add_argument("--wait-seconds", type=float, default=43200)
    args = parser.parse_args()
    root, legacy = args.root.resolve(), args.legacy_root.resolve()
    helper = load_helper(root)
    require(sha(BUDGET_POLICY_SOURCE) == BUDGET_POLICY_SHA, "Budget policy source changed")
    contract = root / "contracts/contract-v0.2.freeze.json"
    require(sha(contract) == FREEZE_SHA, "Original freeze manifest changed")
    doc = helper.freeze(contract, FREEZE_SHA)
    require(sha(args.retry_note) == args.retry_note_sha, "Retry note changed")
    validate_training(helper, legacy, "A1", args.a1_start)
    tasks = Path("data/competition/tasks.jsonl")
    tasks_sha = sha(root / tasks)
    require(sha(legacy / tasks) == tasks_sha, "Initial task dataset differs between roots")
    for path in planned_outputs(root):
        require(not path.exists(), "Retry destination already exists: " + str(path))
    for arm in ARMS:
        require(not (root / ("logs/v025_canary_" + arm + ".log")).exists(), "Canary log already exists")
        for seed in (1, 2, 3):
            require(not (root / ("logs/v025_" + arm + "_s" + str(seed) + ".log")).exists(), "Evaluation log already exists")
    evidence_dir = root / "migration/recovery-v025/evaluation"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    initial = {"registered_at": utc(), "freeze_sha256": FREEZE_SHA, "helper_sha256": HELPER_SHA,
               "runner_sha256": sha(Path(__file__)), "tests_sha256": sha(Path(__file__).with_name("test_recover_eval_v025.py")), "cancellation_tests_sha256": sha(Path(__file__).with_name("test_proxy_cancellation_v025.py")), "task_dataset_sha256": tasks_sha, "retry_note": str(args.retry_note),
               "retry_note_sha256": args.retry_note_sha, "legacy_parent": args.parent,
               "legacy_parent_start": args.parent_start, "outputs": [str(p) for p in planned_outputs(root)],
               "replacement_rule": "v025 trained-arm runs are primary; legacy, v021, v022, v023 and v024 trained-arm runs are diagnostic",
               "backend_response_policy": "nonempty choices with typed assistant messages; successful counts action-bearing responses only; valid empty responses are diagnostic",
               "observer_transport": "ASGI/httpx per-request disconnect cancellation; shared client retained for sibling requests",
               "canary_episode_rule": "after strict routing success, accept only no error or exact configured session/turn/tool budget exhaustion; reject conflicting error fields",
               "budget_policy_source": {"path": str(BUDGET_POLICY_SOURCE), "sha256": BUDGET_POLICY_SHA},
               "backend_classification": {"task_local": ["HTTP400", "HTTP422", "read_timeout", "client_disconnect", "HTTP200_empty_action"],
                                          "infrastructure": ["wrong_alias", "HTTP401", "HTTP403", "HTTP404", "HTTP429", "HTTP5xx",
                                                             "unexpected_HTTP_status", "connection_refusal_or_reset", "unparseable_provider_payload"],
                                          "required": "observed_correct_alias; zero_infrastructure_faults; bounded_request_drain; successful_action_required_for_canaries_only"}}
    write_json(evidence_dir / "operational-manifest.json", initial)
    print(utc(), "waiting for legacy chain completion", flush=True)
    wait_parent(args.parent, args.parent_start, args.wait_seconds)
    assert_idle((root, legacy))
    receipts = []
    (evidence_dir / "legacy-diagnostics").mkdir()
    copy_verified(legacy / "logs/overnight.log", evidence_dir / "legacy-diagnostics/overnight.log", receipts)
    for name, expected in doc["files"].items():
        if name.startswith(("contracts/", "results/", "tools/")):
            continue
        require(sha(legacy / name) == expected, "Legacy frozen input changed: " + name)
    require(sha(root / tasks) == sha(legacy / tasks) == tasks_sha, "Task dataset changed between roots")
    for arm in ARMS:
        validate_training(helper, legacy, arm, getattr(args, arm.lower() + "_start"))
        for suffix in ("", ".meta.json", ".train_log.jsonl"):
            relative = Path("sft/runs/" + arm + suffix)
            copy_verified(legacy / relative, root / relative, receipts)
        copy_verified(legacy / ("logs/train_" + arm + ".log"), root / ("logs/train_" + arm + ".log"), receipts)
        for seed in doc["seeds"]:
            relative = Path("subs/e4b_" + arm + "_s" + str(seed))
            copy_verified(legacy / relative, root / relative, receipts)
            validate_submission(root, arm, seed)
            result = Path("results/" + arm.lower() + "_e4b_s" + str(seed))
            copy_diagnostic(legacy / result, root / result, receipts)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            require(sha(legacy / "subs/e4b_arms_manifest/adapters" / arm / name) ==
                    sha(root / "sft/runs" / arm / name), "Server manifest adapter differs from checkpoint")
    write_json(evidence_dir / "transfer-receipt.json", receipts)
    identity = server_identity(legacy)
    require({BASE, "A1", "A2"} <= {item["id"] for item in helper.api("models")["data"]},
            "Server registry missing required models")
    watched = {}
    for arm in ARMS:
        for seed in doc["seeds"]:
            folder = root / ("subs/e4b_" + arm + "_s" + str(seed))
            watched[str(folder)] = inventory(folder)
        for base in (root, legacy):
            folder = base / "sft/runs" / arm
            watched[str(folder)] = inventory(folder)
    manifest_adapters = legacy / "subs/e4b_arms_manifest/adapters"
    watched[str(manifest_adapters)] = inventory(manifest_adapters)

    def verify():
        helper.freeze(contract, FREEZE_SHA)
        require(sha(BUDGET_POLICY_SOURCE) == BUDGET_POLICY_SHA, "Budget policy source changed during execution")
        require(sha(args.retry_note) == args.retry_note_sha, "Retry note changed during execution")
        require(sha(Path(__file__)) == initial["runner_sha256"], "Recovery runner changed")
        require(sha(Path(__file__).with_name("test_recover_eval_v025.py")) == initial["tests_sha256"], "Recovery tests changed")
        require(sha(Path(__file__).with_name("test_proxy_cancellation_v025.py")) == initial["cancellation_tests_sha256"], "Cancellation tests changed")
        require(server_identity(legacy) == identity, "Serving instance changed")
        for folder, expected in watched.items():
            require(inventory(Path(folder)) == expected, "Adapter/submission changed: " + folder)
        require(sha(root / tasks) == sha(legacy / tasks) == tasks_sha, "Task dataset changed between roots")

    env = dict(os.environ)
    env.pop("MODEL_PROXY_URL", None)
    env.update(OPENAI_BASE_URL="http://127.0.0.1:8000/v1", OPENAI_API_BASE="http://127.0.0.1:8000/v1",
               OPENAI_API_KEY="EMPTY", LOCAL_API_KEY="EMPTY", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OTEL_SDK_DISABLED="true", LITELLM_TELEMETRY="False",
               LITELLM_LOCAL_MODEL_COST_MAP="True")
    gates = {"server": identity, "task_dataset_sha256": sha(root / tasks), "watched_files": watched, "arms": {}}
    for arm in ARMS:
        verify()
        helper.effect(arm)
        routing = canary(helper, root, arm, doc["effective_budget"], env)
        verify()
        gates["arms"][arm] = {"passed_at": utc(), "effect_exceeds_base_repeat_drift": True, "routing": routing}
    gates["all_passed_at"] = utc()
    write_json(evidence_dir / "gates.json", gates)
    for arm in ARMS:
        for seed in doc["seeds"]:
            assert_idle((root, legacy))
            verify()
            result = root / ("results/v025_" + arm.lower() + "_e4b_s" + str(seed))
            require(not result.exists(), "Retry destination appeared")
            started = utc()
            routing = observe_run(helper, arm,
                                  helper.eval_command(root / ("subs/e4b_" + arm + "_s" + str(seed)), result,
                                                      doc["holdout_ids"], doc["effective_budget"]),
                                  root / ("logs/v025_" + arm + "_s" + str(seed) + ".log"), env, root / "data/competition")
            verify()
            rows = validate_rows(result / "task_results.jsonl", doc["holdout_ids"])
            write_json(evidence_dir / (arm + "_s" + str(seed) + "-complete.json"),
                       dict(rows, started_at=started, completed_at=utc(), routing=routing, gates_sha256=sha(evidence_dir / "gates.json")))
            print(utc(), arm, "seed", seed, "completed; outcomes sealed", flush=True)
    write_json(evidence_dir / "completed.json", {"completed_at": utc(), "outcomes_opened": False})
    print(utc(), "six qualified retries completed; outcomes sealed", flush=True)


if __name__ == "__main__":
    main()
