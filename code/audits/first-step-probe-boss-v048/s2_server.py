#!/usr/bin/env python3
"""S2 diagnostic vLLM server: start | wait | stop.

start: the exact v043 api_server argv (migration/mask-evaluations-v043/evaluation/server-ready.json) with three
substitutions only: --port 8048, --gpu-memory-utilization 0.75 (headroom for full-vocabulary prompt logprobs), and
--lora-modules m0, m1, zerob (+ v035_M0_s1, the alias the copied v043_M0_s1 submission requests in S3). Environment =
the v043 server environment (run_v032_probes.environment + contract server_environment + VllmServer.build_env) with
all compile/kernel caches redirected into this directory. Own process group; log in s2/server.log.
wait: until /v1/models lists the base and all aliases; record the LoRA-loading log lines.
stop: SIGTERM to the owned process group, SIGKILL only if it survives 90 s; confirm the group is empty and the GPU idle."""
import datetime
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/first-step-probe-boss-v048"
S2 = OUT + "/s2"
PY = "HOME_DIR/miniconda3/envs/swegemma/bin/python"
READY_V043 = ROOT + "/migration/mask-evaluations-v043/evaluation/server-ready.json"
CONTRACT = ROOT + "/contracts/mask-evaluations-v043-runtime-DRAFT.json"
SERVED = "gemma-4-e4b-it-qat-w4a16-ct"
PORT, GPU_UTIL = 8048, "0.75"
LORAS = [("m0", ROOT + "/sft/runs/A1_M0_v035"), ("m1", ROOT + "/sft/runs/A1_M1_v035"), ("zerob", OUT + "/adapters/zerob"),
         ("v035_M0_s1", OUT + "/s3/sub_v043_M0_s1/adapters/v035_M0_s1")]
CACHES = {"VLLM_CACHE_ROOT": "vllm", "VLLM_CONFIG_ROOT": "vllm_config", "TRITON_CACHE_DIR": "triton",
          "TORCHINDUCTOR_CACHE_DIR": "inductor", "XDG_CACHE_HOME": "xdg", "HF_HOME": "hf"}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def v043_argv():
    members = json.load(open(READY_V043))["members"].values()
    return next(m["argv"] for m in members if "vllm.entrypoints.openai.api_server" in m["argv"])


def build_argv(old):
    out, i = [], 0
    while i < len(old):
        arg = old[i]
        if arg in ("--port", "--gpu-memory-utilization"):
            out += [arg, str(PORT) if arg == "--port" else GPU_UTIL]
            i += 2
        elif arg == "--lora-modules":
            out += [arg] + ["%s=%s" % kv for kv in LORAS]
            i += 1
            while i < len(old) and not old[i].startswith("--"):
                i += 1
        else:
            out.append(arg)
            i += 1
    return out


def server_env():
    env = dict(os.environ)
    env.pop("MODEL_PROXY_URL", None)
    for key in list(env):
        if key.startswith(("VLLM_", "CUDA_", "PYTORCH_", "NCCL_")) or key in ("OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
            env.pop(key)
    env.update(json.load(open(CONTRACT))["server_environment"])
    env.update(OPENAI_BASE_URL="http://127.0.0.1:%d/v1" % PORT, OPENAI_API_BASE="http://127.0.0.1:%d/v1" % PORT,
               OPENAI_API_KEY="EMPTY", LOCAL_API_KEY="EMPTY", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OTEL_SDK_DISABLED="true", LITELLM_TELEMETRY="False", LITELLM_LOCAL_MODEL_COST_MAP="True",
               PYTHONDONTWRITEBYTECODE="1")
    torch_lib = os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin), "lib")
    env["LD_LIBRARY_PATH"] = torch_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for key, sub in CACHES.items():
        env[key] = S2 + "/cache/" + sub
        os.makedirs(env[key], exist_ok=True)
    return env


def group_members(pgid):
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                stat = open("/proc/%s/stat" % name).read()
            except OSError:
                continue
            if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
                pids.append(int(name))
    return pids


def gpu():
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    return used, apps


def start():
    rec = json.load(open(OUT + "/lock-receipt.json"))
    if rec.get("status") != "held" or not os.path.exists("/proc/%d" % rec["pid"]):
        sys.exit("ABORT: leases not held")
    used, apps = gpu()
    if used > 100 or apps:
        sys.exit("ABORT: GPU not idle before the server start (%d MiB, %r)" % (used, apps))
    os.makedirs(S2, exist_ok=True)
    old, env = v043_argv(), server_env()
    argv = build_argv(old)
    with open(S2 + "/server.log", "x") as log:
        child = subprocess.Popen(argv, cwd=S2, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
    changed = sorted(k for k in set(env) | set(os.environ) if env.get(k) != os.environ.get(k))
    with open(S2 + "/server-launched.json", "w") as f:
        json.dump({"pid": child.pid, "pgid": os.getpgid(child.pid), "start_utc": utc(), "argv": argv, "v043_argv": old,
                   "env_changed_keys": changed, "env_selected": {k: env.get(k) for k in changed if "KEY" not in k}}, f, indent=1)
    print(json.dumps({"pid": child.pid, "argv_tail": argv[-14:]}))


def wait(limit_s=1200):
    launched = json.load(open(S2 + "/server-launched.json"))
    want = {SERVED} | {k for k, _ in LORAS}
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if not group_members(launched["pgid"]):
            sys.exit("ABORT: server group exited before readiness")
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=5) as r:
                ids = {m["id"] for m in json.loads(r.read())["data"]}
            if want <= ids:
                lines = [l.rstrip() for l in open(S2 + "/server.log", errors="replace") if "LoRA" in l]
                loaded = [l for l in lines if "Loaded new LoRA adapter" in l]
                with open(S2 + "/lora_load_lines.txt", "w") as f:
                    f.write("\n".join(lines) + "\n")
                info = {"ready_utc": utc(), "seconds": round(time.time() - t0, 1), "models": sorted(ids),
                        "lora_loaded_lines": len(loaded),
                        "all_aliases_in_load_log": all(any("'%s'" % k in l for l in loaded) for k, _ in LORAS)}
                with open(S2 + "/server-ready.json", "w") as f:
                    json.dump(info, f, indent=1)
                print(json.dumps(info))
                return
        except OSError:
            pass
        time.sleep(3)
    sys.exit("ABORT: readiness timeout")


def stop():
    launched = json.load(open(S2 + "/server-launched.json"))
    pgid, t0, method = launched["pgid"], time.time(), "TERM"
    if group_members(pgid):
        os.killpg(pgid, signal.SIGTERM)
    while group_members(pgid) and time.time() - t0 < 90:
        time.sleep(1)
    if group_members(pgid):
        os.killpg(pgid, signal.SIGKILL)
        method = "TERM then KILL after 90 s"
        time.sleep(5)
    time.sleep(3)
    used, apps = gpu()
    info = {"stopped_utc": utc(), "method": method, "group_empty": not group_members(pgid), "seconds": round(time.time() - t0, 1),
            "gpu_used_mib_after": used, "gpu_compute_apps_after": apps}
    with open(S2 + "/server-stopped.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    {"start": start, "wait": wait, "stop": stop}[sys.argv[1]]()
