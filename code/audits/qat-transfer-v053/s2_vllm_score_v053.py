#!/usr/bin/env python3
"""Stage 2 vLLM scoring client: copy of analysis/first-step-probe-boss-v048/s2_vllm_score.py. Every v048 ledger item
through the diagnostic server's completions endpoint with prompt = the stored token-id list, max_tokens 1, echo true,
prompt_logprobs 5, temperature 0, for the base (served name) and each adapter alias of s2/server-launched.json. Two
layouts: sequential (one request in flight) and 8 concurrent requests. Logprob and rank of each target token come from
the returned prompt_logprobs (rank '>5' when the target id is absent). Rows: vllm_scores_v053.jsonl, stack vllm_qat_nopc."""
import concurrent.futures
import datetime
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/qat-transfer-v053"
LEDGER = ROOT + "/analysis/first-step-probe-boss-v048/ledger.jsonl"
URL = "http://127.0.0.1:8048/v1/completions"
SERVED = "gemma-4-e4b-it-qat-w4a16-ct"
RESERVE_S = 5 * 60  # GPU time kept for the server stop
sys.path.insert(0, OUT)
import probe_lib as lib  # noqa: E402  (byte copy of the v048 probe_lib.py)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def request(item, model):
    body = {"model": model, "prompt": item["ids"], "max_tokens": 1, "echo": True, "prompt_logprobs": 5, "temperature": 0.0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    pl = choice["prompt_logprobs"]
    tokens = lib.parse_prompt_logprobs(pl, item["ids"], item["target_span"])
    return {"logprob": [t["logprob"] for t in tokens], "rank": [t["rank"] for t in tokens],
            "top5_ids": [t["top5_ids"] for t in tokens], "top5_logprob": [t["top5_logprob"] for t in tokens],
            "usage_prompt_tokens": resp.get("usage", {}).get("prompt_tokens"), "prompt_logprobs_len": len(pl),
            "prompt_tokens_match": resp.get("usage", {}).get("prompt_tokens") == len(item["ids"]) and len(pl) == len(item["ids"]),
            "request_seconds": round(time.time() - t0, 2)}


def main():
    rec = read_json(OUT + "/lock-receipt.json")
    if rec.get("status") != "held" or not os.path.exists("/proc/%d" % rec["pid"]):
        sys.exit("ABORT: leases not held")
    if not os.path.exists(OUT + "/s2/server-ready.json"):
        sys.exit("ABORT: server not ready")
    deadline = read_json(OUT + "/gpu_deadline.json")["deadline_epoch"]
    cells = [("base", SERVED)] + [(alias, alias) for alias, _ in read_json(OUT + "/s2/server-launched.json")["loras"]]
    with open(LEDGER) as f:
        items = [json.loads(line) for line in f]
    t0, passes, skipped, last = time.time(), [], [], 0.0
    with open(OUT + "/vllm_scores_v053.jsonl", "x") as out:
        for layout in ("sequential", "concurrent8"):
            for cell, model in cells:
                if time.time() + last > deadline - RESERVE_S:
                    skipped.append({"cell": cell, "layout": layout, "reason": "GPU time ceiling (reserve for the stop)"})
                    continue
                ps = time.time()
                if layout == "sequential":
                    results = [request(it, model) for it in items]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        results = list(pool.map(lambda it: request(it, model), items))
                for it, r in zip(items, results):
                    out.write(json.dumps({"stack": "vllm_qat_nopc", "cell": cell, "layout": layout, "model": model,
                                          "item_id": it["item_id"], "target_ids": it["target_ids"], **r}) + "\n")
                out.flush()
                last = time.time() - ps
                passes.append({"cell": cell, "layout": layout, "requests": len(items), "seconds": round(last, 1),
                               "prompt_tokens_match": sum(r["prompt_tokens_match"] for r in results),
                               "targets_absent": sum(1 for r in results for x in r["rank"] if x == ">5")})
                print(json.dumps(passes[-1]), flush=True)
    with open(OUT + "/s2/score_stage.json", "x") as f:
        json.dump({"stage": "S2-vllm-score", "end_utc": utc(), "seconds": round(time.time() - t0, 1), "passes": passes,
                   "skipped": skipped}, f, indent=1)
    print(json.dumps({"S2": "done", "seconds": round(time.time() - t0, 1), "skipped": skipped}))


if __name__ == "__main__":
    main()
