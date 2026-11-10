#!/usr/bin/env python3
"""S2 scoring client: every ledger item through the diagnostic server's completions endpoint with prompt = the stored
token-id list, max_tokens 1, echo true, prompt_logprobs 5, temperature 0, for the base (served name) and each adapter
alias (m0, zerob, m1). Two layouts: sequential (one request in flight) and 8 concurrent requests. Logprob and rank of
each target token come from the returned prompt_logprobs (rank '>5' when the target id is absent)."""
import concurrent.futures
import datetime
import json
import os
import sys
import time
import urllib.request

ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/first-step-probe-boss-v048"
URL = "http://127.0.0.1:8048/v1/completions"
CELLS = [("base", "gemma-4-e4b-it-qat-w4a16-ct"), ("m0", "m0"), ("zerob", "zerob"), ("m1", "m1")]
RESERVE_S = 20 * 60  # GPU time kept for S3 and the server stop
sys.path.insert(0, OUT)
import probe_lib as lib  # noqa: E402


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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
    rec = json.load(open(OUT + "/lock-receipt.json"))
    if rec.get("status") != "held" or not os.path.exists("/proc/%d" % rec["pid"]):
        sys.exit("ABORT: leases not held")
    if not os.path.exists(OUT + "/s2/server-ready.json"):
        sys.exit("ABORT: server not ready")
    deadline = json.load(open(OUT + "/gpu_deadline.json"))["deadline_epoch"]
    items = [json.loads(line) for line in open(OUT + "/ledger.jsonl")]
    done = {}
    if os.path.exists(OUT + "/vllm_scores.jsonl"):  # a resumed run keeps complete passes and skips them
        for line in open(OUT + "/vllm_scores.jsonl"):
            row = json.loads(line)
            done[(row["cell"], row["layout"])] = done.get((row["cell"], row["layout"]), 0) + 1
    t0, passes, skipped, last = time.time(), [], [], 0.0
    with open(OUT + "/vllm_scores.jsonl", "a") as out:
        for layout in ("sequential", "concurrent8"):
            for cell, model in CELLS:
                if done.get((cell, layout)) == len(items):
                    continue
                if time.time() + last > deadline - RESERVE_S:
                    skipped.append({"cell": cell, "layout": layout, "reason": "GPU time ceiling (reserve for S3 + stop)"})
                    continue
                ps = time.time()
                if layout == "sequential":
                    results = [request(it, model) for it in items]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        results = list(pool.map(lambda it: request(it, model), items))
                for it, r in zip(items, results):
                    out.write(json.dumps({"stack": "vllm", "cell": cell, "layout": layout, "model": model,
                                          "item_id": it["item_id"], "target_ids": it["target_ids"], **r}) + "\n")
                out.flush()
                last = time.time() - ps
                passes.append({"cell": cell, "layout": layout, "requests": len(items), "seconds": round(last, 1),
                               "prompt_tokens_match": sum(r["prompt_tokens_match"] for r in results),
                               "targets_absent": sum(1 for r in results for x in r["rank"] if x == ">5")})
                print(json.dumps(passes[-1]), flush=True)
    with open(OUT + "/s2/score_stage.json", "w") as f:
        json.dump({"stage": "S2-score", "end_utc": utc(), "seconds": round(time.time() - t0, 1), "passes": passes,
                   "skipped": skipped}, f, indent=1)
    print(json.dumps({"S2": "done", "seconds": round(time.time() - t0, 1), "skipped": skipped}))


if __name__ == "__main__":
    main()
