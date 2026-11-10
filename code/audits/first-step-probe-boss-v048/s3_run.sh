#!/bin/bash
# S3: one task (rich_2725) through the evaluation harness with the copied v043_M0_s1 submission (byte-identical copy;
# only the base URL differs, via MODEL_PROXY_URL / OPENAI_BASE_URL -> the logging proxy -> the diagnostic server).
# Mirrors the canary path (tools/overnight_v02.py:150-157 eval_command; v043 effective budget with concurrency 1).
# Whole harness capped by a user systemd scope (24 GiB, no swap) and a 15-minute wall limit; TMPDIR and caches in s3/.
set -u
ROOT=PROJECT_ROOT/gemma4
OUT=$ROOT/analysis/first-step-probe-boss-v048
PY=HOME_DIR/miniconda3/envs/swegemma/bin/python
PORT=$($PY -c "import json; print(json.load(open('$OUT/s3/proxy.json'))['port'])")
mkdir -p "$OUT/s3/tmp" "$OUT/s3/cache"
export MODEL_PROXY_URL=http://127.0.0.1:$PORT/v1 OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENAI_API_BASE=http://127.0.0.1:$PORT/v1
export OPENAI_API_KEY=EMPTY LOCAL_API_KEY=EMPTY HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OTEL_SDK_DISABLED=true
export LITELLM_TELEMETRY=False LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$OUT/s3/tmp XDG_CACHE_HOME=$OUT/s3/cache
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/s3/harness.start"
cd "$OUT/s3" || exit 9
timeout 900 systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 \
  $PY $ROOT/run_eval.py --tasks $ROOT/data/competition/tasks.jsonl --snapshots-dir $ROOT/data/competition/snapshots \
  --submission-dir "$OUT/s3/sub_v043_M0_s1" --results-dir "$OUT/s3/results" --models-yaml $ROOT/models.yaml \
  --task-ids rich_2725 --sandbox subprocess --display quiet \
  --max-time-minutes 4 --max-tool-calls 28 --max-turns 80 --timeout-seconds 240 --concurrency 1 \
  > "$OUT/s3/harness.log" 2>&1
echo $? > "$OUT/s3/harness.exit"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/s3/harness.end"
