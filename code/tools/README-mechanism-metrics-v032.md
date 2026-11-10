# Mechanism metrics v0.3.2

Versioned correction of `mechanism_metrics.py`; the original producer, tests, and `analysis/mechanism-v031/` remain historical artifacts. This version changes M5 observation classification and adds explicit provenance. M1-M4, M6, task filtering, aggregations, and outcome-field exclusion retain the original definitions in [README-mechanism-metrics.md](README-mechanism-metrics.md).

## The M5 correction

The observed file-tool failure shape is an observation whose `content` is JSON such as `{"error": "mandatory input parameters are not present"}`, with no status, exit code, stdout or stderr. A nonblank string in `error` is now `error`, rather than silently falling through to `silent`.

- Present nonblank `error` string: `error`.
- Present empty/whitespace `error` string: apply the original status/exit-code/stderr/stdout rules.
- Present nonstring `error`, including null, booleans, numbers, arrays and objects: `unknown`, excluded from M5 stratum denominators.
- Missing `error`: original behavior. In particular, a genuine successful command with exit code 0 and blank stdout/stderr remains `silent`. No status is invented from missing fields.

The input shape correction can move M5 eligible pairs and repeat rates. It does not alter calls, traces, task membership, outcomes, or any other metric definition.

## M1 and M6 are retained-trace proxies

M1 is **the per-task share of recorded agent steps carrying observed nonempty thinking text**, followed by the original run mean of defined task shares. It is a step-based presence proxy, not a provider thinking-token rate. Recorded ATIF cannot reliably distinguish an absent raw thinking channel from an empty channel.

M6 is **fixed-tokenizer retokenization of retained nonblank thinking text**: `len(tokenizer(text, add_special_tokens=True)['input_ids'])`. It is not the provider's original thinking-token count. There is no chat-template application. Special tokens were included by the original implicit Hugging Face default; the option is now explicit with the same policy.

The production commands below use the local E4B tokenizer, not word counts. The CLI's original no-tokenizer mode is retained for synthetic tests and explicitly marked `whitespace_words` / `diagnostic_only` in provenance. Do not use that mode for the production mechanism tables.

Fixed tokenizer identifier: `PROJECT_ROOT/gemma4/models/gemma-4-e4b-it`.

| Asset | SHA256 |
|---|---|
| tokenizer.json | cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f |
| tokenizer_config.json | 97af7f9a43d9c26b4ae9558b39c2d6ffed60b9869f2cf2b9a5282b05c6dac67c |
| chat_template.jinja | 55572b8d3c8342044e25874c73fe5234b661fa0a57a57f6ef75b58e03d7d959a |

## Output and provenance

Summary JSON remains a list of run summaries. Per-task JSONL remains exactly one (run, task) record per line, without metadata-only rows. Metric field names remain compatible with the original producer.

Each invocation writes a sidecar, defaulting to `<summary-stem>.provenance.json`; `--provenance PATH` can specify it explicitly. It records:

- generation `mechanism-v032`, producer path/hash and original producer hash;
- tokenizer identifier, relevant asset hashes, deterministic asset fingerprint, Transformers version and tokenization method/options;
- M1 step-proxy and M6 retained-text labels and units;
- loop/output-token settings, task-ID file hash, run-result input hashes, and metric output paths/hashes.

Tokenizer assets are hashed before and after computation; a change aborts before writing metric outputs. Model weight shards are neither loaded nor hashed by the provenance helper. `resolved`, `test_exit_code` and trace `final_metrics` remain unused by metric computation.

## Generation commands (run only after review)

These Bash commands create fresh v0.3.2 outputs from the same nine runs and the existing 40-ID population file. They do not overwrite v0.3.1. They have **not** been executed as part of the implementation task.

```bash
project_dir=PROJECT_ROOT/gemma4
python_bin=HOME_DIR/miniconda3/envs/swegemma/bin/python
output_dir="$project_dir/analysis/mechanism-v032"
mkdir -p "$output_dir"
run_args=()
for run_name in a0_e4b_s1 a0_e4b_s2 a0_e4b_s3 \
                v025_a1_e4b_s1 v025_a1_e4b_s2 v025_a1_e4b_s3 \
                v025_a2_e4b_s1 v025_a2_e4b_s2 v025_a2_e4b_s3; do
  run_args+=(--run-dir "$project_dir/results/$run_name")
done
common=("$python_bin" "$project_dir/tools/mechanism_metrics_v032.py" \
        "${run_args[@]}" --tokenizer "$project_dir/models/gemma-4-e4b-it" \
        --loop-threshold 5 --max-output-tokens 8192)
"${common[@]}" --task-ids "$project_dir/analysis/mechanism-v031/primary40_ids.txt" \
  --json "$output_dir/summary_primary40.json" \
  --per-task "$output_dir/per_task_primary40.jsonl" \
  --provenance "$output_dir/summary_primary40.provenance.json"
"${common[@]}" --json "$output_dir/summary_all48.json" \
  --per-task "$output_dir/per_task_all48.jsonl" \
  --provenance "$output_dir/summary_all48.provenance.json"
```

The primary invocation yields 40 task records per run; the all-task invocation uses all 48 per run. Existing producer behavior permits reusing a named output path, so use only the new v0.3.2 paths shown. Root creates the separate sealed input manifest after review and generation; it is not synthesized by this metric tool.

## Tests

```bash
cd PROJECT_ROOT/gemma4/tools
HOME_DIR/miniconda3/envs/swegemma/bin/python -m unittest -v test_mechanism_metrics_v032.py
```

The 28 original tests are retained in the versioned test copy; the fake-tokenizer fixture now checks explicit `add_special_tokens=True`. Seven additional tests cover error-only file-tool observations, unsupported error types, preserved silent success, M5 pair reassignment, unchanged JSONL row structure, fixed-tokenizer sidecar options/hashes, and asset-fingerprint changes. No real mechanism output is recomputed by these tests.
