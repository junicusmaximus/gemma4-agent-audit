# sft_example_audit

Read-only, CPU-only, standard-library audit (Python 3.8+) of how SWE-agent trajectories were packed into
LoRA-SFT chunks, per arm key and per arm x tier. Every field is defined in the module docstring.

## Definitions in brief
- History-dropped chunk: starts at a later item of its chain, so the header is followed by ONE later tool response. Orphan: history-dropped and, after the last `<|turn>` before the first loss run, a `<tool_response|>` id but no `<|tool_response>` id.
- Loss run: maximal run of supervised tokens (one per step). Empty thought: `<channel|>` follows the optional opener at once. Categories: channel markers, thought, tool call, tool-response opener, turn end, other.
- Text flags per run: `__public_sft_output=` wrapper, canonical `find /workspace -maxdepth 1|2 -not -path` opener, `call:submit_patch`. Joint table: history-dropped x empty thought x (wrapper or canonical opener).
- Served reference: share of served LLM steps whose `prompt_tokens` exceeds the arm's training `loss_pos_p90`.

## Run
```
python sft_example_audit.py --data trajectories.jsonl --arms arms.json --marker-ids markers.json [--arm-keys a,b,c] [--trace-dir DIR ...] [--limit N] --out OUT_DIR
```
`--arm-keys` defaults to the seven arm names (an absent key is reported as missing); `--trace-dir` (repeatable) reads `DIR/traces/trace_*.json`; `--limit N` audits the first N trajectories. The data file is streamed; a progress line goes to stderr every 200 trajectories. Only `chunks.jsonl`, `summary.json`, `summary.md`, `inputs.sha256` (sha256 and size of the inputs, trace-file counts) are written to OUT_DIR, overwriting; a run that would overwrite an input is refused.

## markers.json
Dump the nine marker ids and the ids of `<|channel>thought` + newline from the tokenizer, e.g.
```
{"<bos>": 2, "<|turn>": 105, "<turn|>": 106, "<|channel>": 100, "<channel|>": 101, "<|tool_call>": 48, "<tool_call|>": 49, "<|tool_response>": 50, "<tool_response|>": 51, "channel_open_ids": [100, 45518, 107]}
```

## Tests
`py -3 -m unittest test_sft_example_audit -v` (toy vocabulary, hand-computed values, no real data; run on Python 3.8 and 3.13).

## Conservative readings where the brief left room
- `alignment_uncertain` is also set when a chain's train-segment count differs from its loss-run count (a dropped item would shift the offsets); text flags are then null and the runs appear as `text_unknown`.
- Text-flag shares use runs with known flags; aggregate `loss_tokens` are recounted from labels (recorded `n_loss_tokens` mismatches are counted).
- Orphan span: after the last `<|turn>` id (excluded; none means the whole prefix). Chunk median: mean of the two middle values. `canonical_opener`: any `run_command` marker immediately followed by the find pattern.
- Served LLM step: `metrics` dict with a numeric `prompt_tokens`; "above" is strictly greater; unreadable traces are counted. `chain_first_step` is the minimum over chunks present, so a dropped leading item is invisible to it.
