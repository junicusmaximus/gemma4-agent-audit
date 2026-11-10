# Exploration-informed paired mechanism intervals

`analyze_mechanism_v031.py` is separate from the frozen v0.2 primary analyzer. It consumes the corrected `mechanism-v032` metrics generation. All intervals are descriptive secondary analyses specified after exploratory behavior and primary outcomes were seen. No metric in this report is a confirmatory primary endpoint.

## Inputs and execution

Execution requires an explicitly pinned input manifest. No actual-data execution has been performed during implementation. After independent review, use:

```sh
HOME_DIR/miniconda3/envs/swegemma/bin/python tools/analyze_mechanism_v031.py \
  --root PROJECT_ROOT/gemma4 \
  --input-manifest /ABSOLUTE/PATH/TO/REVIEWED-MANIFEST.json \
  --manifest-sha256 REVIEWED_MANIFEST_SHA256 \
  --output-dir PROJECT_ROOT/gemma4/analysis/mechanism-ci-v031
```

Manifest schema (paths may be relative to the canonical project root; resolved paths must remain inside it):

```json
{
  "schema_version": 1,
  "input_generation": "mechanism-v032",
  "metrics_path": "analysis/mechanism-v032/per_task_primary40.jsonl",
  "metrics_sha256": "PINNED_SHA256",
  "task_ids_path": "analysis/mechanism-v032/primary40_ids.txt",
  "task_ids_sha256": "PINNED_SHA256",
  "producer_path": "tools/REVIEWED_CORRECTED_PRODUCER.py",
  "producer_sha256": "PINNED_SHA256",
  "sealed_report_sha256": "b2a2ebba991f8fb2c61d548beaf054c03d4163ce0c4e61f0f12b77b0083d30a4"
}
```

The analyzer requires the exact pinned corrected producer SHA256, 36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d. An unknown or comment-modified producer is rejected even if the manifest is re-pinned. It requires 40 IDs matching the sealed report, all nine expected A0/A1/A2-by-seed runs, one record per task/run, and count/rate consistency. The manifest, producer, data and population hashes are recorded and rechecked before writing. Existing output directories are refused.

## Estimands

- M1: fraction of agent steps with observed nonempty thinking text. It is not an LLM-call probability or a claim about internal reasoning.
- M2: loop indicator (`max_repeat >= 5`) and the secondary max-repeat count.
- M3: parsed `submit_patch` presence.
- M4: each fixed opener category and any of the three target opener patterns.
- M5: task-level repeated-next-call fractions in error/silent/normal observation strata, using the corrected producer classification.
- M6: excluded from these intervals. Saved thought lengths require explicit unit/tokenizer provenance; they are not automatically generation-token counts.

Each task carries its three seed positions. For each metric/comparison, first average each arm's observed seed values within that task. Use only tasks where both arms have at least one observed seed value. Bootstrap these paired task means together, 20,000 draws, NumPy `Generator(PCG64)`, seed `20261103`, linear 2.5%/97.5% quantiles. Proportions also have percentage-point effects; max-repeat uses call-count units.

M5 support is conditional on observed eligibility and differs by stratum/comparison. Report each arm's eligible task count, common paired count, per-task seed coverage and values. Never replace missing strata with zero. A separate sensitivity restricts to tasks with all three seed values in each arm. With zero pairs the effect and interval are undefined; with one pair the point difference is reported but its CI is undefined, rather than a misleading zero-width interval.

A1-A0, A2-A0 and A2-A1 are all exploratory. Intervals have no multiplicity correction and do not identify causal mechanisms or establish equivalence at a floor. The study remains conditional on one public repository and one training run per trained arm.

## Validation

```sh
cd PROJECT_ROOT/gemma4/tools
HOME_DIR/miniconda3/envs/swegemma/bin/python -m unittest -v test_analyze_mechanism_v031.py
```

Synthetic tests cover paired task clustering, missing-stratum support, seed coverage, sparse-pair CI handling, deterministic RNG, strict input/provenance checks, directory collisions and input changes during computation. They do not read real experiment outcomes or use the GPU.
