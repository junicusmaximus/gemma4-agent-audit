# Registered serving-probe analysis v0.3.2

The separate `analyze_probes_v032.py` implements the analysis plan registered before the four new serving conditions began. It does not change the frozen v0.2 primary report or the earlier exploratory mechanism analyzer.

## Qualification before decoding

The analyzer first requires the all-four completion marker, all four run receipts, both phase gate files, activation, and both server launch/readiness records. A stopped generation is rejected. It then verifies:

- Both exact registered plan hashes and sidecars, their linkage, the four-condition intervention matrix, budgets, source pins and directory inventories.
- Phase aliases, launch/readiness/server identities, chronology, all specified canaries, exact accepted canary budget errors, correct routing, drained requests and absence of infrastructure/malformed-response faults.
- Each completed run's result-file hash, phase-gate hash, result path, routing receipt and chronology; the final-server linkage.
- The three historical seed-1 baseline result hashes and all 144 baseline trace hashes against historical source manifest `f6a1138406313e6e27a41d59fc442ca51b0b727230b19fdc3c5568945034c262`.
- The new conditions' trace inventory and missingness, captured only after the entire batch completes. Historical missingness cannot silently change. New missing or unreadable traces remain explicitly missing for mechanism metrics.
- Approved metric/bootstrap/row-validator source hashes and the fixed local E4B tokenizer assets.

Only after those checks does it decode task-result rows or new traces. Outcome rows require all 48 unique task IDs, literal boolean resolution, the pinned serializer schema and no resolution/error contradiction. Model task failures remain unsolved; infrastructure-invalid or incomplete generations are rejected. Input hashes and missingness are rechecked before any report is written.

## Registered comparisons

Every comparison uses a single decoding seed and paired whole-task resampling: 20,000 draws, NumPy PCG64, seed 20261103, linear percentile interval. All seven contrasts are reported over primary40 and all48:

1. A1a8 - A1
2. A2a8 - A2
3. A1_t07 - A1
4. A0_t07 - A0
5. A1_t07 - A0_t07
6. A1a8 - A0
7. A2a8 - A0

Resolution and M1-M5 use the fixed observable definitions from the corrected producer. M1 measures nonempty retained thinking-text presence per recorded agent step. M5 uses common support where both compared conditions have an eligible transition; no absent denominator is replaced with zero. The report includes marginal eligible-task counts, paired task IDs and transition counts. Fewer than two paired tasks yields an undefined interval with a reason.

M6 is descriptive: retained thinking text re-tokenized with the pinned local E4B tokenizer and `add_special_tokens=True`. It is not the provider's generated-reasoning token count. The condition summary reports the median across tasks of their within-task median text lengths; absent thinking text remains undefined. Per-task values and tokenizer provenance remain in JSON.

The comparisons are descriptive sensitivity experiments on one public repository, with one decoding seed per new condition. No multiplicity adjustment, causal mechanism claim or equivalence claim follows from these intervals. Timing across server sessions is descriptive.

## Execution after review and sealing

```sh
HOME_DIR/miniconda3/envs/swegemma/bin/python tools/analyze_probes_v032.py \
  --root PROJECT_ROOT/gemma4 \
  --analysis-plan-sha f63610c9babf5754183135b84b0a9a09a2d31a5d3e793931394650d7e44df9f9 \
  --execution-plan-sha d1b13d9873f44f967114c566fff81880c31357c2f6db3d1b811687ac792f2630 \
  --output-dir PROJECT_ROOT/gemma4/analysis/probes-v032
```

The output directory must be fresh. `report.json` includes all contrasts, mechanisms, coverage, per-task data and input/source/tokenizer hashes. `report.md` lists every registered resolution contrast. `output-manifest.json` records hashes of both report artifacts.

No actual new probe outcomes were accessed during implementation. Tests use synthetic inputs and no GPU. The root controller must seal the reviewed analyzer/test hashes and both plan hashes before launching its waiting analysis process.
