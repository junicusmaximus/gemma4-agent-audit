# Different Collapse Outcomes After Finetuning on Successful Trajectories

A controlled study of Gemma 4 E4B: self generated versus teacher trajectories at equal updates with QAT and serving format audits

Published 2026-11-10: https://github.com/junicusmaximus/gemma4-agent-audit

This is the public verification package of the paper. It holds the reports and per-task tables and contracts and pre-registrations and code and figures that support the claims K1 to K9 and Tables 1 and 2 and the external panel paragraph. The folder is complete on its own. It has no symlinks. Size: 342 files and about 47 MB.

## Licence and scope

Copyright 2026 Jun Hyung Lee.

- Own code is released under the Apache License 2.0 (`LICENSE`). This covers `code/` and `verify/` and the plotting scripts in `figures/`.
- Own documents and figures and tables are released under Creative Commons Attribution 4.0 International (`LICENSE-CC-BY-4.0`). This covers the reports and per-task tables and contracts and pre-registrations and figures and the text files written for this package.
- Third-party material keeps its own licence and attribution. It is not included and nothing here relicenses it (see `NOTICE.md`).

## How this work was produced

Claude and Codex assisted with literature review, experiment planning, code development and execution, data analysis and interpretation, and manuscript drafting and editing. The author directed the work and takes responsibility for its methods, reported results, conclusions and final manuscript. In the files, Boss is the orchestrating Claude Code session that planned, delegated and ran experiments. Codex is an OpenAI Codex session that ran or verified work in alternating roles. oracle and momus are read only reviewer agents. None of them is an author.

## What is here

| folder | content |
|---|---|
| `preregistrations/` | pre-registrations and addenda and amendments and change records and the decisions that fix a rule and their freeze receipts |
| `contracts/` | pinned runtime and analysis contracts with their `.sha256` sidecars (for generations that were never pinned the draft that is their recorded contract) |
| `analyses/<name>/` | per analysis: `report.json` and `report.md` and the per-task table(s) and the run's `completed.json` |
| `code/` | the metric producer and analyzers and runners and pin scripts and training wrappers and audit scripts; `code/templates/` holds the diff of our edits to the Gemma chat template and a script that applies it |
| `figures/` | the figures (png and svg) and `figure_data.csv` (one row per plotted value with source file and hash) and the plotting scripts and the figures README |
| `verify/` | `verify_paper_numbers.py` and `synthetic_fixture.py` and `verify_paper_numbers.ipynb` |
| `MANIFEST.tsv` and `SHA256SUMS` | every file with size and hashes and class and change |

## Hashes

A file whose change is not "none" in `MANIFEST.tsv` is a public copy and is not byte-identical to the sealed original. The paper cites the sha256 of the sealed originals as provenance. Verification of the public package uses the public hashes (`public_sha256` in `MANIFEST.tsv` and `SHA256SUMS`). In the public copies the only edits are machine paths and one host name field. Machine paths are replaced by the placeholders `PROJECT_ROOT` and `HOME_DIR` and the host name field by `HOST_NAME`. Nothing else inside a sealed file was changed.

`MANIFEST.tsv` has the columns logical path and bytes and public_sha256 and original_sha256 and class and change. The class is sealed copy or code or document or figure or new. The change is none or paths or new. A file marked new was written for this package and has no original.

## Verify the numbers

This needs Python 3.10 or newer and numpy. It needs no internet.

```
python3 verify/synthetic_fixture.py          # tests the bootstrap and the label rule on hand-made tables
python3 verify/verify_paper_numbers.py       # recomputes the paper's numbers and prints one PASS or FAIL or NOT RECOMPUTABLE line per check
sha256sum -c SHA256SUMS                      # file integrity of the public copies
```

`verify/verify_paper_numbers.ipynb` does the same in a notebook. It finds the package in a Kaggle input folder or next to itself or through the environment variable `PAPER_PACKAGE`.

The script recomputes four groups of numbers.

1. The resolved counts per arm and decoding seed of every row of Table 1.
2. The paired task-bootstrap intervals for every two-arm report and the intervals printed in the paper. The bootstrap uses 20,000 resamples and `numpy.random.Generator(PCG64(20261103))` and the per-task mean over the three decoding seeds of treated minus baseline and linear 2.5 and 97.5 % quantiles.
3. The loop and submit rates from `per_task.jsonl` and the frozen termination labels. NON-COLLAPSED means loop at most 0.30 and submit at least 0.80. COLLAPSED means loop above 0.40 or submit below 0.70.
4. The repair counts and submit counts of the external panel.

Numbers that need agent traces are printed as `NOT RECOMPUTABLE from public tables` with the reason. They are never counted as PASS. This applies to loop and submit for A0 and A1 and A2 (no trace table) and to loop and the labels of the external panel (the external table has no loop flag). The script exits with status 1 if any check fails. If a file is removed from the package the checks that need it are reported as NOT RECOMPUTABLE.

## Claim to files (public files only)

| claim | result files | pre-registration and contract | code |
|---|---|---|---|
| K1 (SFT regression on the control-eligible primary40; anchors A0cap and A0mc) | `analyses/v02-v025/`; `analyses/a0-capped-v052/`; `analyses/matched-context-v066/`; `analyses/mechanism-v031/` | `preregistrations/contracts/`; `preregistrations/freeze-receipts/`; `contracts/` (a0-capped-v052 and matched-context-v066-a0mc) | `code/tools/analyze_contract_v02_v025.py`; `code/tools/analyze_a0_capped_v052.py`; `code/tools/mechanism_metrics_v032.py` |
| K2 (loops and non-submission and repeats; dissociation) | `analyses/mechanism-ci-v031/` and the reports of K1 and K4 to K9 | `contracts/mechanism-ci-v031.execution-plan.json` | `code/tools/analyze_mechanism_v031.py` |
| K3 (teacher targets against packed targets against rollouts) | `analyses/teacher-behaviour-boss-v045/` (tables only); `analyses/mask-control-v043/` | none (descriptive) | `code/audits/teacher-behaviour-boss-v045/` |
| K4 (base alignment; Table 2) | `analyses/first-step-probe-boss-v048/`; `analyses/qat-transfer-v053/`; `analyses/m0q-eval-v054/` | `preregistrations/notes/` (first-step probe and QAT-base fixture); `contracts/` (m0q-eval-v054) | `code/audits/first-step-probe-boss-v048/`; `code/audits/qat-transfer-v053/`; `code/tools/analyze_m0q_eval_v054.py`; `code/tools/dequant_qat_v053.py` |
| K5 (serving-form changes) | `analyses/matched-context-v051/`; `analyses/matched-context-v055/` | `preregistrations/notes/` (matched-context); `contracts/` (matched-context-v051 and v055) | `code/tools/analyze_matched_context_v051.py`; `code/tools/analyze_matched_context_v055.py`; `code/templates/` |
| K6 (masking on bf16) | `analyses/mask-control-v043/` | `preregistrations/notes/prereg-2026-11-05-mask-control-mechanism-predictions.md`; `contracts/` (mask-control-analysis and mask-evaluations) | `code/tools/analyze_mask_control_v044.py`; `code/sft/train_mask_control_v035.py` |
| K7 (history preservation) | `analyses/matched-context-v058/` | `preregistrations/notes/prereg-2026-11-06-history-pack-Hq1-v057.md`; `contracts/` (matched-context-v058) | `code/tools/analyze_matched_context_v058.py`; `code/sft/build_history_pack_v057.py`; `code/sft/check_history_pack_v057.py` |
| K8 (masked arm on the QAT base with two training seeds) | `analyses/matched-context-v062/`; `analyses/matched-context-v066/`; `analyses/m1q-vs-a0mc-v067/`; `analyses/matched-context-v067/` | `preregistrations/notes/prereg-2026-11-07-m1q-mask-on-qat-base-v062.md`; `preregistrations/notes/prereg-2026-11-07-m1q-seed2-and-a0mc-branches-v067.md`; `contracts/` (v062 and v066 and v067) | `code/tools/analyze_matched_context_v062.py`; `code/tools/analyze_m1q_vs_a0mc_v067.py`; `code/sft/train_m1q_v062.py` |
| K9 (self-generated against teacher-subset arm; confirmation seed) | `analyses/matched-context-v064a/`; `analyses/matched-context-v064t/`; `analyses/matched-context-v064c/`; `analyses/onpolicy-cells-v064/`; `analyses/onpolicy-v064/` | `preregistrations/notes/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md`; `preregistrations/notes/prereg-2026-11-08-onpolicy-confirmation-readings-v069.md`; `contracts/` (v064a and v064c and v064t and onpolicy) | `code/tools/analyze_matched_context_v064a.py`; `code/tools/analyze_onpolicy_cells_v064.py`; `code/sft/train_onpolicy_v064r2.py` |
| External-panel paragraph | `analyses/external-e4b-v068/`; `analyses/codex-v070-git-r2/` | `preregistrations/codex-v070-git-r1/`; `preregistrations/codex-v070-git-r2/`; `preregistrations/notes/v068-execution-change-record-2026-11-09.md`; `contracts/codex-v070-git-r2/` | `code/codex-v070-git-r2/`; `code/tools/analyze_external_e4b_v068_r22.py` |
| Table 1 | one analysis per row. A0 and A1 and A2: `analyses/v02-v025/`. M0 and M1: `analyses/mask-control-v043/`. M0mc: `analyses/matched-context-v051/`. A0cap: `analyses/a0-capped-v052/`. M0q: `analyses/m0q-eval-v054/`. M0qmc: `analyses/matched-context-v055/`. Hq1mc: `analyses/matched-context-v058/`. M1qmc: `analyses/matched-context-v062/`. M1qmc-s2: `analyses/matched-context-v067/`. A0mc: `analyses/matched-context-v066/`. A_onpol: `analyses/matched-context-v064a/`. T_sub: `analyses/matched-context-v064t/`. A_onpol-s2: `analyses/matched-context-v064c/` | as above | `verify/verify_paper_numbers.py` (repair column) |
| Table 2 | `analyses/first-step-probe-boss-v048/report.json`; `analyses/qat-transfer-v053/report.json`; `analyses/qat-transfer-v053/m0q/report.json` (score files are not published) | see K4 | see K4 |
| Fig. 1 | `figures/rev3/fig1.png`; `figures/rev3/fig1.svg`; `figures/figure_data.csv`; aggregates in `analyses/sft-audit-v033/full/summary.json` | none | `code/tools/sft_example_audit.py`; `code/audits/serialization-audit-boss-v046/`; `figures/rev3/make_figures_rev3.py` |
| Fig. 2a and 2b | `figures/rev3/fig2a.png`; `figures/rev3/fig2b.png` (and svg); `figures/rev3/fig2.png`; `figures/figure_data.csv` | none | `figures/rev3/make_figures_rev3.py` (reads only `figure_data.csv`) |
| Fig. 3 | not built | | |

## Sealed-file hash prefixes cited in the paper

The paper cites sealed files by the first 8 hex digits of the sha256 of the sealed original (`original_sha256` in `MANIFEST.tsv`). The table lists every cited file of this package with its prefix and the place where the claim register (skeleton v0.18 sections C and G and H and v0.1 section G) lists it. For a file whose change is "paths" the prefix refers to the original and not to the public copy.

| file | prefix and place in the claim register |
|---|---|
| `analyses/a0-capped-v052/report.json` | d5850a7e (v0.18 C) |
| `analyses/codex-v070-git-r2/completed.json` | 93d7994a (v0.18 G) |
| `analyses/codex-v070-git-r2/controls-evidence/control-report.json` | 80259214 (v0.18 C/v0.18 G) |
| `analyses/codex-v070-git-r2/readout/per_task.jsonl` | 211304ac (v0.18 G) |
| `analyses/codex-v070-git-r2/readout/report.json` | a682c9e5 (v0.18 C/v0.18 G) |
| `analyses/codex-v070-git-r2/readout/report.md` | 866624dc (v0.18 G) |
| `analyses/codex-v070-git-r2/terminal.json` | decd66d5 (v0.18 G) |
| `analyses/external-e4b-v068/report.json` | 1c15a47f (v0.18 C/v0.18 G) |
| `analyses/m0q-eval-v054/per_task.jsonl` | 79427bb2 (v0.18 G) |
| `analyses/m0q-eval-v054/report.json` | fbbffc9f (v0.18 C) |
| `analyses/m0q-eval-v054/report.md` | 18cd8692 (v0.18 G) |
| `analyses/m1q-vs-a0mc-v067/report.json` | a62b7f90 (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/m1q-vs-a0mc-v067/report.md` | 634c9c72 (v0.18 C/v0.18 H) |
| `analyses/mask-control-v043/report.json` | 1b8d360b (v0.18 C) |
| `analyses/matched-context-v051/per_task.jsonl` | 24733f02 (v0.18 G) |
| `analyses/matched-context-v051/report.json` | ea0743c9 (v0.18 C) |
| `analyses/matched-context-v051/report.md` | 404edda3 (v0.18 G) |
| `analyses/matched-context-v055/per_task.jsonl` | 682e3492 (v0.18 G) |
| `analyses/matched-context-v055/report.json` | 16399065 (v0.18 C) |
| `analyses/matched-context-v055/report.md` | ffbcda8e (v0.18 G) |
| `analyses/matched-context-v058/per_task.jsonl` | 58bf384f (v0.18 G) |
| `analyses/matched-context-v058/report.json` | 8c564a96 (v0.18 C) |
| `analyses/matched-context-v058/report.md` | b2c84d70 (v0.18 C) |
| `analyses/matched-context-v062/per_task.jsonl` | 2f991e81 (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/matched-context-v062/report.json` | 70f20fc0 (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/matched-context-v062/report.md` | e1da790b (v0.18 C/v0.18 H) |
| `analyses/matched-context-v064a/per_task.jsonl` | 28015851 (v0.18 C/v0.18 G) |
| `analyses/matched-context-v064a/report.json` | 8f32f58b (v0.18 C/v0.18 G) |
| `analyses/matched-context-v064c/per_task.jsonl` | 91c6e001 (v0.18 C/v0.18 G) |
| `analyses/matched-context-v064c/report.json` | 6b275013 (v0.18 C/v0.18 G) |
| `analyses/matched-context-v064t/per_task.jsonl` | 65be6662 (v0.18 C/v0.18 G) |
| `analyses/matched-context-v064t/report.json` | e99fcb88 (v0.18 C/v0.18 G) |
| `analyses/matched-context-v066/per_task.jsonl` | 72fc58d9 (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/matched-context-v066/report.json` | c448ed2e (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/matched-context-v067/per_task.jsonl` | f5f41561 (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/matched-context-v067/report.json` | 10a004d0 (v0.18 C/v0.18 G/v0.18 H) |
| `analyses/matched-context-v067/report.md` | 0165deaa (v0.18 C/v0.18 H) |
| `analyses/mechanism-ci-v031/report.json` | 6df5d7b7 (v0.18 C/v0.18 H) |
| `analyses/mechanism-ci-v031/report.md` | da3317da (v0.18 C/v0.18 H) |
| `analyses/mechanism-v031/primary40_ids.txt` | c7125743 (v0.1 G) |
| `analyses/onpolicy-cells-v064/report.json` | 13251c1c (v0.18 C/v0.18 G) |
| `analyses/oom-controls-v043/final48/report.json` | a3b4548b (v0.1 G) |
| `analyses/oom-replay-v043/report.json` | 536e2c3e (v0.1 G) |
| `analyses/probes-v032/report.md` | 485a234f (v0.1 G) |
| `analyses/repro-definition-audit-v040/results-v2/report.json` | 3c2bd7ed (v0.1 G) |
| `analyses/sft-audit-v033/full/summary.json` | 9420a0ed (v0.1 G/v0.18 D) |
| `analyses/teacher-behaviour-boss-v045/report.json` | ba1f513c (v0.18 C/v0.18 H) |
| `analyses/teacher-behaviour-boss-v045/report.md` | 40d37375 (v0.18 H) |
| `analyses/v02-v025/report.json` | b2a2ebba (v0.1 H/v0.18 C) |
| `code/audits/serialization-audit-boss-v046/audit.py` | 5d26f4cd (v0.1 G) |
| `code/tools/analyze_m1q_vs_a0mc_v067.py` | 911e4ab9 (v0.18 C) |
| `code/tools/analyze_mask_control_v042.py` | 23915fb0 (v0.1 G) |
| `code/tools/analyze_mask_control_v044.py` | 2334f6ad (v0.1 G) |
| `code/tools/analyze_mechanism_v031.py` | 821f2294 (v0.1 G) |
| `code/tools/analyze_onpolicy_cells_v064.py` | 2623bd11 (v0.18 C) |
| `code/tools/dequant_qat_v053.py` | a582efc0 (v0.1 G) |
| `code/tools/mechanism_metrics_v032.py` | 36042f6d (v0.1 G/v0.18 C) |
| `contracts/a0-capped-v052-analysis-DRAFT.json` | ea39c4f4 (v0.18 C) |
| `contracts/codex-v070-git-r2/external-e4b-v070git2-analysis-PINNED.json` | 80a2c182 (v0.18 G) |
| `contracts/codex-v070-git-r2/external-e4b-v070git2-runtime-PINNED.json` | d495a1bc (v0.18 G) |
| `contracts/external-e4b-v068-r22-runtime-PINNED.json` | 97882642 (v0.18 G) |
| `contracts/m0q-eval-v054-analysis-DRAFT.json` | 5fd70d4e (v0.18 C) |
| `contracts/m0q-eval-v054-runtime-DRAFT.json` | 75c82f37 (v0.18 G) |
| `contracts/m1q-vs-a0mc-v067-analysis-DRAFT.json` | 7b434cf0 (v0.18 C) |
| `contracts/matched-context-v055-analysis-DRAFT.json` | faf9298e (v0.18 C) |
| `contracts/matched-context-v058-analysis-PINNED.json` | 997d44da (v0.18 C) |
| `contracts/matched-context-v058-runtime-PINNED.json` | 42baf646 (v0.18 G) |
| `contracts/matched-context-v062-analysis-PINNED.json` | 91295f7c (v0.18 C) |
| `contracts/matched-context-v062-runtime-PINNED.json` | 3717c8d0 (v0.18 C/v0.18 G) |
| `contracts/matched-context-v064a-analysis-PINNED.json` | ae461ea4 (v0.18 C) |
| `contracts/matched-context-v064a-runtime-PINNED.json` | fc8953fe (v0.18 G) |
| `contracts/matched-context-v064c-analysis-PINNED.json` | ad37ac46 (v0.18 C) |
| `contracts/matched-context-v064c-runtime-PINNED.json` | 898b5254 (v0.18 G) |
| `contracts/matched-context-v064t-analysis-PINNED.json` | 20758db0 (v0.18 C) |
| `contracts/matched-context-v064t-runtime-PINNED.json` | eadc98e9 (v0.18 G) |
| `contracts/matched-context-v066-a0mc-analysis-PINNED.json` | a070f299 (v0.18 C) |
| `contracts/matched-context-v066-a0mc-runtime-PINNED.json` | 0f5e04b2 (v0.18 C/v0.18 G) |
| `contracts/matched-context-v067-analysis-PINNED.json` | 97ade321 (v0.18 C) |
| `contracts/matched-context-v067-runtime-PINNED.json` | 1b8e16f2 (v0.18 C/v0.18 G) |
| `preregistrations/codex-v070-git-r1/DECISION-B.json` | 8a94ad30 (v0.18 C/v0.18 G) |
| `preregistrations/codex-v070-git-r1/amendment-FROZEN.md` | 8ca2602b (v0.18 G) |
| `preregistrations/codex-v070-git-r2/amendment-FROZEN.md` | dff250b2 (v0.18 C/v0.18 G) |
| `preregistrations/decisions/2026-11-09-v070-cache-rule-k-disclose.md` | 32f7625a (v0.18 G) |
| `preregistrations/freeze-receipts/contract-v0.2.freeze.json` | a7a301b7 (v0.18 C) |
| `preregistrations/notes/prereg-2026-11-05-capped-A0-and-M1-nothink-probes.md` | cc043e03 (v0.1 H/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-05-mask-control-mechanism-predictions.md` | a24baec7 (v0.1 H/v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-06-first-step-probe-addendum-v2.md` | 8e37c3da (v0.1 H/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-06-first-step-probe.md` | 5305fedb (v0.1 H/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-06-history-pack-Hq1-v057.md` | c90558df (v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-06-matched-context-M0q-v055.md` | 82ce6787 (v0.18 C) |
| `preregistrations/notes/prereg-2026-11-06-matched-context-eval.md` | 86bc2ebb (v0.1 H/v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-06-qat-base-lora-fixture-and-M0q.md` | e11d8b75 (v0.1 H/v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065.md` | b602a5d3 (v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-07-m1q-mask-on-qat-base-v062.md` | 36972c9e (v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-07-m1q-seed2-and-a0mc-branches-v067.md` | 766c1e45 (v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md` | 0893728c (v0.18 C/v0.18 H) |
| `preregistrations/notes/prereg-2026-11-08-external-contrast-onpolicy-v070.md` | b7b4d4be (v0.18 C) |
| `preregistrations/notes/prereg-2026-11-08-onpolicy-confirmation-readings-v069.md` | fabeddb2 (v0.18 C/v0.18 H) |
| `preregistrations/notes/v068-execution-change-record-2026-11-09.md` | 6a0e5942 (v0.18 G) |

Files of the study that carry a cited prefix but are not in this package (adapters and training data and notes) are listed in `NOTICE.md` by class.
