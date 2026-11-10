# External-repository panel v070 (termination contrast): analysis

Generation: all_v070_evaluations_qualified (stop: None). Eligible tasks n_e = 12, complete tasks n_c = 12, completed rollouts 108 of 108. Arms: A_onpol_ts1 (A_onpol, training seed 1), A_onpol_ts2 (A_onpol, training seed 2), T_sub (T_sub, training seed 1).

## Reading

the termination contrast was also observed on an 12-task external panel (A_onpol training seeds 1 and 2: loop 0.000 and 0.000, submit 1.000 and 1.000; T_sub, one training seed: loop 0.861, submit 0.083)

Cell E1 applies: the planned paired set is complete and the report is sealed before the result freeze (2026-11-10T09:00:00Z).

## Termination over the complete tasks (reading)

| arm | family | training seed | loop | submit attempt | label | think | M5 error |
|---|---|---|---|---|---|---|---|
| A_onpol_ts1 | A_onpol | 1 | 0.000 | 1.000 | NON-COLLAPSED | 0.912 | 0.037 |
| A_onpol_ts2 | A_onpol | 2 | 0.000 | 1.000 | NON-COLLAPSED | 0.901 | 0.016 |
| T_sub | T_sub | 1 | 0.861 | 0.083 | COLLAPSED | 0.000 | 0.759 |

## Repair, always reported (pooled resolved over the complete tasks, per decoding seed)

| arm | training seed | resolved | of | decoding seeds 1, 2, 3 |
|---|---|---|---|---|
| A_onpol_ts1 | 1 | 6 | 36 | 4, 1, 1 |
| A_onpol_ts2 | 2 | 7 | 36 | 3, 2, 2 |
| T_sub | 1 | 0 | 36 | 0, 0, 0 |

**A0mc-ext is withheld: v068 had an invalid submit path; no reused comparator rows or confidence intervals are reported.**

No A0mc-ext comparator is shown: v068 submit path was infrastructure-invalid; withheld by the amendment.

## Cache state (declared difference (k))

Rule fixed in the analysis manifest: k-disclose (cache state is not an identity leaf: the cache files that differ between the starting cache list of v070 and that of v068 attempt 2 are disclosed as declared difference (k) and the comparator follows the frozen identity leaves only).

(k) v070 starts with the bytecode caches present after v068 attempt 2, so its starting cache list can differ from the starting list of attempt 2. Python reads .pyc files: a module whose cache changed may load from the cache instead of being compiled from source. Leaving cache files out of the dependency-tree hash is not a claim that caches cannot affect execution.

Starting cache lists of v070 and of v068 attempt 2 (path, sha256): equal. Added 0, removed 0, changed 0; same content with another modification time 0.

## Pin record

- v068 contract (r21, identity leaves): 8450ed230af0593fce6d38e41f3b6d0ca5856892cf0ed4e672af1a61e7d31eb8
- v068 attempt-2 contract (r22): 9788264284a296e36ec07da7726259db1aae87759a487fd3b889a90ce80f3db2
- comparator identity: withheld by amendment; identity mismatches: none
- A_onpol_ts1: adapter ad0910f5faa92256ace68230cd8e0f7a1ebe7bf209ab5b95f66140a25ca547d5
- A_onpol_ts2: adapter f9e0664295ea7cf6a60f964330726ad2d6c0f220160d64aef1abb01c867107a3
- T_sub: adapter 7122a9d67cd0dc5dab5cd3c240db88084031177e2b6422d002a2e47c71d882de
- A_onpol_ts1: 0 of 1 training repositories are panel repositories
- A_onpol_ts2: 0 of 1 training repositories are panel repositories
- T_sub: 0 of 18 training repositories are panel repositories

## Rows that are not completed

- not eligible at the pin: apiflask__apiflask-659 (INCOMPLETE_probe), googleapis__python-api-core-197 (INCOMPLETE_probe)
- excluded: {}
- invocations: {"completed": 108, "planned": 108}

## Canary rows (apart, never in the readings)

- A_onpol_ts1 decoding seed 1: completed
- A_onpol_ts2 decoding seed 1: completed
- T_sub decoding seed 1: completed

## Feasibility projection logged after the canaries (descriptive)

{"canary_mean_seconds": 70.85985841571043, "canary_wall_seconds": [37.551513290032744, 95.8843720946461, 79.14368986245245], "cap_seconds": 10800, "descriptive_only": true, "panel_invocations": 108, "projected_fraction_of_cap": 0.1971841088124482, "projected_panel_seconds": 1913.2161772241818, "projected_seconds_from_ready": 2129.5883751744404, "projected_within_cap": true, "ready_elapsed_seconds": 216.37219795025885, "rule": "ceil(panel invocations / workers) x mean canary wall time, added to the time since server-ready; no effect on launch or stops", "staging_seconds": 8.263857431709766, "startup_seconds": 54.04562735185027, "workers": 4}

## Limitations

- Execution order: v070 ran after v068 attempts 1 and 2 on the same environment trees.
- Cache accumulation: bytecode caches written by the earlier attempts were present at v070's start; their differences from attempt 2's start are listed under (k).

## Not claimed

- A panel statement only: no hidden-set estimate, no generalisation beyond these repositories, no mechanism, no pooling across seeds or arms; with small n_c an absence of contrast or repair is not evidence of absence. A termination cell is a statement about termination behaviour only and never implies repair transfer.
