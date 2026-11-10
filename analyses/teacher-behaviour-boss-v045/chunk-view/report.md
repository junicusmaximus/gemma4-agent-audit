# Chunk view of the supervised targets: M0.jsonl

Groups: first = steps[0] == 0; later = steps[0] > 0; later_orphan = later with a tool response right after the header; later_chain_start = later without it (chain > 0 summarizer or continuation chunk). Details: README.md.

| metric | first | later | all | later_orphan | later_chain_start |
|---|---|---|---|---|---|
| chunks (share of all) | 726 (0.437) | 935 (0.563) | 1661 (1.000) | 897 (0.540) | 38 (0.023) |
| supervised completions (mean per chunk) | 12925 (17.80) | 12274 (13.13) | 25199 (15.17) | 12127 (13.52) | 147 (3.87) |
| supervised tool calls parsed | 12905 | 12238 | 25143 | 12127 | 111 |
| supervised tokens (share of all) | 1541259 (0.351) | 2850390 (0.649) | 4391649 (1.000) | 2809882 (0.640) | 40508 (0.009) |
| mean chunk length, tokens | 15222 | 10557 | 12596 | 10718 | 6747 |
| first supervised action: find_opener | 0.949 (689/726) | 0.000 (0/935) | 0.415 (689/1661) | 0.000 (0/897) | 0.000 (0/38) |
| first supervised action: wrapper_opener | 0.000 (0/726) | 0.442 (413/935) | 0.249 (413/1661) | 0.460 (413/897) | 0.000 (0/38) |
| first supervised action: repro_script_opener | 0.000 (0/726) | 0.065 (61/935) | 0.037 (61/1661) | 0.068 (61/897) | 0.000 (0/38) |
| first supervised action: other | 0.051 (37/726) | 0.471 (440/935) | 0.287 (477/1661) | 0.472 (423/897) | 0.447 (17/38) |
| first supervised action: no_tool_call | 0.000 (0/726) | 0.022 (21/935) | 0.013 (21/1661) | 0.000 (0/897) | 0.553 (21/38) |
| loop in chunk (max multiplicity >= 5) | 0.001 (1/726) | 0.002 (2/935) | 0.002 (3/1661) | 0.001 (1/897) | 0.026 (1/38) |
| max multiplicity mean / median | 1.42 / 1.0 | 1.54 / 1.0 | 1.49 / 1.0 | 1.57 / 1.0 | 0.89 / 0.0 |
| distinct-call ratio, mean over chunks with a call | 0.972 (n=726) | 0.958 (n=914) | 0.964 (n=1640) | 0.959 (n=897) | 0.897 (n=17) |
| chunk has a submit_patch call (extra row) | 0.050 (36/726) | 0.739 (691/935) | 0.438 (727/1661) | 0.753 (675/897) | 0.421 (16/38) |
| M5 repeat after error: mean (pooled; chunks with >= 1 eligible, eligible transitions) | 0.002 (pooled 0.011; chunks 435/726, elig 785) | 0.002 (pooled 0.009; chunks 433/935, elig 764) | 0.002 (pooled 0.010; chunks 868/1661, elig 1549) | 0.000 (pooled 0.000; chunks 422/897, elig 737) | 0.071 (pooled 0.259; chunks 11/38, elig 27) |
| M5 repeat after silent: mean (pooled; chunks with >= 1 eligible, eligible transitions) | 0.000 (pooled 0.000; chunks 726/726, elig 4991) | 0.000 (pooled 0.000; chunks 835/935, elig 4342) | 0.000 (pooled 0.000; chunks 1561/1661, elig 9333) | 0.000 (pooled 0.000; chunks 824/897, elig 4295) | 0.000 (pooled 0.000; chunks 11/38, elig 47) |
| M5 repeat after normal: mean (pooled; chunks with >= 1 eligible, eligible transitions) | 0.000 (pooled 0.000; chunks 724/726, elig 6403) | 0.000 (pooled 0.000; chunks 856/935, elig 6218) | 0.000 (pooled 0.000; chunks 1580/1661, elig 12621) | 0.000 (pooled 0.000; chunks 848/897, elig 6198) | 0.000 (pooled 0.000; chunks 8/38, elig 20) |
| thought present, pooled over completions | 0.041 (534/12925) | 0.012 (151/12274) | 0.027 (685/25199) | 0.000 (5/12127) | 0.993 (146/147) |
| thought present, per-chunk mean | 0.053 | 0.042 | 0.047 | 0.003 | 0.974 |
| history-free actions (share of ALL supervised calls) | 0 (0.0000) | 897 (0.0357) | 897 (0.0357) | 897 (0.0357) | 0 (0.0000) |
| history-free actions (share of the group's own calls) | 0.000 (0/12905) | 0.073 (897/12238) | 0.036 (897/25143) | 0.074 (897/12127) | 0.000 (0/111) |

Type check: first/direct 726, later/direct 19, later/orphan_response 897, later/replayed_history 19.
Mismatches: first chunks with an orphan response 0; later chunks without one 38.
Round trip: header of e1b0af5fbbb3eab6341a2b49e equal True (first differing offset None); 1661 of 1661 chunks decode to the stored segments.
Malformed chunks 0, unreadable lines 0, trajectories missing from the source 0.
