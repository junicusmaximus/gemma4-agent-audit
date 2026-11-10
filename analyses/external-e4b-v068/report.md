# External-repository panel v068 (M1q observation): analysis

Generation: all_v068_evaluations_qualified (stop: None). Eligible tasks n_e = 12, complete tasks n_c = 12. Arms: A0mc, M0q, M1q, M1qs2. M1q-s2 included at the pin.

## Reading

on the external panel the base resolved nothing (0/36), so the panel cannot inform the trained arms (M1q 0, M0q 0)

Cell: z = 0.

Beside it, with x replaced by the M1q-s2 count (never pooled): on the external panel the base resolved nothing (0/36), so the panel cannot inform the trained arms (M1q 0, M0q 0)

## Paired task-bootstrap intervals (20000 resamples, seed 20261103, over the complete tasks)

- M1q - M0q: uncertain (+0.00 pp [+0.00, +0.00])
- M1q - A0mc: uncertain (+0.00 pp [+0.00, +0.00])

## Pooled resolved over the complete tasks

| arm | resolved | of | per seed |
|---|---|---|---|
| A0mc | 0 | 36 | {'1': 0, '2': 0, '3': 0} |
| M0q | 0 | 36 | {'1': 0, '2': 0, '3': 0} |
| M1q | 0 | 36 | {'1': 0, '2': 0, '3': 0} |
| M1qs2 | 0 | 36 | {'1': 0, '2': 0, '3': 0} |

## Rows that are not completed

- not eligible at the pin: apiflask__apiflask-659 (INCOMPLETE_probe), googleapis__python-api-core-197 (INCOMPLETE_probe)
- excluded: {}
- invocations: {"completed": 144, "planned": 144}

## Descriptive (three-seed means over the complete tasks)

| arm | loop | submit | M5 error | think |
|---|---|---|---|---|
| A0mc | 0.3888888888888889 | 0.9444444444444443 | 0.4091955840619661 | 0.7316930915579934 |
| M0q | 0.8333333333333334 | 0.5555555555555555 | 0.37053521409372886 | 0.0 |
| M1q | 0.8888888888888888 | 0.5833333333333334 | 0.5546108376473521 | 0.7555590915843305 |
| M1qs2 | 0.8888888888888888 | 0.4444444444444444 | 0.7322994026757726 | 0.7249122355226746 |

## Canary rows (apart, never in the readings)

- A0mc s1: completed
- M0q s1: completed
- M1q s1: completed
- M1qs2 s1: completed

## Not claimed

- A panel statement only: never a hidden-set estimate; no generalisation beyond these repositories; no mechanism; no pooling across seeds or arms; no recovery, restores or causes wording; with small n_e an absence of repair is not evidence of absence.
- After this read DEV14 is relabelled as a main-track development set (declared exposure); CONFIRM14 stays sealed.
