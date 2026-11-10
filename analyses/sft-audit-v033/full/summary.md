# SFT example audit

Trajectories audited: 1670 (limit: None). Definitions: module docstring of sft_example_audit.py. Input hashes: inputs.sha256.

## Arms

| arm key | status | ids listed | trajectories matched |
|---|---|---|---|
| arm_ordinary | ok | 764 | 764 |
| arm_short | ok | 764 | 764 |
| arm_ordinary_tokmatched | ok | 726 | 726 |
| all_resolved | ok | 1670 | 1670 |
| arm_ordinary_multi | ok | 492 | 492 |
| arm_short_multi | ok | 492 | 492 |
| arm_short_calls | ok | 492 | 492 |

## Population

| group | trajectories | chunks | runs | loss tokens |
|---|---|---|---|---|
| arm_ordinary | 764 | 1751 | 26553 | 4619779 |
| arm_ordinary / external_A | 12 | 21 | 325 | 64739 |
| arm_ordinary / external_B | 449 | 1045 | 15953 | 2747398 |
| arm_ordinary / external_B2 | 265 | 605 | 9598 | 1614384 |
| arm_ordinary / native | 38 | 80 | 677 | 193258 |
| arm_short | 764 | 1660 | 26081 | 4426681 |
| arm_short / external_A | 10 | 18 | 269 | 54252 |
| arm_short / external_B | 508 | 1151 | 17936 | 3030883 |
| arm_short / external_B2 | 196 | 407 | 7058 | 1127327 |
| arm_short / native | 50 | 84 | 818 | 214219 |
| arm_ordinary_tokmatched | 726 | 1661 | 25199 | 4391649 |
| arm_ordinary_tokmatched / external_A | 12 | 21 | 325 | 64739 |
| arm_ordinary_tokmatched / external_B | 426 | 991 | 15129 | 2606988 |
| arm_ordinary_tokmatched / external_B2 | 251 | 572 | 9085 | 1532756 |
| arm_ordinary_tokmatched / native | 37 | 77 | 660 | 187166 |
| all_resolved | 1670 | 3694 | 56702 | 9801140 |
| all_resolved / external_A | 14 | 25 | 382 | 75358 |
| all_resolved / external_B | 729 | 1638 | 25165 | 4302299 |
| all_resolved / external_B2 | 824 | 1835 | 29389 | 4904470 |
| all_resolved / native | 103 | 196 | 1766 | 519013 |
| arm_ordinary_multi | 492 | 1087 | 16594 | 2855064 |
| arm_ordinary_multi / external_A | 2 | 3 | 56 | 10487 |
| arm_ordinary_multi / external_B | 193 | 420 | 6394 | 1081810 |
| arm_ordinary_multi / external_B2 | 265 | 605 | 9598 | 1614384 |
| arm_ordinary_multi / native | 32 | 59 | 546 | 148383 |
| arm_short_multi | 492 | 996 | 16122 | 2661966 |
| arm_short_multi / external_B | 252 | 526 | 8377 | 1365295 |
| arm_short_multi / external_B2 | 196 | 407 | 7058 | 1127327 |
| arm_short_multi / native | 44 | 63 | 687 | 169344 |
| arm_short_calls | 492 | 1037 | 15516 | 2709515 |
| arm_short_calls / external_B | 421 | 906 | 14008 | 2359985 |
| arm_short_calls / external_B2 | 22 | 49 | 785 | 131886 |
| arm_short_calls / native | 49 | 82 | 723 | 217644 |

## History-dropped chunks and orphan tool responses

| group | history-dropped chunks | history-dropped loss tokens | orphan chunks (of history-dropped) |
|---|---|---|---|
| arm_ordinary | 947 (54.08%) | 2972212 (64.34%) | 947 (100.00%) |
| arm_ordinary / external_A | 9 (42.86%) | 12759 (19.71%) | 9 (100.00%) |
| arm_ordinary / external_B | 596 (57.03%) | 1898995 (69.12%) | 596 (100.00%) |
| arm_ordinary / external_B2 | 340 (56.20%) | 1055412 (65.38%) | 340 (100.00%) |
| arm_ordinary / native | 2 (2.50%) | 5046 (2.61%) | 2 (100.00%) |
| arm_short | 864 (52.05%) | 2651318 (59.89%) | 864 (100.00%) |
| arm_short / external_A | 8 (44.44%) | 12290 (22.65%) | 8 (100.00%) |
| arm_short / external_B | 643 (55.86%) | 1988399 (65.60%) | 643 (100.00%) |
| arm_short / external_B2 | 211 (51.84%) | 645583 (57.27%) | 211 (100.00%) |
| arm_short / native | 2 (2.38%) | 5046 (2.36%) | 2 (100.00%) |
| arm_ordinary_tokmatched | 897 (54.00%) | 2809882 (63.98%) | 897 (100.00%) |
| arm_ordinary_tokmatched / external_A | 9 (42.86%) | 12759 (19.71%) | 9 (100.00%) |
| arm_ordinary_tokmatched / external_B | 565 (57.01%) | 1794678 (68.84%) | 565 (100.00%) |
| arm_ordinary_tokmatched / external_B2 | 321 (56.12%) | 997399 (65.07%) | 321 (100.00%) |
| arm_ordinary_tokmatched / native | 2 (2.60%) | 5046 (2.70%) | 2 (100.00%) |
| all_resolved | 1934 (52.36%) | 5959676 (60.81%) | 1934 (100.00%) |
| all_resolved / external_A | 11 (44.00%) | 17256 (22.90%) | 11 (100.00%) |
| all_resolved / external_B | 909 (55.49%) | 2806798 (65.24%) | 909 (100.00%) |
| all_resolved / external_B2 | 1011 (55.10%) | 3123605 (63.69%) | 1011 (100.00%) |
| all_resolved / native | 3 (1.53%) | 12017 (2.32%) | 3 (100.00%) |
| arm_ordinary_multi | 569 (52.35%) | 1751956 (61.36%) | 569 (100.00%) |
| arm_ordinary_multi / external_A | 1 (33.33%) | 469 (4.47%) | 1 (100.00%) |
| arm_ordinary_multi / external_B | 227 (54.05%) | 691383 (63.91%) | 227 (100.00%) |
| arm_ordinary_multi / external_B2 | 340 (56.20%) | 1055412 (65.38%) | 340 (100.00%) |
| arm_ordinary_multi / native | 1 (1.69%) | 4692 (3.16%) | 1 (100.00%) |
| arm_short_multi | 486 (48.80%) | 1431062 (53.76%) | 486 (100.00%) |
| arm_short_multi / external_B | 274 (52.09%) | 780787 (57.19%) | 274 (100.00%) |
| arm_short_multi / external_B2 | 211 (51.84%) | 645583 (57.27%) | 211 (100.00%) |
| arm_short_multi / native | 1 (1.59%) | 4692 (2.77%) | 1 (100.00%) |
| arm_short_calls | 513 (49.47%) | 1541367 (56.89%) | 513 (100.00%) |
| arm_short_calls / external_B | 485 (53.53%) | 1442853 (61.14%) | 485 (100.00%) |
| arm_short_calls / external_B2 | 27 (55.10%) | 91543 (69.41%) | 27 (100.00%) |
| arm_short_calls / native | 1 (1.22%) | 6971 (3.20%) | 1 (100.00%) |

## Empty thought blocks

| group | empty-thought runs | empty-thought loss tokens | thought tokens |
|---|---|---|---|
| arm_ordinary | 25851 (97.36%) | 4409303 (95.44%) | 137975 |
| arm_ordinary / external_A | 299 (92.00%) | 47159 (72.84%) | 9027 |
| arm_ordinary / external_B | 15953 (100.00%) | 2747398 (100.00%) | 0 |
| arm_ordinary / external_B2 | 9598 (100.00%) | 1614384 (100.00%) | 0 |
| arm_ordinary / native | 1 (0.15%) | 362 (0.19%) | 128948 |
| arm_short | 25243 (96.79%) | 4197589 (94.82%) | 149336 |
| arm_short / external_A | 248 (92.19%) | 39017 (71.92%) | 7579 |
| arm_short / external_B | 17936 (100.00%) | 3030883 (100.00%) | 0 |
| arm_short / external_B2 | 7058 (100.00%) | 1127327 (100.00%) | 0 |
| arm_short / native | 1 (0.12%) | 362 (0.17%) | 141757 |
| arm_ordinary_tokmatched | 24514 (97.28%) | 4187265 (95.35%) | 133269 |
| arm_ordinary_tokmatched / external_A | 299 (92.00%) | 47159 (72.84%) | 9027 |
| arm_ordinary_tokmatched / external_B | 15129 (100.00%) | 2606988 (100.00%) | 0 |
| arm_ordinary_tokmatched / external_B2 | 9085 (100.00%) | 1532756 (100.00%) | 0 |
| arm_ordinary_tokmatched / native | 1 (0.15%) | 362 (0.19%) | 124242 |
| all_resolved | 54907 (96.83%) | 9261791 (94.50%) | 365538 |
| all_resolved / external_A | 352 (92.15%) | 54660 (72.53%) | 9892 |
| all_resolved / external_B | 25165 (100.00%) | 4302299 (100.00%) | 0 |
| all_resolved / external_B2 | 29389 (100.00%) | 4904470 (100.00%) | 0 |
| all_resolved / native | 1 (0.06%) | 362 (0.07%) | 355646 |
| arm_ordinary_multi | 16044 (96.69%) | 2704698 (94.73%) | 97754 |
| arm_ordinary_multi / external_A | 51 (91.07%) | 8142 (77.64%) | 1448 |
| arm_ordinary_multi / external_B | 6394 (100.00%) | 1081810 (100.00%) | 0 |
| arm_ordinary_multi / external_B2 | 9598 (100.00%) | 1614384 (100.00%) | 0 |
| arm_ordinary_multi / native | 1 (0.18%) | 362 (0.24%) | 96306 |
| arm_short_multi | 15436 (95.74%) | 2492984 (93.65%) | 109115 |
| arm_short_multi / external_B | 8377 (100.00%) | 1365295 (100.00%) | 0 |
| arm_short_multi / external_B2 | 7058 (100.00%) | 1127327 (100.00%) | 0 |
| arm_short_multi / native | 1 (0.15%) | 362 (0.21%) | 109115 |
| arm_short_calls | 14793 (95.34%) | 2491871 (91.97%) | 149111 |
| arm_short_calls / external_B | 14008 (100.00%) | 2359985 (100.00%) | 0 |
| arm_short_calls / external_B2 | 785 (100.00%) | 131886 (100.00%) | 0 |
| arm_short_calls / native | 0 (0.00%) | 0 (0.00%) | 149111 |

## Loss-token categories (share of loss tokens)

| group | channel_markers | thought | tool_call | tool_response_opener | turn_end | other |
|---|---|---|---|---|---|---|
| arm_ordinary | 0.63% | 2.99% | 95.53% | 0.57% | 0.00% | 0.29% |
| arm_ordinary / external_A | 0.56% | 13.94% | 85.00% | 0.50% | 0.00% | 0.00% |
| arm_ordinary / external_B | 0.63% | 0.00% | 98.79% | 0.58% | 0.00% | 0.00% |
| arm_ordinary / external_B2 | 0.64% | 0.00% | 98.76% | 0.59% | 0.00% | 0.00% |
| arm_ordinary / native | 0.44% | 66.72% | 25.61% | 0.32% | 0.03% | 6.88% |
| arm_short | 0.64% | 3.37% | 95.06% | 0.59% | 0.00% | 0.34% |
| arm_short / external_A | 0.55% | 13.97% | 84.98% | 0.50% | 0.00% | 0.00% |
| arm_short / external_B | 0.64% | 0.00% | 98.77% | 0.59% | 0.00% | 0.00% |
| arm_short / external_B2 | 0.68% | 0.00% | 98.70% | 0.63% | 0.00% | 0.00% |
| arm_short / native | 0.47% | 66.17% | 26.02% | 0.35% | 0.03% | 6.95% |
| arm_ordinary_tokmatched | 0.62% | 3.03% | 95.47% | 0.57% | 0.00% | 0.30% |
| arm_ordinary_tokmatched / external_A | 0.56% | 13.94% | 85.00% | 0.50% | 0.00% | 0.00% |
| arm_ordinary_tokmatched / external_B | 0.63% | 0.00% | 98.79% | 0.58% | 0.00% | 0.00% |
| arm_ordinary_tokmatched / external_B2 | 0.64% | 0.00% | 98.77% | 0.59% | 0.00% | 0.00% |
| arm_ordinary_tokmatched / native | 0.44% | 66.38% | 25.87% | 0.32% | 0.03% | 6.95% |
| all_resolved | 0.63% | 3.73% | 94.71% | 0.58% | 0.00% | 0.35% |
| all_resolved / external_A | 0.56% | 13.13% | 85.80% | 0.51% | 0.00% | 0.00% |
| all_resolved / external_B | 0.64% | 0.00% | 98.78% | 0.58% | 0.00% | 0.00% |
| all_resolved / external_B2 | 0.65% | 0.00% | 98.75% | 0.60% | 0.00% | 0.00% |
| all_resolved / native | 0.43% | 68.52% | 24.13% | 0.31% | 0.03% | 6.59% |
| arm_ordinary_multi | 0.63% | 3.42% | 95.02% | 0.58% | 0.00% | 0.34% |
| arm_ordinary_multi / external_A | 0.59% | 13.81% | 85.07% | 0.53% | 0.00% | 0.00% |
| arm_ordinary_multi / external_B | 0.64% | 0.00% | 98.76% | 0.59% | 0.00% | 0.00% |
| arm_ordinary_multi / external_B2 | 0.64% | 0.00% | 98.76% | 0.59% | 0.00% | 0.00% |
| arm_ordinary_multi / native | 0.46% | 64.90% | 27.64% | 0.34% | 0.03% | 6.63% |
| arm_short_multi | 0.66% | 4.10% | 94.20% | 0.60% | 0.00% | 0.43% |
| arm_short_multi / external_B | 0.67% | 0.00% | 98.72% | 0.61% | 0.00% | 0.00% |
| arm_short_multi / external_B2 | 0.68% | 0.00% | 98.70% | 0.63% | 0.00% | 0.00% |
| arm_short_multi / native | 0.50% | 64.43% | 27.91% | 0.37% | 0.03% | 6.75% |
| arm_short_calls | 0.63% | 5.50% | 92.77% | 0.57% | 0.00% | 0.52% |
| arm_short_calls / external_B | 0.65% | 0.00% | 98.76% | 0.59% | 0.00% | 0.00% |
| arm_short_calls / external_B2 | 0.65% | 0.00% | 98.76% | 0.60% | 0.00% | 0.00% |
| arm_short_calls / native | 0.42% | 68.51% | 24.25% | 0.30% | 0.03% | 6.48% |

## Text flags (share of runs with known flags)

| group | wrapper | canonical opener | submit_patch | runs with known flags |
|---|---|---|---|---|
| arm_ordinary | 14096 (53.09%) | 1126 (4.24%) | 777 (2.93%) | 26553 |
| arm_ordinary / external_A | 172 (52.92%) | 18 (5.54%) | 12 (3.69%) | 325 |
| arm_ordinary / external_B | 8672 (54.36%) | 688 (4.31%) | 449 (2.81%) | 15953 |
| arm_ordinary / external_B2 | 5252 (54.72%) | 420 (4.38%) | 265 (2.76%) | 9598 |
| arm_ordinary / native | 0 (0.00%) | 0 (0.00%) | 51 (7.53%) | 677 |
| arm_short | 13793 (52.89%) | 1096 (4.20%) | 783 (3.00%) | 26081 |
| arm_short / external_A | 140 (52.04%) | 16 (5.95%) | 10 (3.72%) | 269 |
| arm_short / external_B | 9770 (54.47%) | 773 (4.31%) | 508 (2.83%) | 17936 |
| arm_short / external_B2 | 3883 (55.02%) | 307 (4.35%) | 196 (2.78%) | 7058 |
| arm_short / native | 0 (0.00%) | 0 (0.00%) | 69 (8.44%) | 818 |
| arm_ordinary_tokmatched | 13374 (53.07%) | 1068 (4.24%) | 739 (2.93%) | 25199 |
| arm_ordinary_tokmatched / external_A | 172 (52.92%) | 18 (5.54%) | 12 (3.69%) | 325 |
| arm_ordinary_tokmatched / external_B | 8226 (54.37%) | 654 (4.32%) | 426 (2.82%) | 15129 |
| arm_ordinary_tokmatched / external_B2 | 4976 (54.77%) | 396 (4.36%) | 251 (2.76%) | 9085 |
| arm_ordinary_tokmatched / native | 0 (0.00%) | 0 (0.00%) | 50 (7.58%) | 660 |
| all_resolved | 29799 (52.55%) | 2471 (4.36%) | 1717 (3.03%) | 56702 |
| all_resolved / external_A | 198 (51.83%) | 21 (5.50%) | 14 (3.66%) | 382 |
| all_resolved / external_B | 13620 (54.12%) | 1140 (4.53%) | 729 (2.90%) | 25165 |
| all_resolved / external_B2 | 15981 (54.38%) | 1310 (4.46%) | 824 (2.80%) | 29389 |
| all_resolved / native | 0 (0.00%) | 0 (0.00%) | 150 (8.49%) | 1766 |
| arm_ordinary_multi | 8763 (52.81%) | 724 (4.36%) | 505 (3.04%) | 16594 |
| arm_ordinary_multi / external_A | 32 (57.14%) | 2 (3.57%) | 2 (3.57%) | 56 |
| arm_ordinary_multi / external_B | 3479 (54.41%) | 302 (4.72%) | 193 (3.02%) | 6394 |
| arm_ordinary_multi / external_B2 | 5252 (54.72%) | 420 (4.38%) | 265 (2.76%) | 9598 |
| arm_ordinary_multi / native | 0 (0.00%) | 0 (0.00%) | 45 (8.24%) | 546 |
| arm_short_multi | 8460 (52.47%) | 694 (4.30%) | 511 (3.17%) | 16122 |
| arm_short_multi / external_B | 4577 (54.64%) | 387 (4.62%) | 252 (3.01%) | 8377 |
| arm_short_multi / external_B2 | 3883 (55.02%) | 307 (4.35%) | 196 (2.78%) | 7058 |
| arm_short_multi / native | 0 (0.00%) | 0 (0.00%) | 63 (9.17%) | 687 |
| arm_short_calls | 8030 (51.75%) | 702 (4.52%) | 513 (3.31%) | 15516 |
| arm_short_calls / external_B | 7594 (54.21%) | 664 (4.74%) | 421 (3.01%) | 14008 |
| arm_short_calls / external_B2 | 436 (55.54%) | 38 (4.84%) | 22 (2.80%) | 785 |
| arm_short_calls / native | 0 (0.00%) | 0 (0.00%) | 70 (9.68%) | 723 |

## Loss position and integrity counters

| group | loss pos p50 / p90 / max | dropped items | alignment-uncertain chunks | category-sum violations | loss-count mismatches |
|---|---|---|---|---|---|
| arm_ordinary | 8773 / 14617 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary / external_A | 9125 / 14115 / 16316 | 0 | 0 | 0 | 0 |
| arm_ordinary / external_B | 8864 / 14665 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary / external_B2 | 8646 / 14662 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary / native | 8567 / 13373 / 16243 | 0 | 0 | 0 | 0 |
| arm_short | 8650 / 14622 / 16383 | 0 | 0 | 0 | 0 |
| arm_short / external_A | 8696 / 13942 / 16314 | 0 | 0 | 0 | 0 |
| arm_short / external_B | 8727 / 14684 / 16383 | 0 | 0 | 0 | 0 |
| arm_short / external_B2 | 8617 / 14675 / 16383 | 0 | 0 | 0 | 0 |
| arm_short / native | 8021 / 12930 / 16191 | 0 | 0 | 0 | 0 |
| arm_ordinary_tokmatched | 8752 / 14618 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary_tokmatched / external_A | 9125 / 14115 / 16316 | 0 | 0 | 0 | 0 |
| arm_ordinary_tokmatched / external_B | 8838 / 14672 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary_tokmatched / external_B2 | 8625 / 14655 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary_tokmatched / native | 8575 / 13427 / 16243 | 0 | 0 | 0 | 0 |
| all_resolved | 8747 / 14610 / 16383 | 0 | 0 | 0 | 0 |
| all_resolved / external_A | 8971 / 14164 / 16316 | 0 | 0 | 0 | 0 |
| all_resolved / external_B | 8824 / 14659 / 16383 | 0 | 0 | 0 | 0 |
| all_resolved / external_B2 | 8756 / 14680 / 16383 | 0 | 0 | 0 | 0 |
| all_resolved / native | 8239 / 13263 / 16243 | 0 | 0 | 0 | 0 |
| arm_ordinary_multi | 8762 / 14604 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary_multi / external_A | 10704 / 15491 / 16316 | 0 | 0 | 0 | 0 |
| arm_ordinary_multi / external_B | 9052 / 14673 / 16382 | 0 | 0 | 0 | 0 |
| arm_ordinary_multi / external_B2 | 8646 / 14662 / 16383 | 0 | 0 | 0 | 0 |
| arm_ordinary_multi / native | 8242 / 12954 / 16243 | 0 | 0 | 0 | 0 |
| arm_short_multi | 8556 / 14612 / 16383 | 0 | 0 | 0 | 0 |
| arm_short_multi / external_B | 8704 / 14713 / 16383 | 0 | 0 | 0 | 0 |
| arm_short_multi / external_B2 | 8617 / 14675 / 16383 | 0 | 0 | 0 | 0 |
| arm_short_multi / native | 7542 / 12368 / 15434 | 0 | 0 | 0 | 0 |
| arm_short_calls | 8744 / 14595 / 16383 | 0 | 0 | 0 | 0 |
| arm_short_calls / external_B | 8862 / 14678 / 16383 | 0 | 0 | 0 | 0 |
| arm_short_calls / external_B2 | 8475 / 14802 / 16373 | 0 | 0 | 0 | 0 |
| arm_short_calls / native | 8050 / 12935 / 16243 | 0 | 0 | 0 | 0 |

## Joint table

Cells without runs are omitted here; summary.json lists all eight per group.

| group | history_dropped | empty_thought | wrapper or canonical_opener | runs | loss tokens |
|---|---|---|---|---|---|
| arm_ordinary | False | False | False | 686 | 199541 |
| arm_ordinary | False | False | True | 11 | 3890 |
| arm_ordinary | False | True | False | 5813 | 813395 |
| arm_ordinary | False | True | True | 7198 | 630741 |
| arm_ordinary | True | False | False | 4 | 6653 |
| arm_ordinary | True | False | True | 1 | 392 |
| arm_ordinary | True | True | False | 4828 | 2111741 |
| arm_ordinary | True | True | True | 8012 | 853426 |
| arm_ordinary / external_A | False | False | False | 13 | 11691 |
| arm_ordinary / external_A | False | False | True | 11 | 3890 |
| arm_ordinary / external_A | False | True | False | 98 | 23725 |
| arm_ordinary / external_A | False | True | True | 144 | 12674 |
| arm_ordinary / external_A | True | False | False | 1 | 1607 |
| arm_ordinary / external_A | True | False | True | 1 | 392 |
| arm_ordinary / external_A | True | True | False | 23 | 6802 |
| arm_ordinary / external_A | True | True | True | 34 | 3958 |
| arm_ordinary / external_B | False | True | False | 3529 | 466931 |
| arm_ordinary / external_B | False | True | True | 4300 | 381472 |
| arm_ordinary / external_B | True | True | False | 3064 | 1359540 |
| arm_ordinary / external_B | True | True | True | 5060 | 539455 |
| arm_ordinary / external_B2 | False | True | False | 2185 | 322377 |
| arm_ordinary / external_B2 | False | True | True | 2754 | 236595 |
| arm_ordinary / external_B2 | True | True | False | 1741 | 745399 |
| arm_ordinary / external_B2 | True | True | True | 2918 | 310013 |
| arm_ordinary / native | False | False | False | 673 | 187850 |
| arm_ordinary / native | False | True | False | 1 | 362 |
| arm_ordinary / native | True | False | False | 3 | 5046 |
| arm_short | False | False | False | 825 | 219177 |
| arm_short | False | False | True | 8 | 2870 |
| arm_short | False | True | False | 5986 | 898822 |
| arm_short | False | True | True | 7481 | 654494 |
| arm_short | True | False | False | 4 | 6653 |
| arm_short | True | False | True | 1 | 392 |
| arm_short | True | True | False | 4377 | 1862489 |
| arm_short | True | True | True | 7399 | 781784 |
| arm_short / external_A | False | False | False | 11 | 10366 |
| arm_short / external_A | False | False | True | 8 | 2870 |
| arm_short / external_A | False | True | False | 79 | 18313 |
| arm_short / external_A | False | True | True | 118 | 10413 |
| arm_short / external_A | True | False | False | 1 | 1607 |
| arm_short / external_A | True | False | True | 1 | 392 |
| arm_short / external_A | True | True | False | 22 | 6792 |
| arm_short / external_A | True | True | True | 29 | 3499 |
| arm_short / external_B | False | True | False | 4162 | 596158 |
| arm_short / external_B | False | True | True | 5101 | 446326 |
| arm_short / external_B | True | True | False | 3231 | 1410258 |
| arm_short / external_B | True | True | True | 5442 | 578141 |
| arm_short / external_B2 | False | True | False | 1744 | 283989 |
| arm_short / external_B2 | False | True | True | 2262 | 197755 |
| arm_short / external_B2 | True | True | False | 1124 | 445439 |
| arm_short / external_B2 | True | True | True | 1928 | 200144 |
| arm_short / native | False | False | False | 814 | 208811 |
| arm_short / native | False | True | False | 1 | 362 |
| arm_short / native | True | False | False | 3 | 5046 |
| arm_ordinary_tokmatched | False | False | False | 669 | 193449 |
| arm_ordinary_tokmatched | False | False | True | 11 | 3890 |
| arm_ordinary_tokmatched | False | True | False | 5533 | 782271 |
| arm_ordinary_tokmatched | False | True | True | 6859 | 602157 |
| arm_ordinary_tokmatched | True | False | False | 4 | 6653 |
| arm_ordinary_tokmatched | True | False | True | 1 | 392 |
| arm_ordinary_tokmatched | True | True | False | 4551 | 1992879 |
| arm_ordinary_tokmatched | True | True | True | 7571 | 809958 |
| arm_ordinary_tokmatched / external_A | False | False | False | 13 | 11691 |
| arm_ordinary_tokmatched / external_A | False | False | True | 11 | 3890 |
| arm_ordinary_tokmatched / external_A | False | True | False | 98 | 23725 |
| arm_ordinary_tokmatched / external_A | False | True | True | 144 | 12674 |
| arm_ordinary_tokmatched / external_A | True | False | False | 1 | 1607 |
| arm_ordinary_tokmatched / external_A | True | False | True | 1 | 392 |
| arm_ordinary_tokmatched / external_A | True | True | False | 23 | 6802 |
| arm_ordinary_tokmatched / external_A | True | True | True | 34 | 3958 |
| arm_ordinary_tokmatched / external_B | False | True | False | 3359 | 447430 |
| arm_ordinary_tokmatched / external_B | False | True | True | 4102 | 364880 |
| arm_ordinary_tokmatched / external_B | True | True | False | 2890 | 1282821 |
| arm_ordinary_tokmatched / external_B | True | True | True | 4778 | 511857 |
| arm_ordinary_tokmatched / external_B2 | False | True | False | 2075 | 310754 |
| arm_ordinary_tokmatched / external_B2 | False | True | True | 2613 | 224603 |
| arm_ordinary_tokmatched / external_B2 | True | True | False | 1638 | 703256 |
| arm_ordinary_tokmatched / external_B2 | True | True | True | 2759 | 294143 |
| arm_ordinary_tokmatched / native | False | False | False | 656 | 181758 |
| arm_ordinary_tokmatched / native | False | True | False | 1 | 362 |
| arm_ordinary_tokmatched / native | True | False | False | 3 | 5046 |
| all_resolved | False | False | False | 1775 | 520348 |
| all_resolved | False | False | True | 12 | 4207 |
| all_resolved | False | True | False | 12847 | 1935926 |
| all_resolved | False | True | True | 15811 | 1380983 |
| all_resolved | True | False | False | 7 | 14402 |
| all_resolved | True | False | True | 1 | 392 |
| all_resolved | True | True | False | 9803 | 4187423 |
| all_resolved | True | True | True | 16446 | 1757459 |
| all_resolved / external_A | False | False | False | 15 | 13714 |
| all_resolved / external_A | False | False | True | 12 | 4207 |
| all_resolved / external_A | False | True | False | 116 | 26244 |
| all_resolved / external_A | False | True | True | 160 | 13937 |
| all_resolved / external_A | True | False | False | 2 | 2385 |
| all_resolved / external_A | True | False | True | 1 | 392 |
| all_resolved / external_A | True | True | False | 30 | 9452 |
| all_resolved / external_A | True | True | True | 46 | 5027 |
| all_resolved / external_B | False | True | False | 5867 | 864847 |
| all_resolved / external_B | False | True | True | 7167 | 630654 |
| all_resolved / external_B | True | True | False | 4538 | 1991950 |
| all_resolved / external_B | True | True | True | 7593 | 814848 |
| all_resolved / external_B2 | False | True | False | 6863 | 1044473 |
| all_resolved / external_B2 | False | True | True | 8484 | 736392 |
| all_resolved / external_B2 | True | True | False | 5235 | 2186021 |
| all_resolved / external_B2 | True | True | True | 8807 | 937584 |
| all_resolved / native | False | False | False | 1760 | 506634 |
| all_resolved / native | False | True | False | 1 | 362 |
| all_resolved / native | True | False | False | 5 | 12017 |
| arm_ordinary_multi | False | False | False | 545 | 144654 |
| arm_ordinary_multi | False | False | True | 3 | 1020 |
| arm_ordinary_multi | False | True | False | 3702 | 551414 |
| arm_ordinary_multi | False | True | True | 4640 | 406020 |
| arm_ordinary_multi | True | False | False | 2 | 4692 |
| arm_ordinary_multi | True | True | False | 2858 | 1227778 |
| arm_ordinary_multi | True | True | True | 4844 | 519486 |
| arm_ordinary_multi / external_A | False | False | False | 2 | 1325 |
| arm_ordinary_multi / external_A | False | False | True | 3 | 1020 |
| arm_ordinary_multi / external_A | False | True | False | 19 | 5412 |
| arm_ordinary_multi / external_A | False | True | True | 26 | 2261 |
| arm_ordinary_multi / external_A | True | True | False | 1 | 10 |
| arm_ordinary_multi / external_A | True | True | True | 5 | 459 |
| arm_ordinary_multi / external_B | False | True | False | 1497 | 223263 |
| arm_ordinary_multi / external_B | False | True | True | 1860 | 167164 |
| arm_ordinary_multi / external_B | True | True | False | 1116 | 482369 |
| arm_ordinary_multi / external_B | True | True | True | 1921 | 209014 |
| arm_ordinary_multi / external_B2 | False | True | False | 2185 | 322377 |
| arm_ordinary_multi / external_B2 | False | True | True | 2754 | 236595 |
| arm_ordinary_multi / external_B2 | True | True | False | 1741 | 745399 |
| arm_ordinary_multi / external_B2 | True | True | True | 2918 | 310013 |
| arm_ordinary_multi / native | False | False | False | 543 | 143329 |
| arm_ordinary_multi / native | False | True | False | 1 | 362 |
| arm_ordinary_multi / native | True | False | False | 2 | 4692 |
| arm_short_multi | False | False | False | 684 | 164290 |
| arm_short_multi | False | True | False | 3875 | 636841 |
| arm_short_multi | False | True | True | 4923 | 429773 |
| arm_short_multi | True | False | False | 2 | 4692 |
| arm_short_multi | True | True | False | 2407 | 978526 |
| arm_short_multi | True | True | True | 4231 | 447844 |
| arm_short_multi / external_B | False | True | False | 2130 | 352490 |
| arm_short_multi / external_B | False | True | True | 2661 | 232018 |
| arm_short_multi / external_B | True | True | False | 1283 | 533087 |
| arm_short_multi / external_B | True | True | True | 2303 | 247700 |
| arm_short_multi / external_B2 | False | True | False | 1744 | 283989 |
| arm_short_multi / external_B2 | False | True | True | 2262 | 197755 |
| arm_short_multi / external_B2 | True | True | False | 1124 | 445439 |
| arm_short_multi / external_B2 | True | True | True | 1928 | 200144 |
| arm_short_multi / native | False | False | False | 684 | 164290 |
| arm_short_multi / native | False | True | False | 1 | 362 |
| arm_short_multi / native | True | False | False | 2 | 4692 |
| arm_short_calls | False | False | False | 721 | 210673 |
| arm_short_calls | False | True | False | 3585 | 569408 |
| arm_short_calls | False | True | True | 4429 | 388067 |
| arm_short_calls | True | False | False | 2 | 6971 |
| arm_short_calls | True | True | False | 2476 | 1069824 |
| arm_short_calls | True | True | True | 4303 | 464572 |
| arm_short_calls / external_B | False | True | False | 3415 | 546864 |
| arm_short_calls / external_B | False | True | True | 4211 | 370268 |
| arm_short_calls / external_B | True | True | False | 2335 | 1005386 |
| arm_short_calls / external_B | True | True | True | 4047 | 437467 |
| arm_short_calls / external_B2 | False | True | False | 170 | 22544 |
| arm_short_calls / external_B2 | False | True | True | 218 | 17799 |
| arm_short_calls / external_B2 | True | True | False | 141 | 64438 |
| arm_short_calls / external_B2 | True | True | True | 256 | 27105 |
| arm_short_calls / native | False | False | False | 721 | 210673 |
| arm_short_calls / native | True | False | False | 2 | 6971 |

## Served reference

| trace dir | traces | unreadable | steps with metrics | LLM steps without metrics | other steps | n | prompt p50 | prompt p90 | prompt max | above train p90: arm_ordinary | above train p90: arm_short | above train p90: arm_ordinary_tokmatched | above train p90: all_resolved | above train p90: arm_ordinary_multi | above train p90: arm_short_multi | above train p90: arm_short_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| results/a0_e4b_s1 | 48 | 0 | 615 | 52 | 99 | 615 | 9478 | 13893 | 17300 | 3.90% | 3.90% | 3.90% | 3.90% | 3.90% | 3.90% | 4.07% |
| results/a0_e4b_s2 | 48 | 0 | 574 | 50 | 96 | 574 | 9359 | 13925 | 16990 | 4.88% | 4.88% | 4.88% | 4.88% | 4.88% | 4.88% | 4.88% |
| results/a0_e4b_s3 | 48 | 0 | 577 | 49 | 97 | 577 | 9176 | 13973 | 17053 | 3.47% | 3.47% | 3.47% | 3.47% | 3.47% | 3.47% | 3.64% |
| pooled | 144 | 0 | 1766 | 151 | 292 | 1766 | 9362 | 13935 | 17300 | 4.08% | 4.08% | 4.08% | 4.08% | 4.08% | 4.08% | 4.19% |

Train loss_pos_p90 per arm:

| arm | loss_pos_p90 |
|---|---|
| arm_ordinary | 14617 |
| arm_short | 14622 |
| arm_ordinary_tokmatched | 14618 |
| all_resolved | 14610 |
| arm_ordinary_multi | 14604 |
| arm_short_multi | 14612 |
| arm_short_calls | 14595 |

