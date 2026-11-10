#!/usr/bin/env python3
"""Tiny hand-made tables with known answers: tests the bootstrap and the label rule without any project data.

Python 3.10+, numpy only. Usage:  python3 synthetic_fixture.py     (exit 1 on any failure)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_paper_numbers import DRAWS, SEED, label, paired_bootstrap  # noqa: E402

# tasks x 3 decoding seeds, 1 = resolved
TREATED = [[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]]
ZEROS = [[0, 0, 0]] * 4
ONES5 = [[1, 1, 1]] * 5
ZEROS5 = [[0, 0, 0]] * 5


def naive_bootstrap(treated, baseline, draws, seed):
    """Plain-Python version: same random indices, explicit loops, manual linear quantile."""
    delta = [sum(t) / 3 - sum(b) / 3 for t, b in zip(treated, baseline)]
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, len(delta), size=(draws, len(delta))).tolist()
    means = sorted(sum(delta[j] for j in row) / len(row) for row in idx)

    def q(p):
        pos = p * (len(means) - 1)
        lo = int(pos)
        return means[lo] + (means[min(lo + 1, len(means) - 1)] - means[lo]) * (pos - lo)
    return 100 * q(0.025), 100 * q(0.975)


def main():
    fails = []

    def check(name, ok, detail=""):
        print("%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    a = paired_bootstrap(TREATED, ZEROS)
    check("effect of the 4-task table is 25 pp", abs(a["effect_pp"] - 25.0) < 1e-12, "got %.12f" % a["effect_pp"])
    check("seed effects are [50, 25, 0] pp", np.allclose(a["seed_effects_pp"], [50, 25, 0]), str(a["seed_effects_pp"]))
    lo, hi = naive_bootstrap(TREATED, ZEROS, DRAWS, SEED)
    check("bootstrap interval equals the plain-Python version", abs(a["ci95_pp"][0] - lo) < 1e-9 and abs(a["ci95_pp"][1] - hi) < 1e-9,
          "[%.4f, %.4f]" % tuple(a["ci95_pp"]))
    check("interval brackets the effect", a["ci95_pp"][0] <= a["effect_pp"] <= a["ci95_pp"][1])
    b = paired_bootstrap(ONES5, ZEROS5)
    check("all-resolved vs none: 100 pp [100, 100]", b["effect_pp"] == 100.0 and b["ci95_pp"] == [100.0, 100.0], str(b))
    c = paired_bootstrap(ZEROS5, ZEROS5)
    check("identical arms: 0 pp [0, 0]", c["effect_pp"] == 0.0 and c["ci95_pp"] == [0.0, 0.0], str(c))
    d = paired_bootstrap(ZEROS, TREATED)
    check("swapping the arms negates effect and interval", abs(d["effect_pp"] + a["effect_pp"]) < 1e-9 and
          abs(d["ci95_pp"][0] + a["ci95_pp"][1]) < 1e-9 and abs(d["ci95_pp"][1] + a["ci95_pp"][0]) < 1e-9)
    try:
        paired_bootstrap([[1, 0]], [[0, 0]])
        check("a table with two seeds is rejected", False)
    except ValueError:
        check("a table with two seeds is rejected", True)
    # frozen label rule: NON-COLLAPSED loop <= 0.30 and submit >= 0.80; COLLAPSED loop > 0.40 or submit < 0.70
    cases = [(0.008, 0.958, "NON-COLLAPSED"), (0.30, 0.80, "NON-COLLAPSED"), (0.683, 0.142, "COLLAPSED"), (0.10, 0.60, "COLLAPSED"),
             (0.41, 0.95, "COLLAPSED"), (0.35, 0.90, "INTERMEDIATE"), (0.40, 0.70, "INTERMEDIATE"), (0.20, 0.75, "INTERMEDIATE")]
    for lp, sb, want in cases:
        check("label(loop=%.3f, submit=%.3f) = %s" % (lp, sb, want), label(lp, sb) == want, "got " + label(lp, sb))
    print("FIXTURE: %d failed" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
