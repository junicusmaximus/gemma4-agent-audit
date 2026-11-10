#!/usr/bin/env python3
"""Recompute the numbers of the paper from the per-task tables of this package and compare them with the sealed reports.

Python 3.10+, numpy only. Usage:  python3 verify_paper_numbers.py [PACKAGE_ROOT]
PACKAGE_ROOT defaults to the parent of this file's folder. Exit status 1 if any check FAILs.

Each check prints one line:  PASS|FAIL|NOT RECOMPUTABLE  name  recomputed=...  sealed=...  [paper=...]
A check whose input file is not in the package, or whose input cannot come from a per-task table, is reported as
NOT RECOMPUTABLE and is never counted as PASS.

What is recomputed
  (1) resolved counts per arm and decoding seed (Table 1) from the per-task resolved flags in the sealed reports
  (2) the paired task-bootstrap intervals (20,000 resamples, numpy Generator(PCG64(20261103)), per-task mean over the three
      decoding seeds of treated minus baseline, linear 2.5 / 97.5 % quantiles) for every two-arm report, and the intervals
      printed in the paper
  (3) loop and submit rates from per_task.jsonl and the frozen termination labels
  (4) the repair counts of the external panel
"""
import json
import os
import sys

import numpy as np

SEED, DRAWS = 20261103, 20000
LOOP_OK, SUBMIT_OK = 0.30, 0.80      # NON-COLLAPSED: loop <= 0.30 and submit >= 0.80
LOOP_BAD, SUBMIT_BAD = 0.40, 0.70    # COLLAPSED: loop > 0.40 or submit < 0.70
TOL = 1e-9


def paired_bootstrap(treated, baseline, draws=DRAWS, seed=SEED):
    """treated, baseline: tasks x 3 arrays of 0/1. Returns effect_pp, ci95_pp, seed_effects_pp (percentage points)."""
    treated, baseline = np.asarray(treated, dtype=float), np.asarray(baseline, dtype=float)
    if treated.shape != baseline.shape or treated.ndim != 2 or treated.shape[1] != 3 or treated.shape[0] == 0:
        raise ValueError("expected paired task-by-three-seed arrays")
    delta = (treated - baseline).mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, len(delta), size=(draws, len(delta)))
    lo, hi = np.quantile(delta[idx].mean(axis=1), [0.025, 0.975], method="linear")
    return {"effect_pp": float(100 * delta.mean()), "ci95_pp": [float(100 * lo), float(100 * hi)],
            "seed_effects_pp": (100 * (treated.mean(axis=0) - baseline.mean(axis=0))).tolist()}


def label(loop, submit):
    if loop <= LOOP_OK and submit >= SUBMIT_OK:
        return "NON-COLLAPSED"
    if loop > LOOP_BAD or submit < SUBMIT_BAD:
        return "COLLAPSED"
    return "INTERMEDIATE"


class Results:
    def __init__(self):
        self.rows = []

    def add(self, status, name, recomputed="", sealed="", paper="", note=""):
        self.rows.append({"status": status, "name": name, "recomputed": str(recomputed), "sealed": str(sealed),
                          "paper": str(paper), "note": note})

    def compare(self, name, recomputed, sealed, paper=None, tol=TOL):
        def same(a, b):
            if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            if isinstance(a, str) or isinstance(b, str):
                return a == b
            return abs(float(a) - float(b)) <= tol
        ok = same(recomputed, sealed) and (paper is None or same(recomputed, paper))
        self.add("PASS" if ok else "FAIL", name, recomputed, sealed, "" if paper is None else paper)

    def na(self, name, reason):
        self.add("NOT RECOMPUTABLE", name, note="NOT RECOMPUTABLE from public tables: " + reason)

    def counts(self):
        out = {"PASS": 0, "FAIL": 0, "NOT RECOMPUTABLE": 0}
        for r in self.rows:
            out[r["status"]] += 1
        return out

    def format(self):
        lines = []
        for r in self.rows:
            if r["status"] == "NOT RECOMPUTABLE":
                lines.append("%-16s %s  %s" % (r["status"], r["name"], r["note"]))
            else:
                extra = "  paper=%s" % r["paper"] if r["paper"] else ""
                lines.append("%-16s %s  recomputed=%s  sealed=%s%s" % (r["status"], r["name"], r["recomputed"], r["sealed"], extra))
        c = self.counts()
        lines.append("SUMMARY: PASS %d  FAIL %d  NOT RECOMPUTABLE %d" % (c["PASS"], c["FAIL"], c["NOT RECOMPUTABLE"]))
        return "\n".join(lines)


def fmt(x):
    return "%.2f" % (round(float(x), 2) + 0.0)


class Package:
    def __init__(self, root):
        self.root = root
        self.cache = {}

    def path(self, rel):
        return os.path.join(self.root, rel)

    def json(self, rel):
        if rel not in self.cache:
            p = self.path(rel)
            self.cache[rel] = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None
        return self.cache[rel]

    def jsonl(self, rel):
        key = rel + "#l"
        if key not in self.cache:
            p = self.path(rel)
            self.cache[key] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.isfile(p) else None
        return self.cache[key]


# Table 1 of the manuscript: (row, analysis folder, arm key in that report, repair per decoding seed printed in the paper)
TABLE1 = [
    ("A0", "v02-v025", "A0", [3, 2, 2]), ("A1", "v02-v025", "A1", [0, 0, 0]), ("A2", "v02-v025", "A2", [0, 0, 0]),
    ("M0", "mask-control-v043", "M0", [0, 0, 0]), ("M1", "mask-control-v043", "M1", [0, 0, 0]),
    ("M0mc", "matched-context-v051", "v051_M0mc", [0, 0, 0]), ("A0cap", "a0-capped-v052", "A0cap", [5, 5, 5]),
    ("M0q", "m0q-eval-v054", "M0q", [0, 0, 0]), ("M0qmc", "matched-context-v055", "v055_M0qmc", [0, 0, 0]),
    ("Hq1mc", "matched-context-v058", "v058_Hq1mc", [0, 0, 0]), ("M1qmc", "matched-context-v062", "v062_M1qmc", [4, 2, 4]),
    ("M1qmc-s2", "matched-context-v067", "v067_M1qs2mc", [3, 2, 1]), ("A0mc", "matched-context-v064a", "v066_A0mc", [4, 3, 5]),
    ("A_onpol", "matched-context-v064a", "v064_A_onpol", [5, 3, 2]), ("T_sub", "matched-context-v064t", "v064_T_sub", [0, 0, 0]),
    ("A_onpol-s2", "matched-context-v064c", "v064_A_onpol_c2", [3, 2, 4]),
]
# two-arm reports (second arm = treated, first = baseline) and the interval printed in the paper, if any
TWO_ARM = ["mask-control-v043", "matched-context-v051", "a0-capped-v052", "m0q-eval-v054", "matched-context-v055", "matched-context-v058",
           "matched-context-v062", "matched-context-v067", "matched-context-v064a", "matched-context-v064t", "matched-context-v064c"]
PAPER_INTERVALS = {   # (folder): (label, effect_pp, lo, hi) as printed in the manuscript
    "a0-capped-v052": ("A0cap - M0", 12.50, 4.17, 22.50),
    "matched-context-v062": ("M1qmc - M0qmc", 8.33, 1.67, 16.67),
    "matched-context-v067": ("M1qmc-s2 - M0qmc", 5.00, 0.83, 10.00),
    "matched-context-v064a": ("A_onpol - A0mc", -1.67, -6.67, 2.50),
    "matched-context-v064t": ("T_sub - A0mc", -10.00, -17.52, -3.33),
    "matched-context-v064c": ("A_onpol-s2 - A0mc", -2.50, -6.67, 0.00),
}
HOLDOUT_LABELS = {"matched-context-v064a": ("v064_A_onpol", "NON-COLLAPSED"), "matched-context-v064t": ("v064_T_sub", "COLLAPSED"),
                  "matched-context-v064c": ("v064_A_onpol_c2", "NON-COLLAPSED")}


def primary(rep):
    return rep["populations"]["primary40"]["primary"]


def flags(p, arm):
    return np.array([t[arm] for t in p["per_task"]], dtype=bool)


def check_resolved(pkg, res):
    for row, folder, arm, paper in TABLE1:
        name = "Table 1 repair %s (%s)" % (row, folder)
        rep = pkg.json("analyses/%s/report.json" % folder)
        if rep is None:
            res.na(name, "analyses/%s/report.json is not in the package" % folder)
            continue
        try:
            if folder == "v02-v025":
                p = rep["populations"]["primary40"]
                ids = p["task_ids"]
                rec = [sum(bool(rep["per_task"][t][arm][str(s)]["resolved"]) for t in ids) for s in (1, 2, 3)]
                sealed = [int(round(p["arms"][arm]["seed_resolution_rates"][str(s)] * len(ids))) for s in (1, 2, 3)]
            else:
                p = primary(rep)
                rec = flags(p, arm).sum(axis=0).tolist()
                sealed = p["arms"][arm]["resolved_by_seed"]
            res.compare(name, rec, sealed, paper, tol=0)
        except Exception as e:  # unexpected layout
            res.add("FAIL", name, note="error: %r" % (e,))
    rep = pkg.json("analyses/matched-context-v066/report.json")
    if rep is not None:
        try:
            res.compare("A0mc sealed in v066 equals A0mc flags in v064a", pkg_a0mc(pkg), rep["populations"]["primary40"]["resolved_by_seed"]["v066_A0mc"], [4, 3, 5], tol=0)
        except Exception as e:
            res.add("FAIL", "A0mc v066 cross-check", note="error: %r" % (e,))


def pkg_a0mc(pkg):
    rep = pkg.json("analyses/matched-context-v064a/report.json")
    return flags(primary(rep), "v066_A0mc").sum(axis=0).tolist()


def check_bootstrap(pkg, res):
    rep = pkg.json("analyses/v02-v025/report.json")
    if rep is None:
        res.na("bootstrap A1 - A0 (holdout, 40 tasks)", "analyses/v02-v025/report.json is not in the package")
    else:
        try:
            p = rep["populations"]["primary40"]
            ids = p["task_ids"]
            arr = {a: np.array([[bool(rep["per_task"][t][a][str(s)]["resolved"]) for s in (1, 2, 3)] for t in ids]) for a in ("A0", "A1", "A2")}
            for key, (tr, ba) in {"A1-A0": ("A1", "A0"), "A2-A0": ("A2", "A0"), "A2-A1": ("A2", "A1")}.items():
                b = paired_bootstrap(arr[tr], arr[ba])
                s = p["comparisons"][key]
                paper = [-5.83, -12.50, -0.83] if key == "A1-A0" else None
                res.compare("bootstrap %s (holdout, 40 tasks) effect and CI (pp)" % key, [b["effect_pp"]] + b["ci95_pp"],
                            [s["effect_pp"]] + s["ci95_pp"], None)
                if paper:
                    res.compare("paper-printed A1 - A0: -5.83 [-12.50, -0.83] (2 decimals)", [round(x, 2) for x in [b["effect_pp"]] + b["ci95_pp"]], paper, None, tol=1e-9)
        except Exception as e:
            res.add("FAIL", "bootstrap v02-v025", note="error: %r" % (e,))
    for folder in TWO_ARM:
        rep = pkg.json("analyses/%s/report.json" % folder)
        name = "bootstrap %s" % folder
        if rep is None:
            res.na(name, "analyses/%s/report.json is not in the package" % folder)
            continue
        try:
            p = primary(rep)
            base, treat = list(p["arms"])[:2]
            b = paired_bootstrap(flags(p, treat), flags(p, base))
            res.compare("%s: %s - %s effect and CI (pp)" % (name, treat, base), [b["effect_pp"]] + b["ci95_pp"], [p["effect_pp"]] + p["ci95_pp"], None)
            res.compare("%s seed effects (pp)" % name, b["seed_effects_pp"], p["seed_effects_pp"], None)
            if folder in PAPER_INTERVALS:
                lab, e, lo, hi = PAPER_INTERVALS[folder]
                got = [round(x, 2) + 0.0 for x in [b["effect_pp"]] + b["ci95_pp"]]
                res.compare("paper-printed %s: %s [%s, %s] (2 decimals)" % (lab, fmt(e), fmt(lo), fmt(hi)), got, [e, lo, hi], None, tol=1e-9)
        except Exception as e:
            res.add("FAIL", name, note="error: %r" % (e,))


def run_rates(rows, ids, run):
    sel = [r for r in rows if (r["run"] == run or r["run"].endswith("_" + run)) and r["instance_id"] in ids]
    if not sel:
        return None
    return float(np.mean([bool(r["loop"]) for r in sel])), float(np.mean([bool(r["submit_called"]) for r in sel])), len(sel)


def check_termination(pkg, res):
    res.na("loop and submit rates and labels for A0, A1, A2 (v02-v025)", "that analysis has no trace-derived per-task table (loop and submit come from traces, which are not published)")
    ids_ref = None
    rep_a = pkg.json("analyses/matched-context-v064a/report.json")
    if rep_a is not None:
        ids_ref = {t["instance_id"] for t in primary(rep_a)["per_task"]}
    for folder in [f for f in TWO_ARM] + ["matched-context-v066"]:
        rows = pkg.jsonl("analyses/%s/per_task.jsonl" % folder)
        rep = pkg.json("analyses/%s/report.json" % folder)
        name = "loop/submit %s" % folder
        if rows is None or rep is None:
            res.na(name, "analyses/%s/per_task.jsonl or report.json is not in the package" % folder)
            continue
        try:
            ids = ids_ref if folder == "matched-context-v066" else {t["instance_id"] for t in primary(rep)["per_task"]}
            if ids is None:
                res.na(name, "no primary40 task list available")
                continue
            summ = rep["populations"]["primary40"]["per_run_summaries"]
            arm_means = {}
            for arm, runs in summ.items():
                per_run = []
                for rr in runs:
                    got = run_rates(rows, ids, rr["run"])
                    if got is None:
                        res.na("%s run %s" % (name, rr["run"]), "no rows of this run in per_task.jsonl")
                        continue
                    res.compare("%s run %s loop, submit" % (name, rr["run"]), list(got[:2]), [rr["loop_rate"], rr["submit_rate"]], None, tol=1e-12)
                    per_run.append(got[:2])
                if len(per_run) == 3:
                    arm_means[arm] = tuple(float(np.mean([x[i] for x in per_run])) for i in (0, 1))
            if folder in HOLDOUT_LABELS:
                arm, want = HOLDOUT_LABELS[folder]
                sealed = rep["readings"]["arm"]["primary40"]["termination"]
                lp, sb = arm_means[arm]
                res.compare("%s %s three-seed loop, submit" % (name, arm), [lp, sb], [sealed["loop"], sealed["submit"]], None, tol=1e-12)
                res.compare("%s %s frozen label" % (name, arm), label(lp, sb), sealed["label"], want)
        except Exception as e:
            res.add("FAIL", name, note="error: %r" % (e,))


def check_external(pkg, res):
    rep = pkg.json("analyses/codex-v070-git-r2/readout/report.json")
    rows = pkg.jsonl("analyses/codex-v070-git-r2/readout/per_task.jsonl")
    if rep is None or rows is None:
        res.na("external panel (v070)", "analyses/codex-v070-git-r2/readout/ is not in the package")
    else:
        try:
            paper_repair = {"A_onpol_ts1": 6, "A_onpol_ts2": 7, "T_sub": 0}
            paper_submit = {"A_onpol_ts1": 36, "A_onpol_ts2": 36, "T_sub": 3}
            for arm in ("A_onpol_ts1", "A_onpol_ts2", "T_sub"):
                sel = [r for r in rows if r["arm"] == arm]
                resolved = sum(bool(r["resolved"]) for r in sel)
                sealed = rep["repair"]["pooled"][arm]["complete_tasks"]
                res.compare("external repair %s: resolved of %d" % (arm, len(sel)), [resolved, len(sel)], [sealed["resolved"], sealed["denominator"]], [paper_repair[arm], 36], tol=0)
                parsed = sum(r["submission_diagnostics"]["parsed_submit_attempts"] >= 1 for r in sel)
                sealed_submit = rep["readings"]["termination"][arm]["submit"]
                res.compare("external submit %s: parsed submit_patch attempts" % arm, parsed, round(sealed_submit * len(sel)), paper_submit[arm], tol=0)
                res.na("external loop and frozen label %s" % arm, "the external per-task table has no loop flag; loop is derived from traces, which are not published")
        except Exception as e:
            res.add("FAIL", "external panel (v070)", note="error: %r" % (e,))
    rows = pkg.jsonl("analyses/external-e4b-v068/per_task.jsonl")
    if rows is None:
        res.na("external panel v068 (every arm 0/36)", "analyses/external-e4b-v068/per_task.jsonl is not in the package")
    else:
        try:
            per_arm = {}
            for r in rows:
                per_arm.setdefault(r["arm"], []).append(bool(r["resolved"]))
            res.compare("external v068 resolved per arm (all arms 0 of 36, harness defect)", sorted((a, sum(v), len(v)) for a, v in per_arm.items()),
                        sorted((a, 0, 36) for a in per_arm), None, tol=0)
        except Exception as e:
            res.add("FAIL", "external panel v068", note="error: %r" % (e,))


def run_checks(root):
    pkg, res = Package(root), Results()
    for fn in (check_resolved, check_bootstrap, check_termination, check_external):
        fn(pkg, res)
    return res


def main(argv):
    root = argv[1] if len(argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = run_checks(root)
    print(res.format())
    return 1 if res.counts()["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
