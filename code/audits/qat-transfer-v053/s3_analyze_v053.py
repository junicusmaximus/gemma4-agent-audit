#!/usr/bin/env python3
"""Stage 1 / Stage 2 analysis of the QAT-transfer check v053 (CPU), rules as frozen in the re-freeze section of
briefing/research/gemma-4-2026/prereg-2026-11-06-qat-base-lora-fixture-and-M0q.md. Per item and span: mean target
log-prob and argmax accuracy (probe_lib.span_stats, as in the v048 analysis); effects are paired item differences;
intervals are item-level percentile bootstraps (20,000 draws, seed 20261103).
  s1              -> stage1.json; exit 0 = continue to Stage 2, 20 = stop (R >= 0.9)
  validity STAGE  -> fixture validity (HF effect on the T-fit text span >= 1.0); exit 0 valid, 21 invalid
  s2              -> stage2.json (pass rules, optional F_bf control)
Run with the train env (numpy)."""
import json
import os
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/qat-transfer-v053"
V048 = ROOT + "/analysis/first-step-probe-boss-v048"
LEDGER = V048 + "/ledger.jsonl"
sys.path.insert(0, OUT)
import probe_lib as lib  # noqa: E402  (byte copy of the v048 probe_lib.py)

HGEN, TFIT = "H-gen", "T-fit"
POSITIONS = {"depth": 19, "closing_quote": 39}  # target positions (v048 report.md profile: 19 = `depth`, 39 = `<|"|>`)
R_SUPPORT, R_REFUTE, VALID_MIN, HGEN_MIN_WITHIN = 0.6, 0.9, 1.0, 45


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def per_item(rows, spans, stack, cell, layout, kind=None, span="text"):
    """{item_id: span_stats} of one (stack, cell, layout), optionally one item kind."""
    out = {}
    for r in rows:
        if (r["stack"], r["cell"], r["layout"]) != (stack, cell, layout):
            continue
        it = spans[r["item_id"]]
        if kind and it["kind"] != kind:
            continue
        if r["target_ids"] != it["target_ids"]:
            raise ValueError("target ids differ from the ledger: " + r["item_id"])
        st = lib.span_stats([{"logprob": lp, "rank": rk} for lp, rk in zip(r["logprob"], r["rank"])], it, span)
        if st is not None:
            out[r["item_id"]] = st
    return out


def effects(rows, spans, stack, cell, layout, kind, span="text", base_rows=None):
    """{item_id: cell - base} of the span mean log-prob (base from base_rows when given, else the same rows)."""
    a = per_item(rows, spans, stack, cell, layout, kind, span)
    b = per_item(base_rows or rows, spans, stack, "base", layout, kind, span)
    return {k: a[k]["mean_logprob"] - b[k]["mean_logprob"] for k in sorted(set(a) & set(b))}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def ratio_ci(num, den, draws=lib.BOOT_DRAWS, seed=lib.BOOT_SEED):
    """mean(num) / mean(den) with a paired item bootstrap (same resampled items in both)."""
    import numpy as np
    x, y = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(draws, len(x)))
    r = x[idx].mean(axis=1) / y[idx].mean(axis=1)
    lo, hi = np.quantile(r, [0.025, 0.975])
    return float(x.mean() / y.mean()), float(lo), float(hi)


def r_label(r):
    if r <= R_SUPPORT:
        return "R <= 0.6: supports the weights branch"
    if r >= R_REFUTE:
        return "R >= 0.9: refutes the weights branch (no retraining; the vLLM LoRA path is inspected instead)"
    return "0.6 < R < 0.9: between; Stage 2 decides"


def top1(rows, stack, cell, layout):
    return {r["item_id"]: [t[0] if t else None for t in r["top5_ids"]] for r in rows
            if (r["stack"], r["cell"], r["layout"]) == (stack, cell, layout)}


def argmax_agreement(a, b, spans, kind, span="text"):
    """Share of span positions (target coordinates) where two stacks have the same top-1 id."""
    same = total = 0
    for k in sorted(set(a) & set(b)):
        it = spans[k]
        if it["kind"] != kind or it[span + "_span"] is None:
            continue
        lo, hi = (x - it["target_span"][0] for x in it[span + "_span"])
        for p in range(lo, hi):
            total += 1
            same += a[k][p] is not None and a[k][p] == b[k][p]
    return {"positions": total, "agree": same, "share": same / total if total else None}


def cell_table(sources, spans, cells):
    """mean over items of span mean log-prob and accuracy, per group / span / stack / cell."""
    table = {}
    for kind in (HGEN, TFIT):
        for span in lib.SPANS:
            for label, (rows, stack, layout) in sources.items():
                for cell in cells:
                    v = per_item(rows, spans, stack, cell, layout, kind, span)
                    if v:
                        table.setdefault(kind, {}).setdefault(span, {}).setdefault(label, {})[cell] = {
                            "n": len(v), "mean_logprob": mean(x["mean_logprob"] for x in v.values()),
                            "accuracy": mean(x["accuracy"] for x in v.values())}
    return table


def spread(values):
    values = [abs(v) for v in values]
    return {"n": len(values), "p95": lib.p95(values), "max": max(values), "mean": mean(values)} if values else {"n": 0}


def boot(values):
    values = list(values)
    m, lo, hi = lib.bootstrap_mean_ci(values)
    return {"n": len(values), "mean": m, "ci95": [lo, hi]}


def engine_floor_from_v048(spans=None, scores=V048 + "/vllm_v22_scores.jsonl", hf=V048 + "/hf_scores.jsonl",
                           report=V048 + "/report.json"):
    """p95 over H-gen items of |vLLM unquantized base - HF bf16 base| (text span mean log-prob), the add-on's
    base_engine_unq_minus_hf (s4b_v22.py): computed from the probe's score rows (stack vllm_unq, sequential; stack hf,
    single) when they exist, else taken from its report.json; the report.json value is kept as a cross-check."""
    reported = None
    if os.path.exists(report):
        try:
            reported = read_json(report)["v22_addon"]["weights_vs_engine"]["spans"]["text"]["base_engine_unq_minus_hf"]["p95_abs"]
        except (KeyError, TypeError):
            pass
    if spans is not None and os.path.exists(scores):
        a = per_item([r for r in load(scores) if r["stack"] == "vllm_unq"], spans, "vllm_unq", "base", "sequential", HGEN)
        b = per_item([r for r in load(hf) if r["stack"] == "hf"], spans, "hf", "base", "single", HGEN)
        d = [a[k]["mean_logprob"] - b[k]["mean_logprob"] for k in sorted(set(a) & set(b))]
        if d:
            return lib.p95(d), ("computed from v048 vllm_v22_scores.jsonl (vllm_unq base, sequential) vs hf_scores.jsonl (hf "
                                "base, single), H-gen text, n=%d; v048 report.json value %r" % (len(d), reported))
    if reported is not None:
        return reported, "v048 report.json: v22_addon.weights_vs_engine.spans.text.base_engine_unq_minus_hf.p95_abs"
    return None, "unquantized-vLLM rows absent (scores file and report.json)"


def stage1(rows_dq, rows_bf, rows_vq, spans, floor, floor_source):
    """1a parity of the two bases on QAT weights (HF-dq vs vLLM-QAT) and 1b R = M0 effect HF-dq / HF-bf16 (H-gen text)."""
    sources = {"HF-dq": (rows_dq, "hf_dequant", "single"), "HF-bf16 (v048)": (rows_bf, "hf", "single"),
               "vLLM-QAT (v048)": (rows_vq, "vllm", "sequential")}
    dq_base = per_item(rows_dq, spans, "hf_dequant", "base", "single")
    vq_base = per_item(rows_vq, spans, "vllm", "base", "sequential")
    gaps = {k: abs(dq_base[k]["mean_logprob"] - vq_base[k]["mean_logprob"]) for k in sorted(set(dq_base) & set(vq_base))}
    threshold = max(0.05, 1.5 * floor) if floor is not None else 0.05
    a, b = top1(rows_dq, "hf_dequant", "base", "single"), top1(rows_vq, "vllm", "base", "sequential")
    agree = argmax_agreement(a, b, spans, HGEN)
    hg = spread(gaps[k] for k in gaps if spans[k]["kind"] == HGEN)
    parity = {"rule": "pass if p95 over H-gen items of |HF-dq base - vLLM-QAT base| (text) <= max(0.05, 1.5 x engine "
                      "floor) AND argmax agreement on H-gen text positions >= 0.98",
              "engine_floor": floor, "engine_floor_source": floor_source, "threshold": threshold,
              "threshold_note": None if floor is not None else "engine floor absent: threshold is the 0.05 lower bound only",
              "hgen": hg, "tfit": spread(gaps[k] for k in gaps if spans[k]["kind"] == TFIT), "pooled": spread(gaps.values()),
              "argmax_agreement_hgen_text": agree,
              "argmax_agreement_hgen_target": argmax_agreement(a, b, spans, HGEN, "target"),
              "argmax_agreement_tfit_text": argmax_agreement(a, b, spans, TFIT),
              "pass": hg["n"] > 0 and hg["p95"] <= threshold and (agree["share"] or 0) >= 0.98}
    e_dq = effects(rows_dq, spans, "hf_dequant", "m0", "single", HGEN)
    e_bf = effects(rows_bf, spans, "hf", "m0", "single", HGEN)
    e_vq = effects(rows_vq, spans, "vllm", "m0", "sequential", HGEN)
    keys = sorted(set(e_dq) & set(e_bf))
    r, lo, hi = ratio_ci([e_dq[k] for k in keys], [e_bf[k] for k in keys])
    t_dq, t_bf = (effects(rows, spans, st, "m0", "single", TFIT) for rows, st in ((rows_dq, "hf_dequant"), (rows_bf, "hf")))
    transfer = {"n": len(keys), "effect_hf_dq": boot(e_dq[k] for k in keys), "effect_hf_bf16_v048": boot(e_bf[k] for k in keys),
                "effect_vllm_qat_v048": boot(e_vq.values()), "R": r, "R_ci95": [lo, hi], "label": r_label(r),
                "R_ci_crosses_a_boundary": lo <= R_SUPPORT < hi or lo < R_REFUTE <= hi,
                "descriptive_R_vllm_qat_over_hf_bf16": mean(e_vq.values()) / mean(e_bf.values()),
                "descriptive_R_vllm_qat_over_hf_dq": mean(e_vq.values()) / mean(e_dq.values()),
                "descriptive_tfit_R": mean(t_dq.values()) / mean(t_bf.values())}
    zero = effects(rows_dq, spans, "hf_dequant", "zerob", "single", None)
    layout = [x for c in ("base", "zerob", "m0") for x in effects_layout(rows_dq, spans, "hf_dequant", c)]
    return {"cells": cell_table(sources, spans, ("base", "zerob", "m0")), "parity_1a": parity, "transfer_1b": transfer,
            "zero_b_hf_dq": {"n": len(zero), "max_abs": max(abs(x) for x in zero.values()) if zero else None},
            "hf_dq_layout_floor_text": spread(layout), "item_gaps_1a": gaps,
            "decision": "stop" if r >= R_REFUTE else "continue"}


def effects_layout(rows, spans, stack, cell):
    """single - batch8 per item (text) of one HF cell: the HF layout floor."""
    a = per_item(rows, spans, stack, cell, "single")
    b = per_item(rows, spans, stack, cell, "batch8")
    return [a[k]["mean_logprob"] - b[k]["mean_logprob"] for k in sorted(set(a) & set(b))]


def validity(rows, spans, stack, cell):
    e = effects(rows, spans, stack, cell, "single", TFIT)
    m = mean(e.values())
    return {"rule": "HF adapter effect on the T-fit text span >= 1.0 nats/token", "n": len(e), "per_item": e, "mean": m,
            "threshold": VALID_MIN, "valid": len(e) == 5 and m is not None and m >= VALID_MIN}


def position_check(hf_rows, vq_rows, spans, hf_stack, cell, kind):
    """Mean target log-prob and modal argmax at `depth` and the closing quote, HF vs vLLM (same adapter)."""
    pick = lambda rows, st, lay: {r["item_id"]: r for r in rows if (r["stack"], r["cell"], r["layout"]) == (st, cell, lay)  # noqa: E731
                                  and spans[r["item_id"]]["kind"] == kind}
    hf, vq = pick(hf_rows, hf_stack, "single"), pick(vq_rows, "vllm_qat_nopc", "sequential")
    keys = sorted(set(hf) & set(vq))
    out = {}
    for name, pos in POSITIONS.items():
        lp_hf, lp_vq = [hf[k]["logprob"][pos] for k in keys], [vq[k]["logprob"][pos] for k in keys]
        if not keys or None in lp_hf + lp_vq:
            out[name] = {"pos": pos, "n": len(keys), "pass": False, "note": "missing log-probs"}
            continue
        arg_hf, arg_vq = ([r[k]["top5_ids"][pos][0] for k in keys] for r in (hf, vq))
        modal = lambda xs: statistics.mode(xs)  # noqa: E731
        delta = mean(lp_vq) - mean(lp_hf)
        out[name] = {"pos": pos, "target_ids": sorted({spans[k]["target_ids"][pos] for k in keys}), "n": len(keys),
                     "mean_hf": mean(lp_hf), "mean_vllm": mean(lp_vq), "delta": delta,
                     "max_abs_item_delta": max(abs(x - y) for x, y in zip(lp_vq, lp_hf)),
                     "modal_argmax_hf": modal(arg_hf), "modal_argmax_vllm": modal(arg_vq),
                     "items_same_argmax": sum(x == y for x, y in zip(arg_hf, arg_vq)),
                     "pass": abs(delta) <= 0.5 and modal(arg_hf) == modal(arg_vq)}
    return out


def pass_rules(hf_rows, vq_rows, spans, gaps, hf_stack="hf_dequant", cell="fdq"):
    e_hf = {k: effects(hf_rows, spans, hf_stack, cell, "single", k) for k in (HGEN, TFIT)}
    e_vq = {k: effects(vq_rows, spans, "vllm_qat_nopc", cell, "sequential", k) for k in (HGEN, TFIT)}

    def items(kind):
        rows = {}
        for k in sorted(set(e_hf[kind]) & set(e_vq[kind])):
            bound, d = max(0.10, 2 * gaps[k]), e_vq[kind][k] - e_hf[kind][k]
            rows[k] = {"E_hf": e_hf[kind][k], "E_vllm": e_vq[kind][k], "diff": d, "bound": bound, "within": abs(d) <= bound}
        return rows
    t, h = items(TFIT), items(HGEN)
    acc = {name: mean(x["accuracy"] for x in per_item(rows, spans, st, cell, lay, TFIT, span).values())
           for name, rows, st, lay, span in (("hf_text", hf_rows, hf_stack, "single", "text"),
                                             ("vllm_text", vq_rows, "vllm_qat_nopc", "sequential", "text"),
                                             ("hf_target", hf_rows, hf_stack, "single", "target"),
                                             ("vllm_target", vq_rows, "vllm_qat_nopc", "sequential", "target"))}
    pos = {kind: position_check(hf_rows, vq_rows, spans, hf_stack, cell, kind) for kind in (HGEN, TFIT)}
    rules = {"R1_tfit_every_item_within_bound": {"items": t, "within": sum(r["within"] for r in t.values()), "n": len(t),
                                                 "pass": len(t) == 5 and all(r["within"] for r in t.values())},
             "R2_hgen_at_least_45_of_48_within_bound": {"within": sum(r["within"] for r in h.values()), "n": len(h),
                                                        "max_abs_diff": max(abs(r["diff"]) for r in h.values()) if h else None,
                                                        "pass": len(h) == 48 and sum(r["within"] for r in h.values()) >= HGEN_MIN_WITHIN},
             "R3_depth_and_closing_quote": {"positions": pos, "pass": all(p["pass"] for g in pos.values() for p in g.values())},
             "R4_tfit_accuracy_gap": {**acc, "gap_hf_minus_vllm_text": acc["hf_text"] - acc["vllm_text"],
                                      "gap_hf_minus_vllm_target": acc["hf_target"] - acc["vllm_target"],
                                      "pass": acc["hf_text"] - acc["vllm_text"] <= 0.01}}
    failing = [k for k, v in rules.items() if not v["pass"]]
    return {"rules": rules, "hgen_items": h, "verdict": "PASS" if not failing else "FAIL: " + ", ".join(failing),
            "E_mean": {kind: {"hf": mean(e_hf[kind].values()), "vllm": mean(e_vq[kind].values())} for kind in (HGEN, TFIT)}}


def fbf_control(hf_bf_rows, vq_rows, spans):
    """Matched control on the bf16 base: must reproduce the gap (mean dE <= -0.3 x E_HF, T-fit text)."""
    out = {}
    for kind in (TFIT, HGEN):
        e_hf = effects(hf_bf_rows, spans, "hf_bf16", "fbf", "single", kind)
        e_vq = effects(vq_rows, spans, "vllm_qat_nopc", "fbf", "sequential", kind)
        keys = sorted(set(e_hf) & set(e_vq))
        d, ehf = mean(e_vq[k] - e_hf[k] for k in keys), mean(e_hf[k] for k in keys)
        out[kind] = {"n": len(keys), "E_hf": ehf, "E_vllm": mean(e_vq[k] for k in keys), "mean_dE": d,
                     "criterion_minus_0p3_E_hf": -0.3 * ehf, "reproduces_gap": d is not None and d <= -0.3 * ehf}
    out["label"] = ("F_bf reproduces the gap (fixture sensitive)" if out[TFIT]["reproduces_gap"] else
                    "F_bf does not reproduce the gap: fixture insensitive; Stage 1b decides")
    return out


def repeat(rows_a, stack_a, layout_a, rows_b, stack_b, layout_b, spans):
    """Per-item |difference| of the base text mean log-prob between two sessions of the same stack."""
    a = per_item(rows_a, spans, stack_a, "base", layout_a)
    b = per_item(rows_b, spans, stack_b, "base", layout_b)
    return spread(a[k]["mean_logprob"] - b[k]["mean_logprob"] for k in sorted(set(a) & set(b)))


def fixture_stage():
    """The valid fixture run: s2dq, else the 48-epoch rerun s2dq48; None when neither is valid."""
    for stage in ("s2dq", "s2dq48"):
        path = OUT + "/validity_%s.json" % stage
        if os.path.exists(path) and read_json(path)["valid"]:
            return stage
    return None


def stage2(spans):
    s1 = read_json(OUT + "/stage1.json")
    stage = fixture_stage()
    res = {"fixture_stage": stage, "validity": {s: read_json(OUT + "/validity_%s.json" % s) for s in ("s2dq", "s2dq48")
                                                if os.path.exists(OUT + "/validity_%s.json" % s)}}
    if stage is None:
        res["verdict"] = "VOID: no fixture adapter reached the validity threshold"
        return res
    hf_rows, vq_rows = load(OUT + "/hf_scores_%s.jsonl" % stage), load(OUT + "/vllm_scores_v053.jsonl")
    res["fdq"] = pass_rules(hf_rows, vq_rows, spans, s1["item_gaps_1a"])
    res["verdict"] = res["fdq"]["verdict"]
    res["repeats"] = {"vllm_base_nopc_vs_v048": repeat(vq_rows, "vllm_qat_nopc", "sequential", load(V048 + "/vllm_scores.jsonl"),
                                                       "vllm", "sequential", spans),
                      "hf_dq_base_stage2_vs_stage1": repeat(hf_rows, "hf_dequant", "single", load(OUT + "/hf_scores_s1.jsonl"),
                                                            "hf_dequant", "single", spans)}
    if os.path.exists(OUT + "/hf_scores_s2bf.jsonl") and any(r["cell"] == "fbf" for r in vq_rows):
        res["fbf_control"] = fbf_control(load(OUT + "/hf_scores_s2bf.jsonl"), vq_rows, spans)
    return res


def write_new(path, obj):
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=1) + "\n")


def main(argv):
    spans = {r["item_id"]: r for r in load(LEDGER)}
    if argv[1] == "s1":
        floor, source = engine_floor_from_v048(spans)
        res = stage1(load(OUT + "/hf_scores_s1.jsonl"), load(V048 + "/hf_scores.jsonl"), load(V048 + "/vllm_scores.jsonl"),
                     spans, floor, source)
        write_new(OUT + "/stage1.json", res)
        t = res["transfer_1b"]
        print(json.dumps({"1a_pass": res["parity_1a"]["pass"], "1a_hgen_p95": res["parity_1a"]["hgen"]["p95"],
                          "1a_threshold": res["parity_1a"]["threshold"], "argmax": res["parity_1a"]["argmax_agreement_hgen_text"],
                          "R": t["R"], "R_ci95": t["R_ci95"], "label": t["label"], "decision": res["decision"]}))
        return 20 if res["decision"] == "stop" else 0
    if argv[1] == "validity":
        stage = argv[2]
        stack, cell = ("hf_bf16", "fbf") if stage == "s2bf" else ("hf_dequant", "fdq")
        res = validity(load(OUT + "/hf_scores_%s.jsonl" % stage), spans, stack, cell)
        write_new(OUT + "/validity_%s.json" % stage, res)
        print(json.dumps({k: res[k] for k in ("n", "mean", "valid")}))
        return 0 if res["valid"] else 21
    if argv[1] == "s2":
        res = stage2(spans)
        write_new(OUT + "/stage2.json", res)
        print(json.dumps({"fixture_stage": res["fixture_stage"], "verdict": res["verdict"],
                          "fbf": res.get("fbf_control", {}).get("label")}))
        return 0
    sys.exit("usage: s1 | validity STAGE | s2")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
