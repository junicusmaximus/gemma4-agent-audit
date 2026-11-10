"""Amendment v2.2 analysis (CPU), called by s4_analyze.py when the add-on score files exist. Rules fixed before any add-on
score exists. MDE(span) = 0.5 x |HF T-fit M0 - base| on that span (S1 cells). Positive control: m0_a128 - base in the
same session; 'large' = item-bootstrap CI excludes 0 AND |mean| >= MDE; 'applied' needs a large change on the text span
in both stacks. Weights vs engine (H-gen, M0): engine = (M0-base)[vLLM unquantized] - (M0-base)[HF]; weights =
(M0-base)[vLLM QAT] - (M0-base)[vLLM unquantized]; material iff |mean| >= MDE or the argmax flips at the first action
token / control token on >= 5 of 48 items. Orphan items (n = 3): per-item values; 'difference beyond noise on span X'
iff all three items move in the same direction by more than the HF layout floor of S1 on that span."""
import collections
import json
import os

PRIMARY = {"hf": "single", "vllm": "sequential", "hf_v22": "single", "vllm_nopc": "sequential", "vllm_unq": "sequential"}
REPEAT = {"hf_v22": "batch8", "vllm_nopc": "grouped8", "vllm_unq": "grouped8"}


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def per_item(lib, spans, scores):
    """val[(stack, cell, layout)][span][item_id] = span_stats; also top-1 ids per position."""
    val = collections.defaultdict(lambda: collections.defaultdict(dict))
    top1 = {}
    for s in scores:
        it = spans[s["item_id"]]
        toks = [{"logprob": lp, "rank": rk} for lp, rk in zip(s["logprob"], s["rank"])]
        for sp in lib.SPANS:
            st = lib.span_stats(toks, it, sp)
            if st is not None:
                val[(s["stack"], s["cell"], s["layout"])][sp][s["item_id"]] = st
        top1[(s["stack"], s["cell"], s["layout"], s["item_id"])] = [t[0] if t else None for t in s["top5_ids"]]
    return val, top1


def diffs(a, b, keys=None):
    keys = sorted(set(a) & set(b) if keys is None else keys)
    return keys, [a[k]["mean_logprob"] - b[k]["mean_logprob"] for k in keys]


def boot(lib, d, mde=None):
    if not d:
        return None
    m, lo, hi = lib.bootstrap_mean_ci(d)
    row = {"n": len(d), "mean": m, "ci95": [lo, hi]}
    if mde is not None:
        row.update(mde=mde, large=(lo > 0 or hi < 0) and abs(m) >= mde)
    return row


def ids_of(spans, kind):
    return {k for k, v in spans.items() if v["kind"] == kind}


def analyse(OUT, lib, ledger, base_scores, an, tok=None):
    ledger_v22 = load(OUT + "/ledger_v22.jsonl")
    spans = {r["item_id"]: r for r in ledger + ledger_v22}
    add = load(OUT + "/hf_scores_v22.jsonl") + load(OUT + "/vllm_v22_scores.jsonl")
    val, top1 = per_item(lib, spans, base_scores + add)
    get = lambda st, c, sp, kind=None, lay=None: {k: v for k, v in val[(st, c, lay or PRIMARY[st])][sp].items()  # noqa: E731
                                                  if kind is None or spans[k]["kind"] == kind}
    mde = {sp: 0.5 * abs(an["T-fit"][sp]["within"]["hf:m0-base:mean_logprob"]["mean"]) for sp in lib.SPANS}
    out = {"mde": mde, "declaration": "fix decisions rest on the training-fit items (T-fit, T-orphan); the 48 held-out "
                                      "items are descriptive"}
    pc = {}
    for st in ("hf_v22", "vllm_nopc"):
        for kind in ("H-gen", "T-fit"):
            for sp in lib.SPANS:
                base, a128 = get(st, "base", sp, kind), get(st, "m0_a128", sp, kind)
                if base and a128:
                    pc["%s|%s|%s" % (st, kind, sp)] = boot(lib, diffs(a128, base)[1], mde[sp])
    applied = {st: bool(pc.get("%s|H-gen|text" % st)) and pc["%s|H-gen|text" % st]["large"] for st in ("hf_v22", "vllm_nopc")}
    out["positive_control"] = {"effects_m0_a128_minus_base": pc, "large_text_change": applied,
                               "label": "not measurable" if not pc else
                               "adapter applied in both stacks (large text-span change)" if all(applied.values()) else
                               "large change not seen in: " + ", ".join(k for k, v in applied.items() if not v)}
    rep = {}
    for st_new, st_old, layouts in (("hf_v22", "hf", (("single", "single"), ("batch8", "batch8"))),
                                    ("vllm_nopc", "vllm", (("sequential", "sequential"), ("grouped8", "concurrent8")))):
        for new_l, old_l in layouts:
            for sp in lib.SPANS:
                a, b = val[(st_new, "base", new_l)][sp], val[(st_old, "base", old_l)][sp]
                d = diffs(a, b)[1]
                if d:
                    rep["%s:%s vs %s:%s|%s" % (st_new, new_l, st_old, old_l, sp)] = {"n": len(d), "max_abs": max(abs(x) for x in d),
                                                                                    "items_nonzero": sum(1 for x in d if x != 0.0)}
    for st in REPEAT:
        for sp in lib.SPANS:
            d = [x for c in ("base", "m0", "zerob", "m1", "m0_a128")
                 for x in diffs(val[(st, c, PRIMARY[st])][sp], val[(st, c, REPEAT[st])][sp])[1]]
            if d:
                rep["%s layout %s vs %s|%s" % (st, PRIMARY[st], REPEAT[st], sp)] = {"n": len(d), "max_abs": max(abs(x) for x in d),
                                                                                    "items_nonzero": sum(1 for x in d if x != 0.0)}
    out["repeat_differences"] = rep
    dec = {}
    hgen = ids_of(spans, "H-gen")
    for sp in lib.SPANS:
        eff = {}
        for st in ("hf", "vllm_unq", "vllm"):
            m0, b = get(st, "m0", sp, "H-gen"), get(st, "base", sp, "H-gen")
            if m0 and b:
                eff[st] = dict(zip(*diffs(m0, b)))
        row = {"mde": mde[sp]}
        for name, (x, y) in (("engine_unq_minus_hf", ("vllm_unq", "hf")), ("weights_qat_minus_unq", ("vllm", "vllm_unq")),
                             ("total_qat_minus_hf", ("vllm", "hf"))):
            if x in eff and y in eff:
                keys = sorted(set(eff[x]) & set(eff[y]))
                row[name] = boot(lib, [eff[x][k] - eff[y][k] for k in keys], mde[sp])
        for name, (x, y) in (("base_engine_unq_minus_hf", ("vllm_unq", "hf")), ("base_weights_qat_minus_unq", ("vllm", "vllm_unq"))):
            a, b = get(x, "base", sp, "H-gen"), get(y, "base", sp, "H-gen")
            if a and b:
                d = diffs(a, b)[1]
                row[name] = {"p95_abs": lib.p95(d), "mean": sum(d) / len(d)}
        g_rule = {}
        for name, (x, y) in (("engine_unq_vs_hf", ("vllm_unq", "hf")), ("weights_qat_vs_unq", ("vllm", "vllm_unq"))):
            bx, by = get(x, "base", sp, "H-gen"), get(y, "base", sp, "H-gen")
            mx, my = get(x, "m0", sp, "H-gen"), get(y, "m0", sp, "H-gen")
            if bx and by and mx and my:
                floor = lib.p95(diffs(bx, by)[1])
                d = diffs(mx, my)[1]
                ci = lib.bootstrap_mean_ci(d)
                g_rule[name] = dict(lib.beyond_noise(d, floor, ci), mean=ci[0], ci95=[ci[1], ci[2]], base_floor_p95=floor)
        row["g_rule_m0_row_vs_base_floor"] = g_rule
        dec[sp] = row
    flips = {}
    for name, (x, y) in (("engine_unq_vs_hf", ("vllm_unq", "hf")), ("weights_qat_vs_unq", ("vllm", "vllm_unq"))):
        for pos_name, pos in (("first_action_token_pos4", 4), ("control_token_pos42", 42)):
            n = sum(1 for k in hgen if (x, "m0", PRIMARY[x], k) in top1 and (y, "m0", PRIMARY[y], k) in top1
                    and top1[(x, "m0", PRIMARY[x], k)][pos] != top1[(y, "m0", PRIMARY[y], k)][pos])
            flips["%s|%s" % (name, pos_name)] = n
    material = [("%s on %s" % (n, sp)) for sp, r in dec.items() for n in ("engine_unq_minus_hf", "weights_qat_minus_unq")
                if r.get(n) and r[n]["large"]] + [k for k, n in flips.items() if n >= 5]
    out["weights_vs_engine"] = {"spans": dec, "argmax_flips_m0": flips, "label": "not measurable (no unquantized vLLM cells)"
                                if not any("engine_unq_minus_hf" in r for r in dec.values()) else
                                ("material: " + "; ".join(material)) if material else "within numeric drift"}
    orphan = sorted(ids_of(spans, "T-orphan"))
    rows, labels = [], {}
    for k in orphan:
        r = {"item_id": k, "terminator": spans[k].get("terminator"), "target_tokens": len(spans[k]["target_ids"])}
        for st in ("hf_v22", "vllm_nopc"):
            for c in ("base", "m0", "zerob", "m1", "m0_a128"):
                for sp in lib.SPANS:
                    v = val[(st, c, PRIMARY[st])][sp].get(k)
                    if v:
                        r["%s:%s:%s" % (st, c, sp)] = [round(v["mean_logprob"], 4) if v["mean_logprob"] is not None else None,
                                                       round(v["accuracy"], 3)]
        rows.append(r)
    for st in ("hf_v22", "vllm_nopc"):
        for sp in lib.SPANS:
            m0, b = val[(st, "m0", PRIMARY[st])][sp], val[(st, "base", PRIMARY[st])][sp]
            keys = [k for k in orphan if k in m0 and k in b]
            if len(keys) == len(orphan) and keys:
                d = [m0[k]["mean_logprob"] - b[k]["mean_logprob"] for k in keys]
                floor = an["T-fit"][sp]["floors"].get("layout_hf")
                same = all(x > floor for x in d) or all(x < -floor for x in d)
                labels["%s|%s" % (st, sp)] = {"m0_minus_base": [round(x, 4) for x in d], "floor": floor,
                                              "label": ("difference beyond noise on span " + sp) if same else "none observed within this probe"}
    out["orphan_items"] = {"rows": rows, "m0_vs_base": labels}
    stage = next((p for p in (OUT + "/s6_nopc_r2/score_stage.json", OUT + "/s6_nopc/score_stage.json") if os.path.exists(p)), "")
    out["tokenize_check"] = json.load(open(stage)).get("tokenize") if os.path.exists(stage) else None
    out["terminator_census"] = json.load(open(OUT + "/terminators_v22.json"))["terminator_census"] if os.path.exists(OUT + "/terminators_v22.json") else None
    prof = {}
    for pos in (3, 4, 15, 18, 19, 21, 31, 38, 39, 42):
        cells = {}
        for st, c in (("hf", "m0"), ("vllm_unq", "base"), ("vllm_unq", "m0"), ("vllm", "m0"), ("hf_v22", "m0_a128"), ("vllm_nopc", "m0_a128")):
            lps = [s for s in add + base_scores if s["stack"] == st and s["cell"] == c and s["layout"] == PRIMARY[st]
                   and spans[s["item_id"]]["kind"] == "H-gen"]
            if lps:
                cells[st + ":" + c] = round(sum(s["logprob"][pos] for s in lps) / len(lps), 3)
        prof[pos] = cells
    out["position_means_hgen"] = prof
    out["extended_cells"] = extended_cells(lib, val, spans)
    out["positions_every_cell"] = positions_every_cell(add + base_scores, spans, tok)
    return out


ROWS = [("hf", "base", "HF bf16 base"), ("hf", "m0", "HF bf16 + M0"), ("hf", "zerob", "HF bf16 + zero-B"), ("hf", "m1", "HF bf16 + M1"),
        ("hf_v22", "m0_a128", "HF bf16 + M0 alpha128"), ("vllm_unq", "base", "vLLM unquantized bf16 base"),
        ("vllm_unq", "m0", "vLLM unquantized bf16 + M0"), ("vllm", "base", "vLLM QAT base"), ("vllm", "m0", "vLLM QAT + M0"),
        ("vllm", "zerob", "vLLM QAT + zero-B"), ("vllm", "m1", "vLLM QAT + M1"), ("vllm_nopc", "base", "vLLM QAT base, no prefix cache"),
        ("vllm_nopc", "m0_a128", "vLLM QAT + M0 alpha128")]
ORPHAN_ROWS = [(st, c, "%s %s" % ({"hf_v22": "HF bf16", "vllm_nopc": "vLLM QAT"}[st], c))
               for st in ("hf_v22", "vllm_nopc") for c in ("base", "m0", "zerob", "m1", "m0_a128")]


def extended_cells(lib, val, spans):
    """Mean over items (primary layout) of per-item text mean log-prob / accuracy, scaffold and control mean log-prob."""
    out = []
    for group, rows in (("H-gen", ROWS), ("T-fit", ROWS), ("T-orphan", ORPHAN_ROWS)):
        for st, c, label in rows:
            per = {sp: {k: v for k, v in val[(st, c, PRIMARY[st])][sp].items() if spans[k]["kind"] == group} for sp in lib.SPANS}
            if not per["text"]:
                continue
            mean = lambda sp, m: (sum(v[m] for v in per[sp].values()) / len(per[sp])) if per[sp] else None  # noqa: E731
            out.append({"group": group, "stack": st, "cell": c, "label": label, "n": len(per["text"]),
                        "text_mean_logprob": mean("text", "mean_logprob"), "text_accuracy": mean("text", "accuracy"),
                        "scaffold_mean_logprob": mean("scaffold", "mean_logprob"), "control_mean_logprob": mean("control", "mean_logprob")})
    return out


def positions_every_cell(scores, spans, tok):
    """Per cell (primary layout): mean log-prob, share ranked 1 and most frequent argmax at the thought close <channel|>
    (H-gen/T-fit target position 3; orphan position 0), `depth` (19) and the closing <|"|> (39); orphans also the terminator."""
    quote = '<|"|>'
    pos_sets = {"H-gen": {"<channel|>": 3, "depth": 19, quote: 39}, "T-fit": {"<channel|>": 3, "depth": 19, quote: 39}}
    agg = collections.defaultdict(lambda: [0.0, 0, 0, collections.Counter()])
    for s in scores:
        if s["layout"] != PRIMARY.get(s["stack"]):
            continue
        it = spans[s["item_id"]]
        pos = pos_sets.get(it["kind"]) or ({"<channel|>": 0, "terminator": len(it["target_ids"]) - 1} if it["kind"] == "T-orphan" else {})
        for name, k in pos.items():
            a = agg[(it["kind"], s["stack"], s["cell"], name)]
            a[0] += s["logprob"][k]
            a[1] += 1
            a[2] += s["rank"][k] == 1
            top = s["top5_ids"][k][0] if s["top5_ids"][k] else None
            a[3][tok.convert_ids_to_tokens(top) if (tok is not None and top is not None) else top] += 1
    return [{"group": g, "stack": st, "cell": c, "token": name, "mean_logprob": a[0] / a[1], "share_rank1": a[2] / a[1],
             "argmax": a[3].most_common(1)[0][0], "n": a[1]} for (g, st, c, name), a in sorted(agg.items())]


def fmt(x):
    return "-" if x is None else "%.3f" % x if isinstance(x, float) else str(x)


def md_lines(v):
    """Compact markdown tables for report.md."""
    L = ["MDE per span (0.5 x HF T-fit M0 effect): " + ", ".join("%s %s" % (k, fmt(x)) for k, x in v["mde"].items()), "",
         "Positive control m0_a128 - base (same session): " + v["positive_control"]["label"], "",
         "| stack / group / span | mean | 95% CI | large |", "|---|---|---|---|"]
    for k, e in v["positive_control"]["effects_m0_a128_minus_base"].items():
        if e:
            L.append("| %s | %s | [%s, %s] | %s |" % (k.replace("|", " / "), fmt(e["mean"]), fmt(e["ci95"][0]), fmt(e["ci95"][1]), e["large"]))
    L += ["", "Weights vs engine, H-gen, M0 (" + v["weights_vs_engine"]["label"] + "):", "",
          "| span | engine (vLLM unq - HF) | weights (vLLM QAT - vLLM unq) | total (vLLM QAT - HF) | MDE |", "|---|---|---|---|---|"]
    cell = lambda e: "-" if not e else "%s [%s, %s]%s" % (fmt(e["mean"]), fmt(e["ci95"][0]), fmt(e["ci95"][1]), " material" if e.get("large") else "")  # noqa: E731
    for sp, r in v["weights_vs_engine"]["spans"].items():
        L.append("| %s | %s | %s | %s | %s |" % (sp, cell(r.get("engine_unq_minus_hf")), cell(r.get("weights_qat_minus_unq")),
                                              cell(r.get("total_qat_minus_hf")), fmt(r["mde"])))
    L += ["", "Section G rule per axis (M0 row difference vs the base-row floor of the same axis, H-gen):", "",
          "| span | axis | mean | 95% CI | base floor p95 | items above | beyond noise |", "|---|---|---|---|---|---|---|"]
    for sp, r in v["weights_vs_engine"]["spans"].items():
        for ax, g in r.get("g_rule_m0_row_vs_base_floor", {}).items():
            L.append("| %s | %s | %s | [%s, %s] | %s | %s/%s | %s |" % (sp, ax, fmt(g["mean"]), fmt(g["ci95"][0]), fmt(g["ci95"][1]),
                     fmt(g["base_floor_p95"]), g["items_above_floor"], g["n_items"], g["beyond_noise"]))
    L += ["", "Extended cell table (mean over items; text = tool-call span, scaffold = empty-thought block or lone close, control = terminator):", "",
          "| group | cell | n | text mean log-prob | text accuracy | scaffold mean | control mean |", "|---|---|---|---|---|---|---|"]
    for r in v.get("extended_cells", []):
        L.append("| %s | %s | %d | %s | %s | %s | %s |" % (r["group"], r["label"], r["n"], fmt(r["text_mean_logprob"]), fmt(r["text_accuracy"]),
                                                       fmt(r["scaffold_mean_logprob"]), fmt(r["control_mean_logprob"])))
    esc = lambda s: str(s).replace(chr(10), chr(92) + "n").replace("|", chr(92) + "|")  # noqa: E731
    L += ["", "Position-level log-probs in every cell (primary layouts; mean log-prob / share ranked 1 / most frequent argmax):", "",
          "| group | stack | cell | token | mean log-prob | share rank 1 | argmax |", "|---|---|---|---|---|---|---|"]
    for r in v.get("positions_every_cell", []):
        L.append("| %s | %s | %s | `%s` | %s | %s | `%s` |" % (r["group"], r["stack"], r["cell"], esc(r["token"]), fmt(r["mean_logprob"]),
                                                          fmt(r["share_rank1"]), esc(r["argmax"])))
    L += ["", "Argmax flips (M0, H-gen): " + json.dumps(v["weights_vs_engine"]["argmax_flips_m0"]), "",
          "Position means, H-gen (target positions; 19 = `depth`, 39 = closing quote): ", "", "```", json.dumps(v["position_means_hgen"]), "```", "",
          "Orphan-start items (T-orphan), M0 - base: " + json.dumps({k: [x["label"], x["m0_minus_base"]] for k, x in v["orphan_items"]["m0_vs_base"].items()}), "",
          "```", json.dumps(v["orphan_items"]["rows"], indent=0), "```", "",
          "Repeat differences (max |delta| of per-item mean log-prob):", "", "```", json.dumps(v["repeat_differences"], indent=0), "```", "",
          "/tokenize check: " + json.dumps(v["tokenize_check"]), "", "Terminator census: " + json.dumps(v["terminator_census"]), "",
          v["declaration"], ""]
    return L
