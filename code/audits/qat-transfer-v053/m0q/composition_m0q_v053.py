#!/usr/bin/env python3
"""Composition check of M0q (train env, CPU). Criteria fixed 2026-11-05 ~18:4x UTC while M0q was training, before any
M0q score existed (prereg section 3, its re-freeze acceptance, the lead's pre-authorization); E = M0q - base on the
text span, per item, within one stack:
  C1  H-gen diff-of-diffs mean(E_vLLM - E_HF) over the 48 items within +-0.10 nats/token;
  C2  per item |E_vLLM - E_HF| <= max(0.10, 2 x the item's Stage-1a base gap) for >= 45/48 H-gen and 5/5 T-fit items;
  C3  opener accuracy gap (target span, mean over items) HF - vLLM <= 0.01, on H-gen and on T-fit;
  C4  greedy step-1 generation of M0q on hgen:rich_2725: no literal "-max 2", and a parsed run_command call whose
      command contains "-maxdepth".
PASS = C1 and C2 and C3 and C4. Informational: M0q effects next to M0's (HF-bf16, HF-dq, vLLM-QAT), the `depth` and
closing-quote positions, the same-session sealed-M0 rows and its greedy generation (sensitivity control), and the
repeat of the vLLM base against Stage 2. Writes composition.json; exit 0 PASS, 30 FAIL."""
import importlib.util
import json
import re
import sys

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
PARENT = ROOT + "/analysis/qat-transfer-v053"
SESSION = PARENT + "/m0q"
spec = importlib.util.spec_from_file_location("s3_analyze_v053", PARENT + "/s3_analyze_v053.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
H, T = an.HGEN, an.TFIT
TOOL_CALL = re.compile(r'<\|tool_call>call:run_command\{command:<\|"\|>(.*?)<\|"\|>\}<tool_call\|>', re.S)


def generation_check(row):
    text = row.get("text") or ""
    m = TOOL_CALL.search(text)
    cmd = m.group(1) if m else None
    return {"model": row["model"], "finish_reason": row.get("finish_reason"),
            "completion_tokens": (row.get("usage") or {}).get("completion_tokens"),
            "literal_max_2": text.count("-max 2"), "max_not_followed_by_depth": len(re.findall(r"-max(?!depth)", text)),
            "tool_call_parsed": m is not None, "command": cmd, "maxdepth_in_command": bool(cmd and "-maxdepth" in cmd),
            "text_head": text[:400], "pass": text.count("-max 2") == 0 and bool(cmd and "-maxdepth" in cmd)}


def criteria(hf, vq, spans, gaps, gens):
    e_hf = {k: an.effects(hf, spans, "hf_dequant", "m0q", "single", k) for k in (H, T)}
    e_vq = {k: an.effects(vq, spans, "vllm_qat_nopc", "m0q", "sequential", k) for k in (H, T)}
    keys = {k: sorted(set(e_hf[k]) & set(e_vq[k])) for k in (H, T)}
    dd = [e_vq[H][k] - e_hf[H][k] for k in keys[H]]
    c1 = dict(an.boot(dd) if dd else {"n": 0, "mean": None}, rule="|mean(E_vLLM - E_HF)| <= 0.10 over 48 H-gen items",
              **{"pass": len(dd) == 48 and abs(an.mean(dd)) <= 0.10})
    within = {g: {k: abs(e_vq[g][k] - e_hf[g][k]) <= max(0.10, 2 * gaps[k]) for k in keys[g]} for g in (H, T)}
    c2 = {"hgen_within": sum(within[H].values()), "hgen_n": len(within[H]), "tfit_within": sum(within[T].values()),
          "tfit_n": len(within[T]), "max_abs_diff_hgen": max(abs(x) for x in dd) if dd else None,
          "pass": len(within[H]) == 48 and sum(within[H].values()) >= 45 and len(within[T]) == 5 and all(within[T].values())}
    acc = {}
    for g in (H, T):
        a = an.per_item(hf, spans, "hf_dequant", "m0q", "single", g, "target")
        b = an.per_item(vq, spans, "vllm_qat_nopc", "m0q", "sequential", g, "target")
        acc[g] = {"hf": an.mean(x["accuracy"] for x in a.values()), "vllm": an.mean(x["accuracy"] for x in b.values())}
        acc[g]["gap_hf_minus_vllm"] = acc[g]["hf"] - acc[g]["vllm"]
    c3 = dict(acc, **{"pass": all(acc[g]["gap_hf_minus_vllm"] <= 0.01 for g in (H, T))})
    c4 = generation_check(gens["m0q"]) if "m0q" in gens else {"pass": False, "note": "no generation row"}
    rules = {"C1_diff_of_diffs": c1, "C2_items_within_bound": c2, "C3_opener_accuracy_gap": c3, "C4_greedy_step1": c4}
    failing = [k for k, v in rules.items() if not v["pass"]]
    return rules, ("PASS" if not failing else "FAIL: " + ", ".join(failing)), e_hf, e_vq


def informational(hf, vq, spans, e_hf, e_vq, gens):
    s1 = an.read_json(PARENT + "/stage1.json")["transfer_1b"]
    m0_vq = an.effects(vq, spans, "vllm_qat_nopc", "m0", "sequential", H)
    return {"effects_text_mean": {g: {"m0q_hf_dq": an.mean(e_hf[g].values()), "m0q_vllm_qat": an.mean(e_vq[g].values())} for g in (H, T)},
            "m0_reference_hgen_text": {"hf_bf16_v048": s1["effect_hf_bf16_v048"]["mean"], "hf_dq_stage1": s1["effect_hf_dq"]["mean"],
                                       "vllm_qat_v048": s1["effect_vllm_qat_v048"]["mean"],
                                       "vllm_qat_this_session": an.mean(m0_vq.values())},
            "m0q_hgen_text_effect_over_m0_hf_bf16": an.mean(e_hf[H].values()) / s1["effect_hf_bf16_v048"]["mean"],
            "positions": {g: an.position_check(hf, vq, spans, "hf_dequant", "m0q", g) for g in (H, T)},
            "vllm_base_repeat_vs_stage2": an.repeat(vq, "vllm_qat_nopc", "sequential", an.load(PARENT + "/vllm_scores_v053.jsonl"),
                                                    "vllm_qat_nopc", "sequential", spans),
            "m0_greedy_control": generation_check(gens["m0"]) if "m0" in gens else None}


def main():
    spans = {r["item_id"]: r for r in an.load(an.LEDGER)}
    gaps = an.read_json(PARENT + "/stage1.json")["item_gaps_1a"]
    hf, vq = an.load(SESSION + "/hf_scores_m0q.jsonl"), an.load(SESSION + "/vllm_scores_v053.jsonl")
    gens = {r["model"]: r for r in an.load(SESSION + "/generations.jsonl")}
    rules, verdict, e_hf, e_vq = criteria(hf, vq, spans, gaps, gens)
    res = {"verdict": verdict, "rules": rules, "informational": informational(hf, vq, spans, e_hf, e_vq, gens)}
    an.write_new(SESSION + "/composition.json", res)
    c = rules
    print(json.dumps({"verdict": verdict, "C1_mean": c["C1_diff_of_diffs"]["mean"], "C2": [c["C2_items_within_bound"]["hgen_within"],
                      c["C2_items_within_bound"]["tfit_within"]], "C3": {g: c["C3_opener_accuracy_gap"][g]["gap_hf_minus_vllm"] for g in (H, T)},
                      "C4": {k: c["C4_greedy_step1"].get(k) for k in ("literal_max_2", "maxdepth_in_command", "finish_reason")}}))
    return 0 if verdict == "PASS" else 30


if __name__ == "__main__":
    sys.exit(main())
