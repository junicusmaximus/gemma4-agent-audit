#!/usr/bin/env python3
"""Compaction and context-loss audit of the sealed evaluation traces (read-only; CPU; tokenizer + Jinja only).
Locates ADK token-threshold compactions with the serialization audit's rule, rebuilds every served prompt with that
audit's reconstruction (imported) and classifies each compaction by exact match against vLLM's prompt_tokens. Call
identity: tools/mechanism_metrics_v032.py (imported); loop/submit: sealed per-task files. Descriptive; see README.md."""
import sys

sys.dont_write_bytecode = True  # importing the reference modules must not create __pycache__ next to them
import argparse  # noqa: E402
import collections  # noqa: E402
import contextlib  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
from multiprocessing import Pool  # noqa: E402

ROOT = "PROJECT_ROOT/gemma4"
sys.path.insert(0, ROOT + "/analysis/serialization-audit-boss-v046")
sys.path.append(ROOT + "/tools")
import audit  # noqa: E402  serialization audit v046 (sha256 5d26f4cd...)
import mechanism_metrics_v032 as MM  # noqa: E402  sealed producer (sha256 36042f6d...)

RUNS = tuple("v043_%s_s%d" % (a, s) for a in ("M0", "M1") for s in (1, 2, 3)) + tuple("a0_e4b_s%d" % s for s in (1, 2, 3))
SUBS = {"a0_e4b_s%d" % s: "e4b_budgetfit_s%d" % s for s in (1, 2, 3)}  # HOME_DIR/gemma4/e4b_batch.sh:15
SEALED = (ROOT + "/analysis/mask-control-v043/per_task.jsonl", ROOT + "/analysis/mechanism-v032/per_task_all48.jsonl")
PRIMARY40 = ROOT + "/analysis/mechanism-v032/primary40_ids.txt"
REFERENCE = ("v043_M0_s1", "rich_2725")  # serialization-audit-boss-v046/README.md:32-34
SEALED_KEYS = ("loop", "max_repeat", "submit_called", "n_tool_calls", "taxonomy")
CLASSES = ("empty_dropped", "summary_present", "no_compaction_event", "undetermined", "no_candidates")
APPLIED = ("empty_dropped", "summary_present", "undetermined")  # classes after which the regime changes
TOK = None


class AnyCount(int):
    """Equal to every count, so audit.serving_prompts keeps its no-visible-text form at that call (audit.py:176)."""
    __eq__, __hash__ = (lambda self, other: True), int.__hash__


def arm_of(run):
    return "A0" if run.startswith("a0_") else run.split("_")[1]


def crossings(calls):
    """0-based calls whose previous call reached the threshold: ADK's check before each call (audit.py:165-166)."""
    return [k for k in range(1, len(calls)) if (calls[k - 1]["prompt_tokens"] or 0) >= audit.THRESHOLD]


@contextlib.contextmanager
def compaction_plan(calls, plan):
    """While audit.serving_prompts runs, ADK's candidate selection follows a per-crossing plan: 'apply' (ADK as is),
    'skip' (no compaction event) or 'pre_nudge' (post-invocation run: the session before the trailing nudge)."""
    from google.adk.apps import compaction
    real, order = compaction._events_to_compact_for_token_threshold, iter(crossings(calls))

    def select(*, events, event_retention_size):
        how = plan.get(next(order), "apply")
        if how == "skip":
            return []
        while how == "pre_nudge" and events and events[-1].author == "user" and not events[-1].actions.compaction:
            events = events[:-1]
        return real(events=events, event_retention_size=event_retention_size)
    compaction._events_to_compact_for_token_threshold = select
    try:
        yield
    finally:
        compaction._events_to_compact_for_token_threshold = real


def served(tok, items, calls, sub, plan, upto=None, any_last=False):
    """audit.serving_prompts on calls[:upto] under a plan; any_last keeps the no-visible-text form at the last call."""
    sel = list(calls[:upto] if upto else calls)
    if any_last:
        sel[-1] = dict(sel[-1], prompt_tokens=AnyCount(0))
    with compaction_plan(sel, plan):
        return audit.serving_prompts(tok, items, sel, sub)[0]


def exact(lens, vllm, k):
    return vllm[k] is not None and lens[k] == vllm[k]


def delta_ok(lens, vllm, k):
    """The served prompt grew by what vLLM recorded since the previous call (audit.py:273-275)."""
    return k > 0 and None not in (vllm[k], vllm[k - 1]) and lens[k] - lens[k - 1] == vllm[k] - vllm[k - 1]


def classify_event(dropped_exact, skip_consistent, summary_tokens, reliable_before):
    """Class of one located compaction; first match wins (README, 'Classification')."""
    if dropped_exact:
        return "empty_dropped"
    if skip_consistent:
        return "no_compaction_event"
    if reliable_before and summary_tokens is not None and summary_tokens > 0:
        return "summary_present"
    return "undetermined"


def nudges_between(items, calls, k):
    return sum(items[i][0] == "user" for i in range(calls[k - 1]["item"] + 1, calls[k]["item"]))


def classify_compactions(tok, items, calls, sub):
    """Walk the crossings in order; each decision fixes the session state that later crossings start from."""
    vllm, plan, events, explained = [c["prompt_tokens"] for c in calls], {}, [], set()
    for k in crossings(calls):
        base = served(tok, items, calls, sub, {**plan, k: "apply"}, k + 1)
        lens, form = [len(s["ids"]) for s in base], base[k]["compacted"]
        ev = {"call": k + 1, "prev_prompt_tokens": vllm[k - 1], "vllm_prompt_tokens": vllm[k], "summary_text_in_trace": False,
              "nudges_before": nudges_between(items, calls, k), "timing": "request_processor"}
        if form is None:  # at most event_retention_size candidates: ADK appends no event either
            events.append(dict(ev, **{"class": "no_candidates"}))
            continue
        reliable = all(exact(lens, vllm, j) or delta_ok(lens, vllm, j) or j in explained for j in range(k))
        dropped = lens[k] if form == "no_visible_text" else len(
            served(tok, items, calls, sub, {**plan, k: "apply"}, k + 1, True)[k]["ids"])
        ev.update(dropped_form_tokens=dropped, reliable_before=reliable)
        skip_ok, how = False, "apply"
        if dropped != vllm[k]:
            skip = [len(s["ids"]) for s in served(tok, items, calls, sub, {**plan, k: "skip"}, k + 1)]
            skip_ok, ev["no_compaction_form_tokens"] = exact(skip, vllm, k) or delta_ok(skip, vllm, k), skip[k]
            if not skip_ok and ev["nudges_before"]:
                pre = len(served(tok, items, calls, sub, {**plan, k: "pre_nudge"}, k + 1, True)[k]["ids"])
                ev["dropped_form_tokens_post_invocation"] = pre
                if pre == vllm[k]:
                    how, ev["timing"] = "pre_nudge", "post_invocation"
        if form == "placeholder_text" and vllm[k] is not None:
            ev["placeholder_form_tokens"] = lens[k]
            ev["inferred_summary_tokens"] = vllm[k] - lens[k] + len(audit.enc(tok, audit.SUMMARY))  # audit.py:276-277
        ev["class"] = classify_event(dropped == vllm[k] or how == "pre_nudge", skip_ok,
                                     ev.get("inferred_summary_tokens"), reliable)
        plan[k] = "skip" if ev["class"] == "no_compaction_event" else how
        explained.update([k] if ev["class"] in ("summary_present", "undetermined") else [])
        events.append(ev)
    return plan, events


def user_texts(messages):
    """Text of every user-role message of one served request (lite_llm._get_completion_inputs form)."""
    out = []
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            out.append(c if isinstance(c, str) else "".join(p.get("text") or "" for p in c or [] if isinstance(p, dict)))
    return out


def tool_call_positions(trace):
    """(LLM call number, MM identity, function name) per parsed tool call, in MM.flatten_calls order. A step
    without metrics belongs to the LLM call of the step before it (audit.py:103-107)."""
    out, k = [], 0
    for st in trace["steps"]:
        if not isinstance(st, dict) or st.get("source") != "agent":
            continue
        k += st.get("metrics") is not None
        out += [(max(k, 1), MM.tool_call_identity(c), c.get("function_name")) for c in MM.parsed_tool_calls(st)]
    return out


def loop_onset(positions, n=MM.DEFAULT_LOOP_THRESHOLD):
    """The first identity to reach n occurrences (the sealed loop rule, MM.repeat_stats): calls of its first and
    n-th occurrence; plus the longest run of consecutive identical calls and the call where it starts."""
    seen, first, onset = collections.Counter(), {}, None
    for k, ident, _ in positions:
        first.setdefault(ident, k)
        seen[ident] += 1
        if onset is None and seen[ident] == n:
            onset = {"function": ident[0], "first_call": first[ident], "nth_call": k}
    run, best = 0, (0, None)
    for i, (k, ident, _) in enumerate(positions):
        run = run + 1 if i and ident == positions[i - 1][1] else 1
        best = (run, positions[i - run + 1][0]) if run > best[0] else best
    return {"max_repeat": max(seen.values(), default=0), "onset": onset, "longest_run": list(best)}


def order(onset_call, event_call):
    """'after' when the onset call was itself served after the event, else 'before'; None if either is absent."""
    if onset_call is None or event_call is None:
        return None
    return "after" if onset_call >= event_call else "before"


def init():
    global TOK
    TOK = audit.load_tok(audit.SERVED_DIR)


def audit_trace(job):
    run, path, sealed, primary = job
    tok, trace = TOK, json.load(open(path))
    suffix, items, calls, bad = max([(x,) + audit.trace_items(trace, x) for x in ("", "\n")],  # audit.py:264-265
                                    key=lambda v: sum(audit.emitted_residual(tok, c) == 0 for c in v[2]))
    sub = ROOT + "/subs/" + SUBS.get(run, run)
    plan, events = classify_compactions(tok, items, calls, sub)
    final = served(tok, items, calls, sub, plan)
    vllm, lens = [c["prompt_tokens"] for c in calls], [len(s["ids"]) for s in final]
    task = next(s["message"] for s in trace["steps"] if (s.get("extra") or {}).get("event_type") == "task_prompt")
    applied = {e["call"]: e["class"] for e in events if e["class"] in APPLIED}
    rows, regime = [], "before_compaction"
    for k, s in enumerate(final):
        regime = "after_" + applied[k + 1] if k + 1 in applied else regime
        users = user_texts(s["messages"])
        rows.append({"call": k + 1, "step": calls[k]["step"], "vllm": vllm[k], "served": lens[k],
                     "exact": exact(lens, vllm, k), "delta_ok": delta_ok(lens, vllm, k), "regime": regime,
                     "user_turns": len(users), "task_present": any(task in u for u in users)})
    marks = sorted(applied) + [len(rows) + 1]
    for e in events:
        if e["class"] in APPLIED:
            seg = rows[e["call"]:next(m for m in marks if m > e["call"]) - 1]
            e["post_deltas_ok"] = [sum(r["delta_ok"] for r in seg), len(seg)]
    loss = [r for r in rows if r["regime"] == "after_empty_dropped"]
    undet, n = sum(r["regime"] == "after_undetermined" for r in rows), len(rows)
    first_loss, first_cmp = (loss[0]["call"] if loss else None), (min(applied) if applied else None)
    pos = tool_call_positions(trace)
    lo = loop_onset(pos)
    onset = (lo["onset"] or {}).get("first_call") if sealed["loop"] else None
    submits = [k for k, _, name in pos if name == "submit_patch"]
    return {"run": run, "arm": arm_of(run), "seed": int(run[-1]), "instance_id": os.path.basename(path)[6:-5],
            "primary40": primary, "trace": path, "trace_sha256": audit.sha(path), "submission": sub,
            "thought_suffix": suffix, "unmapped": dict(bad), "calls": n, "crossings": len(crossings(calls)),
            "calls_with_prompt_tokens": sum(v is not None for v in vllm),
            "compactions": events, "first_compaction_call": first_cmp, "first_context_loss_call": first_loss,
            "context_loss_calls": len(loss), "context_loss_calls_exact": sum(r["exact"] for r in loss),
            "context_loss_calls_without_user_turn": sum(r["user_turns"] == 0 for r in loss),
            "undetermined_regime_calls": undet,
            "context_loss_share": None if undet or not n else round(len(loss) / n, 4),
            "context_loss_share_bounds": [round(len(loss) / n, 4), round((len(loss) + undet) / n, 4)] if n else None,
            "calls_without_raw_task_prompt": sum(not r["task_present"] for r in rows),
            "unexplained_mismatch_calls": [r["call"] for r in rows if not (r["exact"] or r["delta_ok"])
                                           and applied.get(r["call"]) not in ("summary_present", "undetermined")],
            "sealed": {k: sealed[k] for k in SEALED_KEYS}, "recomputed_max_repeat": lo["max_repeat"],
            "loop_onset": lo["onset"], "longest_identical_run": lo["longest_run"],
            "loop_onset_vs_first_compaction": order(onset, first_cmp),
            "loop_onset_vs_context_loss": order(onset, first_loss),
            "loop_nth_vs_context_loss": order((lo["onset"] or {}).get("nth_call") if sealed["loop"] else None, first_loss),
            "submit_calls": submits,
            "submit_after_context_loss": None if first_loss is None else any(k >= first_loss for k in submits),
            "rows": rows}


def aggregate(recs):
    """Counts with denominators for one population; the mean share only over traces whose regimes are determined."""
    submitted = lambda rs: [sum(bool(r["sealed"]["submit_called"]) for r in rs), len(rs)]  # noqa: E731
    ev = collections.Counter(e["class"] for r in recs for e in r["compactions"])
    loss = [r for r in recs if r["first_context_loss_call"] is not None]
    clean = [r for r in recs if r["first_context_loss_call"] is None and not r["undetermined_regime_calls"]]
    shares = [r["context_loss_share"] for r in recs if r["context_loss_share"] is not None]
    loops = [r for r in recs if r["sealed"]["loop"]]
    return {"traces": len(recs), "traces_with_crossing": sum(r["crossings"] > 0 for r in recs),
            "traces_with_compaction": sum(r["first_compaction_call"] is not None for r in recs),
            "events": {c: ev[c] for c in CLASSES}, "traces_with_context_loss": len(loss),
            "traces_with_undetermined_regime": sum(r["undetermined_regime_calls"] > 0 for r in recs),
            "calls": sum(r["calls"] for r in recs), "context_loss_calls": sum(r["context_loss_calls"] for r in recs),
            "calls_without_raw_task_prompt": sum(r["calls_without_raw_task_prompt"] for r in recs),
            "mean_context_loss_share": round(statistics.mean(shares), 4) if shares else None,
            "mean_share_traces": len(shares), "loops": len(loops),
            "loops_with_compaction": sum(r["first_compaction_call"] is not None for r in loops),
            "loop_onset_after_first_compaction": sum(r["loop_onset_vs_first_compaction"] == "after" for r in loops),
            "loops_with_context_loss": sum(r["first_context_loss_call"] is not None for r in loops),
            "loop_onset_after_context_loss": sum(r["loop_onset_vs_context_loss"] == "after" for r in loops),
            "loop_nth_after_context_loss": sum(r["loop_nth_vs_context_loss"] == "after" for r in loops),
            "submit_with_context_loss": submitted(loss), "submit_without_context_loss": submitted(clean),
            "submit_after_context_loss": [sum(bool(r["submit_after_context_loss"]) for r in loss), len(loss)]}


def groups(recs):
    """Aggregates per arm x seed, then per arm over its seeds."""
    out = {}
    for arm in ("M0", "M1", "A0"):
        rs = [r for r in recs if r["arm"] == arm]
        for seed in sorted({r["seed"] for r in rs}):
            out["%s s%d" % (arm, seed)] = aggregate([r for r in rs if r["seed"] == seed])
        if rs:
            out["%s all" % arm] = aggregate(rs)
    return out


def reproduction(recs):
    """The serialization audit's finding on its M0 trace (its README.md:32-34; report.md rows of calls 6-48)."""
    r = next((x for x in recs if (x["run"], x["instance_id"]) == REFERENCE), None)
    if r is None:
        return {"available": False}
    first = next((e for e in r["compactions"] if e["class"] in APPLIED), {})
    lost = [x for x in r["rows"] if x["regime"] == "after_empty_dropped"]
    got = {"first_compaction_call": first.get("call"), "class": first.get("class"),
           "vllm_prompt_tokens": first.get("vllm_prompt_tokens"), "dropped_form_tokens": first.get("dropped_form_tokens"),
           "context_loss_calls": [lost[0]["call"], lost[-1]["call"], len(lost)] if lost else None,
           "user_turns_in_those_calls": sorted({x["user_turns"] for x in lost})}
    want = {"first_compaction_call": 6, "class": "empty_dropped", "vllm_prompt_tokens": 3731,
            "dropped_form_tokens": 3731, "context_loss_calls": [6, 47, 42], "user_turns_in_those_calls": [0]}
    return {"available": True, "expected": want, "observed": got, "reproduced": got == want}


def cross_checks(recs):
    rows = [x for r in recs for x in r["rows"]]
    return {"max_repeat_sealed_equals_recomputed": [sum(r["sealed"]["max_repeat"] == r["recomputed_max_repeat"]
                                                        for r in recs), len(recs)],
            "submit_sealed_equals_trace": [sum(bool(r["sealed"]["submit_called"]) == bool(r["submit_calls"])
                                               for r in recs), len(recs)],
            "calls_exact": [sum(x["exact"] for x in rows), len(rows)],
            "calls_exact_or_delta": [sum(x["exact"] or x["delta_ok"] for x in rows), len(rows)],
            "unexplained_mismatch_calls": [sum(len(r["unexplained_mismatch_calls"]) for r in recs), len(rows)],
            "unexplained_mismatch_where": {"%s/%s" % (r["run"], r["instance_id"]): r["unexplained_mismatch_calls"]
                                           for r in recs if r["unexplained_mismatch_calls"]},
            "traces_with_unmapped_steps": [sum(bool(r["unmapped"]) for r in recs), len(recs)]}


def evidence(recs):
    """Per arm: the token evidence behind the event classes."""
    out = {}
    for arm in ("M0", "M1", "A0"):
        evs = [e for r in recs if r["arm"] == arm for e in r["compactions"]]
        present = [e for e in evs if e["class"] == "summary_present"]
        size = sorted(e["inferred_summary_tokens"] for e in present)
        gap = sorted(e["vllm_prompt_tokens"] - e["dropped_form_tokens"] for e in present)
        post, cuts = [e["post_deltas_ok"] for e in evs if "post_deltas_ok" in e], ((50, "<50"), (500, "50-499"),
                                                                                    (2000, "500-1999"), (1e9, ">=2000"))
        bins = collections.Counter(next(lab for lim, lab in cuts if s < lim) for s in size)
        out[arm] = {"events": len(evs), "classes": dict(collections.Counter(e["class"] for e in evs)),
                    "inferred_summary_tokens_min_median_max": [size[0], statistics.median(size), size[-1]] if size else None,
                    "inferred_summary_tokens_bins": {lab: bins[lab] for _, lab in cuts},
                    "prompt_after_compaction_above_uncompacted_form": sum(
                        e["vllm_prompt_tokens"] > e["no_compaction_form_tokens"] for e in present),
                    "vllm_minus_dropped_form_min": gap[0] if gap else None,
                    "events_with_nudge_before": sum(e["nudges_before"] > 0 for e in evs),
                    "post_invocation_timing": sum(e["timing"] == "post_invocation" for e in evs),
                    "post_compaction_deltas_ok": [sum(p[0] for p in post), sum(p[1] for p in post)]}
    return out


def case_rows(recs):
    return [{"run": r["run"], "task": r["instance_id"], "primary40": r["primary40"], "calls": r["calls"],
             "first_loss_call": r["first_context_loss_call"], "loss_calls": r["context_loss_calls"],
             "loss_calls_exact": r["context_loss_calls_exact"],
             "loss_calls_without_user_turn": r["context_loss_calls_without_user_turn"],
             "loop": r["sealed"]["loop"], "max_repeat": r["sealed"]["max_repeat"],
             "onset_first_nth": [(r["loop_onset"] or {}).get("first_call"), (r["loop_onset"] or {}).get("nth_call")],
             "onset_vs_loss": r["loop_onset_vs_context_loss"], "submit": r["sealed"]["submit_called"],
             "submit_after_loss": r["submit_after_context_loss"]} for r in recs if r["first_context_loss_call"]]


def undetermined_rows(recs):
    return [{"run": r["run"], "task": r["instance_id"], **{k: e.get(k) for k in (
        "call", "vllm_prompt_tokens", "dropped_form_tokens", "inferred_summary_tokens", "reliable_before")},
        "unexplained_mismatch_calls": r["unexplained_mismatch_calls"]}
        for r in recs for e in r["compactions"] if e["class"] == "undetermined"]


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def write_md(rep, path):
    rc, L = rep["reproduction_check"], ["# Compaction and context-loss audit (boss v050)", "",
                                        "Generated by `compaction_audit.py` from `report.json`. Descriptive counts only; "
                                        "method and caveats in `README.md`.", ""]
    L += ["## Reproduction check (v043_M0_s1, rich_2725)", "", "Reproduced: **%s**. Observed: `%s`." % (
        rc.get("reproduced"), json.dumps(rc.get("observed"))), ""]
    f = "%d/%d".__mod__
    head = ["arm", "seed", "traces", "with compaction", "with context loss", "undetermined regime", "loss calls / calls",
            "mean loss share (traces)", "calls without raw task prompt", "loops, onset after loss / loops",
            "loops, onset after 1st compaction / loops", "submit, with loss", "submit, without loss"]
    for pop, gs in rep["aggregates"].items():
        L += ["## Aggregates, %s" % pop, ""] + table(head, [g.split() + [
            a["traces"], a["traces_with_compaction"], a["traces_with_context_loss"], a["traces_with_undetermined_regime"],
            f((a["context_loss_calls"], a["calls"])), "%s (%d)" % (a["mean_context_loss_share"], a["mean_share_traces"]),
            f((a["calls_without_raw_task_prompt"], a["calls"])),
            f((a["loop_onset_after_context_loss"], a["loops"])), f((a["loop_onset_after_first_compaction"], a["loops"])),
            f(tuple(a["submit_with_context_loss"])), f(tuple(a["submit_without_context_loss"]))]
            for g, a in gs.items()]) + [""]
    L += ["## Compaction events by class, all48", ""] + table(["arm", "seed", "crossings"] + list(CLASSES), [
        g.split() + [sum(a["events"].values())] + [a["events"][c] for c in CLASSES]
        for g, a in rep["aggregates"]["all48"].items()]) + [""]
    L += ["## Evidence per arm", "", "```", json.dumps(rep["evidence"], indent=1), "```", ""]
    L += ["## Context-loss cases (all48)", ""] + table(
        ["run", "task", "primary40", "calls", "first loss call", "loss calls (exact; no user turn)", "loop",
         "max_repeat", "loop onset 1st / 5th call", "onset vs loss", "submit", "submit after loss"],
        [[c["run"], c["task"], c["primary40"], c["calls"], c["first_loss_call"], "%d (%d; %d)" % (
            c["loss_calls"], c["loss_calls_exact"], c["loss_calls_without_user_turn"]), c["loop"], c["max_repeat"],
          "%s / %s" % tuple(c["onset_first_nth"]), c["onset_vs_loss"], c["submit"], c["submit_after_loss"]]
         for c in rep["context_loss_cases"]]) + [""]
    L += ["## Undetermined compaction events", ""] + (table(list(rep["undetermined_events"][0]), [
        list(u.values()) for u in rep["undetermined_events"]]) if rep["undetermined_events"] else ["None."]) + [""]
    L += ["## Cross-checks", "", "```", json.dumps(rep["cross_checks"], indent=1), "```", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def load_jobs(runs):
    sealed = {}
    for p in SEALED:
        with open(p) as fh:
            for row in map(json.loads, fh):
                sealed[(row["run"], row["instance_id"])] = {k: row.get(k) for k in SEALED_KEYS}
    with open(PRIMARY40) as fh:
        primary = {x.strip() for x in fh if x.strip()}
    if len(primary) != 40:
        raise SystemExit("primary40_ids.txt: expected 40 ids, found %d" % len(primary))
    jobs = []
    for run in runs:
        paths = sorted(glob.glob("%s/results/%s/traces/trace_*.json" % (ROOT, run)))
        if len(paths) != 48:
            raise SystemExit("%s: expected 48 traces, found %d" % (run, len(paths)))
        for p in paths:
            key = (run, os.path.basename(p)[6:-5])
            if key not in sealed:
                raise SystemExit("no sealed per-task row for %s %s" % key)
            jobs.append((run, p, sealed[key], key[1] in primary))
    return jobs


def inputs(runs):
    import importlib.metadata as md
    from google.adk import apps
    adk = os.path.dirname(os.path.dirname(apps.__file__))
    files = [audit.__file__, MM.__file__, *SEALED, PRIMARY40, ROOT + "/run_eval.py"] + sorted(
        {"%s/subs/%s/agent.yaml" % (ROOT, SUBS.get(r, r)) for r in runs})
    adk_files = ["apps/compaction.py", "apps/llm_event_summarizer.py", "flows/llm_flows/compaction.py",
                 "flows/llm_flows/contents.py", "runners.py", "models/lite_llm.py"]
    return {"sha256": {os.path.relpath(p, ROOT): audit.sha(p) for p in files},
            "adk_sha256": {p: audit.sha(os.path.join(adk, p)) for p in adk_files},
            "versions": {p: md.version(p) for p in ("google-adk", "transformers", "litellm")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compaction and context-loss audit of evaluation traces (read-only).")
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    jobs = load_jobs(a.runs)
    with Pool(a.workers, initializer=init) as pool:
        recs = pool.map(audit_trace, jobs, chunksize=1)
    rep = {"generation": "compaction-audit-boss-v050", "threshold": audit.THRESHOLD, "retention": audit.RETENTION,
           "loop_threshold": MM.DEFAULT_LOOP_THRESHOLD, "runs": a.runs, "inputs": inputs(a.runs),
           "reproduction_check": reproduction(recs), "cross_checks": cross_checks(recs), "evidence": evidence(recs),
           "aggregates": {"all48": groups(recs), "primary40": groups([r for r in recs if r["primary40"]])},
           "context_loss_cases": case_rows(recs), "undetermined_events": undetermined_rows(recs)}
    with open(os.path.join(a.out, "per_trace.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    write_md(rep, os.path.join(a.out, "report.md"))
    print(json.dumps({"reproduction_check": rep["reproduction_check"], "cross_checks": rep["cross_checks"]}))


if __name__ == "__main__":
    main()
