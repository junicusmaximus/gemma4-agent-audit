#!/usr/bin/env python3
"""Train-vs-serve serialization audit of evaluation traces (read-only; CPU; tokenizer + jinja only).
(a) serving messages: ADK session rebuilt from the ATIF trace, run through ADK's own builders offline; (b) vLLM 0.19.1
message normalisation + served chat template; (c) the same conversation packed by sft/build_sft.py's own build_chains +
chunk_chain (kit-renderer prefixes); (d) piece-level diff of (b) against (c) per LLM call, one category per span."""
import argparse
import asyncio
import collections
import copy
import difflib
import hashlib
import itertools
import json
import os
import sys

ROOT = "PROJECT_ROOT/gemma4"
SERVED_DIR = ROOT + "/models/gemma-4-e4b-it-qat-w4a16-ct"
BASE_DIR = ROOT + "/models/gemma-4-e4b-it"
KIT_NATIVE = "HOME_DIR/gemma4/external/gemma-4-serving-and-lora-verification-kit/lora_native_resolved.jsonl"
TRAJ = ROOT + "/sft/data/trajectories.jsonl"
M0_DATA = ROOT + "/sft/data/mask_control_v035/M0.jsonl"
SERVER_READY = ROOT + "/migration/mask-evaluations-v043/evaluation/server-ready.json"
FORCED = "<|channel>thought\n"
OPENER, EMPTY = [100, 45518, 107], [100, 45518, 107, 101]
STRUCT = {2, 105, 106, 100, 101, 48, 49, 50, 51, 98}  # bos, turn, channel, tool_call, tool_response, think
SPECIAL = ["<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
           "<|tool_response>", "<tool_response|>", "<|think|>"]
THRESHOLD, RETENTION = 14336, 5  # run_eval.py:50-55 COMPACTION_SETTINGS (token_threshold, event_retention_size)
MAX_LEN = 16384  # sft/README.md:16 --max-seq-len
SUMMARY = "[compaction summary: text not recorded in the trace]"
CATS = ["empty_thought_block", "arg_key_order", "newline_before_channel_close", "tool_response_wrapper",
        "system_header_think", "generation_prompt_opener", "other"]

def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_tok(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)

def enc(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]

def vllm_conversation(messages):
    """vllm/entrypoints/chat_utils.py:1503-1552 ('openai' content format) and _postprocess_messages 1555-1578."""
    conv = []
    for m in messages:
        content = m.get("content")
        parts = [] if content is None else [{"type": "text", "text": content}] if isinstance(content, str) else content
        out = {"role": m["role"], "content": [{"type": "text", "text": p["text"]} for p in parts if p.get("type") == "text"]}
        if m["role"] == "assistant" and m.get("tool_calls") is not None:
            out["tool_calls"] = copy.deepcopy(list(m["tool_calls"]))
        if m["role"] == "assistant" and m.get("reasoning") is not None:  # only 'reasoning' is read (1512, 1537-1540)
            out["reasoning"] = out["reasoning_content"] = m["reasoning"]
        if m["role"] == "tool" and "tool_call_id" in m:
            out["tool_call_id"] = m["tool_call_id"]
        out.update({"name": m["name"]} if isinstance(m.get("name"), str) else {})
        conv.append(out)
    for out in conv:
        calls = out.get("tool_calls") if out["role"] == "assistant" else None
        if calls is not None and not calls:
            out.pop("tool_calls")
        for c in calls or []:
            a = c["function"].get("arguments")
            c["function"]["arguments"] = (a if isinstance(a, (dict, list)) else json.loads(a)) if a else {}
    return conv

def render(tok, conv, tools, kwargs):
    text = tok.apply_chat_template(conv, tools=tools or None, add_generation_prompt=True, tokenize=False, **kwargs)
    return text, enc(tok, text)

def fmt(v):
    """Sampled tool-argument value (template format_argument form, keys kept in sampled order)."""
    if isinstance(v, dict):
        return "{" + ",".join("%s:%s" % (k, fmt(x)) for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    return '<|"|>' + v + '<|"|>' if isinstance(v, str) else ("true" if v else "false") if isinstance(v, bool) else str(v)

def action_text(calls):
    return "".join("<|tool_call>call:%s{%s}<tool_call|>" % (n, ",".join("%s:%s" % (k, fmt(v)) for k, v in a.items()))
                   for n, a in calls)

def trace_items(trace, suffix=""):
    """Session contents + one record per LLM call (agent_runner.py:700-711, 1021-1024; atif_converter.py:42-140; trace.py:207-252)."""
    from google.genai import types
    items, calls, bad = [], [], collections.Counter()
    for st in trace["steps"]:
        src, ev, text = st.get("source"), (st.get("extra") or {}).get("event_type"), st.get("message") or ""
        if (src, ev) in (("user", "task_prompt"), ("system", "continuation_nudge")):
            items.append(("user", types.Content(role="user", parts=[types.Part(text=text)])))
        if src != "agent" or ev not in (None, "thinking", "final", "text"):
            bad["unmapped_step_%s_%s" % (src, ev)] += (src, ev) not in (
                ("user", "task_prompt"), ("system", "continuation_nudge"), ("system", "system_instruction"))
            continue
        text += suffix if ev == "thinking" else ""  # the trace strips thoughts (trace.py:249); suffix restores a trailing newline
        parts = [types.Part(text=text, thought=True)] if ev == "thinking" else [types.Part(text=text)] if ev else []
        fcs = [("chatcmpl-tool-%03d-%s" % (len(calls), tc["tool_call_id"]), tc["function_name"], tc.get("arguments") or {})
               for tc in st.get("tool_calls") or []]
        parts += [types.Part(function_call=types.FunctionCall(id=i, name=n, args=a)) for i, n, a in fcs]
        if st.get("metrics") is None:  # usage is attached once per LLM call (trace.py:288-296): same call
            if not (calls and items[-1][0] == "model"):
                bad["agent_step_without_metrics"] += 1
                continue
            items[-1][1].parts.extend(parts)
        else:
            items.append(("model", types.Content(role="model", parts=parts)))
            calls.append({"step": st["step_id"], "item": len(items) - 1, "prompt_tokens": st["metrics"].get("prompt_tokens"),
                          "completion_tokens": st["metrics"].get("completion_tokens"), "thought": "", "text": "", "calls": []})
        calls[-1]["thought" if ev == "thinking" else "text"] += text if ev else ""
        calls[-1]["calls"] += [(n, a) for _, n, a in fcs]
        obs = st.get("observation")
        bad["tool_call_without_single_observation"] += bool(fcs) and (obs is None or len(fcs) != 1)
        if not fcs or obs is None:
            continue
        try:
            resp = json.loads(obs["content"])  # trace.py:219 json.dumps(unwrap_tool_response(fr)), ensure_ascii=True
        except (TypeError, ValueError):
            bad["observation_not_json"] += 1
            continue
        bad["observation_possibly_unwrapped"] += isinstance(resp, dict) and set(resp) <= {"raw", "result"}
        items.append(("tool", types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=fcs[0][0], name=fcs[0][1], response=resp))])))
    return items, calls, +bad

def serving_prompts(tok, items, calls, sub):
    """Per call: the messages the harness sends (ADK code path, offline) and the vLLM-rendered prompt."""
    from pathlib import Path
    from adk_submission.resolvers.generation import apply_thinking_config_to_model  # also installs the LiteLLM patch
    from adk_submission.yaml_loader import load_yaml
    from google.adk.apps.compaction import _events_to_compact_for_token_threshold
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions, EventCompaction
    from google.adk.flows.llm_flows import contents as adk_contents
    from google.adk.models import lite_llm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.tools.function_tool import FunctionTool
    from google.genai import types
    from swegemma.tools import create_tools
    cfg = load_yaml(Path(sub) / "agent.yaml", Path(sub))
    gen = types.GenerateContentConfig(**cfg["generate_content_config"])
    llm = apply_thinking_config_to_model(lite_llm.LiteLlm(model="openai/" + cfg["model"]), gen)
    kwargs = dict(llm._additional_args["extra_body"]["chat_template_kwargs"])
    tools = [FunctionTool(create_tools(None)[n]) for n in cfg["tools"]]
    identity = 'You are an agent. Your internal name is "%s".' % cfg["name"]  # flows/llm_flows/identity.py
    loop, session, out, nxt = asyncio.new_event_loop(), [], [], 0

    def build():
        req = LlmRequest(model=llm.model, contents=adk_contents._get_contents(None, session, cfg["name"]),
                         config=gen.model_copy(deep=True))
        req.append_instructions([cfg["instruction"]])
        req.append_instructions([identity])
        req.append_tools(tools)
        llm._maybe_append_user_content(req)
        lite_llm._append_fallback_user_content_if_missing(req)
        msgs, tool_dicts, _, _ = loop.run_until_complete(lite_llm._get_completion_inputs(req, llm.model))
        return (msgs, tool_dicts) + render(tok, vllm_conversation(msgs), tool_dicts, kwargs)
    for k, call in enumerate(calls):
        for i in range(nxt, call["item"]):
            session.append(Event(author="user" if items[i][0] == "user" else cfg["name"], content=items[i][1],
                                 timestamp=float(i + 1), invocation_id="inv"))
        nxt, form = call["item"], None
        todo = _events_to_compact_for_token_threshold(events=session, event_retention_size=RETENTION) if k and (
            calls[k - 1]["prompt_tokens"] or 0) >= THRESHOLD else []  # apps/compaction.py:443-449, before each call
        # The summary text is not traced: keep the form that reproduces vLLM's prompt_tokens. A summary without
        # visible text is dropped by contents._present_other_agent_message; text becomes 'For context:[model] said:'.
        for form in (("no_visible_text", "placeholder_text") if todo else ()):
            part = types.Part(text="", thought=True) if form == "no_visible_text" else types.Part(text=SUMMARY)
            comp = EventCompaction(start_timestamp=todo[0].timestamp, end_timestamp=todo[-1].timestamp,
                                   compacted_content=types.Content(role="model", parts=[part]))
            session.append(Event(author="user", timestamp=session[-1].timestamp + 0.5, invocation_id="cmp%d" % k,
                                 actions=EventActions(compaction=comp)))
            msgs, tool_dicts, text, ids = build()
            if len(ids) == call["prompt_tokens"] or form == "placeholder_text":
                break
            session.pop()
        if not todo:
            msgs, tool_dicts, text, ids = build()
        out.append({"messages": msgs, "text": text, "ids": ids, "compacted": form})
    loop.close()
    params = dict(llm._additional_args, **cfg["generate_content_config"])
    params.update(model=llm.model, system_instruction=cfg["instruction"] + "\n\n" + identity)
    return out, kwargs, tool_dicts, params

def sampled_completion(call, prefix, max_tokens):
    """Training-form completion: opener only when the prefix lacks it, thought, close, action, stop token."""
    body = ("" if prefix.endswith(FORCED) else FORCED) + call["thought"] + "<channel|>"
    if call["calls"]:
        return body + action_text(call["calls"]) + "<|tool_response>"
    return body + call["text"] + ("" if (call["completion_tokens"] or 0) >= max_tokens else "<turn|>")

def training_contexts(tok, prefixes, completions):
    """Context before every completion as sft/build_sft.py packs it (build_chains 144-171, chunk_chain 174-198)."""
    sys.path.insert(0, ROOT + "/sft")
    import build_sft
    tid = {m: tok.convert_tokens_to_ids(m) for m in (build_sft.TOOL_RESPONSE, build_sft.TURN_END, build_sft.CHANNEL_CLOSE)}
    chains, outcomes = build_sft.build_chains([{"prefix": p, "completion": c} for p, c in zip(prefixes, completions)], tok, tid)
    ctx, where = {}, {}
    for ci, items in enumerate(ch["items"] for ch in chains):
        for n, (first, last, _) in enumerate(build_sft.chunk_chain(items, MAX_LEN)[0]):
            cur = list(items[0]["pre_ids"]) if first else []
            for j in range(first, last + 1):
                cur += items[j]["pre_ids"]
                ctx[items[j]["step"]], where[items[j]["step"]] = list(cur), "chain %d chunk %d pos %d" % (ci, n, j - first)
                cur += items[j]["comp_ids"]
    return ctx, where, dict(outcomes)

def pieces(ids):
    """Split ids at structural special tokens; each special token is its own piece."""
    out = []
    for special, grp in itertools.groupby(ids, key=lambda t: t in STRUCT):
        out += [(t,) for t in grp] if special else [tuple(grp)]
    return out

def classify(dec, t, s, before, after, at_end, in_header):
    """One category per differing span; t/s = training/serving token ids of the span."""
    if at_end:
        return "generation_prompt_opener"
    if (t == EMPTY and not s) or (s == EMPTY and not t):
        return "empty_thought_block"
    dt, ds = dec(t), dec(s)
    if after == 101 and (ds == dt + "\n" or dt == ds + "\n"):
        return "newline_before_channel_close"
    if in_header or 98 in t + s:
        return "system_header_think"
    if before == 48 and sorted(dt) == sorted(ds):
        return "arg_key_order"
    if before == 50 or {50, 51} & set(t + s):
        return "tool_response_wrapper"
    return "other"

def diff_step(tok, train, serve):
    """(first differing token index or None, [spans]) for training vs serving ids."""
    first = next((i for i, (a, b) in enumerate(zip(train, serve)) if a != b),
                 None if len(train) == len(serve) else min(len(train), len(serve)))
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
    tp, sp = pieces(train), pieces(serve)
    hdr = next((i for i, p in enumerate(tp) if p == (106,)), len(tp))
    spans = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, tp, sp, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        t, s = [x for p in tp[i1:i2] for x in p], [x for p in sp[j1:j2] for x in p]
        before = tp[i1 - 1][0] if i1 and len(tp[i1 - 1]) == 1 and tp[i1 - 1][0] in STRUCT else None
        after = tp[i2][0] if i2 < len(tp) and len(tp[i2]) == 1 and tp[i2][0] in STRUCT else None
        cat = classify(dec, t, s, before, after, i2 == len(tp) and j2 == len(sp), i1 < hdr)
        same = sum(b.size for b in difflib.SequenceMatcher(None, t, s, autojunk=False).get_matching_blocks()) if t and s else 0
        spans.append({"category": cat, "train_tokens": len(t) - same, "serve_tokens": len(s) - same,
                      "train_text": dec(t)[:160], "serve_text": dec(s)[:160]})
    return first, spans

def emitted_residual(tok, call):
    """completion_tokens - (action + stop) on tool turns; thought turns also subtract opener + thought + close."""
    if not call["calls"] or call["completion_tokens"] is None:
        return None
    r = call["completion_tokens"] - len(enc(tok, action_text(call["calls"]))) - 1
    return r - len(enc(tok, FORCED + call["thought"] + "<channel|>")) if call["thought"] else r

def audit_trace(tok, path):
    sub = ROOT + "/subs/" + os.path.basename(os.path.dirname(os.path.dirname(path)))
    trace = json.load(open(path))  # thought suffix: the candidate that reproduces the most completion_tokens counts
    suffix, items, calls, bad = max([(x,) + trace_items(trace, x) for x in ("", "\n")],
                                    key=lambda v: sum(emitted_residual(tok, c) == 0 for c in v[2]))
    served, kwargs, tool_dicts, params = serving_prompts(tok, items, calls, sub)
    kit = [s["text"] + FORCED if s["text"].endswith("<tool_response|>") else s["text"] for s in served]
    # unbroken: a call right after a compaction starts a joinable chain (the training form of one uninterrupted chain)
    variants = {"build_sft": kit, "unbroken": [s["text"] if s["compacted"] else p for s, p in zip(served, kit)]}
    rows = [{"call": k + 1, "trace_step": c["step"], "served_tokens": len(s["ids"]), "vllm_prompt_tokens": c["prompt_tokens"],
             "exact_match": len(s["ids"]) == c["prompt_tokens"], "compaction_form": s["compacted"],
             "residual_completion_tokens": emitted_residual(tok, c)} for k, (c, s) in enumerate(zip(calls, served))]
    for prev, rec in zip(rows, rows[1:]):
        if not rec["compaction_form"] and None not in (rec["vllm_prompt_tokens"], prev["vllm_prompt_tokens"]):
            rec["delta_match"] = rec["served_tokens"] - prev["served_tokens"] == rec["vllm_prompt_tokens"] - prev["vllm_prompt_tokens"]
        if rec["compaction_form"] == "placeholder_text" and rec["vllm_prompt_tokens"] is not None:
            rec["inferred_summary_tokens"] = rec["vllm_prompt_tokens"] - rec["served_tokens"] + len(enc(tok, SUMMARY))
    out = {}
    for name, pre in variants.items():
        ctx, where, outcomes = training_contexts(tok, pre, [sampled_completion(c, p, params.get("max_output_tokens", 8192))
                                                            for c, p in zip(calls, pre)])
        totals = {c: collections.Counter() for c in CATS}
        for k, (rec, s) in enumerate(zip(rows, served)):
            if name == "unbroken" and s["compacted"]:  # the chain-start context keeps the kit prefix (with opener)
                ctx[k], where[k] = enc(tok, kit[k]), "chain start (kit prefix)"
            if k not in ctx:
                bad["call_without_training_context_" + name] += 1
                continue
            first, spans = diff_step(tok, ctx[k], s["ids"])
            rec[name] = {"train_tokens": len(ctx[k]), "train_position": where[k], "first_diff": first, "n_spans": len(spans),
                         "categories": dict(collections.Counter(x["category"] for x in spans)), "spans_head": spans[:3],
                         "other_spans": [x for x in spans if x["category"] == "other"]}
            for x in spans:
                totals[x["category"]].update(spans=1, train_tokens=x["train_tokens"], serve_tokens=x["serve_tokens"])
            for c in set(x["category"] for x in spans):
                totals[c]["calls"] += 1
        out[name] = {"build_sft_join_outcomes": outcomes, "category_totals": {
            c: dict(v, share_of_calls=round(v["calls"] / max(len(rows), 1), 3)) for c, v in totals.items()}}
    big = max(served, key=lambda s: s["ids"].count(51))  # the served prompt with the most tool responses
    ids = big["ids"]
    return {"trace": path, "trace_sha256": sha(path), "submission": sub, "calls": len(rows), "unmapped": dict(bad), "thought_suffix": suffix,
            "chat_template_kwargs": kwargs, "request_params": params, "tools_sent": [t["function"]["name"] for t in tool_dicts or []],
            "variants": out, "rows": rows, "busiest_served_prompt": {
                "call": served.index(big) + 1, "message_shapes": [{"role": m["role"], "keys": sorted(x for x, v in m.items() if v is not None)}
                                                                  for m in big["messages"][:8]],
                "token_after_tool_response_end": dict(collections.Counter(tok.convert_ids_to_tokens(b) for a, b in zip(ids, ids[1:]) if a == 51)),
                "token_before_tool_call": dict(collections.Counter(tok.convert_ids_to_tokens(a) for a, b in zip(ids, ids[1:]) if b == 48))},
            "residual_completion_tokens": dict(collections.Counter(r["residual_completion_tokens"] for r in rows)),
            "tools_json": tool_dicts, "call1_ids": served[0]["ids"] if served else []}

def step1_check(tok, rt):
    """Independent render of system + user alone (no ADK, no normaliser) against call 1 of the pipeline."""
    task = next(s["message"] for s in json.load(open(rt["trace"]))["steps"] if (s.get("extra") or {}).get("event_type") == "task_prompt")
    out = {"pipeline_call1_tokens": len(rt["call1_ids"]), "vllm_prompt_tokens_call1": rt["rows"][0]["vllm_prompt_tokens"]}
    for name, wrap in (("openai_parts", lambda x: [{"type": "text", "text": x}]), ("plain_string", lambda x: x)):
        msgs = [{"role": "system", "content": wrap(rt["request_params"]["system_instruction"])}, {"role": "user", "content": wrap(task)}]
        r = tok.apply_chat_template(msgs, tools=copy.deepcopy(rt["tools_json"]), add_generation_prompt=True, tokenize=True,
                                    **rt["chat_template_kwargs"])
        ids = list(r["input_ids"] if hasattr(r, "keys") else r)
        out[name] = {"tokens": len(ids), "equals_pipeline_call1": ids == rt["call1_ids"]}
    return out

def native_checks(tok, all_steps):
    """Training-side sanity: kit native requests rendered with the served template vs stored prefixes and ids."""
    with open(TRAJ) as f:
        t = next(x for x in map(json.loads, f) if x["tier"] == "native")
    res, agg = {"traj_id": t["traj_id"]}, collections.Counter()
    with open(KIT_NATIVE) as f:
        for rec in map(json.loads, f):
            if not all_steps and rec["id"] != t["traj_id"]:
                continue
            for i, ex in enumerate(rec["sft_examples"]):
                rq = ex["request"]
                text, ids = render(tok, vllm_conversation(rq["messages"]), rq.get("tools"), rq.get("chat_template_kwargs") or {})
                kit = text + FORCED if text.endswith("<tool_response|>") else text
                if rec["id"] == t["traj_id"] and i == 0:
                    res.update(header_text_equal=text == t["text_segments"][0]["text"], header_tokens=len(ids),
                               header_ids_equal=ids == t["chunks"][0]["input_ids"][:len(ids)])
                agg.update(steps=1, redacted_steps=not ex.get("served_ids_unchanged", True),
                           render_ends_tool_response=text.endswith("<tool_response|>"),
                           stored_prefix_ends_opener=ex["prefix"].endswith(FORCED), render_equals_stored_prefix=text == ex["prefix"],
                           render_plus_opener_equals_stored_prefix=kit == ex["prefix"],
                           render_plus_opener_ids_equal_served_ids=enc(tok, kit) == ex["prompt_token_ids"])
            if not all_steps:
                break
    res["all_native_steps" if all_steps else "first_trajectory_steps"] = dict(agg)
    return res

def m0_grammar(tok, path):
    """What the M0 training sequences contain around tool responses, before tool calls and at completion starts."""
    c, after, before = collections.Counter(), collections.Counter(), collections.Counter()
    with open(path) as f:
        for line in f:
            for ch in json.loads(line)["chunks"]:
                ids, lab = ch["input_ids"], ch["labels"]
                c.update(chunks=1, header_has_think=98 in ids[:ids.index(106) if 106 in ids else len(ids)],
                         label_differs_from_input=sum(1 for a, b in zip(ids, lab) if b != -100 and a != b))
                sup = [x != -100 for x in lab]
                cut = [i for i in range(len(ids)) if i == 0 or sup[i] != sup[i - 1]] + [len(ids)]
                for n, (a, b) in enumerate(zip(cut, cut[1:])):
                    h = ids[a:a + 4]
                    if sup[a]:
                        c["completion_starts_" + ("empty_block" if h == EMPTY else "opener_thought" if h[:3] == OPENER
                                                  else "channel_close" if h[:1] == [101] else "thought_text_or_other")] += 1
                    elif n:
                        c.update(context_spans_after_a_completion=1, context_spans_ending_with_opener=ids[b - 3:b] == OPENER)
                after.update(b for a, b in zip(ids, ids[1:]) if a == 51)
                before.update(a for a, b in zip(ids, ids[1:]) if b == 48)
    c["token_after_tool_response_end"] = {tok.convert_ids_to_tokens(k): v for k, v in after.most_common(5)}
    c["token_before_tool_call"] = {tok.convert_ids_to_tokens(k): v for k, v in before.most_common(5)}
    return dict(c)

def environment(tok):
    toks = {"served": tok, "training_base": load_tok(BASE_DIR)}
    tmap = {n: {s: t.convert_tokens_to_ids(s) for s in SPECIAL} for n, t in toks.items()}
    pair = lambda a, b: {"served": a, "training_base": b, "equal": a == b}
    shas = {f: pair(sha(SERVED_DIR + "/" + f), sha(BASE_DIR + "/" + f)) for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    argv = next(m["argv"] for m in json.load(open(SERVER_READY))["members"].values() if "vllm.entrypoints.openai.api_server" in m["argv"])
    flags = {x: argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
             for i, x in enumerate(argv) if x.startswith("--") and x not in ("--lora-modules", "--host", "--port", "--model")}
    flags.update({x: "absent (served dir files apply)" for x in ("--chat-template", "--default-chat-template-kwargs", "--generation-config") if x not in flags})
    flags["generation_config.eos_token_id"] = json.load(open(SERVED_DIR + "/generation_config.json"))["eos_token_id"]
    flags["eos_tokens"] = tok.convert_ids_to_tokens(flags["generation_config.eos_token_id"])
    return {"special_tokens": tmap, "special_tokens_equal": tmap["served"] == tmap["training_base"], "sha256": shas,
            "special_tokens_single_id": {n: all(enc(t, s) == [t.convert_tokens_to_ids(s)] for s in SPECIAL) for n, t in toks.items()},
            "vllm_flags": flags}

def verdict(rep):
    lines = []
    for t in rep["traces"]:
        for var, v in t["variants"].items():
            parts = ["%s %d spans in %d/%d calls (train %d / serve %d tokens)" % (c, x["spans"], x["calls"], t["calls"],
                     x["train_tokens"], x["serve_tokens"]) for c, x in v["category_totals"].items() if x.get("spans")]
            lines.append("%s [%s]: %s" % ("/".join(t["trace"].split("/")[-3::2]), var,
                                          "NOT identical; " + "; ".join(parts) if parts else "IDENTICAL in all %d calls" % t["calls"]))
    return lines

def write_md(rep, path):
    env, L = rep["environment"], ["# Train-vs-serve serialization audit (boss v046)", "", "Generated by audit.py from report.json.", "", "## Verdict", ""]
    L += ["- " + v for v in rep["verdict"]] + ["", "## Tokenizers, templates, server flags", "", "| token | served id | training-base id |", "|---|---|---|"]
    L += ["| `%s` | %d | %d |" % (s, env["special_tokens"]["served"][s], env["special_tokens"]["training_base"][s]) for s in SPECIAL]
    L += ["", "Maps equal: %s; single ids: %s." % (env["special_tokens_equal"], env["special_tokens_single_id"]), "",
          "| file | served sha256 | training-base sha256 | equal |", "|---|---|---|---|"]
    L += ["| %s | %s | %s | %s |" % (f, v["served"][:16], v["training_base"][:16], v["equal"]) for f, v in env["sha256"].items()]
    L += ["", "vLLM flags (v043 server-ready.json argv): `%s`" % json.dumps(env["vllm_flags"]), ""]
    cell = lambda v: "%s / %s / %s" % (v["train_tokens"], v["first_diff"], ", ".join(
        "%s:%d" % kv for kv in sorted(v["categories"].items())) or "none") if v else "-"
    for t in rep["traces"]:
        L += ["## Trace %s" % t["trace"], "", "sha256 %s; submission %s; calls %d; unmapped %s; chat_template_kwargs %s. Residual "
              "completion tokens: %s; thought suffix %r. Served prompt with most tool responses: %s." % (
                  t["trace_sha256"][:16], t["submission"], t["calls"], t["unmapped"] or "none", t["chat_template_kwargs"],
                  t["residual_completion_tokens"], t["thought_suffix"], json.dumps(t["busiest_served_prompt"])), "",
            "Cells for build_sft / unbroken: train tokens / first differing token / categories (spans).", "",
            "| call | step | served | vLLM | exact | delta | compaction | build_sft | unbroken |", "|" + "---|" * 9]
        L += ["| %d | %s | %d | %s | %s | %s | %s%s | %s | %s |" % (
            r["call"], r["trace_step"], r["served_tokens"], r["vllm_prompt_tokens"], r["exact_match"], r.get("delta_match", "-"),
            r["compaction_form"] or "-", " ~%s tok" % r["inferred_summary_tokens"] if "inferred_summary_tokens" in r else "",
            cell(r.get("build_sft")), cell(r.get("unbroken"))) for r in t["rows"]]
        L += ["", "| variant | category | spans | calls | share of calls | train tokens | serve tokens |", "|" + "---|" * 7]
        L += ["| %s | %s | %d | %d | %.3f | %d | %d |" % (var, c, x.get("spans", 0), x.get("calls", 0), x["share_of_calls"],
              x.get("train_tokens", 0), x.get("serve_tokens", 0)) for var, v in t["variants"].items() for c, x in v["category_totals"].items()]
        L += ["", "build_sft join outcomes: %s" % {var: v["build_sft_join_outcomes"] for var, v in t["variants"].items()}, ""]
    for title, key in (("Sanity checks", "sanity"), ("M0 training sequences", "m0_grammar")):
        L += ["## " + title, "", "```", json.dumps(rep.get(key), indent=1, ensure_ascii=False), "```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Train-vs-serve serialization audit (read-only).")
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--native-all", action="store_true", help="check every native kit step, not one trajectory")
    ap.add_argument("--m0-grammar", action="store_true", help="scan the M0 training sequences")
    a = ap.parse_args()
    tok = load_tok(SERVED_DIR)
    rep = {"environment": environment(tok), "traces": [audit_trace(tok, p) for p in a.trace]}
    rep["sanity"] = {"step1_independent_render": step1_check(tok, rep["traces"][0]), "native": native_checks(tok, a.native_all)}
    for t in rep["traces"]:
        del t["call1_ids"], t["tools_json"]
    rep["m0_grammar"] = m0_grammar(tok, M0_DATA) if a.m0_grammar else None
    rep["verdict"] = verdict(rep)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    write_md(rep, os.path.join(a.out, "report.md"))
    print("\n".join(rep["verdict"]))
    print(json.dumps(rep["sanity"]))

if __name__ == "__main__":
    main()
