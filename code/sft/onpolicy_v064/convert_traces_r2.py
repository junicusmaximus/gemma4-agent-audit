#!/usr/bin/env python3
"""v064-r2 Stage 2, step 1: ATIF traces of resolved E4B base rollouts -> kit trajectory rows (boss v064-r2).

r2 of convert_traces.py (d9f7edf7; superseded, left unedited) for prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md
(FROZEN 0893728c...; arm A_onpol, gate G1). Changes from r1: the Stage-1 r2 kept manifest (schema onpolicy-kept-v064r2:
every resolved eligible row, the runner's `selected` flag informational); generating_model "gemma-4-e4b-it (base, v052
runtime)"; per-row G1 checks (labels only on generated turns, every tool call paired with its response, supervised
tokens against the server's completion tokens); a call the packer would drop makes the trace Unsupported. The harness
tracer (adk_eval_core tracing/trace.py b426e17c, atif_converter.py 1d35490d) records one step per response part. The
converter rebuilds every model call in the kit's text form and hands the calls to the FROZEN kit packer sft/build_sft.py
(1064a6ad; pack_record -> build_chains -> chunk_chain, imported by path, never edited; kit tokenizer; max_seq_len 16384),
so a row has the schema and the packing of sft/data/trajectories.jsonl. The model's own text is kept as recorded.
  calls     a step with token counts starts a model call, unless it carries the timestamp of the open call (a second
            usage record of the same response); the steps without counts that follow (visible text, tool call, answer)
            belong to it. Unsupported (reported with the reason by build_arms, never changed): parallel tool calls,
            unknown step kinds, a response with neither an answer nor a tool call, a tool call without its result
            before the next call, special-token markers inside recorded text.
  header    <bos><|turn>system\n<|think|>\n + system instruction + the agent-identity / tool-declaration block that
            follows the same instruction in the first kit row's header (verbatim) + <turn|>\n<|turn>user\n + task
            prompt + <turn|>\n<|turn>model\n; its token count minus the BOS must equal the first call's prompt_tokens.
  call      <|channel>thought\n + thought [+ \n] + <channel|> + visible text + <|tool_call>call:NAME{args}<tool_call|> +
            <|tool_response> (an answer ends with <turn|>). The tracer strips the thought, so the newline before
            <channel|> is restored iff that variant's token count equals the call's completion_tokens; a call matching
            neither variant (or without a thought: rendered with an empty thought, as the kit renders teacher turns)
            is rendered without it and its token offset is reported, not gated.
  result    response:NAME{value:<|"|>JSON<|"|>}<tool_response|>, JSON = the recorded tool result re-dumped with
            ensure_ascii=False: the tracer escapes non-ASCII, the model saw it unescaped (v052 traces: next-call
            prompt_tokens); re-dumping with the defaults gives the recorded text back exactly.
  nudge     a harness continuation nudge after an answer: \n<|turn>user\n + text + <turn|>\n<|turn>model\n
  kit       as in every kit row, the thought opener of each call after the first ends the preceding prompt segment
            (labels -100) and the first call's completion starts with it; both arms share this serialisation.
The harness compacts long histories and the trace keeps no summaries: a trajectory is one chain with its full history,
which the packer cuts into 16,384-token chunks (header repeated) exactly as for the kit.
  --selftest [TRACE ...]  convert the traces (default: three v052 A0cap traces: a plain run, a response with visible
                          text before its tool call, a nudge and a thought-less answer), read every row back and compare
                          it text-identically with its trace, check the chunks, the header tokens, per-call and history
                          tokens; writes nothing. PASS iff every trace converts with one chain, an exact round trip, no
                          malformed chunk, no G1 label or pairing problem and a matching header (per-call token offsets
                          and the supervised-token difference are reported only)
  --kept MANIFEST         the same for every row of the Stage-1 r2 kept manifest (uncapped; build_arms_r2.py applies the
                          selection rule); writes nothing
build_arms_r2.py imports load_tools() and convert_kept()."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys

sys.dont_write_bytecode = True

ROOT = "PROJECT_ROOT/gemma4"
SFT = ROOT + "/sft"
BUILD_SFT = SFT + "/build_sft.py"
KIT = SFT + "/data/trajectories.jsonl"
TOKENIZER_DIR = ROOT + "/models/gemma-4-e4b-it-qat-w4a16-ct"
PINS = {BUILD_SFT: "1064a6add5d36ebbbfa1af9f5fbe0b8ed0c347746e472b5385d9699986ece399",
        KIT: "2ecfcb22d3fcc7cb2fa94092c4dea7cd94dfa4a4ae743afa03b3a16ff39230b3",
        TOKENIZER_DIR + "/tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        TOKENIZER_DIR + "/tokenizer_config.json": "a284d1243b62be31faa9c13e1c28cece940c4abaa7bd9ad87b94f61b40687200"}
SELFTEST = [ROOT + "/results/v052_A0cap_s1/traces/trace_rich_2725.json",
            ROOT + "/results/v052_A0cap_s3/traces/trace_rich_2725.json",
            ROOT + "/results/v052_A0cap_s3/traces/trace_rich_3061.json"]
MAX_LEN, TIER, BOS_OFFSET = 16384, "onpolicy_v064", 1  # the server's prompt_tokens do not count the BOS
Q, OPENER, CLOSE = '<|"|>', "<|channel>thought\n", "<channel|>"
CALL, CALL_END, RESP, RESP_END = "<|tool_call>call:", "<tool_call|>", "<|tool_response>", "<tool_response|>"
TURN_END, HEAD = "<turn|>", "<bos><|turn>system\n<|think|>\n"
TO_USER, TO_MODEL, NUDGE_OPEN = "<turn|>\n<|turn>user\n", "<turn|>\n<|turn>model\n", "\n<|turn>user\n"
MARKERS = (Q, "<|channel>", CLOSE, "<|tool_call>", CALL_END, RESP, RESP_END, "<|turn>", TURN_END)
REPO = re.compile(r"software engineering task for repository (\S+?)\.\n")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null")
BAD_KEY = re.compile(r"[:,{}\[\]]")


GEN_MODEL = "gemma-4-e4b-it (base, v052 runtime)"  # the generating-policy label Boss fixed for A_onpol rows
KEPT_SCHEMA = "onpolicy-kept-v064r2"


class Unsupported(ValueError):
    """A trace outside the structure the renderer supports; reported with the reason, never changed."""


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def token_ids(tok, packer):
    """Token ids of the G1 label check; each format marker must be one distinct known token."""
    ids = {name: tok.convert_tokens_to_ids(text) for name, text in (("resp", RESP), ("turn_end", TURN_END),
                                                                    ("resp_end", RESP_END), ("turn_open", "<|turn>"),
                                                                    ("bos", "<bos>"))}
    if len(set(ids.values())) != len(ids) or any(type(v) is not int or v == tok.unk_token_id for v in ids.values()):
        raise ValueError("a format marker is not one known token")
    ids["opener"] = packer.encode(tok, OPENER)
    return ids


def load_tools():
    """The pinned inputs, the frozen packer (imported by path), the kit tokenizer and the first kit header."""
    for path, sha in PINS.items():
        if sha256_file(path) != sha:
            raise ValueError("pinned input changed: " + path)
    spec = importlib.util.spec_from_file_location("build_sft_frozen", BUILD_SFT)
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
    if (packer.TOOL_RESPONSE, packer.TURN_END, packer.CHANNEL_CLOSE, packer.GEN_PROMPT) != (RESP, TURN_END, CLOSE, OPENER):
        raise ValueError("the kit packer's markers differ from the converter's")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    with open(KIT, encoding="utf-8") as f:
        kit_header = json.loads(f.readline())["text_segments"][0]["text"]
    return {"packer": packer, "tok": tok, "kit_header": kit_header,
            "tid": {m: tok.convert_tokens_to_ids(m) for m in (RESP, TURN_END, CLOSE)},
            "encode": lambda text: packer.encode(tok, text), "count": lambda text: len(packer.encode(tok, text)),
            "ids": token_ids(tok, packer)}


def plain(text, where):
    """Recorded text that renders and parses unambiguously: a string without any of the format's markers."""
    if not isinstance(text, str):
        raise Unsupported(where + " is not text")
    hit = [m for m in MARKERS if m in text]
    if hit:
        raise Unsupported("%s contains the marker %s" % (where, hit[0]))
    return text


def render_value(v):
    """Tool-call argument syntax of the kit: <|"|>-quoted strings, bare numbers / true / false / null, [..] lists and
    {key:value} objects in the recorded key order."""
    if isinstance(v, str):
        return Q + plain(v, "a string argument") + Q
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ",".join(render_value(x) for x in v) + "]"
    if isinstance(v, dict):
        if any(not isinstance(k, str) or not k or BAD_KEY.search(k) or Q in k for k in v):
            raise Unsupported("an argument key cannot be rendered unambiguously")
        return "{" + ",".join(k + ":" + render_value(x) for k, x in v.items()) + "}"
    raise Unsupported("argument type %s" % type(v).__name__)


def parse_value(s, i):
    """Inverse of render_value from position i; returns (value, next position)."""
    if s.startswith(Q, i):
        j = s.find(Q, i + len(Q))
        if j < 0:
            raise ValueError("unterminated string at %d" % i)
        return s[i + len(Q):j], j + len(Q)
    if s.startswith("[", i) or s.startswith("{", i):
        is_list, end = s[i] == "[", "]" if s[i] == "[" else "}"
        out, i = ([] if is_list else {}), i + 1
        if s.startswith(end, i):
            return out, i + 1
        while True:
            if is_list:
                value, i = parse_value(s, i)
                out.append(value)
            else:
                j = s.index(":", i)
                value, nxt = parse_value(s, j + 1)
                out[s[i:j]] = value
                i = nxt
            if s.startswith(",", i):
                i += 1
            elif s.startswith(end, i):
                return out, i + 1
            else:
                raise ValueError("bad separator at %d" % i)
    m = NUMBER.match(s, i)
    if not m:
        raise ValueError("bad value at %d" % i)
    return json.loads(m.group(0)), m.end()


def kind(step):
    return step.get("source"), (step.get("extra") or {}).get("event_type")


def add_part(call, step, event, where):
    """One trace step into its model call, in the tracer's part order: thought, visible text, tool call, tool result."""
    if call["observation"] is not None:
        raise Unsupported(where + ": a part after the tool result of its model call")
    message = step.get("message") or ""
    if event == "thinking":
        if call["steps"]:
            raise Unsupported(where + ": a thought after other parts of its model call")
        call["thought"] = plain(message, where + " thought")
    elif event in ("text", "final"):
        if call["visible"] is not None or call["calls"]:
            raise Unsupported(where + ": visible text after other parts of its model call")
        call["visible"], call["final"] = plain(message, where + " text"), event == "final"
    elif message:
        raise Unsupported(where + ": a tool-call step with text")
    for t in step.get("tool_calls") or []:
        call["calls"].append({"name": plain(t.get("function_name"), where + " tool name"),
                              "arguments": t.get("arguments") or {}})
    obs = step.get("observation")
    if obs is not None:
        if not step.get("tool_calls"):
            raise Unsupported(where + ": a tool result without its tool call")
        call["observation"], call["tool_name"] = obs.get("content"), (obs.get("extra") or {}).get("tool_name")
    call["steps"].append(step.get("step_id"))


def read_trace(path):
    """(system instruction, task prompt, events, generating model) of an ATIF trace; events are model calls (module doc)
    and continuation nudges."""
    with open(path, encoding="utf-8") as f:
        steps = json.load(f)["steps"]
    if len(steps) < 3 or kind(steps[0]) != ("system", "system_instruction") or kind(steps[1]) != ("user", "task_prompt"):
        raise Unsupported("the trace does not open with the system instruction and the task prompt")
    events = []
    for step in steps[2:]:
        source, event = kind(step)
        where = "step %s" % step.get("step_id")
        if (source, event) == ("system", "continuation_nudge"):
            events.append({"kind": "nudge", "text": plain(step.get("message"), where), "steps": [step.get("step_id")]})
            continue
        if source != "agent" or event not in ("thinking", "text", "final", None):
            raise Unsupported("%s: unsupported step %s/%s" % (where, source, event))
        stamp = (step.get("extra") or {}).get("timestamp")
        call = events[-1] if events and events[-1]["kind"] == "call" else None
        if step.get("metrics") and call is not None and stamp == call["timestamp"]:
            call["extra_usage"].append(step.get("step_id"))
        elif step.get("metrics"):
            call = {"kind": "call", "thought": "", "visible": None, "final": False, "calls": [], "observation": None,
                    "tool_name": None, "metrics": step["metrics"], "timestamp": stamp, "steps": [], "extra_usage": []}
            events.append(call)
        elif call is None:
            raise Unsupported(where + ": a model step outside a model call")
        add_part(call, step, event, where)
    check_events(events)
    model = next((s.get("model_name") for s in steps if s.get("source") == "agent" and s.get("model_name")), None)
    return plain(steps[0].get("message"), "system instruction"), plain(steps[1].get("message"), "task prompt"), \
        events, model


def check_events(events):
    """The response structure the renderer supports (module doc); anything else is Unsupported."""
    if not events or events[0]["kind"] != "call":
        raise Unsupported("the trace does not start with a model call")
    for i, e in enumerate(events):
        nxt = events[i + 1] if i + 1 < len(events) else None
        if e["kind"] == "nudge":
            if events[i - 1]["kind"] != "call" or events[i - 1]["calls"]:
                raise Unsupported("a continuation nudge that does not follow an answer")
            continue
        if len(e["calls"]) > 1:
            raise Unsupported("parallel tool calls (steps %s)" % e["steps"])
        if e["calls"]:
            if e["final"]:
                raise Unsupported("a final answer with a tool call (steps %s)" % e["steps"])
            if e["observation"] is None and nxt is not None:
                raise Unsupported("a tool call without its tool result (steps %s)" % e["steps"])
            if e["observation"] is not None and e["tool_name"] != e["calls"][0]["name"]:
                raise Unsupported("a tool result of another tool (steps %s)" % e["steps"])
        elif not e["visible"]:
            raise Unsupported("a response with neither an answer nor a tool call (steps %s)" % e["steps"])
        elif nxt is not None and nxt["kind"] != "nudge":
            raise Unsupported("a model call after an answer without a nudge (steps %s)" % e["steps"])


def model_view(content, where):
    """The tool result as the model saw it (module doc), after checking that it is the tracer's JSON dump."""
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        raise Unsupported(where + ": the tool result is not JSON")
    if json.dumps(value) != content:
        raise Unsupported(where + ": the tool result is not the tracer's JSON dump")
    return plain(json.dumps(value, ensure_ascii=False), where + " tool result")


def answer_part(e):
    """What follows <channel|>: the visible text, the tool call and the stop marker."""
    calls = "".join(CALL + c["name"] + render_value(c["arguments"]) + CALL_END for c in e["calls"])
    return (e["visible"] or "") + calls + (RESP if e["calls"] else TURN_END)


def tool_result(e):
    return ("response:" + e["calls"][0]["name"] + "{value:" + Q + model_view(e["observation"], "steps %s" % e["steps"])
            + Q + "}" + RESP_END)


def fit(e, count):
    """(newline before <channel|>, token offset of the rendered completion against the call's completion_tokens)."""
    want = (e["metrics"] or {}).get("completion_tokens")
    flat = count(OPENER + e["thought"] + CLOSE + answer_part(e))
    if flat == want:
        return False, 0
    if count(OPENER + e["thought"] + "\n" + CLOSE + answer_part(e)) == want:
        return True, 0
    return False, (flat - want) if isinstance(want, int) else None


def build_examples(system, user, events, kit_header, count):
    """(header, the packer's sft_examples, per-call fidelity) in the kit's prefix / completion form (module doc)."""
    if not kit_header.startswith(HEAD + system) or TO_USER not in kit_header:
        raise ValueError("the trace's system instruction does not open the first kit header")
    header = kit_header[:kit_header.index(TO_USER)] + TO_USER + user + TO_MODEL
    if count(header) - BOS_OFFSET != events[0]["metrics"].get("prompt_tokens"):
        raise ValueError("header tokens differ from the first call's prompt_tokens")
    exs, fidelity, text, pending, nudges = [], [], header, None, ""
    for e in events:
        if e["kind"] == "nudge":
            nudges += NUDGE_OPEN + e["text"] + TO_MODEL
            continue
        newline, offset = fit(e, count)
        if exs:
            text += (tool_result(pending) if pending is not None else "") + nudges + OPENER
            comp = ""
        else:
            comp = OPENER
        comp += e["thought"] + ("\n" if newline else "") + CLOSE + answer_part(e)
        exs.append({"prefix": text, "completion": comp})
        fidelity.append({"steps": e["steps"], "newline": newline, "offset": offset, "extra_usage": e["extra_usage"]})
        text += comp
        pending, nudges = (e if e["calls"] else None), ""
    return header, exs, fidelity


def chunk_problems(row, header_ids):
    """Malformed-chunk check: ids and labels aligned (labels equal to ids where supervised), at most 16,384 tokens, the
    recorded counts right, at least one target after the causal shift, and the unsupervised trajectory header first."""
    bad, n = [], len(header_ids)
    for k, c in enumerate(row["chunks"]):
        ids, lab = c["input_ids"], c["labels"]
        if not len(ids) == len(lab) == c["n_tokens"] or len(ids) > MAX_LEN:
            bad.append("chunk %d: length" % k)
        elif any(y != -100 and y != x for x, y in zip(ids, lab)):
            bad.append("chunk %d: id / label alignment" % k)
        elif c["n_loss_tokens"] != sum(y != -100 for y in lab) or not any(y != -100 for y in lab[1:]):
            bad.append("chunk %d: target count" % k)
        elif ids[:n] != header_ids or any(y != -100 for y in lab[:n]):
            bad.append("chunk %d: header" % k)
    return bad


def label_problems(row, ids):
    """G1 masking check on token ids. Every chunk starts unsupervised and ends supervised; every supervised run is one
    generated turn (exactly one stop token, <|tool_response> or <turn|>, as its last token; no <|turn>, <tool_response|>
    or <bos>); the thought opener stands exactly once before it (ending the prompt run or opening the turn); every
    prompt run after a chunk's first holds a tool result end or a user turn."""
    bad, opener, stops = [], ids["opener"], (ids["resp"], ids["turn_end"])
    prompt_only = (ids["turn_open"], ids["resp_end"], ids["bos"])
    for k, c in enumerate(row["chunks"]):
        x, lab, runs, start = c["input_ids"], c["labels"], [], 0
        for i in range(1, len(x) + 1):
            if i == len(x) or (lab[i] == -100) != (lab[start] == -100):
                runs.append((lab[start] != -100, x[start:i]))
                start = i
        if not runs or runs[0][0] or not runs[-1][0]:
            bad.append("chunk %d: does not start unsupervised and end supervised" % k)
            continue
        for j in range(1, len(runs), 2):
            prev, toks = runs[j - 1][1], runs[j][1]
            if sum(t in stops for t in toks) != 1 or toks[-1] not in stops or any(t in prompt_only for t in toks):
                bad.append("chunk %d run %d: not one generated turn" % (k, j))
            if (prev[-len(opener):] == opener) == (toks[:len(opener)] == opener):
                bad.append("chunk %d run %d: thought opener not exactly once" % (k, j))
            if j > 1 and ids["resp_end"] not in prev and ids["turn_open"] not in prev:
                bad.append("chunk %d run %d: prompt run without a tool result or user turn" % (k, j))
    return bad


def pairing_problems(row):
    """G1 pairing check on the text segments: a completion ending with <|tool_response> holds exactly one tool call and,
    unless it is the last completion, the next prompt segment opens with that tool's response; an answer holds no tool
    call and is never followed by a tool response."""
    segs, bad = row["text_segments"], []
    for k in range(1, len(segs), 2):
        comp, nxt = segs[k]["text"], (segs[k + 1]["text"] if k + 1 < len(segs) else None)
        names = re.findall(re.escape(CALL) + r"([^{]*)[{]", comp)
        if comp.endswith(RESP):
            if len(names) != 1:
                bad.append("completion %d: %d tool calls" % (k, len(names)))
            elif nxt is not None and not nxt.startswith("response:" + names[0] + "{value:"):
                bad.append("completion %d: %s not answered by the next segment" % (k, names[0]))
        elif names or (nxt is not None and nxt.startswith("response:")):
            bad.append("completion %d: an answer paired with a tool call or a tool response" % k)
    return bad


def loss_vs_server(events, fidelity, row, n_opener):
    """Supervised tokens minus the server's generated tokens (each call's completion_tokens plus its rendering offset,
    less the opener moved into the prompt for every call after the first): 0 when exactly the generated turns are
    labelled; None when a call lacks a count."""
    wants = [(e["metrics"] or {}).get("completion_tokens") for e in events if e["kind"] == "call"]
    if any(type(w) is not int for w in wants) or any(f["offset"] is None for f in fidelity):
        return None
    return row["loss_tokens"] - (sum(w + f["offset"] for w, f in zip(wants, fidelity)) - n_opener * (len(wants) - 1))


def parse_answer(text, where):
    """Inverse of answer_part: (visible text, tool calls)."""
    if text.endswith(TURN_END):
        return text[:-len(TURN_END)], []
    if not text.endswith(RESP) or CALL not in text:
        raise ValueError(where + ": no tool call or stop marker")
    body = text[:-len(RESP)]
    start = body.index(CALL)
    visible, calls = body[:start], []
    while start < len(body):
        if not body.startswith(CALL, start):
            raise ValueError(where + ": text after a tool call")
        brace = body.index("{", start + len(CALL))
        args, end = parse_value(body, brace)
        if not body.startswith(CALL_END, end):
            raise ValueError(where + ": unterminated tool call")
        calls.append({"name": body[start + len(CALL):brace], "arguments": args})
        start = end + len(CALL_END)
    return visible, calls


def parse_row(row):
    """The trace content a row carries, read back from its text segments alone (inverse of build_examples)."""
    segs = row["text_segments"]
    if (len(segs) % 2 or {s["chain"] for s in segs} != {0}
            or [s["train"] for s in segs] != [i % 2 == 1 for i in range(len(segs))]):
        raise ValueError("segments are not one chain of header + completion / prompt pairs")
    header = segs[0]["text"]
    if not header.endswith(TO_MODEL) or TO_USER not in header:
        raise ValueError("header layout")
    cut = header.index(TO_USER)
    got = {"system_part": header[:cut], "user": header[cut + len(TO_USER):-len(TO_MODEL)], "events": []}
    events = got["events"]
    for k in range(1, len(segs), 2):
        comp = segs[k]["text"]
        if k == 1:
            if not comp.startswith(OPENER):
                raise ValueError("first completion without the thought opener")
            comp = comp[len(OPENER):]
        else:
            prompt = segs[k - 1]["text"]
            if not prompt.endswith(OPENER):
                raise ValueError("segment %d does not end with the thought opener" % (k - 1))
            body = prompt[:-len(OPENER)]
            m = re.fullmatch(r"response:([^{]*)\{value:" + re.escape(Q) + r"(.*)" + re.escape(Q) + r"\}"
                             + re.escape(RESP_END), body, re.S)
            if m:
                events[-1]["observation"] = json.dumps(json.loads(m.group(2)))  # the tracer's escaping again
                events[-1]["result_name"] = m.group(1)
            elif body.startswith(NUDGE_OPEN) and body.endswith(TO_MODEL):
                events.append({"kind": "nudge", "text": body[len(NUDGE_OPEN):-len(TO_MODEL)]})
            else:
                raise ValueError("segment %d is neither a tool result nor a nudge" % (k - 1))
        thought, sep, rest = comp.partition(CLOSE)
        if not sep:
            raise ValueError("completion %d without <channel|>" % k)
        visible, calls = parse_answer(rest, "completion %d" % k)
        events.append({"kind": "call", "thought": thought[:-1] if thought.endswith("\n") else thought,
                       "visible": visible, "calls": calls, "observation": None})
    return got


def comparable(events):
    """The trace events a row can carry: up to the last model call, without the last call's tool result."""
    last = max(i for i, e in enumerate(events) if e["kind"] == "call")
    out = []
    for i, e in enumerate(events[:last + 1]):
        if e["kind"] == "nudge":
            out.append({"kind": "nudge", "text": e["text"]})
            continue
        out.append({"kind": "call", "thought": e["thought"], "visible": e["visible"] or "", "calls": e["calls"],
                    "observation": e["observation"] if i < last and e["calls"] else None})
    return out


def round_trip(row, system, user, events, kit_header):
    """Differences between the row read back and the trace, text-identical (arguments as order-preserving JSON)."""
    got, diffs = parse_row(row), []
    if got["system_part"] != kit_header[:kit_header.index(TO_USER)] or not got["system_part"].startswith(HEAD + system):
        diffs.append("system instruction / tool block")
    if got["user"] != user:
        diffs.append("task prompt")
    want = comparable(events)
    if len(want) != len(got["events"]):
        diffs.append("event count %d != %d" % (len(got["events"]), len(want)))
    for i, (a, b) in enumerate(zip(want, got["events"])):
        if a["kind"] != b["kind"]:
            diffs.append("event %d kind" % i)
        elif a["kind"] == "nudge":
            if a["text"] != b["text"]:
                diffs.append("event %d nudge text" % i)
        else:
            for key in ("thought", "visible", "observation"):
                if a[key] != b[key]:
                    diffs.append("event %d %s" % (i, key))
            if json.dumps(a["calls"]) != json.dumps(b["calls"]):
                diffs.append("event %d tool call" % i)
            if a["observation"] is not None and b.get("result_name") != a["calls"][0]["name"]:
                diffs.append("event %d tool result name" % i)
    return diffs


def history(events, fidelity, count):
    """Next-call prompt_tokens against completion + rendered tool result (+1 for the newline the server's history adds
    before <channel|> when the sample had none): exact / compaction (prompt shrank) / other / skipped (offset)."""
    calls = [(i, e) for i, e in enumerate(events) if e["kind"] == "call"]
    out = collections.Counter()
    for (i, a), fa, (j, b) in zip(calls, fidelity, calls[1:]):
        if j != i + 1 or not a["calls"]:
            continue
        if fa["offset"] != 0:
            out["skipped"] += 1
            continue
        delta = b["metrics"]["prompt_tokens"] - a["metrics"]["prompt_tokens"]
        want = a["metrics"]["completion_tokens"] + (0 if fa["newline"] else 1) + count(tool_result(a))
        out["compaction" if delta < 0 else ("exact" if delta == want else "other")] += 1
    return out


def convert(path, task_id, traj_id, tools):
    """One trace -> (kit row, info): conversion, packing, round trip, chunk check and token checks."""
    system, user, events, model = read_trace(path)
    repo = REPO.search(user)
    if repo is None:
        raise Unsupported("no repository in the task prompt")
    header, exs, fidelity = build_examples(system, user, events, tools["kit_header"], tools["count"])
    rec = {"id": traj_id, "sft_examples": exs, "generating_model": GEN_MODEL,
           "tool_calls": sum(len(e["calls"]) for e in events if e["kind"] == "call")}
    row, _, c = tools["packer"].pack_record(rec, TIER, task_id, repo.group(1).lower(), tools["tok"], tools["tid"],
                                            MAX_LEN)
    if c["chains"] != 1 or c["out_exact"] != len(exs) - 1:
        raise ValueError(traj_id + ": the packer did not join every call into one chain")
    if c["dropped_items"]:
        raise Unsupported("%d call(s) longer than max_seq_len with the header and prompt segment" % c["dropped_items"])
    offsets = [f["offset"] for f in fidelity if f["offset"] != 0]
    info = {"traj_id": traj_id, "task_id": task_id, "trace": path, "trace_model": model, "calls": len(exs),
            "tool_calls": rec["tool_calls"],
            "nudges": sum(e["kind"] == "nudge" for e in events), "merged_usage_steps": sum(
                len(f["extra_usage"]) for f in fidelity), "newline": sum(f["newline"] for f in fidelity),
            "no_newline": sum(not f["newline"] and f["offset"] == 0 for f in fidelity), "token_offsets": offsets,
            "history": dict(history(events, fidelity, tools["count"])), "chunks": len(row["chunks"]),
            "dropped_steps": c["dropped_items"], "dropped_loss": c["dropped_loss"], "total_tokens": row["total_tokens"],
            "loss_tokens": row["loss_tokens"], "malformed": chunk_problems(row, tools["encode"](header)),
            "round_trip": round_trip(row, system, user, events, tools["kit_header"]),
            "labels": label_problems(row, tools["ids"]), "pairing": pairing_problems(row),
            "loss_vs_server": loss_vs_server(events, fidelity, row, len(tools["ids"]["opener"]))}
    return row, info


def trace_file(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def run_rows(run):
    """{instance_id: row} of a run's task_results.jsonl; complete lines only (an interrupted run may end mid-line)."""
    with open(os.path.join(run, "task_results.jsonl"), encoding="utf-8") as f:
        lines = f.read().split(chr(10))[:-1]
    return {r["instance_id"]: r for r in map(json.loads, (x for x in lines if x.strip()))}


def read_kept(path, eligible=None):
    """The Stage-1 r2 kept manifest (tools/run_onpolicy_gen_v064r2.py kept_manifest, schema onpolicy-kept-v064r2),
    checked: generation v064, revision r2; its task list free of rich_ ids and equal to the eligible list when one is
    given; counts.uncapped and counts.selected equal to its rows; every row resolved, of a listed task, with an integer
    seed, a boolean selected flag, unique (task, seed), its trace named trace_<task>.json, equal to the recorded sha256
    and resolved in its run's task_results.jsonl (the run dir holds traces/)."""
    with open(path, encoding="utf-8") as f:
        kept = json.load(f)
    tasks, rows, counts = kept.get("tasks") or [], kept.get("kept") or [], kept.get("counts") or {}
    if ((kept.get("schema"), kept.get("generation"), kept.get("revision")) != (KEPT_SCHEMA, "v064", "r2") or not tasks
            or len(set(tasks)) != len(tasks) or any(t.startswith("rich_") for t in tasks)
            or (eligible is not None and list(tasks) != list(eligible))):
        raise ValueError("kept manifest: schema, revision or task list")
    if (counts.get("uncapped"), counts.get("selected")) != (len(rows), sum(r.get("selected") is True for r in rows)):
        raise ValueError("kept manifest: counts differ from its rows")
    seen, results = set(), {}
    for e in rows:
        key, trace = (e.get("task_id"), e.get("seed")), trace_file(e.get("trace_path") or "")
        if (e.get("resolved") is not True or key[0] not in tasks or type(key[1]) is not int or key in seen
                or type(e.get("selected")) is not bool or os.path.basename(trace) != "trace_%s.json" % key[0]):
            raise ValueError("kept manifest row %s" % (key,))
        seen.add(key)
        if sha256_file(trace) != e.get("trace_sha256"):
            raise ValueError("trace changed since Stage 1: " + trace)
        run = os.path.dirname(os.path.dirname(trace))
        if run not in results:
            results[run] = run_rows(run)
        if (results[run].get(key[0]) or {}).get("resolved") is not True:
            raise ValueError("not resolved in its run's task_results.jsonl: %s seed %s" % key)
    return kept


def convert_kept(path, tools, eligible=None):
    """(kept manifest, converted, excluded) over every kept row in manifest order, uncapped (build_arms_r2.py applies
    the selection rule): converted = [(row, kit row, info)], excluded = [{traj_id, task_id, seed, trace, reason}] for
    Unsupported traces. A round-trip difference, a malformed chunk, a G1 label or pairing problem, or a repository other
    than the row's raises (a converter fault, never an exclusion)."""
    kept, converted, excluded = read_kept(path, eligible), [], []
    for e in kept["kept"]:
        traj_id = "v064_%s_s%d" % (e["task_id"], e["seed"])
        try:
            row, info = convert(trace_file(e["trace_path"]), e["task_id"], traj_id, tools)
        except Unsupported as err:
            excluded.append({"traj_id": traj_id, "task_id": e["task_id"], "seed": e["seed"], "trace": e["trace_path"],
                             "reason": str(err)})
            continue
        problems = info["round_trip"] + info["malformed"] + info["labels"] + info["pairing"]
        if problems or row["repository"] != str(e.get("repo")).lower():
            raise ValueError("%s: %s, repository %s" % (traj_id, problems, row["repository"]))
        converted.append((e, row, info))
    return kept, converted, excluded


def line(info):
    return ("%s calls %d (newline %d, no newline %d, token offsets %s) nudges %d merged-usage steps %d history %s "
            "chunks %d dropped steps %d tokens %d loss %d (vs server %s) malformed %s labels %s pairing %s "
            "round trip %s" % (
                info["traj_id"], info["calls"], info["newline"], info["no_newline"], info["token_offsets"] or "none",
                info["nudges"], info["merged_usage_steps"], json.dumps(info["history"], sort_keys=True),
                info["chunks"], info["dropped_steps"], info["total_tokens"], info["loss_tokens"],
                "%+d" % info["loss_vs_server"] if info["loss_vs_server"] is not None else "n/a",
                info["malformed"] or "none", info["labels"] or "ok", info["pairing"] or "ok",
                "EXACT" if not info["round_trip"] else "DIFF " + str(info["round_trip"])))


def selftest(paths, tools):
    ok, unsupported = 0, 0
    for path in paths:
        run, name = path.split("/")[-3], os.path.basename(path)[len("trace_"):-len(".json")]
        try:
            _, info = convert(path, name, run + "_" + name, tools)
        except Unsupported as err:
            unsupported += 1
            print("SELFTEST %s_%s UNSUPPORTED %s" % (run, name, err), flush=True)
            continue
        good = not (info["round_trip"] or info["malformed"] or info["labels"] or info["pairing"])
        ok += good
        print("SELFTEST " + line(info), flush=True)
    passed = ok == len(paths)
    print("SELFTEST %s: %d of %d traces convert with an exact round trip, no malformed chunk and no G1 label or "
          "pairing problem (%d unsupported)" % (
        "PASS" if passed else "FAIL", ok, len(paths), unsupported), flush=True)
    return 0 if passed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", nargs="*", metavar="TRACE", help="default: the three v052 traces of the module doc")
    mode.add_argument("--kept", metavar="MANIFEST", help="Stage-1 r2 kept manifest (migration/onpolicy-v064/...)")
    opts = ap.parse_args(argv)
    tools = load_tools()
    if opts.selftest is not None:
        return selftest([trace_file(p) for p in opts.selftest] or SELFTEST, tools)
    kept, converted, excluded = convert_kept(opts.kept, tools)
    for _, _, info in converted:
        print("CONVERTED " + line(info), flush=True)
    for x in excluded:
        print("EXCLUDED %s: %s" % (x["traj_id"], x["reason"]), flush=True)
    print("KEPT %d rows (uncapped): %d converted, %d excluded, %d chunks" % (
        len(kept["kept"]), len(converted), len(excluded), sum(len(r["chunks"]) for _, r, _ in converted)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
