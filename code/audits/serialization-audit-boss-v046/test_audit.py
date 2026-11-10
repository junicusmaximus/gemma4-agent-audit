"""Tests that pin the audit classifier on a tiny synthetic conversation (served tokenizer and template; CPU only).

Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_audit.py
"""
import json

import audit

TOK = audit.load_tok(audit.SERVED_DIR)
HEAD = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Fix the bug."}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
    "type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"}}, "required": ["filepath"]}}}]
KW = {"enable_thinking": True}
VALUE_WRAPPER = 'response:read_file{value:<|"|>{"status": "ok"}<|"|>}<tool_response|>'


def served(args, reasoning=None):
    """Serving form: OpenAI messages, vLLM normalisation, served template; prompt after one tool turn."""
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps(args)}}
    turn = {"role": "assistant", "content": None, "tool_calls": [call]}
    if reasoning is not None:
        turn["reasoning"] = reasoning
    msgs = HEAD + [turn, {"role": "tool", "tool_call_id": "c1", "content": '{"status": "ok"}'}]
    return audit.render(TOK, audit.vllm_conversation(msgs), TOOLS, KW)[1]


def trained(args, thought="", wrapper=VALUE_WRAPPER):
    """Training form of the same conversation: header, sampled completion, result segment, forced opener."""
    header = audit.render(TOK, audit.vllm_conversation(HEAD), TOOLS, KW)[0]
    completion = audit.FORCED + thought + "<channel|>" + audit.action_text([("read_file", args)]) + "<|tool_response>"
    return audit.enc(TOK, header + completion + wrapper + audit.FORCED)


def categories(train_ids, serve_ids):
    return sorted(s["category"] for s in audit.diff_step(TOK, train_ids, serve_ids)[1])


def test_empty_thought_block_dropped_and_opener_missing():
    args = {"filepath": "a.py", "start_line": 3}
    first, spans = audit.diff_step(TOK, trained(args), served(args))
    assert {s["category"]: (s["train_tokens"], s["serve_tokens"]) for s in spans} == {
        "empty_thought_block": (4, 0), "generation_prompt_opener": (3, 0)}
    assert len(spans) == 2 and first is not None


def test_argument_key_order():
    args = {"start_line": 3, "filepath": "a.py"}  # sampled order; the template sorts the keys
    assert categories(trained(args), served(args)) == ["arg_key_order", "empty_thought_block", "generation_prompt_opener"]


def test_tool_response_wrapper():
    mapping = 'response:read_file{status:<|"|>ok<|"|>}<tool_response|>'  # mapping branch of format_tool_response_block
    args = {"filepath": "a.py"}
    assert categories(trained(args, wrapper=mapping), served(args)) == [
        "empty_thought_block", "generation_prompt_opener", "tool_response_wrapper"]


def test_newline_before_channel_close():
    args = {"filepath": "a.py"}
    assert categories(trained(args, thought="Look.\n"), served(args, reasoning="Look.\n")) == [
        "generation_prompt_opener", "newline_before_channel_close"]


def test_think_token_in_header_and_identical_input():
    think = audit.render(TOK, audit.vllm_conversation(HEAD), TOOLS, KW)[1]
    plain = audit.render(TOK, audit.vllm_conversation(HEAD), TOOLS, {"enable_thinking": False})[1]
    assert categories(think, plain) == ["system_header_think"]
    assert audit.diff_step(TOK, think, list(think)) == (None, [])


def test_vllm_normalisation():
    conv = audit.vllm_conversation([{"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"b": 1, "a": 2}'}}]}])
    assert conv[0]["content"] == [] and conv[0]["reasoning_content"] == "r"
    assert list(conv[0]["tool_calls"][0]["function"]["arguments"]) == ["b", "a"]
