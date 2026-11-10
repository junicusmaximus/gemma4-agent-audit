"""Tests for compaction_audit.py on synthetic ATIF step lists (served tokenizer and template; CPU only).

Run: PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest -q -p no:cacheprovider test_compaction_audit.py
"""
import json

import compaction_audit as ca

audit = ca.audit
SUB = ca.ROOT + "/subs/v043_M0_s1"  # read-only: agent.yaml (instruction, tools, generation config)
ca.init()


def step(i, source, event=None, metrics=None, tool=None, obs=None, message=""):
    s = {"step_id": i, "source": source, "message": message, "extra": {"event_type": event} if event else {}}
    if metrics is not None:
        s["metrics"] = metrics
    if tool:
        s["tool_calls"] = [{"tool_call_id": "t%d" % i, "function_name": tool[0], "arguments": tool[1]}]
    if obs is not None:
        s["observation"] = {"content": json.dumps(obs)}
    return s


def trace(prompts, big=None):
    """Task prompt, then one tool-call step per LLM call with recorded prompt_tokens; a large observation at index big."""
    steps = [step(1, "system", "system_instruction", message="sys"), step(2, "user", "task_prompt", message="Fix a.py.")]
    for k, p in enumerate(prompts):
        content = "x = 1\n" * 4000 if k == big else "ok"
        steps.append(step(k + 3, "agent", metrics={"prompt_tokens": p, "completion_tokens": 5},
                          tool=("read_file", {"filepath": "f%d.py" % k}), obs={"status": "ok", "content": content}))
    return {"steps": steps}


def test_crossings_follow_the_previous_calls_recorded_prompt_tokens():
    items, calls, bad = audit.trace_items(trace([100, 14336, 500, None, 14400, 300, 14335, 200]))
    assert not bad and [c["prompt_tokens"] for c in calls] == [100, 14336, 500, None, 14400, 300, 14335, 200]
    assert ca.crossings(calls) == [2, 5]  # at the threshold counts; None counts as 0; 14,335 does not


def classify(last):
    """Calls 1-7 recorded with their exact served lengths (call 7 above the threshold), call 8 recorded as last(...)."""
    plain = audit.trace_items(trace([0] * 8, big=5))
    lens = [len(s["ids"]) for s in ca.served(ca.TOK, plain[0], plain[1], SUB, {})]  # no crossing: plain growth
    assert lens[6] >= audit.THRESHOLD > lens[5]
    probe = audit.trace_items(trace(lens[:7] + [0], big=5))
    dropped = len(ca.served(ca.TOK, probe[0], probe[1], SUB, {}, None, True)[7]["ids"])
    items, calls, _ = audit.trace_items(trace(lens[:7] + [last(lens, dropped)], big=5))
    plan, events = ca.classify_compactions(ca.TOK, items, calls, SUB)
    assert [e["call"] for e in events] == [8] and events[0]["dropped_form_tokens"] == dropped
    return events[0], plan


def test_empty_summary_classified_only_by_an_exact_match():
    ev, plan = classify(lambda lens, dropped: dropped)
    assert ev["class"] == "empty_dropped" and plan == {7: "apply"} and ev["reliable_before"]
    ev, plan = classify(lambda lens, dropped: lens[7])  # vLLM saw the uncompacted prompt: no event appended
    assert ev["class"] == "no_compaction_event" and plan == {7: "skip"}
    ev, _ = classify(lambda lens, dropped: dropped + 60)  # something visible was added: a summary with text
    assert ev["class"] == "summary_present" and ev["inferred_summary_tokens"] > 0
    ev, _ = classify(lambda lens, dropped: dropped - 2)  # smaller than the empty form: contradiction
    assert ev["class"] == "undetermined" and ev["inferred_summary_tokens"] < 0


def test_classify_event_priority():
    assert ca.classify_event(True, True, 50, True) == "empty_dropped"
    assert ca.classify_event(False, True, 50, True) == "no_compaction_event"
    assert ca.classify_event(False, False, 50, True) == "summary_present"
    assert ca.classify_event(False, False, 50, False) == "undetermined"  # unexplained mismatch before the crossing
    assert ca.classify_event(False, False, None, True) == "undetermined"


def test_loop_onset_ordering_with_canonical_keys_and_metricless_steps():
    m = {"prompt_tokens": 10, "completion_tokens": 1}
    steps = [step(1, "user", "task_prompt", message="task"), step(2, "agent", metrics=m, tool=("read_file", {"filepath": "a"})),
             step(3, "agent", metrics=m, tool=("run_command", {"command": "ls"})),
             step(4, "agent", metrics=m, tool=("run_command", {"command": "pytest", "timeout": 60})),
             step(5, "agent", "thinking", metrics=m, message="think"),  # call 4 starts with a thought ...
             step(6, "agent", tool=("run_command", {"timeout": 60, "command": "pytest"})),  # ... and acts here
             step(7, "agent", "final", metrics=m, message="done?")]  # call 5: text only, no tool call
    steps += [step(8 + i, "agent", metrics=m, tool=("run_command", {"command": "pytest", "timeout": 60})) for i in range(4)]
    pos = ca.tool_call_positions({"steps": steps})
    assert [k for k, _, _ in pos] == [1, 2, 3, 4, 6, 7, 8, 9]
    lo = ca.loop_onset(pos)
    assert lo["onset"] == {"function": "run_command", "first_call": 3, "nth_call": 8}
    assert lo["max_repeat"] == 6 and lo["longest_run"] == [6, 3]
    assert ca.order(3, 6) == "before" and ca.order(6, 6) == "after" and ca.order(None, 6) is None


def test_user_turns_and_task_detection():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "Fix a.py."}]},
            {"role": "assistant", "content": None}, {"role": "user", "content": "For context:[model] said: x"}]
    users = ca.user_texts(msgs)
    assert users == ["Fix a.py.", "For context:[model] said: x"] and ca.user_texts(msgs[2:3]) == []
