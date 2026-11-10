"""Tests for teacher_behaviour.py on small synthetic records (pytest, or plain: python3 test_teacher_behaviour.py)."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_behaviour as tb  # noqa: E402

NL = chr(10)
Q = tb.QUOTE
READ_A = "filepath:" + Q + "a.py" + Q + ",start_line:1"
READ_A_REORDERED = "start_line:1,filepath:" + Q + "a.py" + Q
READ_B = "filepath:" + Q + "b.py" + Q + ",start_line:1"
READ_C = "filepath:" + Q + "c.py" + Q
PAYLOADS = {
    "normal": {"status": "ok", "stdout": "text", "stderr": "", "exit_code": 0},
    "silent": {"status": "ok", "stdout": "", "stderr": "", "exit_code": 0},
    "error": {"status": "ok", "stdout": "", "stderr": "boom", "exit_code": 1},
    "failed": {"status": "error", "stdout": "", "stderr": "", "exit_code": 0},
}


def step(name, args="", kind="normal", thought=""):
    """One model step: a call of `name` whose result has the given kind (name None: final text-only answer)."""
    return {"name": name, "args": args, "kind": kind, "thought": thought}


def completion(spec, first):
    call = "<|tool_call>call:%s{%s}<tool_call|>" % (spec["name"], spec["args"]) if spec["name"] else ""
    stop = "<|tool_response>" if spec["name"] else "<turn|>"
    return (tb.CHANNEL_OPEN if first else "") + spec["thought"] + tb.CHANNEL_CLOSE + call + stop


def result(spec):
    payload = json.dumps(PAYLOADS[spec["kind"]])
    return "response:%s{value:%s%s%s}<tool_response|><|channel>thought%s" % (spec["name"], Q, payload, Q, NL)


def chain_segments(index, specs, header):
    """Header, then completion/result pairs; the last step of a chain has no result segment."""
    segments = [{"chain": index, "text": header, "train": False}]
    for k, spec in enumerate(specs):
        segments.append({"chain": index, "text": completion(spec, k == 0), "train": True})
        if k + 1 < len(specs):
            segments.append({"chain": index, "text": result(spec), "train": False})
    return segments


def make_record(specs, think=True):
    header = tb.SYSTEM_START + (tb.THINK_TOKEN + NL if think else "") + "prompt<turn|>" + NL + "<|turn>model" + NL
    return {"traj_id": "t1", "task_id": "task", "tier": "external_B", "generating_model": "m", "n_steps": len(specs),
            "tool_calls": sum(1 for s in specs if s["name"]) - int(bool(specs[-1]["name"])), "n_chains": 1,  # answered calls only
            "text_segments": chain_segments(0, specs, header)}


def row_of(specs, think=True):
    return tb.analyse_trajectory(make_record(specs, think))


def test_loop_threshold_with_canonical_argument_order():
    calls = [READ_A, READ_A_REORDERED, READ_A, READ_A_REORDERED, READ_A]
    looped = row_of([step("read_file", a) for a in calls] + [step("submit_patch")])
    assert looped["status"] == "ok"
    assert (looped["max_repeat"], looped["loop"]) == (5, True)  # key order does not matter
    assert abs(looped["distinct_ratio"] - 2 / 6) < 1e-12
    calls = [READ_A, READ_A_REORDERED, READ_A, READ_A_REORDERED, READ_B]
    below = row_of([step("read_file", a) for a in calls] + [step("submit_patch")])
    assert (below["max_repeat"], below["loop"]) == (4, False)  # one below the threshold of 5
    assert looped["submit_called"] and below["opener"] == "other"


def test_opener_classes():
    find = "command:" + Q + "find /workspace -maxdepth 2 -not -path '*/.*' -print" + Q
    wrapper = "command:" + Q + "__public_sft_output=$(ls)" + Q
    shebang = "#!/usr/bin/env python3" + NL
    repro = "content:" + Q + shebang + "# Reproduce the issue" + NL + "print(1)" + Q + ",filepath:" + Q + "r.py" + Q
    plain_script = "content:" + Q + shebang + "print(1)" + Q + ",filepath:" + Q + "r.py" + Q
    final, later = step(None), step("write_file", repro)  # a later repro-like call must not change the class of the first call
    expected = [([step("run_command", find), later, final], "find_opener"),
                ([step("run_command", wrapper), later, final], "wrapper_opener"),
                ([step("write_file", repro), step("read_file", READ_A), final], "repro_script_opener"),
                ([step("write_file", plain_script), later, final], "other"),
                ([step("read_file", READ_A), later, final], "other"),
                ([final], "no_tool_call")]
    for specs, opener in expected:
        assert row_of(specs)["opener"] == opener, opener


def test_m5_strata_and_repeat():
    specs = [step("read_file", READ_A, "error"), step("read_file", READ_A, "error"),
             step("read_file", READ_B, "silent"), step("read_file", READ_B, "normal"),
             step("read_file", READ_C, "normal"), step("read_file", READ_C, "failed"),
             step("submit_patch")]  # calls 1-2: A A (error), 3-4: B B (silent, normal), 5-6: C C (normal, failed status)
    nonrecovery = row_of(specs)["nonrecovery"]
    assert nonrecovery["error"] == {"eligible": 3, "identical": 1, "repeat_rate": 1 / 3}  # steps 1, 2 and 6 (failed status)
    assert nonrecovery["silent"] == {"eligible": 1, "identical": 1, "repeat_rate": 1.0}
    assert nonrecovery["normal"] == {"eligible": 2, "identical": 1, "repeat_rate": 0.5}
    assert nonrecovery["unknown_excluded"] == 0


def test_argument_parser_follows_the_serving_parser():
    text = "a:" + Q + "x,y" + Q + ",n:12,flag:true,nested:{k:" + Q + "v" + Q + ",arr:[" + Q + "p" + Q + ",3]},list:[" + Q + "q" + Q + ",{x:1}]"
    assert tb.parse_items(text, True) == {"a": "x,y", "n": "12", "flag": "true", "nested": {"k": "v", "arr": ["p", "3"]},
                                          "list": ["q", {"x": "1"}]}  # bare values stay raw strings
    assert tb.parse_items('start_line":70,end_line":80', True) == {'start_line"': "70", 'end_line"': "80"}  # malformed but accepted
    assert tb.parse_items("a:" + Q + "unterminated", True) == {"a": "unterminated"}
    assert tb.parse_items("a : " + Q + "x" + Q + " , " + Q + "k" + Q + ":1", True) == {"a": "x", "k": "1"}  # spaces, quoted key
    calls, errors = tb.parse_calls("<|tool_call>call:run_command{command:" + Q + "ls" + Q + "}<|tool_response>")  # call never closed
    assert calls == [{"function_name": "run_command", "arguments": {"command": "ls"}}] and errors == []


def test_thought_presence_counts_nonblank_thought_blocks():
    specs = [step("read_file", READ_A, thought="plan"), step("read_file", READ_B), step("read_file", READ_C, thought="  " + NL),
             step("submit_patch")]
    row = row_of(specs)
    assert (row["n_steps"], row["n_thinking_steps"], row["think_rate"]) == (4, 1, 0.25)
    assert tb.thought_text(tb.CHANNEL_OPEN + "abc" + tb.CHANNEL_CLOSE + "x") == "abc"
    assert tb.thought_text("abc" + tb.CHANNEL_CLOSE + "x") == "abc"  # later steps: the opener sits in the previous result segment
    assert row["first_header_has_think_token"] is True and row_of(specs, think=False)["first_header_has_think_token"] is False


def test_summarizer_chain_and_chain_final_call_have_no_result():
    header = tb.SYSTEM_START + tb.THINK_TOKEN + NL + "prompt<turn|>" + NL + "<|turn>model" + NL
    summary_header = tb.SUMMARIZER_START + NL + "<|turn>user" + NL + "history<turn|>" + NL + "<|turn>model" + NL
    segments = (chain_segments(0, [step("read_file", READ_A), step("read_file", READ_B)], header)
                + chain_segments(1, [step(None, thought="summary")], summary_header)
                + chain_segments(2, [step("read_file", READ_B), step("submit_patch")], header))
    record = {**make_record([step("submit_patch")]), "n_steps": 5, "tool_calls": 3, "n_chains": 3, "text_segments": segments}
    row = tb.analyse_trajectory(record)
    assert row["status"] == "ok" and row["n_summarizer_steps"] == 1
    assert row["nonrecovery"]["unknown_excluded"] == 1  # B (last call of chain 0) -> B: its result is not in the chain
    assert row["nonrecovery"]["normal"]["eligible"] == 2


def test_format_problems_are_reported_not_skipped():
    record = make_record([step("read_file", READ_A), step("submit_patch")])
    assert any("completions" in e for e in tb.analyse_trajectory({**record, "n_steps": 3})["errors"])
    segments = [{**s} for s in record["text_segments"]]
    segments[1] = {**segments[1], "text": tb.CHANNEL_CLOSE + "<|tool_call>call:<tool_call|><|tool_response>"}
    row = tb.analyse_trajectory({**record, "text_segments": segments})
    assert row["status"] == "malformed" and any("call:NAME" in e for e in row["errors"])
    try:
        tb.analyse_trajectory({"traj_id": "x"})
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("an unusable record must raise ValueError")


def test_cli_end_to_end_on_a_tiny_file():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        keep = make_record([step("read_file", READ_A), step("submit_patch")])
        other = {**make_record([step("submit_patch")]), "traj_id": "t2"}
        (folder / "data.jsonl").write_bytes((json.dumps(keep) + NL + json.dumps(other) + NL).encode("utf-8"))
        (folder / "arms.json").write_bytes(json.dumps({"arm_x": ["t1"], "arm_y": ["t1", "t2"]}).encode("utf-8"))
        argv = ["--arm", "arm_x", "--arm", "arm_y", "--data", str(folder / "data.jsonl"),
                "--arms-file", str(folder / "arms.json"), "--out-dir", str(folder / "out")]
        assert tb.main(argv) == 0
        report = json.loads((folder / "out" / "report.json").read_text(encoding="utf-8"))
        assert [report["arms"][a]["n"] for a in ("arm_x", "arm_y")] == [1, 2]
        assert report["arms"]["arm_y"]["submit_rate"] == 1.0 and report["malformed_trajectories"] == 0
        assert len((folder / "out" / "per_trajectory.jsonl").read_text(encoding="utf-8").splitlines()) == 2
        assert "| n trajectories (well-formed) | 1 | 2 |" in (folder / "out" / "report.md").read_text(encoding="utf-8")
        assert tb.main(argv[:-4] + ["--arms-file", str(folder / "nope.json")]) == 2  # unreadable input -> status 2


if __name__ == "__main__":  # fallback runner when pytest is not installed
    tests = sorted((name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print("FAIL", name, type(exc).__name__, exc)
    print("%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
