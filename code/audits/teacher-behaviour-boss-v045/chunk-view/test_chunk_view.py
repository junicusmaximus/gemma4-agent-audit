#!/usr/bin/env python3
"""Tests for chunk_view.py on tiny synthetic id/label sequences (pytest-compatible, also runnable as a plain script)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunk_view as cv  # noqa: E402

NL, Q = chr(10), '<|"|>'
CONTROL = {2: "<bos>", 105: "<|turn>", 106: "<turn|>", 107: NL, 4368: "model", 100: "<|channel>", 45518: "thought",
           101: "<channel|>", 48: "<|tool_call>", 49: "<tool_call|>", 50: "<|tool_response>", 51: "<tool_response|>"}
OK, SILENT, BOOM = ('{"status": "ok", "stdout": "x", "exit_code": 0}', '{"status": "ok", "stdout": "", "exit_code": 0}',
                    '{"status": "ok", "stdout": "", "stderr": "boom", "exit_code": 1}')


class Vocab:
    """Fake tokenizer: the control ids of the real one plus one new id per distinct piece of text."""

    def __init__(self):
        self.pieces, self.ids = dict(CONTROL), {v: k for k, v in CONTROL.items()}

    def encode(self, text):
        if text not in self.ids:
            self.ids[text] = 1000 + len(self.ids)
            self.pieces[self.ids[text]] = text
        return self.ids[text]

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def header(v):
    return [2, 105, v.encode("system and task"), 106, 107, 105, 4368, 107]


def call(v, name, arg, thought="", opener=False):
    """Supervised ids: [opener] [thought] <channel|> <|tool_call>call:NAME{command:ARG}<tool_call|> <|tool_response>."""
    body = "call:%s{command:%s%s%s}" % (name, Q, arg, Q)
    return ([100, 45518, 107] if opener else []) + ([v.encode(thought)] if thought else []) + [101, 48, v.encode(body), 49, 50]


def answer(v):
    return [101, v.encode("done"), 106]


def reply(v, name, body):
    """Unsupervised ids: the tool response of a call, ending with the thought opener of the next completion."""
    return [v.encode("response:%s{value:%s%s%s}" % (name, Q, body, Q)), 51, 100, 45518, 107]


def chunk(parts, steps, chain=0):
    """A record of M0.jsonl from (ids, supervised) parts."""
    ids = [i for part, _ in parts for i in part]
    labels = [i if sup else -100 for part, sup in parts for i in part]
    return {"chain": chain, "steps": steps, "n_tokens": len(ids), "n_loss_tokens": sum(x != -100 for x in labels),
            "input_ids": ids, "labels": labels}


def first_chunk(v):
    parts = [(header(v), False), (call(v, "run_command", "ls", "plan", True), True), (reply(v, "run_command", OK), False),
             (call(v, "read_file", "a.py"), True)]
    return chunk(parts, [0, 1])


def orphan_chunk(v, last=None):
    parts = [(header(v), False), (reply(v, "run_command", OK), False), (call(v, "edit_file", "x"), True),
             (reply(v, "edit_file", OK), False), (last or call(v, "submit_patch", "p"), True)]
    return chunk(parts, [5, 6])


def test_label_runs():
    assert cv.label_runs([-100, -100, 5, 6, -100, 7, -100]) == [(2, 4), (5, 6)]
    assert cv.label_runs([]) == [] and cv.label_runs([-100, -100]) == [] and cv.label_runs([1, 2]) == [(0, 2)]
    v = Vocab()
    c = first_chunk(v)
    assert len(cv.label_runs(c["labels"])) == 2 and cv.header_end(c["input_ids"]) == len(header(v))


def test_chunk_type_and_start_kind():
    v = Vocab()
    chain_start = chunk([(header(v), False), (answer(v), True)], [7, 7], chain=1)  # summarizer-like: steps[0] > 0, no orphan
    opener_only = chunk([(header(v), False), ([100, 45518, 107], False), (call(v, "read_file", "b.py"), True)], [9, 9], chain=2)
    replayed = chunk([(header(v), False), (call(v, "read_file", "old", "t", True) + reply(v, "read_file", OK), False),
                      (call(v, "read_file", "b.py"), True)], [11, 11], chain=2)  # history re-serialized in a continuation header
    rows = [cv.analyse_chunk(c, v.decode) for c in (first_chunk(v), orphan_chunk(v), chain_start, opener_only, replayed)]
    assert [(r["type"], r["start_kind"]) for r in rows] == [("first", "direct"), ("later", "orphan_response"), ("later", "direct"),
                                                            ("later", "opener_only"), ("later", "replayed_history")]
    assert cv.chunk_type([0, 3]) == "first" and cv.chunk_type([1, 2]) == "later" and cv.chunk_type([4, 9]) == "later"
    groups = {k: len(g) for k, g in cv.group_rows(rows).items()}
    assert groups == {"first": 1, "later": 4, "all": 5, "later_orphan": 1, "later_chain_start": 3}
    check = cv.type_check(rows)
    assert check["later_by_steps_rule"] == 4 and check["later_by_decoded_orphan_response"] == 1
    assert check["mismatch_later_but_no_orphan_response"] == 3 and check["mismatch_first_but_orphan_response"] == 0
    assert [r["history_free_actions"] for r in rows] == [0, 1, 0, 0, 0]  # only the orphan start lacks its preceding call


def test_history_free_counting():
    v = Vocab()
    ends_with_answer = orphan_chunk(v, last=answer(v))
    final_only = chunk([(header(v), False), (reply(v, "run_command", OK), False), (answer(v), True)], [8, 8])
    rows = [cv.analyse_chunk(c, v.decode) for c in (first_chunk(v), orphan_chunk(v), ends_with_answer, final_only)]
    assert [r["history_free_actions"] for r in rows] == [0, 1, 1, 0]  # a first completion without a call is not an action
    totals = cv.totals_of(rows)
    assert totals["calls"] == 2 + 2 + 1 + 0 and totals["chunks"] == 4
    later = cv.summarize(cv.group_rows(rows)["later"], totals)
    assert later["history_free_actions"] == 2 and later["history_free_share_of_all_calls"] == 2 / 5
    assert later["history_free_share_of_group_calls"] == 2 / 3
    assert cv.summarize(cv.group_rows(rows)["first"], totals)["history_free_actions"] == 0
    assert later["n_loss_tokens"] + cv.summarize(cv.group_rows(rows)["first"], totals)["n_loss_tokens"] == totals["loss_tokens"]


def test_loop_and_m5_within_chunk():
    v = Vocab()
    parts = [(header(v), False)]
    for _ in range(5):
        parts += [(call(v, "run_command", "pytest"), True), (reply(v, "run_command", BOOM), False)]
    row = cv.analyse_chunk(chunk(parts + [(call(v, "run_command", "pytest"), True)], [0, 5]), v.decode)
    assert row["status"] == "ok" and row["n_calls"] == 6 and row["max_repeat"] == 6 and row["loop"] is True
    assert row["first_action"] == "other" and row["nonrecovery"]["silent"]["eligible"] == 0
    assert row["nonrecovery"]["error"] == {"eligible": 5, "identical": 5, "repeat_rate": 1.0}
    alt = [(header(v), False)]
    for k in range(4):
        alt += [(call(v, "run_command", "ab"[k % 2]), True)] + ([(reply(v, "run_command", SILENT), False)] if k < 3 else [])
    row = cv.analyse_chunk(chunk(alt, [0, 3]), v.decode)
    assert row["loop"] is False and row["max_repeat"] == 2
    assert row["nonrecovery"]["silent"] == {"eligible": 3, "identical": 0, "repeat_rate": 0.0}


def test_thought_presence_controls_and_malformed():
    v = Vocab()
    row = cv.analyse_chunk(first_chunk(v), v.decode)
    assert row["status"] == "ok" and row["n_thinking"] == 1 and [c["thought"] for c in row["completions"]] == [True, False]
    assert row["completions"][0]["controls"]["<|channel>"] == 1 and row["completions"][0]["calls"][0]["arguments"] == {"command": "ls"}
    body = v.encode("call:read_file{command:" + Q + "a" + Q + "}")
    no_close = cv.analyse_chunk(chunk([(header(v), False), ([48, body, 49, 50], True)], [0, 0]), v.decode)
    assert no_close["status"] == "malformed" and any("no channel close marker" in e for e in no_close["errors"])
    cut = cv.analyse_chunk(chunk([(header(v), False), ([101, v.encode("partial thought")], True)], [0, 0]), v.decode)
    assert any("stop token missing" in e for e in cut["errors"]) and cut["n_calls"] == 0 and cut["first_action"] == "no_tool_call"


def test_roundtrip_expected_text_and_first_difference():
    v = Vocab()
    c = orphan_chunk(v)
    head, res0 = v.decode(header(v)), v.decode(reply(v, "run_command", OK))
    comp5, res5, comp6 = (v.decode(call(v, "edit_file", "x")), v.decode(reply(v, "edit_file", OK)), v.decode(call(v, "submit_patch", "p")))
    steps = [(0, "c%d" % i, "r%d" % i) for i in range(4)] + [(0, "prev", res0), (0, comp5, res5), (0, comp6, "TAIL")]
    expected = cv.expected_chunk_text(c, {0: head}, steps)
    assert expected == v.decode(c["input_ids"]) == head + res0 + comp5 + res5 + comp6
    assert cv.analyse_chunk(c, v.decode, expected)["roundtrip_diff"] is None
    broken = cv.analyse_chunk(c, v.decode, expected[:20] + "X" + expected[21:])
    assert broken["roundtrip_diff"] == 20 and broken["status"] == "malformed"
    chain_start = chunk([(header(v), False), (answer(v), True)], [7, 7], chain=1)  # the previous step belongs to chain 0
    assert cv.expected_chunk_text(chain_start, {1: head}, [(0, "a", "b")] * 7 + [(1, v.decode(answer(v)), "TAIL")]) == head + v.decode(answer(v))
    assert cv.first_difference("abc", "abd") == 2 and cv.first_difference("abc", "abc") is None and cv.first_difference("ab", "abc") == 2 and cv.first_difference("abc", "ab") == 2


def run_all():
    names = sorted(name for name in globals() if name.startswith("test_"))
    failed = 0
    for name in names:
        try:
            globals()[name]()
        except Exception as exc:  # report every failing test, then exit non-zero
            failed += 1
            print("FAIL %s: %r" % (name, exc))
    print("%d passed, %d failed" % (len(names) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_all())
