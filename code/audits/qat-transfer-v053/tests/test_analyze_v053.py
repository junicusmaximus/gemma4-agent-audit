#!/usr/bin/env python3
"""Unit tests of s3_analyze_v053.py decision rules on synthetic score rows (train env, CPU, seconds)."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
OUT = "PROJECT_ROOT/gemma4/analysis/qat-transfer-v053"
spec = importlib.util.spec_from_file_location("s3_analyze_v053", OUT + "/s3_analyze_v053.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

TARGET = list(range(1000, 1043))  # 43 target ids: scaffold 0-3, text 4-41, control 42


def ledger():
    spans = {}
    for i in range(53):
        kind, p = ("H-gen", 100 + i) if i < 48 else ("T-fit", 200 + i)
        k = ("hgen:%d" if kind == "H-gen" else "tfit:%d") % i
        spans[k] = {"item_id": k, "kind": kind, "target_ids": TARGET, "target_span": [p, p + 43],
                    "scaffold_span": [p, p + 4], "text_span": [p + 4, p + 42], "control_span": [p + 42, p + 43]}
    return spans


def rows(spans, stack, cell, layout, lp, kinds=("H-gen", "T-fit"), edit=None):
    """One row per item; log-prob lp at every position, rank 1, top-1 = target; edit(item_id, row) may change it."""
    out = []
    for k, it in spans.items():
        if it["kind"] not in kinds:
            continue
        r = {"stack": stack, "cell": cell, "layout": layout, "item_id": k, "target_ids": TARGET,
             "logprob": [lp] * 43, "rank": [1] * 43, "top5_ids": [[t, 1, 2, 3, 4] for t in TARGET]}
        if edit:
            edit(k, r)
        out.append(r)
    return out


class Stage1(unittest.TestCase):
    def setUp(self):
        self.spans = ledger()
        self.bf = rows(self.spans, "hf", "base", "single", -2.5) + rows(self.spans, "hf", "m0", "single", -0.1)   # effect 2.4
        self.vq = rows(self.spans, "vllm", "base", "sequential", -1.5) + rows(self.spans, "vllm", "m0", "sequential", -0.5)

    def dq(self, m0_lp):
        return (rows(self.spans, "hf_dequant", "base", "single", -1.5) + rows(self.spans, "hf_dequant", "zerob", "single", -1.5)
                + rows(self.spans, "hf_dequant", "m0", "single", m0_lp) + rows(self.spans, "hf_dequant", "base", "batch8", -1.5))

    def test_support_and_parity(self):
        res = an.stage1(self.dq(-0.54), self.bf, self.vq, self.spans, 0.2, "test")   # effect 0.96 -> R 0.4
        t = res["transfer_1b"]
        self.assertAlmostEqual(t["R"], 0.4)
        self.assertTrue(t["label"].startswith("R <= 0.6"))
        self.assertEqual(res["decision"], "continue")
        self.assertTrue(res["parity_1a"]["pass"])
        self.assertAlmostEqual(res["parity_1a"]["threshold"], 0.3)
        self.assertEqual(res["parity_1a"]["argmax_agreement_hgen_text"]["share"], 1.0)
        self.assertEqual(res["zero_b_hf_dq"]["max_abs"], 0.0)

    def test_refute_and_between(self):
        self.assertEqual(an.stage1(self.dq(0.9), self.bf, self.vq, self.spans, None, "absent")["decision"], "stop")  # R 1.0
        res = an.stage1(self.dq(0.18), self.bf, self.vq, self.spans, None, "absent")                                # R 0.7
        self.assertTrue(res["transfer_1b"]["label"].startswith("0.6 < R < 0.9"))
        self.assertEqual(res["parity_1a"]["threshold"], 0.05)

    def test_parity_fails_on_gap_or_argmax(self):
        far = rows(self.spans, "hf_dequant", "base", "single", -1.2) + rows(self.spans, "hf_dequant", "m0", "single", -0.5)
        self.assertFalse(an.stage1(far, self.bf, self.vq, self.spans, 0.1, "t")["parity_1a"]["pass"])   # gap 0.3 > 0.15

        def flip(k, r):
            r["top5_ids"][10] = [7, 1, 2, 3, 4]
        flipped = rows(self.spans, "hf_dequant", "base", "single", -1.5, edit=flip) + rows(self.spans, "hf_dequant", "m0", "single", -0.5)
        res = an.stage1(flipped, self.bf, self.vq, self.spans, 0.1, "t")
        self.assertAlmostEqual(res["parity_1a"]["argmax_agreement_hgen_text"]["share"], 37 / 38)
        self.assertFalse(res["parity_1a"]["pass"])

    def test_engine_floor_lookup(self):
        tmp = tempfile.mkdtemp(prefix="an053-")
        try:
            path, scores, hf = (os.path.join(tmp, n) for n in ("report.json", "scores.jsonl", "hf.jsonl"))
            self.assertIsNone(an.engine_floor_from_v048(None, scores, hf, path)[0])
            with open(path, "w") as f:
                json.dump({"v22_addon": {"weights_vs_engine": {"spans": {"text": {"base_engine_unq_minus_hf": {"p95_abs": 0.07}}}}}}, f)
            self.assertEqual(an.engine_floor_from_v048(None, scores, hf, path)[0], 0.07)
            spans = ledger()
            with open(scores, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows(spans, "vllm_unq", "base", "sequential", -1.0, ("H-gen",))
                                + rows(spans, "vllm_nopc", "base", "sequential", -9.0)))
            with open(hf, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows(spans, "hf", "base", "single", -1.2)))
            floor, source = an.engine_floor_from_v048(spans, scores, hf, path)
            self.assertAlmostEqual(floor, 0.2)
            self.assertIn("n=48", source)
        finally:
            shutil.rmtree(tmp)


class Stage2(unittest.TestCase):
    def setUp(self):
        self.spans = ledger()
        self.gaps = {k: 0.01 for k in self.spans}
        self.hf = rows(self.spans, "hf_dequant", "base", "single", -2.0) + rows(self.spans, "hf_dequant", "fdq", "single", -0.5)

    def vq(self, fdq_lp=-0.55, edit=None):
        return (rows(self.spans, "vllm_qat_nopc", "base", "sequential", -2.0)
                + rows(self.spans, "vllm_qat_nopc", "fdq", "sequential", fdq_lp, edit=edit))

    def test_all_pass(self):
        res = an.pass_rules(self.hf, self.vq(), self.spans, self.gaps)
        self.assertEqual(res["verdict"], "PASS")
        self.assertAlmostEqual(res["E_mean"]["T-fit"]["hf"], 1.5)

    def test_each_rule_can_fail(self):
        def hgen4(k, r):
            if k in ("hgen:0", "hgen:1", "hgen:2", "hgen:3"):
                r["logprob"] = [-0.8] * 43
        self.assertIn("R2", an.pass_rules(self.hf, self.vq(edit=hgen4), self.spans, self.gaps)["verdict"])
        self.assertIn("R1", an.pass_rules(self.hf, self.vq(-0.75), self.spans, self.gaps)["verdict"])

        def depth(k, r):
            r["logprob"][19] = -2.0
            r["top5_ids"][19] = [5, 1, 2, 3, 4]
        res = an.pass_rules(self.hf, self.vq(edit=depth), self.spans, self.gaps)
        self.assertIn("R3", res["verdict"])
        self.assertNotIn("R1", res["verdict"])

        def acc(k, r):
            r["rank"][10] = 2
        self.assertIn("R4", an.pass_rules(self.hf, self.vq(edit=acc), self.spans, self.gaps)["verdict"])

    def test_fbf_control_and_validity(self):
        hf = rows(self.spans, "hf_bf16", "base", "single", -2.0) + rows(self.spans, "hf_bf16", "fbf", "single", 0.0)
        vq = rows(self.spans, "vllm_qat_nopc", "base", "sequential", -2.0) + rows(self.spans, "vllm_qat_nopc", "fbf", "sequential", -1.0)
        self.assertTrue(an.fbf_control(hf, vq, self.spans)["T-fit"]["reproduces_gap"])
        vq2 = rows(self.spans, "vllm_qat_nopc", "base", "sequential", -2.0) + rows(self.spans, "vllm_qat_nopc", "fbf", "sequential", -0.1)
        self.assertFalse(an.fbf_control(hf, vq2, self.spans)["T-fit"]["reproduces_gap"])
        self.assertTrue(an.validity(self.hf, self.spans, "hf_dequant", "fdq")["valid"])
        weak = rows(self.spans, "hf_dequant", "base", "single", -2.0) + rows(self.spans, "hf_dequant", "fdq", "single", -1.2)
        self.assertFalse(an.validity(weak, self.spans, "hf_dequant", "fdq")["valid"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
