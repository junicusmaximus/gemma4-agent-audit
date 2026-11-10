#!/usr/bin/env python3
"""Unit tests of sft/train_qatbase_v053.py (train env, CPU, no model, no GPU): the frozen trainers are bound by
path to one module object, the fixture cut and group order, base-override-only and every refusal gate.
Run with PYTHONDONTWRITEBYTECODE=1 and TMPDIR inside analysis/qat-transfer-v053/ (temporary dirs are removed)."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
spec = importlib.util.spec_from_file_location("train_qatbase_v053", ROOT + "/sft/train_qatbase_v053.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

import numpy as np  # noqa: E402

M0_ARGS = {"rank": 16, "alpha": 16, "lr": 0.0001, "epochs": 1, "max_seq_len": 16384, "grad_accum": 8,
           "warmup_ratio": 0.05, "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 20261103, "variant": "M0",
           "base": ROOT + "/models/gemma-4-e4b-it", "out": ROOT + "/sft/runs/A1_M0_v035", "device": "cuda",
           "served_base_name": "google/gemma-4-e4b-it-qat-w4a16-ct"}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def synthetic(tmp):
    """5 trajectories (chunk 0: prompt, completion 0 ending in 50, tool text, completion 1; plus a chunk 1) and a
    ledger with 2 H-gen rows and the 5 T-fit rows built the way s0_items.py builds them."""
    rng = np.random.default_rng(0)
    seqs, rows = [], [{"item_id": "hgen:x%d" % i, "kind": "H-gen", "ids": [1, 2, 3]} for i in range(2)]
    for t in range(5):
        p, n = 20 + t, 7 + t
        prompt = rng.integers(200, 1000, p).tolist()
        comp0 = rng.integers(200, 1000, n - 1).tolist() + [50]
        tail_ctx, comp1 = rng.integers(200, 1000, 9).tolist(), rng.integers(200, 1000, 4).tolist()
        ids = np.asarray(prompt + comp0 + tail_ctx + comp1, dtype=np.int32)
        lab = np.asarray([-100] * p + comp0 + [-100] * 9 + comp1, dtype=np.int32)
        seqs.append(("traj%d" % t, ids, lab))
        seqs.append(("traj%d" % t, ids[:10].copy(), np.full(10, -100, dtype=np.int32)))
        rows.append({"item_id": "tfit:traj%d" % t, "kind": "T-fit", "source": {"traj_id": "traj%d" % t, "chunk": 0},
                     "target_span": [p, p + n], "ids": prompt + comp0})
    path = os.path.join(tmp, "ledger.jsonl")
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return seqs, path


class Imports(unittest.TestCase):
    def test_trainers_bound_to_one_module(self):
        original, codex = w.load_trainers()
        self.assertIs(codex.original, original)
        self.assertIs(sys.modules["train_lora"], original)
        for name in ("build_model", "save_adapter", "check_gpu", "sequence_loss", "lr_at"):
            self.assertTrue(callable(getattr(original, name)), name)
        for name in ("verify_pair", "fixed_denominator_train"):
            self.assertTrue(callable(getattr(codex, name)), name)
        self.assertEqual(original.SERVED_BASE, M0_ARGS["served_base_name"])


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wq053-")
        self.seqs, self.ledger = synthetic(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cut_after_completion0(self):
        out, items = w.fixture_tfit5(self.seqs, self.ledger, sha(self.ledger))
        self.assertEqual([o[0] for o in out], ["traj%d" % t for t in range(5)])
        for (tid, ids, lab), it in zip(out, items):
            s, e = it["target_span"]
            self.assertEqual(ids.tolist(), it["ids"])
            self.assertEqual(lab.tolist(), [-100] * s + it["ids"][s:e])
            self.assertEqual(int(ids[-1]), 50)

    def test_mismatch_and_changed_ledger_refused(self):
        with self.assertRaises(ValueError):
            w.fixture_tfit5(self.seqs, self.ledger, "0" * 64)
        with open(self.ledger) as f:
            rows = [json.loads(x) for x in f]
        rows[3]["ids"][0] += 1
        with open(self.ledger, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(ValueError):
            w.fixture_tfit5(self.seqs, self.ledger, sha(self.ledger))

    def test_groups_follow_trainer_order(self):
        out, _ = w.fixture_tfit5(self.seqs, self.ledger, sha(self.ledger))
        groups = w.fixture_groups(out, 24, 5, 20261103)
        rng, order = random.Random(20261103), []
        for _ in range(24):
            perm = list(range(5))
            rng.shuffle(perm)
            order += perm
        self.assertEqual(len(groups), 24)
        self.assertEqual(sum((g["sequence_indices"] for g in groups), []), order)
        total = sum(int((s[2][1:] != -100).sum()) for s in out)
        self.assertEqual({g["original_denominator"] for g in groups}, {total})
        self.assertEqual(total, sum(7 + t for t in range(5)))


class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wq053-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_overrides_base_and_out_only(self):
        args = argparse.Namespace(**dict(M0_ARGS, base="/x/dequant", out="/x/F"))
        self.assertEqual(w.check_overrides(args, M0_ARGS, fixture=False), ["base", "out"])
        args.epochs, args.grad_accum = 24, 5
        self.assertEqual(w.check_overrides(args, M0_ARGS, fixture=True), ["base", "epochs", "grad_accum", "out"])
        with self.assertRaises(ValueError):
            w.check_overrides(args, M0_ARGS, fixture=False)
        args.lr = 2e-4
        with self.assertRaises(ValueError):
            w.check_overrides(args, M0_ARGS, fixture=True)

    def test_verify_base(self):
        base = os.path.join(self.tmp, "dq")
        os.mkdir(base)
        for name, text in (("model-00001-of-00001.safetensors", "w"), ("config.json", "{}")):
            with open(os.path.join(base, name), "w") as f:
                f.write(text)
        manifest = {"shards": [{"name": "model-00001-of-00001.safetensors",
                                "sha256": sha(os.path.join(base, "model-00001-of-00001.safetensors"))}],
                    "files_sha256": {"config.json": sha(os.path.join(base, "config.json"))}, "counts": {"x": 1}}
        with open(os.path.join(base, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        bases = {"dequant": base, "bf16": base}
        self.assertEqual(w.verify_base("dequant", bases)["counts"], {"x": 1})
        with open(os.path.join(base, "extra.txt"), "w") as f:
            f.write("x")
        with self.assertRaises(ValueError):
            w.verify_base("dequant", bases)
        os.remove(os.path.join(base, "extra.txt"))
        with open(os.path.join(base, "config.json"), "w") as f:
            f.write("{ }")
        with self.assertRaises(ValueError):
            w.verify_base("dequant", bases)
        with open(os.path.join(base, "model.safetensors"), "w") as f:
            f.write("bf16")
        self.assertEqual(w.verify_base("bf16", bases, sha(os.path.join(base, "model.safetensors")))["kind"], "bf16")
        with self.assertRaises(ValueError):
            w.verify_base("bf16", bases, "0" * 64)

    def test_init_gate(self):
        self.assertIsNone(w.check_init("a" * 64, 258, "a" * 64))
        for digest, n in (("b" * 64, 258), ("a" * 64, 257)):
            with self.assertRaises(SystemExit):
                w.check_init(digest, n, "a" * 64)

    def test_cli_guards(self):
        out = os.path.join(self.tmp, "F")
        for argv in (["--out", out, "--execute"], ["--out", out, "--epochs", "3"], ["--fixture", "tfit5", "--out", out],
                     ["--out", out, "--base", "bf16"]):
            with self.assertRaises(SystemExit, msg=str(argv)):
                w.parse(argv)
        opts = w.parse(["--fixture", "tfit5", "--epochs", "24", "--grad-accum", "5", "--out", out, "--execute"])
        self.assertEqual((opts.base, opts.epochs, opts.grad_accum, opts.out), ("dequant", 24, 5, out))
        self.assertTrue(w.parse(["--out", out, "--execute", "--m0q-go"]).m0q_go)

    def test_prepare_refuses_existing_out(self):
        out = os.path.join(self.tmp, "F")
        os.mkdir(out)
        with self.assertRaises(FileExistsError):
            w.prepare(w.parse(["--fixture", "tfit5", "--epochs", "1", "--grad-accum", "5", "--out", out]), None, None)


if __name__ == "__main__":
    unittest.main(verbosity=1)
