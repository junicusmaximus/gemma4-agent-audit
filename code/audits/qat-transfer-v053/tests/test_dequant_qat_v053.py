#!/usr/bin/env python3
"""Unit tests of tools/dequant_qat_v053.py on synthetic checkpoints (swegemma env, CPU, seconds).
Run with PYTHONDONTWRITEBYTECODE=1 and TMPDIR inside analysis/qat-transfer-v053/ (temporary dirs are removed)."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
spec = importlib.util.spec_from_file_location("dequant_qat_v053", ROOT + "/tools/dequant_qat_v053.py")
dq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dq)

import torch  # noqa: E402
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32  # noqa: E402
from compressed_tensors.quantization import QuantizationScheme  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

WEIGHTS = {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 32, "num_bits": 4,
           "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
           "symmetric": True, "type": "int", "zp_dtype": None}
QC = {"config_groups": {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None,
                                    "targets": ["Linear"], "weights": WEIGHTS}},
      "format": "pack-quantized", "kv_cache_scheme": None, "quant_method": "compressed-tensors",
      "quantization_status": "compressed", "sparsity_config": {}, "transform_config": {}}
MODULES = (("model.language_model.layers.0.mlp.down_proj", 4, 64), ("model.language_model.layers.0.self_attn.q_proj", 8, 32))
PLAIN = ((dq.EMBED, (16, 8)), ("model.language_model.layers.0.input_layernorm.weight", (8,)),
         ("model.language_model.layers.0.layer_scalar", (1,)), ("model.language_model.zero_dim_probe", ()))


def bits(t):
    return t.contiguous().view(torch.int16)


def make_packed(rows, cols, seed, codes=None):
    g = torch.Generator().manual_seed(seed)
    if codes is None:
        codes = torch.randint(-8, 8, (rows, cols), generator=g, dtype=torch.int8)
    scale = (torch.rand(rows, cols // 32, generator=g) * 0.02 + 1e-4).to(torch.bfloat16)
    expected = (codes.float() * scale.float().repeat_interleave(32, dim=1)).to(torch.bfloat16)
    return {"weight_packed": pack_to_int32(codes, 4), "weight_scale": scale,
            "weight_shape": torch.tensor([rows, cols])}, expected, codes


def scheme():
    return QuantizationScheme.model_validate(QC["config_groups"]["group_0"])


def make_checkpoint(root, weights=None, text_q=None, text_b=None):
    """Synthetic QAT dir (2 packed modules, 4 plain tensors, lm_head) and bf16 reference dir."""
    src, ref = os.path.join(root, "qat"), os.path.join(root, "bf16")
    os.makedirs(src)
    os.makedirs(ref)
    tensors, expected, ref_tensors = {}, {}, {}
    for i, (m, r, c) in enumerate(MODULES):
        packed, exp, _ = make_packed(r, c, i)
        tensors.update({m + "." + k: v for k, v in packed.items()})
        expected[m + ".weight"] = exp
        ref_tensors[m + ".weight"] = (exp.float() * 1.01).to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    for k, shape in PLAIN:
        tensors[k] = expected[k] = ref_tensors[k] = torch.randn(shape, generator=g).to(torch.bfloat16)
    ref_tensors[PLAIN[1][0]] = (tensors[PLAIN[1][0]].float() + 0.5).to(torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(16, 8, generator=g).to(torch.bfloat16)
    save_file(tensors, os.path.join(src, "model.safetensors"), metadata={"format": "pt"})
    save_file(ref_tensors, os.path.join(ref, "model.safetensors"), metadata={"format": "pt"})
    qc = json.loads(json.dumps(QC))
    qc["config_groups"]["group_0"]["weights"].update(weights or {})
    text = {"hidden_size": 8, "tie_word_embeddings": True}
    with open(os.path.join(src, "config.json"), "w") as f:
        json.dump({"text_config": dict(text, **(text_q or {})), "quantization_config": qc, "transformers_version": "5.8.0.dev0"}, f)
    with open(os.path.join(ref, "config.json"), "w") as f:
        json.dump({"text_config": dict(text, **(text_b or {})), "transformers_version": "5.5.0.dev0"}, f)
    for n in dq.TOKENIZER_FILES:
        with open(os.path.join(src, n), "w") as f:
            f.write("synthetic " + n)
    return src, ref, expected


class Unpack(unittest.TestCase):
    def test_independent_equals_compressed_tensors_and_reference(self):
        for seed, (r, c) in enumerate(((4, 64), (3, 96), (16, 32), (5, 256))):
            packed, expected, codes = make_packed(r, c, seed)
            ours, our_codes = dq.independent_dequant(packed["weight_packed"], packed["weight_scale"], [r, c])
            theirs = dq.ct_decompress(packed, scheme())
            self.assertTrue(torch.equal(bits(ours), bits(expected)))
            self.assertTrue(torch.equal(bits(theirs), bits(expected)))
            self.assertTrue(torch.equal(torch.from_numpy(our_codes.copy()), codes))

    def test_extreme_codes_and_sign_bit(self):
        for value in (7, -8, 0, -1):
            codes = torch.full((2, 64), value, dtype=torch.int8)
            packed, expected, _ = make_packed(2, 64, 3, codes=codes)
            ours, _ = dq.independent_dequant(packed["weight_packed"], packed["weight_scale"], [2, 64])
            self.assertTrue(torch.equal(bits(ours), bits(expected)), value)

    def test_round_to_nearest_even_matches_torch(self):
        g = torch.Generator().manual_seed(11)
        codes = torch.arange(-8, 8, dtype=torch.int8).repeat(64, 2)          # 64 rows x 32 columns, every code
        scale = torch.randn(64, 1, generator=g).abs().add(1e-3).to(torch.bfloat16)
        packed = {"weight_packed": pack_to_int32(codes, 4), "weight_scale": scale, "weight_shape": torch.tensor([64, 32])}
        ours, _ = dq.independent_dequant(packed["weight_packed"], scale, [64, 32])
        ref = (codes.float() * scale.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(bits(ours), bits(ref)))
        self.assertTrue(torch.equal(bits(dq.ct_decompress(packed, scheme())), bits(ref)))


class Layout(unittest.TestCase):
    def test_plan_shards_oversize_alone(self):
        self.assertEqual(dq.plan_shards({"a": 5, "b": 10, "c": 3}, 6), [["a"], ["b"], ["c"]])
        self.assertEqual(dq.plan_shards({"a": 2, "b": 2, "c": 2}, 4), [["a", "b"], ["c"]])

    def test_classify_rejects_incomplete_or_zero_point(self):
        base = {"m.weight_packed", "m.weight_scale", "m.weight_shape", "x.weight"}
        self.assertEqual(dq.classify(base), (["m"], ["x.weight"]))
        with self.assertRaises(ValueError):
            dq.classify(base - {"m.weight_scale"})
        with self.assertRaises(ValueError):
            dq.classify(base | {"m.weight_zero_point"})
        with self.assertRaises(ValueError):
            dq.classify(base | {"m.weight"})

    def test_validate_rejects_other_schemes(self):
        for change in ({"group_size": 64}, {"symmetric": False}, {"num_bits": 8}, {"actorder": "group"}):
            qc = json.loads(json.dumps(QC))
            qc["config_groups"]["group_0"]["weights"].update(change)
            with self.assertRaises(ValueError, msg=str(change)):
                dq.validate_quantization({"quantization_config": qc})
        self.assertEqual(dq.validate_quantization({"quantization_config": QC})["weights"]["group_size"], 32)


class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dq053-")
        self.src, self.ref, self.expected = make_checkpoint(self.tmp)
        self.out, self.report = os.path.join(self.tmp, "out"), os.path.join(self.tmp, "build.json")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_build(self, **kw):
        args = dict(src_dir=self.src, ref_dir=self.ref, out_dir=self.out, report_path=self.report, max_bytes=200,
                    expect_packed=2, expect_plain=4, src_model_sha=None, threads=1)
        args.update(kw)
        return dq.build(**args)

    def test_end_to_end(self):
        rep = self.run_build()
        self.assertEqual(rep["status"], "PASS", rep["problems"])
        index = dq.read_json(os.path.join(self.out, "model.safetensors.index.json"))
        shards = sorted(set(index["weight_map"].values()))
        self.assertGreaterEqual(len(shards), 2)
        loaded = {}
        for s in shards:
            loaded.update(load_file(os.path.join(self.out, s)))
        self.assertEqual(set(loaded), set(self.expected))
        self.assertNotIn("lm_head.weight", loaded)
        for k, v in self.expected.items():
            self.assertTrue(torch.equal(bits(loaded[k]), bits(v)), k)
        self.assertEqual(index["metadata"]["total_size"], sum(v.numel() * 2 for v in self.expected.values()))
        cfg = dq.read_json(os.path.join(self.out, "config.json"))
        self.assertNotIn("quantization_config", cfg)
        self.assertEqual(cfg["text_config"], dq.read_json(os.path.join(self.ref, "config.json"))["text_config"])
        for n in dq.TOKENIZER_FILES:
            self.assertEqual(dq.sha256_file(os.path.join(self.out, n)), dq.sha256_file(os.path.join(self.src, n)))
        man = dq.read_json(os.path.join(self.out, "manifest.json"))
        self.assertEqual(rep["manifest_sha256"], dq.sha256_file(os.path.join(self.out, "manifest.json")))
        self.assertEqual(man["counts"]["dequantized_torch_equal_independent"], 2)
        self.assertEqual(man["counts"]["byte_equal_after_write"], 4)
        self.assertEqual(list(man["dropped"]), ["lm_head.weight"])
        self.assertEqual(man["config"]["top_level_keys_differing_from_reference"], ["transformers_version"])
        for row in man["shards"]:
            self.assertEqual(row["sha256"], dq.sha256_file(os.path.join(self.out, row["name"])))
        for k, row in man["tensors"].items():
            self.assertEqual(row["sha256"], dq.tensor_sha(loaded[k]), k)
            self.assertEqual(row["origin"], "dequantized" if k.endswith("_proj.weight") else "byte_copy")
        for n, sha in man["files_sha256"].items():
            self.assertEqual(sha, dq.sha256_file(os.path.join(self.out, n)), n)

    def test_refuses_existing_output(self):
        os.mkdir(self.out)
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.assertEqual(os.listdir(self.out), [])

    def test_refuses_before_writing(self):
        for kw in ({"expect_packed": 3}, {"src_model_sha": "0" * 64}):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.run_build(**kw)
            self.assertFalse(os.path.exists(self.out), kw)
        shutil.rmtree(self.src)
        shutil.rmtree(self.ref)
        self.src, self.ref, _ = make_checkpoint(self.tmp, weights={"group_size": 64})
        with self.assertRaises(ValueError):
            self.run_build()
        self.assertFalse(os.path.exists(self.out))

    def test_text_config_null_key_rename_only(self):
        shutil.rmtree(self.src)
        shutil.rmtree(self.ref)
        self.src, self.ref, _ = make_checkpoint(self.tmp, text_q={"moe_intermediate_size": None},
                                                text_b={"expert_intermediate_size": None})
        self.assertEqual(self.run_build()["status"], "PASS")
        ref_text = dq.read_json(os.path.join(self.ref, "config.json"))["text_config"]
        self.assertEqual(dq.read_json(os.path.join(self.out, "config.json"))["text_config"], ref_text)
        man = dq.read_json(os.path.join(self.out, "manifest.json"))
        self.assertEqual(man["config"]["qat_text_config_differences_all_null"],
                         ["expert_intermediate_size", "moe_intermediate_size"])
        self.assertTrue(man["config"]["text_config_equals_reference"])

    def test_text_config_real_difference_refused(self):
        shutil.rmtree(self.src)
        shutil.rmtree(self.ref)
        self.src, self.ref, _ = make_checkpoint(self.tmp, text_q={"hidden_size": 9})
        with self.assertRaises(ValueError):
            self.run_build()
        self.assertFalse(os.path.exists(self.out))

    def test_compare_informational(self):
        rep = dq.compare(src_dir=self.src, ref_dir=self.ref, report_path=os.path.join(self.tmp, "cmp.json"), threads=1)
        rows = {r["module"]: r for r in rep["modules"]}
        self.assertEqual(len(rows), 2)
        for r in rows.values():
            self.assertAlmostEqual(r["rel_dq"], 0.01, delta=0.004)   # reference = 1.01 x the dequantized weight
            self.assertGreater(r["rel_rtn"], 0)
        plain = {r["key"]: r for r in rep["plain"]}
        self.assertEqual(sum(r["byte_equal"] for r in plain.values()), 3)
        self.assertFalse(plain[PLAIN[1][0]]["byte_equal"])
        self.assertGreater(plain[PLAIN[1][0]]["rel_l2"], 0)
        self.assertFalse(rep["summary"]["lm_head"]["qat_lm_head_byte_equal_qat_embed_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
