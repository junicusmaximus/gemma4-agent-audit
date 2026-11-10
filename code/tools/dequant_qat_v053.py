#!/usr/bin/env python3
"""Dequantized bf16 copy of the Gemma 4 E4B QAT w4a16 checkpoint (boss v053, Stage 0 of
briefing/research/gemma-4-2026/prereg-2026-11-06-qat-base-lora-fixture-and-M0q.md, re-freeze section).

Run every subcommand with PYTHONDONTWRITEBYTECODE=1.
  build        swegemma env. Every pack-quantized Linear (weight_packed / weight_scale / weight_shape) is decompressed
               with compressed-tensors' own PackedQuantizationCompressor.decompress (compressors/pack_quantized/base.py
               :79-119, unpack in helpers.py:126-159, _dequantize in quantization/lifecycle/forward_helpers.py) and the
               result is compared bit for bit (torch.equal on int16 views) with an independent numpy unpack (int32 words
               as little-endian bytes, low nibble first, code = nibble - 8, exact fp32 product with the bf16 scale,
               round to nearest even to bf16 by bit arithmetic). Every other tensor is copied byte for byte, except
               lm_head.weight, which is dropped: the config ties it to embed_tokens, vLLM ties and skips the stored
               tensor (vllm/model_executor/models/gemma4.py:1511-1512, 1672-1673) and transformers keeps two different
               stored tensors untied (transformers/modeling_utils.py:2712-2732). config.json = the QAT config without
               quantization_config, with the bf16 dir's text_config verbatim (the two text_configs may differ only in
               null-valued keys); tokenizer files copied from the QAT dir; shards of at most 4e9 bytes (a single larger
               tensor gets a shard of its own); manifest.json last.
  verify-load  any env with transformers (swegemma, train). from_pretrained of the copy on CPU (bf16, sdpa): no
               missing / unexpected / mismatched keys, lm_head tied to embed_tokens (same data_ptr), every loaded
               tensor equal to the file.
  compare      swegemma env, informational. QAT lm_head vs embed_tokens; plain tensors QAT vs bf16; per packed module
               ||W_dq - W_bf16|| / ||W_bf16|| against the round-to-nearest 4-bit error of W_bf16 under the same scheme
               (compressed-tensors calculate_qparams + fake_quantize, min/max per group of 32)."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time

sys.dont_write_bytecode = True

ROOT = "PROJECT_ROOT/gemma4"
SRC = ROOT + "/models/gemma-4-e4b-it-qat-w4a16-ct"
REF = ROOT + "/models/gemma-4-e4b-it"
OUT = ROOT + "/models/gemma-4-e4b-it-qat-dequant-bf16"
REPORTS = ROOT + "/analysis/qat-transfer-v053"
PACKED = ("weight_packed", "weight_scale", "weight_shape")
UNSUPPORTED = ("weight_zero_point", "weight_g_idx", "weight_global_scale")
DROPPED = ("lm_head.weight",)
EMBED = "model.language_model.embed_tokens.weight"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
MAX_SHARD_BYTES = 4 * 10 ** 9
EXPECT_PACKED, EXPECT_PLAIN = 343, 1733
SCHEME = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32,
          "actorder": None, "dynamic": False, "block_structure": None}
SRC_MODEL_SHA = "19d89d2a4b3e401a3eae75ec08ab112810421a036c91d8c8226b686be7fe0c5d"  # v048 ledger_meta.json
REF_MODEL_SHA = "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"  # v048 ledger_meta.json
LORA_TARGET = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"  # train_lora:28
BLOCK_ROWS = 8192


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path):
    """safetensors header -> ({name: {dtype, shape, start, end}} with absolute file offsets, metadata)."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    meta = header.pop("__metadata__", None)
    base = 8 + size
    return {k: {"dtype": v["dtype"], "shape": list(v["shape"]), "start": base + v["data_offsets"][0],
                "end": base + v["data_offsets"][1]} for k, v in header.items()}, meta


def range_sha(fh, start, end, block=1 << 24):
    fh.seek(start)
    h, left = hashlib.sha256(), end - start
    while left > 0:
        chunk = fh.read(min(block, left))
        if not chunk:
            raise IOError("short read at %d" % fh.tell())
        h.update(chunk)
        left -= len(chunk)
    return h.hexdigest()


def tensor_sha(t):
    """sha256 of the raw bytes of a CPU tensor (no copy)."""
    import torch
    return hashlib.sha256(memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj, indent=1):
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=indent) + "\n")


def validate_quantization(cfg):
    """Exactly one pack-quantized INT4 symmetric group-32 weight-only group over Linear; returns that group."""
    qc = cfg.get("quantization_config")
    if not qc:
        raise ValueError("no quantization_config")
    expect_top = {"quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed"}
    problems = ["%s=%r" % (k, qc.get(k)) for k, v in expect_top.items() if qc.get(k) != v]
    groups = qc.get("config_groups") or {}
    if len(groups) != 1:
        problems.append("config_groups=%d" % len(groups))
    group = next(iter(groups.values()), {})
    if group.get("targets") != ["Linear"] or group.get("input_activations") or group.get("output_activations"):
        problems.append("group targets/activations")
    if group.get("format") not in (None, "pack-quantized"):
        problems.append("group format=%r" % group.get("format"))
    weights = group.get("weights") or {}
    problems += ["weights.%s=%r" % (k, weights.get(k)) for k, v in SCHEME.items() if weights.get(k) != v]
    if qc.get("sparsity_config") or qc.get("transform_config") or qc.get("kv_cache_scheme"):
        problems.append("sparsity/transform/kv_cache present")
    if problems:
        raise ValueError("unsupported quantization_config: " + "; ".join(problems))
    return group


def classify(keys):
    """(packed module prefixes, plain tensor names), both sorted; raises on an incomplete or unsupported packed layout."""
    keys = set(keys)
    packed = sorted(k[:-len(".weight_packed")] for k in keys if k.endswith(".weight_packed"))
    members = {m + "." + s for m in packed for s in PACKED}
    problems = {"missing": sorted(x for x in members if x not in keys),
                "unsupported": sorted(k for k in keys if k.rsplit(".", 1)[-1] in UNSUPPORTED),
                "weight_clash": sorted(m + ".weight" for m in packed if m + ".weight" in keys),
                "orphan": sorted(k for k in keys if k.rsplit(".", 1)[-1] in PACKED and k not in members)}
    if any(problems.values()):
        raise ValueError("packed layout: %s" % {k: v[:3] for k, v in problems.items() if v})
    return packed, sorted(keys - members)


def ct_decompress(tensors, scheme):
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    return PackedQuantizationCompressor.decompress(dict(tensors), scheme)["weight"]


def independent_dequant(packed, scale, shape):
    """Without compressed-tensors: int32 words as little-endian bytes, low nibble first, code = nibble - 8; exact fp32
    product with the bf16 scale (group = cols / scale cols), rounded to nearest even to bf16 by bit arithmetic.
    Returns (bf16 tensor, int8 codes as numpy)."""
    import numpy as np
    import torch
    if sys.byteorder != "little":
        raise RuntimeError("little-endian host required")
    rows, cols = int(shape[0]), int(shape[1])
    words = packed.contiguous().numpy()
    if words.dtype != np.int32 or scale.dtype != torch.bfloat16:
        raise TypeError("expected int32 weight_packed and bf16 weight_scale")
    raw = words.view(np.uint8).reshape(rows, -1)
    codes = np.empty((rows, raw.shape[1] * 2), dtype=np.int8)
    codes[:, 0::2] = (raw & 0x0F).astype(np.int8) - 8
    codes[:, 1::2] = (raw >> 4).astype(np.int8) - 8
    codes = codes[:, :cols]
    s32 = (scale.contiguous().view(torch.int16).numpy().view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    group = cols // s32.shape[1]
    if group * s32.shape[1] != cols or s32.shape[0] != rows:
        raise ValueError("scale shape %s does not tile weight shape %s" % (s32.shape, (rows, cols)))
    product = codes.astype(np.float32) * np.repeat(s32, group, axis=1)
    if not np.isfinite(product).all():
        raise ValueError("non-finite dequantized value")
    bits = product.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
    return torch.from_numpy(rounded.view(np.int16)).view(torch.bfloat16), codes


def dequant_module(src, module, scheme):
    """compressed-tensors decompress of one packed module, checked bit for bit against the independent unpack."""
    import torch
    t = {s: src.get_tensor(module + "." + s) for s in PACKED}
    shape = [int(x) for x in t["weight_shape"].tolist()]
    weight = ct_decompress(t, scheme).contiguous()
    other, codes = independent_dequant(t["weight_packed"], t["weight_scale"], shape)
    if weight.dtype != torch.bfloat16 or list(weight.shape) != shape:
        raise ValueError("decompress dtype/shape %s %s for %s" % (weight.dtype, list(weight.shape), module))
    if not torch.equal(weight.view(torch.int16), other.view(torch.int16)):
        raise ValueError("independent unpack disagrees with compressed-tensors: " + module)
    scale = t["weight_scale"].float()
    if float(scale.min()) <= 0:
        raise ValueError("non-positive scale: " + module)
    return weight, {"shape": shape, "torch_equal_independent": True, "codes_min": int(codes.min()),
                    "codes_max": int(codes.max()), "scale_min": float(scale.min()), "scale_max": float(scale.max()),
                    "sha256": tensor_sha(weight)}


def plan_shards(sizes, max_bytes):
    """Sorted keys packed into consecutive shards of at most max_bytes; a larger single tensor gets its own shard."""
    shards, current, used = [], [], 0
    for key in sorted(sizes):
        if current and used + sizes[key] > max_bytes:
            shards.append(current)
            current, used = [], 0
        current.append(key)
        used += sizes[key]
    if current:
        shards.append(current)
    return shards


def null_key_delta(a, b):
    """Keys that differ between two text_configs; raises unless each of them is null or absent on both sides (the QAT
    config, written by transformers 5.8, names the null MoE width moe_intermediate_size; the bf16 config, 5.5, names
    it expert_intermediate_size)."""
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) or (k in a) != (k in b))
    real = [k for k in diff if a.get(k) is not None or b.get(k) is not None]
    if real:
        raise ValueError("text_config of the QAT dir differs from the bf16 dir in non-null keys: %s" % real)
    return diff


def prepare(src_dir, ref_dir, src_model_sha, expect_packed, expect_plain):
    """Configs, scheme, source header, counts and output sizes; nothing is written. config.json = the QAT config without
    quantization_config and with the bf16 dir's text_config verbatim (they may differ only in null-valued keys)."""
    from compressed_tensors.quantization import QuantizationScheme
    from safetensors import safe_open
    src_cfg = read_json(os.path.join(src_dir, "config.json"))
    ref_cfg = read_json(os.path.join(ref_dir, "config.json"))
    group = validate_quantization(src_cfg)
    delta = null_key_delta(src_cfg.get("text_config") or {}, ref_cfg.get("text_config") or {})
    out_cfg = {k: v for k, v in src_cfg.items() if k != "quantization_config"}
    out_cfg["text_config"] = ref_cfg["text_config"]
    src_file = os.path.join(src_dir, "model.safetensors")
    src_sha = sha256_file(src_file)
    if src_model_sha and src_sha != src_model_sha:
        raise ValueError("QAT model.safetensors sha %s != pinned %s" % (src_sha, src_model_sha))
    header, _ = read_header(src_file)
    packed, plain = classify(header)
    kept = [k for k in plain if k not in DROPPED]
    if len(packed) != expect_packed or len(kept) != expect_plain or len(plain) - len(kept) != len(DROPPED):
        raise ValueError("counts: packed %d, kept plain %d, dropped %d" % (len(packed), len(kept), len(plain) - len(kept)))
    with safe_open(src_file, "pt") as src:
        shapes = {m + ".weight": [int(x) for x in src.get_tensor(m + ".weight_shape").tolist()] for m in packed}
    sizes = {k: header[k]["end"] - header[k]["start"] for k in kept}
    sizes.update({k: 2 * math.prod(v) for k, v in shapes.items()})
    return {"src_cfg": src_cfg, "ref_cfg": ref_cfg, "out_cfg": out_cfg, "group": group, "text_config_null_key_delta": delta,
            "scheme": QuantizationScheme.model_validate(group), "src_file": src_file, "src_sha": src_sha,
            "header": header, "packed": packed, "kept": kept, "shapes": shapes, "sizes": sizes}


def write_shards(plan, out_dir, max_bytes):
    """Write the shards; dequantized tensors are checked against the independent unpack before they are written."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    shards = plan_shards(plan["sizes"], max_bytes)
    names = ["model-%05d-of-%05d.safetensors" % (i + 1, len(shards)) for i in range(len(shards))]
    packed, weight_map, dq = set(plan["packed"]), {}, {}
    with safe_open(plan["src_file"], "pt") as src:
        for name, keys in zip(names, shards):
            tensors = {}
            for key in keys:
                module = key[:-len(".weight")] if key.endswith(".weight") else None
                if module in packed:
                    tensors[key], dq[key] = dequant_module(src, module, plan["scheme"])
                else:
                    tensors[key] = src.get_tensor(key)
                weight_map[key] = name
            save_file(tensors, os.path.join(out_dir, name), metadata={"format": "pt"})
            del tensors
    return names, weight_map, dq


def verify_written(plan, out_dir, names, weight_map, dq):
    """Re-read every written tensor from disk: byte copies equal the source byte range, dequantized tensors equal the
    checked compressed-tensors result; dtype, shape and key set as planned."""
    header, tensors, problems, counts = plan["header"], {}, [], {"byte_equal": 0, "dequantized_equal": 0}
    with open(plan["src_file"], "rb") as sf:
        for name in names:
            out_header, _ = read_header(os.path.join(out_dir, name))
            with open(os.path.join(out_dir, name), "rb") as of:
                for key, h in sorted(out_header.items()):
                    digest = range_sha(of, h["start"], h["end"])
                    if key in dq:
                        ok = (digest, h["dtype"], h["shape"]) == (dq[key]["sha256"], "BF16", dq[key]["shape"])
                        source = {s: range_sha(sf, header[key[:-6] + s]["start"], header[key[:-6] + s]["end"])
                                  for s in PACKED}
                        counts["dequantized_equal"] += ok
                    else:
                        s = header[key]
                        ok = (digest, h["dtype"], h["shape"]) == (range_sha(sf, s["start"], s["end"]), s["dtype"], s["shape"])
                        source = key
                        counts["byte_equal"] += ok
                    if not ok or weight_map.get(key) != name:
                        problems.append("mismatch or unplanned: " + key)
                    tensors[key] = {"sha256": digest, "dtype": h["dtype"], "shape": h["shape"], "shard": name,
                                    "origin": "dequantized" if key in dq else "byte_copy", "source": source}
    problems += ["missing: " + k for k in sorted(set(weight_map) - set(tensors))]
    return tensors, counts, problems


def copy_side_files(src_dir, out_dir, out_cfg):
    """config.json (the QAT config without quantization_config) and byte copies of the tokenizer files."""
    with open(os.path.join(out_dir, "config.json"), "x") as f:
        f.write(json.dumps(out_cfg, indent=2) + "\n")
    for name in TOKENIZER_FILES:
        shutil.copyfile(os.path.join(src_dir, name), os.path.join(out_dir, name))
        if sha256_file(os.path.join(src_dir, name)) != sha256_file(os.path.join(out_dir, name)):
            raise IOError("copy mismatch: " + name)


def env_versions():
    import importlib
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("torch", "numpy", "safetensors", "compressed_tensors", "transformers"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def make_manifest(plan, src_dir, ref_dir, out_dir, names, tensors, counts, max_bytes, started):
    header = plan["header"]
    with open(plan["src_file"], "rb") as sf:
        dropped = {k: {"sha256": range_sha(sf, header[k]["start"], header[k]["end"]), "dtype": header[k]["dtype"],
                       "shape": header[k]["shape"], "reason": "config ties lm_head to embed_tokens; vLLM ties it and "
                       "skips the stored tensor; transformers would keep a stored, different tensor untied"} for k in DROPPED}
    shards = []
    for name in names:
        keys = [k for k, v in tensors.items() if v["shard"] == name]
        size = sum(plan["sizes"][k] for k in keys)
        shards.append({"name": name, "bytes": os.path.getsize(os.path.join(out_dir, name)),
                       "sha256": sha256_file(os.path.join(out_dir, name)), "tensors": len(keys), "tensor_bytes": size,
                       "single_tensor_over_limit": keys if size > max_bytes else []})
    src_files = {n: sha256_file(os.path.join(src_dir, n)) for n in ("config.json",) + TOKENIZER_FILES}
    src_files["model.safetensors"] = plan["src_sha"]
    out_cfg, ref_cfg = plan["out_cfg"], plan["ref_cfg"]
    tool = os.path.abspath(__file__)
    return {"created_utc_start": started, "created_utc_end": utc(), "tool": {"path": tool, "sha256": sha256_file(tool)},
            "env": env_versions(), "source_dir": src_dir, "source_files_sha256": src_files, "reference_bf16_dir": ref_dir,
            "reference_config_json_sha256": sha256_file(os.path.join(ref_dir, "config.json")),
            "removed_quantization_config": plan["src_cfg"]["quantization_config"],
            "config": {"origin": "QAT config.json without quantization_config; text_config = the bf16 dir's text_config "
                                 "verbatim", "text_config_equals_reference": out_cfg.get("text_config") == ref_cfg.get("text_config"),
                       "qat_text_config_differences_all_null": plan["text_config_null_key_delta"],
                       "top_level_keys_differing_from_reference":
                           sorted(k for k in set(out_cfg) | set(ref_cfg) if out_cfg.get(k) != ref_cfg.get(k))},
            "method": {"dequantize": "compressed_tensors PackedQuantizationCompressor.decompress", "check":
                       "bitwise torch.equal (int16 views) against an independent numpy unpack", "plain": "byte copy",
                       "dropped": list(DROPPED)},
            "counts": counts, "dropped": dropped, "max_shard_bytes": max_bytes, "shards": shards,
            "index_total_size": sum(plan["sizes"].values()),
            "files_sha256": {n: sha256_file(os.path.join(out_dir, n))
                             for n in ("model.safetensors.index.json", "config.json") + TOKENIZER_FILES},
            "tensors": tensors}


def build(src_dir=SRC, ref_dir=REF, out_dir=OUT, report_path=REPORTS + "/dequant_build.json", max_bytes=MAX_SHARD_BYTES,
          expect_packed=EXPECT_PACKED, expect_plain=EXPECT_PLAIN, src_model_sha=SRC_MODEL_SHA, threads=4):
    """Fresh output dir only; the manifest is written last and only when every check passed."""
    import torch
    torch.set_num_threads(threads)
    t0, started = time.time(), utc()
    if os.path.lexists(out_dir) or os.path.lexists(report_path):
        raise FileExistsError("fresh output required: %s / %s" % (out_dir, report_path))
    plan = prepare(src_dir, ref_dir, src_model_sha, expect_packed, expect_plain)
    os.mkdir(out_dir)
    names, weight_map, dq = write_shards(plan, out_dir, max_bytes)
    tensors, written, problems = verify_written(plan, out_dir, names, weight_map, dq)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "x") as f:
        f.write(json.dumps({"metadata": {"total_size": sum(plan["sizes"].values())}, "weight_map": weight_map},
                           indent=2) + "\n")
    copy_side_files(src_dir, out_dir, plan["out_cfg"])
    counts = {"source_tensors": len(plan["header"]), "packed_modules": len(plan["packed"]),
              "dequantized_torch_equal_independent": sum(r["torch_equal_independent"] for r in dq.values()),
              "dequantized_written_equal": written["dequantized_equal"], "byte_copied": len(plan["kept"]),
              "byte_equal_after_write": written["byte_equal"], "dropped": len(DROPPED), "tensors_out": len(tensors)}
    expected = {"packed_modules": expect_packed, "dequantized_torch_equal_independent": expect_packed,
                "dequantized_written_equal": expect_packed, "byte_copied": expect_plain,
                "byte_equal_after_write": expect_plain, "tensors_out": expect_packed + expect_plain}
    problems += ["count %s=%d expected %d" % (k, counts[k], v) for k, v in expected.items() if counts[k] != v]
    report = {"stage": "S0 build", "status": "FAIL" if problems else "PASS", "started_utc": started, "out_dir": out_dir,
              "counts": counts, "problems": problems, "shards": len(names)}
    if not problems:
        manifest = make_manifest(plan, src_dir, ref_dir, out_dir, names, tensors, counts, max_bytes, started)
        write_json(os.path.join(out_dir, "manifest.json"), manifest)
        report.update(manifest_sha256=sha256_file(os.path.join(out_dir, "manifest.json")), shard_rows=manifest["shards"],
                      source_files_sha256=manifest["source_files_sha256"], env=manifest["env"],
                      config_keys_differing_from_bf16=manifest["config"]["top_level_keys_differing_from_reference"])
    report.update(ended_utc=utc(), seconds=round(time.time() - t0, 1),
                  disk_bytes=sum(os.path.getsize(os.path.join(out_dir, n)) for n in os.listdir(out_dir)))
    write_json(report_path, report)
    return report


def verify_load(model_dir=OUT, report_path=None):
    """from_pretrained on CPU in the current env; key lists, tie and file equality of every loaded tensor."""
    import torch
    from safetensors import safe_open
    from transformers import Gemma4ForConditionalGeneration
    if os.path.lexists(report_path):
        raise FileExistsError(report_path)
    t0, started = time.time(), utc()
    model, info = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    load_s = round(time.time() - t0, 1)
    lists = {k: sorted(str(x) for x in v) for k, v in info.items()}
    try:
        attr_tie = model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    except AttributeError as exc:
        attr_tie = "attribute path missing: %s" % exc
    api_tie = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    state = model.state_dict()
    weight_map = read_json(os.path.join(model_dir, "model.safetensors.index.json"))["weight_map"]
    equal, unequal, absent = 0, [], []
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(model_dir, shard), "pt") as f:
            for key in f.keys():
                if key not in state:
                    absent.append(key)
                    continue
                t, p = f.get_tensor(key), state[key]
                same = p.dtype == t.dtype and tuple(p.shape) == tuple(t.shape) and torch.equal(p, t)
                equal += same
                if not same:
                    unequal.append(key)
    failing = [k for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if lists.get(k)]
    ok = not failing and attr_tie is True and api_tie and not unequal and not absent
    report = {"stage": "S0 verify-load", "status": "PASS" if ok else "FAIL", "started_utc": started, "ended_utc": utc(),
              "load_seconds": load_s, "seconds": round(time.time() - t0, 1), "model_dir": model_dir,
              "env": env_versions(), "loading_info": lists, "failing_lists": failing,
              "lm_head_data_ptr_equals_embed_tokens": attr_tie, "output_embeddings_tied_to_input": api_tie,
              "file_tensors": len(weight_map), "loaded_equal_to_file": equal, "unequal": unequal[:20],
              "absent_from_model": absent[:20], "state_dict_keys_not_in_file": sorted(set(state) - set(weight_map)),
              "model_class": type(model).__name__, "params": sum(p.numel() for p in model.parameters()),
              "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
    write_json(report_path, report)
    return report


def blockwise_diff(fa, ka, fb, kb):
    """||a - b|| / ||b||, cosine and max |a - b| (float32 blocks, float64 sums) of two safetensors entries."""
    import torch
    sa, sb = fa.get_slice(ka), fb.get_slice(kb)
    shape = list(sa.get_shape())
    if shape != list(sb.get_shape()):
        return {"comparable": False, "shape_a": shape, "shape_b": list(sb.get_shape())}
    if len(shape) < 2:
        blocks = [(fa.get_tensor(ka), fb.get_tensor(kb))]
    else:
        blocks = ((sa[i:i + BLOCK_ROWS], sb[i:i + BLOCK_ROWS]) for i in range(0, shape[0], BLOCK_ROWS))
    dd = aa = bb = ab = mx = 0.0
    for a, b in blocks:
        a, b = a.float(), b.float()
        d = a - b
        dd += float(d.square().sum(dtype=torch.float64))
        aa += float(a.square().sum(dtype=torch.float64))
        bb += float(b.square().sum(dtype=torch.float64))
        ab += float((a * b).sum(dtype=torch.float64))
        mx = max(mx, float(d.abs().max())) if d.numel() else mx
    return {"comparable": True, "rel_l2": math.sqrt(dd / bb) if bb else None,
            "cosine": ab / math.sqrt(aa * bb) if aa and bb else None, "max_abs": mx}


def module_errors(src, ref, module, scheme):
    """Per packed module: QAT dequantized vs bf16, and the round-to-nearest error of the bf16 weight itself."""
    import torch
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.utils.helpers import calculate_qparams
    t = {s: src.get_tensor(module + "." + s) for s in PACKED}
    shape = [int(x) for x in t["weight_shape"].tolist()]
    w_dq, codes = independent_dequant(t["weight_packed"], t["weight_scale"], shape)
    w_bf = ref.get_tensor(module + ".weight")
    if list(w_bf.shape) != shape:
        raise ValueError("shape mismatch for " + module)
    gs, args = scheme.weights.group_size, scheme.weights
    grouped = w_bf.unflatten(-1, (-1, gs))
    scale_rtn, zero_rtn = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    w_rtn = fake_quantize(w_bf, scale_rtn, zero_rtn, args)
    wb = w_bf.float()
    norm = wb.norm()
    s_qat = t["weight_scale"].float().repeat_interleave(gs, dim=1)
    q_own = torch.clamp(torch.round(wb / s_qat), -8, 7)
    rel = lambda w: float((w.float() - wb).norm() / norm)  # noqa: E731
    row = {"module": module, "kind": re.sub(r"\.\d+\.", ".N.", module).replace("model.language_model.", ""),
           "lora_target": bool(re.fullmatch(LORA_TARGET, module)), "numel": wb.numel(), "rel_dq": rel(w_dq),
           "rel_rtn": rel(w_rtn), "rel_rtn_with_qat_scales": rel((q_own * s_qat).to(torch.bfloat16)),
           "code_agreement_with_qat_scales": float((q_own == torch.from_numpy(codes).float()).float().mean()),
           "scale_ratio_qat_over_rtn_median": float((t["weight_scale"].float() / scale_rtn.float()).median()),
           "cosine_dq_bf16": float((w_dq.float() * wb).sum() / (w_dq.float().norm() * norm))}
    row["ratio_dq_over_rtn"] = row["rel_dq"] / row["rel_rtn"]
    return row


def quantiles(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "min": float(v.min()), "p05": float(np.quantile(v, .05)), "median": float(np.median(v)),
            "p95": float(np.quantile(v, .95)), "max": float(v.max())} if v.size else {"n": 0}


def plain_category(key):
    for needle, cat in (("vision", "vision"), ("audio", "audio"), ("embed_tokens_per_layer", "per_layer_embedding_table"),
                        ("embed_tokens", "embed_tokens"), ("norm", "norms"), ("layer_scalar", "layer_scalar")):
        if needle in key:
            return cat
    return "other_text"


def compare_plain(src_file, ref_file, src_header, ref_header, keys):
    """Byte equality from the raw ranges; difference statistics only for tensors that are not byte-equal."""
    from safetensors import safe_open
    rows = []
    with open(src_file, "rb") as sa, open(ref_file, "rb") as rb, safe_open(src_file, "pt") as fa, \
            safe_open(ref_file, "pt") as fb:
        for k in keys:
            a, b = src_header[k], ref_header[k]
            equal = range_sha(sa, a["start"], a["end"]) == range_sha(rb, b["start"], b["end"]) and a["shape"] == b["shape"]
            row = {"key": k, "category": plain_category(k), "byte_equal": equal}
            if not equal:
                row.update(blockwise_diff(fa, k, fb, k))
            rows.append(row)
    return rows


def summarize(module_rows, plain_rows, head):
    fields = ("ratio_dq_over_rtn", "rel_dq", "rel_rtn", "rel_rtn_with_qat_scales", "code_agreement_with_qat_scales",
              "scale_ratio_qat_over_rtn_median", "cosine_dq_bf16")
    groups = {"all_packed": module_rows, "lora_targets": [r for r in module_rows if r["lora_target"]],
              "per_layer_projections": [r for r in module_rows if not r["lora_target"]]}
    mods = {g: {"n": len(rows), **{f: quantiles([r[f] for r in rows]) for f in fields}} for g, rows in groups.items()}
    mods["by_kind_median_ratio"] = {k: quantiles([r["ratio_dq_over_rtn"] for r in module_rows if r["kind"] == k])["median"]
                                    for k in sorted({r["kind"] for r in module_rows})}
    cats = {}
    for c in sorted({r["category"] for r in plain_rows}):
        rows = [r for r in plain_rows if r["category"] == c]
        diff = [r["rel_l2"] for r in rows if not r["byte_equal"] and r.get("rel_l2") is not None]
        cats[c] = {"n": len(rows), "byte_equal": sum(r["byte_equal"] for r in rows), "rel_l2_unequal": quantiles(diff)}
    return {"packed_modules": mods, "plain_tensors": cats, "lm_head": head}


def compare(src_dir=SRC, ref_dir=REF, report_path=REPORTS + "/dequant_compare.json", threads=4):
    """Informational comparisons (nothing here gates the build)."""
    import torch
    from compressed_tensors.quantization import QuantizationScheme
    from safetensors import safe_open
    torch.set_num_threads(threads)
    if os.path.lexists(report_path):
        raise FileExistsError(report_path)
    t0, started = time.time(), utc()
    src_file, ref_file = os.path.join(src_dir, "model.safetensors"), os.path.join(ref_dir, "model.safetensors")
    scheme = QuantizationScheme.model_validate(validate_quantization(read_json(os.path.join(src_dir, "config.json"))))
    src_header, ref_header = read_header(src_file)[0], read_header(ref_file)[0]
    packed, plain = classify(src_header)
    with safe_open(src_file, "pt") as src, safe_open(ref_file, "pt") as ref:
        module_rows = [module_errors(src, ref, m, scheme) for m in packed]
        head = {"qat_lm_head_vs_qat_embed_tokens": blockwise_diff(src, DROPPED[0], src, EMBED),
                "qat_lm_head_vs_bf16_embed_tokens": blockwise_diff(src, DROPPED[0], ref, EMBED)}
    with open(src_file, "rb") as f:
        a, b = src_header[DROPPED[0]], src_header[EMBED]
        head["qat_lm_head_byte_equal_qat_embed_tokens"] = range_sha(f, a["start"], a["end"]) == range_sha(f, b["start"], b["end"])
    plain_rows = compare_plain(src_file, ref_file, src_header, ref_header, [k for k in plain if k not in DROPPED])
    report = {"stage": "S0 compare (informational)", "started_utc": started, "ended_utc": utc(),
              "seconds": round(time.time() - t0, 1), "source_dir": src_dir, "reference_dir": ref_dir,
              "rtn_baseline": "compressed-tensors calculate_qparams (symmetric, absmax / 7.5 per group of 32, bf16 "
                              "scales) + fake_quantize of W_bf16; rel_x = ||W_x - W_bf16|| / ||W_bf16||",
              "summary": summarize(module_rows, plain_rows, head), "modules": module_rows, "plain": plain_rows,
              "env": env_versions()}
    write_json(report_path, report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--report", default=REPORTS + "/dequant_build.json")
    v = sub.add_parser("verify-load")
    v.add_argument("--report", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--report", default=REPORTS + "/dequant_compare.json")
    for p in (b, c):
        p.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        rep = build(report_path=args.report, threads=args.threads)
        print(json.dumps({k: rep.get(k) for k in ("status", "counts", "problems", "manifest_sha256", "seconds", "disk_bytes")}))
    elif args.cmd == "verify-load":
        rep = verify_load(report_path=args.report)
        print(json.dumps({k: rep.get(k) for k in ("status", "failing_lists", "lm_head_data_ptr_equals_embed_tokens",
                                                   "loaded_equal_to_file", "file_tensors", "state_dict_keys_not_in_file",
                                                   "seconds")}))
    else:
        rep = compare(report_path=args.report, threads=args.threads)
        print(json.dumps({"seconds": rep["seconds"], "summary": rep["summary"]["packed_modules"]["all_packed"]["ratio_dq_over_rtn"]}))
    return 0 if rep.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
