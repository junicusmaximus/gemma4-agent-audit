#!/usr/bin/env python3
"""HF teacher-forced scoring for the QAT-transfer check v053: a copy of analysis/first-step-probe-boss-v048/s1_hf_score.py
with the base swapped. Same ledger (v048 ledger.jsonl, 48 H-gen + 5 T-fit items, same target ids), same LoRA structure
(train_lora TARGET_REGEX, r16/alpha16, PEFT autocast), same loss path (hidden -> lm_head -> tanh softcap -> float
log_softmax), same layouts (one item per forward without attention mask; batches of 8 with right padding + mask).
Changes: BASE = the dequantized QAT copy (or the bf16 checkpoint for the F_bf control); the cells per --stage; the v048
abort on "adapter meta base != loaded base" became a declared cross-base cell (abort only if the observed cross-base
set differs from the declaration); outputs carry the stage name. Run with the train env (PEFT); --cpu-dry-run with CUDA_VISIBLE_DEVICES="" checks the code path.
  --stage=s1      base, zerob, m0 on the dequantized base (m0 and zerob are declared cross-base cells)
  --stage=s2dq    base, fdq   on the dequantized base     (fixture F_dq, sft/runs/F_dq_v053)
  --stage=s2dq48  base, fdq   on the dequantized base     (48-epoch rerun, sft/runs/F_dq_v053_e48)
  --stage=s2bf    base, fbf   on the bf16 base            (control F_bf, sft/runs/F_bf_v053)"""
import collections
import contextlib
import datetime
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/qat-transfer-v053"
V048 = ROOT + "/analysis/first-step-probe-boss-v048"
LEDGER, LEDGER_SHA = V048 + "/ledger.jsonl", "bca7ae2274a38b7d57ea17d5785031dc6c9d061a2cdfef8a7b1184ce696d6273"
BASES = {"dequant": ROOT + "/models/gemma-4-e4b-it-qat-dequant-bf16", "bf16": ROOT + "/models/gemma-4-e4b-it"}
M0, ZEROB = ROOT + "/sft/runs/A1_M0_v035", V048 + "/adapters/zerob"
STAGES = {  # stage: (base, {cell: (adapter dir, meta json with the training base)}, declared cross-base cells)
    "s1": ("dequant", {"zerob": (ZEROB, M0 + ".meta.json"), "m0": (M0, M0 + ".meta.json")}, {"zerob", "m0"}),
    "s2dq": ("dequant", {"fdq": (ROOT + "/sft/runs/F_dq_v053", ROOT + "/sft/runs/F_dq_v053/meta.json")}, set()),
    "s2dq48": ("dequant", {"fdq": (ROOT + "/sft/runs/F_dq_v053_e48", ROOT + "/sft/runs/F_dq_v053_e48/meta.json")}, set()),
    "s2bf": ("bf16", {"fbf": (ROOT + "/sft/runs/F_bf_v053", ROOT + "/sft/runs/F_bf_v053/meta.json")}, set()),
}
BATCH, STAGE_BUDGET_S, GPU_TOTAL_S = 8, 15 * 60, 60 * 60
STAGE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--stage=")), "s1")
DRY = "--cpu-dry-run" in sys.argv  # code-path check on CPU with 3 truncated items; outputs go to dryrun/, no lease needed
OUT_S1 = OUT + "/dryrun" if DRY else OUT
sys.path.insert(0, OUT)
import probe_lib as lib  # noqa: E402  (byte copy of the v048 probe_lib.py)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def gate():
    """Leases held by the live holder process, GPU idle, deadline file written (first GPU stage)."""
    rec = read_json(OUT + "/lock-receipt.json")
    if rec.get("status") != "held" or not os.path.exists("/proc/%d" % rec["pid"]):
        sys.exit("ABORT: leases are not held by a live holder: %s" % rec.get("status"))
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if used > 100 or apps:
        sys.exit("ABORT: GPU not idle (%d MiB, apps %r)" % (used, apps))
    if not os.path.exists(OUT + "/gpu_deadline.json"):
        with open(OUT + "/gpu_deadline.json", "w") as f:
            json.dump({"gpu_start_utc": utc(), "gpu_start_epoch": time.time(), "deadline_epoch": time.time() + GPU_TOTAL_S}, f)
    return rec


def module_info(pm, name, file_state, lora_layer_cls):
    """PEFT modules that carry adapter `name`, their layers/projections, dtype, and a tensor-by-tensor file comparison."""
    import torch
    from peft import get_peft_model_state_dict
    mods = [n for n, m in pm.named_modules() if isinstance(m, lora_layer_cls) and name in m.lora_A]
    layers = sorted({int(x) for n in mods for x in re.findall(r"\.layers\.(\d+)\.", n)})
    loaded = get_peft_model_state_dict(pm, adapter_name=name)
    equal = sum(1 for k, v in file_state.items() if k in loaded and torch.equal(loaded[k].detach().to("cpu", torch.bfloat16), v))
    dtypes = collections.Counter(str(p.dtype) for n, p in pm.named_parameters() if "." + name + "." in n and "lora_" in n)
    return {"modules": len(mods), "module_names": mods, "layers": layers, "n_layers": len(layers),
            "projections": dict(collections.Counter(n.rsplit(".", 1)[-1] for n in mods)),
            "tensors_in_file": len(file_state), "tensors_loaded": len(loaded), "tensors_equal_to_file": equal,
            "key_sets_equal": set(loaded) == set(file_state), "param_dtypes": dict(dtypes)}


def score(inner, softcap, items, batched, dev):
    import torch
    import torch.nn.functional as F
    width = max(len(it["ids"]) for it in items)
    ids = torch.zeros((len(items), width), dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for b, it in enumerate(items):
        ids[b, :len(it["ids"])] = torch.tensor(it["ids"])
        mask[b, :len(it["ids"])] = 1
    kwargs = {"input_ids": ids.to(dev), "use_cache": False}
    if batched:
        kwargs["attention_mask"] = mask.to(dev)
    with torch.inference_mode():
        hidden = inner.model(**kwargs).last_hidden_state
        out = []
        for b, it in enumerate(items):
            s, e = it["target_span"]
            logits = F.linear(hidden[b, s - 1:e - 1], inner.lm_head.weight)
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            lp = logits.float().log_softmax(-1)
            tgt = torch.tensor(it["ids"][s:e], device=lp.device)
            tlp = lp.gather(1, tgt[:, None])[:, 0]
            rank = (lp >= tlp[:, None]).sum(1)
            top = lp.topk(5, dim=-1)
            out.append({"logprob": tlp.tolist(), "rank": rank.tolist(), "top5_ids": top.indices.tolist(),
                        "top5_logprob": top.values.tolist()})
    return out


def truncate(it, keep=40):
    """Dry run only: the last `keep` prompt tokens + the target, spans shifted."""
    cut = max(0, it["target_span"][0] - keep)
    new = dict(it, ids=it["ids"][cut:])
    for k in ("target_span", "scaffold_span", "text_span", "control_span"):
        new[k] = [it[k][0] - cut, it[k][1] - cut] if it[k] else None
    return new


def base_identity(kind):
    """dequant: every shard and side file re-hashed against manifest.json, plus the manifest sha; bf16: model sha."""
    base = BASES[kind]
    if kind == "bf16":
        return {"dir": base, "model_sha256": sha256_file(base + "/model.safetensors")}
    manifest = read_json(base + "/manifest.json")
    files = {r["name"]: r["sha256"] for r in manifest["shards"]}
    files.update(manifest["files_sha256"])
    changed = [n for n, sha in sorted(files.items()) if sha256_file(os.path.join(base, n)) != sha]
    if changed:
        sys.exit("ABORT: dequantized base differs from its manifest: %s" % changed)
    return {"dir": base, "manifest_sha256": sha256_file(base + "/manifest.json"), "files_verified": len(files)}


def main():
    if STAGE not in STAGES:
        sys.exit("ABORT: unknown stage %r" % STAGE)
    base_kind, adapters, declared_cross = STAGES[STAGE]
    BASE = BASES[base_kind]
    if DRY:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            sys.exit("ABORT: the dry run must run with CUDA_VISIBLE_DEVICES set to an empty string")
        os.makedirs(OUT_S1, exist_ok=True)
        rec, deadline = {"pid": None}, time.time() + 3600
    else:
        rec = gate()
        deadline = read_json(OUT + "/gpu_deadline.json")["deadline_epoch"]
    stage_start, t0 = utc(), time.time()
    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    sys.path.insert(0, ROOT + "/sft")
    import train_lora  # trainer constants only (TARGET_REGEX); nothing is trained
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
           "python": sys.version.split()[0], "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "lock_holder_pid": rec["pid"]}
    if sha256_file(LEDGER) != LEDGER_SHA:
        sys.exit("ABORT: v048 ledger changed")
    identity = base_identity(base_kind)
    table = read_json(ROOT + "/analysis/serialization-audit-boss-v046/report.json")["environment"]["special_tokens"]["training_base"]
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    special = {s: tok.convert_tokens_to_ids(s) for s in table}
    acfg = {k: read_json(v[0] + "/adapter_config.json") for k, v in adapters.items()}
    meta_base = {k: read_json(v[1])["args"]["base"] for k, v in adapters.items()}
    cross = {k for k, b in meta_base.items() if os.path.normpath(b) != os.path.normpath(BASE)}
    checks = {"stage": STAGE, "special_ids": special, "special_ids_equal_audit_table": special == table, "base_loaded": BASE,
              "base_identity": identity, "meta_json_base": meta_base, "cross_base_cells": sorted(cross),
              "cross_base_cells_declared": sorted(declared_cross), "cross_base_as_declared": cross == declared_cross,
              "adapter_config_base_model_name_or_path": {k: v["base_model_name_or_path"] for k, v in acfg.items()},
              "adapter_config_target_modules": {k: v["target_modules"] for k, v in acfg.items()},
              "adapter_sha256": {k: sha256_file(v[0] + "/adapter_model.safetensors") for k, v in adapters.items()},
              "trainer_target_regex": train_lora.TARGET_REGEX, "ledger_sha256": LEDGER_SHA}
    if not checks["special_ids_equal_audit_table"] or not checks["cross_base_as_declared"]:
        sys.exit("ABORT: tokenizer/base check failed: %s" % json.dumps(checks))
    items = [json.loads(line) for line in open(LEDGER)]
    dev = "cpu" if DRY else "cuda"
    if DRY:
        items = [truncate(x) for x in items[:2] + [i for i in items if i["kind"] == "T-fit"][:1]]
    model = Gemma4ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                           device_map=None if DRY else {"": 0})
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    checks["lm_head_tied_to_embed_tokens"] = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    names = [n for n, _ in model.named_modules()]
    checks["trainer_regex_matches"] = sum(1 for n in names if re.fullmatch(train_lora.TARGET_REGEX, n))
    cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=train_lora.TARGET_REGEX)
    first = next(iter(adapters))
    pm = get_peft_model(model, cfg, adapter_name=first)
    modules = {}
    for name, (path, _) in adapters.items():
        if name != first:
            pm.add_adapter(name, cfg)
        state = load_file(path + "/adapter_model.safetensors")
        res = set_peft_model_state_dict(pm, state, adapter_name=name)
        modules[name] = module_info(pm, name, state, LoraLayer)
        modules[name]["unexpected_keys"] = list(getattr(res, "unexpected_keys", []))
        modules[name]["missing_lora_keys_for_adapter"] = [k for k in getattr(res, "missing_keys", []) if "lora_" in k and "." + name + "." in k]
    pm.eval()
    inner = pm.base_model.model
    softcap = inner.config.get_text_config().final_logit_softcapping
    load_s = round(time.time() - t0, 1)
    passes, skipped, last = [], [], 0.0
    cells = ["base"] + list(adapters)
    out = open(OUT_S1 + "/hf_scores_%s.jsonl" % STAGE, "x")
    for layout in ("single", "batch8"):
        for cell in cells:
            now = time.time()
            if now - t0 + last > STAGE_BUDGET_S or now + last > deadline:
                skipped.append({"cell": cell, "layout": layout, "reason": "time budget"})
                continue
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            groups = [[it] for it in items] if layout == "single" else [
                grp[i:i + BATCH] for k in ("H-gen", "T-fit") for grp in [[it for it in items if it["kind"] == k]]
                for i in range(0, len(grp), BATCH)]
            if cell != "base":
                pm.set_adapter(cell)
            ps = time.time()
            fallback = 0
            for grp in groups:
                try:
                    with pm.disable_adapter() if cell == "base" else contextlib.nullcontext():
                        res = score(inner, softcap, grp, layout == "batch8", dev)
                except torch.cuda.OutOfMemoryError:  # batched layout only: halve the batch, keep it a padded batch
                    if layout != "batch8":
                        raise
                    torch.cuda.empty_cache()
                    fallback += 1
                    res = []
                    for half in (grp[:len(grp) // 2], grp[len(grp) // 2:]):
                        with pm.disable_adapter() if cell == "base" else contextlib.nullcontext():
                            res += score(inner, softcap, half, True, dev)
                for it, r in zip(grp, res):
                    out.write(json.dumps({"stack": "hf_" + base_kind, "stage": STAGE, "cell": cell, "layout": layout,
                                          "item_id": it["item_id"], "target_ids": it["target_ids"], **r}) + "\n")
            out.flush()
            last = time.time() - ps
            passes.append({"cell": cell, "layout": layout, "forwards": len(groups), "oom_halved_batches": fallback, "seconds": round(last, 1),
                           "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev == "cuda" else None,
                           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if dev == "cuda" else None})
            print(json.dumps(passes[-1]), flush=True)
    out.close()
    del pm, model, inner
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    report = {"stage": STAGE, "stage_start_utc": stage_start, "stage_end_utc": utc(), "seconds": round(time.time() - t0, 1),
              "load_seconds": load_s, "env": env, "checks": checks, "passes": passes, "skipped": skipped,
              "modules": {k: {x: y for x, y in v.items() if x != "module_names"} for k, v in modules.items()}}
    with open(OUT_S1 + "/hf_modules_%s.json" % STAGE, "x") as f:
        json.dump({k: v["module_names"] for k, v in modules.items()}, f, indent=0)
    with open(OUT_S1 + "/hf_stage_%s.json" % STAGE, "x") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"stage": STAGE, "status": "done", "seconds": report["seconds"], "skipped": skipped,
                      "modules": {k: [v["modules"], v["tensors_equal_to_file"], v["n_layers"]] for k, v in modules.items()}}))


if __name__ == "__main__":
    main()
