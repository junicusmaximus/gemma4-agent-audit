#!/usr/bin/env python3
"""CPU smoke test of the wrapper's training path (train env, CUDA hidden): the same calls as
sft/train_qatbase_v053.py execute() on the dequantized base, with the device set to cpu, the 5 fixture sequences
cropped to 96-token windows (train_lora.crop) and a single update. Reports the LoRA init hash (compared with M0's,
informational on CPU), module/tensor counts and names_match_probe; the adapter goes to smoke/F_smoke (this directory)."""
import importlib.util
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/qat-transfer-v053"
spec = importlib.util.spec_from_file_location("train_qatbase_v053", ROOT + "/sft/train_qatbase_v053.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def main():
    import numpy as np
    import torch
    from peft.tuners.lora import LoraLayer
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("ABORT: run with CUDA_VISIBLE_DEVICES set to an empty string")
    t0 = time.time()
    original, codex = w.load_trainers()
    opts = w.parse(["--fixture", "tfit5", "--epochs", "1", "--grad-accum", "5", "--out", OUT + "/smoke/F_smoke"])
    state = w.prepare(opts, codex, original)
    args = state["args"]
    args.device = "cpu"
    original.PROBE_DIR = state["manifest"]["probe_dir"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = original.build_model(args)
    n_lora = sum(isinstance(m, LoraLayer) for m in model.modules())
    digest = w.init_hash(model)
    seqs = original.crop(state["seqs"], 96)
    groups = w.fixture_groups(seqs, 1, 5, args.seed)
    logs, seconds, seen = codex.fixed_denominator_train(args, seqs, groups, model)
    n_tensors, names_match = original.save_adapter(args, model)
    m0 = state["m0_meta"]["initial_trainable_parameters_sha256"]
    print("SMOKE " + json.dumps({"init_hash": digest, "init_hash_equals_m0_on_cpu_path": digest == m0, "lora_modules": n_lora,
                                 "adapter_tensors": n_tensors, "names_match_probe": names_match, "updates": len(logs),
                                 "loss": logs[0]["loss"], "loss_tokens": logs[0]["loss_tokens"], "train_seconds": round(seconds, 1),
                                 "total_seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
