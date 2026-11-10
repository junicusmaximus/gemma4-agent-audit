#!/usr/bin/env python3
"""LoRA training on the dequantized QAT base (boss v053): a thin wrapper around the frozen trainers.

Imports by file path, without editing them: build_model / save_adapter / check_gpu from sft/train_lora.py and
verify_pair / fixed_denominator_train from the Codex trainer sft/train_mask_control_v035.py. Overrides ONLY the base
path and the output dir; data, update groups, hyper-parameters, loss normalisation, seeding order and LoRA init are
those of the sealed M0 run (contracts/mask-control-v0351-APPROVED.json, sft/runs/A1_M0_v035.meta.json).
  --fixture tfit5 --epochs N --grad-accum 5  the 5 T-fit items of the v048 ledger as training sequences (M0 chunk 0
                                             cut after completion 0); --base dequant (F_dq) or bf16 (F_bf control)
  no --fixture                               M0q: the full M0 data on the dequantized base (needs --m0q-go as well)
Without --execute: CPU inspection of every identity gate (no model, no GPU). With --execute: GPU training that refuses
to start unless the LoRA init hash equals M0's and 258 LoRA modules exist; adapter, meta.json and train_log.jsonl are
written into --out (a fresh directory). Run with PYTHONDONTWRITEBYTECODE=1 in the train env."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True

ROOT = "PROJECT_ROOT/gemma4"
SFT = ROOT + "/sft"
TRAINER = SFT + "/train_lora.py"
CODEX = SFT + "/train_mask_control_v035.py"
PREPARE = SFT + "/prepare_mask_control_v035.py"
CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"
CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
M0_META = SFT + "/runs/A1_M0_v035.meta.json"
LEDGER = ROOT + "/analysis/first-step-probe-boss-v048/ledger.jsonl"
LEDGER_SHA = "bca7ae2274a38b7d57ea17d5785031dc6c9d061a2cdfef8a7b1184ce696d6273"  # v048 ledger_meta.json
BASES = {"dequant": ROOT + "/models/gemma-4-e4b-it-qat-dequant-bf16", "bf16": ROOT + "/models/gemma-4-e4b-it"}
BF16_MODEL_SHA = "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
N_TFIT, LORA_MODULES, ADAPTER_TENSORS = 5, 258, 516


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_trainers():
    """train_lora and the Codex prepare module are registered under their import names first, so the Codex trainer's
    `import train_lora as original` binds the very module object used here."""
    original = load_module("train_lora", TRAINER)
    load_module("prepare_mask_control_v035", PREPARE)
    codex = load_module("train_mask_control_v035", CODEX)
    if codex.original is not original:
        raise RuntimeError("the Codex trainer bound a different train_lora module")
    return original, codex


def verify_base(kind, bases=BASES, bf16_sha=BF16_MODEL_SHA):
    """dequant: manifest.json present, directory listing as built, every shard and side file equal to the manifest.
    bf16: model.safetensors equal to the pinned sha."""
    base = bases[kind]
    if kind == "bf16":
        sha = sha256_file(os.path.join(base, "model.safetensors"))
        if sha != bf16_sha:
            raise ValueError("bf16 model.safetensors changed: " + sha)
        return {"kind": kind, "dir": base, "model_sha256": sha}
    manifest_path = os.path.join(base, "manifest.json")
    manifest = read_json(manifest_path)
    files = {r["name"]: r["sha256"] for r in manifest["shards"]}
    files.update(manifest["files_sha256"])
    if set(os.listdir(base)) != set(files) | {"manifest.json"}:
        raise ValueError("dequantized base dir listing differs from its manifest")
    for name, sha in sorted(files.items()):
        if sha256_file(os.path.join(base, name)) != sha:
            raise ValueError("dequantized base file changed: " + name)
    return {"kind": kind, "dir": base, "manifest_sha256": sha256_file(manifest_path), "counts": manifest["counts"]}


def fixture_tfit5(seqs, ledger_path=LEDGER, ledger_sha=LEDGER_SHA):
    """The T-fit items of the v048 ledger as training sequences: chunk 0 of each trajectory in the verified M0 data,
    cut after the end of its first supervised run; ids and the supervised span must equal the ledger item."""
    import numpy as np
    if sha256_file(ledger_path) != ledger_sha:
        raise ValueError("v048 ledger changed")
    with open(ledger_path) as f:
        items = [row for row in map(json.loads, f) if row["kind"] == "T-fit"]
    if len(items) != N_TFIT:
        raise ValueError("expected %d T-fit items, found %d" % (N_TFIT, len(items)))
    first = {}
    for tid, ids, lab in seqs:
        first.setdefault(tid, (ids, lab))
    out = []
    for it in items:
        tid = it["source"]["traj_id"]
        ids, lab = first[tid]
        start, end = it["target_span"]
        cut_ids, cut_lab = ids[:end].copy(), lab[:end].copy()
        sup = np.nonzero(cut_lab != -100)[0]
        if (cut_ids.tolist() != it["ids"] or sup.size != end - start or sup.size == 0 or int(sup[0]) != start
                or (cut_lab[start:end] != cut_ids[start:end]).any() or it["source"].get("chunk") != 0):
            raise ValueError("T-fit item does not match chunk 0 of the M0 data: " + tid)
        out.append((tid, cut_ids, cut_lab))
    return out, items


def fixture_groups(seqs, epochs, grad_accum, seed):
    """train_lora.train's order (random.Random(seed), one shuffle per epoch, consecutive grad_accum slices) as frozen
    groups with that loop's denominators (count of labels[1:] != -100 in the group)."""
    rng, order = random.Random(seed), []
    for _ in range(epochs):
        perm = list(range(len(seqs)))
        rng.shuffle(perm)
        order += perm
    groups = []
    for s in range(0, len(order), grad_accum):
        idx = order[s:s + grad_accum]
        denominator = sum(int((seqs[i][2][1:] != -100).sum()) for i in idx)
        groups.append({"update": len(groups) + 1, "sequence_indices": idx, "original_denominator": denominator,
                       "m1_remaining_targets": denominator})
    return groups


def check_overrides(args, m0_args, fixture):
    """Differences from the sealed M0 args: base and out only (plus epochs / grad_accum for the fixture)."""
    diff = sorted(k for k in set(vars(args)) | set(m0_args) if vars(args).get(k) != m0_args.get(k))
    allowed = {"base", "out"} | ({"epochs", "grad_accum"} if fixture else set())
    if set(diff) - allowed:
        raise ValueError("arguments differ from M0 beyond the declared overrides: %s" % sorted(set(diff) - allowed))
    return diff


def init_hash(model):
    """Codex trainer's initial_trainable_parameters_sha256 (train_mask_control_v035.py:144-148), same computation."""
    import torch
    initial = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    h = hashlib.sha256()
    for name, tensor in sorted(initial.items()):
        h.update(name.encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def check_init(digest, n_lora, expected_digest):
    if digest != expected_digest or n_lora != LORA_MODULES:
        raise SystemExit("REFUSED: init hash %s (M0 %s), LoRA modules %d (expected %d); nothing trained"
                         % (digest, expected_digest, n_lora, LORA_MODULES))


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", choices=["tfit5"], help="train on the 5 T-fit items of the v048 ledger")
    ap.add_argument("--epochs", type=int, help="fixture only")
    ap.add_argument("--grad-accum", type=int, help="fixture only")
    ap.add_argument("--base", choices=sorted(BASES), default="dequant")
    ap.add_argument("--out", required=True, help="fresh adapter directory")
    ap.add_argument("--execute", action="store_true", help="GPU training (default: CPU inspection only)")
    ap.add_argument("--m0q-go", action="store_true", help="required with --execute for the full M0q run")
    opts = ap.parse_args(argv)
    if opts.fixture and (not opts.epochs or not opts.grad_accum):
        ap.error("--fixture needs --epochs and --grad-accum")
    if not opts.fixture and (opts.epochs or opts.grad_accum or opts.base != "dequant"):
        ap.error("--epochs, --grad-accum and --base bf16 exist for the fixture only")
    if opts.execute and not opts.fixture and not opts.m0q_go:
        ap.error("the full M0q run needs --m0q-go (separate pre-registered go)")
    opts.out = os.path.normpath(os.path.abspath(opts.out))
    return opts


def prepare(opts, codex, original):
    """Every CPU gate: contract, Codex verify_pair, base files, fixture data, declared overrides."""
    if os.path.lexists(opts.out):
        raise FileExistsError("fresh output required: " + opts.out)
    if sha256_file(CONTRACT) != CONTRACT_SHA:
        raise ValueError("approved contract changed")
    manifest, sequences, groups = codex.verify_pair(read_json(CONTRACT))
    base = verify_base(opts.base)
    m0_meta = read_json(M0_META)
    args = argparse.Namespace(**manifest["hyperparameters"], variant="M0", base=BASES[opts.base], out=opts.out,
                              device="cuda", served_base_name=original.SERVED_BASE)
    seqs, fixture = sequences[0], None
    if opts.fixture:
        seqs, items = fixture_tfit5(sequences[0])
        args.epochs, args.grad_accum = opts.epochs, opts.grad_accum
        groups = fixture_groups(seqs, args.epochs, args.grad_accum, args.seed)
        fixture = {"name": opts.fixture, "ledger": LEDGER, "ledger_sha256": LEDGER_SHA,
                   "items": [it["item_id"] for it in items], "sequence_tokens": [len(s[1]) for s in seqs],
                   "targets_per_sequence": [int((s[2][1:] != -100).sum()) for s in seqs],
                   "group_orders": [g["sequence_indices"] for g in groups]}
    overrides = check_overrides(args, m0_meta["args"], bool(opts.fixture))
    return {"args": args, "seqs": seqs, "groups": groups, "manifest": manifest, "base": base, "m0_meta": m0_meta,
            "fixture": fixture, "overrides": overrides}


def execute(opts, state, original, codex):
    """Codex main's seeding order and init-hash computation, the init gate, training, adapter and records."""
    import numpy as np
    import torch
    from peft.tuners.lora import LoraLayer
    args = state["args"]
    original.PROBE_DIR = state["manifest"]["probe_dir"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    original.check_gpu(2.0)
    torch.cuda.manual_seed_all(args.seed)
    model = original.build_model(args)
    torch.cuda.reset_peak_memory_stats()
    n_lora = sum(isinstance(m, LoraLayer) for m in model.modules())
    digest = init_hash(model)
    check_init(digest, n_lora, state["m0_meta"]["initial_trainable_parameters_sha256"])
    logs, seconds, seen = codex.fixed_denominator_train(args, state["seqs"], state["groups"], model)
    n_tensors, names_match = original.save_adapter(args, model)
    with open(os.path.join(args.out, "train_log.jsonl"), "x") as f:
        f.write("".join(json.dumps(r) + "\n" for r in logs))
    return {"initial_trainable_parameters_sha256": digest, "init_hash_equals_m0": True, "lora_modules": n_lora,
            "adapter_tensors": n_tensors, "names_match_probe": names_match, "seconds": round(seconds, 1),
            "tokens": seen, "updates": len(logs), "loss_first": logs[0]["loss"], "loss_last": logs[-1]["loss"],
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "gates_pass": n_tensors == ADAPTER_TENSORS and names_match is True}


def main(argv=None):
    opts = parse(argv)
    started = utc()
    original, codex = load_trainers()
    state = prepare(opts, codex, original)
    record = {"wrapper": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
              "mode": "fixture-" + opts.fixture if opts.fixture else "m0q", "started_utc": started,
              "args": vars(state["args"]), "overrides_vs_m0": state["overrides"], "base": state["base"],
              "contract_sha256": CONTRACT_SHA, "codex_manifest_sha256": sha256_file(SFT + "/data/mask_control_v035/manifest.json"),
              "frozen_code_sha256": {p: sha256_file(p) for p in (TRAINER, CODEX, PREPARE)},
              "m0_initial_trainable_parameters_sha256": state["m0_meta"]["initial_trainable_parameters_sha256"],
              "updates_planned": len(state["groups"]), "sequences": len(state["seqs"]), "fixture": state["fixture"]}
    if not opts.execute:
        print("INSPECTION " + json.dumps(record), flush=True)
        return 0
    record.update(execute(opts, state, original, codex), ended_utc=utc())
    with open(os.path.join(opts.out, "meta.json"), "x") as f:
        f.write(json.dumps(record, indent=1) + "\n")
    print("SUMMARY " + json.dumps({k: record[k] for k in ("mode", "initial_trainable_parameters_sha256", "lora_modules",
                                                          "adapter_tensors", "names_match_probe", "updates", "loss_first",
                                                          "loss_last", "seconds", "gates_pass")}), flush=True)
    return 0 if record["gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
