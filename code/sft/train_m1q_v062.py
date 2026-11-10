#!/usr/bin/env python3
"""LoRA training of M1q on the dequantized QAT base (boss v062): train_qatbase_v053.py (M0q) with one change, the variant.

M1q = the mask-control M1 recipe (variant M1 of the frozen Codex trainer sft/train_mask_control_v035.py: empty-thought
targets masked, the update's M1 remaining-target count enforced, M0's original denominators) on the dequantized QAT base,
pre-registered in prereg-2026-11-07-m1q-mask-on-qat-base-v062.md (its sha is a leaf of the training contract).
Imports by file path, without editing them: build_model / save_adapter / check_gpu from sft/train_lora.py and
verify_pair / fixed_denominator_train from the Codex trainer. Overrides ONLY the base path and the output dir of the
sealed M1 run (sft/runs/A1_M1_v035.meta.json); data, update groups, hyper-parameters, loss normalisation, seeding order
and LoRA init are those of the sealed M0 / M1 / M0q runs (contracts/mask-control-v0351-APPROVED.json). The mask and the
denominator logic are the frozen ones and are not re-implemented here.
  --verify            CPU only: training contract and pins, sealed references, data files, dequantized base files, the
                      Codex verify_pair on the M0/M1 pair, and the groups / denominators / M1 target count of the
                      arrays that would be trained; takes no lease, uses no GPU
  --smoke             GPU: the --verify gates, then the first 3 updates into sft/runs/A1_M1q_v062_smoke
  --execute --m1q-go  GPU: the --verify gates, then all 208 updates into sft/runs/A1_M1q_v062
Both GPU stages check the contract passed with --contract before any lease is touched. --execute refuses unless it has
status APPROVED and a prereg sha that is filled in and equal to the frozen file. --smoke leaves a receipt only (never
evaluated), so it also runs from the DRAFT while the prereg leaf is still PENDING_FREEZE; any other status is refused,
and a filled leaf must equal its file. Each stage then runs in its own process: it takes the evaluation and training
leases (flock, non-blocking; refuses while either is held), seeds 20261103, builds a fresh LoRA whose init hash must
equal M0's (9dfc7ccc...) and trains with a fresh AdamW and the stateless LR schedule. There is no resume, checkpoint,
adapter or output option; both run dirs are pinned and must not exist, so smoke state can never reach the main run.
Run with PYTHONDONTWRITEBYTECODE=1 in the train env and without an external flock on the lease files."""
import argparse
import hashlib
import importlib.util
import json
import math
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
V035_CONTRACT = ROOT + "/contracts/mask-control-v035-APPROVED.json"
V035_CONTRACT_SHA = "207889e935122d1a057e6acaca3a265afaff7ec43516d2ae5d40c9de81c75e31"
M0_CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"  # the contract recorded by the sealed M0 / M1 / M0q
M0_CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
CONTRACT = ROOT + "/contracts/m1q-training-v062-DRAFT.json"
SEALED_METAS = {"m0": SFT + "/runs/A1_M0_v035.meta.json", "m1": SFT + "/runs/A1_M1_v035.meta.json",
                "m0q": SFT + "/runs/A1_M0q_v053/meta.json"}
DATA_DIR = SFT + "/data/mask_control_v035"
DATA_FILES = ("M0.jsonl", "M1.jsonl", "identity_ledger.jsonl", "update_groups.json", "manifest.json")
OUTS = {"smoke": SFT + "/runs/A1_M1q_v062_smoke", "main": SFT + "/runs/A1_M1q_v062"}
SMOKE_UPDATES = 3  # about 3.5 min of compute at the M0q rate
LOCKS = (ROOT + "/migration/recovery-screen-v034/evaluation.lock", ROOT + "/migration/mask-control-v0351/training.lock")
BASE = ROOT + "/models/gemma-4-e4b-it-qat-dequant-bf16"
BASE_MANIFEST_SHA = "20f1c2f9a52eea099c7ab95fca57c150d4e3b1946663136f11d11e0f50849ce5"
INIT_SHA = "9dfc7ccc71fc4631d51a61dad8a889040e76bd9c87a2546fe24ec3b769e20066"  # M0 / M1 / M0q initial trainable parameters
SERVED_BASE = "google/gemma-4-e4b-it-qat-w4a16-ct"
VARIANT = "M1"
EXPECTED = {"sequences": 1661, "updates": 208, "m0_targets": 4391649, "m1_targets": 4365092, "tokens": 20922041}
SAME_IN_BOTH_CONTRACTS = ("source_hashes", "artifact_hashes", "hyperparameters", "base", "probe_dir", "fresh_outputs",
                          "manifest_sha256", "trajectories", "chunks", "m0_shifted_targets", "m1_shifted_targets",
                          "masked_shifted_targets", "update_groups")
SAME_IN_MANIFEST = ("source_hashes", "artifact_hashes", "hyperparameters", "base", "probe_dir", "fresh_outputs")
PENDING = "PENDING_FREEZE"  # the prereg leaf of the training contract until the prereg is frozen
LORA_MODULES, ADAPTER_TENSORS = 258, 516


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


def dig(obj, keys):
    for k in keys:
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def check_reference_contracts(v035, m0_contract):
    """The v035 contract named by the prereg and the v0351 contract recorded by the sealed runs carry the same science:
    sources, artifacts, hyper-parameters, base, probe, outputs, manifest and every count."""
    bad = [k for k in SAME_IN_BOTH_CONTRACTS if v035.get(k) != m0_contract.get(k)]
    if bad:
        raise ValueError("v035 and v0351 contracts differ in: " + ", ".join(bad))


def expected_population(v035):
    """The population the v035 contract declares (m1_shifted_targets included); it must equal this wrapper's constants."""
    expected = {"sequences": v035["chunks"], "updates": v035["update_groups"], "m0_targets": v035["m0_shifted_targets"],
                "m1_targets": v035["m1_shifted_targets"], "tokens": EXPECTED["tokens"]}
    if expected != EXPECTED:
        raise ValueError("v035 contract population differs from this wrapper's: " + json.dumps(expected))
    return expected


def check_sealed_runs(metas):
    """The sealed M0, M1 and M0q runs share the init hash, token count and contract that M1q must reproduce."""
    for name, meta in sorted(metas.items()):
        if (meta["initial_trainable_parameters_sha256"], meta["tokens"], meta["contract_sha256"]) != (
                INIT_SHA, EXPECTED["tokens"], M0_CONTRACT_SHA):
            raise ValueError("sealed %s run differs from the pinned init hash, token count or contract" % name)


def check_data_files(files, m0_contract):
    """The five data files equal the sealed contract's artifact hashes and manifest sha."""
    want = {os.path.basename(p): s for p, s in m0_contract["artifact_hashes"].items()}
    want["manifest.json"] = m0_contract["manifest_sha256"]
    bad = sorted(k for k in set(files) | set(want) if files.get(k) != want.get(k))
    if bad:
        raise ValueError("data files differ from the sealed contract: " + ", ".join(bad))


def check_manifest(manifest, m0_contract):
    """Codex main's approved-contract drift check (train_mask_control_v035.py:127-129)."""
    bad = [k for k in SAME_IN_MANIFEST if m0_contract[k] != manifest[k]]
    if bad or manifest["m1_shifted_targets"] != EXPECTED["m1_targets"]:
        raise ValueError("approved contract drift against the data manifest: " + ", ".join(bad or ["m1_shifted_targets"]))


def check_contract(contract, v035, files, wrapper_sha, code, metas_sha):
    """The training contract pins exactly this wrapper, its constants, the v035 contract's M1 target count, the data
    files, the frozen code and the sealed references."""
    want = {("arm",): "M1q", ("variant",): VARIANT, ("wrapper", "sha256"): wrapper_sha,
            ("v035_contract", "sha256"): V035_CONTRACT_SHA,
            ("v035_contract", "m1_shifted_targets"): v035["m1_shifted_targets"],
            ("reference_contract", "sha256"): M0_CONTRACT_SHA, ("base", "dir"): BASE,
            ("base", "manifest_sha256"): BASE_MANIFEST_SHA, ("initial_trainable_parameters_sha256",): INIT_SHA,
            ("hyperparameters",): v035["hyperparameters"], ("data", "dir"): DATA_DIR,
            ("data", "expected_sha256"): files, ("data", "sequences"): EXPECTED["sequences"],
            ("data", "updates"): EXPECTED["updates"], ("data", "m0_shifted_targets"): EXPECTED["m0_targets"],
            ("data", "m1_shifted_targets"): EXPECTED["m1_targets"], ("data", "input_tokens"): EXPECTED["tokens"],
            ("stages", "smoke", "out"): OUTS["smoke"], ("stages", "smoke", "updates"): SMOKE_UPDATES,
            ("stages", "main", "out"): OUTS["main"], ("stages", "main", "updates"): EXPECTED["updates"],
            ("leases", "paths"): list(LOCKS), ("code_sha256", "trainer"): code[TRAINER],
            ("code_sha256", "codex_trainer"): code[CODEX], ("code_sha256", "prepare"): code[PREPARE]}
    for name, sha in metas_sha.items():
        want[("sealed_runs", name, "sha256")] = sha
    bad = [".".join(keys) for keys, value in want.items() if dig(contract, keys) != value]
    if bad:
        raise ValueError("training contract differs from this wrapper: " + ", ".join(bad))


def check_prereg(contract, require_frozen):
    """The pre-registration named by the contract. Its sha is PENDING_FREEZE until the freeze; --verify accepts that,
    a GPU stage does not. Once filled in, the sha must equal the file."""
    leaf = contract.get("prereg") or {}
    path, sha = leaf.get("path"), leaf.get("sha256")
    if sha == PENDING:
        if require_frozen:
            raise SystemExit("REFUSED: the pre-registration is not frozen (prereg sha256 is %s); nothing trained"
                             % PENDING)
        return {"path": path, "sha256": PENDING}
    try:
        actual = sha256_file(path)
    except (OSError, TypeError):
        actual = None
    if not isinstance(sha, str) or len(sha) != 64 or actual != sha:
        raise ValueError("the frozen pre-registration is missing or changed")
    return {"path": path, "sha256": sha}


def check_approved(contract):
    """The main run starts only from an APPROVED copy of the training contract; DRAFT or any other status is refused."""
    if contract.get("status") != "APPROVED":
        raise SystemExit("REFUSED: training contract status is %r, not APPROVED; nothing trained" % contract.get("status"))


def check_gpu_contract(contract, stage):
    """What a GPU stage needs from the contract before any lease is touched. main: status APPROVED and a frozen prereg.
    smoke (a receipt only): status DRAFT or APPROVED, and a prereg leaf that is pending or equal to its file."""
    if stage == "main":
        check_approved(contract)
        check_prereg(contract, True)
        return
    if contract.get("status") not in ("DRAFT", "APPROVED"):
        raise SystemExit("REFUSED: training contract status is %r, not DRAFT or APPROVED; nothing trained"
                         % contract.get("status"))
    check_prereg(contract, False)


def verify_base(base=BASE):
    """train_qatbase_v053.verify_base, dequantized branch: manifest.json equal to the pinned sha, directory listing as
    built, every shard and side file equal to the manifest."""
    manifest_path = os.path.join(base, "manifest.json")
    if sha256_file(manifest_path) != BASE_MANIFEST_SHA:
        raise ValueError("dequantized base manifest changed")
    manifest = read_json(manifest_path)
    files = {r["name"]: r["sha256"] for r in manifest["shards"]}
    files.update(manifest["files_sha256"])
    if set(os.listdir(base)) != set(files) | {"manifest.json"}:
        raise ValueError("dequantized base dir listing differs from its manifest")
    for name, sha in sorted(files.items()):
        if sha256_file(os.path.join(base, name)) != sha:
            raise ValueError("dequantized base file changed: " + name)
    return {"kind": "dequant", "dir": base, "manifest_sha256": BASE_MANIFEST_SHA, "counts": manifest["counts"]}


def shifted_targets(labels):
    """Supervised positions after the causal shift: the count that the Codex trainer's denominators use."""
    return int((labels[1:] != -100).sum())


def check_population(sequences, groups, expected):
    """What is trained is what was checked: each update carries exactly M0's original denominator of M0 targets and the
    update's m1_remaining_targets of M1 targets (the counts fixed_denominator_train compares), every sequence is in
    exactly one update, and the totals equal the declared population, M1's target count included."""
    m0, m1 = sequences
    if (len(m0), len(m1), len(groups)) != (expected["sequences"], expected["sequences"], expected["updates"]):
        raise ValueError("sequence or update count differs from the declared population")
    t0, t1 = [shifted_targets(s[2]) for s in m0], [shifted_targets(s[2]) for s in m1]
    used, denominator, remaining = [], 0, 0
    for g in groups:
        idx = g["sequence_indices"]
        if sum(t0[i] for i in idx) != g["original_denominator"] or sum(t1[i] for i in idx) != g["m1_remaining_targets"]:
            raise ValueError("update %s: targets differ from the frozen denominators" % g["update"])
        used += idx
        denominator += g["original_denominator"]
        remaining += g["m1_remaining_targets"]
    if sorted(used) != list(range(len(m1))):
        raise ValueError("sequences are not used exactly once")
    tokens = sum(len(s[1]) for s in m1)
    got = {"m0_targets": sum(t0), "m1_targets": sum(t1), "tokens": tokens}
    sums = {"m0_targets": denominator, "m1_targets": remaining}
    if got != {k: expected[k] for k in got} or sums != {k: expected[k] for k in sums}:
        raise ValueError("population totals differ from the declared M0 / M1 target counts and tokens: " + json.dumps(got))
    return {"sequences": len(m1), "updates": len(groups), "m0_targets": sum(t0), "m1_targets": sum(t1),
            "masked_targets": sum(t0) - sum(t1), "original_denominator_sum": denominator,
            "m1_remaining_targets_sum": remaining, "tokens": tokens, "max_len": max(len(s[1]) for s in m1)}


def select_sequences(sequences, variant=VARIANT):
    """The arrays the Codex trainer's main trains for a variant (train_mask_control_v035.py:154): M0 or M1."""
    return sequences[0 if variant == "M0" else 1]


def estimates(seqs, groups):
    """Training time at the M0q rate (its tokens / seconds): the smoke's first updates and the full run."""
    meta = read_json(SEALED_METAS["m0q"])
    rate = meta["tokens"] / meta["seconds"]
    tokens = [sum(len(seqs[x][1]) for x in g["sequence_indices"]) for g in groups]
    smoke, total = sum(tokens[:SMOKE_UPDATES]), sum(tokens)
    return {"m0q_tokens_per_s": round(rate, 1), "smoke_updates": SMOKE_UPDATES, "smoke_tokens": smoke,
            "smoke_minutes_at_m0q_rate": round(smoke / rate / 60, 1), "main_seconds_at_m0q_rate": round(total / rate),
            "main_hours_at_m0q_rate": round(total / rate / 3600, 2), "max_update_tokens": max(tokens)}


def verify(contract_path, require_frozen=False):
    """--verify: training contract and pins, sealed references, data files, dequantized base, the Codex verify_pair on
    the M0/M1 pair, and the groups / denominators / M1 target count of the arrays that would be trained."""
    contract = read_json(contract_path)
    prereg = check_prereg(contract, require_frozen)
    for path, sha in ((V035_CONTRACT, V035_CONTRACT_SHA), (M0_CONTRACT, M0_CONTRACT_SHA)):
        if sha256_file(path) != sha:
            raise ValueError("approved mask-control contract changed: " + path)
    v035, m0_contract = read_json(V035_CONTRACT), read_json(M0_CONTRACT)
    check_reference_contracts(v035, m0_contract)
    expected = expected_population(v035)
    metas = {name: read_json(path) for name, path in SEALED_METAS.items()}
    check_sealed_runs(metas)
    files = {n: sha256_file(os.path.join(DATA_DIR, n)) for n in DATA_FILES}
    check_data_files(files, m0_contract)
    code = {p: sha256_file(p) for p in (TRAINER, CODEX, PREPARE)}
    check_contract(contract, v035, files, sha256_file(os.path.abspath(__file__)), code,
                   {name: sha256_file(path) for name, path in SEALED_METAS.items()})
    base = verify_base()
    original, codex = load_trainers()
    manifest, sequences, groups = codex.verify_pair(m0_contract)
    check_manifest(manifest, m0_contract)
    population = check_population(sequences, groups, expected)
    seqs = select_sequences(sequences)
    return {"contract": contract, "prereg": prereg, "v035": v035, "m0_contract": m0_contract, "metas": metas,
            "files": files, "code": code, "base": base, "original": original, "codex": codex, "seqs": seqs,
            "groups": groups, "population": population, "estimates": estimates(seqs, groups)}


def build_args(hyperparameters, out):
    """The Codex main's argument namespace for variant M1 (train_mask_control_v035.py:130-132), on the dequantized base."""
    return argparse.Namespace(**hyperparameters, variant=VARIANT, base=BASE, out=out, device="cuda",
                              served_base_name=SERVED_BASE)


def check_overrides(args, ref_args, allowed):
    """Differences from a sealed run's args, which must stay inside the declared overrides."""
    diff = sorted(k for k in set(vars(args)) | set(ref_args) if vars(args).get(k) != ref_args.get(k))
    extra = sorted(set(diff) - set(allowed))
    if extra:
        raise ValueError("arguments differ from the sealed run beyond the declared overrides: %s" % extra)
    return diff


def fresh_out(stage, outs=OUTS):
    """The stage's pinned run dir: apart from the other stage's dir and not existing (no resume, no reuse)."""
    out, other = outs[stage], outs["smoke" if stage == "main" else "main"]
    if out == other or out.startswith(other + "/") or other.startswith(out + "/"):
        raise ValueError("smoke and main run dirs overlap")
    if os.path.lexists(out):
        raise FileExistsError("fresh output required: " + out)
    return out


def stage_groups(stage, groups):
    """smoke: the first SMOKE_UPDATES updates of the M1 schedule; main: all of them."""
    return groups[:SMOKE_UPDATES] if stage == "smoke" else groups


def take_leases(paths=LOCKS):
    """The evaluation and training leases of the gemma4 runners (fcntl.flock, exclusive, non-blocking), held by the
    returned open files until the process exits; refuses while another job holds either."""
    import fcntl
    held = []
    for path in paths:
        f = open(path, "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            for g in held:
                g.close()
            raise SystemExit("REFUSED: lease held by another job: %s; nothing trained" % path)
        held.append(f)
    return held


def gates_pass(n_tensors, names_match, logs, updates):
    """Adapter layout as the probe's, every logged loss and gradient norm finite, and every planned update trained."""
    finite = all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in logs)
    return n_tensors == ADAPTER_TENSORS and names_match is True and finite and len(logs) == updates > 0


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="CPU gates only; no lease, no model, no GPU")
    mode.add_argument("--smoke", action="store_true", help="GPU: the first %d updates into %s; contract status DRAFT or "
                      "APPROVED" % (SMOKE_UPDATES, OUTS["smoke"]))
    mode.add_argument("--execute", action="store_true", help="GPU: all 208 updates into %s" % OUTS["main"])
    ap.add_argument("--m1q-go", action="store_true", help="required with --execute (pre-registered go)")
    ap.add_argument("--contract", default=CONTRACT, help="training contract")
    opts = ap.parse_args(argv)
    if opts.m1q_go and not opts.execute:
        ap.error("--m1q-go belongs to --execute")
    if opts.execute and not opts.m1q_go:
        ap.error("--execute needs --m1q-go (separate pre-registered go)")
    opts.stage = "smoke" if opts.smoke else ("main" if opts.execute else None)
    return opts


def prepare(opts):
    """Every CPU gate of a GPU stage: a fresh pinned run dir, the --verify gates (a frozen prereg is required for the main
    run only) and the declared overrides against the sealed M1 run and the M0q run."""
    out = fresh_out(opts.stage)
    state = verify(opts.contract, require_frozen=opts.stage == "main")
    args = build_args(state["m0_contract"]["hyperparameters"], out)
    overrides = {"vs_sealed_m1": check_overrides(args, state["metas"]["m1"]["args"], {"base", "out"}),
                 "vs_m0q": check_overrides(args, state["metas"]["m0q"]["args"], {"variant", "out"})}
    return dict(state, args=args, groups=stage_groups(opts.stage, state["groups"]), overrides=overrides)


def execute(state, original, codex):
    """Codex main's seeding order and init-hash computation, the init gate, then training from that fresh LoRA init
    with a fresh optimizer into the stage's fresh run dir. Nothing is loaded from an earlier run."""
    import numpy as np
    import torch
    from peft.tuners.lora import LoraLayer
    args = state["args"]
    if original.SERVED_BASE != SERVED_BASE:
        raise ValueError("train_lora.SERVED_BASE changed")
    original.PROBE_DIR = state["m0_contract"]["probe_dir"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    original.check_gpu(2.0)
    torch.cuda.manual_seed_all(args.seed)
    model = original.build_model(args)
    torch.cuda.reset_peak_memory_stats()
    n_lora = sum(isinstance(m, LoraLayer) for m in model.modules())
    digest = init_hash(model)
    check_init(digest, n_lora, INIT_SHA)
    logs, seconds, seen = codex.fixed_denominator_train(args, state["seqs"], state["groups"], model)
    n_tensors, names_match = original.save_adapter(args, model)
    with open(os.path.join(args.out, "train_log.jsonl"), "x") as f:
        f.write("".join(json.dumps(r) + "\n" for r in logs))
    return {"initial_trainable_parameters_sha256": digest, "init_hash_equals_m0": True, "lora_modules": n_lora,
            "adapter_tensors": n_tensors, "names_match_probe": names_match, "seconds": round(seconds, 1),
            "tokens": seen, "updates": len(logs), "loss_first": logs[0]["loss"], "loss_last": logs[-1]["loss"],
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "gates_pass": gates_pass(n_tensors, names_match, logs, len(state["groups"]))}


def main(argv=None):
    opts = parse(argv)
    started = utc()
    if opts.verify:
        state = verify(opts.contract)
        print("VERIFY " + json.dumps({"contract_status": state["contract"].get("status"), "prereg": state["prereg"],
                                      "population": state["population"],
                                      "m1_shifted_targets_v035_contract": state["v035"]["m1_shifted_targets"],
                                      "estimates": state["estimates"], "data_files_sha256": state["files"],
                                      "frozen_code_sha256": state["code"], "base": state["base"]}), flush=True)
        return 0
    check_gpu_contract(read_json(opts.contract), opts.stage)  # before any lease is touched
    leases = take_leases()  # held by this process until it exits
    state = prepare(opts)
    args = state["args"]
    record = {"wrapper": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
              "mode": "m1q-" + opts.stage, "started_utc": started, "args": vars(args),
              "overrides": state["overrides"], "base": state["base"],
              "training_contract": {"path": os.path.abspath(opts.contract), "sha256": sha256_file(opts.contract),
                                    "status": state["contract"].get("status")},
              "prereg": state["prereg"], "v035_contract_sha256": V035_CONTRACT_SHA,
              "m0_contract_sha256": M0_CONTRACT_SHA, "data_files_sha256": state["files"],
              "population": state["population"], "estimates": state["estimates"],
              "leases": {"paths": list(LOCKS), "held": len(leases)}, "frozen_code_sha256": state["code"],
              "m0_initial_trainable_parameters_sha256": INIT_SHA, "updates_planned": len(state["groups"]),
              "sequences": len(state["seqs"]),
              "resume": "none: fresh LoRA init (hash gate), fresh AdamW, stateless LR schedule, seed %d" % args.seed}
    record.update(execute(state, state["original"], state["codex"]), ended_utc=utc())
    with open(os.path.join(args.out, "meta.json"), "x") as f:
        f.write(json.dumps(record, indent=1) + "\n")
    print("SUMMARY " + json.dumps({k: record[k] for k in ("mode", "initial_trainable_parameters_sha256", "lora_modules",
                                                          "adapter_tensors", "names_match_probe", "updates", "loss_first",
                                                          "loss_last", "seconds", "gates_pass")}), flush=True)
    return 0 if record["gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
