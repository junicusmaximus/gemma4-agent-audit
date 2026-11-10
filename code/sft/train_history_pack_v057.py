#!/usr/bin/env python3
"""LoRA training of Hq1 on the dequantized QAT base (boss v057): train_qatbase_v053.py (M0q) with one change, the data.

Hq1 = the M0q recipe with the rule S history packing of proposal-2026-11-05-history-arm-A1-hist.md (sections 1, 2, 6),
pre-registered in prereg-2026-11-06-history-pack-Hq1-v057.md (FROZEN; its sha is pinned below). Data: the output of
build_history_pack_v057.py --write in sft/data/history_pack_v057: H1.jsonl, parent_ledger.jsonl and an explicit
update_groups.json that puts every sub-chunk into its parent's M0 update. Imports by file path, without editing them:
build_model / save_adapter / check_gpu from sft/train_lora.py and fixed_denominator_train from the Codex trainer
sft/train_mask_control_v035.py. Codex's verify_pair (pinned to the M0/M1 pair) is replaced by the Hq1 gates below; base,
hyper-parameters, the 208 updates with M0's original denominators, seeding order and the LoRA init gate are M0q's.
  --verify            CPU only: frozen prereg, training contract, pinned M0 reference, candidate files equal to the
                      packer manifest and to the contract, the six pack checks against M0 (pair verification) and the
                      groups / denominators of the int32 arrays that would be trained; takes no lease, uses no GPU
  --smoke             GPU: the --verify gates, base files, then the first 3 updates into sft/runs/A1_Hq1_v057_smoke
  --execute --hq1-go  GPU: the --verify gates, base files, then all 208 updates into sft/runs/A1_Hq1_v057
Each GPU stage refuses unless the contract passed with --contract has status APPROVED (a DRAFT is refused before any
lease is touched), then runs in its own process: it takes the evaluation and training leases (flock, non-blocking;
refuses while either is held), seeds 20261103, builds a fresh LoRA whose init hash must equal M0's (9dfc7ccc...) and
trains with a fresh AdamW and the stateless LR schedule. There is no resume, checkpoint, adapter or output option; both run dirs are
pinned and must not exist, so smoke state can never reach the main run. Run with PYTHONDONTWRITEBYTECODE=1 in the train
env and without an external flock on the lease files."""
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
PACKER = SFT + "/build_history_pack_v057.py"
CHECKER = SFT + "/check_history_pack_v057.py"
M0_CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"
M0_CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
CONTRACT = ROOT + "/contracts/history-pack-v057-training-DRAFT.json"
PREREG = "PROJECT_ROOT/briefing/research/gemma-4-2026/prereg-2026-11-06-history-pack-Hq1-v057.md"
PREREG_SHA = "c90558dfebd77a903ada7e2bfc2d06ffedebde6c35d160165f8a5e4257214367"  # re-frozen copy on 151, 08:13:43 UTC
M0_META = SFT + "/runs/A1_M0_v035.meta.json"
M0Q_META = SFT + "/runs/A1_M0q_v053/meta.json"
REF_DIR = SFT + "/data/mask_control_v035"
DATA_DIR = SFT + "/data/history_pack_v057"
DATA_FILES = ("H1.jsonl", "parent_ledger.jsonl", "update_groups.json")
OUTS = {"smoke": SFT + "/runs/A1_Hq1_v057_smoke", "main": SFT + "/runs/A1_Hq1_v057"}
SMOKE_UPDATES = 3  # about 4.4 min of compute at the M0q rate, inside the prereg's <= 10-min smoke cell
LOCKS = (ROOT + "/migration/recovery-screen-v034/evaluation.lock", ROOT + "/migration/mask-control-v0351/training.lock")
BASE = ROOT + "/models/gemma-4-e4b-it-qat-dequant-bf16"
BASE_MANIFEST_SHA = "20f1c2f9a52eea099c7ab95fca57c150d4e3b1946663136f11d11e0f50849ce5"
INIT_SHA = "9dfc7ccc71fc4631d51a61dad8a889040e76bd9c87a2546fe24ec3b769e20066"  # M0 / M0q initial trainable parameters
SERVED_BASE = "google/gemma-4-e4b-it-qat-w4a16-ct"
EXPECTED = {"sequences": 1696, "updates": 208, "targets": 4391649}
CANDIDATE_STATUS = "CANDIDATE_UNAPPROVED_NO_TRAINING"
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


def load_pack_modules():
    """The packer (RULES) and the checker, imported from sft/ the way the packer imports the checker."""
    if SFT not in sys.path:
        sys.path.insert(0, SFT)
    import build_history_pack_v057 as packer
    import check_history_pack_v057 as checker
    return packer, checker


def check_contract(contract, m0_contract):
    """The training contract pins exactly this wrapper, its constants, the frozen prereg, M0's hyper-parameters and the
    pack code."""
    want = {("wrapper", "sha256"): sha256_file(os.path.abspath(__file__)), ("prereg", "sha256"): PREREG_SHA,
            ("reference_contract", "sha256"): M0_CONTRACT_SHA, ("base", "dir"): BASE,
            ("base", "manifest_sha256"): BASE_MANIFEST_SHA, ("initial_trainable_parameters_sha256",): INIT_SHA,
            ("hyperparameters",): m0_contract["hyperparameters"], ("data", "dir"): DATA_DIR,
            ("stages", "smoke", "out"): OUTS["smoke"], ("stages", "smoke", "updates"): SMOKE_UPDATES,
            ("stages", "main", "out"): OUTS["main"], ("stages", "main", "updates"): EXPECTED["updates"],
            ("code_sha256", "packer"): sha256_file(PACKER), ("code_sha256", "checker"): sha256_file(CHECKER)}
    bad = []
    for keys, value in want.items():
        got = contract
        for k in keys:
            got = got.get(k) if isinstance(got, dict) else None
        if got != value:
            bad.append(".".join(keys))
    if bad:
        raise ValueError("training contract differs from this wrapper: " + ", ".join(bad))


def to_sequences(records):
    """(traj_id, int32 input_ids, int32 labels) of every candidate chunk in file order: the Codex trainer's tuples."""
    import numpy as np
    return [(r["traj_id"], np.asarray(c["input_ids"], dtype=np.int32), np.asarray(c["labels"], dtype=np.int32))
            for r in records for c in r["chunks"]]


def check_sequences_and_groups(seqs, groups, ref_groups, max_len):
    """What is trained is what was checked: every sequence aligned and <= max_len; M0's updates in M0's order with M0's
    original denominators; each update's sequences carry exactly that many shifted targets (the count that
    fixed_denominator_train compares per update); every sequence in exactly one update."""
    targets = []
    for _, ids, lab in seqs:
        if len(ids) != len(lab) or len(ids) > max_len or ((lab != -100) & (lab != ids)).any():
            raise ValueError("sequence length or id/label alignment gate failed")
        targets.append(int((lab[1:] != -100).sum()))
    if len(groups) != len(ref_groups):
        raise ValueError("update count %d differs from M0's %d" % (len(groups), len(ref_groups)))
    used = []
    for g, r in zip(groups, ref_groups):
        idx = g.get("sequence_indices") or []
        same = (g.get("update"), g.get("original_denominator"), g.get("parent_sequence_indices")) == (
            r["update"], r["original_denominator"], r["sequence_indices"])
        if not same or any(not isinstance(x, int) or not 0 <= x < len(seqs) for x in idx):
            raise ValueError("update %s differs from M0's" % r["update"])
        if sum(targets[x] for x in idx) != r["original_denominator"]:
            raise ValueError("update %s: targets differ from M0's original denominator" % r["update"])
        used += idx
    if sorted(used) != list(range(len(seqs))):
        raise ValueError("sequences are not used exactly once")
    return {"sequences": len(seqs), "updates": len(groups), "targets": sum(targets),
            "tokens": sum(len(s[1]) for s in seqs), "max_len": max(len(s[1]) for s in seqs)}


def verify_streams(ref_records, ref_groups, cand_records, ledger_rows, cand_groups, rules, **kw):
    """Pair verification (the six pack checks against M0) and the groups / denominators of the sequences that would be
    trained; cand_records() returns a fresh iterator over the candidate records. Raises unless everything passes."""
    _, checker = load_pack_modules()
    report = checker.check_streams(ref_records, cand_records(), ledger_rows, ref_groups, cand_groups, rules, **kw)
    failed = [n for n in map(str, checker.NAMES) if report[n]["status"] != "PASS"]
    if failed:
        raise ValueError("pack checks failed: " + ", ".join(failed))
    seqs = to_sequences(cand_records())
    return report, seqs, check_sequences_and_groups(seqs, cand_groups, ref_groups, rules["max_len"])


def estimates(seqs, groups):
    """Training time at the M0q rate (its tokens / seconds): the smoke's first updates and the full run."""
    meta = read_json(M0Q_META)
    rate = meta["tokens"] / meta["seconds"]
    tokens = [sum(len(seqs[x][1]) for x in g["sequence_indices"]) for g in groups]
    return {"m0q_tokens_per_s": round(rate, 1), "smoke_updates": SMOKE_UPDATES,
            "smoke_tokens": sum(tokens[:SMOKE_UPDATES]),
            "smoke_minutes_at_m0q_rate": round(sum(tokens[:SMOKE_UPDATES]) / rate / 60, 1),
            "main_hours_at_m0q_rate": round(sum(tokens) / rate / 3600, 2), "max_update_tokens": max(tokens)}


def verify(contract_path):
    """--verify: frozen prereg, contract, pinned M0 reference, candidate files, pair verification, groups, denominators."""
    packer, checker = load_pack_modules()
    if sha256_file(PREREG) != PREREG_SHA:
        raise ValueError("the frozen pre-registration changed")
    if sha256_file(M0_CONTRACT) != M0_CONTRACT_SHA:
        raise ValueError("approved M0 contract changed")
    m0_contract, contract = read_json(M0_CONTRACT), read_json(contract_path)
    check_contract(contract, m0_contract)
    for name in ("M0.jsonl", "update_groups.json"):
        path = os.path.join(REF_DIR, name)
        if sha256_file(path) != m0_contract["artifact_hashes"][path]:
            raise ValueError("M0 reference changed: " + name)
    manifest = read_json(os.path.join(DATA_DIR, "manifest.json"))
    files = {n: sha256_file(os.path.join(DATA_DIR, n)) for n in DATA_FILES}
    if (manifest.get("status") != CANDIDATE_STATUS or manifest.get("all_checks_pass") is not True
            or manifest.get("written_sha256") != files or contract["data"]["expected_sha256"] != files):
        raise ValueError("candidate files differ from the packer manifest or the training contract")
    ref_groups = read_json(os.path.join(REF_DIR, "update_groups.json"))
    cand_groups = read_json(os.path.join(DATA_DIR, "update_groups.json"))
    report, seqs, summary = verify_streams(
        checker.read_jsonl(os.path.join(REF_DIR, "M0.jsonl")), ref_groups,
        lambda: checker.read_jsonl(os.path.join(DATA_DIR, "H1.jsonl")),
        checker.read_jsonl(os.path.join(DATA_DIR, "parent_ledger.jsonl")), cand_groups, packer.RULES)
    if {k: summary[k] for k in EXPECTED} != EXPECTED or summary["tokens"] != contract["data"]["input_tokens"]:
        raise ValueError("candidate population differs from the training contract: " + json.dumps(summary))
    return {"contract": contract, "m0_contract": m0_contract, "seqs": seqs, "groups": cand_groups, "summary": summary,
            "files": files, "checks": {n: report[n]["status"] for n in map(str, checker.NAMES)},
            "estimates": estimates(seqs, cand_groups)}


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


def check_overrides(args, m0_args):
    """Differences from the sealed M0 args: base and out only (the M0q overrides)."""
    diff = sorted(k for k in set(vars(args)) | set(m0_args) if vars(args).get(k) != m0_args.get(k))
    if set(diff) - {"base", "out"}:
        raise ValueError("arguments differ from M0 beyond the declared overrides: %s" % sorted(set(diff) - {"base", "out"}))
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
    """smoke: the first SMOKE_UPDATES updates of the Hq1 schedule; main: all of them."""
    return groups[:SMOKE_UPDATES] if stage == "smoke" else groups


def check_approved(contract):
    """GPU modes run only from an APPROVED copy of the training contract; DRAFT or any other status is refused."""
    if contract.get("status") != "APPROVED":
        raise SystemExit("REFUSED: training contract status is %r, not APPROVED; nothing trained" % contract.get("status"))


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


def gates_pass(n_tensors, names_match, logs):
    """Adapter layout as the probe's and every logged loss and gradient norm finite."""
    finite = all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in logs)
    return n_tensors == ADAPTER_TENSORS and names_match is True and finite and bool(logs)


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
    mode.add_argument("--smoke", action="store_true", help="GPU: the first %d updates into %s" % (SMOKE_UPDATES,
                                                                                               OUTS["smoke"]))
    mode.add_argument("--execute", action="store_true", help="GPU: all 208 updates into %s" % OUTS["main"])
    ap.add_argument("--hq1-go", action="store_true", help="required with --execute (pre-registered go)")
    ap.add_argument("--contract", default=CONTRACT, help="training contract")
    opts = ap.parse_args(argv)
    if opts.hq1_go and not opts.execute:
        ap.error("--hq1-go belongs to --execute")
    if opts.execute and not opts.hq1_go:
        ap.error("--execute needs --hq1-go (separate pre-registered go)")
    opts.stage = "smoke" if opts.smoke else ("main" if opts.execute else None)
    return opts


def prepare(opts):
    """Every CPU gate of a GPU stage: the --verify gates, a fresh pinned run dir, the M0 / M0q init hash, base files
    and the declared overrides."""
    state = verify(opts.contract)
    out = fresh_out(opts.stage)
    m0_meta, m0q_meta = read_json(M0_META), read_json(M0Q_META)
    if not m0_meta["initial_trainable_parameters_sha256"] == m0q_meta["initial_trainable_parameters_sha256"] == INIT_SHA:
        raise ValueError("M0 / M0q init hashes differ from the pinned one")
    base = verify_base()
    args = argparse.Namespace(**state["m0_contract"]["hyperparameters"], variant="M0", base=BASE, out=out,
                              device="cuda", served_base_name=SERVED_BASE)
    overrides = check_overrides(args, m0_meta["args"])
    return dict(state, args=args, base=base, overrides=overrides, groups=stage_groups(opts.stage, state["groups"]))


def execute(opts, state, original, codex):
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
            "gates_pass": gates_pass(n_tensors, names_match, logs)}


def main(argv=None):
    opts = parse(argv)
    started = utc()
    if opts.verify:
        state = verify(opts.contract)
        print("VERIFY " + json.dumps({"checks": state["checks"], "summary": state["summary"],
                                      "estimates": state["estimates"], "files_sha256": state["files"],
                                      "prereg_sha256": PREREG_SHA}), flush=True)
        return 0
    check_approved(read_json(opts.contract))  # before any lease is touched
    leases = take_leases()  # held by this process until it exits
    state = prepare(opts)
    args = state["args"]
    record = {"wrapper": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
              "mode": "hq1-" + opts.stage, "started_utc": started, "args": vars(args),
              "overrides_vs_m0": state["overrides"], "base": state["base"],
              "training_contract": {"path": os.path.abspath(opts.contract), "sha256": sha256_file(opts.contract)},
              "prereg_sha256": PREREG_SHA, "m0_contract_sha256": M0_CONTRACT_SHA, "data_files_sha256": state["files"],
              "pack_checks": state["checks"], "estimates": state["estimates"],
              "leases": {"paths": list(LOCKS), "held": len(leases)},
              "frozen_code_sha256": {p: sha256_file(p) for p in (TRAINER, CODEX, PREPARE, PACKER, CHECKER)},
              "m0_initial_trainable_parameters_sha256": INIT_SHA, "updates_planned": len(state["groups"]),
              "sequences": len(state["seqs"]),
              "resume": "none: fresh LoRA init (hash gate), fresh AdamW, stateless LR schedule, seed %d" % args.seed}
    original, codex = load_trainers()
    record.update(execute(opts, state, original, codex), ended_utc=utc())
    with open(os.path.join(args.out, "meta.json"), "x") as f:
        f.write(json.dumps(record, indent=1) + "\n")
    print("SUMMARY " + json.dumps({k: record[k] for k in ("mode", "initial_trainable_parameters_sha256", "lora_modules",
                                                          "adapter_tensors", "names_match_probe", "updates", "loss_first",
                                                          "loss_last", "seconds", "gates_pass")}), flush=True)
    return 0 if record["gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
