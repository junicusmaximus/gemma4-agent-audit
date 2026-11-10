#!/usr/bin/env python3
"""LoRA training of the v064-r2 arms A_onpol and T_sub on the dequantized QAT base (boss v064-r2): train_m1q_v062.py
with the data, arm, variant and output names replaced (the v057 data-override pattern). r2 of train_onpolicy_v064.py
(7b6c4b2e; superseded, left unedited): the contract must pin the frozen v064-r2 prereg (0893728c...), its addendum
(b602a5d3...) and the Stage-0 eligible list (78b856a2...), each checked against its file, the r2 converter / builder
code, and data manifests of revision r2. Both arms train the same optimizer update count U.

A_onpol = on-policy rejection SFT on resolved E4B base rollouts; T_sub = a tier-stratified seed-20261103 subset of the
M0q training trajectories matched to the same optimizer update count U; pre-registered in
prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md (its sha is a leaf of the training contract). Data: the arm's own
dir sft/data/onpolicy_v064/<arm>/ written by sft/onpolicy_v064/build_arms_r2.py (trajectories.jsonl, M0.jsonl,
identity_ledger.jsonl, update_groups.json, manifest.json), every file sha and count pinned in the training contract.
Imports by file path, without editing them: build_model / save_adapter / check_gpu from sft/train_lora.py and
fixed_denominator_train from the Codex trainer (variant M0: each update's targets must equal its original denominator).
The Codex verify_pair reads only mask_control_v035, so the arm's ledger digests, shifted-target counts and update groups
are recomputed here with the frozen prepare helpers (digest / shifted / target_identity / make_groups). Recipe = the
sealed M0q run: the hyper-parameters of contracts/mask-control-v0351-APPROVED.json, the dequantized base, M0q's init
hash; the only argument that differs from M0q's is the output dir; one pass of U updates, U identical for both arms.
  --verify --arm A              CPU only: training contract and pins, the sealed M0q reference, the dequantized base
                                files and, once the contract's data leaves are pinned, the arm's files, ledger, update
                                groups and population; takes no lease, uses no GPU
  --smoke --arm A               GPU: the --verify gates, then the first 3 updates into the arm's smoke dir
  --execute --arm A --v064-go   GPU: the --verify gates, then all U updates into the arm's main dir
Both GPU stages check the contract passed with --contract before any lease is touched: the arm's data leaves must be
pinned (not PENDING_DATA or UNMATCHED). --execute refuses unless the contract has status APPROVED and a prereg sha that is
filled in and equal to the frozen file. --smoke leaves a receipt only (never evaluated), so it also runs from a DRAFT;
any other status is refused, and a filled prereg leaf must equal its file. Each stage then runs in its own process: it
takes the evaluation and training leases (flock, non-blocking; refuses while either is held), seeds 20261103, builds a
fresh LoRA whose init hash must equal M0's (9dfc7ccc...) and trains with a fresh AdamW and the stateless LR schedule.
There is no resume, checkpoint, adapter or output option; all four run dirs are pinned and must not exist, so smoke
state can never reach a main run. Run with PYTHONDONTWRITEBYTECODE=1 in the train env and without an external flock on
the lease files."""
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
PACKER = SFT + "/build_sft.py"
CONVERTER = SFT + "/onpolicy_v064/convert_traces_r2.py"
BUILDER = SFT + "/onpolicy_v064/build_arms_r2.py"
CODE = {"trainer": TRAINER, "codex_trainer": CODEX, "prepare": PREPARE, "packer": PACKER, "converter": CONVERTER,
        "build_arms": BUILDER}
M0_CONTRACT = ROOT + "/contracts/mask-control-v0351-APPROVED.json"  # the contract recorded by the sealed M0 / M1 / M0q
M0_CONTRACT_SHA = "02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde"
CONTRACT = ROOT + "/contracts/onpolicy-training-v064r2-DRAFT.json"
BRIEF = "PROJECT_ROOT/briefing/research/gemma-4-2026/"
PREREG = BRIEF + "prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md"
PREREG_SHA = "0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60"
PINNED_FILES = {"addendum": (BRIEF + "prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065.md",
                             "b602a5d366002bc5335fcc7a94ada72cba71535b3a63551f74d28bd9a8f39a56"),
                "eligible": (ROOT + "/migration/onpolicy-v064/eligible.json",
                             "78b856a2382068b86a1f5b3ea0624ef01b5962f4b1c98ecdf38f32edeb082625")}
REVISION = "r2"
SEALED_METAS = {"m0q": SFT + "/runs/A1_M0q_v053/meta.json"}
DATA_ROOT = SFT + "/data/onpolicy_v064"
ARMS = {"A_onpol": {"data": DATA_ROOT + "/A_onpol", "smoke": SFT + "/runs/A1_onpol_v064_smoke",
                    "main": SFT + "/runs/A1_onpol_v064"},
        "T_sub": {"data": DATA_ROOT + "/T_sub", "smoke": SFT + "/runs/A1_tsub_v064_smoke",
                  "main": SFT + "/runs/A1_tsub_v064"}}
DATA_FILES = ("trajectories.jsonl", "M0.jsonl", "identity_ledger.jsonl", "update_groups.json", "manifest.json")
DATA_KEYS = ("trajectories", "packed_sequences", "duplication_m", "sequences", "updates", "shifted_targets",
             "input_tokens")
DATA_STATUS = "DRAFT_DATA_NO_TRAINING"
SMOKE_UPDATES = 3  # about 3 min of compute at the M0q rate
LOCKS = (ROOT + "/migration/recovery-screen-v034/evaluation.lock", ROOT + "/migration/mask-control-v0351/training.lock")
BASE = ROOT + "/models/gemma-4-e4b-it-qat-dequant-bf16"
BASE_MANIFEST_SHA = "20f1c2f9a52eea099c7ab95fca57c150d4e3b1946663136f11d11e0f50849ce5"
INIT_SHA = "9dfc7ccc71fc4631d51a61dad8a889040e76bd9c87a2546fe24ec3b769e20066"  # M0 / M1 / M0q initial trainable parameters
SERVED_BASE = "google/gemma-4-e4b-it-qat-w4a16-ct"
VARIANT = "M0"
PENDING = "PENDING_FREEZE"  # the prereg leaf of the training contract until the prereg is frozen
PENDING_DATA = "PENDING_DATA"  # the data leaves until "PIN v064-train"
LORA_MODULES, ADAPTER_TENSORS = 258, 516
MAX_LEN = 16384


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
    prepare = load_module("prepare_mask_control_v035", PREPARE)
    codex = load_module("train_mask_control_v035", CODEX)
    if codex.original is not original:
        raise RuntimeError("the Codex trainer bound a different train_lora module")
    return original, prepare, codex


def dig(obj, keys):
    for k in keys:
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def check_sealed_run(meta):
    """The sealed M0q run: the init hash, contract, variant and base that both v064 arms reproduce."""
    args = meta.get("args") or {}
    if (meta.get("initial_trainable_parameters_sha256"), meta.get("contract_sha256"), args.get("variant"),
            args.get("base")) != (INIT_SHA, M0_CONTRACT_SHA, VARIANT, BASE):
        raise ValueError("sealed M0q run differs from the pinned init hash, contract, variant or base")


def data_leaf(contract, arm):
    """The arm's pinned data leaf, or None while it is PENDING_DATA (or UNMATCHED)."""
    leaf = dig(contract, ("arms", arm, "data"))
    return leaf if isinstance(leaf, dict) else None


def check_contract(contract, wrapper_sha, code, metas_sha, m0_contract):
    """The training contract pins exactly this wrapper, its constants, M0's hyper-parameters, the frozen and v064 code
    and the sealed M0q run. Each arm's data leaf is PENDING_DATA (T_sub also UNMATCHED), with the same value as its main
    stage's update count, or is pinned with its dir, five file shas and positive counts, its main stage running exactly
    its U updates; pinned arms share one U."""
    want = {("generation",): "v064", ("revision",): REVISION, ("prereg", "path"): PREREG,
            ("prereg", "sha256"): PREREG_SHA, ("variant",): VARIANT, ("wrapper", "sha256"): wrapper_sha,
            ("reference_contract", "sha256"): M0_CONTRACT_SHA, ("base", "dir"): BASE,
            ("base", "manifest_sha256"): BASE_MANIFEST_SHA, ("initial_trainable_parameters_sha256",): INIT_SHA,
            ("hyperparameters",): m0_contract["hyperparameters"], ("leases", "paths"): list(LOCKS)}
    for name, sha in code.items():
        want[("code_sha256", name)] = sha
    for name, (path, sha) in PINNED_FILES.items():
        want[("pinned_files", name)] = {"path": path, "sha256": sha}
    for name, sha in metas_sha.items():
        want[("sealed_runs", name, "sha256")] = sha
    bad, updates = [], {}
    for arm, paths in sorted(ARMS.items()):
        want[("arms", arm, "stages", "smoke", "out")] = paths["smoke"]
        want[("arms", arm, "stages", "smoke", "updates")] = SMOKE_UPDATES
        want[("arms", arm, "stages", "main", "out")] = paths["main"]
        leaf = dig(contract, ("arms", arm, "data"))
        if isinstance(leaf, dict):
            want[("arms", arm, "data", "dir")] = paths["data"]
            want[("arms", arm, "stages", "main", "updates")] = leaf.get("updates")
            if sorted(leaf.get("expected_sha256") or {}) != sorted(DATA_FILES) or any(
                    type(leaf.get(k)) is not int or leaf[k] <= 0 for k in DATA_KEYS):
                bad.append("arms.%s.data (five file shas and positive counts)" % arm)
            updates[arm] = leaf.get("updates")
        elif leaf not in ((PENDING_DATA, "UNMATCHED") if arm == "T_sub" else (PENDING_DATA,)):
            bad.append("arms.%s.data" % arm)
        else:
            want[("arms", arm, "stages", "main", "updates")] = leaf
    bad += [".".join(keys) for keys, value in want.items() if dig(contract, keys) != value]
    if len(set(updates.values())) > 1:
        bad.append("arms: unequal update counts %s" % json.dumps(updates, sort_keys=True))
    if bad:
        raise ValueError("training contract differs from this wrapper: " + ", ".join(bad))


def check_pinned_files(contract):
    """The addendum and the Stage-0 eligible list named by the contract equal their files."""
    bad = [name for name, leaf in sorted((contract.get("pinned_files") or {}).items())
           if not isinstance(leaf, dict) or sha256_file(leaf.get("path") or "/nonexistent") != leaf.get("sha256")]
    if bad or sorted(contract.get("pinned_files") or {}) != sorted(PINNED_FILES):
        raise ValueError("pinned files missing or changed: " + ", ".join(bad or ["names"]))
    return contract["pinned_files"]


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


def check_gpu_contract(contract, stage, arm):
    """What a GPU stage needs from the contract before any lease is touched: the arm's data leaves pinned; main: status
    APPROVED and a frozen prereg; smoke (a receipt only): status DRAFT or APPROVED, and a prereg leaf that is pending or
    equal to its file."""
    if data_leaf(contract, arm) is None:
        raise SystemExit("REFUSED: the %s data leaf is %r, not pinned; nothing trained"
                         % (arm, dig(contract, ("arms", arm, "data"))))
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


def ledger_entry_ok(e, key, ids, lab, prepare):
    """A ledger entry names this sequence's position and carries its digests and shifted-target counts (M1 = M0)."""
    labels, targets, identity = prepare.digest(lab), prepare.shifted(lab), prepare.target_identity(lab)
    return ((e["sequence_index"], e["row"], e["traj_id"], e["copy"], e["chunk_index"]) == key
            and e["input_sha256"] == prepare.digest(ids) and e["m0_labels_sha256"] == e["m1_labels_sha256"] == labels
            and e["m0_shifted_targets"] == e["m1_shifted_targets"] == targets and e["changed_original_positions"] == []
            and e["m0_target_identity_sha256"] == e["m1_target_identity_sha256"] == identity)


def load_data(folder, leaf, prepare):
    """The arm's files as pinned, its manifest, and the Codex trainer's tuples (traj_id, int32 ids, int32 labels) of
    M0.jsonl in file order, each sequence checked against its ledger entry with the frozen prepare helpers; the update
    groups must equal prepare.make_groups(ledger)."""
    import numpy as np
    files = {n: sha256_file(os.path.join(folder, n)) for n in DATA_FILES}
    if files != leaf["expected_sha256"]:
        raise ValueError("data files differ from the training contract")
    manifest = read_json(os.path.join(folder, "manifest.json"))
    if (manifest.get("status"), manifest.get("revision"), manifest.get("variant"), manifest.get("files_sha256")) != (
            DATA_STATUS, REVISION, VARIANT, {n: s for n, s in files.items() if n != "manifest.json"}):
        raise ValueError("data manifest differs from its files")
    if any(manifest["counts"].get(k) != leaf[k] for k in DATA_KEYS):
        raise ValueError("data manifest counts differ from the training contract")
    with open(os.path.join(folder, "identity_ledger.jsonl")) as f:
        ledger = [json.loads(line) for line in f]
    seqs = []
    with open(os.path.join(folder, "M0.jsonl")) as f:
        for row_no, line in enumerate(f):
            r = json.loads(line)
            for ci, c in enumerate(r["chunks"]):
                ids, lab = c["input_ids"], c["labels"]
                if not len(ids) == len(lab) <= MAX_LEN or any(y != -100 and y != x for x, y in zip(ids, lab)):
                    raise ValueError("length or id / label alignment gate failed")
                if len(seqs) >= len(ledger) or not ledger_entry_ok(
                        ledger[len(seqs)], (len(seqs), row_no, r["traj_id"], r["copy"], ci), ids, lab, prepare):
                    raise ValueError("ledger identity mismatch at sequence %d" % len(seqs))
                seqs.append((r["traj_id"], np.asarray(ids, dtype=np.int32), np.asarray(lab, dtype=np.int32)))
    groups = read_json(os.path.join(folder, "update_groups.json"))
    if len(seqs) != len(ledger) or groups != prepare.make_groups(ledger):
        raise ValueError("update-group identity mismatch")
    return seqs, groups, manifest


def check_population(seqs, groups, leaf):
    """What is trained is what was checked: each update carries exactly its original denominator of shifted targets
    (the count fixed_denominator_train compares for M0), every sequence is in exactly one update, and the totals equal
    the contract's data leaf."""
    t = [shifted_targets(s[2]) for s in seqs]
    used = []
    for g in groups:
        idx = g["sequence_indices"]
        if sum(t[i] for i in idx) != g["original_denominator"] or g["m1_remaining_targets"] != g["original_denominator"]:
            raise ValueError("update %s: targets differ from the frozen denominators" % g["update"])
        used += idx
    if sorted(used) != list(range(len(seqs))):
        raise ValueError("sequences are not used exactly once")
    got = {"sequences": len(seqs), "updates": len(groups), "shifted_targets": sum(t),
           "input_tokens": sum(len(s[1]) for s in seqs)}
    if got != {k: leaf[k] for k in got}:
        raise ValueError("population differs from the training contract: " + json.dumps(got))
    return dict(got, max_len=max(len(s[1]) for s in seqs), last_update_sequences=len(groups[-1]["sequence_indices"]))


def estimates(seqs, groups):
    """Training time at the M0q rate (its tokens / seconds): the smoke's first updates and the full run."""
    meta = read_json(SEALED_METAS["m0q"])
    rate = meta["tokens"] / meta["seconds"]
    tokens = [sum(len(seqs[x][1]) for x in g["sequence_indices"]) for g in groups]
    smoke, total = sum(tokens[:SMOKE_UPDATES]), sum(tokens)
    return {"m0q_tokens_per_s": round(rate, 1), "smoke_updates": SMOKE_UPDATES, "smoke_tokens": smoke,
            "smoke_minutes_at_m0q_rate": round(smoke / rate / 60, 1), "main_seconds_at_m0q_rate": round(total / rate),
            "main_hours_at_m0q_rate": round(total / rate / 3600, 2), "max_update_tokens": max(tokens)}


def verify(contract_path, arm, require_frozen=False):
    """--verify: training contract and pins, the sealed M0q reference, the dequantized base and, once pinned, the arm's
    data files, ledger, update groups and population."""
    contract = read_json(contract_path)
    prereg = check_prereg(contract, require_frozen)
    if sha256_file(M0_CONTRACT) != M0_CONTRACT_SHA:
        raise ValueError("approved mask-control contract changed: " + M0_CONTRACT)
    m0_contract = read_json(M0_CONTRACT)
    metas = {name: read_json(path) for name, path in SEALED_METAS.items()}
    check_sealed_run(metas["m0q"])
    code = {name: sha256_file(path) for name, path in CODE.items()}
    check_contract(contract, sha256_file(os.path.abspath(__file__)), code,
                   {name: sha256_file(path) for name, path in SEALED_METAS.items()}, m0_contract)
    pinned = check_pinned_files(contract)
    base = verify_base()
    original, prepare, codex = load_trainers()
    state = {"contract": contract, "prereg": prereg, "pinned_files": pinned, "m0_contract": m0_contract,
             "metas": metas, "code": code,
             "base": base, "original": original, "codex": codex, "data": dig(contract, ("arms", arm, "data")),
             "files": None, "seqs": None, "groups": None, "population": None, "estimates": None}
    leaf = data_leaf(contract, arm)
    if leaf is None:
        return state
    seqs, groups, _ = load_data(ARMS[arm]["data"], leaf, prepare)
    return dict(state, files=leaf["expected_sha256"], seqs=seqs, groups=groups,
                population=check_population(seqs, groups, leaf), estimates=estimates(seqs, groups))


def build_args(hyperparameters, out):
    """The Codex main's argument namespace for variant M0 (train_mask_control_v035.py:130-132), on the dequantized base."""
    return argparse.Namespace(**hyperparameters, variant=VARIANT, base=BASE, out=out, device="cuda",
                              served_base_name=SERVED_BASE)


def check_overrides(args, ref_args, allowed):
    """Differences from a sealed run's args, which must stay inside the declared overrides."""
    diff = sorted(k for k in set(vars(args)) | set(ref_args) if vars(args).get(k) != ref_args.get(k))
    extra = sorted(set(diff) - set(allowed))
    if extra:
        raise ValueError("arguments differ from the sealed run beyond the declared overrides: %s" % extra)
    return diff


def fresh_out(stage, arm, arms=ARMS):
    """The stage's pinned run dir: apart from the other three v064 run dirs and not existing (no resume, no reuse)."""
    out = arms[arm][stage]
    for other in [p[s] for a, p in arms.items() for s in ("smoke", "main") if (a, s) != (arm, stage)]:
        if out == other or out.startswith(other + "/") or other.startswith(out + "/"):
            raise ValueError("v064 run dirs overlap")
    if os.path.lexists(out):
        raise FileExistsError("fresh output required: " + out)
    return out


def stage_groups(stage, groups):
    """smoke: the first SMOKE_UPDATES updates of the arm's schedule; main: all of them."""
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
    mode.add_argument("--smoke", action="store_true", help="GPU: the first %d updates into the arm's smoke dir; "
                      "contract status DRAFT or APPROVED" % SMOKE_UPDATES)
    mode.add_argument("--execute", action="store_true", help="GPU: all U updates into the arm's main dir")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS), help="the v064 arm")
    ap.add_argument("--v064-go", action="store_true", help="required with --execute (pre-registered go)")
    ap.add_argument("--contract", default=CONTRACT, help="training contract")
    opts = ap.parse_args(argv)
    if opts.v064_go and not opts.execute:
        ap.error("--v064-go belongs to --execute")
    if opts.execute and not opts.v064_go:
        ap.error("--execute needs --v064-go (separate pre-registered go)")
    opts.stage = "smoke" if opts.smoke else ("main" if opts.execute else None)
    return opts


def prepare(opts):
    """Every CPU gate of a GPU stage: a fresh pinned run dir, the --verify gates (a frozen prereg is required for the main
    run only) and the declared override against the sealed M0q run (the output dir)."""
    out = fresh_out(opts.stage, opts.arm)
    state = verify(opts.contract, opts.arm, require_frozen=opts.stage == "main")
    if state["seqs"] is None:
        raise SystemExit("REFUSED: the %s data is not pinned; nothing trained" % opts.arm)
    args = build_args(state["m0_contract"]["hyperparameters"], out)
    overrides = {"vs_m0q": check_overrides(args, state["metas"]["m0q"]["args"], {"out"})}
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
        state = verify(opts.contract, opts.arm)
        print("VERIFY " + json.dumps({"arm": opts.arm, "contract_status": state["contract"].get("status"),
                                      "prereg": state["prereg"], "pinned_files": state["pinned_files"],
                                      "data": state["data"] if state["seqs"] is None else ARMS[opts.arm]["data"],
                                      "data_files_sha256": state["files"], "population": state["population"],
                                      "estimates": state["estimates"], "frozen_code_sha256": state["code"],
                                      "base": state["base"]}), flush=True)
        return 0
    check_gpu_contract(read_json(opts.contract), opts.stage, opts.arm)  # before any lease is touched
    leases = take_leases()  # held by this process until it exits
    state = prepare(opts)
    args = state["args"]
    record = {"wrapper": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
              "mode": "v064r2-%s-%s" % (opts.arm, opts.stage), "arm": opts.arm, "started_utc": started,
              "args": vars(args), "overrides": state["overrides"], "base": state["base"],
              "training_contract": {"path": os.path.abspath(opts.contract), "sha256": sha256_file(opts.contract),
                                    "status": state["contract"].get("status")},
              "prereg": state["prereg"], "pinned_files": state["pinned_files"], "m0_contract_sha256": M0_CONTRACT_SHA,
              "data_dir": ARMS[opts.arm]["data"],
              "data_files_sha256": state["files"], "population": state["population"], "estimates": state["estimates"],
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
