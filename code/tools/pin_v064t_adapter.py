#!/usr/bin/env python3
"""Pins the finished T_sub run into the v064t runtime contract. Run only on Boss's literal PIN v064t.

The pinned contract is the staged draft with every pending field filled from the finished run, written BESIDE the draft
(contracts/matched-context-v064t-runtime-PINNED.json plus its .sha256 sidecar); the draft is never touched. The tool refuses unless
  - sft/runs/A1_tsub_v064/meta.json exists with gates_pass true, updates == U (the update count that the approved training contract
    contracts/onpolicy-training-v064r2-APPROVED.json pins for this arm) and M0's init hash 9dfc7ccc (init_hash_equals_m0 true),
  - the draft is the verified one (its sha equals --draft-sha and its sidecar),
  - qualified_arm passes on the finished run (wrapper receipt, contract, data files, adapter layout, update log),
  - the contract re-derived from the current files differs from the draft exactly at the pending fields,
  - the pinned contract passes verify_runtime from a private temporary copy before it is installed,
  - no pinned contract exists yet (exclusive create).
--dry-run does everything except the installation. CPU only: no lock, no server, no GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def refuse(message):
    raise SystemExit('REFUSED: ' + message)


def finished_run_ready(runner):
    """The first gate: meta.json exists with gates_pass true, updates == U of the approved training contract and M0's init hash."""
    meta_path = runner.ADAPTER / 'meta.json'
    if not meta_path.is_file():
        refuse(str(meta_path) + ' does not exist; the training has not finished')
    try:
        updates = runner.approved_training()[2]
    except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
        refuse('the approved training contract does not pin this arm: ' + type(error).__name__ + ': ' + str(error))
    meta = json.loads(meta_path.read_text())
    if meta.get('gates_pass') is not True or meta.get('updates') != updates or type(meta.get('updates')) is not int:
        refuse('meta.json has gates_pass %r and updates %r, need true and %d (U of the approved training contract)'
               % (meta.get('gates_pass'), meta.get('updates'), updates))
    if meta.get('initial_trainable_parameters_sha256') != runner.M0_INIT_SHA or meta.get('init_hash_equals_m0') is not True:
        refuse('meta.json init hash %r (init_hash_equals_m0 %r): need M0 %s and true'
               % (meta.get('initial_trainable_parameters_sha256'), meta.get('init_hash_equals_m0'), runner.M0_INIT_SHA[:8]))
    return meta


def changed_leaves(before, after, runner):
    a, b = dict(runner.leaf_paths(before)), dict(runner.leaf_paths(after))
    if set(a) != set(b):
        refuse('the re-derived contract has a different shape than the draft: ' + str(sorted(set(a) ^ set(b))[:5]))
    return {path for path in a if a[path] != b[path]}


def run(runner, draft_sha, dry_run=False):
    pinned, draft_path = runner.RUNTIME, runner.RUNTIME_DRAFT
    meta = finished_run_ready(runner)
    if pinned.exists() or Path(str(pinned) + '.sha256').exists():
        refuse('a pinned contract already exists: ' + str(pinned))
    if runner.sha(draft_path) != draft_sha or Path(str(draft_path) + '.sha256').read_text().split()[0] != draft_sha:
        refuse('the staged draft is not the verified one (sha or sidecar differs from --draft-sha)')
    draft = json.loads(draft_path.read_text())
    if runner.sha(runner.RUNNER_FILE) != draft['runner']['sha256']:
        refuse('the runner differs from the one pinned in the draft')
    try:
        qualified = runner.qualified_arm()
        final = runner.contract_doc(runner.readout(), qualified)
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        refuse('the finished run does not qualify: ' + type(error).__name__ + ': ' + str(error))
    pending = set(runner.pending_paths(draft))
    changed = changed_leaves(draft, final, runner)
    if changed != pending or runner.pending_paths(final):
        refuse('the pinned contract must differ from the draft exactly at the pending fields; unexpected: ' +
               str(sorted(changed ^ pending)[:8]))
    text = runner.contract_text(final)
    digest = hashlib.sha256(text.encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='.pin-v064t-', dir=str(pinned.parent)) as folder:
        candidate = Path(folder) / pinned.name
        candidate.write_text(text)
        Path(str(candidate) + '.sha256').write_text(digest + '  ' + pinned.name + chr(10))
        original, runner.RUNTIME = runner.RUNTIME, candidate
        try:
            runner.verify_runtime(digest)
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            refuse('the pinned contract does not verify, nothing installed: ' + type(error).__name__ + ': ' + str(error))
        finally:
            runner.RUNTIME = original
        report = dict(status='dry_run_verified' if dry_run else 'pinned', runtime=str(pinned), runtime_sha256=digest,
                      draft_sha256=draft_sha, pending_fields_filled=len(pending), adapter_artifacts=qualified['artifacts'],
                      summary=qualified['summary'], meta_sha256=qualified['artifacts'][str(runner.ADAPTER / 'meta.json')],
                      training_contract_sha256=qualified['training_contract_sha256'], updates=qualified['updates'],
                      meta_mode=meta['mode'], next='python -B tools/analyze_matched_context_v064t.py --prepare --runtime-sha ' + digest)
        if not dry_run:
            os.link(str(candidate) + '.sha256', str(pinned) + '.sha256')
            try:
                os.link(str(candidate), str(pinned))
            except BaseException:
                os.unlink(str(pinned) + '.sha256')
                raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--draft-sha', required=True, help='sha256 of the verified staged draft contract')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_matched_context_v064t as runner
    runner.RUNNER_FILE = Path(runner.__file__).resolve()
    print(json.dumps(run(runner, args.draft_sha, args.dry_run), indent=1))


if __name__ == '__main__':
    main()
