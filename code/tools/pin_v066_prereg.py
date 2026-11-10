#!/usr/bin/env python3
"""Pins the frozen defining pre-registration into the v066 (A0mc) runtime contract. Run only on Boss's literal PIN v066.

The pinned contract is the staged draft with its pending fields (sha, status line and Comparator E0 text of prereg v064-r2) filled from
the FROZEN file, written BESIDE the draft (contracts/matched-context-v066-a0mc-runtime-PINNED.json plus its .sha256 sidecar); the draft
is never touched. The tool refuses unless
  - the pre-registration carries one FROZEN status line and one Comparator E0 section,
  - the draft is the verified one (its sha equals --draft-sha and its sidecar),
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


def prereg_frozen(runner):
    """The first gate: the defining pre-registration is FROZEN and has its Comparator E0 section."""
    try:
        return runner.prereg_binding()
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        refuse('the defining pre-registration does not qualify: ' + type(error).__name__ + ': ' + str(error))


def changed_leaves(before, after, runner):
    a, b = dict(runner.leaf_paths(before)), dict(runner.leaf_paths(after))
    if set(a) != set(b):
        refuse('the re-derived contract has a different shape than the draft: ' + str(sorted(set(a) ^ set(b))[:5]))
    return {path for path in a if a[path] != b[path]}


def run(runner, draft_sha, dry_run=False):
    pinned, draft_path = runner.RUNTIME, runner.RUNTIME_DRAFT
    binding = prereg_frozen(runner)
    if pinned.exists() or Path(str(pinned) + '.sha256').exists():
        refuse('a pinned contract already exists: ' + str(pinned))
    if runner.sha(draft_path) != draft_sha or Path(str(draft_path) + '.sha256').read_text().split()[0] != draft_sha:
        refuse('the staged draft is not the verified one (sha or sidecar differs from --draft-sha)')
    draft = json.loads(draft_path.read_text())
    if runner.sha(runner.RUNNER_FILE) != draft['runner']['sha256']:
        refuse('the runner differs from the one pinned in the draft')
    try:
        final = runner.contract_doc(runner.readout(), binding)
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        refuse('the contract does not re-derive from the current files: ' + type(error).__name__ + ': ' + str(error))
    pending = set(runner.pending_paths(draft))
    changed = changed_leaves(draft, final, runner)
    if changed != pending or runner.pending_paths(final):
        refuse('the pinned contract must differ from the draft exactly at the pending fields; unexpected: ' +
               str(sorted(changed ^ pending)[:8]))
    text = runner.contract_text(final)
    digest = hashlib.sha256(text.encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='.pin-v066-', dir=str(pinned.parent)) as folder:
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
                      draft_sha256=draft_sha, pending_fields_filled=len(pending),
                      preregistration={key: binding[key] for key in ('path', 'sha256', 'status')}, next='python -B tools/analyze_matched_context_v066.py --prepare --runtime-sha ' + digest)
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
    import run_matched_context_v066 as runner
    runner.RUNNER_FILE = Path(runner.__file__).resolve()
    print(json.dumps(run(runner, args.draft_sha, args.dry_run), indent=1))


if __name__ == '__main__':
    main()
