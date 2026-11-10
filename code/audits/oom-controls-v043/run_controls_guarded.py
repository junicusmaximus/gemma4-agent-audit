"""Reuse frozen official gold/empty verifier with the v043 sandbox memory guard."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

ROOT = Path('PROJECT_ROOT/gemma4')
OUT = ROOT / 'analysis/oom-controls-v043/final48'
sys.path.insert(0, str(ROOT / 'tools'))
from sandbox_memory_guard_v043 import MemoryGuard, qualify_guard_receipt


def main():
    selection = json.loads((OUT / 'selection.json').read_text())
    guard_source = ROOT / 'tools/sandbox_memory_guard_v043.py'
    assert hashlib.sha256(guard_source.read_bytes()).hexdigest() == selection['expected_guard_sha256']
    source = ROOT / 'controls/run_controls.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == selection['source']['controls/run_controls.py']
    spec = importlib.util.spec_from_file_location('original_controls_v043', source)
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    controls.GEMMA = ROOT
    controls.DATA = ROOT / 'data/competition'
    controls.SPLITS_PATH = ROOT / 'splits/v1.json'
    sys.argv = [str(source), '--ids', ','.join(selection['ids']), '--out', str(OUT / 'controls'), '--concurrency', str(selection['concurrency'])]
    guard = MemoryGuard(OUT / 'guard.json')
    guard.install()
    success = False
    error = None
    try:
        controls.main()
        success = True
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        guard.close(completed=success, error=error)
        if success:
            qualify_guard_receipt(OUT / 'guard.json')


if __name__ == '__main__':
    main()
