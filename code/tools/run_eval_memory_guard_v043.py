#!/usr/bin/env python
"""Run the unchanged local evaluator with the v043 sandbox memory guard."""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import sys
from sandbox_memory_guard_v043 import MemoryGuard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard-receipt', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    expected = Path(__file__).resolve().parents[1] / 'run_eval.py'
    if not command or Path(command[0]).resolve() != expected:
        parser.error('Guard wrapper must invoke the canonical run_eval.py')
    # Execute the canonical environment setup before importing sandbox dependencies.
    sys.path.insert(0, str(expected.parent))
    driver = runpy.run_path(str(expected), run_name='gemma4_guarded_eval_v043')
    guard = MemoryGuard(args.guard_receipt)
    guard.install()
    completed, error = False, None
    try:
        returncode = driver['main'](command[1:])
        if returncode not in (None, 0):
            raise SystemExit(returncode)
        completed = True
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        payload = guard.close(completed=completed, error=error)
        if completed and not payload['completed']:
            raise RuntimeError('Memory guard cleanup failed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
