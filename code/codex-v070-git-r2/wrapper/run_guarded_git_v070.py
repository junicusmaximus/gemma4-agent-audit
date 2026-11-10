#!/usr/bin/env python
"""Explicit v070 Git preparation before the unchanged v043 guarded evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path("PROJECT_ROOT/gemma4")
GUARD = ROOT / "tools/run_eval_memory_guard_v043.py"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    expected = ROOT / "run_eval.py"
    if not command or Path(command[0]).resolve() != expected:
        parser.error("Only the canonical run_eval.py may be executed")
    # Canonical offline/environment initialization precedes every harness import.
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(expected), run_name="v070_git_environment_preload")
    import git_baseline_v070 as repair
    repair.install()
    sys.path.insert(0, str(GUARD.parent))
    previous = sys.argv
    completed, error = False, None
    try:
        sys.argv = [str(GUARD), "--guard-receipt", str(args.guard_receipt), "--", *command]
        try:
            runpy.run_path(str(GUARD), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.argv = previous
        repair.write_receipt(str(args.guard_receipt) + ".git-setup.json",
                             completed=completed, error=error, entry_wrapper=Path(__file__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
