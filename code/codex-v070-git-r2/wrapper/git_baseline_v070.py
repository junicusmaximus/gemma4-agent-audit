"""Experiment-only checked Git baseline preparation; installed harness files are untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import threading

AMENDMENT_SHA256 = "dff250b2870b185069ec5a86b9b7a2e811ed9fc0b65a728049f93049e8d57a99"
ERROR_PREFIX = "V070_GIT_SETUP_ERROR"
_ENTRIES = []
_LOCK = threading.Lock()
_INVENTORY_CODE = """
import hashlib,json,os,stat
root=os.path.realpath('/workspace')
rows=[]
for base,dirs,files in os.walk(root,followlinks=False):
    dirs[:]=sorted(d for d in dirs if d!='.git')
    for name in sorted(files):
        p=os.path.join(base,name)
        rel=os.path.relpath(p,root)
        st=os.lstat(p)
        if stat.S_ISLNK(st.st_mode):
            value='link:'+os.readlink(p)
        elif stat.S_ISREG(st.st_mode):
            with open(p,'rb') as f:value='file:'+hashlib.sha256(f.read()).hexdigest()
        else:
            raise RuntimeError('unsupported source inode: '+rel)
        rows.append((rel,value))
payload=json.dumps(sorted(rows),ensure_ascii=True,separators=(',',':')).encode()
print('V070_INVENTORY='+json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),'files':len(rows)}))
"""


class GitBaselineError(RuntimeError):
    pass


def _checked(docker, container_id, command):
    result = docker.exec(container_id, command)
    if getattr(result, "timed_out", False) or result.exit_code != 0:
        detail = (result.stderr or result.stdout or "")[-1500:]
        raise GitBaselineError(f"{ERROR_PREFIX}: exit={result.exit_code}: {detail}")
    return result.stdout or ""


def _inventory(docker, container_id):
    text = _checked(docker, container_id, "python3 -c " + shlex.quote(_INVENTORY_CODE))
    lines = [line.removeprefix("V070_INVENTORY=") for line in text.splitlines()
             if line.startswith("V070_INVENTORY=")]
    if len(lines) != 1:
        raise GitBaselineError(f"{ERROR_PREFIX}: source inventory missing")
    return json.loads(lines[0])


def setup_baseline_commit(docker, container_id, commit_msg="baseline"):
    """Preserve source bytes, initialize an owned repository, and verify the baseline."""
    entry = {"baseline_label": str(commit_msg), "completed": False}
    try:
        before = _inventory(docker, container_id)
        entry.update(source_before_sha256=before["sha256"], source_files=before["files"])
        command = (
            "set -eu; cd /workspace; "
            "test ! -L .git; test ! -f .git; "
            "git init --quiet .; "
            'test "$(git rev-parse --show-toplevel)" = "$(pwd -P)"; '
            "git config --local user.name 'Gemma v070 baseline'; "
            "git config --local user.email 'v070-baseline@localhost'; "
            "git add -A; "
            "git -c core.hooksPath=/dev/null commit --allow-empty --quiet -m "
            + shlex.quote(str(commit_msg)) + "; "
            "git tag -f _swegemma_baseline >/dev/null; "
            "git diff --quiet HEAD; "
            'test -z "$(git status --porcelain --untracked-files=all)"; '
            'printf "V070_GIT_ROOT=%s\\n" "$(git rev-parse --show-toplevel)"; '
            'printf "V070_GIT_HEAD=%s\\n" "$(git rev-parse HEAD)"; '
            'printf "V070_GIT_TAG=%s\\n" "$(git rev-parse _swegemma_baseline)"'
        )
        text = _checked(docker, container_id, command)
        meta = dict(line.split("=", 1) for line in text.splitlines()
                    if line.startswith(("V070_GIT_ROOT=", "V070_GIT_HEAD=", "V070_GIT_TAG=")))
        after = _inventory(docker, container_id)
        if before != after:
            raise GitBaselineError(f"{ERROR_PREFIX}: source contents changed during Git preparation")
        head, tag = meta.get("V070_GIT_HEAD"), meta.get("V070_GIT_TAG")
        if not head or head != tag or not meta.get("V070_GIT_ROOT"):
            raise GitBaselineError(f"{ERROR_PREFIX}: baseline identity unavailable")
        entry.update(source_after_sha256=after["sha256"], git_root=meta["V070_GIT_ROOT"],
                     head=head, baseline_tag=tag, clean=True, completed=True)
    except BaseException as exc:
        entry["error"] = str(exc)
        if isinstance(exc, GitBaselineError):
            raise
        raise GitBaselineError(f"{ERROR_PREFIX}: {type(exc).__name__}: {exc}") from exc
    finally:
        with _LOCK:
            _ENTRIES.append(entry)


def install():
    """Bind only the three baseline helpers, preserving all other harness functions."""
    import swegemma.harness.container_setup as setup
    import swegemma.harness.agent_runner as agent
    import swegemma.harness.verification as verification
    for module in (setup, agent, verification):
        module.setup_baseline_commit = setup_baseline_commit


def entries():
    with _LOCK:
        return [dict(entry) for entry in _ENTRIES]


def write_receipt(path, *, completed, error=None, entry_wrapper=None):
    path = Path(path)
    wrapper = Path(entry_wrapper) if entry_wrapper is not None else None
    doc = dict(schema="v070-git-setup-v1", amendment_sha256=AMENDMENT_SHA256,
               completed=bool(completed), error=error, entries=entries())
    if wrapper:
        doc["wrapper_sha256"] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, sort_keys=True, indent=2)
        stream.write("\n")
