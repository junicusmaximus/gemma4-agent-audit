#!/usr/bin/env python
"""Synthetic API integration check; never reads the external panel or model outcomes."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import time
import uuid

ROOT=Path("PROJECT_ROOT/gemma4")
WORK=Path("PROJECT_ROOT/codex/experiments/v070-git-r2")
PYTHON=Path("HOME_DIR/miniconda3/envs/swegemma/bin/python")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    run_id="synthetic-"+uuid.uuid4().hex[:10]
    out=WORK/"controls"/run_id
    out.mkdir(exist_ok=False)
    (out/"snapshots").mkdir()
    rows=[]
    gold="diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n@@ -1,2 +1,2 @@\n def value():\n-    return 0\n+    return 1\n"
    tests="diff --git a/tests/test_module.py b/tests/test_module.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_module.py\n@@ -0,0 +1,4 @@\n+from module import value\n+\n+def test_value():\n+    assert value() == 1\n"
    iid="test_mock_repo_v070-1"
    snapshot=out/"snapshots"/(iid+".tgz")
    with tarfile.open(snapshot,"w:gz") as tar:
        raw=b"def value():\n    return 0\n"
        info=tarfile.TarInfo("module.py");info.size=len(raw)
        tar.addfile(info,io.BytesIO(raw))
    taskfile=out/"tasks.jsonl"
    taskfile.write_text(json.dumps(dict(instance_id=iid,repo="test/mock_repo_v070",base_commit="123456",
        problem_statement="Synthetic fixture: value must return one.",patch=gold,test_patch=tests,
        FAIL_TO_PASS=["tests/test_module.py::test_value"],PASS_TO_PASS=[]))+"\n")
    task=dict(instance_id=iid,taskfile=str(taskfile),snapshot=str(snapshot),snapshot_sha256=sha(snapshot))
    sys.path.insert(0,str(ROOT/"tools"))
    spec=importlib.util.spec_from_file_location("v070_synthetic_service_reference",ROOT/"tools/run_external_e4b_v070.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    module=runner.v060();contract,_=runner.panel_plan(module)
    service_dir=out/"services";service_dir.mkdir()
    locks=[]
    try:
        for p in runner.LOCKS:
            handle=Path(p).open("a+");fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        for kind in ("gold","empty"):
            home=ROOT/"results/extctl_v060/_rebuild/tmp/v070git2-controls"/(run_id+"-"+kind)
            home.mkdir(parents=True,exist_ok=False)
            case=dict(task=task,kind=kind,output=str(out/kind),guard_receipt=str(out/(kind+".guard.json")),
                budget=dict(max_time_minutes=4,max_tool_calls=28,max_turns=80,timeout_seconds=240))
            path=out/(kind+".case.json");path.write_text(json.dumps(case,sort_keys=True,indent=2)+"\n")
            env={**os.environ,**module.child_env(contract,home)}
            log=out/(kind+".log")
            bounded=runner.MemoryBoundedService(time.monotonic()+1800,service_dir)
            try:
                bounded([str(PYTHON),"-B",str(WORK/"controls/control_case_v070.py"),
                    "--case",str(path),"--case-sha256",sha(path)],log,env,ROOT/"data/competition")
                rec=json.loads((out/kind/"case-report.json").read_text())
                rec["memory_guard"]=runner.guard_qualification(case["guard_receipt"])
                rows.append(rec)
                print(json.dumps(dict(kind=kind,passed=rec["passed"],resolved=rec["resolved"],
                                      test_exit_code=rec["test_exit_code"],collected=rec["junit"]["collected"])),flush=True)
                if not rec["passed"]:raise RuntimeError("Synthetic API fixture failed")
            finally:
                end=service_dir/(log.name+".service-end.json")
                if end.exists() and json.loads(end.read_text()).get("cgroup_empty") is True:
                    shutil.rmtree(home)
        report=dict(schema="v070-synthetic-api-integration-v1",status="PASS",rows=rows,
                    completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_pins={str(WORK/x):sha(WORK/x) for x in ["controls/control_case_v070.py",
                    "wrapper/git_baseline_v070.py","wrapper/run_guarded_git_v070.py"]})
        (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        print("SYNTHETIC_REPORT",out/"report.json",sha(out/"report.json"),flush=True)
    finally:
        for handle in reversed(locks):handle.close()


if __name__=="__main__":main()
