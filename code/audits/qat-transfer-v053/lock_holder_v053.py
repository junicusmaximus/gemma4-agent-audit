#!/usr/bin/env python3
"""GPU lease holder for the QAT-transfer check v053 (copy of analysis/first-step-probe-boss-v048/lock_holder_v22.py with
this directory's receipt, release flag and monitor files): refuses unless nvidia-smi shows <= 100 MiB used and no compute apps,
then holds BOTH leases with fcntl.flock(LOCK_EX | LOCK_NB) on files opened with mode 'a' (never truncated, never
deleted; tools/run_mask_evaluations_v043.py:538-541) until OUT/RELEASE appears or MAX_HOLD_S passes. Writes
lock-receipt.json (pid, start/end UTC) and samples GPU memory every 2 s into gpu_monitor.csv."""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = "PROJECT_ROOT/gemma4"
OUT = ROOT + "/analysis/qat-transfer-v053"
LOCKS = [ROOT + "/migration/recovery-screen-v034/evaluation.lock", ROOT + "/migration/mask-control-v0351/training.lock"]
MAX_HOLD_S = 75 * 60  # safety net only; the stages release earlier via OUT/RELEASE


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def gpu_state():
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   text=True, timeout=30).split()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True, timeout=30).strip()
    return max(int(x) for x in used), [a.strip() for a in apps.splitlines() if a.strip()]


def write(receipt):
    tmp = OUT + "/lock-receipt.json.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(receipt, indent=1) + "\n")
    os.replace(tmp, OUT + "/lock-receipt.json")


def main():
    receipt = {"pid": os.getpid(), "host": os.uname().nodename, "locks": LOCKS, "requested_utc": utc(), "mode": "a",
               "flags": "LOCK_EX|LOCK_NB"}
    used, apps = gpu_state()
    receipt["gpu_before"] = {"memory_used_mib": used, "compute_apps": apps}
    if used > 100 or apps:
        receipt.update(status="refused_gpu_busy", end_utc=utc())
        write(receipt)
        return 2
    missing = [p for p in LOCKS if not os.path.isfile(p)]
    if missing:  # mode 'a' would create the file; never write under migration/
        receipt.update(status="refused_lock_file_missing", missing=missing, end_utc=utc())
        write(receipt)
        return 3
    handles = []
    for path in LOCKS:
        handle = open(path, "a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            receipt.update(status="refused_lock_busy", busy=path, error=str(error), end_utc=utc())
            write(receipt)
            return 4
        handles.append(handle)
    receipt.update(status="held", start_utc=utc(), lock_inodes={p: os.fstat(h.fileno()).st_ino for p, h in zip(LOCKS, handles)})
    write(receipt)
    started = time.monotonic()
    with open(OUT + "/gpu_monitor.csv", "a") as mon:
        while not os.path.exists(OUT + "/RELEASE") and time.monotonic() - started < MAX_HOLD_S:
            try:
                used, apps = gpu_state()
                mon.write("%s,%d,%d\n" % (utc(), used, len(apps)))
            except Exception as error:  # keep holding; record the sampling failure
                mon.write("%s,error,%s\n" % (utc(), type(error).__name__))
            mon.flush()
            time.sleep(2)
    reason = "release_flag" if os.path.exists(OUT + "/RELEASE") else "max_hold_reached"
    for handle in handles:
        handle.close()  # closing our own descriptors releases our flocks; nothing else is touched
    receipt.update(status="released", end_utc=utc(), release_reason=reason,
                   held_seconds=round(time.monotonic() - started, 1))
    write(receipt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
