"""Bounded unique-user-service launcher for CPU memory guard controls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('PROJECT_ROOT/gemma4')
OUT = ROOT / 'analysis/oom-controls-v043/final48'
UNIT = 'gemma4-memory-controls-v043-final48-20261105.service'
ENV = dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}')
ENV.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path={ENV["XDG_RUNTIME_DIR"]}/bus')
PROPS = ['Delegate=memory', 'DelegateSubgroup=supervisor', 'OOMPolicy=continue', 'MemoryMax=24G', 'MemorySwapMax=0', 'RuntimeMaxSec=2400', 'TimeoutStopSec=30', 'KillMode=control-group', 'ExitType=cgroup']
KEYS = ['LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainCode', 'ExecMainStatus', 'InvocationID', 'ControlGroup', 'MemoryPeak', 'MemorySwapPeak', 'MemoryMax', 'MemorySwapMax', 'Delegate', 'DelegateSubgroup', 'OOMPolicy']


def show():
    args = ['systemctl', '--user', 'show', UNIT]
    args += ['--property=' + ','.join(KEYS)]
    result = subprocess.run(args, env=ENV, capture_output=True, text=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def main():
    if (OUT / 'service-end.json').exists() or (OUT / 'guard.json').exists():
        raise RuntimeError('Refusing to reuse control output generation')
    prior = show()
    if prior.get('LoadState') != 'not-found':
        raise RuntimeError('Owned unit name already present: ' + str(prior))
    argv = ['systemd-run', '--user', '--wait', '--pipe', '--unit=' + UNIT, '--working-directory=' + str(ROOT)]
    argv += ['--property=' + prop for prop in PROPS]
    argv += ['HOME_DIR/miniconda3/envs/swegemma/bin/python', str(OUT / 'run_controls_guarded.py')]
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT / 'tools/sandbox_memory_guard_v043.py', OUT / 'run_controls_guarded.py', OUT / 'selection.json']}
    start = {'utc': datetime.now(timezone.utc).isoformat(), 'argv': argv, 'source_sha256': sources, 'prior': prior}
    (OUT / 'service-start.json').write_text(json.dumps(start, indent=2) + '\n')
    observed = []
    deadline = time.monotonic() + 2500
    with (OUT / 'service.log').open('x') as log:
        child = subprocess.Popen(argv, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            live = show()
            if live.get('InvocationID') and live.get('ControlGroup'):
                if not observed:
                    observed.append(live)
                elif len(observed) == 1:
                    observed.append(live)
                else:
                    observed[1] = live
                (OUT / 'service-observed.json').write_text(json.dumps(observed, indent=2) + '\n')
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned control service exceeded launcher deadline')
            time.sleep(0.5)
    end = show()
    group = end.get('ControlGroup') or (observed[-1].get('ControlGroup', '') if observed else '')
    group_path = Path('/sys/fs/cgroup') / group.lstrip('/') if group else None
    events = (group_path / 'cgroup.events').read_text() if group_path and (group_path / 'cgroup.events').exists() else 'populated 0\n'
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'returncode': child.returncode, 'service': end, 'observed': observed, 'systemd_run_success': child.returncode == 0 and 'Finished with result: success' in (OUT / 'service.log').read_text(), 'cgroup_path': str(group_path) if group_path else None, 'cgroup_exists': group_path.exists() if group_path else None, 'cgroup_events': events, 'cgroup_empty': bool(group_path) and 'populated 0' in events}
    (OUT / 'service-end.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
