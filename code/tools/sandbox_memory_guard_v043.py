#!/usr/bin/env python
"""Aggregate cgroup-v2 sandbox memory guard for local evaluation v043.

Requires a delegated service with DelegateSubgroup=supervisor and OOMPolicy=continue.
This adds resource accounting, not security isolation from hostile same-user code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

MEMORY_MAX = 4 * 1024**3
MEMORY_SWAP_MAX = 0


def counters(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}


def service_root() -> Path:
    relative = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    current = Path('/sys/fs/cgroup') / relative.lstrip('/')
    if current.name != 'supervisor' or not current.parent.name.endswith('.service'):
        raise RuntimeError('Memory guard requires service DelegateSubgroup=supervisor')
    root = current.parent
    if 'memory' not in (root / 'cgroup.controllers').read_text().split():
        raise RuntimeError('Memory controller not delegated')
    return root


def drain(group: Path) -> None:
    (group / 'cgroup.kill').write_text('1')
    deadline = time.monotonic() + 5
    while counters(group / 'cgroup.events')['populated']:
        if time.monotonic() > deadline:
            raise RuntimeError(f'Owned command cgroup failed to drain: {group.name}')
        time.sleep(.01)


def join_and_exec(group: str, ready: str, argv: list[str]) -> None:
    target = Path(group)
    allowed = service_root() / 'sandbox-memory-v043'
    if not target.is_relative_to(allowed) or len(target.relative_to(allowed).parts) != 2:
        raise RuntimeError('Launcher cgroup is outside owned command subtree')
    (target / 'cgroup.procs').write_text(str(os.getpid()))
    Path(ready).write_text('joined')
    os.execvpe(argv[0], argv, os.environ)


class MemoryGuard:
    def __init__(self, receipt: Path, *, memory_max: int = MEMORY_MAX):
        self.service = service_root()
        self.root = self.service / 'sandbox-memory-v043'
        self.receipt = receipt
        self.memory_max = memory_max
        self.lock = threading.RLock()
        self.sandboxes: dict[str, Path] = {}
        self.closing_sandboxes: set[str] = set()
        self.records: list[dict] = []
        self.cancelled_before_launch: list[dict] = []
        self.closed_sandboxes: list[dict] = []
        self.failures: list[str] = []
        self.installed = False
        if receipt.exists() or receipt.with_suffix('.commands.jsonl').exists():
            raise RuntimeError('Guard evidence destination already exists')
        if (self.service / 'memory.max').read_text().strip() == 'max':
            raise RuntimeError('Guard requires finite outer MemoryMax')
        (self.service / 'cgroup.subtree_control').write_text('+memory')
        if any(int((group / 'memory.oom.group').read_text()) for group in (self.service, self.service / 'supervisor')):
            raise RuntimeError('Outer service and evaluator must not use group OOM kill')
        if int((self.service / 'memory.swap.max').read_text()) != 0:
            raise RuntimeError('Guard requires outer MemorySwapMax=0')
        self.root.mkdir()
        (self.root / 'cgroup.subtree_control').write_text('+memory')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        self.journal = receipt.with_suffix('.commands.jsonl').open('x', encoding='utf-8')
        self.service_events_before = counters(self.service / 'memory.events.local')

    def sandbox(self, cwd: Path) -> Path:
        key = hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()
        with self.lock:
            if key in self.closing_sandboxes:
                raise RuntimeError('Command attempted after sandbox stop began')
            if key not in self.sandboxes:
                group = self.root / key
                group.mkdir()
                (group / 'memory.max').write_text(str(self.memory_max))
                (group / 'memory.swap.max').write_text('0')
                (group / 'cgroup.subtree_control').write_text('+memory')
                self.sandboxes[key] = group
            return self.sandboxes[key]

    def execute(self, command, cwd, env, timeout=300, *, shell=None, max_output_chars=10 * 1024 * 1024):
        try:
            return self._execute(command, cwd, env, timeout, shell=shell, max_output_chars=max_output_chars)
        except BaseException as exc:
            with self.lock:
                self.failures.append(f'Guard execution: {type(exc).__name__}: {exc}')
            raise

    def _execute(self, command, cwd, env, timeout=300, *, shell=None, max_output_chars=10 * 1024 * 1024):
        if not isinstance(command, list) or (shell is not False):
            raise RuntimeError('Unexpected sandbox execution signature; refusing unguarded command')
        with self.lock:
            key = hashlib.sha256(str(Path(cwd).resolve()).encode()).hexdigest()
            if key in self.closing_sandboxes:
                self.cancelled_before_launch.append({'sandbox': key, 'reason': 'sandbox-stopped-before-launch'})
                return self.module.ExecutionResult(status='error', stdout='', stderr='Sandbox stopped before command launch', exit_code=-1, duration_seconds=0)
            group = self.sandbox(Path(cwd)) / uuid.uuid4().hex
            group.mkdir()
            (group / 'memory.oom.group').write_text('1')
        ready = self.receipt.parent / f'.join-{group.name}'
        started = time.monotonic()
        result = None
        error = None
        try:
            result = self.original_execute(
                [sys.executable, '-I', str(Path(__file__).resolve()), '--join', str(group), str(ready), '--', *command],
                cwd, env, timeout, shell=False, max_output_chars=max_output_chars,
            )
            if not ready.exists() or ready.read_text() != 'joined':
                raise RuntimeError('Launcher did not confirm cgroup join')
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            try:
                drain(group)
                ready.unlink(missing_ok=True)
                record = {
                    'sandbox': group.parent.name, 'command_id': group.name,
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                    'status': result.status if result is not None else None,
                    'exit_code': result.exit_code if result is not None else None,
                    'error': error, 'memory_peak': int((group / 'memory.peak').read_text()),
                    'memory_events': counters(group / 'memory.events'),
                    'cgroup_empty': counters(group / 'cgroup.events')['populated'] == 0,
                }
                group.rmdir()
                with self.lock:
                    self.records.append(record)
                    self.journal.write(json.dumps(record, sort_keys=True) + '\n')
                    self.journal.flush()
            except BaseException as exc:
                with self.lock:
                    self.failures.append(f'Command cleanup: {type(exc).__name__}: {exc}')
                raise

    def release(self, cwd: Path) -> None:
        key = hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()
        with self.lock:
            self.closing_sandboxes.add(key)
            group = self.sandboxes.get(key)
            if group is None:
                return
        # Cancellation of asyncio.to_thread does not stop its command thread.
        # Kill its owned subtree, then let execute's finally collect the receipt.
        deadline = time.monotonic() + 6
        while True:
            children = [p for p in group.iterdir() if p.is_dir()]
            if not children:
                break
            for child in children:
                # Do not kill a launcher between cgroup join and its confirmation.
                if not (self.receipt.parent / f'.join-{child.name}').exists():
                    continue
                try:
                    drain(child)
                except FileNotFoundError:
                    pass  # The execute thread already collected this child.
            if time.monotonic() > deadline:
                raise RuntimeError('Sandbox stop could not join active command cleanup')
            time.sleep(.01)
        with self.lock:
            if counters(group / 'cgroup.events')['populated'] != 0:
                raise RuntimeError('Sandbox stop has surviving descendants')
            self.closed_sandboxes.append({
                'sandbox': key, 'memory_peak': int((group / 'memory.peak').read_text()),
                'memory_events': counters(group / 'memory.events'),
                'memory_max': int((group / 'memory.max').read_text()),
                'memory_swap_max': int((group / 'memory.swap.max').read_text()),
            })
            group.rmdir()
            del self.sandboxes[key]

    def install(self) -> None:
        import swegemma.sandbox.subprocess as module
        if self.installed:
            raise RuntimeError('Guard already installed')
        self.module = module
        self.original_execute = module.execute_subprocess_command
        self.original_stop = module.SubprocessManager.stop
        guard = self

        def stop(manager, sandbox_id):
            entry = manager.sandboxes.get(sandbox_id)
            if entry is not None:
                key = hashlib.sha256(str(Path(entry['workspace']).resolve()).encode()).hexdigest()
                with guard.lock:
                    guard.closing_sandboxes.add(key)
            try:
                if entry is not None:
                    guard.release(Path(entry['workspace']))
            except BaseException as exc:
                guard.failures.append(f'Sandbox stop: {type(exc).__name__}: {exc}')
                raise
            finally:
                # Keep the workspace valid until all admitted commands have joined
                # and been collected. Preserve the original manager's disk cleanup.
                guard.original_stop(manager, sandbox_id)

        self.stop_wrapper = stop
        module.execute_subprocess_command = self.execute
        module.SubprocessManager.stop = stop
        self.installed = True

    def close(self, *, completed: bool, error: str | None = None) -> dict:
        try:
            for key in list(self.sandboxes):
                group = self.sandboxes[key]
                if any(p.is_dir() for p in group.iterdir()):
                    raise RuntimeError('Guard close has uncollected command cgroup')
                # Recover cwd-independent shutdown for any sandbox not explicitly stopped.
                if counters(group / 'cgroup.events')['populated']:
                    raise RuntimeError('Guard close has live sandbox processes')
                self.closed_sandboxes.append({
                    'sandbox': key, 'memory_peak': int((group / 'memory.peak').read_text()),
                    'memory_events': counters(group / 'memory.events'),
                    'memory_max': int((group / 'memory.max').read_text()),
                    'memory_swap_max': int((group / 'memory.swap.max').read_text()),
                })
                group.rmdir()
                del self.sandboxes[key]
            self.root.rmdir()
        except BaseException as exc:
            self.failures.append(f'Guard close: {type(exc).__name__}: {exc}')
        finally:
            if self.installed:
                self.module.execute_subprocess_command = self.original_execute
                self.module.SubprocessManager.stop = self.original_stop
            self.journal.close()
        after = counters(self.service / 'memory.events.local')
        payload = {
            'schema': 'sandbox-memory-guard-v043',
            'completed': bool(completed and not self.failures), 'error': error,
            'failures': self.failures,
            'config': {'memory_max': self.memory_max, 'memory_swap_max': 0},
            'command_count': len(self.records),
            'cancelled_before_launch_count': len(self.cancelled_before_launch),
            'cancelled_before_launch': self.cancelled_before_launch,
            'oom_command_count': sum(r['memory_events'].get('oom_kill', 0) > 0 for r in self.records),
            'commands': self.records, 'sandboxes': self.closed_sandboxes,
            'all_command_cgroups_empty': all(r['cgroup_empty'] for r in self.records),
            'all_sandbox_cgroups_removed': not self.sandboxes and not self.root.exists(),
            'service_memory_events_local_before': self.service_events_before,
            'service_memory_events_local_after': after,
            'supervisor_memory_events_local': counters(self.service / 'supervisor' / 'memory.events.local'),
            'service_memory_current': int((self.service / 'memory.current').read_text()),
            'service_memory_peak': int((self.service / 'memory.peak').read_text()),
            'service_memory_max': (self.service / 'memory.max').read_text().strip(),
            'service_memory_swap_max': int((self.service / 'memory.swap.max').read_text()),
        }
        with self.receipt.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return payload


def qualify_guard_receipt(path: Path, *, memory_max: int = MEMORY_MAX, memory_swap_max: int = 0) -> dict:
    data = json.loads(Path(path).read_text())
    journal = Path(path).with_suffix('.commands.jsonl')
    if [json.loads(line) for line in journal.read_text().splitlines()] != data['commands']:
        raise RuntimeError('Command journal does not match final receipt')
    if data['schema'] != 'sandbox-memory-guard-v043' or data['config'] != {
        'memory_max': memory_max, 'memory_swap_max': memory_swap_max,
    }:
        raise RuntimeError('Memory guard configuration mismatch')
    if not data['completed'] or data['error'] is not None or data['failures']:
        raise RuntimeError('Memory guard execution incomplete')
    if data['cancelled_before_launch_count'] != len(data['cancelled_before_launch']) or any(r['reason'] != 'sandbox-stopped-before-launch' for r in data['cancelled_before_launch']):
        raise RuntimeError('Pre-launch cancellation accounting invalid')
    if not data['command_count'] or data['command_count'] != len(data['commands']):
        raise RuntimeError('Memory guard command accounting incomplete')
    if any(not r['cgroup_empty'] or r['error'] is not None for r in data['commands']):
        raise RuntimeError('Command execution or cleanup incomplete')
    if not data['sandboxes'] or {r['sandbox'] for r in data['commands']} != {r['sandbox'] for r in data['sandboxes']}:
        raise RuntimeError('Sandbox lifetime accounting incomplete')
    if not data['all_command_cgroups_empty'] or not data['all_sandbox_cgroups_removed']:
        raise RuntimeError('Memory guard cleanup incomplete')
    for record in data['sandboxes']:
        if record['memory_max'] != memory_max or record['memory_swap_max'] != memory_swap_max:
            raise RuntimeError('Sandbox memory restriction mismatch')
    before, after = data['service_memory_events_local_before'], data['service_memory_events_local_after']
    if any(after.get(k, 0) != before.get(k, 0) for k in ('oom', 'oom_kill', 'oom_group_kill')):
        raise RuntimeError('Evaluation service itself encountered OOM')
    if any(data['supervisor_memory_events_local'].get(k, 0) for k in ('oom', 'oom_kill', 'oom_group_kill')):
        raise RuntimeError('Evaluator supervisor encountered OOM')
    if data['service_memory_swap_max'] != 0:
        raise RuntimeError('Evaluation service swap restriction mismatch')
    return data


if __name__ == '__main__':
    if len(sys.argv) < 6 or sys.argv[1] != '--join' or sys.argv[4] != '--':
        raise SystemExit('Internal launcher: --join GROUP READY -- COMMAND [ARGS]')
    join_and_exec(sys.argv[2], sys.argv[3], sys.argv[5:])
