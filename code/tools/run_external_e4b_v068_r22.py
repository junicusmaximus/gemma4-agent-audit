"""External-panel v068, revision r22 (implements the frozen pre-registration 8061b917): the E4B QAT w4a16 base (A0mc) and the M0q, M1q and M1q-s2 adapters on the eligible DEV14 tasks of the v060 external set.

One official harness invocation per task (the per-repository router environment cannot be shared), four invocations in flight, one vLLM session
that serves the base and every LoRA alias under the matched serving form of v055, capped sandboxes as in v062. Per run (arm, seed) the rows are merged
and descriptive; no reading, contrast or sentence lives in this file. r21 adds to r1: non-terminal ledger rows are not eligible (fewer than 4 eligible tasks: no run), the
exclusion rule and the stop rules (canary fault, two excluded rows in one arm, a server restart, 4 h from server-ready) with partial-generation qualification, the M1q-s2
arm included iff its meta.json shows gates_pass true at the pin, and a repeatable pin until an external rollout exists. r22 changes r21 only as the frozen
v068 execution-change record (6a0e5942, section 3) says: the dependency trees are compared without their bytecode caches, the pin bridges that baseline to
the r21 admission hashes, and attempt 2 runs as generation v068a2 on new paths."""
from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import sys
import urllib.request
import uuid
from types import SimpleNamespace

from prepare_v032_probes import BASE, inventory, load_recovery, require, sha, write_json
import run_v032_probes as serving
import run_31b_recovery_screen_v034 as services
import run_mask_control_v0351 as training

ROOT = Path('PROJECT_ROOT/gemma4')
SCHEMA, GENERATION = 'external-e4b-v068-r22-runtime', 'v068a2'
# RUNTIME is the pinned contract the tools read (written by --pin beside the staged draft, never over it).
RUNTIME = ROOT / 'contracts/external-e4b-v068-r22-runtime-PINNED.json'
RUNTIME_DRAFT = ROOT / 'contracts/external-e4b-v068-r22-runtime-DRAFT.json'
LEDGER_SNAPSHOT = ROOT / 'contracts/external-e4b-v068-r21-dev-ledger-PINNED.json'  # the DEV rows of the v060 ledger as the r21 pin step copied them; r22 only reads it
PACKAGE = ROOT / 'migration/external-e4b-v068'
BUILD = ROOT / 'analysis/external-e4b-boss-v068-r22'
TASKS = PACKAGE / 'tasks.json'
EVIDENCE = PACKAGE / 'attempt2/evaluation'
MANIFEST = ROOT / 'subs/v068a2_E4B_manifest'
LOCKS = (ROOT / 'migration/recovery-screen-v034/evaluation.lock', ROOT / 'migration/mask-control-v0351/training.lock')
GUARD = ROOT / 'tools/run_eval_memory_guard_v043.py'
GUARD_SOURCE_SHA = '5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GUARD_WRAPPER_SHA = 'c04928a3406c1175098f1217dd7b955dcec61aec1147537fbd68bf59de8bbbd5'
REPLAY_RECEIPT = ROOT / 'analysis/oom-replay-v043/report.json'
SERVE = ROOT / 'tools/serve_vllm_matched_v055.py'
SERVE_PARENT = ROOT / 'tools/serve_vllm_matched_v051.py'
SERVE_PARENT_SHA = '683c7ab9c53f1b66a248f019e2e8a13b6df1ac12ad74b53df0bb5f6b862c8633'
SERVE_ORIGINAL = ROOT / 'serve_vllm.py'
SERVE_ORIGINAL_SHA = 'c4b513cc5bb87d4bc3741623a4bedac41cfca5b8bdf95c0a4743b8f9371d576a'
TEMPLATE = ROOT / 'analysis/matched-context-boss-v049/chat_template.matched.jinja'
TEMPLATE_SHA = '855bbbbbef10d51177c5d7eca19a3f3d39a11d9a80c142a9f19784029515a437'
# Parent generation: the v062 runner is the template of this one (runner.diff is against it); its pinned contract supplies the pins, budgets and server facts.
RUNNER_PARENT = ROOT / 'tools/run_matched_context_v062.py'
RUNNER_PARENT_SHA = '621cd9363e0523ed6391705c9329d003213c09f34eb8d3ff3ab78047d059da2b'
PARENT_RUNTIME = ROOT / 'contracts/matched-context-v062-runtime-PINNED.json'
PARENT_RUNTIME_SHA = '3717c8d036ae17cf2f4713458b5fe31a36cff95903dadf4e9175eb2efee9aefe'
PARENT_EVIDENCE = ROOT / 'migration/matched-context-v062/evaluation'
COMPARATOR_EVIDENCE = ROOT / 'migration/matched-context-v055/evaluation'
# The external set: the frozen v060 contract, runner and the evaluator-only task files they name.
V060_CONTRACT = ROOT / 'contracts/external-controls-v060-APPROVED-r3.json'
V060_CONTRACT_SHA = 'ef147ef8bdf11c5f98ae50c370651c9576ad4d7f48c61882927516bbae3c6162'
V060_CORE_SHA = 'b65da663154d6b6db31e9032e85abd910d78b970cab56caf115623448cdb9cfc'
V060_RUNNER = ROOT / 'tools/run_external_controls_v060.py'
V060_RUNNER_SHA = '538a6d6a3738833efdde179a3e2fb04ebb8af4764898d87884bde633c26507eb'
V060_LEDGER = ROOT / 'migration/external-controls-v060/ledger.json'
V060_ROWS_DEV = 14
REBUILD = ROOT / 'results/extctl_v060/_rebuild'
BYTE_TRACE = ROOT / 'results/v043_M0_s1/traces/trace_rich_2725.json'
BYTE_CALLS = (1, 3, 7)
BYTE_ALIAS = BASE
# The v066 expectation (request model = the base name) is reused as it is: the render of calls 1, 3, 7 does not depend on the adapter.
BYTE_EXPECTED = ROOT / 'migration/matched-context-v066/byte-check-expected.json'
BYTE_EXPECTED_SHA = 'f6f3bd84004313567206c985a649eb8d7545ac123e80d7f5ff3d3cb85566ea9e'
SERVE_DIFF, RUNNER_DIFF = ROOT / 'migration/matched-context-v055/serve.diff', BUILD / 'runner.diff'
TOKENIZE_URL = 'http://127.0.0.1:8000/tokenize'
FORMAT_TEXT = 'Detected the chat template content format to be '
SEEDS, WORKERS = (1, 2, 3), 4
TOTAL, RESERVE, INVOCATION_SECONDS = 21600, 600, 1200
PENDING_LEDGER, PENDING_TRAINING = 'PENDING_LEDGER', 'PENDING_TRAINING'
PENDING_MARKERS = (PENDING_LEDGER, PENDING_TRAINING)
# The frozen v068 pre-registration (read, never edited) and the rules it fixes for this runner.
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-external-panel-m1q-v068.md'
PREREG_SHA = '8061b917294dc8406f68f40740c832e424c8f167ebba18fab0ea26f52e351f7e'
# Attempt 2 (the frozen v068 execution-change record): the record, the r21 pin and runner it bridges to, the cache audit and the attempt-1 cache paths.
CHANGE_RECORD = ROOT.parent / 'briefing/research/gemma-4-2026/v068-execution-change-record-2026-11-09.md'
CHANGE_RECORD_SHA = '6a0e59426bc076c62d89bb78028e867bc6afdd38d86e8a9d2c14d60742868ae6'
R21_RUNTIME = ROOT / 'contracts/external-e4b-v068-r21-runtime-PINNED.json'
R21_RUNTIME_SHA = '8450ed230af0593fce6d38e41f3b6d0ca5856892cf0ed4e672af1a61e7d31eb8'
R21_RUNNER = ROOT / 'tools/run_external_e4b_v068_r21.py'
R21_RUNNER_SHA = 'e590244237ef8ffe8132fd1e74092a340f978b8f478b2664655770a850a7a727'
LEDGER_SNAPSHOT_SHA = '79c26c72b5b0377f2578253f2e5ef81486084553c5008ae8cbae3cc1564c9fc9'
AUDIT = ROOT.parent / 'briefing/research/gemma-4-2026/artifacts/codex-v068-cache-audit-2026-11-09/audit.json'
AUDIT_SHA = 'b4457509ca9cc485e4cd3929745cff77cbf0b031a399e39e1943aea66f2e4dff'
BRIDGE_CANDIDATES = PACKAGE / 'attempt2-bridge-candidates.txt'
BRIDGE_CANDIDATES_SHA = 'a7bfafc5fe289b5a29170a73be001273f91ffebc3712fc34c97e52f22de05928'
ATTEMPT2_INPUTS = ((CHANGE_RECORD, CHANGE_RECORD_SHA), (R21_RUNTIME, R21_RUNTIME_SHA), (R21_RUNNER, R21_RUNNER_SHA), (AUDIT, AUDIT_SHA),
                   (BRIDGE_CANDIDATES, BRIDGE_CANDIDATES_SHA))
MIN_ELIGIBLE, MAX_EXCLUDED_PER_ARM, CAP_SECONDS = 4, 2, 14400
LATEST_LAUNCH = '2026-11-09T12:00:00Z'
STOP_RULES = ('canary', 'excluded_rows', 'server_restart', 'time_cap')
NORMAL_MARKERS = ('Agent exceeded session timeout', 'Agent exceeded turns budget', 'Agent exceeded tool call budget', 'ContextWindowExceededError',
                  'Agent completed execution without calling submit_patch')
TOO_FEW_TEMPLATE = 'the external panel had fewer than 4 eligible tasks (n_e)'
STATUS_ALL, STATUS_EXCLUSIONS = 'all_v068_evaluations_qualified', 'v068_evaluations_qualified_with_exclusions'
STATUS_STOPPED, STATUS_FAULT = 'v068_stopped_by_rule', 'v068_stopped_by_fault'
LIVE_RECHECK_SECONDS = 2  # a failed server check is repeated once after this pause before it counts as a restart
EXCLUDABLE = (RuntimeError, ValueError, KeyError, OSError, subprocess.SubprocessError)  # any other exception of an invocation is a controller fault, never an excluded row
DISK_MIN_FREE = 50 * 2 ** 30
M1QS2_SEED, M1QS2_CONTRACT = 20261104, ROOT / 'contracts/m1q-s2-training-v067-APPROVED.json'
ARTIFACT_PARTS = ('traces', 'patches', 'test_outputs')
ADAPTER_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'meta.json', 'train_log.jsonl')
ARMS = (  # slot order = run order inside a seed
    dict(arm='A0mc', alias=BASE, adapter=None, label='E4B QAT w4a16 base, no adapter, matched template'),
    dict(arm='M0q', alias='v068_M0q', adapter=ROOT / 'sft/runs/A1_M0q_v053', label='M0q adapter (unmasked control)'),
    dict(arm='M1q', alias='v068_M1q', adapter=ROOT / 'sft/runs/A1_M1q_v062', label='M1q adapter (mask recipe on the QAT base)'),
    dict(arm='M1qs2', alias='v068_M1qs2', adapter=ROOT / 'sft/runs/A1_M1q_s2_v067', label='M1q-s2 adapter (second training seed 20261104)'))

MEMORY_CORRECTION = dict(version='v043_sandbox_cgroup_v1',
    per_sandbox=dict(memory_max=4294967296, memory_swap_max=0),
    service=dict(MemoryMax=51539607552, MemorySwapMax=0, Delegate='memory', DelegateSubgroup='supervisor', OOMPolicy='continue'),
    outer_memory_events_local_zero=['max', 'oom', 'oom_kill', 'oom_group_kill'],
    host_min_available_bytes=68719476736,
    scope='Commands routed through SubprocessManager.exec: agent, verification pytest, and setup commands using that path',
    exceptions='Direct manager.start git/materialization and other direct subprocess host helpers remain under the outer 48GiB service cap only',
    comparability='Frozen numerical thresholds retained descriptively; this infrastructure correction is not evidence of historical exchangeability',
    global_oom_policy='No kernel CONSTRAINT_NONE events after admission journal cursor through final cleanup')


MEMORY_PROPERTIES = ['Delegate', 'DelegateControllers', 'DelegateSubgroup', 'OOMPolicy', 'MemoryMax',
                     'MemorySwapMax', 'Type', 'ExitType', 'KillMode']


def service_control(unit):
    names = ['LoadState', 'InvocationID', 'ControlGroup', 'MainPID', 'ActiveState', 'SubState',
             'Result', 'ExecMainStatus'] + MEMORY_PROPERTIES
    result = subprocess.run(['systemctl', '--user', 'show', unit, '--property=' + ','.join(names)],
                            capture_output=True, text=True, timeout=10)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def verify_memory_properties(state):
    require(state.get('Delegate') == 'yes' and 'memory' in state.get('DelegateControllers', '').split() and
            state.get('DelegateSubgroup') == 'supervisor' and state.get('OOMPolicy') == 'continue' and
            state.get('MemoryMax') == str(MEMORY_CORRECTION['service']['MemoryMax']) and
            state.get('MemorySwapMax') == '0' and state.get('Type') == 'exec' and
            state.get('ExitType') == 'cgroup' and state.get('KillMode') == 'control-group',
            'Actual delegated memory/service properties differ; payload not admitted')


class MemoryBoundedService(services.BoundedService):
    """Versioned v034 call mechanics with pre-release admission of delegated memory limits."""
    def __call__(self, command, log, env=None, cwd=None):
        remaining = self.deadline - time.monotonic() - RESERVE
        require(remaining > 1, 'Insufficient deadline allocation before evaluation')
        unit = 'gemma-v068-eval-' + uuid.uuid4().hex + '.service'
        require(service_control(unit).get('LoadState') == 'not-found', 'Transient unit collision')
        argv = ['systemd-run', '--user', '--wait', '--pipe', '--service-type=exec',
                '--expand-environment=no', '--unit=' + unit, '--working-directory=' + str(cwd),
                '--property=ExitType=cgroup', '--property=KillMode=control-group',
                '--property=Restart=no', '--property=TimeoutStartSec=15', '--property=RuntimeRandomizedExtraSec=0',
                '--property=SendSIGKILL=yes', '--property=RuntimeMaxSec=' + str(max(1, int(remaining))),
                '--property=TimeoutStopSec=' + str(self.stop_seconds),
                '--property=Delegate=memory', '--property=DelegateSubgroup=supervisor',
                '--property=OOMPolicy=continue', '--property=MemoryMax=51539607552', '--property=MemorySwapMax=0']
        argv += ['--setenv=' + key for key in sorted(env or {})]
        gate = self.receipt_dir / (unit + '.release')
        argv += [sys.executable, str(Path(services.__file__).resolve()), '--owned-service-payload', str(gate), str(self.gate_seconds)]
        argv += [str(arg) for arg in command]
        log = Path(log)
        require(not log.exists(), 'Refuse evaluation log overwrite')
        identity, client, error = None, None, None
        try:
            with log.open('x') as stream:
                client = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                ready = time.monotonic() + self.gate_seconds
                while identity is None:
                    state = service_control(unit)
                    if state.get('InvocationID') and state.get('ControlGroup'):
                        identity = {key: state[key] for key in ('InvocationID', 'ControlGroup')}
                        main_pid = state.get('MainPID')
                    else:
                        require(client.poll() is None and time.monotonic() < ready, 'Service identity unavailable; qualification uncertain')
                        time.sleep(.05)
                verify_memory_properties(state)
                write_json(self.receipt_dir / (log.name + '.service-start.json'), dict(identity, unit=unit,
                    main_pid=main_pid, environment_names=sorted(env or {}), memory_properties={key: state[key] for key in MEMORY_PROPERTIES}))
                gate.touch(exist_ok=False)
                client.wait(timeout=remaining + self.stop_seconds + 15)
                require(services.cgroup_empty(identity['ControlGroup']), 'Evaluation cgroup still has processes')
                require(client.returncode == 0, 'Evaluation service failed: ' + str(client.returncode))
        except BaseException as exc:
            error = exc
            if identity is not None and not services.cgroup_empty(identity['ControlGroup']):
                state = service_control(unit)
                require(all(state.get(k) == v for k, v in identity.items()), 'Service identity changed before stop')
                subprocess.run(['systemctl', '--user', 'stop', unit], check=True, timeout=self.stop_seconds + 15)
                require(services.cgroup_empty(identity['ControlGroup']), 'Owned evaluation cgroup cleanup failed')
            if client is not None:
                client.wait(timeout=25)
            raise
        finally:
            state = service_control(unit)
            empty = identity is not None and services.cgroup_empty(identity['ControlGroup'])
            write_json(self.receipt_dir / (log.name + '.service-end.json'), dict(unit=unit, identity=identity,
                state=state, cgroup_empty=empty, returncode=None if client is None else client.returncode,
                error=None if error is None else type(error).__name__ + ': ' + str(error)))


def host_memory_admission():
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].split()[0]
              for line in Path('/proc/meminfo').read_text().splitlines()}
    available = int(fields['MemAvailable']) * 1024
    require(available >= MEMORY_CORRECTION['host_min_available_bytes'], 'Less than 64GiB host memory available')
    return dict(available_bytes=available, required_bytes=MEMORY_CORRECTION['host_min_available_bytes'])


def kernel_cursor():
    result = subprocess.run(['journalctl', '--quiet', '-k', '-n', '1', '--output=json'],
                            capture_output=True, text=True, check=True, timeout=15)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    require(len(rows) == 1 and rows[0].get('__CURSOR'), 'Kernel OOM monitor cursor unavailable')
    return rows[0]['__CURSOR']


def global_oom_check(cursor):
    result = subprocess.run(['journalctl', '--quiet', '-k', '--after-cursor', cursor,
                             '--grep=CONSTRAINT_NONE', '--output=json'], capture_output=True, text=True, timeout=15)
    require(result.returncode in (0, 1) and not result.stderr.strip(), 'Kernel OOM monitor failed')
    require(not result.stdout.strip(), 'Global host OOM occurred during recovery generation')
    return dict(start_cursor=cursor, no_global_oom=True, checked_at=time.time())


def guard_qualification(path):
    from sandbox_memory_guard_v043 import qualify_guard_receipt
    data = qualify_guard_receipt(path, **MEMORY_CORRECTION['per_sandbox'])
    require(int(data['service_memory_max']) == MEMORY_CORRECTION['service']['MemoryMax'] and
            data['service_memory_swap_max'] == 0, 'Outer evaluation memory cap changed')
    before, after = data['service_memory_events_local_before'], data['service_memory_events_local_after']
    require(all(after.get(key, 0) == before.get(key, 0) == 0 for key in MEMORY_CORRECTION['outer_memory_events_local_zero']),
            'Outer evaluation memory pressure/OOM makes generation unqualified')
    return dict(path=str(path), sha256=sha(path), command_count=data['command_count'],
                oom_command_count=data['oom_command_count'], service_memory_peak=data['service_memory_peak'],
                outer_memory_events_local_before=before, outer_memory_events_local_after=after,
                recursive_cleanup=True)


def read(path):
    return json.loads(Path(path).read_text())


def leaf_paths(value, prefix=''):
    """Path and value of every scalar leaf of a JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, prefix + '/' + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, prefix + '/' + str(index))
    else:
        yield prefix, value


def pending_paths(doc):
    """Leaves of a contract that still hold the pending marker (the M1q adapter pins before the pin step)."""
    return [path for path, value in leaf_paths(doc) if value in PENDING_MARKERS]


class Halt(RuntimeError):
    """A pre-registered stop rule fired. No new invocation starts, running ones finish, the completed rows stay valid and are listed."""
    def __init__(self, rule, detail):
        if rule not in STOP_RULES:
            raise ValueError('Unknown stop rule: ' + rule)
        super().__init__('stop rule ' + rule + ': ' + detail)
        self.rule, self.detail = rule, detail


class Fault(RuntimeError):
    """A fault of the controller or the host (global OOM, existing temporary directory): the generation stops and does not qualify, as in v062."""


class AbnormalError(RuntimeError):
    """The task row of a qualified invocation carries an error outside the normal-outcome markers: the row is excluded."""


def row_problem(row, markers=NORMAL_MARKERS):
    """Why a task row cannot be an evaluation row (the v056/v061 exclusion rule), None when it can. A task's own budget overrun is a normal outcome."""
    if type(row.get('resolved')) is not bool:
        return 'resolved flag is not boolean'
    for key in ('duration_seconds', 'total_llm_calls', 'tool_calls'):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return 'invalid ' + key
    for key in ('error', 'error_message'):
        value = row.get(key)
        if value is not None and not any(marker in str(value) for marker in markers):
            return 'harness error: ' + str(value)[:160]
    return None


def eligibility_doc():
    return dict(rule='ELIGIBLE rows of the v060 ledger (DEV14 only, frozen order) at the pin step; NOT_RUN, INCOMPLETE_<stage> and INELIGIBLE_* rows are '
                     'not eligible, are listed and never waited for; CONFIRM14 is never read',
                minimum_eligible=MIN_ELIGIBLE, too_few_sentence=TOO_FEW_TEMPLATE, normal_outcome_error_markers=list(NORMAL_MARKERS),
                exclusion='a runner qualification failure of the invocation or a task error outside the normal-outcome markers')


def stops_doc():
    return dict(canary='a canary with zero parsed tool calls, a routing fault or an infrastructure or qualification failure that leaves its row unqualifiable stops '
                       'the whole generation; a row whose error is a normal outcome passes (canary acceptance = the literal v068 sentence, per decision note 97e9f6ee)',
                excluded_rows_per_arm=MAX_EXCLUDED_PER_ARM, server_restart=True, cap_seconds_from_server_ready=CAP_SECONDS,
                latest_launch=LATEST_LAUNCH, on_stop='no new invocation starts; running ones finish; completed rows are kept and listed')


def qualify_service(folder, name, expected_end=None):
    start_path = folder / (name + '.log.service-start.json')
    end_path = folder / (name + '.log.service-end.json')
    if expected_end is not None:
        require(sha(end_path) == expected_end, 'Service receipt hash changed')
    start, end = read(start_path), read(end_path)
    identity = {key: start[key] for key in ('InvocationID', 'ControlGroup')}
    prefix = 'v068' if folder == EVIDENCE else 'v034'
    require(re.fullmatch('gemma-' + prefix + r'-eval-[0-9a-f]{32}\.service', start['unit']) is not None and
            re.fullmatch('[0-9a-f]{32}', identity['InvocationID']) is not None and
            identity['ControlGroup'].startswith('/user.slice/') and int(start['main_pid']) > 0,
            'Invalid dedicated service identity')
    require(end['unit'] == start['unit'] and end['identity'] == identity and end['cgroup_empty'] is True and
            type(end['returncode']) is int and end['returncode'] == 0 and end['error'] is None,
            'Service completion/cgroup did not qualify')
    require(services.cgroup_empty(identity['ControlGroup']), 'Recorded service cgroup is active')
    if folder == EVIDENCE:
        verify_memory_properties(start['memory_properties'])
    return dict(start_sha256=sha(start_path), end_sha256=sha(end_path), unit=start['unit'], identity=identity)


def submission_agent(text, alias):
    marker = 'model: ' + BASE + '\n'
    require(text.count(marker) == 1 and '\nadapter:' not in text, 'Unexpected original agent config')
    return text.replace(marker, 'model: ' + alias + '\n')


def v060():
    """The frozen v060 runner as a module: its plan (frozen order, halves, evaluator task files, FAIL_TO_PASS ids) and its child environment."""
    return load_module(V060_RUNNER, V060_RUNNER_SHA)


def panel_plan(module=None):
    """The v060 contract and its 14 DEV tasks in the frozen selection order; the CONFIRM half is not listed anywhere in this file."""
    module = module or v060()
    contract, file_sha, core_sha = module.load_contract(V060_CONTRACT)
    require(file_sha == V060_CONTRACT_SHA and core_sha == V060_CORE_SHA, 'The v060 contract changed')
    dev = [task for task in module.load_plan(contract) if task['half'] == 'DEV']
    require(len(dev) == V060_ROWS_DEV and [task['instance_id'] for task in dev] == contract['split']['dev'], 'The DEV14 plan changed')
    for task in dev:
        require(task['task_lines'] == 1 and task['task_id'] == task['instance_id'] and Path(task['taskfile']).is_file() and
                re.fullmatch(r'[A-Za-z0-9_-]+', task['instance_id']) is not None, 'Evaluator task file or id unusable: ' + task['instance_id'])
    return contract, dev


def panel_row(task):
    return dict(instance_id=task['instance_id'], rank=task['rank'], repository=task['repository'], environment_id=task['environment_id'],
                taskfile=task['taskfile'], taskfile_sha256=sha(task['taskfile']), snapshot_sha256=task['snapshot_sha256'],
                snapshot_bytes=task['snapshot_bytes'], f2p_count=len(task['f2p']))


def dev_ledger(module, contract, dev, path=None):
    """The v060 ledger rows of the DEV half in frozen order, each equal to what v060 derives from its own stage records. A row need not be
    terminal: NOT_RUN and INCOMPLETE_<stage> rows are simply not eligible, the pre-registration never waits for them. The CONFIRM rows are neither read nor copied.
    `path` is the snapshot of these rows that the pin step wrote (the same shape as the ledger); without it the live ledger is read, by the pin step only."""
    path = V060_LEDGER if path is None else Path(path)
    require(path.is_file(), 'The v060 ledger does not exist yet' if path == V060_LEDGER else 'The pinned DEV ledger snapshot does not exist')
    data = read(path)
    require(data['schema'] == 'external-controls-v060' and data['contract_core_sha256'] == V060_CORE_SHA and
            data['runner_sha256'] == V060_RUNNER_SHA, 'The ledger belongs to another v060 contract or runner')
    by_id = {row['instance_id']: row for row in data['rows']}
    rows = [by_id.get(task['instance_id']) for task in dev]
    require(all(row is not None and row['half'] == 'DEV' and row['rank'] == task['rank'] for row, task in zip(rows, dev)),
            'Ledger rows differ from the frozen DEV14 plan')
    rule = contract['eligibility_implementation']['fail_to_pass_failure_rule']
    require(all(module.derive(row, rule) == {'status': row['status'], 'reasons': row['reasons']} for row in rows),
            'A ledger row contradicts its own stage records')
    return rows


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def ledger_digest(rows):
    """What the pinned contract keeps of the DEV rows: the eligibility rule, a digest of the rows and the status of every task."""
    return dict(status_rule='ELIGIBLE', dev_rows_sha256=sha_text(json.dumps(rows, sort_keys=True)), statuses={row['instance_id']: row['status'] for row in rows})


def not_eligible(rows):
    """The DEV rows that are not ELIGIBLE with their ledger status (INELIGIBLE_*, NOT_RUN or INCOMPLETE_<stage>), listed in the contract."""
    return [dict(instance_id=row['instance_id'], rank=row['rank'], status=row['status']) for row in rows if row['status'] != 'ELIGIBLE']


def m1qs2_gates_pass():
    """True iff meta.json of the second-seed adapter shows gates_pass true now; absent, unreadable or any other value is not ready."""
    directory = next(arm['adapter'] for arm in ARMS if arm['arm'] == 'M1qs2')
    try:
        return read(directory / 'meta.json').get('gates_pass') is True
    except (OSError, ValueError, AttributeError):
        return False


def rollout_exists():
    """True once an attempt-2 rollout has started: the server was launched, or any v068a2 result directory or log exists. Staging, the tree evidence
    of the pin and a refused activation are not a rollout."""
    return ((EVIDENCE / 'server-launched.json').exists() or any((ROOT / 'logs').glob('v068a2_*')) or any((ROOT / 'results').glob('v068a2_*')))


def pin_evidence_only(doc):
    """Before the activation the attempt-2 evidence directory holds exactly the tree evidence files of the pin, unchanged (3(c))."""
    return EVIDENCE.is_dir() and {str(path): sha(path) for path in EVIDENCE.iterdir()} == doc['pin']['tree_evidence']


def launch_window_open():
    """The pre-registration sets 2026-11-09 12:00 UTC as the latest launch."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) < LATEST_LAUNCH


def cache_file(path):
    """The exclusion rule of the v068 change record 3(a): lstat says regular file (never a symlink or special file), the immediate parent directory
    is named __pycache__ and the name ends in .pyc. Every other path is hashed exactly as v060 tree_hash hashes it."""
    return stat.S_ISREG(os.lstat(path).st_mode) and path.parent.name == '__pycache__' and path.name.endswith('.pyc')


def tree_lines(module, root, keep=lambda path: True):
    """The lines v060 tree_hash hashes (relative path, NUL, sha256 and LF for every file below root, in its order, with its sha_file), kept files only."""
    root = Path(root)
    return [(path, path.relative_to(root).as_posix() + chr(0) + module.sha_file(path) + chr(10)) for path in sorted(root.rglob('*'))
            if path.is_file() and keep(path)]


def filtered_tree_hash(module, root):
    """v060 tree_hash of a dependency tree without its bytecode caches (3(a)), with v060 sha_text: the comparison hash of attempt 2."""
    return module.sha_text(''.join(line for _, line in tree_lines(module, root, lambda path: not cache_file(path))))


def build_products(module, task, cache, tree):
    """The v060 build products of one eligible task, hashed: snapshot, router and, once per environment, the dependency tree without its bytecode
    caches (env_tree_filtered_sha256, 3(b)); the pinned admission hash env_tree_sha256 is never recomputed."""
    env = REBUILD / 'envs' / task['environment_id']
    snapshot = REBUILD / 'snapshots' / (task['instance_id'] + '.tgz')
    require(snapshot.is_file() and sha(snapshot) == task['snapshot_sha256'], 'Snapshot missing or changed: ' + task['instance_id'])
    router = env / 'router-site-packages' / 'sitecustomize.py'
    require(router.is_file() and (env / 'site-packages').is_dir(), 'Router or dependency target missing: ' + task['environment_id'])
    products = dict(snapshot=str(snapshot), router=str(router.parent), router_sha256=sha(router))
    if tree:
        if task['environment_id'] not in cache:
            cache[task['environment_id']] = filtered_tree_hash(module, env / 'site-packages')
        products['env_tree_filtered_sha256'] = cache[task['environment_id']]
    return products


def adapter_pins(directory):
    return {name: sha(Path(directory) / name) for name in ADAPTER_FILES}


def arm_rows(m1qs2=None):
    """The arms of the contract. M1q-s2 holds the pending marker until the pin step fills it (a pins dict) or drops it (False)."""
    rows = []
    for arm in ARMS:
        if arm['arm'] == 'M1qs2':
            if m1qs2 is False:
                continue
            files = m1qs2 if m1qs2 is not None else {name: PENDING_TRAINING for name in ADAPTER_FILES}
        else:
            files = adapter_pins(arm['adapter']) if arm['adapter'] else None
        rows.append(dict(arm=arm['arm'], alias=arm['alias'], label=arm['label'],
                         adapter=None if files is None else dict(path=str(arm['adapter']), files=files)))
    return rows


def lora_aliases(arms):
    return [arm['alias'] for arm in arms if arm['adapter'] is not None]


def run_rows(arms):
    """Run order: per decoding seed every arm in slot order. One run = one arm under one seed over all eligible tasks."""
    return [dict(name='v068a2_%s_s%d' % (arm['arm'], seed), arm=arm['arm'], alias=arm['alias'], decoding_seed=seed,
                 adapter_source=None if arm['adapter'] is None else arm['adapter']['path'],
                 baseline_config=str(ROOT / ('subs/e4b_budgetfit_s%d' % seed)),
                 submission_dir=str(ROOT / ('subs/v068a2_%s_s%d' % (arm['arm'], seed))),
                 result_dir=str(ROOT / ('results/v068a2_%s_s%d' % (arm['arm'], seed))))
            for seed in SEEDS for arm in arms]


def canary_rows(arms):
    """One strict canary per arm on the canary task with the seed-1 submission of that arm."""
    first = {run['arm']: run for run in run_rows(arms) if run['decoding_seed'] == 1}
    return [dict(first[arm['arm']], name='v068a2_canary_' + arm['arm'], result_dir=str(ROOT / ('results/v068a2_canary_' + arm['arm'])))
            for arm in arms]


def frozen_view(doc):
    """Everything the stages iterate over, derived from the contract."""
    return dict(arms=doc['arms'], runs=run_rows(doc['arms']), canaries=canary_rows(doc['arms']), tasks=doc['panel']['eligible'],
                canary_task=doc['panel']['canary_task'], evaluation_config=doc['evaluation'])


def admission_receipts(doc, replay_sha):
    """The v043 replay receipt (guard mechanics, byte compatibility, OOM containment). The gold40 controls receipt of v062 is bound to the Rich
    primary40 and cannot cover the external tasks, and the v060 controls ran without the guard: both facts are declared in the contract."""
    require(replay_sha, 'The byte-compatibility replay receipt is mandatory')
    replay = training.checked_json(REPLAY_RECEIPT, replay_sha)
    require(Path(str(REPLAY_RECEIPT) + '.sha256').read_text().split()[0] == replay_sha, 'Admission sidecar mismatch')
    guard_sha = doc['pins'][str(ROOT / 'tools/sandbox_memory_guard_v043.py')]
    require(replay['schema'] == 'oom-replay-v043' and replay['completed'] is True and
            replay['guard_source_sha256'] == guard_sha and replay['class_identity_preserved'] is True,
            'Replay guard/source/class identity failed')
    deterministic = replay['deterministic_replays']
    require(type(deterministic['count']) is int and deterministic['count'] == 4 and deterministic['all_semantic_fields_equal'] is True and
            set(deterministic['compared_fields']) == {'status', 'stdout', 'stderr', 'exit_code'} and
            deterministic['duration_excluded'] is True and replay['allocation_5g']['off_success'] is True and
            replay['allocation_5g']['on_oom_contained'] is True and replay['aggregate_descendants']['off_success'] is True and
            replay['aggregate_descendants']['on_oom_contained'] is True,
            'Replay byte compatibility or OOM containment failed')
    outer_replay = replay['outer_services']
    require(outer_replay['MemoryMax'] == 12884901888 and outer_replay['MemorySwapMax'] == 0 and
            outer_replay['RuntimeMaxSec'] == 180 and outer_replay['all_exited'] is True and
            outer_replay['all_cgroups_empty'] is True, 'Replay outer-service cleanup failed')
    for mapping in (replay['source_pins'], replay['source_snapshot_pins'], replay['evidence_pins']):
        require(bool(mapping), 'Admission source/result evidence pins missing')
        for name, digest in mapping.items():
            require(sha(ROOT / name) == digest, 'Admission evidence/source pin changed: ' + name)
    return dict(replay_path=str(REPLAY_RECEIPT), replay_sha256=replay_sha, byte_compatibility=True,
                controls='not applicable: the v043 gold40 controls are bound to the Rich primary40 and the v060 controls ran without the guard')


def eligible_rows(module, dev, rows, tree):
    """The eligible DEV tasks in frozen order with their hashed build products (the dependency tree only when asked)."""
    cache = {}
    return [dict(panel_row(task), **build_products(module, task, cache, tree)) for task, row in zip(dev, rows) if row['status'] == 'ELIGIBLE']


def verify_panel(panel, tree):
    """The pinned panel equals the frozen plan and the pinned snapshot of the DEV rows (its hash, its header, every row re-derived from its stage records
    as at the pin step, the digest and the eligibility), and its build products hash as pinned now. The live v060 ledger is read by the pin step only:
    whatever v060 does afterwards cannot change this verification."""
    module = v060()
    contract, dev = panel_plan(module)
    require(panel['frozen_order'] == [panel_row(task) for task in dev] and panel['half'] == 'DEV', 'The frozen DEV14 table changed')
    pinned = panel['ledger']
    require(pinned['snapshot']['path'] == str(LEDGER_SNAPSHOT) and sha(LEDGER_SNAPSHOT) == pinned['snapshot']['sha256'], 'The pinned DEV ledger snapshot changed')
    rows = dev_ledger(module, contract, dev, LEDGER_SNAPSHOT)
    digest = ledger_digest(rows)
    require(all(pinned[key] == digest[key] for key in ('status_rule', 'dev_rows_sha256', 'statuses')), 'The pinned DEV ledger digest differs from its snapshot')
    now = eligible_rows(module, dev, rows, tree)
    drop = ('env_tree_sha256',) if tree else ('env_tree_sha256', 'env_tree_filtered_sha256')  # the pinned admission hash is never recomputed (3(b))
    keep = lambda item: {key: value for key, value in item.items() if key not in drop}
    require([keep(item) for item in panel['eligible']] == [keep(item) for item in now] and len(now) >= MIN_ELIGIBLE and
            panel['canary_task'] == now[0]['instance_id'] and panel['not_eligible'] == not_eligible(rows),
            'The eligible tasks or their build products changed')


def verify_runtime(expected, full=True):
    """Pinned contract: sidecar, identity, file pins, scientific mapping. `full` adds the model inventory and the dependency trees."""
    doc = training.checked_json(RUNTIME, expected)
    require(Path(str(RUNTIME) + '.sha256').read_text().split()[0] == expected, 'Runtime sidecar mismatch')
    require(doc['schema'] == SCHEMA and doc['generation'] == GENERATION and doc['status'] == 'PINNED' and doc['root'] == str(ROOT) and
            doc['total_seconds'] == TOTAL and doc['cleanup_seconds'] == RESERVE, 'Runtime identity/bounds changed')
    require(not pending_paths(doc), 'The contract still holds pending pins; the pin step has not run')
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Runtime/package pin changed: ' + name)
    if full:
        for name, files in doc['directories'].items():
            require(inventory(name) == files, 'Runtime inventory changed: ' + name)
    verify_panel(doc['panel'], tree=full)
    require(doc['chat_template'] == dict(path=str(TEMPLATE), sha256=TEMPLATE_SHA) and doc['pins'][str(TEMPLATE)] == TEMPLATE_SHA and
            doc['server_script'] == str(SERVE) and doc['pins'][str(SERVE_ORIGINAL)] == SERVE_ORIGINAL_SHA and
            doc['byte_check']['expected'] == str(BYTE_EXPECTED) and doc['pins'][str(BYTE_EXPECTED)] == BYTE_EXPECTED_SHA,
            'Matched-context template/server/byte-check identity changed')
    arms = doc['arms']
    require(doc['runs'] == run_rows(arms) and doc['canaries'] == canary_rows(arms) and doc['memory_correction'] == MEMORY_CORRECTION and
            doc['aliases'] == lora_aliases(arms) and doc['expected_vllm_argv'] == expected_vllm_argv(comparator_vllm_argv(), lora_aliases(arms)),
            'Scientific mapping, memory correction or server argv changed')
    for arm in arms:
        if arm['adapter'] is not None:
            require(adapter_pins(arm['adapter']['path']) == arm['adapter']['files'] and
                    all(doc['pins'].get(str(Path(arm['adapter']['path']) / name)) == digest for name, digest in arm['adapter']['files'].items()),
                    'Adapter files changed or are not pinned: ' + arm['arm'])
    require(doc['revision'] == 'r22' and doc['preregistration'] == dict(path=str(PREREG), sha256=PREREG_SHA) and sha(PREREG) == PREREG_SHA,
            'The v068 pre-registration changed')
    require(doc['attempt2'] == attempt2_doc(), 'The attempt-2 inputs differ from the v068 change record')
    require(doc['eligibility'] == eligibility_doc() and doc['stops'] == stops_doc(), 'Eligibility or stop rules differ from the pre-registration')
    decision, included = doc['pin']['m1qs2'], any(arm['arm'] == 'M1qs2' for arm in arms)
    require(decision['included'] is included and decision['gates_pass_at_pin'] is included,
            'The M1q-s2 arm differs from the gates_pass decision recorded at the pin')
    return doc, frozen_view(doc)


def cases(frozen):
    """Canaries first (one arm each, the canary task only), then the runs in run order over every eligible task."""
    return [(run, frozen['tasks'][:1], True) for run in frozen['canaries']] + [(run, frozen['tasks'], False) for run in frozen['runs']]


def invocation_stem(run, task):
    return run['name'] + '__' + task['instance_id']


def invocation_log(run, task):
    return ROOT / 'logs' / (invocation_stem(run, task) + '.log')


def invocation_dir(run, task):
    return Path(run['result_dir']) / task['instance_id']


def invocation_tmp(run, task):
    """TMPDIR and HOME of one invocation. The router of the v060 environments activates in an owned workspace only below the rebuild root."""
    return REBUILD / 'tmp' / GENERATION / run['name'] / task['instance_id']


def destinations(frozen):
    paths = [MANIFEST, ROOT / 'logs/v068a2_server.log', REBUILD / 'tmp' / GENERATION]  # the evidence directory exists from the pin on (pin_evidence_only)
    for run, tasks, _ in cases(frozen):
        paths += [Path(run['result_dir']), Path(run['submission_dir'])] if run in frozen['runs'] else [Path(run['result_dir'])]
        for task in tasks:
            paths += [invocation_log(run, task), Path(str(invocation_log(run, task)) + '.routing.json')]
    return paths


def outside_adapter(files, agent=False):
    """Staged file hashes without the adapter directories and, unless asked, the alias-bearing agent.yaml (both differ from v062 by construction)."""
    return {name: digest for name, digest in files.items() if not name.startswith('adapters/') and (agent or name != 'agent.yaml')}


def copy_adapter(arm, parent):
    destination = parent / 'adapters' / arm['alias']
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        source = Path(arm['adapter']['path']) / name
        require(sha(source) == arm['adapter']['files'][name], 'Pinned adapter source changed before copy: ' + str(source))
        shutil.copyfile(source, destination / name)
        require(sha(destination / name) == sha(source), 'Adapter copy changed')


def stage_submissions(frozen, manifest=None):
    """Server manifest (base agent.yaml plus every LoRA alias) and one submission per run: the e4b_budgetfit baseline of the seed, the alias in
    agent.yaml and the adapter of the arm. Outside adapters and alias the bytes must equal the v062 matched-context submissions."""
    manifest = manifest or MANIFEST
    reference = read(PARENT_EVIDENCE / 'staged-inputs.json')['directories']
    manifest.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT / 'subs/e4b_budgetfit_s1/agent.yaml', manifest / 'agent.yaml')
    require(outside_adapter(inventory(manifest), agent=True) == outside_adapter(reference[str(ROOT / 'subs/v062_M1qmc_manifest')], agent=True),
            'Staged manifest differs from the v062 manifest outside the adapters')
    arms = {arm['arm']: arm for arm in frozen['arms']}
    for arm in frozen['arms']:
        if arm['adapter'] is not None:
            copy_adapter(arm, manifest)
    for run in frozen['runs']:
        baseline, target, arm = Path(run['baseline_config']), Path(run['submission_dir']), arms[run['arm']]
        shutil.copytree(baseline, target)
        (target / 'agent.yaml').write_text(submission_agent((baseline / 'agent.yaml').read_text(), run['alias']))
        if arm['adapter'] is not None:
            copy_adapter(arm, target)
        added = set() if arm['adapter'] is None else {'adapters/%s/%s' % (run['alias'], name)
                                                      for name in ('adapter_config.json', 'adapter_model.safetensors')}
        require(set(inventory(target)) == set(inventory(baseline)) | added, 'Unexpected submission file set')
        for name, digest in inventory(baseline).items():
            require((target / name).read_text() == submission_agent((baseline / name).read_text(), run['alias'])
                    if name == 'agent.yaml' else sha(target / name) == digest, 'Non-alias config bytes changed')
        require(outside_adapter(inventory(target)) == outside_adapter(
                reference[str(ROOT / 'subs' / ('v062_M1qmc_s%d' % run['decoding_seed']))]),
                'Staged submission differs from the v062 submission bytes outside the adapter and alias')
    from adk_submission import discover_adapters
    require(set(discover_adapters(manifest).adapters) == set(lora_aliases(frozen['arms'])), 'Official alias discovery differs')
    directories = {str(path): inventory(path) for path in [manifest] + [Path(run['submission_dir']) for run in frozen['runs']]}
    return dict(directories=directories, adapters={arm['arm']: arm['adapter'] for arm in frozen['arms']})


def verify_staged(staged):
    for name, files in staged['directories'].items():
        require(inventory(name) == files, 'Staged alias/model inputs changed')
    for adapter in staged['adapters'].values():
        if adapter is not None:
            require(adapter_pins(adapter['path']) == adapter['files'], 'Original trained adapter changed')


def registry_gate(cards, frozen):
    records = {card['id']: card for card in cards['data']}
    aliases = set(lora_aliases(frozen['arms']))
    require(len(records) == len(cards['data']) and set(records) == {BASE, *aliases}, 'Unexpected served model/alias set')
    require(Path(records[BASE]['root']).resolve() == ROOT / 'models' / BASE, 'Wrong served base weights')
    for alias in aliases:
        require(Path(records[alias]['root']).resolve() == MANIFEST / 'adapters' / alias and
                records[alias]['parent'] == BASE, 'Wrong adapter root/parent registration: ' + alias)
    return records


def server_identity(recovery, pid, start, manifest):
    """run_v032_probes.server_identity (lines 74-106) with the v055 server script instead of serve_vllm.py."""
    rows = recovery.processes()
    require(pid in rows and rows[pid]["start"] == str(start), "Server PID identity changed")
    row = rows[pid]
    args = row["argv"]
    require(len(args) == 5 and args[1] == str(SERVE)
            and args[3] == BASE and Path(args[4]).resolve() == Path(manifest).resolve(),
            "Server arguments or manifest changed")
    group = os.getpgid(pid)
    require(group != os.getpgrp(), "Refuse to signal controller process group")
    members = serving.group_members(recovery, group)
    descendants = {pid}
    for _ in rows:
        expanded = descendants | {p for p, v in rows.items() if v["parent"] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    require(set(members) <= descendants and pid in members, "Server group has unrelated processes")
    inodes = {line.split()[9] for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
              if line.split()[1] == "0100007F:1F40" and line.split()[3] == "0A"}
    require(len(inodes) == 1, "Expected one localhost:8000 listener")
    inode = next(iter(inodes))
    owners = []
    for member in members:
        try:
            if any(os.readlink(fd) == "socket:[" + inode + "]"
                   for fd in Path(f"/proc/{member}/fd").iterdir()):
                owners.append(member)
        except (FileNotFoundError, PermissionError):
            continue
    require(bool(owners), "Listener is not in the verified server group")
    return {"pid": pid, "start": str(start), "manifest": str(manifest), "group": group,
            "members": members, "listener_inode": inode, "listener_owners": owners}


def stop_server(recovery, identity, grace_seconds=60):
    """run_v032_probes.stop_server (lines 117-131) calling the v055 server_identity above."""
    current = server_identity(recovery, identity["pid"], identity["start"], identity["manifest"])
    require(current == identity, "Server group changed before shutdown")
    require(serving.same_members(identity["members"], serving.group_members(recovery, identity["group"])),
            "Server membership changed immediately before shutdown")
    os.killpg(identity["group"], signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        remaining = serving.group_members(recovery, identity["group"])
        require(all(pid in identity["members"] and serving.same_process(identity["members"][pid], row)
                    for pid, row in remaining.items()), "Unexpected process appeared during shutdown")
        if not remaining:
            return
        time.sleep(0.5)
    require(False, "Verified server group did not exit after SIGTERM; no escalation attempted")


def server_live(recovery, launched, ready=None):
    live = server_identity(recovery, launched['pid'], launched['start'], str(MANIFEST))
    require(live['members'][launched['pid']]['argv'] == launched['argv'], 'Evaluation server argv changed')
    if ready is not None:
        require(all(live[key] == ready[key] for key in ('pid', 'start', 'group', 'listener_inode', 'listener_owners')) and
                set(live['members']) == set(ready['members']) and
                all(serving.same_process(ready['members'][pid], row) for pid, row in live['members'].items()),
                'Evaluation server process/listener identities changed')
    return live


def start_server(recovery, helper, env, frozen, deadline):
    argv = [str(helper.PY), str(SERVE), str(ROOT / 'models' / BASE), BASE, str(MANIFEST)]
    with (ROOT / 'logs/v068a2_server.log').open('x') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launched = dict(pid=child.pid, start=recovery.stat(child.pid)['start'], argv=argv, group=child.pid)
    write_json(EVIDENCE / 'server-launched.json', launched)
    until = min(deadline - RESERVE, time.monotonic() + 1200)
    while time.monotonic() < until:
        require(child.poll() is None, 'Evaluation server exited before readiness')
        try:
            cards = helper.api('models')
            if len(cards['data']) == 1 + len(lora_aliases(frozen['arms'])):
                records = registry_gate(cards, frozen)
                ready = server_live(recovery, launched)
                write_json(EVIDENCE / 'server-registry.json', records)
                write_json(EVIDENCE / 'server-ready.json', services.normalize(ready))
                return launched, ready
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise TimeoutError('Evaluation server readiness deadline exceeded')


def expected_vllm_argv(recorded, aliases, manifest=None):
    """The recorded v055 vLLM argv (template flag included) with this generation's aliases (--max-loras, --lora-modules)."""
    manifest = manifest or MANIFEST
    start, end, loras = recorded.index('--lora-modules'), recorded.index('--served-model-name'), recorded.index('--max-loras')
    require(loras < start < end and recorded[-2:] == ['--chat-template', str(TEMPLATE)] and '--chat-template' not in recorded[:-2],
            'Unexpected v055 vLLM argv layout')
    argv = recorded[:start + 1] + [alias + '=' + str(manifest / 'adapters' / alias) for alias in sorted(aliases)] + recorded[end:]
    argv[loras + 1] = str(len(aliases))
    return argv


def comparator_vllm_argv():
    ready = read(COMPARATOR_EVIDENCE / 'server-ready.json')
    require(len(ready['listener_owners']) == 1, 'Comparator listener owner ambiguous')
    return ready['members'][str(ready['listener_owners'][0])]['argv']


def tokenize(body):
    request = urllib.request.Request(TOKENIZE_URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def server_log_formats(owner):
    """Content formats logged by the listening API server; its stdout is the log BaseInferenceServer opened in /tmp."""
    log = os.readlink('/proc/' + str(owner) + '/fd/1')
    require(re.fullmatch(r'/tmp/inference_server_8000_[^/]+\.log', log) is not None, 'Unexpected vLLM log destination: ' + log)
    prefix = '(APIServer pid=' + str(owner) + ')'
    found = [re.search(FORMAT_TEXT + r"'([a-z]+)'", line) for line in Path(log).read_text(errors='replace').splitlines()
             if line.startswith(prefix)]
    return log, [match[1] for match in found if match]


def token_rows(expected, responses):
    rows = []
    for call, response in zip(expected['calls'], responses, strict=True):
        tokens, ids = response.get('tokens'), call['matched_ids']
        first = (next((i for i, (a, b) in enumerate(zip(tokens, ids)) if a != b), None if len(tokens) == len(ids)
                      else min(len(tokens), len(ids))) if isinstance(tokens, list) else 0)
        rows.append(dict(call=call['call'], expected_count=len(ids), returned_count=len(tokens) if isinstance(tokens, list) else None,
                         equal=tokens == ids, first_difference=first, error=response.get('error')))
    return rows


def byte_check(ready, frozen, expected):
    """Prereg stop rule: live vLLM argv, 'openai' content format and /tokenize ids of calls 1, 3, 7 equal the offline render."""
    result = dict(endpoint=TOKENIZE_URL, expected=str(BYTE_EXPECTED), expected_sha256=sha(BYTE_EXPECTED), calls=list(BYTE_CALLS))
    try:
        require([call['call'] for call in expected['calls']] == list(BYTE_CALLS), 'Byte-check call set changed')
        owners = ready['listener_owners']
        require(len(owners) == 1, 'Expected one listening vLLM API server')
        result.update(owner=owners[0], argv=ready['members'][owners[0]]['argv'], expected_argv=expected_vllm_argv(
            comparator_vllm_argv(), lora_aliases(frozen['arms'])))
        responses = []
        for call in expected['calls']:
            try:
                responses.append(tokenize(call['request']))
            except Exception as error:
                responses.append(dict(error=type(error).__name__ + ': ' + str(error)[:500]))
        result['log'], result['content_formats'] = server_log_formats(owners[0])
        result['rows'] = token_rows(expected, responses)
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)[:500]
    result['passed'] = ('error' not in result and result['argv'] == result['expected_argv'] and
                        bool(result['content_formats']) and set(result['content_formats']) == {'openai'} and
                        all(row['equal'] for row in result['rows']))
    write_json(EVIDENCE / 'byte-check.json', result)
    require(result['passed'], 'Launch byte check failed before any task; generation stopped without retry')
    return result


def invocation_command(helper, run, task, budget):
    """The frozen helper's command for the official entry, one task, concurrency 1, with the absolute external task file and snapshot directory."""
    command = helper.eval_command(Path(run['submission_dir']), invocation_dir(run, task), [task['instance_id']], budget, 1)
    for flag, value in (('--tasks', task['taskfile']), ('--snapshots-dir', str(REBUILD / 'snapshots'))):
        require(command.count(flag) == 1, 'Unexpected official command layout: ' + flag)
        command[command.index(flag) + 1] = value
    return command


def invocation_env(base, module, contract, run, task):
    """Serving-side variables of the Rich evaluations, overlaid with the v060 child environment (PATH, HOME, TMPDIR, thread and hash settings
    and the PYTHONPATH of the per-repository router) so that the sandbox sees what the v060 controls saw."""
    return {**base, **module.child_env(contract, invocation_tmp(run, task), Path(task['router']))}


def check_artifact_names(output, task):
    """Every artifact of an invocation is a regular file named after its task, so that the run-level links cannot collide."""
    for part in ARTIFACT_PARTS:
        folder = Path(output) / part
        for path in sorted(folder.iterdir()) if folder.is_dir() else []:
            require(path.is_file() and task['instance_id'] in path.name, 'Unexpected artifact layout: ' + str(path))


def clean_tmp(tmp, stem):
    """Removes the temporary directory of an invocation after its receipt exists. A read-only entry is made writable first; what still cannot be
    removed is recorded in the evidence and never stops the generation (housekeeping, not an outcome)."""
    failed = []

    def retry(function, path, error):
        try:
            os.chmod(Path(path).parent, 0o700)
            os.chmod(path, 0o700)
            function(path)
        except OSError as problem:
            failed.append(str(path) + ': ' + str(problem))

    shutil.rmtree(tmp, onexc=retry)
    if failed or tmp.exists():
        write_json(EVIDENCE / (stem + '-tmp-cleanup.json'), dict(path=str(tmp), failed=failed, still_present=tmp.exists()))


def exclusion_path(run, task):
    return EVIDENCE / (invocation_stem(run, task) + '-excluded.json')


def task_states(run, tasks):
    """Where every planned invocation of a run stands, from the files alone: completed (receipt), excluded (exclusion record) or not run (nothing
    started). An invocation that started and left neither record is a fault of the controller."""
    states = dict(completed=[], excluded=[], not_run=[])
    for task in tasks:
        stem = invocation_stem(run, task)
        done, excluded = (EVIDENCE / (stem + '-complete.json')).exists(), exclusion_path(run, task).exists()
        require(not (done and excluded), 'An invocation is both completed and excluded: ' + stem)
        if done or excluded:
            states['completed' if done else 'excluded'].append(task['instance_id'])
        else:
            require(not invocation_log(run, task).exists() and not invocation_dir(run, task).exists(),
                    'An invocation started and left neither a completion nor an exclusion record: ' + stem)
            states['not_run'].append(task['instance_id'])
    return states


def exclude(ctx, run, task, strict, kind, error):
    """Records one excluded row: a runner qualification failure or an error outside the normal-outcome markers. The record is written first; the
    stop rules are checked after it. A restarted server stops the generation; so does a failed canary and the second excluded row of an arm."""
    stem, log, output, tmp = invocation_stem(run, task), invocation_log(run, task), invocation_dir(run, task), invocation_tmp(run, task)
    route, rows, stale = Path(str(log) + '.routing.json'), output / 'task_results.jsonl', EVIDENCE / (stem + '-complete.json')
    record = dict(kind='canary_failure' if strict else kind, reason=type(error).__name__ + ': ' + str(error)[:2000], run=run['name'],
                  arm=run['arm'], alias=run['alias'], decoding_seed=run['decoding_seed'], generation=GENERATION, task=task['instance_id'],
                  scientific_name=run['name'], result_dir=str(output), strict_canary=strict,
                  log_sha256=sha(log) if log.is_file() else None, route_sha256=sha(route) if route.is_file() else None,
                  row=rows.read_text() if rows.is_file() else None, receipt=read(stale) if stale.is_file() else None,
                  tmp_leftover=sorted(path.name for path in tmp.iterdir()) if tmp.is_dir() else None,
                  runtime_sha256=ctx.runtime_sha, staged_sha256=ctx.staged_sha)
    write_json(exclusion_path(run, task), record)
    if stale.is_file():
        stale.unlink()  # a receipt whose qualification failed is kept inside the exclusion record, never beside it
    if tmp.is_dir() and not (EVIDENCE / (stem + '-tmp-cleanup.json')).exists():
        clean_tmp(tmp, stem)  # run_invocation may have tried already; its note stays the record of a leftover
    with ctx.lock:
        listed = ctx.excluded.setdefault(run['arm'], [])
        if not strict:
            listed.append(run['name'] + ':' + task['instance_id'])
        count = len(listed)
        if strict or count >= MAX_EXCLUDED_PER_ARM:
            ctx.stop.set()  # no queued invocation starts while the server is checked below
    ctx.live()
    if strict:
        raise Halt('canary', 'canary ' + run['name'] + ' failed: ' + record['reason'])
    if count >= MAX_EXCLUDED_PER_ARM:
        raise Halt('excluded_rows', '%d excluded rows in arm %s' % (count, run['arm']))


def leaked_service(run, task):
    """True when the service-end record of an invocation shows a dedicated service whose cgroup still had processes: sandboxes leaked beside the
    rows in flight, a host-level fault that excludes nothing."""
    try:
        end = read(EVIDENCE / (invocation_log(run, task).name + '.service-end.json'))
    except (OSError, ValueError):
        return False
    return end.get('identity') is not None and end.get('cgroup_empty') is not True


def invoke(ctx, run, task, strict):
    """One invocation under the pre-registered rules. A canary fault is a stop. For a panel row a runner qualification failure (the invocation or the
    qualification that a later reader repeats) or an error outside the normal-outcome markers excludes the row and the generation goes on; a restarted
    server or the 4 h cap is a stop. An exception outside EXCLUDABLE (a controller error) and a leaked service cgroup are faults, never exclusions."""
    ctx.check_cap()
    try:
        receipt = run_invocation(ctx, run, task, strict)
        qualify_invocation(run, task, strict, ctx.budget)  # the qualification a later reader repeats; a failure excludes the row
    except (Halt, Fault):
        raise
    except EXCLUDABLE as error:
        if leaked_service(run, task):
            raise Fault('The service cgroup of %s still has processes: %s: %s' % (invocation_stem(run, task), type(error).__name__, str(error)[:300])) from error
        exclude(ctx, run, task, strict, 'abnormal_error' if isinstance(error, AbnormalError) else 'qualification_failure', error)
        return None
    except Exception as error:
        raise Fault('Unexpected %s in %s: %s' % (type(error).__name__, invocation_stem(run, task), str(error)[:300])) from error
    ctx.live()
    return receipt


def qualify_exclusion(run, task, strict):
    """The exclusion record of an invocation: identity and binding to this runtime and staged inputs; the run receipt binds its hash."""
    path = exclusion_path(run, task)
    record = read(path)
    require(record['run'] == run['name'] and record['arm'] == run['arm'] and record['alias'] == run['alias'] and record['task'] == task['instance_id'] and
            record['decoding_seed'] == run['decoding_seed'] and record['generation'] == GENERATION and record['scientific_name'] == run['name'] and
            record['result_dir'] == str(invocation_dir(run, task)) and record['strict_canary'] is strict and
            (record['kind'] == 'canary_failure' if strict else record['kind'] in ('qualification_failure', 'abnormal_error')) and
            isinstance(record['reason'], str) and bool(record['reason']), 'Exclusion record identity mismatch')
    require(record['runtime_sha256'] == sha(RUNTIME) and record['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json'),
            'Exclusion record source binding changed')
    log = invocation_log(run, task)
    route = Path(str(log) + '.routing.json')
    require(record['log_sha256'] == (sha(log) if log.is_file() else None) and record['route_sha256'] == (sha(route) if route.is_file() else None),
            'Exclusion record log or route changed')
    if record['kind'] == 'abnormal_error':
        found = [line for line in (record['row'] or '').splitlines() if line.strip()]
        require(len(found) == 1 and row_problem(json.loads(found[0])) is not None, 'An abnormal_error exclusion holds no row with a problem')
    return dict(record=record, sha256=sha(path))


def canary_check(recovery, path, task_id, budget):
    """The v062 canary row check, recorded and never a gate (decision note 97e9f6ee): the pre-registration stops a canary on zero parsed tool calls, a
    routing fault or a failure that leaves its row unqualifiable, and a row whose error is a normal outcome passes."""
    try:
        return dict(passed=True, values=recovery.validate_canary_result(path, task_id, budget))
    except (RuntimeError, ValueError, KeyError) as error:
        return dict(passed=False, reason=type(error).__name__ + ': ' + str(error)[:500])


def run_invocation(ctx, run, task, strict):
    stem, log, output, tmp = invocation_stem(run, task), invocation_log(run, task), invocation_dir(run, task), invocation_tmp(run, task)
    try:
        tmp.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise Fault('The temporary directory of the invocation exists: ' + str(tmp)) from error
    guard_receipt = EVIDENCE / (log.name + '.memory-guard.json')
    command = invocation_command(ctx.helper, run, task, ctx.budget)
    guarded = [command[0], GUARD, '--guard-receipt', guard_receipt, '--', *command[1:]]
    limit = min(ctx.deadline, time.monotonic() + INVOCATION_SECONDS + RESERVE)
    bounded = services.isolated_helper(ctx.helper, MemoryBoundedService(limit, EVIDENCE))
    route = ctx.recovery.observe_run(bounded, run['alias'], guarded, log, invocation_env(ctx.env, ctx.module, ctx.contract, run, task),
                                     ROOT / 'data/competition', strict=strict)
    try:
        oom = global_oom_check(ctx.cursor)
    except Exception as error:
        raise Fault(str(error)) from error  # a host-level event: no exclusion, the generation does not qualify
    write_json(EVIDENCE / (stem + '-oom-check.json'), oom)
    result = output / 'task_results.jsonl'
    rows = ctx.recovery.validate_rows(result, [task['instance_id']])
    problem = row_problem(json.loads(next(line for line in result.read_text().splitlines() if line.strip())))
    if problem:
        raise AbnormalError(problem)  # leaves the row unqualifiable: a canary stop as well as a panel exclusion
    canary = dict(canary_check=canary_check(ctx.recovery, result, task['instance_id'], ctx.budget)) if strict else {}
    check_artifact_names(output, task)
    leftover = sorted(path.name for path in tmp.iterdir())
    receipt = dict(rows, **canary, alias=run['alias'], arm=run['arm'], decoding_seed=run['decoding_seed'], generation=GENERATION, task=task['instance_id'],
                   scientific_name=run['name'], result_dir=str(output), strict_canary=strict, memory_guard=guard_qualification(guard_receipt),
                   global_oom_check=oom, routing=route, route_sha256=sha(Path(str(log) + '.routing.json')),
                   service=qualify_service(EVIDENCE, stem), tmp_leftover=leftover,
                   traces=services.trace_availability(output, [task['instance_id']], require_tools=strict),
                   runtime_sha256=ctx.runtime_sha, staged_sha256=ctx.staged_sha)
    write_json(EVIDENCE / (stem + '-complete.json'), receipt)
    clean_tmp(tmp, stem)
    return receipt


def execute_run(ctx, run, tasks, strict):
    """All tasks of one run: at most WORKERS invocations in flight (one for the canary), frozen order of submission. A stop rule or a fault stops new
    starts; invocations already in flight finish under their own bounds. A fault is raised, without retry. A stop rule is returned in the run
    receipt, which lists what completed, what was excluded and what never started; only the completed rows are merged and linked."""
    Path(run['result_dir']).mkdir(exist_ok=False)
    stop, halts, errors = threading.Event(), [], []
    ctx.stop = stop  # the exclusion rule sets it under the lock, before any other check

    def work(task):
        if stop.is_set():
            return
        try:
            invoke(ctx, run, task, strict)
        except Halt as halt:
            halts.append(halt)
            stop.set()
        except BaseException:
            stop.set()
            raise

    with ThreadPoolExecutor(1 if strict else WORKERS) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    errors.append(error)
        finally:
            stop.set()  # also on an interrupt of the controller: queued invocations must not start
    if errors:
        raise errors[0]
    states = task_states(run, tasks)
    completed = [task for task in tasks if task['instance_id'] in states['completed']]
    merged = None
    if completed:
        lines = []
        for task in completed:
            found = [line for line in (invocation_dir(run, task) / 'task_results.jsonl').read_text().splitlines() if line.strip()]
            require(len(found) == 1, 'One result row per invocation expected')
            lines.append(found[0])
        path = Path(run['result_dir']) / 'task_results.jsonl'
        with path.open('x') as stream:
            stream.write(chr(10).join(lines) + chr(10))
        merged = dict(path=str(path), sha256=sha(path), task_rows=len(lines))
    done = dict(run=run['name'], arm=run['arm'], alias=run['alias'], decoding_seed=run['decoding_seed'], strict_canary=strict,
                task_ids=[task['instance_id'] for task in tasks], states=states, merged=merged, linked=link_artifacts(run, completed),
                invocations={task['instance_id']: sha(EVIDENCE / (invocation_stem(run, task) + '-complete.json')) for task in completed},
                exclusions={task['instance_id']: sha(exclusion_path(run, task)) for task in tasks if task['instance_id'] in states['excluded']},
                halt=dict(rule=halts[0].rule, detail=halts[0].detail) if halts else None)
    write_json(EVIDENCE / (run['name'] + '-run.json'), done)
    return done


def link_artifacts(run, tasks):
    """Run-level traces/, patches/ and test_outputs/ as hard links to the per-invocation files, so that the v062 run-directory layout holds."""
    linked = {}
    for part in ARTIFACT_PARTS:
        target = Path(run['result_dir']) / part
        target.mkdir()
        for task in tasks:
            source = invocation_dir(run, task) / part
            for path in sorted(source.iterdir()) if source.is_dir() else []:
                os.link(path, target / path.name)
        linked[part] = sorted(path.name for path in target.iterdir())
    return linked


def linked_artifacts(run, tasks, linked):
    """The run-level artifact files are exactly the recorded ones and each is the same file as its per-invocation original."""
    for part in ARTIFACT_PARTS:
        target = Path(run['result_dir']) / part
        require(sorted(path.name for path in target.iterdir()) == linked[part], 'Run-level ' + part + ' differ from the recorded ones')
        for task in tasks:
            source = invocation_dir(run, task) / part
            for path in sorted(source.iterdir()) if source.is_dir() else []:
                require(path.samefile(target / path.name), 'Run-level ' + part + ' file is not the invocation file: ' + path.name)
    return linked


def trace_schema(path):
    from analyze_31b_recovery_screen_v034 import strict_json
    trace = strict_json(Path(path).read_text(encoding='utf-8-sig'))
    require(isinstance(trace, dict) and trace.get('schema_version') == 'ATIF-v1.7' and
            isinstance(trace.get('agent'), dict) and isinstance(trace.get('steps'), list), 'Invalid exported trace schema')
    require(all(isinstance(step, dict) and type(step.get('step_id')) is int and step['step_id'] > 0 and
                step.get('source') in ('system', 'user', 'agent') for step in trace['steps']), 'Invalid exported trace steps')
    return trace


def qualify_invocation(run, task, strict, budget):
    stem, log, output = invocation_stem(run, task), invocation_log(run, task), invocation_dir(run, task)
    receipt = read(EVIDENCE / (stem + '-complete.json'))
    require(receipt['alias'] == run['alias'] and receipt['task'] == task['instance_id'] and receipt['strict_canary'] is strict and
            receipt['arm'] == run['arm'] and receipt['decoding_seed'] == run['decoding_seed'] and receipt['generation'] == GENERATION and
            receipt['scientific_name'] == run['name'] and receipt['result_dir'] == str(output), 'Invocation identity/seed/task receipt mismatch')
    require(receipt['runtime_sha256'] == sha(RUNTIME) and receipt['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json'),
            'Run source binding changed')
    import analyze_contract_v02_v025 as qualified
    from analyze_31b_recovery_screen_v034 import outcome_rows
    rows = outcome_rows(output / 'task_results.jsonl', [task['instance_id']], qualified)
    require(row_problem(rows[task['instance_id']]) is None, 'A completed row carries an error outside the normal-outcome markers')
    require(receipt['sha256'] == sha(output / 'task_results.jsonl') and receipt['task_rows'] == 1, 'Invocation result content changed')
    raw_path = Path(str(log) + '.routing.json')
    require(receipt['route_sha256'] == sha(raw_path), 'Raw route receipt changed')
    qualified.verify_observer(read(raw_path), receipt['routing'], run['alias'], strict=strict)
    require(receipt['service'] == qualify_service(EVIDENCE, stem), 'Invocation service qualification changed')
    require(receipt['memory_guard'] == guard_qualification(EVIDENCE / (log.name + '.memory-guard.json')),
            'Invocation memory guard qualification changed')
    oom = read(EVIDENCE / (stem + '-oom-check.json'))
    require(oom == receipt['global_oom_check'] and oom['no_global_oom'] is True, 'Global OOM monitor qualification changed')
    require(receipt['traces'] == services.trace_availability(output, [task['instance_id']], require_tools=strict),
            'Trace content/call receipts changed')
    trace_schema(output / 'traces' / ('trace_' + task['instance_id'] + '.json'))
    require(('canary_check' in receipt) is strict, 'The canary check record belongs to a canary receipt and to no other')
    if strict:
        require(receipt['canary_check'] == canary_check(load_recovery(ROOT), output / 'task_results.jsonl', task['instance_id'], budget),
                'Canary check record changed')
    return dict(receipt=receipt, rows=rows)


def qualify_run(run, tasks, strict, budget):
    """Every completed invocation of the run again from its files, every exclusion record, and the merged run file equal to the completed rows in
    frozen order. A run that never started has no files and no receipt."""
    path = EVIDENCE / (run['name'] + '-run.json')
    ids = [task['instance_id'] for task in tasks]
    states = task_states(run, tasks)
    if not path.exists():
        require(states['not_run'] == ids and not Path(run['result_dir']).exists(), 'A run without a run receipt has files')
        return dict(receipt=None, invocations={}, exclusions={})
    done = read(path)
    completed = [task for task in tasks if task['instance_id'] in states['completed']]
    excluded = [task for task in tasks if task['instance_id'] in states['excluded']]
    checked = {task['instance_id']: qualify_invocation(run, task, strict, budget) for task in completed}
    exclusions = {task['instance_id']: qualify_exclusion(run, task, strict) for task in excluded}
    merged = Path(run['result_dir']) / 'task_results.jsonl'
    names = {task['instance_id']: sha(EVIDENCE / (invocation_stem(run, task) + '-complete.json')) for task in completed}
    require(done['run'] == run['name'] and done['task_ids'] == ids and done['strict_canary'] is strict and done['states'] == states and
            done['merged'] == (dict(path=str(merged), sha256=sha(merged), task_rows=len(completed)) if completed else None) and
            done['invocations'] == names and done['exclusions'] == {i: item['sha256'] for i, item in exclusions.items()},
            'Run completion receipt mismatch')
    require(linked_artifacts(run, completed, done['linked']) == done['linked'], 'Run-level artifact links changed')
    expected = [(invocation_dir(run, task) / 'task_results.jsonl').read_text().strip() for task in completed]
    require((merged.read_text().splitlines() if completed else []) == expected and (bool(completed) or not merged.exists()),
            'Merged run file differs from the per-invocation rows')
    return dict(receipt=done, invocations=checked, exclusions=exclusions)


def stop_evidence(stop, completed, frozen, runs, seen, excluded):
    """What a stop record must be able to show. The started runs are a prefix of the planned order. A stop inside a run is the halt in the receipt of
    the last started run; a stop before a run (the cap or the server check) names the next run, which has no receipt; the final server check names
    no run. A canary stop needs the excluded canary row, an excluded_rows stop an arm with two excluded rows, a time_cap stop the elapsed time."""
    order = [run['name'] for run, _, _ in cases(frozen)]
    started = [name for name in order if runs[name]['receipt'] is not None]
    require(started == order[:len(started)], 'The started runs are not a prefix of the planned order')
    rule, name = stop['rule'], stop['run']
    if seen:
        require(name == started[-1] and runs[name]['receipt']['halt'] == dict(rule=rule, detail=stop['detail']),
                'The recorded stop is not the halt of the last started run')
    elif name is None:
        require(rule == 'server_restart' and started == order, 'A stop without a run needs the final server check after every run')
    else:
        require(rule in ('time_cap', 'server_restart') and len(started) < len(order) and name == order[len(started)],
                'The stopped run has started, or does not follow the started runs')
    if rule == 'canary':
        require(name in {run['name'] for run in frozen['canaries']} and bool(runs[name]['receipt']['states']['excluded']),
                'A canary stop shows no excluded canary row')
    if rule == 'excluded_rows':
        arm = next((run['arm'] for run in frozen['runs'] if run['name'] == name), None)
        require(len(excluded.get(arm, [])) >= MAX_EXCLUDED_PER_ARM, 'An excluded_rows stop shows fewer than two excluded rows in its arm')
    if rule == 'time_cap':
        require(completed['cap_seconds'] == CAP_SECONDS and type(completed['ready_elapsed_seconds']) in (int, float) and
                completed['ready_elapsed_seconds'] >= CAP_SECONDS, 'A time_cap stop precedes the cap')


def qualify_generation(runtime_sha):
    """CPU-only post-stage interface; lifecycle qualification never needs a still-live server. A generation that finished, or that stopped by a
    pre-registered rule with its server cleaned up, qualifies; a generation that stopped by a fault does not."""
    doc, frozen = verify_runtime(runtime_sha, full=False)
    finished, halted = EVIDENCE / 'completed.json', EVIDENCE / 'stopped.json'
    require(finished.exists() != halted.exists(), 'Expected exactly one of completed.json and stopped.json')
    completed = read(finished if finished.exists() else halted)
    stop = completed.get('stop')
    if halted.exists():
        require(completed['status'] == STATUS_STOPPED and isinstance(stop, dict) and stop.get('rule') in STOP_RULES,
                'Evaluation generation stopped by a fault')
    else:
        require(completed['status'] in (STATUS_ALL, STATUS_EXCLUSIONS) and stop is None, 'Generation completion provenance/scope failed')
    require(completed['runtime_sha256'] == runtime_sha and
            completed['cleanup_completed'] is True and completed['analysis_pending'] is True and
            completed['submission_started'] is False and completed['generation'] == GENERATION and
            completed['memory_correction'] == MEMORY_CORRECTION, 'Generation completion provenance/scope failed')
    require(type(completed['elapsed_seconds']) in (int, float) and math.isfinite(completed['elapsed_seconds']) and
            0 <= completed['elapsed_seconds'] <= TOTAL, 'Invalid completed elapsed time')
    proofs = completed['admission_receipts']
    require(proofs == admission_receipts(doc, proofs['replay_sha256']), 'Admission proof changed')
    require({'activation.json', 'staged-inputs.json', 'server-launched.json', 'server-ready.json', 'byte-check.json',
             'server-registry.json', 'server-cleanup.json', 'global-oom-final.json'} <= set(completed['evidence_hashes']),
            'Missing lifecycle hash bindings')
    for name, digest in completed['evidence_hashes'].items():
        require(sha(EVIDENCE / name) == digest, 'Lifecycle/evidence receipt changed: ' + name)
    cleanup = read(EVIDENCE / 'server-cleanup.json')
    require(read(EVIDENCE / 'global-oom-final.json')['no_global_oom'] is True, 'Global OOM invalidates generation')
    require(cleanup['status'] == 'owned_server_exited' and cleanup['group_empty'] is True, 'Server shutdown incomplete')
    launched, ready = read(EVIDENCE / 'server-launched.json'), read(EVIDENCE / 'server-ready.json')
    require(ready['pid'] == launched['pid'] == cleanup['pid'] and ready['start'] == launched['start'] == cleanup['start'] and
            ready['group'] == launched['group'] == cleanup['group'], 'Recorded server lifecycle identity mismatch')
    expected_argv = [str(Path('HOME_DIR/miniconda3/envs/swegemma/bin/python')), str(SERVE), str(ROOT / 'models' / BASE), BASE, str(MANIFEST)]
    require(launched['argv'] == expected_argv and ready['group'] == ready['pid'] and
            ready['members'][str(ready['pid'])]['argv'] == expected_argv and
            ready['members'][str(ready['pid'])]['start'] == ready['start'] and ready['listener_owners'] and
            all(str(pid) in ready['members'] for pid in ready['listener_owners']), 'Recorded model/server ownership mismatch')
    byte = read(EVIDENCE / 'byte-check.json')
    require(byte['passed'] is True and byte['expected_sha256'] == BYTE_EXPECTED_SHA and
            [row['call'] for row in byte['rows']] == list(BYTE_CALLS) and all(row['equal'] for row in byte['rows']) and
            byte['argv'] == byte['expected_argv'] == ready['members'][str(byte['owner'])]['argv'] and
            set(byte['content_formats']) == {'openai'}, 'Launch byte check did not qualify')
    registry_gate({'data': list(read(EVIDENCE / 'server-registry.json').values())}, frozen)
    staged = read(EVIDENCE / 'staged-inputs.json')
    verify_staged(staged)
    require(completed['arms'] == doc['arms'] and completed['panel'] == doc['panel'], 'Arms or panel changed since activation')
    budget, runs = doc['evaluation']['effective_budget'], {}
    for run, tasks, strict in cases(frozen):
        checked = qualify_run(run, tasks, strict, budget)
        require(checked['receipt'] == completed['canaries' if strict else 'runs'][run['name']], 'Run/final completion disagreement')
        runs[run['name']] = checked
    require(set(completed['runs']) | set(completed['canaries']) == set(runs) and
            len(completed['runs']) == len(frozen['runs']) and len(completed['canaries']) == len(frozen['canaries']), 'Incomplete generation')
    receipts = [item['receipt'] for item in runs.values()]
    seen = [item['halt'] for item in receipts if item is not None and item['halt'] is not None]
    panel = [runs[run['name']]['receipt'] for run in frozen['runs']]
    excluded = {arm['arm']: sorted(run['name'] + ':' + task for run, item in zip(frozen['runs'], panel) if run['arm'] == arm['arm'] and item
                                   for task in item['states']['excluded']) for arm in frozen['arms']}
    require(len(seen) <= 1 and (not seen if stop is None else all(halt['rule'] == stop['rule'] and halt['detail'] == stop['detail'] for halt in seen)) and
            completed['excluded'] == {arm: names for arm, names in excluded.items() if names}, 'Stop or exclusion record disagrees with the runs')
    if halted.exists():
        stop_evidence(stop, completed, frozen, runs, seen, excluded)
    if finished.exists():
        every = [item for item in receipts if item is not None]
        require(len(every) == len(runs) and all(not item['states']['not_run'] and item['halt'] is None for item in every) and
                not any(item['states']['excluded'] for item in every if item['strict_canary']) and
                completed['status'] == (STATUS_EXCLUSIONS if any(names for names in excluded.values()) else STATUS_ALL),
                'A finished generation has runs that did not finish')
    return doc, frozen, completed, runs


def listener_count():
    """Listening sockets on 127.0.0.1:8000."""
    return sum(line.split()[1] == '0100007F:1F40' and line.split()[3] == '0A' for line in Path('/proc/net/tcp').read_text().splitlines()[1:])


def v060_running(recovery):
    """Processes of the v060 controls runner: it shares the rebuild root and the CPU with the invocations, and the pinned idle check does not see it."""
    return [dict(pid=value['pid'], argv=value['argv']) for value in recovery.processes().values()
            if 'run_external_controls_v060.py' in {Path(item).name for item in value['argv']}]


def shutdown_server(recovery, launched, ready, deadline, stopped):
    """Verified TERM-only shutdown of the server this controller started. After a pre-registered stop the server may be gone or replaced (a restart
    is a stop rule); the reviewed v034 cleanup then runs instead. Both signal only the verified process group."""
    method = 'verified_TERM_only'
    try:
        live = server_live(recovery, launched, ready)
        stop_server(recovery, live, grace_seconds=min(60, max(1, deadline - time.monotonic())))
    except Exception:
        if not stopped:
            raise
        services.cleanup31(recovery, launched, deadline)
        method = 'reviewed_cleanup31_TERM_only'
    require(not serving.group_members(recovery, launched['group']), 'Evaluation server group remains')
    write_json(EVIDENCE / 'server-cleanup.json', dict(status='owned_server_exited', group_empty=True, pid=launched['pid'],
                                                      start=launched['start'], group=launched['group'], method=method))


def activate(runtime_sha, replay_sha=None):
    require(launch_window_open(), 'Past the latest launch time of the pre-registration: ' + LATEST_LAUNCH)
    doc, frozen = verify_runtime(runtime_sha)
    admission = admission_receipts(doc, replay_sha)
    recovery = load_recovery(ROOT)
    require(all(not path.exists() for path in destinations(frozen)), 'Execution destination collision')
    require((REBUILD / 'tmp').is_dir(), 'The v060 rebuild temporary directory is missing')
    with (ROOT / 'migration/recovery-screen-v034/evaluation.lock').open('a') as old_lease, \
         (training.EVIDENCE.parent / 'training.lock').open('a') as train_lease:
        fcntl.flock(old_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(train_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, deadline = time.monotonic(), time.monotonic() + TOTAL
        require(pin_evidence_only(doc), 'The attempt-2 evidence directory holds other files than the tree evidence of the pin')
        try:
            verify_runtime(runtime_sha)
            admission_receipts(doc, replay_sha)
            training.assert_idle(recovery)
            require(not v060_running(recovery), 'The v060 controls runner is still running')
            memory, cursor = host_memory_admission(), kernel_cursor()
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
            require(not gpu.strip(), 'GPU compute processes remain')
            require(listener_count() == 0, 'Serving listener already occupied')
            write_json(EVIDENCE / 'activation.json', dict(runtime_sha256=runtime_sha, started_at=recovery.utc(), total_seconds=TOTAL,
                       generation=GENERATION, memory_correction=MEMORY_CORRECTION, host_memory=memory, kernel_cursor=cursor,
                       admission_receipts=admission, arms=doc['arms'], panel=doc['panel'], user_manager_failed_units=failed_units(),
                       failed_units_policy='record only; historical failed units do not block (team lead, 2026-11-05)'))
            (REBUILD / 'tmp' / GENERATION).mkdir()
            staged = stage_submissions(frozen)
            write_json(EVIDENCE / 'staged-inputs.json', staged)
            helper = recovery.load_helper(ROOT)
            env = serving.environment(doc)
            launched, ready = start_server(recovery, helper, env, frozen, deadline)
            ready_at = time.monotonic()  # the 4 h cap counts from server-ready
            byte_check(ready, frozen, training.checked_json(BYTE_EXPECTED, BYTE_EXPECTED_SHA))
            module = v060()
            def live():
                try:
                    server_live(recovery, launched, ready)
                except Exception:
                    time.sleep(LIVE_RECHECK_SECONDS)  # the check runs beside loaded invocations: one repeat before a restart is declared
                    try:
                        server_live(recovery, launched, ready)
                    except Exception as error:
                        raise Halt('server_restart', 'the evaluation server changed: ' + str(error)[:300]) from error

            def check_cap():
                if time.monotonic() >= ready_at + CAP_SECONDS:
                    raise Halt('time_cap', 'more than %d s since the server was ready' % CAP_SECONDS)

            ctx = SimpleNamespace(helper=helper, recovery=recovery, env=env, module=module, contract=panel_plan(module)[0],
                                  budget=doc['evaluation']['effective_budget'], deadline=deadline, cursor=cursor,
                                  runtime_sha=runtime_sha, staged_sha=sha(EVIDENCE / 'staged-inputs.json'),
                                  lock=threading.Lock(), excluded={}, live=live, check_cap=check_cap)
            canaries, runs, halt = {}, {}, None
            for run, tasks, strict in cases(frozen):
                require(time.monotonic() < deadline - RESERVE, 'Evaluation launch deadline barrier')
                verify_runtime(runtime_sha, full=False)
                verify_staged(staged)
                training.assert_idle(recovery)
                require(not v060_running(recovery), 'The v060 controls runner is running')
                try:
                    ctx.check_cap()
                    ctx.live()
                    registry_gate(helper.api('models'), frozen)
                    global_oom_check(cursor)
                    done = execute_run(ctx, run, tasks, strict)
                except Halt as stopped:
                    halt = dict(rule=stopped.rule, detail=stopped.detail, run=run['name'])
                    break
                checked = qualify_run(run, tasks, strict, ctx.budget)
                (canaries if strict else runs)[run['name']] = checked['receipt']
                if done['halt'] is not None:
                    halt = dict(done['halt'], run=run['name'])
                    break
                if run['name'] == frozen['canaries'][-1]['name'] or (not strict and run['arm'] == frozen['arms'][-1]['arm']):
                    verify_panel(doc['panel'], True)  # the dependency trees after the canaries and after every seed
            if halt is None:
                try:
                    ctx.live()
                except Halt as stopped:
                    halt = dict(rule=stopped.rule, detail=stopped.detail, run=None)
            for run, _, strict in cases(frozen):
                (canaries if strict else runs).setdefault(run['name'], None)  # runs that never started have no receipt
            shutdown_server(recovery, launched, ready, deadline, halt is not None)
            verify_runtime(runtime_sha)
            verify_staged(staged)
            for run, tasks, strict in cases(frozen):
                qualify_run(run, tasks, strict, ctx.budget)
            require(time.monotonic() <= deadline, 'Evaluation total deadline exceeded')
            write_json(EVIDENCE / 'global-oom-final.json', global_oom_check(cursor))
            hashes = {p.name: sha(p) for p in EVIDENCE.glob('*.json')}
            excluded = {arm: sorted(names) for arm, names in ctx.excluded.items() if names}
            write_json(EVIDENCE / ('stopped.json' if halt else 'completed.json'), dict(
                status=STATUS_STOPPED if halt else (STATUS_EXCLUSIONS if excluded else STATUS_ALL), runtime_sha256=runtime_sha,
                generation=GENERATION, memory_correction=MEMORY_CORRECTION, byte_check_sha256=hashes['byte-check.json'],
                admission_receipts=admission, arms=doc['arms'], panel=doc['panel'], cleanup_completed=True,
                runs=runs, canaries=canaries, evidence_hashes=hashes, elapsed_seconds=time.monotonic() - started,
                ready_elapsed_seconds=time.monotonic() - ready_at, cap_seconds=CAP_SECONDS, stop=halt, excluded=excluded,
                analysis_pending=True, submission_started=False))
        except BaseException as error:
            cleanup = read(EVIDENCE / 'server-cleanup.json') if (EVIDENCE / 'server-cleanup.json').exists() else dict(status='server_not_launched')
            if (EVIDENCE / 'server-launched.json').exists() and not (EVIDENCE / 'server-cleanup.json').exists():
                launched = read(EVIDENCE / 'server-launched.json')
                try:
                    services.cleanup31(recovery, launched, deadline)
                    cleanup = dict(status='owned_server_exited', method='reviewed_cleanup31_TERM_only',
                                   group_empty=not serving.group_members(recovery, launched['group']))
                except BaseException as cleanup_error:
                    cleanup = dict(status='cleanup_failed', reason=str(cleanup_error), launched=launched,
                                   residual_members=services.normalize(serving.group_members(recovery, launched['group'])))
            try:
                offenders = idle_offenders(recovery)
            except BaseException as probe_error:
                offenders = 'unavailable: ' + type(probe_error).__name__ + ': ' + str(probe_error)
            write_json(EVIDENCE / 'stopped.json', dict(status=STATUS_FAULT, type=type(error).__name__, reason=str(error), cleanup=cleanup,
                       elapsed_seconds=time.monotonic() - started, no_automatic_retry=True, idle_offenders_at_stop=offenders))
            raise


def failed_units():
    """Record-only: failed user units at activation (team-lead ruling: historical failures do not block)."""
    result = subprocess.run(['systemctl', '--user', '--failed', '--no-legend', '--plain'], capture_output=True, text=True, timeout=15)
    return dict(returncode=result.returncode, units=[line.split()[0] for line in result.stdout.splitlines() if line.strip()])


def idle_offenders(recovery):
    """Processes the pinned idle check refuses (recover_eval_v025.assert_idle plus training.assert_idle), with argv."""
    roots = {str(training.ROOT), 'HOME_DIR/gemma4'}
    offenders = []
    for value in recovery.processes().values():
        scripts = {Path(v).name for v in value['argv']}
        tied = value['cwd'] in roots or any(any(v.startswith(r + '/') for r in roots) for v in value['argv'])
        if (scripts & {'train_' + 'lora.py', 'run_' + 'eval.py'} and tied) or 'train_mask_control_v035.py' in scripts:
            offenders.append(dict(pid=value['pid'], start=value['start'], cwd=value['cwd'], argv=value['argv']))
    return offenders


def router_probe(module, contract, tasks):
    """The router rule of the v060 environments on this host: python started in a directory named workspace below the rebuild root, with the child
    environment of an invocation, must find the dependency target of the repository on sys.path (the owned-workspace path that agent commands use)."""
    results, root = {}, REBUILD / 'tmp' / (GENERATION + '-probe')
    for task in tasks:
        work = root / task['instance_id'] / 'workspace'
        work.mkdir(parents=True, exist_ok=False)
        try:
            done = subprocess.run([str(module.P(contract, 'python')), '-c', 'import json, sys; print(json.dumps(sys.path))'], cwd=work,
                                  env=module.child_env(contract, work.parent, Path(task['router'])), capture_output=True, text=True, timeout=120)
            paths = json.loads(done.stdout.splitlines()[-1]) if done.returncode == 0 and done.stdout.strip() else []
            results[task['instance_id']] = str(REBUILD / 'envs' / task['environment_id'] / 'site-packages') in paths
        finally:
            shutil.rmtree(work.parent)
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()
    return results


def preflight(runtime_sha=None):
    """CPU-only gate report before activation; takes no lock, launches nothing, reads no outcome; saved with exclusive create."""
    recovery = load_recovery(ROOT)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=15).stdout.split()
    apps = [line for line in subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=15).stdout.splitlines() if line.strip()]
    port = listener_count()
    locks = Path('/proc/locks').read_text()
    held = [str(path) for path in LOCKS if ':' + str(path.stat().st_ino) + ' ' in locks]
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].split()[0] for line in Path('/proc/meminfo').read_text().splitlines()}
    available, disk = int(fields['MemAvailable']) * 1024, shutil.disk_usage(ROOT).free
    offenders, controls = idle_offenders(recovery), v060_running(recovery)
    probe, runtime, collisions = {}, dict(present=RUNTIME.is_file(), sha256=sha(RUNTIME) if RUNTIME.is_file() else None, verified=None), []
    if runtime_sha is not None:
        try:
            doc, frozen = verify_runtime(runtime_sha)
            runtime['verified'] = True
            collisions = [str(path) for path in destinations(frozen) if path.exists()] + ([] if pin_evidence_only(doc) else [str(EVIDENCE)])
            module = v060()
            probe = router_probe(module, panel_plan(module)[0], frozen['tasks'])
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            runtime['verified'] = False
            runtime['reason'] = type(error).__name__ + ': ' + str(error)
    ready = (bool(gpu) and int(gpu[0]) <= 100 and not apps and port == 0 and not held and not offenders and not collisions and not controls and
             launch_window_open() and
             available >= MEMORY_CORRECTION['host_min_available_bytes'] and disk >= DISK_MIN_FREE and runtime['verified'] is True and
             bool(probe) and all(probe.values()))
    report = dict(ready=ready, launch_window_open=launch_window_open(), latest_launch=LATEST_LAUNCH, checked_at=recovery.utc(), gpu_memory_used_mib=int(gpu[0]) if gpu else None, gpu_compute_apps=apps,
                  port_8000_listeners=port, locks_held=held, mem_available_bytes=available, disk_free_bytes=disk,
                  disk_min_free_bytes=DISK_MIN_FREE, destination_collisions=collisions, idle_offenders=offenders,
                  v060_runner_processes=controls, router_probe=probe, failed_units=failed_units(), runtime_contract=runtime)
    PACKAGE.mkdir(exist_ok=True)
    path = PACKAGE / ('preflight-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.json')
    with path.open('x') as stream:
        stream.write(json.dumps(report, indent=1, sort_keys=True) + chr(10))
    return dict(report, receipt=str(path), receipt_sha256=sha(path))


def serve_argv(manifest):
    """The argv the unmodified server script hands to the vLLM launcher for this manifest, captured without starting anything."""
    import adk_submission, contextlib, io, runpy
    captured, original = [], adk_submission.VllmServer.start

    def capture(self):
        captured.append(self.build_cmd())
        raise SystemExit('captured')

    adk_submission.VllmServer.start = capture
    argv = sys.argv
    sys.argv = [str(SERVE), str(ROOT / 'models' / BASE), BASE, str(manifest)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(SERVE), run_name='__main__')
    except SystemExit:
        pass
    finally:
        adk_submission.VllmServer.start, sys.argv = original, argv
    require(len(captured) == 1, 'The server script did not reach the launcher')
    return captured[0]


def dry_run(runtime_sha):
    """Everything an activation does before its first GPU operation, on scratch paths: the staging with its byte checks, the vLLM argv that the staged
    manifest produces, and the command and environment of every invocation. Nothing under subs/, results/, logs/ or the evidence directory is written.
    On the staged draft all DEV14 tasks are assumed eligible and M1q-s2 stands in with the M1q adapter; on the pinned contract the pinned values apply."""
    pinned = RUNTIME.is_file() and sha(RUNTIME) == runtime_sha
    if pinned:
        doc, frozen = verify_runtime(runtime_sha, full=False)
        arms, tasks = doc['arms'], doc['panel']['eligible']
    else:
        verify_draft(runtime_sha)
        doc = read(RUNTIME_DRAFT)
        m1q = next(arm for arm in doc['arms'] if arm['arm'] == 'M1q')['adapter']
        arms = [dict(arm, adapter=m1q) if arm['adapter'] and PENDING_TRAINING in arm['adapter']['files'].values() else arm for arm in doc['arms']]
        tasks = [dict(panel_row(task), router=str(REBUILD / 'envs' / task['environment_id'] / 'router-site-packages'))
                 for task in panel_plan()[1]]
    scratch = BUILD / ('dry-run-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    scratch.mkdir()
    remap = lambda run: dict(run, submission_dir=str(scratch / Path(run['submission_dir']).relative_to(ROOT)),
                             result_dir=str(scratch / Path(run['result_dir']).relative_to(ROOT)))
    frozen = dict(arms=arms, runs=[remap(run) for run in run_rows(arms)], canaries=[remap(run) for run in canary_rows(arms)], tasks=tasks,
                  canary_task=tasks[0]['instance_id'])
    try:
        manifest = scratch / 'subs/v068_E4B_manifest'
        staged = stage_submissions(frozen, manifest)
        require(serve_argv(manifest) == expected_vllm_argv(comparator_vllm_argv(), lora_aliases(arms), manifest),
                'The staged manifest does not produce the expected vLLM argv')
        helper, module = load_recovery(ROOT).load_helper(ROOT), v060()
        env, contract, budget = serving.environment(doc), panel_plan(module)[0], doc['evaluation']['effective_budget']
        entry, seen = str(ROOT / ('run_' + 'eval.py')), set()
        for run, chosen, _ in cases(frozen):
            for task in chosen:
                command = invocation_command(helper, run, task, budget)
                at = lambda flag: command[command.index(flag) + 1]
                require(str(command[1]) == entry and [str(x) for x in command].count(entry) == 1 and at('--tasks') == task['taskfile'] and
                        at('--snapshots-dir') == str(REBUILD / 'snapshots') and at('--task-ids') == task['instance_id'] and
                        command[command.index('--task-ids') + 2] == '--sandbox' and at('--concurrency') == '1' and
                        at('--max-time-minutes') == str(budget['max_time_minutes']) and at('--max-tool-calls') == str(budget['max_tool_calls']) and
                        at('--max-turns') == str(budget['max_turns']) and at('--timeout-seconds') == str(budget['timeout_seconds']),
                        'Invocation command layout differs from the official budgets: ' + invocation_stem(run, task))
                child = invocation_env(env, module, contract, run, task)
                require(child['PYTHONPATH'] == task['router'] and child['TMPDIR'] == child['HOME'] == str(invocation_tmp(run, task)) and
                        child['CUDA_VISIBLE_DEVICES'] == '' and 'MODEL_PROXY_URL' not in child, 'Invocation environment differs')
                seen.add((str(invocation_log(run, task)), str(invocation_dir(run, task)), str(invocation_tmp(run, task))))
        total = sum(len(chosen) for _, chosen, _ in cases(frozen))
        require(len(seen) == total, 'Invocation paths are not unique')
        missing = [path for task in tasks for path in (task['router'], str(REBUILD / 'snapshots' / (task['instance_id'] + '.tgz')))
                   if not Path(path).exists()]
        return dict(mode='dry_run', runtime_sha256=runtime_sha, contract_state='PINNED' if pinned else 'DRAFT: DEV14 assumed eligible, M1q-s2 stand-in',
                    arms=[arm['arm'] for arm in arms], runs=len(frozen['runs']), canaries=len(frozen['canaries']), tasks=len(tasks),
                    invocations=total, workers=WORKERS, serve_argv_equal=True, staged_directories=len(staged['directories']),
                    staged_inventory_sha256=sha_text(json.dumps(staged['directories'], sort_keys=True)), missing_v060_products=len(missing),
                    gpu_operations=False, scratch_removed=True)
    finally:
        shutil.rmtree(scratch)


def write_or_verify(path, text):
    """Reproducible preparation: a new file is written once; a later --prepare must re-derive identical bytes."""
    if Path(path).exists():
        require(Path(path).read_text() == text, 'Preparation is not reproducible: ' + str(path))
        return
    with Path(path).open('x') as stream:
        stream.write(text)


def unified_diff(old, new):
    return ''.join(difflib.unified_diff(Path(old).read_text().splitlines(True), Path(new).read_text().splitlines(True),
                                        str(old), str(new)))


def serve_copy_is_one_flag():
    """The server script is serve_vllm.py with exactly one added argument pair appended to VllmConfig.extra_args."""
    old = 'extra_args=["--served-model-name", sys.argv[2]])'
    new = 'extra_args=["--served-model-name", sys.argv[2], "--chat-template", "' + str(TEMPLATE) + '"])'
    original = SERVE_ORIGINAL.read_text()
    return original.count(old) == 1 and SERVE.read_text() == original.replace(old, new)


def load_module(path, digest):
    require(sha(path) == digest, 'Pinned helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('v068_' + Path(path).stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


DEVIATIONS = [
    'Panel: the DEV14 tasks that the v060 ledger marks ELIGIBLE (gold-pass and empty-fail) at the pin step, in the frozen selection order; the pin step fills the list. NOT_RUN, INCOMPLETE_<stage> and INELIGIBLE_* rows are not eligible and are listed; the pin never waits for them. With fewer than 4 eligible tasks the panel is not run (sentence: the external panel had fewer than 4 eligible tasks (n_e)). CONFIRM14 is neither read nor listed.',
    'Invocations: one official-entry invocation per task (the per-repository router cannot be shared), WORKERS in flight, each in its own dedicated delegated service with its own memory guard, observer proxy and receipts; a run is the merge of its invocations. The effective budget is the Rich one except concurrency 1 per invocation.',
    'Inputs: the evaluator-only per-repository task file and the v060 rebuild snapshots, given by absolute path. The data directory of the entry is the per_repo directory, which holds no graphs, embeddings or wheels, so the code-graph tools and their prompt section are unavailable on the external tasks (declared confound against the Rich evaluations).',
    'Environment: the Rich serving environment overlaid with the v060 child environment (PATH, HOME, TMPDIR, hash and thread settings, router PYTHONPATH). HOME and TMPDIR of an invocation lie below results/extctl_v060/_rebuild/tmp/v068a2 because the router reaches an owned workspace only below the rebuild root; the directory is removed after the invocation.',
    'Exposure: the evaluator-only task rows (gold patch, FAIL_TO_PASS ids) stay readable by the same user inside the subprocess sandbox, as the Rich task file is.',
    'Admission: the v043 replay receipt (guard mechanics) is required; the v043 gold40 controls receipt is bound to the Rich primary40, and the v060 controls ran without the guard, so no receipt shows that the 4 GiB sandbox cap leaves the external gold runs unchanged. Per-invocation guard receipts record the command and OOM counts.',
    'Arms: A0mc is served as the base model name by the same LoRA-enabled engine as the adapters (v066 used an adapter-free engine); the LoRA arms are mounted at once (max-loras = number of adapters, rank 16).',
    'Canary: the first eligible DEV14 task, strict, one per arm with the seed-1 submission, before any run; it is a panel task, so the canary rows are reported separately and never replace panel rows; canary acceptance = the literal v068 sentence, per decision note 97e9f6ee: a canary stops the generation iff it has zero parsed tool calls, a routing fault, or an infrastructure or qualification failure that leaves its row unqualifiable; a row whose error is a normal outcome (a configured budget exhaustion, a context-window error, completion without submit_patch) passes. The result of the v062 canary row check is recorded in the receipt and is not a gate. The failure class is in the stop detail.',
    'Verification: the model inventory is hashed at activation start, at the end and in the preflight; the dependency trees of the eligible environments also after the canaries and after every seed, by their filtered hash (v068 change record 3(a): a regular *.pyc file directly inside a __pycache__ directory is left out, everything else is hashed as v060 tree_hash hashes it; env_tree_sha256 keeps the r21-pinned admission hash and is never recomputed). The verification timeout of the entry is the harness default (300 s, the value of the v060 controls).',
    'Host: preflight and activation (also between runs) refuse while the v060 controls runner is running, because it shares the rebuild root and the CPU and the pinned idle check does not see it. The preflight also runs a router probe: python started in a directory named workspace below the rebuild root with the child environment must find the dependency target of every eligible environment on sys.path.',
    'Housekeeping: the temporary directory of an invocation is removed after its receipt exists; read-only entries are made writable first and a leftover is recorded (tmp-cleanup.json), never fatal. Artifacts must be regular files named after their task; run-level traces/, patches/ and test_outputs/ are hard links.',
    'Exclusion and stops (v068 pre-registration): a panel row is excluded when its invocation fails runner qualification (the checks of qualify_invocation, run on every invocation as it finishes and repeated by every later reader) or its task error lies outside the normal-outcome markers (v056); the exclusion record is written, a receipt that failed qualification is kept inside it, and the generation goes on. A failed canary, the second excluded row of an arm, any server restart (identity of the verified server group and listener; a failed check is repeated once after 2 s) and 4 h from server-ready stop the generation: no new invocation starts, running ones finish, the server is stopped TERM-only and the completed rows are kept and listed (stopped.json). A host-level global OOM, a leaked service cgroup, a controller error (an exception outside RuntimeError, ValueError, KeyError, OSError and SubprocessError) or any other fault stops the generation without a qualifiable record, as in v062.',
    'M1q-s2 arm: included if and only if sft/runs/A1_M1q_s2_v067/meta.json shows gates_pass true at the moment the pin step runs; the pin step records the decision, the meta.json hash and the adapter hash (or --without-arm M1qs2). The adapter must pass the sealed-training checks (main run of the approved second-seed contract, seed 20261104, not a copy of the M1q adapter).',
    'Pin (r22, v068 change record 3(c)-(e)): the pin step reads the r21 pinned contract (8450ed23) and its DEV ledger snapshot contracts/external-e4b-v068-r21-dev-ledger-PINNED.json (79c26c72), both read only (the live v060 ledger is not read), and refuses unless the arms with their adapter pins, the ledger digest, the not-eligible list, the canary task and the r21 keys of every eligible row equal the r21 pin and, for every environment, the full v060 listing now minus exactly the attempt-1 cache paths of migration/external-e4b-v068/attempt2-bridge-candidates.txt (a7bfafc5) hashes to the r21-pinned env_tree_sha256. The eligible rows are the r21 rows unchanged plus env_tree_filtered_sha256. Per environment the pin writes the full listing, the full and the filtered hash and the starting cache list to migration/external-e4b-v068/attempt2/evaluation and pins their sha256; a refusal writes nothing. It refuses while the v060 controls runner is running and is repeatable while no attempt-2 rollout exists (no v068a2 server, result or invocation log); an earlier pinned file and its evidence directory are kept under superseded names. The re-pin after the attempt-1 rollouts is a declared deviation from the pre-registration clause on repeated pins; the arm set cannot change. The pre-registration (8061b917) is a fixed input of the contract.',
    'Time box: activation refuses after 2026-11-09 12:00 UTC, the latest launch of the pre-registration.',
    'Unchanged from v062: template flag and checks, server script, argv layout, byte-check logic and stop rule (expectation of v066, request model = base name), memory correction and guard, budgets, sampling, seeds, TERM-only cleanup, fail-closed without retry for faults.']


def pending_panel(dev):
    return dict(source=dict(contract=str(V060_CONTRACT), contract_sha256=V060_CONTRACT_SHA, core_sha256=V060_CORE_SHA,
                            runner=str(V060_RUNNER), runner_sha256=V060_RUNNER_SHA, ledger=str(V060_LEDGER)),
                half='DEV', frozen_order=[panel_row(task) for task in dev], ledger=PENDING_LEDGER, eligible=PENDING_LEDGER,
                not_eligible=PENDING_LEDGER, canary_task=PENDING_LEDGER, minimum_eligible=MIN_ELIGIBLE,
                rule='status ELIGIBLE in the v060 ledger at the pin step, frozen order; NOT_RUN, INCOMPLETE_<stage> and INELIGIBLE_* rows are not '
                     'eligible and listed; fewer than 4 eligible tasks: no run; CONFIRM14 is never read')


PIN_NAMES = ['contracts/matched-context-v062-runtime-PINNED.json', 'contracts/matched-context-v062-runtime-PINNED.json.sha256',
    'migration/matched-context-v062/evaluation/staged-inputs.json', 'migration/matched-context-v062/evaluation/completed.json',
    'tools/run_matched_context_v066.py', 'contracts/matched-context-v066-a0mc-runtime-PINNED.json',
    'contracts/matched-context-v066-a0mc-runtime-PINNED.json.sha256', 'migration/matched-context-v066/byte-check-expected.json',
    'tools/run_external_controls_v060.py', 'tools/test_run_external_controls_v060.py', 'contracts/external-controls-v060-APPROVED-r3.json',
    'data/external_validation_v036/selection.json', 'external/gemma-4-task-soundness-and-local-results/validation_rebuild.py',
    'external/gemma-4-task-soundness-and-local-results/validation_env_recipes.json', 'models.yaml', 'run_' + 'eval.py',
    'tools/run_external_e4b_v068_r21.py', 'tools/test_run_external_e4b_v068_r21.py', 'tools/analyze_external_e4b_v068_r21.py',
    'tools/test_analyze_external_e4b_v068_r21.py', 'analysis/external-e4b-boss-v068-r21/runner.diff', 'migration/external-e4b-v068/tasks.json',
    'tools/run_external_e4b_v068_r22.py', 'tools/test_run_external_e4b_v068_r22.py', 'tools/analyze_external_e4b_v068_r22.py',
    'tools/test_analyze_external_e4b_v068_r22.py', 'analysis/external-e4b-boss-v068-r22/runner.diff', 'analysis/external-e4b-boss-v068-r22/analyzer.diff']


def contract_pins(parent, dev, arms):
    """Every v062 pin (re-verified now), the v066/v060/v068 files, the 14 evaluator task files and the adapter files that are not pending."""
    pins = dict(parent['pins'])
    for name, digest in pins.items():
        require(sha(name) == digest, 'A v062 pin changed: ' + name)
    pins.update({str(ROOT / name): sha(ROOT / name) for name in PIN_NAMES})
    pins.update({task['taskfile']: sha(task['taskfile']) for task in dev})
    pins[str(PREREG)] = sha(PREREG)
    pins.update({str(path): sha(path) for path, _ in ATTEMPT2_INPUTS})
    for arm in arms:
        if arm['adapter'] is not None:
            pins.update({str(Path(arm['adapter']['path']) / name): digest for name, digest in arm['adapter']['files'].items()
                         if digest not in PENDING_MARKERS})
    require(pins[str(TEMPLATE)] == TEMPLATE_SHA and pins[str(SERVE_ORIGINAL)] == SERVE_ORIGINAL_SHA and pins[str(SERVE_PARENT)] == SERVE_PARENT_SHA and
            pins[str(RUNNER_PARENT)] == RUNNER_PARENT_SHA and pins[str(BYTE_EXPECTED)] == BYTE_EXPECTED_SHA and
            pins[str(V060_CONTRACT)] == V060_CONTRACT_SHA and pins[str(V060_RUNNER)] == V060_RUNNER_SHA and pins[str(PREREG)] == PREREG_SHA,
            'A pre-registered v068 input hash differs')
    require(all(pins[str(path)] == digest for path, digest in ATTEMPT2_INPUTS), 'An attempt-2 input differs from the v068 change record')
    return pins


def attempt2_doc():
    """The fixed inputs of attempt 2 that the contract records (v068 change record 3(d), 3(e)) and its tree rule (3(a), 3(b))."""
    return dict(change_record=dict(path=str(CHANGE_RECORD), sha256=CHANGE_RECORD_SHA), r21_runtime=dict(path=str(R21_RUNTIME), sha256=R21_RUNTIME_SHA),
                r21_runner=dict(path=str(R21_RUNNER), sha256=R21_RUNNER_SHA), audit=dict(path=str(AUDIT), sha256=AUDIT_SHA),
                bridge_candidates=dict(path=str(BRIDGE_CANDIDATES), sha256=BRIDGE_CANDIDATES_SHA),
                ledger_snapshot=dict(path=str(LEDGER_SNAPSHOT), sha256=LEDGER_SNAPSHOT_SHA), evidence=str(EVIDENCE),
                tree_rule='a path below a dependency tree is left out of the comparison hash iff lstat says regular file, its immediate parent directory '
                          'is named __pycache__ and its name ends in .pyc; everything else is hashed as v060 tree_hash hashes it; env_tree_sha256 keeps '
                          'the r21-pinned admission hash and is never recomputed; env_tree_filtered_sha256 is the attempt-2 baseline')


def contract_doc(m1qs2=None, panel=None, status='DRAFT_PREPARED_ONLY', pin=None):
    """The runtime contract as a dict: final everywhere except the pending leaves (eligible tasks, M1q-s2 pins and the pin record)."""
    require(sha(PARENT_RUNTIME) == PARENT_RUNTIME_SHA, 'The v062 pinned contract changed')
    parent, (_, dev), arms = read(PARENT_RUNTIME), panel_plan(), arm_rows(m1qs2)
    pins, aliases = contract_pins(parent, dev, arms), lora_aliases(arms)
    runner = str(ROOT / 'tools/run_external_e4b_v068_r22.py')
    return dict(schema=SCHEMA, revision='r22', status=status, root=str(ROOT), generation=GENERATION, total_seconds=TOTAL, cleanup_seconds=RESERVE,
        workers=WORKERS, invocation_seconds=INVOCATION_SECONDS, memory_correction=MEMORY_CORRECTION,
        guard_source_sha256=GUARD_SOURCE_SHA, guard_wrapper_sha256=GUARD_WRAPPER_SHA, required_replay_receipt=str(REPLAY_RECEIPT),
        pins=pins, directories=parent['directories'], server_environment=parent['server_environment'],
        arms=arms, aliases=aliases, runs=run_rows(arms), canaries=canary_rows(arms), run_order=[run['name'] for run in run_rows(arms)],
        panel=panel if panel is not None else pending_panel(dev),
        preregistration=dict(path=str(PREREG), sha256=PREREG_SHA), eligibility=eligibility_doc(), stops=stops_doc(),
        attempt2=attempt2_doc(),
        pin=pin if pin is not None else PENDING_TRAINING,
        policy='Launch byte check, then one strict routing/tool canary per arm on the canary task, then per decoding seed every arm over all '
               'eligible tasks; one harness invocation per task, WORKERS in flight; no efficacy stopping, analysis, submission or training; '
               'a failed canary, two excluded rows in one arm, a server restart or 4 h from server-ready stop the generation and keep the completed rows; '
               'any other fault stops it without retry',
        server_handoff='Verified TERM-only shutdown; qualification uses immutable recorded lifecycle, not server liveness',
        gpu_operations_during_preparation=False, server_manifest=str(MANIFEST),
        chat_template=dict(path=str(TEMPLATE), sha256=TEMPLATE_SHA),
        server_script=str(SERVE), server_script_sha256=pins[str(SERVE)],
        serve_diff=dict(path=str(SERVE_DIFF), sha256=pins[str(SERVE_DIFF)], base=str(SERVE_PARENT), base_sha256=SERVE_PARENT_SHA,
                        original=str(SERVE_ORIGINAL), original_sha256=SERVE_ORIGINAL_SHA),
        runner=dict(path=runner, sha256=pins[runner], diff=str(RUNNER_DIFF), diff_sha256=pins[str(RUNNER_DIFF)],
                    base=str(R21_RUNNER), base_sha256=R21_RUNNER_SHA),
        expected_vllm_argv=expected_vllm_argv(comparator_vllm_argv(), aliases),
        byte_check=dict(expected=str(BYTE_EXPECTED), expected_sha256=BYTE_EXPECTED_SHA, trace=str(BYTE_TRACE), calls=list(BYTE_CALLS),
                        alias=BYTE_ALIAS, endpoint=TOKENIZE_URL, content_format='openai',
                        stop_rule='Any mismatch or error: byte-check.json, stopped.json, TERM cleanup, no retry'),
        evaluation={key: parent['evaluation'][key] for key in ('effective_budget', 'sampling', 'compaction')},
        parent=dict(runtime=str(PARENT_RUNTIME), runtime_sha256=PARENT_RUNTIME_SHA, runner=str(RUNNER_PARENT), runner_sha256=RUNNER_PARENT_SHA),
        readme=dict(deviations_from_v062_runner=DEVIATIONS))


def contract_text(doc):
    return json.dumps(doc, indent=2, sort_keys=True) + chr(10)


def tasks_text():
    """tasks.json: the frozen DEV14 table; the eligible subset is pinned from the finished v060 ledger."""
    return json.dumps(dict(source=dict(contract=str(V060_CONTRACT), contract_sha256=V060_CONTRACT_SHA, core_sha256=V060_CORE_SHA,
                                       runner=str(V060_RUNNER), runner_sha256=V060_RUNNER_SHA),
                           half='DEV', order='selection_sha256 ascending, as the v060 plan', tasks=[panel_row(task) for task in panel_plan()[1]]),
                      indent=2) + chr(10)


def prepare():
    """Writes the staged draft contract and its by-products once; a later call must re-derive identical bytes."""
    require(sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == GUARD_SOURCE_SHA and sha(GUARD) == GUARD_WRAPPER_SHA,
            'Guard sources are not the independently approved versions')
    require(serve_copy_is_one_flag() and sha(SERVE_PARENT) == SERVE_PARENT_SHA and SERVE.read_bytes() == SERVE_PARENT.read_bytes(),
            'Server script is not the v051 server script (serve_vllm.py plus the one template flag)')
    require(sha(BYTE_EXPECTED) == BYTE_EXPECTED_SHA, 'Byte-check expectation changed')
    PACKAGE.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    require(sha(R21_RUNNER) == R21_RUNNER_SHA, 'The r21 runner changed')
    write_or_verify(RUNNER_DIFF, unified_diff(R21_RUNNER, Path(__file__).resolve()))
    write_or_verify(TASKS, tasks_text())
    write_or_verify(RUNTIME_DRAFT, contract_text(contract_doc()))
    digest = sha(RUNTIME_DRAFT)
    write_or_verify(Path(str(RUNTIME_DRAFT) + '.sha256'), digest + '  ' + RUNTIME_DRAFT.name + chr(10))
    return digest


def verify_draft(expected):
    """Inspection of the staged draft, never a run input: its sidecar, and the contract re-derived now from the current files equals the draft."""
    require(Path(str(RUNTIME_DRAFT) + '.sha256').read_text().split()[0] == expected == sha(RUNTIME_DRAFT), 'Draft sidecar mismatch')
    require(RUNTIME_DRAFT.read_text() == contract_text(contract_doc()), 'The staged draft does not re-derive from the current files')
    return dict(runtime_draft_sha256=expected, pending_fields=len(pending_paths(read(RUNTIME_DRAFT))))


def sealed_m1qs2(attested):
    """The M1q-s2 pins once its training receipts are sealed: files present, the adapter hash equal to the attested one, the receipt gates passed
    and the adapter shaped like M1q (configuration and tensor names, shapes, dtypes)."""
    directory, reference = next(arm['adapter'] for arm in ARMS if arm['arm'] == 'M1qs2'), next(arm['adapter'] for arm in ARMS if arm['arm'] == 'M1q')
    require(all((directory / name).is_file() for name in ADAPTER_FILES), 'M1q-s2 adapter or training receipts unavailable')
    pins, meta, first = adapter_pins(directory), read(directory / 'meta.json'), read(reference / 'meta.json')
    require(pins['adapter_model.safetensors'] == attested, 'The M1q-s2 adapter hash differs from the attested one')
    require(meta.get('gates_pass') is True and meta.get('updates') == meta.get('updates_planned') == 208 and meta.get('names_match_probe') is True,
            'M1q-s2 training receipt gates did not pass')
    require(meta.get('mode') == 'm1q-s2-main' and meta['args']['seed'] == M1QS2_SEED and meta['args']['out'] == str(directory) and
            meta['training_contract']['path'] == str(M1QS2_CONTRACT) and meta['training_contract']['status'] == 'APPROVED',
            'M1q-s2 receipt is not the main run of the approved second-seed contract')
    require(pins['adapter_model.safetensors'] != adapter_pins(reference)['adapter_model.safetensors'] and
            meta['initial_trainable_parameters_sha256'] != first['initial_trainable_parameters_sha256'], 'M1q-s2 repeats the M1q adapter or initialization')
    shape = lambda path: {k: (v['shape'], v['dtype']) for k, v in training.safe_header(path / 'adapter_model.safetensors').items()}
    require(read(directory / 'adapter_config.json') == read(reference / 'adapter_config.json') and shape(directory) == shape(reference),
            'M1q-s2 adapter differs from M1q in configuration, tensor names, shapes or dtypes')
    return pins


def bridge_candidates():
    """The attempt-1 cache paths of the bridge (3(d)), read only from the candidates file whose sha256 the contract pins: sorted lines
    '<environment_id>/<relative_path>' and LF, each split at the first '/'."""
    data = BRIDGE_CANDIDATES.read_bytes()
    require(hashlib.sha256(data).hexdigest() == BRIDGE_CANDIDATES_SHA, 'The bridge candidates file is not the pinned one: ' + str(BRIDGE_CANDIDATES))
    paths = {}
    for line in data.decode('utf-8').split(chr(10))[:-1]:
        environment, relative = line.split('/', 1)
        paths.setdefault(environment, set()).add(relative)
    return paths


def r21_pin():
    """The r21 pinned contract that attempt 2 bridges to: the file, its sidecar and its hash, read only."""
    doc = training.checked_json(R21_RUNTIME, R21_RUNTIME_SHA)
    require(Path(str(R21_RUNTIME) + '.sha256').read_text().split()[0] == R21_RUNTIME_SHA, 'The r21 runtime sidecar does not match its contract')
    return doc


def tree_evidence(module, item, mine):
    """3(c) and the bridge of 3(d) for the dependency tree of one environment: the full listing now (the exact lines v060 tree_hash hashes), its full
    hash, the filtered hash and the starting cache list (every excluded file: path, sha256, bytes, mtime_ns). Refuses unless every candidate path is a
    cache file of the listing and the listing minus exactly the candidate paths hashes to the r21-pinned admission hash env_tree_sha256."""
    root = REBUILD / 'envs' / item['environment_id'] / 'site-packages'
    rows = [(path.relative_to(root).as_posix(), line, cache_file(path)) for path, line in tree_lines(module, root)]
    full = module.sha_text(''.join(line for _, line, _ in rows))
    require(full == module.tree_hash(root), 'The listing does not reproduce v060 tree_hash: ' + item['environment_id'])
    require(mine <= {name for name, _, cached in rows if cached}, 'A bridge candidate is not a cache file of the listing: ' + item['environment_id'])
    bridge = module.sha_text(''.join(line for name, line, _ in rows if name not in mine))
    require(bridge == item['env_tree_sha256'], 'Bridge refused for %s: the listing minus its %d candidate paths hashes to %s, not to the r21-pinned %s'
            % (item['environment_id'], len(mine), bridge, item['env_tree_sha256']))
    cache = []
    for name, line, cached in rows:
        if cached:
            status = os.lstat(root / name)
            cache.append(dict(path=name, sha256=line[len(name) + 1:-1], bytes=status.st_size, mtime_ns=status.st_mtime_ns))
    summary = dict(environment_id=item['environment_id'], tree=str(root), files=len(rows), admission_sha256=item['env_tree_sha256'], full_sha256=full,
                   filtered_sha256=module.sha_text(''.join(line for _, line, cached in rows if not cached)), bridge_sha256=bridge,
                   bridge_candidate_count=len(mine), cache_file_count=len(cache), bridge_candidates=sorted(mine), cache_files=cache)
    return ''.join(line for _, line, _ in rows), summary


def pin(m1qs2_sha=None, without_arm=None):
    """Writes the r22 pinned contract beside the staged draft (v068 change record 3(c), 3(d)). The panel is the r21 pin's: the DEV rows come from the r21
    snapshot (read only; the live v060 ledger is not read) and the eligible rows are the r21 rows unchanged plus env_tree_filtered_sha256. The pin
    refuses unless the arms with their adapter pins (the M1q-s2 rule is r21's), the ledger digest, the not-eligible list, the canary task and the r21 keys
    of every eligible row equal the r21 pin and, for every environment, the full listing now minus exactly its attempt-1 candidate paths hashes to the
    r21-pinned env_tree_sha256. Per environment the full listing, the full and the filtered hash and the starting cache list go to the attempt-2 evidence
    directory and their sha256 into the contract; a refusal writes nothing. A repeated pin is allowed only while no attempt-2 rollout exists; the earlier
    pinned files and evidence directory are then kept under superseded names."""
    require(not rollout_exists(), 'An external rollout exists; the pinned contract can no longer be replaced')
    require(not v060_running(load_recovery(ROOT)), 'The v060 controls runner is running: pin only after the ledger is complete and frozen')
    gates = m1qs2_gates_pass()
    if gates:
        require(bool(m1qs2_sha) and without_arm is None,
                'M1q-s2 meta.json shows gates_pass true: give --m1qs2-adapter-sha; --without-arm M1qs2 is refused')
    else:
        require(without_arm == 'M1qs2' and not m1qs2_sha,
                'M1q-s2 meta.json does not show gates_pass true: pin with --without-arm M1qs2 and without an adapter hash')
    verify_draft(sha(RUNTIME_DRAFT))
    require(sha(PREREG) == PREREG_SHA, 'The v068 pre-registration changed')
    old = r21_pin()
    module = v060()
    contract, dev = panel_plan(module)
    require(sha(LEDGER_SNAPSHOT) == LEDGER_SNAPSHOT_SHA, 'The DEV ledger snapshot is not the r21-pinned file')
    rows = dev_ledger(module, contract, dev, LEDGER_SNAPSHOT)
    without = lambda item, key: {name: value for name, value in item.items() if name != key}
    require([without(item, 'env_tree_sha256') for item in old['panel']['eligible']] == eligible_rows(module, dev, rows, False),
            'The eligible rows (r21 keys) or their build products differ from the r21 pin')
    candidates, evidence = bridge_candidates(), {}
    require(set(candidates) <= {item['environment_id'] for item in old['panel']['eligible']}, 'A bridge candidate names an environment outside the panel')
    for item in old['panel']['eligible']:
        if item['environment_id'] not in evidence:
            evidence[item['environment_id']] = tree_evidence(module, item, candidates.get(item['environment_id'], set()))
    eligible = [dict(item, env_tree_filtered_sha256=evidence[item['environment_id']][1]['filtered_sha256']) for item in old['panel']['eligible']]
    require(len(eligible) >= MIN_ELIGIBLE, TOO_FEW_TEMPLATE.replace('n_e', str(len(eligible))))
    meta = next(arm['adapter'] for arm in ARMS if arm['arm'] == 'M1qs2') / 'meta.json'
    earlier = sorted(sha(path) for path in RUNTIME.parent.glob(RUNTIME.name + '.superseded-*')) + ([sha(RUNTIME)] if RUNTIME.exists() else [])
    files = {EVIDENCE / ('pin-tree-' + name + suffix): data for name, (listing, summary) in evidence.items()
             for suffix, data in (('.txt', listing.encode('utf-8')), ('.json', (json.dumps(summary, indent=1, sort_keys=True) + chr(10)).encode('utf-8')))}
    record = dict(m1qs2=dict(rule='included iff sft/runs/A1_M1q_s2_v067/meta.json shows gates_pass true when --pin runs', gates_pass_at_pin=gates,
                             included=gates, meta_sha256=sha(meta) if meta.is_file() else None, adapter_sha256=m1qs2_sha or None),
                  supersedes=earlier, tree_evidence={str(path): hashlib.sha256(data).hexdigest() for path, data in files.items()},
                  bridge={name: {key: value for key, value in summary.items() if key not in ('bridge_candidates', 'cache_files')}
                          for name, (_, summary) in evidence.items()})
    panel = dict(pending_panel(dev), ledger=dict(ledger_digest(rows), snapshot=dict(path=str(LEDGER_SNAPSHOT), sha256=sha(LEDGER_SNAPSHOT))),
                 eligible=eligible, not_eligible=not_eligible(rows),
                 canary_task=eligible[0]['instance_id'])
    doc = contract_doc(sealed_m1qs2(m1qs2_sha) if gates else False, panel, 'PINNED', record)
    require(not pending_paths(doc), 'Pending leaves remain after pinning')
    require(doc['arms'] == old['arms'] and without(doc['panel'], 'eligible') == without(old['panel'], 'eligible'),
            'The arms, adapter pins, ledger digest, not-eligible list or canary task differ from the r21 pin')
    if RUNTIME.exists():
        for path in (RUNTIME, Path(str(RUNTIME) + '.sha256'), EVIDENCE):
            path.rename(path.with_name(path.name + '.superseded-' + earlier[-1][:8]))
    EVIDENCE.mkdir(parents=True)
    for path, data in files.items():
        with path.open('xb') as stream:
            stream.write(data)
    write_or_verify(RUNTIME, contract_text(doc))
    digest = sha(RUNTIME)
    write_or_verify(Path(str(RUNTIME) + '.sha256'), digest + '  ' + RUNTIME.name + chr(10))
    return dict(runtime_sha256=digest, n_eligible=len(eligible), eligible=[item['instance_id'] for item in eligible],
                not_eligible=panel['not_eligible'], canary_task=panel['canary_task'], arms=[arm['arm'] for arm in doc['arms']],
                ledger_snapshot_sha256=panel['ledger']['snapshot']['sha256'], m1qs2=record['m1qs2'], supersedes=earlier, invocations=len(eligible) * len(doc['runs']) + len(doc['canaries']),
                bridge=record['bridge'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = ('prepare', 'pin', 'preflight', 'dry_run', 'activate')
    for mode in modes:
        parser.add_argument('--' + mode.replace('_', '-'), action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--replay-receipt-sha')
    parser.add_argument('--m1qs2-adapter-sha')
    parser.add_argument('--without-arm', choices=['M1qs2'])
    args = parser.parse_args()
    require(sum(getattr(args, mode) for mode in modes) <= 1, 'Prepare, pin, preflight, dry-run and activate are separate modes')
    if args.prepare:
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(), gpu_operations=False)))
    elif args.pin:
        print(json.dumps(dict(status='pinned', gpu_operations=False, **pin(args.m1qs2_adapter_sha, args.without_arm)), indent=1))
    elif args.preflight:
        report = preflight(args.runtime_sha)
        print(json.dumps(report, indent=1))
        sys.exit(0 if report['ready'] else 3)
    elif args.dry_run:
        require(args.runtime_sha, 'Dry run requires the runtime SHA (staged draft or pinned)')
        print(json.dumps(dry_run(args.runtime_sha), indent=1))
    elif args.activate:
        require(args.runtime_sha, 'Activation requires runtime SHA')
        activate(args.runtime_sha, args.replay_receipt_sha)
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        if RUNTIME_DRAFT.is_file() and sha(RUNTIME_DRAFT) == args.runtime_sha:
            print(json.dumps(dict(status='draft_verified', gpu_operations=False, **verify_draft(args.runtime_sha))))
            return
        doc, frozen = verify_runtime(args.runtime_sha)
        admitted = admission_receipts(doc, args.replay_receipt_sha)['byte_compatibility'] if args.replay_receipt_sha else None
        print(json.dumps(dict(status='pinned_verified', gpu_operations=False, arms=[arm['arm'] for arm in doc['arms']],
                              eligible=[task['instance_id'] for task in frozen['tasks']], admission_receipts_verified=admitted)))


if __name__ == '__main__':
    main()
