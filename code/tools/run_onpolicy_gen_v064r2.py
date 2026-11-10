"""On-policy generation v064-r2: E4B base A0 (LoRA off) on the Stage-0 eligible tasks, a 3-seed pilot then a 7-seed scale stage, through the v052 A0cap machinery."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import sys
import uuid

from prepare_v032_probes import BASE, inventory, load_recovery, require, sha, write_json
import run_v032_probes as serving
import run_31b_recovery_screen_v034 as services
import run_mask_control_v0351 as training

ROOT = Path('PROJECT_ROOT/gemma4')
READOUT = ROOT / 'contracts/mask-control-v035-readout.json'
READOUT_SHA = '6319262b25c19ddcbb0e34a33f5e97325a7d6117a7b8faad8d4a4c7ff772c909'
DRAFT = ROOT / 'contracts/onpolicy-gen-v064r2-runtime-DRAFT.json'
RUNTIME = ROOT / 'contracts/onpolicy-gen-v064r2-runtime-PINNED.json'
PACKAGE = ROOT / 'migration/onpolicy-v064'
RUNNER_DIFF = PACKAGE / 'r2/runner.diff'
SOURCE_RUNNER = ROOT / 'tools/run_a0_capped_v052.py'
SOURCE_RUNNER_SHA = '63ab0049e67b5d9fb1f121e6a18b5573df456eae43b05724830a489a0873d7f9'
R1_RUNNER = ROOT / 'tools/run_onpolicy_gen_v064.py'
R1_RUNNER_SHA = 'af9c4ffae01281a1dbf8fdada0d2acfadfc0d2c003ebd50d77fe22e47b668865'
PREREG = Path('PROJECT_ROOT/briefing/research/gemma-4-2026/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md')
PREREG_SHA = '0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60'
ADDENDUM = Path('PROJECT_ROOT/briefing/research/gemma-4-2026/prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065.md')
ADDENDUM_SHA = 'b602a5d366002bc5335fcc7a94ada72cba71535b3a63551f74d28bd9a8f39a56'
STAGE0 = ROOT / 'contracts/onpolicy-controls-v064b.json'
STAGE0_SHA = '71613c1d3a397b32a609b5fbb2688ab8d59308edc961a167ab192da955cb4103'
ELIGIBLE = PACKAGE / 'eligible.json'
GATE = PACKAGE / 'gate.json'
SPLIT = ROOT / 'splits/v1.json'
SPLIT_SHA = '9c99c412a0e094dca93f6867703e535b3a84c9d4878c5bc867bac8d8b5a59f8d'
MODES = {'pilot': dict(seeds=(1, 2, 3), package=PACKAGE / 'pilot', status='pilot_runs_qualified', start_minimum=60),
         'scale': dict(seeds=tuple(range(4, 11)), package=PACKAGE / 'scale', status='scale_runs_qualified', start_minimum=5400)}
MODE = 'pilot'
EVIDENCE = PACKAGE / 'pilot/evaluation'
MANIFEST = PACKAGE / 'pilot/server-manifest'
KEPT = PACKAGE / 'kept_trajectories.json'
SERVER_LOG = ROOT / 'logs/v064_pilot_A0_server.log'
PREVIOUS_STOPPED = ROOT / 'migration/a0-capped-v048/evaluation/stopped.json'
PREVIOUS_STOPPED_SHA = '814c647b3cc363cb11a1146849953728b2fea396c351e296fd7cde34c84b7a43'
LOCKS = (ROOT / 'migration/recovery-screen-v034/evaluation.lock', ROOT / 'migration/mask-control-v0351/training.lock')
V043_RUNTIME = ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json'
V043_RUNTIME_SHA = '7eaf5d53b56e02d6e45ef29a229748b2716bdee36ae7b4ebf0301450d4c57a63'
GUARD = ROOT / 'tools/run_eval_memory_guard_v043.py'
GUARD_SOURCE_SHA = '5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GUARD_WRAPPER_SHA = 'c04928a3406c1175098f1217dd7b955dcec61aec1147537fbd68bf59de8bbbd5'
CONTROLS_RECEIPT = ROOT / 'analysis/oom-controls-v043/final48/report.json'
REPLAY_RECEIPT = ROOT / 'analysis/oom-replay-v043/report.json'
TOTAL, RESERVE = 28800, 600
GENERATION_CAP, PILOT_CAP, RUN_CAP, CAP_MARGIN = 25200, 7200, 5400, 5
KEEP_PER_TASK, FLOOR_KEPT, FLOOR_TASKS, GATE_TASKS, MIN_ELIGIBLE = 3, 30, 10, 5, 10
CANARY = 'fastapi_11194'
ARM, SEEDS = 'A0', tuple(range(1, 11))
SAMPLING_SHA = {1: '2df78016a612024ed34c3cc1383ecaf35f2735cafdaa1b8d43495934aa45166c',
                2: '7add435195b430d1debfa7fc9d27d8eec37bd52f75ffa619b0326fdbd9c2c7a6',
                3: '9b29a6e21a8dbc327e551bea9b89219be7e76879d07240f9f62d390578d7c2f8',
                4: 'be4541748f781abebb7505c3dd327e348abeaeaae6725af46a2890411199596e',
                5: 'ce07c30ce362b0f4f120d83c3a010f7b7912b701f0d7c5fe8f11388f17c24798',
                6: '87037f54c9fa7741cd33d74dd440042db73881673576ef94b0305f9287024c82',
                7: '55682f741f9a0e7614767262fda2b877d74871ad0c2e6513fb1392a3f8842ea8',
                8: '804aaf39fe83d8fe8872b4dcd2b8d01f2fe0ba8ad6c2286b7820733794fba022',
                9: '6add68aed031a90245dba432d559a00828a00c9a4fcdda1c7d7b98f8fd1f9d07',
                10: 'ed9d8f3801c2d11fe07b29479ca1649bb422b9b3580ff616dc448d50cfe88394'}
CANDIDATES = (
    'fastapi_11355', 'fastapi_12942', 'fastapi_13207', 'fastapi_13537', 'fastapi_13713', 'fastapi_13786', 'fastapi_13920',
    'fastapi_14099', 'fastapi_14246', 'fastapi_14258', 'fastapi_14262', 'fastapi_14266', 'fastapi_14297', 'fastapi_14301',
    'fastapi_14303', 'fastapi_14306', 'fastapi_14349', 'fastapi_14360', 'fastapi_14361', 'fastapi_14371', 'fastapi_14372',
    'fastapi_14419', 'fastapi_14430', 'fastapi_14448', 'fastapi_14455', 'fastapi_14458', 'fastapi_14459', 'fastapi_14463',
    'fastapi_14479', 'fastapi_14482', 'fastapi_14485', 'fastapi_14487', 'fastapi_14492', 'fastapi_14512', 'fastapi_14583',
    'fastapi_14605', 'fastapi_14609', 'fastapi_14616', 'fastapi_14786', 'fastapi_14791', 'fastapi_14794', 'fastapi_14873',
    'fastapi_14953', 'fastapi_14962', 'fastapi_14964', 'fastapi_14978', 'fastapi_14986', 'fastapi_15023', 'fastapi_15030',
    'fastapi_15280', 'fastapi_15588', 'fastapi_15589', 'fastapi_15661', 'fastapi_15763', 'fastapi_15785', 'fastapi_15800',
    'fastapi_5077', 'fastapi_5624', 'fastapi_9425', 'fastapi_9555', 'fastapi_9753', 'httpx_3672', 'requests_6644',
    'requests_7205', 'requests_7309', 'requests_7315', 'requests_7328', 'requests_7427', 'requests_7505',
)
MEMORY_CORRECTION = dict(version='v043_sandbox_cgroup_v1',
    per_sandbox=dict(memory_max=4294967296, memory_swap_max=0),
    service=dict(MemoryMax=51539607552, MemorySwapMax=0, Delegate='memory', DelegateSubgroup='supervisor', OOMPolicy='continue'),
    outer_memory_events_local_zero=['max', 'oom', 'oom_kill', 'oom_group_kill'],
    host_min_available_bytes=68719476736,
    scope='Commands routed through SubprocessManager.exec: agent, verification pytest, and setup commands using that path',
    exceptions='Direct manager.start git/materialization and other direct subprocess host helpers remain under the outer 48GiB service cap only',
    comparability='Frozen numerical thresholds retained descriptively; this infrastructure correction is not evidence of historical exchangeability',
    global_oom_policy='No kernel CONSTRAINT_NONE events after admission journal cursor through final cleanup')


def select_mode(mode):
    """Bind the stage's write-once paths; called once at CLI startup, before any other call."""
    global MODE, EVIDENCE, MANIFEST, SERVER_LOG
    require(mode in MODES, 'Unknown stage: ' + str(mode))
    MODE, package = mode, MODES[mode]['package']
    EVIDENCE, MANIFEST = package / 'evaluation', package / 'server-manifest'
    SERVER_LOG = ROOT / ('logs/v064_' + mode + '_' + ARM + '_server.log')


def eligible_tasks(doc):
    """The only task list this runner runs: the pinned Stage-0 eligible subset of the 69 candidates; never rich_ or the canary."""
    pinned = doc.get('eligible') or {}
    ids = list(pinned.get('task_ids') or [])
    require(pinned.get('sha256') is not None and sha(ELIGIBLE) == pinned['sha256'] and read(ELIGIBLE)['eligible_task_ids'] == ids,
            'Eligible list not pinned or changed')
    require(len(CANDIDATES) == len(set(CANDIDATES)) == 69 and CANARY not in CANDIDATES and len(ids) == len(set(ids)) >= MIN_ELIGIBLE and
            set(ids) <= set(CANDIDATES) and not any(task.startswith('rich_') for task in ids + list(CANDIDATES)),
            'Refuse a task list outside the eligible train-split candidates')
    return ids


def canary_run(mode=None):
    """The stage's strict canary: fastapi_11194 with the seed-1 configuration; its rollout is never kept."""
    name = 'v064_canary_' + (MODE if mode is None else mode) + '_' + ARM + '_s1'
    return dict(name=name, scientific_name=name, arm=ARM, adapter_alias=BASE, adapter_source=None, temperature=.2, decoding_seed=1,
                baseline_config=str(ROOT / 'subs/e4b_budgetfit_s1'), submission_dir=str(ROOT / 'subs' / name),
                result_dir=str(ROOT / 'results' / name), trace_dir=str(ROOT / 'results' / name / 'traces'))


def generation_runs(mode=None):
    spec = MODES[MODE if mode is None else mode]
    runs = []
    for seed in spec['seeds']:
        name = 'v064_gen_' + ARM + '_s' + str(seed)
        runs.append(dict(name=name, scientific_name=name, arm=ARM, adapter_alias=BASE, adapter_source=None,
                         temperature=.2, decoding_seed=seed,
                         baseline_config=str(ROOT / ('subs/e4b_budgetfit_s' + str(seed))),
                         submission_dir=str(ROOT / 'subs' / name), result_dir=str(ROOT / 'results' / name),
                         trace_dir=str(ROOT / 'results' / name / 'traces')))
    return runs


def effective_readout(frozen):
    effective = json.loads(json.dumps(frozen))
    effective['future_runs'] = generation_runs()
    return effective


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
        unit = 'gemma-v064-eval-' + uuid.uuid4().hex + '.service'
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


def admission_receipts(doc, controls_sha, replay_sha):
    require(controls_sha and replay_sha, 'Gold40 controls and byte-compatibility replay receipts are mandatory')
    controls = training.checked_json(CONTROLS_RECEIPT, controls_sha)
    replay = training.checked_json(REPLAY_RECEIPT, replay_sha)
    require(Path(str(CONTROLS_RECEIPT) + '.sha256').read_text().split()[0] == controls_sha and
            Path(str(REPLAY_RECEIPT) + '.sha256').read_text().split()[0] == replay_sha, 'Admission sidecar mismatch')
    frozen = readout()
    primary = frozen['populations']['primary40']['task_ids']
    all_ids = frozen['populations']['all48_sensitivity']['task_ids']
    p, outer = controls['primary40'], controls['memory']
    guard_sha = doc['pins'][str(ROOT / 'tools/sandbox_memory_guard_v043.py')]
    require(controls['schema'] == 'oom-controls-v043' and controls['qualified'] is True and
            controls['frozen_readout_sha256'] == READOUT_SHA and p['ids'] == primary and
            p['expected_count'] == p['gold_resolved_count'] == p['gold_exit_zero_count'] == 40 and
            p['all_gold_pass'] is True, 'Frozen primary40 gold controls did not all qualify')
    require(controls['all48']['ids'] == all_ids and controls['all48']['row_count'] == 48 and
            set(controls['original8']['ids']) == set(all_ids) - set(primary) and
            len(controls['original8']['rows']) == 8, 'Controls population/classification changed')
    require(outer['sandbox_memory_max'] == 4294967296 and outer['sandbox_swap_max'] == 0 and
            outer['service_memory_max'] == 25769803776 and outer['service_memory_swap_max'] == 0 and
            outer['service_result'] == 'success' and outer['service_returncode'] == 0 and
            outer['cgroup_empty'] is True and outer['guard_qualified'] is True, 'Controls memory/cgroup proof failed')
    require(re.fullmatch('[0-9a-f]{32}', outer['service_invocation_id']) is not None, 'Controls service invocation missing')
    for snapshot in (outer['service_properties'], outer['service_properties_last']):
        require(snapshot['InvocationID'] == outer['service_invocation_id'] and snapshot['Delegate'] == 'yes' and
                snapshot['DelegateSubgroup'] == 'supervisor' and snapshot['OOMPolicy'] == 'continue' and
                snapshot['MemoryMax'] == '25769803776' and snapshot['MemorySwapMax'] == '0' and
                snapshot['ControlGroup'].startswith('/user.slice/'), 'Controls actual service properties failed')
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
    require(str(ROOT / 'tools/sandbox_memory_guard_v043.py') in {
        str(ROOT / name) for name in controls['input_sha256']}, 'Controls omit guard source pin')
    for mapping in (controls['input_sha256'], replay['source_pins'], replay['source_snapshot_pins'], replay['evidence_pins']):
        require(bool(mapping), 'Admission source/result evidence pins missing')
        for name, digest in mapping.items():
            require(sha(ROOT / name) == digest, 'Admission evidence/source pin changed: ' + name)
    require(controls['input_sha256'].get('tools/sandbox_memory_guard_v043.py',
                controls['input_sha256'].get(str(ROOT / 'tools/sandbox_memory_guard_v043.py'))) == guard_sha,
            'Controls guard source differs from admitted runtime')
    return dict(controls_path=str(CONTROLS_RECEIPT), controls_sha256=controls_sha,
                replay_path=str(REPLAY_RECEIPT), replay_sha256=replay_sha, gold_primary40=40,
                original8_classified=True, byte_compatibility=True)


def read(path):
    return json.loads(Path(path).read_text())


def readout():
    require(sha(READOUT) == READOUT_SHA and Path(str(READOUT) + '.sha256').read_text().split()[0] == READOUT_SHA,
            'Frozen readout changed')
    doc = read(READOUT)
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Readout source changed: ' + name)
    require(doc['evaluation_config']['effective_budget'] == dict(max_time_minutes=4, max_tool_calls=28,
            max_turns=80, timeout_seconds=240, concurrency=4), 'Frozen effective budget changed')
    require([run['name'] for run in doc['future_runs']] ==
            ['v035_' + arm + '_s' + str(seed) for arm in ('M0', 'M1') for seed in (1, 2, 3)], 'Frozen run order changed')
    require(CANARY not in doc['populations']['all48_sensitivity']['task_ids'], 'Canary overlaps holdout')
    return doc


def snapshot(task):
    return ROOT / 'data/competition/snapshots' / (task + '.tgz')


def train_tasks():
    """Prereg v064-r2: sound train-split tasks of the pinned split in split order; the 69 candidates are that minus the canary."""
    require(sha(SPLIT) == SPLIT_SHA, 'Train/holdout split changed')
    split = read(SPLIT)
    sound = [task for group in split['train_groups'] for task in group if split['soundness'][task]['sound'] == '1']
    holdout = set(split['holdout_rich']) | set(readout()['populations']['all48_sensitivity']['task_ids'])
    require(len(sound) == len(set(sound)) == 70 and CANARY in sound and [t for t in sound if t != CANARY] == list(CANDIDATES),
            'Sound train-split list differs from the pinned candidates')
    require(len(holdout) == 48 and all(task.startswith('rich_') for task in holdout) and not holdout & set(sound),
            'Holdout identity changed or overlaps the train split')
    registry = {json.loads(line)['instance_id'] for line in
                (ROOT / 'data/competition/tasks.jsonl').read_text().splitlines() if line.strip()}
    require(set(sound) <= registry and all(snapshot(task).is_file() for task in sound), 'Task registry entry or snapshot missing')
    return sound


def verify_snapshots(doc, ids):
    """Task snapshots (about 14 GB) are hashed at activation and at finalization, not before every case."""
    for task in ids:
        require(sha(snapshot(task)) == doc['snapshot_sha256'][str(snapshot(task))], 'Task snapshot changed: ' + task)
    return dict(tasks=len(ids), verified_at=time.time())


def verify_runtime(expected, path=None):
    path = RUNTIME if path is None else path
    doc = training.checked_json(path, expected)
    require(Path(str(path) + '.sha256').read_text().split()[0] == expected, 'Runtime sidecar mismatch')
    require(doc['root'] == str(ROOT) and doc['total_seconds'] == TOTAL and doc['cleanup_seconds'] == RESERVE and
            doc['generation'] == 'v064' and doc['revision'] == 'r2' and doc['generation_cap_seconds'] == GENERATION_CAP and
            doc['pilot_cap_seconds'] == PILOT_CAP and doc['run_cap_seconds'] == RUN_CAP, 'Runtime identity/bounds changed')
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Runtime/package pin changed: ' + name)
    for name, files in doc['directories'].items():
        require(inventory(name) == files, 'Runtime inventory changed: ' + name)
    frozen = readout()
    require(doc['generation_runs'] == {mode: generation_runs(mode) for mode in MODES} and
            doc['canary_runs'] == {mode: canary_run(mode) for mode in MODES} and doc['memory_correction'] == MEMORY_CORRECTION and
            doc['candidate_ids'] == list(CANDIDATES) and doc['sampling_sha256'] == {str(s): SAMPLING_SHA[s] for s in SEEDS} and
            doc['prereg'] == dict(path=str(PREREG), sha256=PREREG_SHA) and doc['pins'].get(str(PREREG)) == PREREG_SHA and
            doc['pins'].get(str(STAGE0)) == STAGE0_SHA and doc['pins'].get(str(ADDENDUM)) == ADDENDUM_SHA,
            'Scientific mapping or memory correction changed')
    if doc['status'] == 'PINNED':
        eligible_tasks(doc)
    return doc, effective_readout(frozen)


def qualify_service(folder, name, expected_end=None):
    start_path = folder / (name + '.log.service-start.json')
    end_path = folder / (name + '.log.service-end.json')
    if expected_end is not None:
        require(sha(end_path) == expected_end, 'Service receipt hash changed')
    start, end = read(start_path), read(end_path)
    identity = {key: start[key] for key in ('InvocationID', 'ControlGroup')}
    prefix = 'v064' if folder == EVIDENCE else 'v034'
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


def destinations(frozen):
    paths = [EVIDENCE, MANIFEST, KEPT, SERVER_LOG] + ([GATE] if MODE == 'pilot' else [])
    for run in [canary_run()] + frozen['future_runs']:
        paths += [Path(run['submission_dir']), Path(run['result_dir'])]
        paths += [ROOT / ('logs/' + run['name'] + '.log'), ROOT / ('logs/' + run['name'] + '.log.routing.json')]
    return paths


def stage_submissions(frozen):
    MANIFEST.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / 'subs/e4b_budgetfit_s1/agent.yaml', MANIFEST / 'agent.yaml')
    for run in [canary_run()] + frozen['future_runs']:
        baseline, target = Path(run['baseline_config']), Path(run['submission_dir'])
        agent = (baseline / 'agent.yaml').read_text()
        require(run['adapter_alias'] == BASE and run['adapter_source'] is None and run['temperature'] == .2 and
                run['decoding_seed'] in SEEDS and agent.count('model: ' + BASE + '\n') == 1 and '\nadapter:' not in agent and
                sha(baseline / 'configs/sampling.yaml') == SAMPLING_SHA[run['decoding_seed']], 'Unexpected base alias/seed/sampling')
        shutil.copytree(baseline, target)
        require(inventory(target) == inventory(baseline), 'Base submission byte copy changed')
    from adk_submission import discover_adapters
    folders = [MANIFEST] + [Path(run['submission_dir']) for run in [canary_run()] + frozen['future_runs']]
    require(all(discover_adapters(folder).adapters == {} for folder in folders), 'Adapter-free base staging required')
    return dict(directories={str(path): inventory(path) for path in folders})


def verify_staged(staged):
    for name, files in staged['directories'].items():
        require(inventory(name) == files, 'Staged alias/model inputs changed')


def registry_gate(cards, frozen):
    records = {card['id']: card for card in cards['data']}
    aliases = {run['adapter_alias'] for run in frozen['future_runs']}
    require(aliases == {BASE} and len(records) == len(cards['data']) == 1 and set(records) == {BASE},
            'Unexpected served model/alias set')
    require(Path(records[BASE]['root']).resolve() == ROOT / 'models' / BASE and records[BASE]['parent'] is None,
            'Wrong served base weights')
    return records


def server_live(recovery, launched, ready=None):
    live = serving.server_identity(recovery, launched['pid'], launched['start'], str(MANIFEST))
    require(live['members'][launched['pid']]['argv'] == launched['argv'], 'Evaluation server argv changed')
    if ready is not None:
        require(all(live[key] == ready[key] for key in ('pid', 'start', 'group', 'listener_inode', 'listener_owners')) and
                set(live['members']) == set(ready['members']) and
                all(serving.same_process(ready['members'][pid], row) for pid, row in live['members'].items()),
                'Evaluation server process/listener identities changed')
    return live


def start_server(recovery, helper, env, frozen, deadline):
    argv = [str(helper.PY), str(ROOT / 'serve_vllm.py'), str(ROOT / 'models' / BASE), BASE, str(MANIFEST)]
    with SERVER_LOG.open('x') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launched = dict(pid=child.pid, start=recovery.stat(child.pid)['start'], argv=argv, group=child.pid)
    write_json(EVIDENCE / 'server-launched.json', launched)
    until = min(deadline - RESERVE, time.monotonic() + 1200)
    while time.monotonic() < until:
        require(child.poll() is None, 'Evaluation server exited before readiness')
        try:
            cards = helper.api('models')
            if len(cards['data']) == 1:
                records = registry_gate(cards, frozen)
                ready = server_live(recovery, launched)
                write_json(EVIDENCE / 'server-registry.json', records)
                write_json(EVIDENCE / 'server-ready.json', services.normalize(ready))
                return launched, ready
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise TimeoutError('Evaluation server readiness deadline exceeded')


def qualify_run(name, alias, ids, output, log, strict, budget):
    receipt = read(EVIDENCE / (name + '-complete.json'))
    require(receipt['alias'] == alias and receipt['task_ids'] == ids and receipt['strict_canary'] is strict,
            'Run identity/seed/task receipt mismatch')
    identity = re.fullmatch('(v064_' + ('canary_' + MODE + '_' if strict else 'gen_') + '(A0)_s([0-9]+))', name)
    require(identity is not None and int(identity[3]) in ((1,) if strict else MODES[MODE]['seeds']) and alias == BASE and
            receipt['arm'] == identity[2] and
            type(receipt['decoding_seed']) is int and receipt['decoding_seed'] == int(identity[3]) and
            receipt['runtime_sha256'] == sha(RUNTIME) and receipt['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json') and
            sha(ROOT / 'subs' / identity[1] / 'configs/sampling.yaml') == SAMPLING_SHA[int(identity[3])],
            'Run arm/seed/source binding changed')
    require(receipt['generation'] == 'v064' and receipt['scientific_name'] == identity[1] and
            receipt['result_dir'] == str(output), 'Recovery generation/path association changed')
    import analyze_contract_v02_v025 as qualified
    from analyze_31b_recovery_screen_v034 import outcome_rows
    rows = outcome_rows(output / 'task_results.jsonl', ids, qualified)
    require(receipt['sha256'] == sha(output / 'task_results.jsonl') and receipt['task_rows'] == len(ids), 'Run result content changed')
    raw_path = Path(str(log) + '.routing.json')
    require(receipt['route_sha256'] == sha(raw_path), 'Raw route receipt changed')
    qualified.verify_observer(read(raw_path), receipt['routing'], alias, strict=strict)
    require(receipt['service'] == qualify_service(EVIDENCE, log.stem), 'Run service qualification changed')
    require(receipt['memory_guard'] == guard_qualification(EVIDENCE / (log.name + '.memory-guard.json')),
            'Run memory guard qualification changed')
    oom = read(EVIDENCE / (name + '-oom-check.json'))
    require(oom == receipt['global_oom_check'] and oom['no_global_oom'] is True, 'Global OOM monitor qualification changed')
    require(receipt['traces'] == services.trace_availability(output, ids, require_tools=strict), 'Trace content/call receipts changed')
    for task in ids:
        trace_schema(output / 'traces' / ('trace_' + task + '.json'))
    if strict:
        recovery = load_recovery(ROOT)
        expected = recovery.validate_canary_result(output / 'task_results.jsonl', ids[0], budget)
        require(all(receipt[key] == value for key, value in expected.items()), 'Canary budget/error qualification changed')
    return dict(receipt=receipt, rows=rows)


def trace_schema(path):
    from analyze_31b_recovery_screen_v034 import strict_json
    trace = strict_json(Path(path).read_text(encoding='utf-8-sig'))
    require(isinstance(trace, dict) and trace.get('schema_version') == 'ATIF-v1.7' and
            isinstance(trace.get('agent'), dict) and isinstance(trace.get('steps'), list), 'Invalid exported trace schema')
    require(all(isinstance(step, dict) and type(step.get('step_id')) is int and step['step_id'] > 0 and
                step.get('source') in ('system', 'user', 'agent') for step in trace['steps']), 'Invalid exported trace steps')
    return trace


def qualify_generation(runtime_sha):
    """CPU-only post-stage interface; lifecycle qualification never needs a still-live server."""
    doc, frozen = verify_runtime(runtime_sha)
    require(not (EVIDENCE / 'stopped.json').exists(), 'Evaluation generation stopped')
    completed = read(EVIDENCE / 'completed.json')
    require(completed['status'] in (MODES[MODE]['status'], 'stage_capped_completed_runs_qualified') and completed['mode'] == MODE and
            completed['runtime_sha256'] == runtime_sha and completed['readout_sha256'] == READOUT_SHA and
            completed['cleanup_completed'] is True and completed['analysis_pending'] is True and
            completed['submission_started'] is False and completed['elapsed_seconds'] <= TOTAL,
            'Generation completion provenance/scope/deadline failed')
    require(completed['generation'] == 'v064' and completed['memory_correction'] == MEMORY_CORRECTION,
            'Recovery memory policy/provenance changed')
    proofs = completed['admission_receipts']
    require(proofs == admission_receipts(doc, proofs['controls_sha256'], proofs['replay_sha256']), 'Admission proof changed')
    require(type(completed['elapsed_seconds']) in (int, float) and math.isfinite(completed['elapsed_seconds']) and
            0 <= completed['elapsed_seconds'] <= TOTAL, 'Invalid completed elapsed time')
    require({'activation.json', 'staged-inputs.json', 'server-launched.json', 'server-ready.json',
             'server-registry.json', 'server-cleanup.json', 'global-oom-final.json'} <= set(completed['evidence_hashes']), 'Missing lifecycle hash bindings')
    for name, digest in completed['evidence_hashes'].items():
        require(sha(EVIDENCE / name) == digest, 'Lifecycle/evidence receipt changed: ' + name)
    cleanup = read(EVIDENCE / 'server-cleanup.json')
    require(read(EVIDENCE / 'global-oom-final.json')['no_global_oom'] is True, 'Global OOM invalidates generation')
    require(cleanup['status'] == 'owned_server_exited' and cleanup['group_empty'] is True, 'Server shutdown incomplete')
    launched, ready = read(EVIDENCE / 'server-launched.json'), read(EVIDENCE / 'server-ready.json')
    require(ready['pid'] == launched['pid'] == cleanup['pid'] and ready['start'] == launched['start'] == cleanup['start'] and
            ready['group'] == launched['group'] == cleanup['group'], 'Recorded server lifecycle identity mismatch')
    expected_argv = [str(Path('HOME_DIR/miniconda3/envs/swegemma/bin/python')), str(ROOT / 'serve_vllm.py'),
                     str(ROOT / 'models' / BASE), BASE, str(MANIFEST)]
    require(launched['argv'] == expected_argv and ready['group'] == ready['pid'] and
            ready['members'][str(ready['pid'])]['argv'] == expected_argv and
            ready['members'][str(ready['pid'])]['start'] == ready['start'] and ready['listener_owners'] and
            all(str(pid) in ready['members'] for pid in ready['listener_owners']), 'Recorded model/server ownership mismatch')
    registry_gate({'data': list(read(EVIDENCE / 'server-registry.json').values())}, frozen)
    staged = read(EVIDENCE / 'staged-inputs.json')
    verify_staged(staged)
    runs = {}
    budget = frozen['evaluation_config']['effective_budget']
    ids = eligible_tasks(doc)
    canary = canary_run()['name']
    check = qualify_run(canary, BASE, [CANARY], ROOT / ('results/' + canary), ROOT / ('logs/' + canary + '.log'), True, budget)
    require(set(completed['canaries']) == {canary} and check['receipt'] == completed['canaries'][canary],
            'Canary/final completion disagreement')
    for run in frozen['future_runs']:
        name = run['name']
        if name not in completed['runs']:
            continue
        checked = qualify_run(name, run['adapter_alias'], ids, Path(run['result_dir']),
                              ROOT / ('logs/' + name + '.log'), False, budget)
        require(checked['receipt'] == completed['runs'][name], 'Run/final completion disagreement')
        runs[name] = checked
    require(set(completed['runs']) == set(runs) and (completed['capped'] is not None or len(runs) == len(frozen['future_runs'])),
            'Incomplete v064 generation')
    require(sha(EVIDENCE / 'records.json') == completed['records_sha256'], 'Run records changed')
    return doc, readout(), completed, runs


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k)


def partial_rows(path, ids):
    """Rows of rollouts that finished before an interruption: complete lines only, each checked like a full-run row."""
    from analyze_31b_recovery_screen_v034 import strict_json
    rows = {}
    for line in (Path(path).read_text().split(chr(10))[:-1] if Path(path).is_file() else []):
        if not line.strip():
            continue
        row = strict_json(line)
        task = row.get('instance_id')
        require(task in ids and task not in rows and type(row.get('resolved')) is bool and
                all(type(row.get(field)) is int and row[field] >= 0 for field in ('tool_calls', 'total_llm_calls', 'agent_patch_size')),
                'Interrupted-run row invalid: ' + str(task))
        rows[task] = row
    return rows


def trace_stats(path):
    trace = trace_schema(path)
    metrics = trace.get('final_metrics') or {}
    return dict(n_steps=len(trace['steps']), total_tokens=metrics.get('total_tokens'),
                prompt_tokens=metrics.get('total_prompt_tokens'), completion_tokens=metrics.get('total_completion_tokens'))


def run_record(run, receipt, rows):
    return dict(name=run['name'], seed=run['decoding_seed'], result_dir=receipt['result_dir'], results_sha256=receipt['sha256'],
                rows=rows, traces={task: value['sha256'] for task, value in receipt['traces'].items()}, interrupted=None, skipped={})


def interrupted_record(run, ids, interruption):
    """Completed rollouts of a cut run: rows written before the cut whose traces parse; recorded once beside the evidence."""
    output = Path(run['result_dir'])
    rows, kept, traces, skipped = partial_rows(output / 'task_results.jsonl', ids), {}, {}, {}
    for task, row in rows.items():
        path = output / 'traces' / ('trace_' + task + '.json')
        try:
            trace_schema(path)
        except (OSError, ValueError, RuntimeError) as error:
            skipped[task] = type(error).__name__
            continue
        kept[task], traces[task] = row, sha(path)
    results = output / 'task_results.jsonl'
    record = dict(name=run['name'], seed=run['decoding_seed'], result_dir=str(output),
                  results_sha256=sha(results) if results.is_file() else None, rows=kept, traces=traces,
                  interrupted=interruption, skipped=skipped)
    write_json(EVIDENCE / (run['name'] + '-interrupted.json'), record)
    return record


def kept_manifest(runtime_sha, records, ids, end):
    """Kept iff resolved and eligible; per-task cap 3 by lowest seed (capped rows flagged, uncapped counts kept); k_i per task."""
    entries, attempted, successes = [], dict.fromkeys(ids, 0), dict.fromkeys(ids, 0)
    order = sorted(records.values(), key=lambda record: record['seed'])
    for record in order:
        output = Path(record['result_dir'])
        require(record['results_sha256'] is None or sha(output / 'task_results.jsonl') == record['results_sha256'],
                'Run results changed after recording: ' + record['name'])
        for task in ids:
            row = record['rows'].get(task)
            if row is None:
                continue
            attempted[task] += 1
            if row['resolved'] is not True:
                continue
            trace, patch = output / 'traces' / ('trace_' + task + '.json'), output / 'patches' / (task + '.patch')
            require(sha(trace) == record['traces'][task], 'Kept trace changed after recording: ' + str(trace))
            if not (patch.is_file() and patch.stat().st_size == row['agent_patch_size'] > 0):
                require(record['interrupted'] is not None, 'Kept patch missing: ' + str(patch))
                continue
            successes[task] += 1
            entries.append(dict(task_id=task, seed=record['seed'], run=record['name'], repo=row['repo'], resolved=True,
                                trace_path=str(trace), trace_sha256=record['traces'][task], patch_path=str(patch),
                                patch_sha256=sha(patch), charged_calls=row['tool_calls'], llm_calls=row['total_llm_calls'],
                                duration_seconds=row['duration_seconds'], selected=successes[task] <= KEEP_PER_TASK,
                                interrupted=record['interrupted'], **trace_stats(trace)))
    after = [entry for entry in entries if entry['selected']]
    tasks_after = len({entry['task_id'] for entry in after})
    complete = all(attempted[task] == len(SEEDS) for task in ids)
    return dict(schema='onpolicy-kept-v064r2', generation='v064', revision='r2', runtime_sha256=runtime_sha, end=end, tasks=ids,
                eligible_sha256=sha(ELIGIBLE), canary_excluded=CANARY, keep_per_task=KEEP_PER_TASK,
                rule='kept iff resolved and the task is eligible; selected = the first 3 per task by lowest seed; unselected rows stay listed',
                kept=entries, counts=dict(uncapped=len(entries), selected=len(after), not_selected=len(entries) - len(after),
                                          distinct_tasks_selected=tasks_after),
                floor=dict(selected=len(after), distinct_tasks=tasks_after, min_selected=FLOOR_KEPT, min_tasks=FLOOR_TASKS,
                           met=len(after) >= FLOOR_KEPT and tasks_after >= FLOOR_TASKS),
                per_task={task: dict(attempted=attempted[task], successes=successes[task],
                                     selected=min(successes[task], KEEP_PER_TASK)) for task in ids},
                every_task_completed_ten_seeds=complete, insufficient_wording='k = 10' if complete else 'the attempted k_i',
                runs=[dict(name=r['name'], seed=r['seed'], completed_rollouts=len(r['rows']), interrupted=r['interrupted'],
                           skipped=r['skipped']) for r in order],
                pass_at_k={task: dict({'n': attempted[task], 'c': successes[task]},
                                      **{'pass@' + str(k): pass_at_k(attempted[task], successes[task], k) for k in (1, 5, 10)
                                         if k <= attempted[task]}) for task in ids},
                pass_at_k_note='descriptive generation yield only (unbiased estimator over the completed rollouts per task)')


def scale_gate(runtime_sha, records, seconds, ids, server_seconds):
    """Prereg v064-r2 scale gate from the pilot's own results: >= 5 distinct successful eligible tasks and a time fit."""
    rollouts = sum(len(record['rows']) for record in records.values())
    resolved = [task for record in records.values() for task, row in record['rows'].items() if row['resolved'] is True]
    tasks = sorted(set(resolved))
    per_rollout = sum(seconds.values()) / rollouts if rollouts else None
    scale_rollouts = len(MODES['scale']['seeds']) * len(ids)
    projected = None if per_rollout is None else per_rollout * scale_rollouts
    remaining = GENERATION_CAP - server_seconds
    passed = len(tasks) >= GATE_TASKS and projected is not None and projected <= remaining
    return dict(runtime_sha256=runtime_sha, distinct_successful_tasks=len(tasks), successful_task_ids=tasks, min_tasks=GATE_TASKS,
                resolved_rollouts=len(resolved), rollouts_completed=rollouts, data_run_seconds=sum(seconds.values()),
                seconds_per_rollout=per_rollout, scale_rollouts=scale_rollouts, projected_scale_seconds=projected,
                pilot_server_seconds=server_seconds, remaining_cap_seconds=remaining, passed=passed,
                rule='pass iff distinct successful tasks >= 5 and pilot seconds per rollout x scale rollouts <= 7 h - pilot server seconds',
                sentence=None if passed else 'insufficient on-policy yield in a 3-seed pilot (%d kept, %d tasks)' % (len(resolved), len(tasks)))


def gate_passed(runtime_sha):
    """Stage 1b needs the pilot completed and its gate passed under the same PINNED runtime."""
    folder = MODES['pilot']['package'] / 'evaluation'
    require(GATE.is_file() and (folder / 'completed.json').is_file() and not (folder / 'stopped.json').exists(),
            'Pilot not completed; scale refused')
    gate, completed = read(GATE), read(folder / 'completed.json')
    require(gate['runtime_sha256'] == runtime_sha and gate['passed'] is True and completed['runtime_sha256'] == runtime_sha and
            completed['records_sha256'] == gate['records_sha256'] == sha(folder / 'records.json'),
            'Pilot gate not passed under this runtime; scale refused')
    return dict(gate_sha256=sha(GATE), pilot_server_seconds=gate['pilot_server_seconds'], records_sha256=gate['records_sha256'])


def finish_stage(runtime_sha, ids, records, seconds, server_seconds, capped):
    """Pilot: gate.json, plus the final kept manifest only when the gate fails. Scale: the final kept manifest (pilot + scale)."""
    if MODE == 'pilot':
        gate = dict(scale_gate(runtime_sha, records, seconds, ids, server_seconds), records_sha256=sha(EVIDENCE / 'records.json'),
                    capped=capped)
        write_json(GATE, gate)
        if gate['passed']:
            return dict(gate_sha256=sha(GATE), passed=True, kept_manifest=None)
        write_json(KEPT, kept_manifest(runtime_sha, records, ids, dict(stage='pilot', reason=gate['sentence'])))
        return dict(gate_sha256=sha(GATE), passed=False, kept_manifest=dict(path=str(KEPT), sha256=sha(KEPT)))
    pilot = MODES['pilot']['package'] / 'evaluation/records.json'
    require(sha(pilot) == read(GATE)['records_sha256'], 'Pilot records changed after the gate')
    write_json(KEPT, kept_manifest(runtime_sha, dict(read(pilot), **records), ids,
                                   dict(stage='scale', reason=capped['reason'] if capped else 'all scale runs completed')))
    return dict(kept_manifest=dict(path=str(KEPT), sha256=sha(KEPT)))


def may_start(now, stage_deadline):
    """Prereg v064-r2: a scale run starts only if the remaining generation cap is at least the per-run cap; a pilot run
    starts while its 2 h cap has a minute left (the 90-min start rule is stated for Stage 1b only)."""
    return stage_deadline - now >= MODES[MODE]['start_minimum']


def cap_reached(now, case_end):
    """A failed run counts as capped only when it ended at its RuntimeMaxSec bound."""
    return now >= case_end - CAP_MARGIN


def owner_unit(pid):
    """Systemd unit that owns a process (from /proc/<pid>/cgroup); used for preflight attribution only."""
    try:
        parts = Path('/proc/' + str(pid) + '/cgroup').read_text().strip().split('/')
    except OSError:
        return 'pid ' + str(pid)
    units = [part for part in parts if part.endswith(('.service', '.scope'))]
    return units[-1] if units else 'pid ' + str(pid)


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
        if (scripts & {'train_lora.py', 'run_eval.py'} and tied) or 'train_mask_control_v035.py' in scripts:
            offenders.append(dict(pid=value['pid'], start=value['start'], cwd=value['cwd'], argv=value['argv']))
    return offenders


def preflight():
    """CPU-only gate report before activation; reads no outcome and takes no lock."""
    recovery = load_recovery(ROOT)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=15).stdout.split()
    apps = [line for line in subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=15).stdout.splitlines() if line.strip()]
    port = sum(line.split()[1] == '0100007F:1F40' and line.split()[3] == '0A'
               for line in Path('/proc/net/tcp').read_text().splitlines()[1:])
    locks = Path('/proc/locks').read_text()
    held = [str(path) for path in LOCKS if ':' + str(path.stat().st_ino) + ' ' in locks]
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].split()[0] for line in Path('/proc/meminfo').read_text().splitlines()}
    available = int(fields['MemAvailable']) * 1024
    collisions = [str(path) for path in destinations(effective_readout(readout())) if path.exists()]
    offenders = idle_offenders(recovery)
    owners = {}
    for line in apps:
        owners.setdefault(owner_unit(line.split(',')[0].strip()), []).append('GPU')
    for line in locks.splitlines():
        parts = line.split()
        for path in LOCKS:
            if str(path) in held and len(parts) > 5 and parts[5].endswith(':' + str(path.stat().st_ino)):
                owners.setdefault(owner_unit(parts[4]), []).append(path.name)
    attributed = {name for what in owners.values() for name in what}
    if MODE == 'pilot':
        prior_ok, missing = ELIGIBLE.is_file() and read(ELIGIBLE)['stop'] is False, 'Stage-0 eligible list missing or below 10 tasks'
    else:
        prior_ok, missing = GATE.is_file() and read(GATE)['passed'] is True, 'Pilot scale gate not passed yet'
    blockers = [('GPU busy (' if 'GPU' in what else 'Lease held (') + unit + ': ' + ', '.join(what) + ')'
                for unit, what in sorted(owners.items())]
    checks = [(not gpu, 'GPU query failed'), (bool(gpu) and not apps and int(gpu[0]) > 100, 'GPU memory in use without a compute app'),
              (port != 0, 'Port 8000 listener present'), (bool(offenders), 'Idle-check offenders present'),
              (bool(collisions), 'Destinations exist: ' + ', '.join(collisions[:3])),
              (available < MEMORY_CORRECTION['host_min_available_bytes'], 'Host MemAvailable below 64 GiB'),
              (sha(PREVIOUS_STOPPED) != PREVIOUS_STOPPED_SHA, 'v048 stopped receipt changed'), (not prior_ok, missing)]
    blockers += [text for failed, text in checks if failed]
    blockers += ['Lease held (unattributed): ' + Path(path).name for path in held if Path(path).name not in attributed]
    ready = (bool(gpu) and int(gpu[0]) <= 100 and not apps and port == 0 and not held and not offenders and not collisions and
             available >= MEMORY_CORRECTION['host_min_available_bytes'] and sha(PREVIOUS_STOPPED) == PREVIOUS_STOPPED_SHA and prior_ok)
    return dict(ready=ready, mode=MODE, blockers=blockers, gpu_memory_used_mib=int(gpu[0]) if gpu else None,
                gpu_compute_apps=apps, port_8000_listeners=port,
                locks_held=held, mem_available_bytes=available, destination_collisions=collisions,
                idle_offenders=offenders, failed_units=failed_units(), previous_stopped_intact=sha(PREVIOUS_STOPPED) == PREVIOUS_STOPPED_SHA)


def activate(runtime_sha, controls_sha=None, replay_sha=None):
    doc, frozen = verify_runtime(runtime_sha)
    require(doc['status'] == 'PINNED' and doc['eligible']['sha256'] is not None and
            doc['eligible']['sha256'] == doc['pins'].get(str(ELIGIBLE)), 'Activation requires the PINNED runtime with the eligible list')
    ids = eligible_tasks(doc)
    prerequisite = gate_passed(runtime_sha) if MODE == 'scale' else None
    prior = prerequisite['pilot_server_seconds'] if prerequisite else 0
    admission = admission_receipts(doc, controls_sha, replay_sha)
    recovery = load_recovery(ROOT)
    require(all(not path.exists() for path in destinations(frozen)), 'Execution destination collision')
    with (ROOT / 'migration/recovery-screen-v034/evaluation.lock').open('a') as old_lease, \
         (training.EVIDENCE.parent / 'training.lock').open('a') as train_lease:
        fcntl.flock(old_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(train_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, deadline = time.monotonic(), time.monotonic() + TOTAL
        EVIDENCE.mkdir(parents=True, exist_ok=False)
        canaries, runs, records, seconds, capped, current, ready_at = {}, {}, {}, {}, None, None, None
        try:
            verify_runtime(runtime_sha)
            admission_receipts(doc, controls_sha, replay_sha)
            training.assert_idle(recovery)
            memory = host_memory_admission()
            cursor = kernel_cursor()
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
            require(not gpu.strip(), 'GPU compute processes present; evaluation refused')
            require(not any(line.split()[1] == '0100007F:1F40' and line.split()[3] == '0A'
                    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]), 'Serving listener already occupied')
            train_tasks()
            snapshots = verify_snapshots(doc, ids)
            write_json(EVIDENCE / 'activation.json', dict(runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                       started_at=recovery.utc(), total_seconds=TOTAL, generation_cap_seconds=GENERATION_CAP,
                       pilot_cap_seconds=PILOT_CAP, run_cap_seconds=RUN_CAP, mode=MODE, task_ids=ids, gate=prerequisite,
                       prior_server_seconds=prior, snapshot_check=snapshots,
                       generation='v064', memory_correction=MEMORY_CORRECTION, host_memory=memory, kernel_cursor=cursor,
                       admission_receipts=admission, user_manager_failed_units=failed_units(),
                       failed_units_policy='record only; historical failed units do not block (team lead, 2026-11-05)'))
            staged = stage_submissions(frozen)
            write_json(EVIDENCE / 'staged-inputs.json', staged)
            helper = recovery.load_helper(ROOT)
            env = serving.environment(doc)
            launched, ready = start_server(recovery, helper, env, frozen, deadline)
            ready_at, stage_started = time.monotonic(), recovery.utc()
            cap = PILOT_CAP if MODE == 'pilot' else GENERATION_CAP - prior
            stage_deadline = min(ready_at + cap, deadline - 2 * RESERVE)
            budget = frozen['evaluation_config']['effective_budget']
            cases = [(canary_run(), True)] + [(run, False) for run in frozen['future_runs']]
            done = []
            for run, strict in cases:
                require(time.monotonic() < deadline - RESERVE, 'Evaluation launch deadline barrier')
                if not strict and not may_start(time.monotonic(), stage_deadline):
                    capped = dict(reason='remaining cap below the start minimum; run not started', run=run['name'])
                    break
                verify_runtime(runtime_sha)
                verify_staged(staged)
                training.assert_idle(recovery)
                server_live(recovery, launched, ready)
                registry_gate(helper.api('models'), frozen)
                global_oom_check(cursor)
                name, output = run['name'], Path(run['result_dir'])
                log = ROOT / ('logs/' + name + '.log')
                selected = [CANARY] if strict else eligible_tasks(doc)
                require(selected == ([CANARY] if strict else ids) and not any(task.startswith('rich_') for task in selected),
                        'Refuse a holdout or non-eligible task id')
                command = helper.eval_command(Path(run['submission_dir']), output, selected, budget, 1 if strict else None)
                guarded = [command[0], GUARD, '--guard-receipt', EVIDENCE / (log.name + '.memory-guard.json'), '--', *command[1:]]
                case_start = time.monotonic()
                case_end = min(case_start + RUN_CAP, stage_deadline)
                bounded = services.isolated_helper(helper, MemoryBoundedService(case_end + RESERVE, EVIDENCE))
                current = None if strict else run
                try:
                    route = recovery.observe_run(bounded, run['adapter_alias'], guarded, log, env, ROOT / 'data/competition', strict=strict)
                except Exception as error:
                    end = EVIDENCE / (log.name + '.service-end.json')
                    if strict or not cap_reached(time.monotonic(), case_end) or not (end.is_file() and read(end)['cgroup_empty'] is True):
                        raise
                    capped = dict(reason='per-run, pilot or generation cap reached; completed rollouts kept with this interruption',
                                  run=name, error=type(error).__name__ + ': ' + str(error), unit_result=read(end)['state'].get('Result'))
                    seconds[name] = time.monotonic() - case_start
                    records[name], current = interrupted_record(run, ids, capped), None
                    break
                oom = global_oom_check(cursor)
                write_json(EVIDENCE / (name + '-oom-check.json'), oom)
                rows = (recovery.validate_canary_result(output / 'task_results.jsonl', CANARY, budget) if strict else
                        recovery.validate_rows(output / 'task_results.jsonl', selected))
                receipt = dict(rows, alias=run['adapter_alias'], arm=run['arm'], decoding_seed=run['decoding_seed'],
                    generation='v064', scientific_name=run['scientific_name'], result_dir=str(output),
                    memory_guard=guard_qualification(EVIDENCE / (log.name + '.memory-guard.json')), global_oom_check=oom,
                    task_ids=selected, strict_canary=strict, routing=route,
                    route_sha256=sha(Path(str(log) + '.routing.json')), service=qualify_service(EVIDENCE, log.stem),
                    traces=services.trace_availability(output, selected, require_tools=strict),
                    runtime_sha256=runtime_sha, staged_sha256=sha(EVIDENCE / 'staged-inputs.json'))
                write_json(EVIDENCE / (name + '-complete.json'), receipt)
                checked = qualify_run(name, run['adapter_alias'], selected, output, log, strict, budget)
                (canaries if strict else runs)[run['name']] = checked['receipt']
                if not strict:
                    seconds[name] = time.monotonic() - case_start
                    records[name] = run_record(run, checked['receipt'], checked['rows'])
                current = None
                done.append((run, strict))
                server_live(recovery, launched, ready)
            live = server_live(recovery, launched, ready)
            serving.stop_server(recovery, live, grace_seconds=min(60, max(1, deadline - time.monotonic())))
            require(not serving.group_members(recovery, launched['group']), 'Evaluation server group remains')
            server_seconds = time.monotonic() - ready_at
            write_json(EVIDENCE / 'server-cleanup.json', dict(status='owned_server_exited', group_empty=True,
                       pid=launched['pid'], start=launched['start'], group=launched['group'], method='verified_TERM_only',
                       server_seconds=server_seconds))
            verify_runtime(runtime_sha)
            verify_staged(staged)
            for run, strict in done:
                qualify_run(run['name'], run['adapter_alias'], [CANARY] if strict else ids, Path(run['result_dir']),
                            ROOT / ('logs/' + run['name'] + '.log'), strict, budget)
            verify_snapshots(doc, ids)
            require(time.monotonic() <= deadline, 'Evaluation total deadline exceeded')
            write_json(EVIDENCE / 'global-oom-final.json', global_oom_check(cursor))
            write_json(EVIDENCE / 'records.json', records)
            outcome = finish_stage(runtime_sha, ids, records, seconds, server_seconds, capped)
            hashes = {p.name: sha(p) for p in EVIDENCE.glob('*.json')}
            write_json(EVIDENCE / 'completed.json', dict(status='stage_capped_completed_runs_qualified' if capped else MODES[MODE]['status'],
                mode=MODE, runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                generation='v064', memory_correction=MEMORY_CORRECTION,
                admission_receipts=admission, cleanup_completed=True, gate_prerequisite=prerequisite,
                runs=runs, canaries=canaries, capped=capped, stage_started_at=stage_started,
                runs_incomplete=[item['name'] for item in frozen['future_runs'] if item['name'] not in runs],
                server_seconds=server_seconds, prior_server_seconds=prior, run_wall_seconds=seconds,
                records_sha256=sha(EVIDENCE / 'records.json'), outcome=outcome, evidence_hashes=hashes,
                elapsed_seconds=time.monotonic() - started, analysis_pending=True, submission_started=False))
        except BaseException as error:
            cleanup = read(EVIDENCE / 'server-cleanup.json') if (EVIDENCE / 'server-cleanup.json').exists() else dict(status='server_not_launched')
            if (EVIDENCE / 'server-launched.json').exists() and not (EVIDENCE / 'server-cleanup.json').exists():
                launched = read(EVIDENCE / 'server-launched.json')
                try:
                    result = services.cleanup31(recovery, launched, deadline)
                    cleanup = dict(status='owned_server_exited', method='reviewed_cleanup31_TERM_only',
                                   group_empty=not serving.group_members(recovery, launched['group']))
                except BaseException as cleanup_error:
                    cleanup = dict(status='cleanup_failed', reason=str(cleanup_error), launched=launched,
                                   residual_members=services.normalize(serving.group_members(recovery, launched['group'])))
            try:
                offenders = idle_offenders(recovery)
            except BaseException as probe_error:
                offenders = 'unavailable: ' + type(probe_error).__name__ + ': ' + str(probe_error)
            try:
                if current is not None and current['name'] not in records:
                    records[current['name']] = interrupted_record(current, ids, dict(reason='stopped', run=current['name'],
                                                                                    error=type(error).__name__ + ': ' + str(error)))
                if records and not (EVIDENCE / 'records.json').exists():
                    write_json(EVIDENCE / 'records.json', records)
                if records and not KEPT.exists():
                    pilot = MODES['pilot']['package'] / 'evaluation/records.json'
                    merged = dict(read(pilot), **records) if MODE == 'scale' else records
                    write_json(KEPT, kept_manifest(runtime_sha, merged, ids, dict(stage=MODE, reason='stopped: ' + type(error).__name__)))
                salvage = dict(records=sorted(records), kept_manifest=dict(path=str(KEPT), sha256=sha(KEPT)) if KEPT.exists() else None)
            except BaseException as salvage_error:
                salvage = 'not completed: ' + type(salvage_error).__name__ + ': ' + str(salvage_error)
            write_json(EVIDENCE / 'stopped.json', dict(type=type(error).__name__, reason=str(error), cleanup=cleanup,
                       elapsed_seconds=time.monotonic() - started, no_automatic_retry=True, idle_offenders_at_stop=offenders,
                       server_seconds_until_stop=None if ready_at is None else time.monotonic() - ready_at, salvage=salvage))
            raise


def prepare():
    require(not DRAFT.exists() and not Path(str(DRAFT) + '.sha256').exists(), 'Runtime destination collision')
    readout()
    sound = train_tasks()
    require(sha(SOURCE_RUNNER) == SOURCE_RUNNER_SHA and sha(R1_RUNNER) == R1_RUNNER_SHA and sha(PREREG) == PREREG_SHA and
            sha(STAGE0) == STAGE0_SHA and sha(ADDENDUM) == ADDENDUM_SHA and re.search('^status: FROZEN', PREREG.read_text(), re.M) is not None,
            'Source runner, r1 runner, frozen prereg r2, addendum or Stage-0 contract changed')
    require(read(STAGE0)['candidate_ids'] == list(CANDIDATES), 'Stage-0 candidates differ from this runner')
    require(sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == GUARD_SOURCE_SHA and
            sha(GUARD) == GUARD_WRAPPER_SHA, 'Guard sources are not the independently approved versions')
    with (ROOT / 'results/controls_train/controls.csv').open() as stream:
        controls = [row for row in csv.DictReader(stream) if row['instance_id'] == CANARY]
    require(len(controls) == 1 and controls[0]['gold_resolved'] == 'True' and
            controls[0]['empty_resolved'] == 'False' and controls[0]['empty_exit_code'] == '1', 'Fixed canary controls changed')
    snapshots = {str(snapshot(task)): sha(snapshot(task)) for task in CANDIDATES}
    listing = hashlib.sha256(''.join(task + chr(10) for task in CANDIDATES).encode()).hexdigest()
    repos = lambda ids: {repo: sum(t.split('_')[0] == repo for t in ids) for repo in sorted({t.split('_')[0] for t in ids})}
    names = ['tools/run_onpolicy_gen_v064r2.py', 'tools/test_run_onpolicy_gen_v064r2.py', 'migration/onpolicy-v064/r2/runner.diff',
        'migration/onpolicy-v064/r2/runner-vs-r1.diff', 'migration/onpolicy-v064/r2/design.json',
        'tools/run_onpolicy_gen_v064.py', 'tools/run_controls_capped_v064b.py', 'contracts/onpolicy-controls-v064b.json',
        'contracts/onpolicy-controls-v064b.json.sha256',
        'tools/run_a0_capped_v052.py', 'tools/test_run_a0_capped_v052.py', 'migration/a0-capped-v052/runner.diff',
        'contracts/a0-capped-v052-runtime-DRAFT.json', 'contracts/a0-capped-v052-runtime-DRAFT.json.sha256',
        'migration/a0-capped-v052/runner-v048-to-v052.diff', 'migration/a0-capped-v048/evaluation/stopped.json',
        'tools/run_mask_evaluations_v043.py', 'tools/test_run_mask_evaluations_v043.py',
        'contracts/mask-evaluations-v043-runtime-DRAFT.json', 'contracts/mask-evaluations-v043-runtime-DRAFT.json.sha256',
        'tools/run_eval_memory_guard_v043.py', 'tools/sandbox_memory_guard_v043.py', 'tools/test_sandbox_memory_guard_v043.py',
        'tools/run_mask_control_v0351.py', 'tools/test_run_mask_control_v0351.py',
        'tools/run_31b_recovery_screen_v034.py', 'tools/test_run_31b_recovery_screen_v034.py',
        'tools/run_v032_probes.py', 'tools/prepare_v032_probes.py', 'tools/recover_eval_v025.py',
        'tools/overnight_v02.py', 'tools/analyze_contract_v02_v025.py', 'tools/mechanism_metrics_v032.py',
        'tools/analyze_31b_recovery_screen_v034.py', 'tools/test_analyze_31b_recovery_screen_v034.py',
        'contracts/mask-control-v035-readout.json', 'contracts/mask-control-v035-readout.json.sha256',
        'data/competition/snapshots/' + CANARY + '.tgz', 'results/controls_train/controls.csv',
        'splits/v1.json', 'data/competition/tasks.jsonl', 'serve_vllm.py']
    names += ['subs/e4b_budgetfit_s' + str(seed) + '/configs/sampling.yaml' for seed in SEEDS]
    pins = {str(ROOT / name): sha(ROOT / name) for name in names}
    require(all(pins[str(ROOT / 'subs' / ('e4b_budgetfit_s' + str(seed)) / 'configs/sampling.yaml')] == SAMPLING_SHA[seed]
                for seed in SEEDS), 'Seed sampling configs differ from the pinned shas')
    require(pins[str(V043_RUNTIME)] == V043_RUNTIME_SHA, 'Source v043 runtime contract changed')
    require(pins[str(PREVIOUS_STOPPED)] == PREVIOUS_STOPPED_SHA, 'Previous stopped v048 generation changed')
    package = Path('HOME_DIR/miniconda3/envs/swegemma/lib/python3.12/site-packages')
    for name in ['swegemma/cli.py', 'adk_submission/discovery.py', 'adk_submission/server.py',
        'adk_submission/compiler.py', 'adk_submission/resolvers/models.py', 'adk_submission/resolvers/generation.py',
        'swegemma/evaluate.py', 'swegemma/results.py', 'swegemma/harness/verification.py',
        'swegemma/sandbox/subprocess.py',
        'adk_eval_core/tracing/atif.py', 'adk_eval_core/tracing/atif_converter.py', 'adk_eval_core/tracing/trace.py',
        'vllm/entrypoints/openai/models/serving.py']:
        pins[str(package / name)] = sha(package / name)
    pins.update(read(ROOT / 'contracts/recovery-screen-v034-plan.json')['runtime_pins'])
    pins.update(read(ROOT / 'contracts/recovery-screen-v034-runtime.json')['installed_pins'])
    pins['/usr/bin/journalctl'] = sha('/usr/bin/journalctl')
    pins[str(PREREG)], pins[str(ADDENDUM)] = PREREG_SHA, ADDENDUM_SHA
    doc = dict(status='DRAFT_PREPARED_ONLY', root=str(ROOT), total_seconds=TOTAL, cleanup_seconds=RESERVE, revision='r2',
        generation_cap_seconds=GENERATION_CAP, pilot_cap_seconds=PILOT_CAP, run_cap_seconds=RUN_CAP,
        cap_rule='pilot: 2 h from its server-ready, a run starts while a minute is left; generation cap 7 h = pilot server seconds + '
                 'scale server seconds (server-ready to stop only); a scale run starts only if the remaining cap is at least 90 min; '
                 'per-run cap 90 min; completed rollouts of a cut run are kept and flagged; any stop keeps the completed rollouts',
        generation='v064', generation_runs={mode: generation_runs(mode) for mode in MODES},
        canary_runs={mode: canary_run(mode) for mode in MODES}, memory_correction=MEMORY_CORRECTION,
        candidate_ids=list(CANDIDATES), snapshot_sha256=snapshots,
        candidates=dict(count=len(CANDIDATES), list_sha256=listing, per_repo=repos(CANDIDATES), canary_excluded=CANARY,
                        split=str(SPLIT), split_sha256=SPLIT_SHA, sound_tasks=len(sound), sound_per_repo=repos(sound),
                        snapshots_present=str(len(sound)) + '/70', guard='no rich_ id, no holdout id, canary never a data task'),
        stage0=dict(path=str(STAGE0), sha256=STAGE0_SHA), prereg=dict(path=str(PREREG), sha256=PREREG_SHA),
        addendum=dict(path=str(ADDENDUM), sha256=ADDENDUM_SHA),
        eligible=dict(path=str(ELIGIBLE), sha256=None, task_ids=None, filled_at='PIN, after Stage 0 wrote eligible.json'),
        selection=dict(rule='kept iff resolved and eligible', keep_per_task=KEEP_PER_TASK, order='lowest seeds first',
                       floor_selected=FLOOR_KEPT, floor_distinct_tasks=FLOOR_TASKS, min_eligible=MIN_ELIGIBLE),
        scale_gate=dict(min_distinct_successful_tasks=GATE_TASKS, time_fit='pilot seconds per rollout x 7 x eligible <= 7 h - pilot server seconds',
                        file=str(GATE)), kept_manifest=str(KEPT),
        modes={mode: dict(evidence=str(spec['package'] / 'evaluation'), server_manifest=str(spec['package'] / 'server-manifest'),
                          seeds=list(spec['seeds']), completed_status=spec['status'], start_minimum_seconds=spec['start_minimum'])
               for mode, spec in MODES.items()},
        guard_source_sha256=GUARD_SOURCE_SHA, guard_wrapper_sha256=GUARD_WRAPPER_SHA,
        required_controls_receipt=str(CONTROLS_RECEIPT), required_replay_receipt=str(REPLAY_RECEIPT),
        readout_sha256=READOUT_SHA, pins=pins,
        directories={str(path): inventory(path) for path in
                     [ROOT / 'models' / BASE] + [ROOT / 'subs' / ('e4b_budgetfit_s' + str(seed)) for seed in SEEDS]},
        server_environment=read(V043_RUNTIME)['server_environment'], canary_task=CANARY,
        aliases=[BASE], lora='off: adapter-free manifest gives serve_vllm enable_lora=False and no LoRA flags, as the sealed A0 server',
        sampling_sha256={str(seed): SAMPLING_SHA[seed] for seed in SEEDS},
        runner_diff=dict(path=str(RUNNER_DIFF), sha256=pins[str(RUNNER_DIFF)],
                         source_runner=str(SOURCE_RUNNER), source_runner_sha256=SOURCE_RUNNER_SHA),
        previous_stopped_generation=dict(generation='v048', stopped_receipt=str(PREVIOUS_STOPPED), sha256=PREVIOUS_STOPPED_SHA,
            cause='inferred from timing: the pinned idle check matched another agent provenance-hashing argv naming run_eval.py/train_lora.py; 3/3 canaries qualified, 0 runs'),
        server_manifest=str(MANIFEST), run_order={mode: [run['name'] for run in generation_runs(mode)] for mode in MODES},
        template='serve_vllm.py default template (no --chat-template), as the sealed v052 A0cap server; prereg v064-r2 A_onpol',
        canary_rule='each stage starts with one strict fastapi_11194 canary on a seed-1 copy of the s1 configuration; never kept',
        policy='pilot: canary then seeds 1-3 on the eligible list, then gate.json; scale (only after a passed gate): canary then '
               'seeds 4-10; the kept manifest is written once, by the stage that ends the generation; no conversion or training',
        server_handoff='Verified TERM-only shutdown; qualification uses immutable recorded lifecycle, not server liveness',
        gpu_operations_during_preparation=False)
    write_json(DRAFT, doc)
    digest = sha(DRAFT)
    with Path(str(DRAFT) + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + DRAFT.name + '\n')
    return digest


def pin(draft_sha, write=False):
    """PINNED = the DRAFT plus the Stage-0 eligible list (sha and ids); without write it prints the predicted sha and writes nothing."""
    doc, _ = verify_runtime(draft_sha, DRAFT)
    require(doc['status'] == 'DRAFT_PREPARED_ONLY' and doc['eligible']['sha256'] is None, 'Pin starts from the prepared DRAFT')
    require(ELIGIBLE.is_file(), 'Stage-0 eligible list missing: ' + str(ELIGIBLE))
    eligible = read(ELIGIBLE)
    ids, receipt = eligible['eligible_task_ids'], Path(eligible['receipt']['path'])
    require(eligible['schema'] == 'onpolicy-v064-eligible-1' and eligible['contract_sha256'] == STAGE0_SHA and
            eligible['prereg']['sha256'] == PREREG_SHA and sha(receipt) == eligible['receipt']['sha256'] and
            eligible['stop'] is False and len(ids) == len(set(ids)) >= MIN_ELIGIBLE and set(ids) <= set(CANDIDATES) and
            [task for task in CANDIDATES if task in ids] == ids and
            hashlib.sha256(''.join(task + chr(10) for task in ids).encode()).hexdigest() == eligible['eligible_list_sha256'],
            'Eligible list does not admit generation')
    pinned = json.loads(json.dumps(doc))
    pinned.update(status='PINNED', pinned_from_draft_sha256=draft_sha,
                  eligible=dict(path=str(ELIGIBLE), sha256=sha(ELIGIBLE), task_ids=ids, count=len(ids),
                                list_sha256=eligible['eligible_list_sha256'], receipt=eligible['receipt']))
    pinned['pins'][str(ELIGIBLE)] = sha(ELIGIBLE)
    pinned['pins'][str(receipt)] = eligible['receipt']['sha256']
    predicted = hashlib.sha256((json.dumps(pinned, indent=2, sort_keys=True) + chr(10)).encode()).hexdigest()
    if write:
        require(not RUNTIME.exists() and not Path(str(RUNTIME) + '.sha256').exists(), 'Pinned runtime destination collision')
        write_json(RUNTIME, pinned)
        require(sha(RUNTIME) == predicted, 'Pinned runtime bytes differ from the prediction')
        with Path(str(RUNTIME) + '.sha256').open('x') as stream:
            stream.write(predicted + '  ' + RUNTIME.name + chr(10))
    return dict(status='pin_written' if write else 'pin_dry_run', draft_sha256=draft_sha, pinned_sha256=predicted,
                eligible_sha256=sha(ELIGIBLE), eligible_count=len(ids), written=write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--pin', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--mode', choices=sorted(MODES))
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--controls-receipt-sha')
    parser.add_argument('--replay-receipt-sha')
    args = parser.parse_args()
    require(args.mode is not None or not (args.activate or args.preflight), 'Activation and preflight need --mode pilot or scale')
    require(args.pin or not args.write, '--write belongs to --pin')
    if args.mode is not None:
        select_mode(args.mode)
    if args.preflight:
        require(not (args.prepare or args.activate or args.pin), 'Preflight is read-only and separate')
        report = preflight()
        print(json.dumps(report, indent=1))
        sys.exit(0 if report['ready'] else 3)
    if args.prepare:
        require(not (args.activate or args.pin), 'Preparation cannot activate')
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(), gpu_operations=False)))
    elif args.pin:
        require(args.runtime_sha and not args.activate, 'Pin needs the DRAFT runtime SHA and cannot activate')
        print(json.dumps(pin(args.runtime_sha, args.write)))
    elif args.activate:
        require(args.runtime_sha, 'Activation requires runtime SHA')
        activate(args.runtime_sha, args.controls_receipt_sha, args.replay_receipt_sha)
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        pinned = RUNTIME.exists() and sha(RUNTIME) == args.runtime_sha
        verify_runtime(args.runtime_sha, RUNTIME if pinned else DRAFT)
        print(json.dumps(dict(status='pinned_verified' if pinned else 'draft_verified', gpu_operations=False)))


if __name__ == '__main__':
    main()
