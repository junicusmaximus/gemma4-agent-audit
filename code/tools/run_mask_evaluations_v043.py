"""Fresh v043 rerun of all six frozen scientific evaluations with bounded sandbox memory."""
from __future__ import annotations

import argparse
import csv
import fcntl
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
RUNTIME = ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json'
EVIDENCE = ROOT / 'migration/mask-evaluations-v043/evaluation'
MANIFEST = ROOT / 'subs/v043_mask_manifest'
GUARD = ROOT / 'tools/run_eval_memory_guard_v043.py'
GUARD_SOURCE_SHA = '5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GUARD_WRAPPER_SHA = 'c04928a3406c1175098f1217dd7b955dcec61aec1147537fbd68bf59de8bbbd5'
CONTROLS_RECEIPT = ROOT / 'analysis/oom-controls-v043/final48/report.json'
REPLAY_RECEIPT = ROOT / 'analysis/oom-replay-v043/report.json'
PRESERVATION_SEAL = ROOT / 'analysis/oom-recovery-v043/prior-v039-seal.json'
PRESERVATION_SHA = '0d39aa152757b67f7070ddb786633febed262792e228a2de18cb20557b7367e8'
TRAIN_RUNTIME_SHA = '84453f87abb796603c1b9d8fc5f2e27bc3535b36c1fe24b580430201eca054e1'
TRAIN_PID, TRAIN_START = 3016915, '448966420'
TOTAL, RESERVE = 28800, 600
CANARY = 'fastapi_11194'
MEMORY_CORRECTION = dict(version='v043_sandbox_cgroup_v1',
    per_sandbox=dict(memory_max=4294967296, memory_swap_max=0),
    service=dict(MemoryMax=51539607552, MemorySwapMax=0, Delegate='memory', DelegateSubgroup='supervisor', OOMPolicy='continue'),
    outer_memory_events_local_zero=['max', 'oom', 'oom_kill', 'oom_group_kill'],
    host_min_available_bytes=68719476736,
    scope='Commands routed through SubprocessManager.exec: agent, verification pytest, and setup commands using that path',
    exceptions='Direct manager.start git/materialization and other direct subprocess host helpers remain under the outer 48GiB service cap only',
    comparability='Frozen numerical thresholds retained descriptively; this infrastructure correction is not evidence of historical exchangeability',
    global_oom_policy='No kernel CONSTRAINT_NONE events after admission journal cursor through final cleanup')


def generation_runs(frozen):
    runs = json.loads(json.dumps(frozen['future_runs']))
    for run in runs:
        run['scientific_name'] = run['name']
        run['name'] = run['name'].replace('v035_', 'v043_', 1)
        run['submission_dir'] = str(ROOT / 'subs' / run['name'])
        run['result_dir'] = str(ROOT / 'results' / run['name'])
        run['trace_dir'] = str(Path(run['result_dir']) / 'traces')
    return runs


def effective_readout(frozen):
    effective = json.loads(json.dumps(frozen))
    effective['future_runs'] = generation_runs(frozen)
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
        unit = 'gemma-v043-eval-' + uuid.uuid4().hex + '.service'
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


def verify_preservation_seal():
    require(sha(PRESERVATION_SEAL) == PRESERVATION_SHA, 'Prior v039 preservation seal changed')
    for name, metadata in read(PRESERVATION_SEAL)['files'].items():
        require(sha(name) == metadata['sha256'] and Path(name).stat().st_size == metadata['bytes'],
                'Prior v039 scientific artifact changed: ' + name)


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


def verify_runtime(expected):
    doc = training.checked_json(RUNTIME, expected)
    require(Path(str(RUNTIME) + '.sha256').read_text().split()[0] == expected, 'Runtime sidecar mismatch')
    require(doc['root'] == str(ROOT) and doc['total_seconds'] == TOTAL and doc['cleanup_seconds'] == RESERVE and
            doc['generation'] == 'v043' and doc['training_controller'] == dict(pid=TRAIN_PID, start=TRAIN_START), 'Runtime identity/bounds changed')
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Runtime/package pin changed: ' + name)
    for name, files in doc['directories'].items():
        require(inventory(name) == files, 'Runtime inventory changed: ' + name)
    verify_preservation_seal()
    frozen = readout()
    require(doc['generation_runs'] == generation_runs(frozen) and doc['memory_correction'] == MEMORY_CORRECTION,
            'Scientific mapping or memory correction changed')
    train_doc, draft = training.verify_runtime(TRAIN_RUNTIME_SHA)
    training.approval(training.APPROVED, doc['training_approval_sha256'], draft)
    return doc, effective_readout(frozen)


def qualify_service(folder, name, expected_end=None):
    start_path = folder / (name + '.log.service-start.json')
    end_path = folder / (name + '.log.service-end.json')
    if expected_end is not None:
        require(sha(end_path) == expected_end, 'Service receipt hash changed')
    start, end = read(start_path), read(end_path)
    identity = {key: start[key] for key in ('InvocationID', 'ControlGroup')}
    prefix = 'v043' if folder == EVIDENCE else 'v034'
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


def qualified_training(doc):
    folder = training.EVIDENCE
    require(not (folder / 'stopped.json').exists(), 'Training stopped; evaluation refused')
    done_path = folder / 'completed.json'
    require(done_path.is_file(), 'Training completion unavailable')
    done = read(done_path)
    require(done['status'] == 'training_only_qualified' and done['runtime_sha256'] == TRAIN_RUNTIME_SHA and
            done['approval_sha256'] == doc['training_approval_sha256'] and
            done['evaluation_or_submission_started'] is False and set(done['controls']) == {'M0', 'M1'},
            'Training completion provenance/scope changed')
    require(type(done['elapsed_seconds']) in (int, float) and math.isfinite(done['elapsed_seconds']) and
            0 <= done['elapsed_seconds'] <= training.TOTAL_SECONDS, 'Training deadline invalid')
    _, draft = training.verify_runtime(TRAIN_RUNTIME_SHA)
    contract = training.approval(training.APPROVED, doc['training_approval_sha256'], draft)
    require(done['manifest_sha256'] == draft['manifest_sha256'], 'Training manifest binding changed')
    receipts = {}
    for arm in ('M0', 'M1'):
        receipt = read(folder / (arm + '-complete.json'))
        require(receipt == done['controls'][arm], 'Per-arm completion disagrees')
        name = 'v0351_train_' + arm
        service = qualify_service(folder, name, receipt['service_receipt_sha256'])
        started_ns = (folder / (name + '.log.service-start.json')).stat().st_mtime_ns
        checked = training.validate_training(contract, arm, doc['training_approval_sha256'], started_ns)
        require(checked == {k: v for k, v in receipt.items() if k != 'service_receipt_sha256'},
                'Training artifact/initialization/finite-metric qualification changed')
        receipts[arm] = dict(receipt=receipt, service=service)
    require(receipts['M0']['receipt']['initial_sha256'] == receipts['M1']['receipt']['initial_sha256'],
            'Matched training initialization differs')
    return dict(completed_sha256=sha(done_path), controls=receipts)


def submission_agent(text, alias):
    marker = 'model: ' + BASE + '\n'
    require(text.count(marker) == 1 and '\nadapter:' not in text, 'Unexpected original agent config')
    return text.replace(marker, 'model: ' + alias + '\n')


def destinations(frozen):
    paths = [EVIDENCE, MANIFEST, ROOT / 'logs/v043_server.log']
    for run in frozen['future_runs']:
        paths += [Path(run['submission_dir']), Path(run['result_dir'])]
        for name in ('v043_' + run['name'], 'v043_canary_' + run['name']):
            paths += [ROOT / ('logs/' + name + '.log'), ROOT / ('logs/' + name + '.log.routing.json')]
        paths += [ROOT / ('results/v043_canary_' + run['name'])]
    return paths


def stage_submissions(frozen, qualified):
    MANIFEST.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / 'subs/e4b_budgetfit_s1/agent.yaml', MANIFEST / 'agent.yaml')
    for run in frozen['future_runs']:
        baseline, target = Path(run['baseline_config']), Path(run['submission_dir'])
        require(run['adapter_alias'] == run['scientific_name'] and run['temperature'] == .2 and
                run['decoding_seed'] in (1, 2, 3), 'Unexpected frozen alias/seed')
        shutil.copytree(baseline, target)
        (target / 'agent.yaml').write_text(submission_agent((baseline / 'agent.yaml').read_text(), run['adapter_alias']))
        source = Path(run['adapter_source'])
        for parent in (target, MANIFEST):
            destination = parent / 'adapters' / run['adapter_alias']
            destination.mkdir(parents=True, exist_ok=False)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                require(sha(source / name) == qualified['controls'][run['arm']]['receipt']['artifacts'][str(source / name)],
                        'Qualified adapter source changed before copy')
                shutil.copyfile(source / name, destination / name)
                require(sha(destination / name) == sha(source / name), 'Adapter copy changed')
        expected_files = set(inventory(baseline)) | {
            'adapters/' + run['adapter_alias'] + '/' + name for name in ('adapter_config.json', 'adapter_model.safetensors')}
        require(set(inventory(target)) == expected_files, 'Unexpected submission file set')
        for name, digest in inventory(baseline).items():
            require((target / name).read_text() == submission_agent((baseline / name).read_text(), run['adapter_alias'])
                    if name == 'agent.yaml' else sha(target / name) == digest, 'Non-alias config bytes changed')
    from adk_submission import discover_adapters
    manifest = discover_adapters(MANIFEST)
    require(set(manifest.adapters) == {run['adapter_alias'] for run in frozen['future_runs']}, 'Official alias discovery differs')
    directories = {str(path): inventory(path) for path in [MANIFEST] + [Path(run['submission_dir']) for run in frozen['future_runs']]}
    return dict(directories=directories, training=qualified)


def verify_staged(staged):
    for name, files in staged['directories'].items():
        require(inventory(name) == files, 'Staged alias/model inputs changed')
    for arm in staged['training']['controls'].values():
        for name, digest in arm['receipt']['artifacts'].items():
            require(sha(name) == digest, 'Original trained adapter changed')


def registry_gate(cards, frozen):
    records = {card['id']: card for card in cards['data']}
    aliases = {run['adapter_alias'] for run in frozen['future_runs']}
    require(len(records) == len(cards['data']) and set(records) == {BASE, *aliases}, 'Unexpected served model/alias set')
    require(Path(records[BASE]['root']).resolve() == ROOT / 'models' / BASE, 'Wrong served base weights')
    for alias in aliases:
        require(Path(records[alias]['root']).resolve() == MANIFEST / 'adapters' / alias and
                records[alias]['parent'] == BASE, 'Wrong adapter root/parent registration: ' + alias)
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
    with (ROOT / 'logs/v043_server.log').open('x') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launched = dict(pid=child.pid, start=recovery.stat(child.pid)['start'], argv=argv, group=child.pid)
    write_json(EVIDENCE / 'server-launched.json', launched)
    until = min(deadline - RESERVE, time.monotonic() + 1200)
    while time.monotonic() < until:
        require(child.poll() is None, 'Evaluation server exited before readiness')
        try:
            cards = helper.api('models')
            if len(cards['data']) == 7:
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
    identity = re.fullmatch(r'v035_(M[01])_s([123])', alias)
    require(identity is not None and receipt['arm'] == identity[1] and type(receipt['decoding_seed']) is int and
            receipt['decoding_seed'] == int(identity[2]) and receipt['runtime_sha256'] == sha(RUNTIME) and
            receipt['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json'), 'Run arm/seed/source binding changed')
    require(receipt['generation'] == 'v043' and receipt['scientific_name'] == alias and
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
    require(completed['status'] == 'all_six_evaluations_qualified' and completed['runtime_sha256'] == runtime_sha and completed['readout_sha256'] == READOUT_SHA and
            completed['cleanup_completed'] is True and completed['analysis_pending'] is True and
            completed['submission_started'] is False and completed['elapsed_seconds'] <= TOTAL,
            'Generation completion provenance/scope/deadline failed')
    require(completed['generation'] == 'v043' and completed['memory_correction'] == MEMORY_CORRECTION,
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
    require(completed['training_completed_sha256'] == qualified_training(doc)['completed_sha256'], 'Training source generation changed')
    runs = {}
    budget = frozen['evaluation_config']['effective_budget']
    ids = frozen['populations']['all48_sensitivity']['task_ids']
    for run in frozen['future_runs']:
        name = run['name']
        canary = 'v043_canary_' + name
        check = qualify_run(canary, run['adapter_alias'], [CANARY], ROOT / ('results/' + canary),
                            ROOT / ('logs/' + canary + '.log'), True, budget)
        require(check['receipt'] == completed['canaries'][name], 'Canary/final completion disagreement')
        checked = qualify_run(name, run['adapter_alias'], ids, Path(run['result_dir']),
                              ROOT / ('logs/v043_' + name + '.log'), False, budget)
        require(checked['receipt'] == completed['runs'][name], 'Run/final completion disagreement')
        runs[name] = checked
    require(set(completed['runs']) == set(runs) == set(completed['canaries']), 'Incomplete six-alias generation')
    return doc, readout(), completed, runs


def activate(runtime_sha, queue, wait_seconds, controls_sha=None, replay_sha=None):
    doc, frozen = verify_runtime(runtime_sha)
    admission = admission_receipts(doc, controls_sha, replay_sha)
    recovery = load_recovery(ROOT)
    training.wait_exit(recovery, TRAIN_PID, TRAIN_START, queue, wait_seconds)
    qualified = qualified_training(doc)
    require(all(not path.exists() for path in destinations(frozen)), 'Execution destination collision')
    with (ROOT / 'migration/recovery-screen-v034/evaluation.lock').open('a') as old_lease, \
         (training.EVIDENCE.parent / 'training.lock').open('a') as train_lease:
        fcntl.flock(old_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(train_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, deadline = time.monotonic(), time.monotonic() + TOTAL
        EVIDENCE.mkdir(parents=True, exist_ok=False)
        try:
            verify_runtime(runtime_sha)
            admission_receipts(doc, controls_sha, replay_sha)
            qualified = qualified_training(doc)
            training.assert_idle(recovery)
            memory = host_memory_admission()
            cursor = kernel_cursor()
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
            require(not gpu.strip(), 'GPU compute processes remain after training')
            require(not any(line.split()[1] == '0100007F:1F40' and line.split()[3] == '0A'
                    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]), 'Serving listener already occupied')
            write_json(EVIDENCE / 'activation.json', dict(runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                       training=qualified, started_at=recovery.utc(), total_seconds=TOTAL,
                       generation='v043', memory_correction=MEMORY_CORRECTION, host_memory=memory, kernel_cursor=cursor,
                       admission_receipts=admission))
            staged = stage_submissions(frozen, qualified)
            write_json(EVIDENCE / 'staged-inputs.json', staged)
            helper = recovery.load_helper(ROOT)
            env = serving.environment(doc)
            launched, ready = start_server(recovery, helper, env, frozen, deadline)
            bounded = services.isolated_helper(helper, MemoryBoundedService(deadline, EVIDENCE))
            canaries, runs = {}, {}
            budget = frozen['evaluation_config']['effective_budget']
            ids = frozen['populations']['all48_sensitivity']['task_ids']
            cases = [(run, True) for run in frozen['future_runs']] + [(run, False) for run in frozen['future_runs']]
            for run, strict in cases:
                require(time.monotonic() < deadline - RESERVE, 'Evaluation launch deadline barrier')
                verify_runtime(runtime_sha)
                verify_staged(staged)
                training.assert_idle(recovery)
                server_live(recovery, launched, ready)
                registry_gate(helper.api('models'), frozen)
                global_oom_check(cursor)
                name = 'v043_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + (name if strict else 'v043_' + name) + '.log')
                selected = [CANARY] if strict else ids
                command = helper.eval_command(Path(run['submission_dir']), output, selected, budget, 1 if strict else None)
                guarded = [command[0], GUARD, '--guard-receipt', EVIDENCE / (log.name + '.memory-guard.json'), '--', *command[1:]]
                route = recovery.observe_run(bounded, run['adapter_alias'], guarded, log, env, ROOT / 'data/competition', strict=strict)
                oom = global_oom_check(cursor)
                write_json(EVIDENCE / (name + '-oom-check.json'), oom)
                rows = (recovery.validate_canary_result(output / 'task_results.jsonl', CANARY, budget) if strict else
                        recovery.validate_rows(output / 'task_results.jsonl', selected))
                receipt = dict(rows, alias=run['adapter_alias'], arm=run['arm'], decoding_seed=run['decoding_seed'],
                    generation='v043', scientific_name=run['scientific_name'], result_dir=str(output),
                    memory_guard=guard_qualification(EVIDENCE / (log.name + '.memory-guard.json')), global_oom_check=oom,
                    task_ids=selected, strict_canary=strict, routing=route,
                    route_sha256=sha(Path(str(log) + '.routing.json')), service=qualify_service(EVIDENCE, log.stem),
                    traces=services.trace_availability(output, selected, require_tools=strict),
                    runtime_sha256=runtime_sha, staged_sha256=sha(EVIDENCE / 'staged-inputs.json'))
                write_json(EVIDENCE / (name + '-complete.json'), receipt)
                checked = qualify_run(name, run['adapter_alias'], selected, output, log, strict, budget)
                (canaries if strict else runs)[run['name']] = checked['receipt']
                server_live(recovery, launched, ready)
            live = server_live(recovery, launched, ready)
            serving.stop_server(recovery, live, grace_seconds=min(60, max(1, deadline - time.monotonic())))
            require(not serving.group_members(recovery, launched['group']), 'Evaluation server group remains')
            write_json(EVIDENCE / 'server-cleanup.json', dict(status='owned_server_exited', group_empty=True,
                       pid=launched['pid'], start=launched['start'], group=launched['group'], method='verified_TERM_only'))
            verify_runtime(runtime_sha)
            verify_staged(staged)
            for run, strict in cases:
                name = 'v043_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + (name if strict else 'v043_' + name) + '.log')
                qualify_run(name, run['adapter_alias'], [CANARY] if strict else ids, output, log, strict, budget)
            require(time.monotonic() <= deadline, 'Evaluation total deadline exceeded')
            write_json(EVIDENCE / 'global-oom-final.json', global_oom_check(cursor))
            hashes = {p.name: sha(p) for p in EVIDENCE.glob('*.json')}
            write_json(EVIDENCE / 'completed.json', dict(status='all_six_evaluations_qualified', runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                generation='v043', memory_correction=MEMORY_CORRECTION,
                admission_receipts=admission,
                training_completed_sha256=qualified['completed_sha256'], cleanup_completed=True,
                runs=runs, canaries=canaries, evidence_hashes=hashes, elapsed_seconds=time.monotonic() - started,
                analysis_pending=True, submission_started=False))
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
            write_json(EVIDENCE / 'stopped.json', dict(type=type(error).__name__, reason=str(error), cleanup=cleanup,
                       elapsed_seconds=time.monotonic() - started, no_automatic_retry=True))
            raise


def prepare():
    require(not RUNTIME.exists() and not Path(str(RUNTIME) + '.sha256').exists(), 'Runtime destination collision')
    frozen = readout()
    require(sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == GUARD_SOURCE_SHA and
            sha(GUARD) == GUARD_WRAPPER_SHA, 'Guard sources are not the independently approved versions')
    verify_preservation_seal()
    with (ROOT / 'results/controls_train/controls.csv').open() as stream:
        controls = [row for row in csv.DictReader(stream) if row['instance_id'] == CANARY]
    require(len(controls) == 1 and controls[0]['gold_resolved'] == 'True' and
            controls[0]['empty_resolved'] == 'False' and controls[0]['empty_exit_code'] == '1', 'Fixed canary controls changed')
    train_doc, draft = training.verify_runtime(TRAIN_RUNTIME_SHA)
    approval_sha = sha(training.APPROVED)
    training.approval(training.APPROVED, approval_sha, draft)
    names = ['tools/run_mask_evaluations_v043.py', 'tools/test_run_mask_evaluations_v043.py',
        'tools/run_eval_memory_guard_v043.py', 'tools/sandbox_memory_guard_v043.py', 'tools/test_sandbox_memory_guard_v043.py',
        'tools/run_mask_evaluations_v039.py', 'tools/test_run_mask_evaluations_v039.py',
        'contracts/mask-evaluations-v039-runtime-DRAFT.json',
        'migration/mask-evaluations-v039/evaluation/stopped.json',
        'analysis/oom-recovery-v043/prior-v039-seal.json',
        'tools/run_mask_control_v0351.py', 'tools/test_run_mask_control_v0351.py',
        'tools/run_31b_recovery_screen_v034.py', 'tools/test_run_31b_recovery_screen_v034.py',
        'tools/run_v032_probes.py', 'tools/prepare_v032_probes.py', 'tools/recover_eval_v025.py',
        'tools/overnight_v02.py', 'tools/analyze_contract_v02_v025.py', 'tools/mechanism_metrics_v032.py',
        'tools/analyze_31b_recovery_screen_v034.py', 'tools/test_analyze_31b_recovery_screen_v034.py',
        'contracts/mask-control-v0351-runtime-DRAFT.json', 'contracts/mask-control-v0351-APPROVED.json',
        'contracts/mask-control-v035-readout.json', 'contracts/mask-control-v035-readout.json.sha256',
        'data/competition/snapshots/' + CANARY + '.tgz', 'results/controls_train/controls.csv']
    pins = {str(ROOT / name): sha(ROOT / name) for name in names}
    require(pins[str(ROOT / 'migration/mask-evaluations-v039/evaluation/stopped.json')] ==
            'bda19717158ceeb2694488f93673a17be8ace28661307280c1c2338dbb0f34b4', 'Previous stopped generation changed')
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
    doc = dict(status='DRAFT_PREPARED_ONLY', root=str(ROOT), total_seconds=TOTAL, cleanup_seconds=RESERVE,
        generation='v043', generation_runs=generation_runs(frozen), memory_correction=MEMORY_CORRECTION,
        guard_source_sha256=GUARD_SOURCE_SHA, guard_wrapper_sha256=GUARD_WRAPPER_SHA,
        required_controls_receipt=str(CONTROLS_RECEIPT), required_replay_receipt=str(REPLAY_RECEIPT),
        readout_sha256=READOUT_SHA, training_runtime_sha256=TRAIN_RUNTIME_SHA, training_approval_sha256=approval_sha,
        training_controller=dict(pid=TRAIN_PID, start=TRAIN_START), pins=pins,
        directories={str(ROOT / 'models' / BASE): inventory(ROOT / 'models' / BASE)},
        server_environment=train_doc['server_environment'], canary_task=CANARY,
        aliases=[run['adapter_alias'] for run in frozen['future_runs']],
        server_manifest=str(MANIFEST), run_order=[run['name'] for run in generation_runs(frozen)],
        policy='All six routing/tool canaries, then six fixed48 runs; no efficacy stopping, analysis, submission or new training',
        server_handoff='Verified TERM-only shutdown; qualification uses immutable recorded lifecycle, not server liveness',
        gpu_operations_during_preparation=False)
    write_json(RUNTIME, doc)
    digest = sha(RUNTIME)
    with Path(str(RUNTIME) + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + RUNTIME.name + '\n')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--queue-after-training', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=46800)
    parser.add_argument('--controls-receipt-sha')
    parser.add_argument('--replay-receipt-sha')
    args = parser.parse_args()
    require(math.isfinite(args.wait_seconds) and args.wait_seconds > 0, 'Invalid queue wait bound')
    if args.prepare:
        require(not args.activate, 'Preparation cannot activate')
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(), gpu_operations=False)))
    elif args.activate:
        require(args.runtime_sha, 'Activation requires runtime SHA')
        activate(args.runtime_sha, args.queue_after_training, args.wait_seconds,
                 args.controls_receipt_sha, args.replay_receipt_sha)
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        verify_runtime(args.runtime_sha)
        print(json.dumps(dict(status='prepared_only', gpu_operations=False)))


if __name__ == '__main__':
    main()
