"""Matched-context v051: the sealed M0 adapter served as in v043 plus one vLLM --chat-template flag."""
from __future__ import annotations

import argparse
import csv
import difflib
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import sys
import urllib.request
import uuid

from prepare_v032_probes import BASE, inventory, load_recovery, require, sha, write_json
import run_v032_probes as serving
import run_31b_recovery_screen_v034 as services
import run_mask_control_v0351 as training

ROOT = Path('PROJECT_ROOT/gemma4')
READOUT = ROOT / 'contracts/mask-control-v035-readout.json'
READOUT_SHA = '6319262b25c19ddcbb0e34a33f5e97325a7d6117a7b8faad8d4a4c7ff772c909'
RUNTIME = ROOT / 'contracts/matched-context-v051-runtime-DRAFT.json'
PACKAGE = ROOT / 'migration/matched-context-v051'
EVIDENCE = PACKAGE / 'evaluation'
MANIFEST = ROOT / 'subs/v051_M0mc_manifest'
GUARD = ROOT / 'tools/run_eval_memory_guard_v043.py'
GUARD_SOURCE_SHA = '5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GUARD_WRAPPER_SHA = 'c04928a3406c1175098f1217dd7b955dcec61aec1147537fbd68bf59de8bbbd5'
CONTROLS_RECEIPT = ROOT / 'analysis/oom-controls-v043/final48/report.json'
REPLAY_RECEIPT = ROOT / 'analysis/oom-replay-v043/report.json'
TRAIN_RUNTIME_SHA = '84453f87abb796603c1b9d8fc5f2e27bc3535b36c1fe24b580430201eca054e1'
ADAPTER = ROOT / 'sft/runs/A1_M0_v035/adapter_model.safetensors'
ADAPTER_SHA = '7484c31177fdfe6ec432f1f32eb1f29875810934d011447e443189d9961f75ad'
SERVE = ROOT / 'tools/serve_vllm_matched_v051.py'
SERVE_ORIGINAL = ROOT / 'serve_vllm.py'
SERVE_ORIGINAL_SHA = 'c4b513cc5bb87d4bc3741623a4bedac41cfca5b8bdf95c0a4743b8f9371d576a'
TEMPLATE = ROOT / 'analysis/matched-context-boss-v049/chat_template.matched.jinja'
TEMPLATE_SHA = '855bbbbbef10d51177c5d7eca19a3f3d39a11d9a80c142a9f19784029515a437'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-06-matched-context-eval.md'
PREREG_SHA = '86bc2ebbecc3a60363b8eed0b05a6575f48d1187fc8a1191fb2b301d918d6507'
COMPARATOR_RUNNER = ROOT / 'tools/run_mask_evaluations_v043.py'
COMPARATOR_RUNNER_SHA = '9b39a600adb998117a694df5dfd61514f03e29b467795b58bc449b495f8ca590'
COMPARATOR_RUNTIME = ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json'
COMPARATOR_RUNTIME_SHA = '7eaf5d53b56e02d6e45ef29a229748b2716bdee36ae7b4ebf0301450d4c57a63'
COMPARATOR_EVIDENCE = ROOT / 'migration/mask-evaluations-v043/evaluation'
COMPARATOR_ANALYSIS = ROOT / 'analysis/mask-control-v043/report.json'
COMPARATOR_ANALYSIS_SHA = '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2'
AUDIT = ROOT / 'analysis/serialization-audit-boss-v046/audit.py'
AUDIT_SHA = '5d26f4cdc9aafa2e9ffc1f527fd88d1ccd58497770f6b4a626b04ef5078cee98'
VERIFY = ROOT / 'analysis/matched-context-boss-v049/verify_matched.py'
VERIFY_SHA = '23f5f6d2074b2e5de2a6fd0a5e072f3c9defb56537d990a6e7e8e57d43e91004'
BYTE_TRACE = ROOT / 'results/v043_M0_s1/traces/trace_rich_2725.json'
BYTE_CALLS = (1, 3, 7)
BYTE_EXPECTED = PACKAGE / 'byte-check-expected.json'
SERVE_DIFF, RUNNER_DIFF = PACKAGE / 'serve.diff', PACKAGE / 'runner.diff'
TOKENIZE_URL = 'http://127.0.0.1:8000/tokenize'
FORMAT_TEXT = 'Detected the chat template content format to be '
SLOT_BINDING = dict(M0='v043_M0_s1..s3: sealed comparator, served template', M1='v051_M0mc_s1..s3: matched template',
                    sign='M1 - M0 = M0mc - M0', statistics='tools/analyze_mask_control_v042.py sha 23915fb0, ARMS (M0, M1) unchanged')
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
    runs = json.loads(json.dumps([run for run in frozen['future_runs'] if run['arm'] == 'M0']))
    for run in runs:
        run['scientific_name'] = run['name']
        run['name'] = run['name'].replace('v035_M0_', 'v051_M0mc_', 1)
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
        unit = 'gemma-v051-eval-' + uuid.uuid4().hex + '.service'
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


def verify_runtime(expected):
    doc = training.checked_json(RUNTIME, expected)
    require(Path(str(RUNTIME) + '.sha256').read_text().split()[0] == expected, 'Runtime sidecar mismatch')
    require(doc['root'] == str(ROOT) and doc['total_seconds'] == TOTAL and doc['cleanup_seconds'] == RESERVE and
            doc['generation'] == 'v051', 'Runtime identity/bounds changed')
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Runtime/package pin changed: ' + name)
    for name, files in doc['directories'].items():
        require(inventory(name) == files, 'Runtime inventory changed: ' + name)
    require(doc['chat_template'] == dict(path=str(TEMPLATE), sha256=TEMPLATE_SHA) and doc['pins'][str(TEMPLATE)] == TEMPLATE_SHA and
            doc['server_script'] == str(SERVE) and doc['pins'][str(SERVE_ORIGINAL)] == SERVE_ORIGINAL_SHA and
            doc['byte_check']['expected'] == str(BYTE_EXPECTED) and
            doc['pins'][str(BYTE_EXPECTED)] == doc['byte_check']['expected_sha256'] and doc['slot_binding'] == SLOT_BINDING,
            'Matched-context template/server/byte-check identity changed')
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
    prefix = 'v051' if folder == EVIDENCE else 'v034'
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
            done['evaluation_or_submission_started'] is False and 'M0' in done['controls'],
            'Training completion provenance/scope changed')
    require(type(done['elapsed_seconds']) in (int, float) and math.isfinite(done['elapsed_seconds']) and
            0 <= done['elapsed_seconds'] <= training.TOTAL_SECONDS, 'Training deadline invalid')
    _, draft = training.verify_runtime(TRAIN_RUNTIME_SHA)
    contract = training.approval(training.APPROVED, doc['training_approval_sha256'], draft)
    require(done['manifest_sha256'] == draft['manifest_sha256'], 'Training manifest binding changed')
    receipts = {}
    for arm in ('M0',):
        receipt = read(folder / (arm + '-complete.json'))
        require(receipt == done['controls'][arm], 'Per-arm completion disagrees')
        name = 'v0351_train_' + arm
        service = qualify_service(folder, name, receipt['service_receipt_sha256'])
        started_ns = (folder / (name + '.log.service-start.json')).stat().st_mtime_ns
        checked = training.validate_training(contract, arm, doc['training_approval_sha256'], started_ns)
        require(checked == {k: v for k, v in receipt.items() if k != 'service_receipt_sha256'},
                'Training artifact/initialization/finite-metric qualification changed')
        receipts[arm] = dict(receipt=receipt, service=service)
    require(receipts['M0']['receipt']['artifacts'][str(ADAPTER)] == ADAPTER_SHA, 'Sealed M0 adapter identity changed')
    return dict(completed_sha256=sha(done_path), controls=receipts)


def submission_agent(text, alias):
    marker = 'model: ' + BASE + '\n'
    require(text.count(marker) == 1 and '\nadapter:' not in text, 'Unexpected original agent config')
    return text.replace(marker, 'model: ' + alias + '\n')


def destinations(frozen):
    paths = [EVIDENCE, MANIFEST, ROOT / 'logs/v051_server.log']
    for run in frozen['future_runs']:
        paths += [Path(run['submission_dir']), Path(run['result_dir'])]
        for name in ('v051_' + run['name'], 'v051_canary_' + run['name']):
            paths += [ROOT / ('logs/' + name + '.log'), ROOT / ('logs/' + name + '.log.routing.json')]
        paths += [ROOT / ('results/v051_canary_' + run['name'])]
    return paths


def stage_submissions(frozen, qualified):
    comparator_staged = read(COMPARATOR_EVIDENCE / 'staged-inputs.json')['directories']
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
        require(inventory(target) == comparator_staged[str(ROOT / 'subs' / run['name'].replace('v051_M0mc_', 'v043_M0_', 1))],
                'Staged submission differs from the v043 M0 submission bytes')
    require(inventory(MANIFEST) == {name: digest for name, digest in comparator_staged[str(ROOT / 'subs/v043_mask_manifest')].items()
                                    if not name.startswith('adapters/v035_M1_')}, 'Staged manifest differs from the v043 M0 aliases')
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


def server_identity(recovery, pid, start, manifest):
    """run_v032_probes.server_identity (lines 74-106) with the v051 server script instead of serve_vllm.py."""
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
    """run_v032_probes.stop_server (lines 117-131) calling the v051 server_identity above."""
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
    with (ROOT / 'logs/v051_server.log').open('x') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    launched = dict(pid=child.pid, start=recovery.stat(child.pid)['start'], argv=argv, group=child.pid)
    write_json(EVIDENCE / 'server-launched.json', launched)
    until = min(deadline - RESERVE, time.monotonic() + 1200)
    while time.monotonic() < until:
        require(child.poll() is None, 'Evaluation server exited before readiness')
        try:
            cards = helper.api('models')
            if len(cards['data']) == 1 + len(frozen['future_runs']):
                records = registry_gate(cards, frozen)
                ready = server_live(recovery, launched)
                write_json(EVIDENCE / 'server-registry.json', records)
                write_json(EVIDENCE / 'server-ready.json', services.normalize(ready))
                return launched, ready
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise TimeoutError('Evaluation server readiness deadline exceeded')


def expected_vllm_argv(recorded, aliases):
    """The recorded v043 vLLM argv with this generation's aliases (--max-loras, --lora-modules) plus the one template flag."""
    start, end, loras = recorded.index('--lora-modules'), recorded.index('--served-model-name'), recorded.index('--max-loras')
    require(loras < start < end and '--chat-template' not in recorded, 'Unexpected v043 vLLM argv layout')
    argv = recorded[:start + 1] + [alias + '=' + str(MANIFEST / 'adapters' / alias) for alias in sorted(aliases)] + recorded[end:]
    argv[loras + 1] = str(len(aliases))
    return argv + ['--chat-template', str(TEMPLATE)]


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
            comparator_vllm_argv(), [run['adapter_alias'] for run in frozen['future_runs']]))
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


def qualify_run(name, alias, ids, output, log, strict, budget):
    receipt = read(EVIDENCE / (name + '-complete.json'))
    require(receipt['alias'] == alias and receipt['task_ids'] == ids and receipt['strict_canary'] is strict,
            'Run identity/seed/task receipt mismatch')
    identity = re.fullmatch(r'v035_(M0)_s([123])', alias)
    require(identity is not None and receipt['arm'] == identity[1] and type(receipt['decoding_seed']) is int and
            receipt['decoding_seed'] == int(identity[2]) and receipt['runtime_sha256'] == sha(RUNTIME) and
            receipt['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json'), 'Run arm/seed/source binding changed')
    require(receipt['generation'] == 'v051' and receipt['scientific_name'] == alias and
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
    require(completed['status'] == 'all_three_evaluations_qualified' and completed['runtime_sha256'] == runtime_sha and completed['readout_sha256'] == READOUT_SHA and
            completed['cleanup_completed'] is True and completed['analysis_pending'] is True and
            completed['submission_started'] is False and completed['elapsed_seconds'] <= TOTAL,
            'Generation completion provenance/scope/deadline failed')
    require(completed['generation'] == 'v051' and completed['memory_correction'] == MEMORY_CORRECTION,
            'Recovery memory policy/provenance changed')
    proofs = completed['admission_receipts']
    require(proofs == admission_receipts(doc, proofs['controls_sha256'], proofs['replay_sha256']), 'Admission proof changed')
    require(type(completed['elapsed_seconds']) in (int, float) and math.isfinite(completed['elapsed_seconds']) and
            0 <= completed['elapsed_seconds'] <= TOTAL, 'Invalid completed elapsed time')
    require({'activation.json', 'staged-inputs.json', 'server-launched.json', 'server-ready.json', 'byte-check.json',
             'server-registry.json', 'server-cleanup.json', 'global-oom-final.json'} <= set(completed['evidence_hashes']), 'Missing lifecycle hash bindings')
    for name, digest in completed['evidence_hashes'].items():
        require(sha(EVIDENCE / name) == digest, 'Lifecycle/evidence receipt changed: ' + name)
    cleanup = read(EVIDENCE / 'server-cleanup.json')
    require(read(EVIDENCE / 'global-oom-final.json')['no_global_oom'] is True, 'Global OOM invalidates generation')
    require(cleanup['status'] == 'owned_server_exited' and cleanup['group_empty'] is True, 'Server shutdown incomplete')
    launched, ready = read(EVIDENCE / 'server-launched.json'), read(EVIDENCE / 'server-ready.json')
    require(ready['pid'] == launched['pid'] == cleanup['pid'] and ready['start'] == launched['start'] == cleanup['start'] and
            ready['group'] == launched['group'] == cleanup['group'], 'Recorded server lifecycle identity mismatch')
    expected_argv = [str(Path('HOME_DIR/miniconda3/envs/swegemma/bin/python')), str(SERVE),
                     str(ROOT / 'models' / BASE), BASE, str(MANIFEST)]
    require(launched['argv'] == expected_argv and ready['group'] == ready['pid'] and
            ready['members'][str(ready['pid'])]['argv'] == expected_argv and
            ready['members'][str(ready['pid'])]['start'] == ready['start'] and ready['listener_owners'] and
            all(str(pid) in ready['members'] for pid in ready['listener_owners']), 'Recorded model/server ownership mismatch')
    byte = read(EVIDENCE / 'byte-check.json')
    require(byte['passed'] is True and byte['expected_sha256'] == doc['byte_check']['expected_sha256'] and
            [row['call'] for row in byte['rows']] == list(BYTE_CALLS) and all(row['equal'] for row in byte['rows']) and
            byte['argv'] == byte['expected_argv'] == ready['members'][str(byte['owner'])]['argv'] and
            set(byte['content_formats']) == {'openai'}, 'Launch byte check did not qualify')
    registry_gate({'data': list(read(EVIDENCE / 'server-registry.json').values())}, frozen)
    staged = read(EVIDENCE / 'staged-inputs.json')
    verify_staged(staged)
    require(completed['training_completed_sha256'] == qualified_training(doc)['completed_sha256'], 'Training source generation changed')
    runs = {}
    budget = frozen['evaluation_config']['effective_budget']
    ids = frozen['populations']['all48_sensitivity']['task_ids']
    for run in frozen['future_runs']:
        name = run['name']
        canary = 'v051_canary_' + name
        check = qualify_run(canary, run['adapter_alias'], [CANARY], ROOT / ('results/' + canary),
                            ROOT / ('logs/' + canary + '.log'), True, budget)
        require(check['receipt'] == completed['canaries'][name], 'Canary/final completion disagreement')
        checked = qualify_run(name, run['adapter_alias'], ids, Path(run['result_dir']),
                              ROOT / ('logs/v051_' + name + '.log'), False, budget)
        require(checked['receipt'] == completed['runs'][name], 'Run/final completion disagreement')
        runs[name] = checked
    require(set(completed['runs']) == set(runs) == set(completed['canaries']), 'Incomplete three-alias generation')
    return doc, readout(), completed, runs


def activate(runtime_sha, controls_sha=None, replay_sha=None):
    doc, frozen = verify_runtime(runtime_sha)
    admission = admission_receipts(doc, controls_sha, replay_sha)
    recovery = load_recovery(ROOT)
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
                       generation='v051', memory_correction=MEMORY_CORRECTION, host_memory=memory, kernel_cursor=cursor,
                       admission_receipts=admission))
            staged = stage_submissions(frozen, qualified)
            write_json(EVIDENCE / 'staged-inputs.json', staged)
            helper = recovery.load_helper(ROOT)
            env = serving.environment(doc)
            launched, ready = start_server(recovery, helper, env, frozen, deadline)
            byte_check(ready, frozen, training.checked_json(BYTE_EXPECTED, doc['byte_check']['expected_sha256']))
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
                name = 'v051_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + (name if strict else 'v051_' + name) + '.log')
                selected = [CANARY] if strict else ids
                command = helper.eval_command(Path(run['submission_dir']), output, selected, budget, 1 if strict else None)
                guarded = [command[0], GUARD, '--guard-receipt', EVIDENCE / (log.name + '.memory-guard.json'), '--', *command[1:]]
                route = recovery.observe_run(bounded, run['adapter_alias'], guarded, log, env, ROOT / 'data/competition', strict=strict)
                oom = global_oom_check(cursor)
                write_json(EVIDENCE / (name + '-oom-check.json'), oom)
                rows = (recovery.validate_canary_result(output / 'task_results.jsonl', CANARY, budget) if strict else
                        recovery.validate_rows(output / 'task_results.jsonl', selected))
                receipt = dict(rows, alias=run['adapter_alias'], arm=run['arm'], decoding_seed=run['decoding_seed'],
                    generation='v051', scientific_name=run['scientific_name'], result_dir=str(output),
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
            stop_server(recovery, live, grace_seconds=min(60, max(1, deadline - time.monotonic())))
            require(not serving.group_members(recovery, launched['group']), 'Evaluation server group remains')
            write_json(EVIDENCE / 'server-cleanup.json', dict(status='owned_server_exited', group_empty=True,
                       pid=launched['pid'], start=launched['start'], group=launched['group'], method='verified_TERM_only'))
            verify_runtime(runtime_sha)
            verify_staged(staged)
            for run, strict in cases:
                name = 'v051_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + (name if strict else 'v051_' + name) + '.log')
                qualify_run(name, run['adapter_alias'], [CANARY] if strict else ids, output, log, strict, budget)
            require(time.monotonic() <= deadline, 'Evaluation total deadline exceeded')
            write_json(EVIDENCE / 'global-oom-final.json', global_oom_check(cursor))
            hashes = {p.name: sha(p) for p in EVIDENCE.glob('*.json')}
            write_json(EVIDENCE / 'completed.json', dict(status='all_three_evaluations_qualified', runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                generation='v051', memory_correction=MEMORY_CORRECTION, byte_check_sha256=hashes['byte-check.json'],
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
    spec = importlib.util.spec_from_file_location('v051_' + Path(path).stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def byte_check_expected():
    """Offline matched render of calls 1, 3, 7 of the audited v043 M0 trace (v049 verify_matched.rebuild + v046 audit)."""
    verify = load_module(VERIFY, VERIFY_SHA)
    require(sha(AUDIT) == AUDIT_SHA and sha(TEMPLATE) == TEMPLATE_SHA, 'Audit or matched template changed')
    audit = verify.load_audit()
    tok = audit.load_tok(audit.SERVED_DIR)
    require(tok.chat_template == (ROOT / 'models' / BASE / 'chat_template.jinja').read_text(), 'Tokenizer template is not the served file')
    rebuilt = verify.rebuild(audit, tok, str(BYTE_TRACE))
    model = rebuilt['params']['model']
    require(model.startswith('openai/') and rebuilt['kwargs'] == {'enable_thinking': True}, 'Unexpected harness request shape')
    calls = []
    for number in BYTE_CALLS:
        served = rebuilt['served'][number - 1]
        conversation = audit.vllm_conversation(served['messages'])
        require(audit.render(tok, conversation, rebuilt['tools'], rebuilt['kwargs'])[1] == served['ids'], 'Served re-render differs')
        text, ids = audit.render(tok, conversation, rebuilt['tools'], dict(rebuilt['kwargs'], chat_template=TEMPLATE.read_text()))
        undone, edits = verify.undo_edits(text)
        calls.append(dict(call=number, compaction_form=served['compacted'], served_count=len(served['ids']),
                          matched_count=len(ids), differs_from_served=ids != served['ids'], edits=edits,
                          edits_only=undone == served['text'], matched_ids=ids,
                          request=dict(model=model.split('/', 1)[1], messages=served['messages'], tools=rebuilt['tools'],
                                       chat_template_kwargs=rebuilt['kwargs'], add_generation_prompt=True)))
    require([call['differs_from_served'] for call in calls] == [False, True, True] and all(call['edits_only'] for call in calls),
            'Byte-check calls do not discriminate the matched template as verified in v049')
    doc = dict(schema='matched-context-v051-byte-check-expected', trace=str(BYTE_TRACE), trace_sha256=sha(BYTE_TRACE),
               template=str(TEMPLATE), template_sha256=TEMPLATE_SHA, audit=str(AUDIT), audit_sha256=AUDIT_SHA,
               verify=str(VERIFY), verify_sha256=VERIFY_SHA, tokenizer=audit.SERVED_DIR, endpoint=TOKENIZE_URL,
               comparison='POST /tokenize body = request; response tokens must equal matched_ids exactly', calls=calls)
    return json.dumps(doc, indent=1, sort_keys=True) + '\n'


DEVIATIONS = [
    'Runs: only the three M0 runs of the frozen readout, renamed v051_M0mc_s{1,2,3}; served alias stays v035_M0_s{n} and the adapter source sft/runs/A1_M0_v035 (v043 generation_runs 51-59).',
    'Server script: tools/serve_vllm_matched_v051.py replaces serve_vllm.py (v043 414-415, 504-505); it is serve_vllm.py with "--chat-template", TEMPLATE appended to VllmConfig.extra_args (serve.diff), which VllmServer.build_cmd appends last (adk_submission/server.py:800); the extra_args conditions at server.py:720-724 and 752-757 do not involve it.',
    'vLLM argv = the recorded v043 argv + --chat-template TEMPLATE, except --max-loras 3 and three --lora-modules entries, which build_cmd derives from the three-alias manifest (server.py:759-798); checked live against expected_vllm_argv in the byte check.',
    'Readiness waits for 4 model cards (base + 3 M0 aliases) instead of 7 (v043 425); registry_gate code unchanged.',
    'Unit prefix gemma-v051-eval- (v043 94); service receipt prefix v051 (v043 292); log, result and receipt names v051 (v043 343-350, 416, 578-580, 608-610, 615).',
    'server_identity and stop_server are local copies of run_v032_probes.py:74-106 and 117-131; the only change is the script check args[1] == SERVE instead of basename serve_vllm.py (run_v032_probes.py:79), which would reject the copied script; v043 called serving.server_identity (404) and serving.stop_server (601).',
    'New launch byte check after readiness and before the first canary: live vLLM argv, openai content-format line from the API server log (/proc/<listener>/fd/1), POST /tokenize of calls 1, 3, 7 vs the pinned offline render; any failure writes byte-check.json, then the existing failure path writes stopped.json and TERM-cleans the server (v043 621-634), no retry.',
    'Training coupling kept only for the M0 adapter: training runtime + approval verification (v043 280-281, 318-320) and the M0 receipt/service/validate_training; dropped the M1 receipt loop and the M0=M1 initialisation check (v043 322-333); completion requires M0 in controls instead of the exact set {M0, M1} (v043 314); added the adapter sha 7484c311 check.',
    'Removed the training-controller wait (training.wait_exit, v043 535; TRAIN_PID/TRAIN_START, v043 37, 271, 683) and the --queue-after-training/--wait-seconds options (v043 703-708): training completed 2026-11-05 00:28 UTC and its completion record is checked.',
    'Removed the v039 coupling: preservation seal (v043 34-35, 246-250, 276, 642) and v039 pins/stopped-receipt check (v043 652-655, 665-666); it does not concern the M0 adapter.',
    'Staging adds two identity checks: each v051 submission dir equals the v043 M0 submission inventory of the same seed, and the v051 manifest equals the v043 manifest minus the M1 adapters (both from migration/mask-evaluations-v043/evaluation/staged-inputs.json).',
    'Completion status all_three_evaluations_qualified (v043 484, 615), generation v051, byte_check_sha256; qualify_generation requires and re-checks byte-check.json.',
    'Inspection mode (no --activate) also qualifies the M0 training receipt and, when both receipt shas are given, the admission receipts: a CPU-only preflight (v043 717-719 verified the runtime only).',
    'prepare is write-or-verify (a second run re-derives identical bytes) instead of refusing an existing contract (v043 638); it writes serve.diff, runner.diff and byte-check-expected.json and adds pins for them, the template, the v046/v049 code, the byte trace, the comparator receipts, the prereg and the vLLM render/tokenize sources.',
    'Unchanged: memory correction and guard, admission receipts, budgets, sampling, seeds, canary task and validation, TOTAL 28800 s / RESERVE 600 s, both locks LOCK_EX|LOCK_NB, GPU-empty and port-8000 checks, global OOM checks, cleanup and re-qualification.']


def prepare():
    frozen = readout()
    require(sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == GUARD_SOURCE_SHA and
            sha(GUARD) == GUARD_WRAPPER_SHA, 'Guard sources are not the independently approved versions')
    require(serve_copy_is_one_flag(), 'Server script is not serve_vllm.py plus the one template flag')
    PACKAGE.mkdir(exist_ok=True)
    write_or_verify(SERVE_DIFF, unified_diff(SERVE_ORIGINAL, SERVE))
    write_or_verify(RUNNER_DIFF, unified_diff(COMPARATOR_RUNNER, Path(__file__).resolve()))
    write_or_verify(BYTE_EXPECTED, byte_check_expected())
    with (ROOT / 'results/controls_train/controls.csv').open() as stream:
        controls = [row for row in csv.DictReader(stream) if row['instance_id'] == CANARY]
    require(len(controls) == 1 and controls[0]['gold_resolved'] == 'True' and
            controls[0]['empty_resolved'] == 'False' and controls[0]['empty_exit_code'] == '1', 'Fixed canary controls changed')
    train_doc, draft = training.verify_runtime(TRAIN_RUNTIME_SHA)
    approval_sha = sha(training.APPROVED)
    training.approval(training.APPROVED, approval_sha, draft)
    names = ['tools/run_matched_context_v051.py', 'tools/test_run_matched_context_v051.py', 'tools/serve_vllm_matched_v051.py',
        'serve_vllm.py', 'tools/run_mask_evaluations_v043.py', 'tools/test_run_mask_evaluations_v043.py',
        'tools/run_eval_memory_guard_v043.py', 'tools/sandbox_memory_guard_v043.py', 'tools/test_sandbox_memory_guard_v043.py',
        'contracts/mask-evaluations-v043-runtime-DRAFT.json', 'contracts/mask-evaluations-v043-runtime-DRAFT.json.sha256',
        'migration/mask-evaluations-v043/evaluation/completed.json', 'migration/mask-evaluations-v043/evaluation/staged-inputs.json',
        'migration/mask-evaluations-v043/evaluation/server-ready.json',
        'analysis/mask-control-v043/report.json', 'analysis/mask-control-v043/per_task.jsonl',
        'analysis/matched-context-boss-v049/chat_template.matched.jinja', 'analysis/matched-context-boss-v049/template.diff',
        'analysis/matched-context-boss-v049/README.md', 'analysis/matched-context-boss-v049/verify_matched.py',
        'analysis/matched-context-boss-v049/verify_report.json', 'analysis/matched-context-boss-v049/SHA256SUMS',
        'analysis/serialization-audit-boss-v046/audit.py', 'analysis/serialization-audit-boss-v046/report.json',
        'results/v043_M0_s1/traces/trace_rich_2725.json', 'models/' + BASE + '/chat_template.jinja',
        'sft/runs/A1_M0_v035/adapter_model.safetensors', 'sft/runs/A1_M0_v035/adapter_config.json',
        'tools/run_mask_control_v0351.py', 'tools/test_run_mask_control_v0351.py',
        'tools/run_31b_recovery_screen_v034.py', 'tools/test_run_31b_recovery_screen_v034.py',
        'tools/run_v032_probes.py', 'tools/prepare_v032_probes.py', 'tools/recover_eval_v025.py',
        'tools/overnight_v02.py', 'tools/analyze_contract_v02_v025.py', 'tools/mechanism_metrics_v032.py',
        'tools/analyze_mechanism_v031.py', 'tools/analyze_mask_control_v042.py',
        'tools/analyze_31b_recovery_screen_v034.py', 'tools/test_analyze_31b_recovery_screen_v034.py',
        'contracts/mask-control-v0351-runtime-DRAFT.json', 'contracts/mask-control-v0351-APPROVED.json',
        'contracts/mask-control-v035-readout.json', 'contracts/mask-control-v035-readout.json.sha256',
        'data/competition/snapshots/' + CANARY + '.tgz', 'results/controls_train/controls.csv',
        'migration/matched-context-v051/serve.diff', 'migration/matched-context-v051/runner.diff',
        'migration/matched-context-v051/byte-check-expected.json']
    pins = {str(ROOT / name): sha(ROOT / name) for name in names}
    pins[str(PREREG)] = sha(PREREG)
    require(pins[str(TEMPLATE)] == TEMPLATE_SHA and pins[str(PREREG)] == PREREG_SHA and pins[str(ADAPTER)] == ADAPTER_SHA and
            pins[str(SERVE_ORIGINAL)] == SERVE_ORIGINAL_SHA and pins[str(COMPARATOR_RUNNER)] == COMPARATOR_RUNNER_SHA and
            pins[str(COMPARATOR_RUNTIME)] == COMPARATOR_RUNTIME_SHA and pins[str(COMPARATOR_ANALYSIS)] == COMPARATOR_ANALYSIS_SHA,
            'A pre-registered v051 input hash differs')
    package = Path('HOME_DIR/miniconda3/envs/swegemma/lib/python3.12/site-packages')
    for name in ['swegemma/cli.py', 'adk_submission/discovery.py', 'adk_submission/server.py',
        'adk_submission/compiler.py', 'adk_submission/resolvers/models.py', 'adk_submission/resolvers/generation.py',
        'swegemma/evaluate.py', 'swegemma/results.py', 'swegemma/harness/verification.py',
        'swegemma/sandbox/subprocess.py', 'swegemma/harness/agent_runner.py', 'google/adk/models/lite_llm.py',
        'google/adk/apps/compaction.py',
        'adk_eval_core/tracing/atif.py', 'adk_eval_core/tracing/atif_converter.py', 'adk_eval_core/tracing/trace.py',
        'vllm/entrypoints/openai/models/serving.py', 'vllm/entrypoints/openai/api_server.py',
        'vllm/entrypoints/serve/tokenize/serving.py', 'vllm/entrypoints/serve/tokenize/protocol.py',
        'vllm/entrypoints/serve/render/serving.py', 'vllm/renderers/hf.py', 'vllm/entrypoints/chat_utils.py',
        'vllm/reasoning/gemma4_reasoning_parser.py', 'vllm/reasoning/basic_parsers.py', 'vllm/tool_parsers/gemma4_tool_parser.py']:
        pins[str(package / name)] = sha(package / name)
    pins.update(read(ROOT / 'contracts/recovery-screen-v034-plan.json')['runtime_pins'])
    pins.update(read(ROOT / 'contracts/recovery-screen-v034-runtime.json')['installed_pins'])
    pins['/usr/bin/journalctl'] = sha('/usr/bin/journalctl')
    runs = generation_runs(frozen)
    aliases = [run['adapter_alias'] for run in runs]
    config, populations = frozen['evaluation_config'], frozen['populations']
    comparator_runs = [run['name'].replace('v051_M0mc_', 'v043_M0_', 1) for run in runs]
    doc = dict(status='DRAFT_PREPARED_ONLY', root=str(ROOT), total_seconds=TOTAL, cleanup_seconds=RESERVE,
        generation='v051', generation_runs=runs, memory_correction=MEMORY_CORRECTION,
        guard_source_sha256=GUARD_SOURCE_SHA, guard_wrapper_sha256=GUARD_WRAPPER_SHA,
        required_controls_receipt=str(CONTROLS_RECEIPT), required_replay_receipt=str(REPLAY_RECEIPT),
        readout_sha256=READOUT_SHA, training_runtime_sha256=TRAIN_RUNTIME_SHA, training_approval_sha256=approval_sha,
        pins=pins, directories={str(ROOT / 'models' / BASE): inventory(ROOT / 'models' / BASE)},
        server_environment=train_doc['server_environment'], canary_task=CANARY, aliases=aliases,
        server_manifest=str(MANIFEST), run_order=[run['name'] for run in runs],
        policy='Launch byte check, then three routing/tool canaries (one per M0 alias), then three fixed48 runs; no efficacy stopping, analysis, submission or training',
        server_handoff='Verified TERM-only shutdown; qualification uses immutable recorded lifecycle, not server liveness',
        gpu_operations_during_preparation=False,
        preregistration=dict(path=str(PREREG), sha256=PREREG_SHA),
        chat_template=dict(path=str(TEMPLATE), sha256=TEMPLATE_SHA),
        server_script=str(SERVE), server_script_sha256=pins[str(SERVE)],
        serve_diff=dict(path=str(SERVE_DIFF), sha256=pins[str(SERVE_DIFF)], base=str(SERVE_ORIGINAL), base_sha256=SERVE_ORIGINAL_SHA),
        runner=dict(path=str(ROOT / 'tools/run_matched_context_v051.py'), sha256=pins[str(ROOT / 'tools/run_matched_context_v051.py')],
                    diff=str(RUNNER_DIFF), diff_sha256=pins[str(RUNNER_DIFF)], base=str(COMPARATOR_RUNNER), base_sha256=COMPARATOR_RUNNER_SHA),
        adapter=dict(path=str(ADAPTER), sha256=ADAPTER_SHA),
        expected_vllm_argv=expected_vllm_argv(comparator_vllm_argv(), aliases),
        byte_check=dict(expected=str(BYTE_EXPECTED), expected_sha256=pins[str(BYTE_EXPECTED)], trace=str(BYTE_TRACE),
                        calls=list(BYTE_CALLS), endpoint=TOKENIZE_URL, content_format='openai',
                        stop_rule='Any mismatch or error: byte-check.json, stopped.json, TERM cleanup, no retry'),
        evaluation=dict(effective_budget=config['effective_budget'], sampling=config['sampling'], compaction=config['compaction'],
                        seeds=[1, 2, 3], all48=populations['all48_sensitivity']['task_ids'],
                        primary40=populations['primary40']['task_ids'], clean36=populations['control_clean36_sensitivity']['task_ids']),
        comparator=dict(runtime=str(COMPARATOR_RUNTIME), runtime_sha256=COMPARATOR_RUNTIME_SHA, runner=str(COMPARATOR_RUNNER),
                        runner_sha256=COMPARATOR_RUNNER_SHA, runs=comparator_runs, pairing='task x decoding seed',
                        result_inventories={str(ROOT / 'results' / name): inventory(ROOT / 'results' / name) for name in comparator_runs},
                        analysis_report=str(COMPARATOR_ANALYSIS), analysis_report_sha256=COMPARATOR_ANALYSIS_SHA,
                        per_task=str(ROOT / 'analysis/mask-control-v043/per_task.jsonl'),
                        per_task_sha256=pins[str(ROOT / 'analysis/mask-control-v043/per_task.jsonl')]),
        slot_binding=SLOT_BINDING,
        readings=dict(source=str(PREREG) + ':20-25', population='primary40, three-seed means (v042 prediction_inputs)',
                      MC_i='loop <= 0.30 AND submit >= 0.80 AND M5 repeat-after-error <= 0.20 (M0mc)',
                      MC_ii='|delta loop| <= 0.10 AND |delta submit| <= 0.10 AND |delta M5-error| <= 0.10 (delta = M0mc - M0)',
                      otherwise='intermediate: paired 95% task-bootstrap intervals, no label',
                      resolution='not predicted; pooled M0mc >= 4/120 reported as notable and bounded, never as recovery'),
        readme=dict(deviations_from_v043_runner=DEVIATIONS,
                    flag_source='vLLM 0.19.1 --help line 4 [--chat-template CHAT_TEMPLATE]; read at vllm/entrypoints/openai/api_server.py:346 and passed to chat (369) and tokenize (380-388) serving',
                    tokenize_source='vllm/entrypoints/serve/tokenize/protocol.py:50-145 (TokenizeChatRequest), serving.py:55-122 (same preprocess_chat as chat completions)'))
    write_or_verify(RUNTIME, json.dumps(doc, indent=2, sort_keys=True) + '\n')
    digest = sha(RUNTIME)
    write_or_verify(Path(str(RUNTIME) + '.sha256'), digest + '  ' + RUNTIME.name + '\n')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--controls-receipt-sha')
    parser.add_argument('--replay-receipt-sha')
    args = parser.parse_args()
    if args.prepare:
        require(not args.activate, 'Preparation cannot activate')
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(), gpu_operations=False)))
    elif args.activate:
        require(args.runtime_sha, 'Activation requires runtime SHA')
        activate(args.runtime_sha, args.controls_receipt_sha, args.replay_receipt_sha)
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        doc, _ = verify_runtime(args.runtime_sha)
        training_m0 = qualified_training(doc)['completed_sha256']
        admitted = (admission_receipts(doc, args.controls_receipt_sha, args.replay_receipt_sha)['byte_compatibility']
                    if args.controls_receipt_sha or args.replay_receipt_sha else None)
        print(json.dumps(dict(status='prepared_only', gpu_operations=False, runtime_verified=True,
                              training_completed_sha256=training_m0, admission_receipts_verified=admitted)))


if __name__ == '__main__':
    main()
