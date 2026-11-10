"""v054: the corrected adapter M0q evaluated exactly like v043 M0 (LoRA-on server, v043 template, capped memory)."""
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
RUNTIME = ROOT / 'contracts/m0q-eval-v054-runtime-DRAFT.json'
SOURCES_DRAFT = ROOT / 'contracts/m0q-eval-v054-sources-DRAFT.json'
EVIDENCE = ROOT / 'migration/m0q-eval-v054/evaluation'
MANIFEST = ROOT / 'migration/m0q-eval-v054/server-manifest'
RUNNER_DIFF = ROOT / 'migration/m0q-eval-v054/runner.diff'
V043_RUNTIME = ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json'
V043_RUNTIME_SHA = '7eaf5d53b56e02d6e45ef29a229748b2716bdee36ae7b4ebf0301450d4c57a63'
LOCKS = (ROOT / 'migration/recovery-screen-v034/evaluation.lock', ROOT / 'migration/mask-control-v0351/training.lock')
GUARD = ROOT / 'tools/run_eval_memory_guard_v043.py'
GUARD_SOURCE_SHA = '5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GUARD_WRAPPER_SHA = 'c04928a3406c1175098f1217dd7b955dcec61aec1147537fbd68bf59de8bbbd5'
CONTROLS_RECEIPT = ROOT / 'analysis/oom-controls-v043/final48/report.json'
REPLAY_RECEIPT = ROOT / 'analysis/oom-replay-v043/report.json'
TOTAL, RESERVE = 28800, 600
CANARY = 'fastapi_11194'
ARM, SEEDS = 'M0q', (1, 2, 3)
# M0q training receipts: the v053 wrapper's meta.json/train_log.jsonl replace the v0351 training receipts.
ADAPTER = ROOT / 'sft/runs/A1_M0q_v053'
WRAPPER = ROOT / 'sft/train_qatbase_v053.py'
WRAPPER_SHA = '688231bf609b85a8530ab41531a2460d2c1dde28c8e6fcd337491c12583c4473'
TRAINING_CONTRACT_SHA = '02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde'
M0_ADAPTER, M0_META = ROOT / 'sft/runs/A1_M0_v035', ROOT / 'sft/runs/A1_M0_v035.meta.json'
M0_INIT_SHA = '9dfc7ccc71fc4631d51a61dad8a889040e76bd9c87a2546fe24ec3b769e20066'
DEQUANT = ROOT / 'models/gemma-4-e4b-it-qat-dequant-bf16'
DATA = ROOT / 'sft/data/mask_control_v035'
FROZEN_TRAINERS = [ROOT / ('sft/train_' + 'lora.py'), ROOT / 'sft/train_mask_control_v035.py', ROOT / 'sft/prepare_mask_control_v035.py']
COMPOSITION = ROOT / 'analysis/qat-transfer-v053/composition-check.json'
MEMORY_CORRECTION = dict(version='v043_sandbox_cgroup_v1',
    per_sandbox=dict(memory_max=4294967296, memory_swap_max=0),
    service=dict(MemoryMax=51539607552, MemorySwapMax=0, Delegate='memory', DelegateSubgroup='supervisor', OOMPolicy='continue'),
    outer_memory_events_local_zero=['max', 'oom', 'oom_kill', 'oom_group_kill'],
    host_min_available_bytes=68719476736,
    scope='Commands routed through SubprocessManager.exec: agent, verification pytest, and setup commands using that path',
    exceptions='Direct manager.start git/materialization and other direct subprocess host helpers remain under the outer 48GiB service cap only',
    comparability='Frozen numerical thresholds retained descriptively; this infrastructure correction is not evidence of historical exchangeability',
    global_oom_policy='No kernel CONSTRAINT_NONE events after admission journal cursor through final cleanup')


def generation_runs():
    runs = []
    for seed in SEEDS:
        name, alias = 'v054_' + ARM + '_s' + str(seed), 'v053_' + ARM + '_s' + str(seed)
        runs.append(dict(name=name, scientific_name=alias, arm=ARM, adapter_alias=alias, adapter_source=str(ADAPTER),
                         temperature=.2, decoding_seed=seed, baseline_config=str(ROOT / ('subs/e4b_budgetfit_s' + str(seed))),
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
        unit = 'gemma-v054-eval-' + uuid.uuid4().hex + '.service'
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
            doc['generation'] == 'v054', 'Runtime identity/bounds changed')
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Runtime/package pin changed: ' + name)
    for name, files in doc['directories'].items():
        require(inventory(name) == files, 'Runtime inventory changed: ' + name)
    require(sha(SOURCES_DRAFT) == doc['sources_sha256'], 'Sources draft changed after the runtime contract')
    frozen = readout()
    require(doc['generation_runs'] == generation_runs() and doc['memory_correction'] == MEMORY_CORRECTION,
            'Scientific mapping or memory correction changed')
    return doc, effective_readout(frozen)


def qualify_service(folder, name, expected_end=None):
    start_path = folder / (name + '.log.service-start.json')
    end_path = folder / (name + '.log.service-end.json')
    if expected_end is not None:
        require(sha(end_path) == expected_end, 'Service receipt hash changed')
    start, end = read(start_path), read(end_path)
    identity = {key: start[key] for key in ('InvocationID', 'ControlGroup')}
    prefix = 'v054' if folder == EVIDENCE else 'v034'
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


def qualified_m0q():
    """Substitutes the v0351 training receipts: the v053 wrapper's meta.json and train_log.jsonl, checked against M0."""
    paths = [ADAPTER / 'adapter_model.safetensors', ADAPTER / 'adapter_config.json', ADAPTER / 'meta.json', ADAPTER / 'train_log.jsonl']
    require(all(path.is_file() for path in paths), 'M0q adapter or training receipts unavailable')
    meta, m0 = read(ADAPTER / 'meta.json'), read(M0_META)
    require(meta['mode'] == 'm0q' and meta['fixture'] is None and meta['wrapper']['sha256'] == WRAPPER_SHA == sha(WRAPPER) and
            meta['contract_sha256'] == TRAINING_CONTRACT_SHA and meta['gates_pass'] is True and meta['init_hash_equals_m0'] is True and
            meta['initial_trainable_parameters_sha256'] == meta['m0_initial_trainable_parameters_sha256'] ==
            m0['initial_trainable_parameters_sha256'] == M0_INIT_SHA and meta['lora_modules'] == 258 and
            meta['adapter_tensors'] == 516 and meta['names_match_probe'] is True and meta['updates'] == meta['updates_planned'] == 208 and
            sorted(meta['overrides_vs_m0']) == ['base', 'out'] and meta['codex_manifest_sha256'] == sha(DATA / 'manifest.json') and
            meta['frozen_code_sha256'] == {str(path): sha(path) for path in FROZEN_TRAINERS} and
            meta['base']['kind'] == 'dequant' and meta['base']['dir'] == str(DEQUANT) and
            meta['base']['manifest_sha256'] == sha(DEQUANT / 'manifest.json'), 'M0q wrapper receipt, gates or provenance changed')
    keep = lambda args: {k: v for k, v in args.items() if k not in ('base', 'out')}
    require(keep(meta['args']) == keep(m0['args']) and meta['args']['base'] == str(DEQUANT) and meta['args']['out'] == str(ADAPTER),
            'M0q trainer arguments differ from M0 beyond base and output')
    require(all(type(meta[key]) in (int, float) and math.isfinite(meta[key]) and meta[key] > 0 for key in ('seconds', 'tokens')),
            'Nonfinite or empty M0q trainer summary')
    config, m0_config = read(ADAPTER / 'adapter_config.json'), read(M0_ADAPTER / 'adapter_config.json')
    require({k: v for k, v in config.items() if k != 'base_model_name_or_path'} ==
            {k: v for k, v in m0_config.items() if k != 'base_model_name_or_path'}, 'M0q adapter config differs from M0 beyond the base string')
    tensors, reference = training.safe_header(paths[0]), training.safe_header(M0_ADAPTER / 'adapter_model.safetensors')
    require(len(tensors) == 516 and {k: (v['shape'], v['dtype']) for k, v in tensors.items()} ==
            {k: (v['shape'], v['dtype']) for k, v in reference.items()}, 'M0q adapter tensors differ from M0 in names, shapes or dtypes')
    logs = [json.loads(line) for line in paths[3].read_text().splitlines() if line.strip()]
    groups = read(DATA / 'update_groups.json')
    require(len(logs) == len(groups) == 208, 'M0q training updates incomplete')
    for index, (row, group) in enumerate(zip(logs, groups), 1):
        require(row['step'] == index and row['loss_tokens'] == row['remaining_loss_tokens'] == group['original_denominator'] and
                row['sequences'] == len(group['sequence_indices']), 'Frozen update exposure mismatch')
        for key in ('loss', 'lr', 'grad_norm', 'seconds', 'tokens_per_s', 'peak_alloc_gib'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]), 'Nonfinite training metric: ' + key)
    require(meta['tokens'] == sum(row['tokens'] for row in logs), 'Training token summary mismatch')
    return dict(artifacts={str(path): sha(path) for path in paths}, initial_sha256=meta['initial_trainable_parameters_sha256'],
                updates=len(logs), adapter_tensors=len(tensors), base_model_name_or_path=config['base_model_name_or_path'],
                m0_base_model_name_or_path=m0_config['base_model_name_or_path'], dequant_manifest_sha256=meta['base']['manifest_sha256'],
                wrapper_sha256=WRAPPER_SHA, receipt_substitution='v053 wrapper meta.json + train_log.jsonl replace the v0351 training receipts')


def submission_agent(text, alias):
    marker = 'model: ' + BASE + '\n'
    require(text.count(marker) == 1 and '\nadapter:' not in text, 'Unexpected original agent config')
    return text.replace(marker, 'model: ' + alias + '\n')


def destinations(frozen):
    paths = [EVIDENCE, MANIFEST, ROOT / 'logs/v054_M0q_server.log']
    for run in frozen['future_runs']:
        paths += [Path(run['submission_dir']), Path(run['result_dir'])]
        for name in (run['name'], 'v054_canary_' + run['name']):
            paths += [ROOT / ('logs/' + name + '.log'), ROOT / ('logs/' + name + '.log.routing.json')]
        paths += [ROOT / ('results/v054_canary_' + run['name'])]
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
                require(sha(source / name) == qualified['artifacts'][str(source / name)], 'Qualified adapter source changed before copy')
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
    for name, digest in staged['training']['artifacts'].items():
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
    with (ROOT / 'logs/v054_M0q_server.log').open('x') as stream:
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


def qualify_run(name, alias, ids, output, log, strict, budget):
    receipt = read(EVIDENCE / (name + '-complete.json'))
    require(receipt['alias'] == alias and receipt['task_ids'] == ids and receipt['strict_canary'] is strict,
            'Run identity/seed/task receipt mismatch')
    identity = re.fullmatch(r'v053_(M0q)_s([123])', alias)
    require(identity is not None and receipt['arm'] == identity[1] and type(receipt['decoding_seed']) is int and
            receipt['decoding_seed'] == int(identity[2]) and receipt['runtime_sha256'] == sha(RUNTIME) and
            receipt['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json'), 'Run arm/seed/source binding changed')
    require(receipt['generation'] == 'v054' and receipt['scientific_name'] == alias and
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
    require(completed['status'] == 'all_three_m0q_evaluations_qualified' and completed['runtime_sha256'] == runtime_sha and completed['readout_sha256'] == READOUT_SHA and
            completed['cleanup_completed'] is True and completed['analysis_pending'] is True and
            completed['submission_started'] is False and completed['elapsed_seconds'] <= TOTAL,
            'Generation completion provenance/scope/deadline failed')
    require(completed['generation'] == 'v054' and completed['memory_correction'] == MEMORY_CORRECTION,
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
    require(completed['m0q_training'] == doc['m0q_training'] == qualified_m0q(), 'M0q training qualification changed')
    runs = {}
    budget = frozen['evaluation_config']['effective_budget']
    ids = frozen['populations']['all48_sensitivity']['task_ids']
    for run in frozen['future_runs']:
        name = run['name']
        canary = 'v054_canary_' + name
        check = qualify_run(canary, run['adapter_alias'], [CANARY], ROOT / ('results/' + canary),
                            ROOT / ('logs/' + canary + '.log'), True, budget)
        require(check['receipt'] == completed['canaries'][name], 'Canary/final completion disagreement')
        checked = qualify_run(name, run['adapter_alias'], ids, Path(run['result_dir']),
                              ROOT / ('logs/' + name + '.log'), False, budget)
        require(checked['receipt'] == completed['runs'][name], 'Run/final completion disagreement')
        runs[name] = checked
    require(set(completed['runs']) == set(runs) == set(completed['canaries']), 'Incomplete three-alias M0q generation')
    return doc, readout(), completed, runs


def activate(runtime_sha, controls_sha=None, replay_sha=None):
    doc, frozen = verify_runtime(runtime_sha)
    admission = admission_receipts(doc, controls_sha, replay_sha)
    recovery = load_recovery(ROOT)
    qualified = qualified_m0q()
    require(qualified == doc['m0q_training'], 'M0q training differs from the prepared runtime contract')
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
            qualified = qualified_m0q()
            require(qualified == doc['m0q_training'], 'M0q training differs from the prepared runtime contract')
            training.assert_idle(recovery)
            memory = host_memory_admission()
            cursor = kernel_cursor()
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
            require(not gpu.strip(), 'GPU compute processes present; evaluation refused')
            require(not any(line.split()[1] == '0100007F:1F40' and line.split()[3] == '0A'
                    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]), 'Serving listener already occupied')
            write_json(EVIDENCE / 'activation.json', dict(runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                       m0q_training=qualified, started_at=recovery.utc(), total_seconds=TOTAL,
                       generation='v054', memory_correction=MEMORY_CORRECTION, host_memory=memory, kernel_cursor=cursor,
                       admission_receipts=admission, user_manager_failed_units=failed_units(),
                       failed_units_policy='record only; historical failed units do not block (team lead, 2026-11-05)'))
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
                name = 'v054_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + name + '.log')
                selected = [CANARY] if strict else ids
                command = helper.eval_command(Path(run['submission_dir']), output, selected, budget, 1 if strict else None)
                guarded = [command[0], GUARD, '--guard-receipt', EVIDENCE / (log.name + '.memory-guard.json'), '--', *command[1:]]
                route = recovery.observe_run(bounded, run['adapter_alias'], guarded, log, env, ROOT / 'data/competition', strict=strict)
                oom = global_oom_check(cursor)
                write_json(EVIDENCE / (name + '-oom-check.json'), oom)
                rows = (recovery.validate_canary_result(output / 'task_results.jsonl', CANARY, budget) if strict else
                        recovery.validate_rows(output / 'task_results.jsonl', selected))
                receipt = dict(rows, alias=run['adapter_alias'], arm=run['arm'], decoding_seed=run['decoding_seed'],
                    generation='v054', scientific_name=run['scientific_name'], result_dir=str(output),
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
                name = 'v054_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + name + '.log')
                qualify_run(name, run['adapter_alias'], [CANARY] if strict else ids, output, log, strict, budget)
            require(time.monotonic() <= deadline, 'Evaluation total deadline exceeded')
            write_json(EVIDENCE / 'global-oom-final.json', global_oom_check(cursor))
            hashes = {p.name: sha(p) for p in EVIDENCE.glob('*.json')}
            write_json(EVIDENCE / 'completed.json', dict(status='all_three_m0q_evaluations_qualified', runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                generation='v054', memory_correction=MEMORY_CORRECTION,
                admission_receipts=admission, m0q_training=qualified, cleanup_completed=True,
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
            try:
                offenders = idle_offenders(recovery)
            except BaseException as probe_error:
                offenders = 'unavailable: ' + type(probe_error).__name__ + ': ' + str(probe_error)
            write_json(EVIDENCE / 'stopped.json', dict(type=type(error).__name__, reason=str(error), cleanup=cleanup,
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
    try:
        adapter = dict(qualified=True, **{k: v for k, v in qualified_m0q().items() if k != 'artifacts'})
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        adapter = dict(qualified=False, reason=type(error).__name__ + ': ' + str(error))
    ready = (bool(gpu) and int(gpu[0]) <= 100 and not apps and port == 0 and not held and not offenders and not collisions and
             available >= MEMORY_CORRECTION['host_min_available_bytes'] and adapter['qualified'])
    return dict(ready=ready, gpu_memory_used_mib=int(gpu[0]) if gpu else None, gpu_compute_apps=apps, port_8000_listeners=port,
                locks_held=held, mem_available_bytes=available, destination_collisions=collisions, idle_offenders=offenders,
                failed_units=failed_units(), m0q_adapter=adapter, composition_check_present=COMPOSITION.is_file(),
                sources_draft_present=SOURCES_DRAFT.is_file(), runtime_contract_present=RUNTIME.is_file())


def verify_sources(sources_sha):
    sources = training.checked_json(SOURCES_DRAFT, sources_sha)
    require(Path(str(SOURCES_DRAFT) + '.sha256').read_text().split()[0] == sources_sha, 'Sources draft sidecar mismatch')
    require(sources['root'] == str(ROOT) and sources['generation'] == 'v054' and sources['generation_runs'] == generation_runs() and
            sources['memory_correction'] == MEMORY_CORRECTION, 'Sources identity or mapping changed')
    for name, digest in sources['pins'].items():
        require(sha(name) == digest, 'Sources pin changed since the sources draft: ' + name)
    for name, files in sources['directories'].items():
        require(inventory(name) == files, 'Sources inventory changed: ' + name)
    readout()
    return sources


def prepare_sources():
    """Phase 1 (CPU, before training ends): pin every input that does not need the M0q adapter."""
    require(not SOURCES_DRAFT.exists() and not Path(str(SOURCES_DRAFT) + '.sha256').exists(), 'Sources draft destination collision')
    readout()
    require(sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == GUARD_SOURCE_SHA and
            sha(GUARD) == GUARD_WRAPPER_SHA, 'Guard sources are not the independently approved versions')
    with (ROOT / 'results/controls_train/controls.csv').open() as stream:
        controls = [row for row in csv.DictReader(stream) if row['instance_id'] == CANARY]
    require(len(controls) == 1 and controls[0]['gold_resolved'] == 'True' and
            controls[0]['empty_resolved'] == 'False' and controls[0]['empty_exit_code'] == '1', 'Fixed canary controls changed')
    require(sha(WRAPPER) == WRAPPER_SHA and sha(training.APPROVED) == TRAINING_CONTRACT_SHA and
            read(M0_META)['initial_trainable_parameters_sha256'] == M0_INIT_SHA, 'M0q training provenance inputs changed')
    names = ['tools/run_m0q_eval_v054.py', 'tools/test_run_m0q_eval_v054.py', 'migration/m0q-eval-v054/runner.diff',
        'tools/run_mask_evaluations_v043.py', 'tools/test_run_mask_evaluations_v043.py',
        'contracts/mask-evaluations-v043-runtime-DRAFT.json', 'contracts/mask-evaluations-v043-runtime-DRAFT.json.sha256',
        'tools/run_eval_memory_guard_v043.py', 'tools/sandbox_memory_guard_v043.py', 'tools/test_sandbox_memory_guard_v043.py',
        'tools/run_mask_control_v0351.py', 'tools/test_run_mask_control_v0351.py',
        'tools/run_31b_recovery_screen_v034.py', 'tools/test_run_31b_recovery_screen_v034.py',
        'tools/run_v032_probes.py', 'tools/prepare_v032_probes.py', 'tools/recover_eval_v025.py',
        'tools/overnight_v02.py', 'tools/analyze_contract_v02_v025.py', 'tools/mechanism_metrics_v032.py',
        'tools/analyze_31b_recovery_screen_v034.py', 'tools/test_analyze_31b_recovery_screen_v034.py',
        'contracts/mask-control-v035-readout.json', 'contracts/mask-control-v035-readout.json.sha256',
        'contracts/mask-control-v0351-APPROVED.json', 'sft/train_qatbase_v053.py', 'sft/runs/A1_M0_v035.meta.json',
        'sft/runs/A1_M0_v035/adapter_config.json', 'sft/runs/A1_M0_v035/adapter_model.safetensors',
        'sft/data/mask_control_v035/manifest.json', 'sft/data/mask_control_v035/update_groups.json',
        'models/gemma-4-e4b-it-qat-dequant-bf16/manifest.json',
        'data/competition/snapshots/' + CANARY + '.tgz', 'results/controls_train/controls.csv']
    pins = {str(ROOT / name): sha(ROOT / name) for name in names}
    pins.update({str(path): sha(path) for path in FROZEN_TRAINERS})
    require(pins[str(V043_RUNTIME)] == V043_RUNTIME_SHA, 'Source v043 runtime contract changed')
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
    runs = generation_runs()
    doc = dict(status='SOURCES_DRAFT_ADAPTER_PENDING', root=str(ROOT), total_seconds=TOTAL, cleanup_seconds=RESERVE,
        generation='v054', generation_runs=runs, memory_correction=MEMORY_CORRECTION,
        guard_source_sha256=GUARD_SOURCE_SHA, guard_wrapper_sha256=GUARD_WRAPPER_SHA,
        required_controls_receipt=str(CONTROLS_RECEIPT), required_replay_receipt=str(REPLAY_RECEIPT),
        readout_sha256=READOUT_SHA, pins=pins, directories={str(ROOT / 'models' / BASE): inventory(ROOT / 'models' / BASE)},
        server_environment=read(V043_RUNTIME)['server_environment'], canary_task=CANARY,
        aliases=[run['adapter_alias'] for run in runs], adapter_source=str(ADAPTER), server_manifest=str(MANIFEST),
        run_order=[run['name'] for run in runs],
        training_receipts=dict(substitution='v053 wrapper meta.json and train_log.jsonl inside the adapter dir replace the v0351 '
                               'training evidence (completed.json, per-arm receipts, service receipts)', wrapper=str(WRAPPER),
                               wrapper_sha256=WRAPPER_SHA, training_contract_sha256=TRAINING_CONTRACT_SHA, m0_init_sha256=M0_INIT_SHA,
                               dequant_base=str(DEQUANT)),
        serving='LoRA on: serve_vllm.py with a three-adapter manifest (enable_lora, max_loras 3, max_lora_rank 16); v043 registered '
                'six adapters (max_loras 6); chat template = the v043 default, so no template byte-check stage',
        composition_check=dict(path=str(COMPOSITION), pinned='by --prepare if present'),
        runner_diff=dict(path=str(RUNNER_DIFF), sha256=pins[str(RUNNER_DIFF)],
                         source_runner=str(ROOT / 'tools/run_mask_evaluations_v043.py'),
                         source_runner_sha256=pins[str(ROOT / 'tools/run_mask_evaluations_v043.py')]),
        pending='adapter files, meta.json, train_log.jsonl and the composition check are pinned by --prepare after training',
        policy='Three routing/tool canaries, then three fixed48 M0q runs; no efficacy stopping, analysis, submission or training',
        server_handoff='Verified TERM-only shutdown; qualification uses immutable recorded lifecycle, not server liveness',
        gpu_operations_during_preparation=False)
    write_json(SOURCES_DRAFT, doc)
    digest = sha(SOURCES_DRAFT)
    with Path(str(SOURCES_DRAFT) + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + SOURCES_DRAFT.name + '\n')
    return digest


def prepare(sources_sha):
    """Phase 2 (after training): bind the qualified M0q adapter and its receipts to the unchanged sources draft."""
    require(not RUNTIME.exists() and not Path(str(RUNTIME) + '.sha256').exists(), 'Runtime destination collision')
    sources = verify_sources(sources_sha)
    qualified = qualified_m0q()
    pins = dict(sources['pins'], **qualified['artifacts'])
    composition = dict(path=str(COMPOSITION), present=COMPOSITION.is_file())
    if composition['present']:
        composition['sha256'] = pins[str(COMPOSITION)] = sha(COMPOSITION)
    doc = dict(sources, status='DRAFT_PREPARED_ONLY', sources_sha256=sources_sha, pins=pins, m0q_training=qualified,
               composition_check=composition, pending=None)
    write_json(RUNTIME, doc)
    digest = sha(RUNTIME)
    with Path(str(RUNTIME) + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + RUNTIME.name + '\n')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-sources', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--sources-sha')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--controls-receipt-sha')
    parser.add_argument('--replay-receipt-sha')
    args = parser.parse_args()
    require(sum((args.prepare_sources, args.prepare, args.activate, args.preflight)) <= 1, 'Modes are separate')
    if args.preflight:
        report = preflight()
        print(json.dumps(report, indent=1))
        sys.exit(0 if report['ready'] else 3)
    if args.prepare_sources:
        print(json.dumps(dict(status='sources_prepared_only', sources_sha256=prepare_sources(), gpu_operations=False)))
    elif args.prepare:
        require(args.sources_sha, 'Runtime preparation requires the sources draft SHA')
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(args.sources_sha), gpu_operations=False)))
    elif args.activate:
        require(args.runtime_sha, 'Activation requires runtime SHA')
        activate(args.runtime_sha, args.controls_receipt_sha, args.replay_receipt_sha)
    elif args.sources_sha:
        verify_sources(args.sources_sha)
        print(json.dumps(dict(status='sources_draft_verified', gpu_operations=False)))
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        verify_runtime(args.runtime_sha)
        print(json.dumps(dict(status='prepared_only', gpu_operations=False)))


if __name__ == '__main__':
    main()
