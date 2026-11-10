"""Matched-context v064c: the A_onpol confirmation adapter (second training seed, LoRA init seed 20261104; prereg v064-r2 confirmation trigger, readings v069) under the matched serving form of v055."""
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
# RUNTIME is the pinned contract the tools read (written by tools/pin_v064c_adapter.py beside the staged draft, never over it).
RUNTIME = ROOT / 'contracts/matched-context-v064c-runtime-PINNED.json'
RUNTIME_DRAFT = ROOT / 'contracts/matched-context-v064c-runtime-DRAFT.json'
PACKAGE = ROOT / 'migration/matched-context-v064c'
BUILD = ROOT / 'analysis/matched-context-boss-v064c'
TASKS = PACKAGE / 'tasks.json'
EVIDENCE = PACKAGE / 'evaluation'
MANIFEST = ROOT / 'subs/v064_A_onpol_c2_manifest'
LOCKS = (ROOT / 'migration/recovery-screen-v034/evaluation.lock', ROOT / 'migration/mask-control-v0351/training.lock')
GUARD = ROOT / 'tools/run_eval_memory_guard_v043.py'
GUARD_SOURCE_SHA = '5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GUARD_WRAPPER_SHA = 'c04928a3406c1175098f1217dd7b955dcec61aec1147537fbd68bf59de8bbbd5'
CONTROLS_RECEIPT = ROOT / 'analysis/oom-controls-v043/final48/report.json'
REPLAY_RECEIPT = ROOT / 'analysis/oom-replay-v043/report.json'
SERVE = ROOT / 'tools/serve_vllm_matched_v055.py'
SERVE_PARENT = ROOT / 'tools/serve_vllm_matched_v051.py'
SERVE_PARENT_SHA = '683c7ab9c53f1b66a248f019e2e8a13b6df1ac12ad74b53df0bb5f6b862c8633'
SERVE_ORIGINAL = ROOT / 'serve_vllm.py'
SERVE_ORIGINAL_SHA = 'c4b513cc5bb87d4bc3741623a4bedac41cfca5b8bdf95c0a4743b8f9371d576a'
TEMPLATE = ROOT / 'analysis/matched-context-boss-v049/chat_template.matched.jinja'
TEMPLATE_SHA = '855bbbbbef10d51177c5d7eca19a3f3d39a11d9a80c142a9f19784029515a437'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md'
PREREG_SHA = '0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60'
RUNNER_PARENT = ROOT / 'tools/run_matched_context_v064a.py'
RUNNER_PARENT_SHA = '0b48720497ca998fbc117af8277b9c22e42006591bc2912e1f4c06242d5e8ad3'
# The serving-form comparator is the v055 generation (M0q adapter, matched template): its runner, runtime contract, evidence, manifest and
# analysis outputs. The analysis comparator (slot M0) is the sealed v066 A0mc generation, bound by the analyzer (see SLOT_BINDING).
COMPARATOR_RUNNER = ROOT / 'tools/run_matched_context_v055.py'
COMPARATOR_RUNNER_SHA = 'a24ebfdb154bc3034934d2df9b2a0123fbd038adaeaa105eba39d8d9340f76dd'
# Templates of this generation (readings v069, Arm and evaluation: the v064a machinery re-pinned): the v064a runtime and analysis contracts.
PARENT_RUNTIME = ROOT / 'contracts/matched-context-v064a-runtime-PINNED.json'
PARENT_RUNTIME_SHA = 'fc8953fe54cce583e1b153dcb5a459dd3cc47355ce5fff9d7c71b42e60765afb'
PARENT_ANALYSIS = ROOT / 'contracts/matched-context-v064a-analysis-PINNED.json'
PARENT_ANALYSIS_SHA = 'ae461ea43882a09a3d2ef2e7adbc0121a772eb287c2ca291b44cf5cc82199cc4'
COMPARATOR_RUNTIME = ROOT / 'contracts/matched-context-v055-runtime-DRAFT.json'
COMPARATOR_RUNTIME_SHA = '776c7f7b4810437e285003492be9cc6465cc7c54eb7d9ab59a76f2317a9038d7'
COMPARATOR_EVIDENCE = ROOT / 'migration/matched-context-v055/evaluation'
COMPARATOR_MANIFEST = ROOT / 'subs/v055_M0qmc_manifest'
COMPARATOR_ANALYSIS = ROOT / 'analysis/matched-context-v055/report.json'
COMPARATOR_ANALYSIS_SHA = '163990653084e14e8feb08c1a04b920222d4900f18f8868cdda06d47966713c4'
COMPARATOR_PER_TASK = ROOT / 'analysis/matched-context-v055/per_task.jsonl'
COMPARATOR_PER_TASK_SHA = '682e3492b1ff45bb1b48f0bf53b06d96d93dcb9780a1ffd0b8fed624aeb57b1e'
AUDIT = ROOT / 'analysis/serialization-audit-boss-v046/audit.py'
AUDIT_SHA = '5d26f4cdc9aafa2e9ffc1f527fd88d1ccd58497770f6b4a626b04ef5078cee98'
VERIFY = ROOT / 'analysis/matched-context-boss-v049/verify_matched.py'
VERIFY_SHA = '23f5f6d2074b2e5de2a6fd0a5e072f3c9defb56537d990a6e7e8e57d43e91004'
BYTE_TRACE = ROOT / 'results/v043_M0_s1/traces/trace_rich_2725.json'
BYTE_CALLS = (1, 3, 7)
BYTE_ALIAS = 'v064_A_onpol_c2_s1'
BYTE_EXPECTED = PACKAGE / 'byte-check-expected.json'
SERVE_DIFF, RUNNER_DIFF = ROOT / 'migration/matched-context-v055/serve.diff', BUILD / 'runner.diff'  # the v055 serve script is reused by pin
TOKENIZE_URL = 'http://127.0.0.1:8000/tokenize'
FORMAT_TEXT = 'Detected the chat template content format to be '
SLOT_BINDING = dict(M0='v066_A0mc_s1..s3: comparator (sealed v066 generation, E4B QAT base without adapter), matched template',
                    M1='v064_A_onpol_c2_s1..s3: A_onpol confirmation adapter (LoRA init seed 20261104), matched template',
                    sign='M1 - M0 = A_onpol_c2 - A0mc', statistics='tools/analyze_mask_control_v042.py sha 23915fb0, ARMS (M0, M1) unchanged')
TOTAL, RESERVE = 28800, 600
CANARY = 'fastapi_11194'
ARM, SEEDS = 'A_onpol', (1, 2, 3)
# A_onpol confirmation receipts (second training seed, LoRA init seed 20261104; addendum b602a5d3 section 2, readings v069): the
# confirmation wrapper's own meta.json/train_log.jsonl, written only at the end of the run. The adapter hashes are not code constants:
# they live in the pinned runtime contract (the pending marker in the staged draft until the pin step).
ADAPTER = ROOT / 'sft/runs/A1_onpol_s2_v064'
MODE, PENDING = 'v064r2-A_onpol-s2-main', 'PENDING_TRAINING'
ADAPTER_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'meta.json', 'train_log.jsonl')
RECEIPT = 'confirmation wrapper meta.json + train_log.jsonl are the training receipts'
WRAPPER = ROOT / 'sft/train_onpolicy_s2_v064r2.py'
WRAPPER_SHA = '8a43a115d67441d4e32497de215dda36cbd7cecfc76f2166f14b6a3f3f466ab4'
# The approved confirmation contract (path fixed by Boss) exists only after its own pin and approval: its sha is a pending field of the
# staged draft until the pin step. It must pin the approved v064-r2 A_onpol recipe (the parent contract below) with only the LoRA init seed changed.
TRAINING_CONTRACT = ROOT / 'contracts/onpolicy-confirmation-v064r2-APPROVED.json'
PARENT_TRAINING_CONTRACT = ROOT / 'contracts/onpolicy-training-v064r2-APPROVED.json'
PARENT_TRAINING_CONTRACT_SHA = '8e13817b0ff51d60046f97b47df1eafbf3b11f086f8c2696eb0e96fa2813fd24'
SEED = 20261104  # the LoRA init seed of the confirmation: its init hash is recorded and must differ from M0_INIT_SHA
# The confirmation contract's own prereg (the frozen v069 readings addendum); its pinned files are the parent's two plus the v064-r2 prereg
# and the parent contract (the layout of the confirmation DRAFT bc82c2d3).
READINGS = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-08-onpolicy-confirmation-readings-v069.md'
READINGS_SHA = 'fabeddb2c62e9df81f68c86355ddec7d6caa083fda788dcee61030f84373385b'
M0_CONTRACT_SHA = '02e0a25ebb6c74220caf1c590fc08da46833399d3e8eab53fe174e9fd8bb0fde'
M0_ADAPTER, M0Q_META = ROOT / 'sft/runs/A1_M0_v035', ROOT / 'sft/runs/A1_M0q_v053/meta.json'
M0_INIT_SHA = '9dfc7ccc71fc4631d51a61dad8a889040e76bd9c87a2546fe24ec3b769e20066'
DEQUANT = ROOT / 'models/gemma-4-e4b-it-qat-dequant-bf16'
DEQUANT_MANIFEST_SHA = '20f1c2f9a52eea099c7ab95fca57c150d4e3b1946663136f11d11e0f50849ce5'
DATA = ROOT / 'sft/data/onpolicy_v064/A_onpol'
DATA_FILES = ('trajectories.jsonl', 'M0.jsonl', 'identity_ledger.jsonl', 'update_groups.json', 'manifest.json')
PINNED_FILES = dict(
    addendum=dict(path=str(ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065.md'),
                  sha256='b602a5d366002bc5335fcc7a94ada72cba71535b3a63551f74d28bd9a8f39a56'),
    eligible=dict(path=str(ROOT / 'migration/onpolicy-v064/eligible.json'), sha256='78b856a2382068b86a1f5b3ea0624ef01b5962f4b1c98ecdf38f32edeb082625'))
SUMMARY_KEYS = ('mode', 'initial_trainable_parameters_sha256', 'lora_modules', 'adapter_tensors', 'names_match_probe', 'updates',
                'loss_first', 'loss_last', 'seconds', 'gates_pass')
FROZEN_CODE = dict(trainer=ROOT / ('sft/train_' + 'lora.py'), codex_trainer=ROOT / 'sft/train_mask_control_v035.py',
                   prepare=ROOT / 'sft/prepare_mask_control_v035.py', packer=ROOT / 'sft/build_sft.py',
                   converter=ROOT / 'sft/onpolicy_v064/convert_traces_r2.py', build_arms=ROOT / 'sft/onpolicy_v064/build_arms_r2.py')
COMPOSITION = ROOT / 'analysis/qat-transfer-v053/composition-check.json'
COMPOSITION_SHA = '0a9e835f056ec819a2ec2370dddf125d4a7d3be7020154630f3f8aea70cd8999'
DISK_MIN_FREE = 50 * 2 ** 30
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
        name, alias = 'v064_' + ARM + '_c2_s' + str(seed), 'v064_' + ARM + '_c2_s' + str(seed)
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
        unit = 'gemma-v064c-eval-' + uuid.uuid4().hex + '.service'
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
    """Leaves of a contract that still hold the pending marker (the A_onpol adapter pins before the pin step)."""
    return [path for path, value in leaf_paths(doc) if value == PENDING]


def verify_runtime(expected):
    doc = training.checked_json(RUNTIME, expected)
    require(Path(str(RUNTIME) + '.sha256').read_text().split()[0] == expected, 'Runtime sidecar mismatch')
    require(doc['root'] == str(ROOT) and doc['total_seconds'] == TOTAL and doc['cleanup_seconds'] == RESERVE and
            doc['generation'] == 'v064c', 'Runtime identity/bounds changed')
    require(not pending_paths(doc), 'Adapter pins are still pending training; the pin step has not run')
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
    require(doc['generation_runs'] == generation_runs() and doc['memory_correction'] == MEMORY_CORRECTION,
            'Scientific mapping or memory correction changed')
    artifacts = doc['arm_training']['artifacts']
    require(doc['adapter'] == dict(path=str(ADAPTER), sha256=artifacts[str(ADAPTER / 'adapter_model.safetensors')],
                                   meta_sha256=artifacts[str(ADAPTER / 'meta.json')]) and
            all(doc['pins'].get(name) == digest for name, digest in artifacts.items()) and
            doc['composition_check'] == dict(path=str(COMPOSITION), sha256=COMPOSITION_SHA) and doc['pins'][str(COMPOSITION)] == COMPOSITION_SHA,
            'A_onpol adapter or composition-check identity changed')
    return doc, effective_readout(frozen)


def qualify_service(folder, name, expected_end=None):
    start_path = folder / (name + '.log.service-start.json')
    end_path = folder / (name + '.log.service-end.json')
    if expected_end is not None:
        require(sha(end_path) == expected_end, 'Service receipt hash changed')
    start, end = read(start_path), read(end_path)
    identity = {key: start[key] for key in ('InvocationID', 'ControlGroup')}
    prefix = 'v064c' if folder == EVIDENCE else 'v034'
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


def run_artifacts():
    """The files the pin step hashes from the finished run: the adapter and its receipts, the arm's data files and the approved contract."""
    return [ADAPTER / name for name in ADAPTER_FILES] + [DATA / name for name in DATA_FILES] + [TRAINING_CONTRACT]


def approved_training():
    """The approved confirmation contract: the approved v064-r2 A_onpol recipe (parent contract 8e13817b) with the LoRA init seed 20261104 and
    this run dir. Generation, revision, variant, base, code, sealed M0q run, reference contract, data leaf and update count U equal the parent's;
    its prereg is the frozen v069 readings addendum, its pinned files are the parent's two plus the v064-r2 prereg and the parent contract,
    each equal to its file, and its init hash rule names M0's hash as the reference the recorded one must differ from."""
    require(TRAINING_CONTRACT.is_file(), 'Approved training contract unavailable: ' + str(TRAINING_CONTRACT))
    require(sha(PARENT_TRAINING_CONTRACT) == PARENT_TRAINING_CONTRACT_SHA, 'Parent (v064-r2) training contract changed')
    contract, parent = read(TRAINING_CONTRACT), read(PARENT_TRAINING_CONTRACT)
    entry, original = contract['arms'][ARM], parent['arms'][ARM]
    leaf, updates, init_rule = entry['data'], entry['stages']['main']['updates'], contract.get('init_hash')
    pinned = dict(PINNED_FILES, v064r2_prereg=dict(path=str(PREREG), sha256=PREREG_SHA),
                  parent_contract=dict(path=str(PARENT_TRAINING_CONTRACT), sha256=PARENT_TRAINING_CONTRACT_SHA))
    require(contract['status'] == 'APPROVED' and contract['generation'] == parent['generation'] == 'v064' and
            contract['revision'] == parent['revision'] == 'r2' and contract['wrapper'] == dict(path=str(WRAPPER), sha256=WRAPPER_SHA) and
            contract['variant'] == parent['variant'] and contract['hyperparameters'] == dict(parent['hyperparameters'], seed=SEED) and
            contract['prereg'] == dict(path=str(READINGS), sha256=READINGS_SHA) and sha(READINGS) == READINGS_SHA and
            contract['pinned_files'] == pinned and all(sha(value['path']) == value['sha256'] for value in pinned.values()) and
            isinstance(init_rule, dict) and init_rule.get('m0_reference') == M0_INIT_SHA and
            contract['base'] == parent['base'] == dict(dir=str(DEQUANT), manifest_sha256=DEQUANT_MANIFEST_SHA) and
            contract['code_sha256'] == parent['code_sha256'] == {name: sha(path) for name, path in FROZEN_CODE.items()} and
            contract['sealed_runs'] == parent['sealed_runs'] == dict(m0q=dict(path=str(M0Q_META), sha256=sha(M0Q_META))) and
            contract['reference_contract']['sha256'] == parent['reference_contract']['sha256'] == M0_CONTRACT_SHA and
            entry['stages']['main']['out'] == str(ADAPTER) and isinstance(leaf, dict) and leaf == original['data'] and
            type(updates) is int and updates > 0 and updates == original['stages']['main']['updates'] == leaf['updates'] and
            leaf['dir'] == str(DATA) and leaf['expected_sha256'] == {name: sha(DATA / name) for name in DATA_FILES},
            'Approved confirmation contract does not pin the v064-r2 A_onpol recipe with seed 20261104, this run dir and its data')
    return contract, leaf, updates


def qualified_arm():
    """The confirmation wrapper's own receipts (meta.json, train_log.jsonl) checked against the approved confirmation contract, the sealed M0q
    run, the arm's pinned data files and M0's adapter layout: the v064a gates with the init hash recorded (not M0's) and seed 20261104."""
    paths = [ADAPTER / name for name in ADAPTER_FILES]
    require(all(path.is_file() for path in paths), ARM + ' confirmation adapter or training receipts unavailable')
    contract, leaf, updates = approved_training()
    meta, m0q = read(ADAPTER / 'meta.json'), read(M0Q_META)
    init = meta['initial_trainable_parameters_sha256']
    population = {key: leaf[key] for key in ('sequences', 'updates', 'shifted_targets', 'input_tokens')}
    require(meta['mode'] == MODE and meta['arm'] == ARM and meta['wrapper'] == contract['wrapper'] and WRAPPER_SHA == sha(WRAPPER) and
            meta['training_contract'] == dict(path=str(TRAINING_CONTRACT), sha256=sha(TRAINING_CONTRACT), status='APPROVED') and
            meta['prereg'] == contract['prereg'] and meta['pinned_files'] == contract['pinned_files'] and
            meta['m0_contract_sha256'] == M0_CONTRACT_SHA and meta['gates_pass'] is True and meta['init_hash_equals_m0'] is False and
            isinstance(init, str) and re.fullmatch('[0-9a-f]{64}', init) is not None and init != M0_INIT_SHA and
            meta['m0_initial_trainable_parameters_sha256'] == m0q['initial_trainable_parameters_sha256'] == M0_INIT_SHA and
            meta['lora_modules'] == 258 and meta['adapter_tensors'] == 516 and meta['names_match_probe'] is True and
            meta['updates'] == meta['updates_planned'] == updates and meta['overrides'] == dict(vs_m0q=['out', 'seed']) and
            meta['data_dir'] == str(DATA) and meta['data_files_sha256'] == leaf['expected_sha256'] and
            meta['frozen_code_sha256'] == contract['code_sha256'] and meta['base']['kind'] == 'dequant' and meta['base']['dir'] == str(DEQUANT) and
            meta['base']['manifest_sha256'] == sha(DEQUANT / 'manifest.json') == DEQUANT_MANIFEST_SHA and meta['sequences'] == leaf['sequences'] and
            isinstance(meta['population'], dict) and {key: meta['population'].get(key) for key in population} == population,
            ARM + ' confirmation wrapper receipt, gates or provenance changed')
    keep = lambda args: {k: v for k, v in args.items() if k not in ('out', 'seed')}
    require(keep(meta['args']) == keep(m0q['args']) and meta['args']['base'] == str(DEQUANT) and meta['args']['out'] == str(ADAPTER) and
            meta['args'].get('seed') == SEED != m0q['args'].get('seed'),
            ARM + ' confirmation trainer arguments differ from the sealed M0q run beyond the output and the init seed')
    require(all(type(meta[key]) in (int, float) and math.isfinite(meta[key]) and meta[key] > 0 for key in ('seconds', 'tokens')),
            'Nonfinite or empty ' + ARM + ' trainer summary')
    config, m0_config = read(ADAPTER / 'adapter_config.json'), read(M0_ADAPTER / 'adapter_config.json')
    require({k: v for k, v in config.items() if k != 'base_model_name_or_path'} ==
            {k: v for k, v in m0_config.items() if k != 'base_model_name_or_path'}, ARM + ' adapter config differs from M0 beyond the base string')
    tensors, reference = training.safe_header(paths[0]), training.safe_header(M0_ADAPTER / 'adapter_model.safetensors')
    require(len(tensors) == 516 and {k: (v['shape'], v['dtype']) for k, v in tensors.items()} ==
            {k: (v['shape'], v['dtype']) for k, v in reference.items()}, ARM + ' adapter tensors differ from M0 in names, shapes or dtypes')
    logs = [json.loads(line) for line in paths[3].read_text().splitlines() if line.strip()]
    groups = read(DATA / 'update_groups.json')
    require(len(logs) == len(groups) == updates, ARM + ' training updates incomplete')
    for index, (row, group) in enumerate(zip(logs, groups), 1):
        require(row['step'] == index and row['loss_tokens'] == row['remaining_loss_tokens'] == group['original_denominator'] ==
                group['m1_remaining_targets'] and row['sequences'] == len(group['sequence_indices']), 'Frozen update exposure mismatch')
        for key in ('loss', 'lr', 'grad_norm', 'seconds', 'tokens_per_s', 'peak_alloc_gib'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]), 'Nonfinite training metric: ' + key)
    require(meta['tokens'] == sum(row['tokens'] for row in logs), 'Training token summary mismatch')
    return dict(artifacts={str(path): sha(path) for path in run_artifacts()}, initial_sha256=meta['initial_trainable_parameters_sha256'],
                updates=len(logs), adapter_tensors=len(tensors), base_model_name_or_path=config['base_model_name_or_path'],
                m0_base_model_name_or_path=m0_config['base_model_name_or_path'], dequant_manifest_sha256=meta['base']['manifest_sha256'],
                wrapper_sha256=WRAPPER_SHA, training_contract_sha256=sha(TRAINING_CONTRACT), data_files_sha256=meta['data_files_sha256'],
                summary={key: meta[key] for key in SUMMARY_KEYS}, receipt=RECEIPT)


def pending_qualified():
    """qualified_arm before the run ends: the fields that come from the finished run, the arm data and the approved contract hold the pending
    marker, the rest is final."""
    return dict(artifacts={str(path): PENDING for path in run_artifacts()}, initial_sha256=PENDING, updates=PENDING,
                adapter_tensors=PENDING, base_model_name_or_path=PENDING,
                m0_base_model_name_or_path=read(M0_ADAPTER / 'adapter_config.json')['base_model_name_or_path'],
                dequant_manifest_sha256=sha(DEQUANT / 'manifest.json'), wrapper_sha256=WRAPPER_SHA,
                training_contract_sha256=PENDING, data_files_sha256={name: PENDING for name in DATA_FILES},
                summary={key: PENDING for key in SUMMARY_KEYS}, receipt=RECEIPT)


def submission_agent(text, alias):
    marker = 'model: ' + BASE + '\n'
    require(text.count(marker) == 1 and '\nadapter:' not in text, 'Unexpected original agent config')
    return text.replace(marker, 'model: ' + alias + '\n')


def destinations(frozen):
    paths = [EVIDENCE, MANIFEST, ROOT / 'logs/v064c_server.log']
    for run in frozen['future_runs']:
        paths += [Path(run['submission_dir']), Path(run['result_dir'])]
        for name in (run['name'], 'v064c_canary_' + run['name']):
            paths += [ROOT / ('logs/' + name + '.log'), ROOT / ('logs/' + name + '.log.routing.json')]
        paths += [ROOT / ('results/v064c_canary_' + run['name'])]
    return paths


def outside_adapter(files, agent=False):
    """Staged file hashes without the adapter directories and, unless asked, the alias-bearing agent.yaml (both differ from v055 by construction)."""
    return {name: digest for name, digest in files.items() if not name.startswith('adapters/') and (agent or name != 'agent.yaml')}


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
                require(sha(source / name) == qualified['artifacts'][str(source / name)],
                        'Qualified adapter source changed before copy')
                shutil.copyfile(source / name, destination / name)
                require(sha(destination / name) == sha(source / name), 'Adapter copy changed')
        expected_files = set(inventory(baseline)) | {
            'adapters/' + run['adapter_alias'] + '/' + name for name in ('adapter_config.json', 'adapter_model.safetensors')}
        require(set(inventory(target)) == expected_files, 'Unexpected submission file set')
        for name, digest in inventory(baseline).items():
            require((target / name).read_text() == submission_agent((baseline / name).read_text(), run['adapter_alias'])
                    if name == 'agent.yaml' else sha(target / name) == digest, 'Non-alias config bytes changed')
        require(outside_adapter(inventory(target)) == outside_adapter(
                comparator_staged[str(ROOT / 'subs' / run['name'].replace('v064_A_onpol_c2_', 'v055_M0qmc_', 1))]),
                'Staged submission differs from the v055 M0qmc submission bytes outside the adapter and alias')
    require(outside_adapter(inventory(MANIFEST), agent=True) == outside_adapter(comparator_staged[str(COMPARATOR_MANIFEST)], agent=True),
            'Staged manifest differs from the v055 M0qmc manifest outside the adapters')
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
    with (ROOT / 'logs/v064c_server.log').open('x') as stream:
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
    """The recorded v055 vLLM argv (template flag included) with this generation's aliases (--max-loras, --lora-modules)."""
    start, end, loras = recorded.index('--lora-modules'), recorded.index('--served-model-name'), recorded.index('--max-loras')
    require(loras < start < end and recorded[-2:] == ['--chat-template', str(TEMPLATE)] and '--chat-template' not in recorded[:-2],
            'Unexpected v055 vLLM argv layout')
    argv = recorded[:start + 1] + [alias + '=' + str(MANIFEST / 'adapters' / alias) for alias in sorted(aliases)] + recorded[end:]
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
    identity = re.fullmatch(r'v064_(A_onpol)_c2_s([123])', alias)
    require(identity is not None and receipt['arm'] == identity[1] and type(receipt['decoding_seed']) is int and
            receipt['decoding_seed'] == int(identity[2]) and receipt['runtime_sha256'] == sha(RUNTIME) and
            receipt['staged_sha256'] == sha(EVIDENCE / 'staged-inputs.json'), 'Run arm/seed/source binding changed')
    require(receipt['generation'] == 'v064c' and receipt['scientific_name'] == alias and
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
    require(completed['status'] == 'all_three_a_onpol_c2_evaluations_qualified' and completed['runtime_sha256'] == runtime_sha and completed['readout_sha256'] == READOUT_SHA and
            completed['cleanup_completed'] is True and completed['analysis_pending'] is True and
            completed['submission_started'] is False and completed['elapsed_seconds'] <= TOTAL,
            'Generation completion provenance/scope/deadline failed')
    require(completed['generation'] == 'v064c' and completed['memory_correction'] == MEMORY_CORRECTION,
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
    require(completed['arm_training'] == doc['arm_training'] == qualified_arm(), 'A_onpol training source changed')
    runs = {}
    budget = frozen['evaluation_config']['effective_budget']
    ids = frozen['populations']['all48_sensitivity']['task_ids']
    for run in frozen['future_runs']:
        name = run['name']
        canary = 'v064c_canary_' + name
        check = qualify_run(canary, run['adapter_alias'], [CANARY], ROOT / ('results/' + canary),
                            ROOT / ('logs/' + canary + '.log'), True, budget)
        require(check['receipt'] == completed['canaries'][name], 'Canary/final completion disagreement')
        checked = qualify_run(name, run['adapter_alias'], ids, Path(run['result_dir']),
                              ROOT / ('logs/' + name + '.log'), False, budget)
        require(checked['receipt'] == completed['runs'][name], 'Run/final completion disagreement')
        runs[name] = checked
    require(set(completed['runs']) == set(runs) == set(completed['canaries']), 'Incomplete three-alias generation')
    return doc, readout(), completed, runs


def activate(runtime_sha, controls_sha=None, replay_sha=None):
    doc, frozen = verify_runtime(runtime_sha)
    admission = admission_receipts(doc, controls_sha, replay_sha)
    recovery = load_recovery(ROOT)
    qualified = qualified_arm()
    require(qualified == doc['arm_training'], 'A_onpol training receipt changed since preparation')
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
            qualified = qualified_arm()
            require(qualified == doc['arm_training'], 'A_onpol training receipt changed since preparation')
            training.assert_idle(recovery)
            memory = host_memory_admission()
            cursor = kernel_cursor()
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
            require(not gpu.strip(), 'GPU compute processes remain after training')
            require(not any(line.split()[1] == '0100007F:1F40' and line.split()[3] == '0A'
                    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]), 'Serving listener already occupied')
            write_json(EVIDENCE / 'activation.json', dict(runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                       arm_training=qualified, started_at=recovery.utc(), total_seconds=TOTAL,
                       generation='v064c', memory_correction=MEMORY_CORRECTION, host_memory=memory, kernel_cursor=cursor,
                       admission_receipts=admission, user_manager_failed_units=failed_units(),
                       failed_units_policy='record only; historical failed units do not block (team lead, 2026-11-05)'))
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
                name = 'v064c_canary_' + run['name'] if strict else run['name']
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
                    generation='v064c', scientific_name=run['scientific_name'], result_dir=str(output),
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
                name = 'v064c_canary_' + run['name'] if strict else run['name']
                output = ROOT / ('results/' + name) if strict else Path(run['result_dir'])
                log = ROOT / ('logs/' + name + '.log')
                qualify_run(name, run['adapter_alias'], [CANARY] if strict else ids, output, log, strict, budget)
            require(time.monotonic() <= deadline, 'Evaluation total deadline exceeded')
            write_json(EVIDENCE / 'global-oom-final.json', global_oom_check(cursor))
            hashes = {p.name: sha(p) for p in EVIDENCE.glob('*.json')}
            write_json(EVIDENCE / 'completed.json', dict(status='all_three_a_onpol_c2_evaluations_qualified', runtime_sha256=runtime_sha, readout_sha256=READOUT_SHA,
                generation='v064c', memory_correction=MEMORY_CORRECTION, byte_check_sha256=hashes['byte-check.json'],
                admission_receipts=admission,
                arm_training=qualified, cleanup_completed=True,
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


def preflight(runtime_sha=None):
    """CPU-only gate report before activation; takes no lock, launches nothing, reads no outcome; saved with exclusive create."""
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
    available, disk = int(fields['MemAvailable']) * 1024, shutil.disk_usage(ROOT).free
    collisions = [str(path) for path in destinations(effective_readout(readout())) if path.exists()]
    offenders = idle_offenders(recovery)
    try:
        adapter = dict(qualified=True, **{k: v for k, v in qualified_arm().items() if k != 'artifacts'})
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        adapter = dict(qualified=False, reason=type(error).__name__ + ': ' + str(error))
    pinned = read(RUNTIME)['pins'] if RUNTIME.is_file() else {}
    live = lambda name: (ADAPTER / name).is_file() and sha(ADAPTER / name) == pinned.get(str(ADAPTER / name))
    hashes = dict(template=sha(TEMPLATE) == TEMPLATE_SHA, adapter=live('adapter_model.safetensors'),
                  meta=live('meta.json'), composition=sha(COMPOSITION) == COMPOSITION_SHA,
                  prereg=sha(PREREG) == PREREG_SHA, server_script=SERVE.is_file() and SERVE.read_bytes() == SERVE_PARENT.read_bytes())
    runtime = dict(present=RUNTIME.is_file(), sha256=sha(RUNTIME) if RUNTIME.is_file() else None, verified=None)
    if runtime_sha is not None:
        try:
            verify_runtime(runtime_sha)
            runtime['verified'] = True
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            runtime['verified'] = False
            runtime['reason'] = type(error).__name__ + ': ' + str(error)
    ready = (bool(gpu) and int(gpu[0]) <= 100 and not apps and port == 0 and not held and not offenders and not collisions and
             available >= MEMORY_CORRECTION['host_min_available_bytes'] and disk >= DISK_MIN_FREE and adapter['qualified'] and
             all(hashes.values()) and runtime['verified'] is True)
    report = dict(ready=ready, checked_at=recovery.utc(), gpu_memory_used_mib=int(gpu[0]) if gpu else None, gpu_compute_apps=apps,
                  port_8000_listeners=port, locks_held=held, mem_available_bytes=available, disk_free_bytes=disk,
                  disk_min_free_bytes=DISK_MIN_FREE, destination_collisions=collisions, idle_offenders=offenders,
                  failed_units=failed_units(), arm_adapter=adapter, pinned_hashes_equal=hashes, runtime_contract=runtime,
                  analysis_manifest_present=(ROOT / 'contracts/matched-context-v064c-analysis-PINNED.json').is_file())
    PACKAGE.mkdir(exist_ok=True)
    path = PACKAGE / ('preflight-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.json')
    with path.open('x') as stream:
        stream.write(json.dumps(report, indent=1, sort_keys=True) + chr(10))
    return dict(report, receipt=str(path), receipt_sha256=sha(path))


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
    spec = importlib.util.spec_from_file_location('v064c_' + Path(path).stem, path)
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
                          request=dict(model=BYTE_ALIAS, messages=served['messages'], tools=rebuilt['tools'],
                                       chat_template_kwargs=rebuilt['kwargs'], add_generation_prompt=True)))
    require([call['differs_from_served'] for call in calls] == [False, True, True] and all(call['edits_only'] for call in calls),
            'Byte-check calls do not discriminate the matched template as verified in v049')
    doc = dict(schema='matched-context-v064c-byte-check-expected', trace=str(BYTE_TRACE), trace_sha256=sha(BYTE_TRACE), trace_model=model,
               template=str(TEMPLATE), template_sha256=TEMPLATE_SHA, audit=str(AUDIT), audit_sha256=AUDIT_SHA,
               verify=str(VERIFY), verify_sha256=VERIFY_SHA, tokenizer=audit.SERVED_DIR, endpoint=TOKENIZE_URL,
               comparison='POST /tokenize body = request; response tokens must equal matched_ids exactly', calls=calls)
    return json.dumps(doc, indent=1, sort_keys=True) + '\n'


DEVIATIONS = [
    'Arm: the A_onpol confirmation (second training seed, LoRA init seed 20261104; addendum b602a5d3 section 2), three explicit runs v064_A_onpol_c2_s{1,2,3} served as v064_A_onpol_c2_s{n} from sft/runs/A1_onpol_s2_v064, generation v064c; the receipt arm stays A_onpol. The run definitions are the v064a ones (the v055 ones); contract_doc checks that the v055 runs equal them except name, scientific_name, arm, adapter_alias, adapter_source and the paths.',
    'Training coupling: qualified_arm follows the confirmation wrapper (sft/train_onpolicy_s2_v064r2.py): mode v064r2-A_onpol-s2-main; the init hash is recorded (64 lowercase hex characters, different from M0 9dfc7ccc) with init_hash_equals_m0 false and m0_initial_trainable_parameters_sha256 equal to M0; overrides {vs_m0q: [out, seed]} with seed 20261104 and every other trainer argument equal to the sealed M0q run; meta prereg and pinned files equal to the approved contract. approved_training reads contracts/onpolicy-confirmation-v064r2-APPROVED.json (the layout of DRAFT bc82c2d3) and requires the v064-r2 A_onpol recipe of the parent contract 8e13817b with only the seed changed: generation v064, revision r2, variant, base, code shas, sealed M0q run, reference contract and data leaf (files, U 43) equal to those of the parent; prereg the frozen v069 readings addendum fabeddb2; pinned files exactly the two of the parent plus the v064-r2 prereg 0893728c and the parent contract, each equal to its file; init_hash.m0_reference 9dfc7ccc; the main stage this run dir. Every other v064a gate is unchanged.',
    'Pending fields: the same 42 as v064a (adapter receipts, the five arm data files, the approved confirmation contract and their derived fields), filled by tools/pin_v064c_adapter.py from the finished run only.',
    'Parent and comparator: RUNNER_PARENT is the v064a runner (0b487204), so runner.diff is against it; the v064a PINNED runtime (fc8953fe) and analysis (ae461ea4) contracts are the templates. The serving-form reference stays v055 M0qmc; the analysis comparator (slot M0) stays the sealed v066 A0mc generation; the sealed v064a report is printed beside the confirmation by the analyzer, never pooled.',
    'Pins: the v064a runner, its test, design, byte-check expectation and runner.diff, the v064a PINNED runtime and analysis contracts with their sidecars, the parent training contract and the confirmation wrapper are pinned; the approved confirmation contract, the arm data files and the adapter receipts are pinned from the finished run; the self-pins are the v064c files.',
    'Readings: the verbatim Evaluation line of section Arms and the lines of section Readouts of prereg v064-r2 (the confirmation trigger is one of them); the confirmation sentences of the v069 readings addendum are bound by the analyzer.',
    'Names: unit prefix gemma-v064c-eval-, service receipt prefix v064c, receipts migration/matched-context-v064c/evaluation, manifest subs/v064_A_onpol_c2_manifest, server log logs/v064c_server.log, run logs logs/v064_A_onpol_c2_s{n}.log, canary logs and results v064c_canary_v064_A_onpol_c2_s{n}; completion status all_three_a_onpol_c2_evaluations_qualified; receipts key arm_training.',
    'Unchanged from v064a: everything else, including the template flag and checks, server script, byte check, memory correction and guard, admission receipts, budgets, sampling, seeds, canary task and validation, TOTAL 28800 s / RESERVE 600 s, the locks, GPU-empty and port-8000 checks, global OOM checks and TERM-only cleanup.']


def prereg_readings():
    """The Evaluation line of section Arms and the lines of section Readouts of the frozen pre-registration, verbatim."""
    text = PREREG.read_text(encoding='utf-8')
    section = lambda header: text.split(chr(10) + header, 1)[1].split(chr(10) + '## ', 1)[0].splitlines()[1:]
    return ([line for line in section('## Arms') if line.startswith('- Evaluation:')] +
            [line for line in section('## Readouts') if line.startswith('- ')])


def contract_doc(frozen, qualified):
    """The runtime contract as a dict: final everywhere except where `qualified` carries the pending marker."""
    comparator = read(COMPARATOR_RUNTIME)
    names = ['tools/run_matched_context_v055.py', 'tools/test_run_matched_context_v055.py', 'tools/serve_vllm_matched_v055.py',
        'tools/run_matched_context_v051.py', 'tools/test_run_matched_context_v051.py', 'tools/serve_vllm_matched_v051.py',
        'contracts/matched-context-v051-runtime-DRAFT.json', 'migration/matched-context-v051/byte-check-expected.json',
        'serve_vllm.py', 'tools/run_mask_evaluations_v043.py', 'tools/test_run_mask_evaluations_v043.py',
        'tools/run_m0q_eval_v054.py', 'tools/test_run_m0q_eval_v054.py', 'contracts/m0q-eval-v054-sources-DRAFT.json',
        'contracts/m0q-eval-v054-runtime-DRAFT.json', 'contracts/m0q-eval-v054-runtime-DRAFT.json.sha256',
        'migration/m0q-eval-v054/evaluation/completed.json', 'migration/m0q-eval-v054/evaluation/staged-inputs.json',
        'migration/m0q-eval-v054/evaluation/server-ready.json',
        'analysis/m0q-eval-v054/report.json', 'analysis/m0q-eval-v054/per_task.jsonl',
        'tools/run_eval_memory_guard_v043.py', 'tools/sandbox_memory_guard_v043.py', 'tools/test_sandbox_memory_guard_v043.py',
        'contracts/mask-evaluations-v043-runtime-DRAFT.json', 'contracts/mask-evaluations-v043-runtime-DRAFT.json.sha256',
        'analysis/matched-context-boss-v049/chat_template.matched.jinja', 'analysis/matched-context-boss-v049/template.diff',
        'analysis/matched-context-boss-v049/README.md', 'analysis/matched-context-boss-v049/verify_matched.py',
        'analysis/matched-context-boss-v049/verify_report.json', 'analysis/matched-context-boss-v049/SHA256SUMS',
        'analysis/serialization-audit-boss-v046/audit.py', 'analysis/serialization-audit-boss-v046/report.json',
        'results/v043_M0_s1/traces/trace_rich_2725.json', 'models/' + BASE + '/chat_template.jinja',
        'sft/runs/A1_M0q_v053/adapter_model.safetensors', 'sft/runs/A1_M0q_v053/adapter_config.json',
        'sft/runs/A1_M0q_v053/meta.json', 'sft/runs/A1_M0q_v053/train_log.jsonl',
        'analysis/qat-transfer-v053/composition-check.json', 'sft/train_qatbase_v053.py',
        'sft/runs/A1_M0_v035.meta.json', 'sft/runs/A1_M0_v035/adapter_config.json', 'sft/runs/A1_M0_v035/adapter_model.safetensors',
        'sft/data/mask_control_v035/manifest.json', 'sft/data/mask_control_v035/update_groups.json',
        'models/gemma-4-e4b-it-qat-dequant-bf16/manifest.json',
        'tools/run_mask_control_v0351.py', 'tools/test_run_mask_control_v0351.py',
        'tools/run_31b_recovery_screen_v034.py', 'tools/test_run_31b_recovery_screen_v034.py',
        'tools/run_v032_probes.py', 'tools/prepare_v032_probes.py', 'tools/recover_eval_v025.py',
        'tools/overnight_v02.py', 'tools/analyze_contract_v02_v025.py', 'tools/mechanism_metrics_v032.py',
        'tools/analyze_mechanism_v031.py', 'tools/analyze_mask_control_v042.py',
        'tools/analyze_31b_recovery_screen_v034.py', 'tools/test_analyze_31b_recovery_screen_v034.py',
        'contracts/mask-control-v0351-APPROVED.json',
        'contracts/mask-control-v035-readout.json', 'contracts/mask-control-v035-readout.json.sha256',
        'data/competition/snapshots/' + CANARY + '.tgz', 'results/controls_train/controls.csv',
        'migration/matched-context-v055/serve.diff', 'migration/matched-context-v055/runner.diff',
        'migration/matched-context-v055/runner-vs-v054.diff', 'migration/matched-context-v055/byte-check-expected.json',
        'analysis/matched-context-v055/report.json', 'analysis/matched-context-v055/per_task.jsonl',
        'contracts/matched-context-v055-runtime-DRAFT.json', 'contracts/matched-context-v055-runtime-DRAFT.json.sha256',
        'migration/matched-context-v055/evaluation/completed.json', 'migration/matched-context-v055/evaluation/staged-inputs.json',
        'migration/matched-context-v055/evaluation/server-ready.json',
        'tools/run_matched_context_v062.py', 'tools/test_run_matched_context_v062.py', 'migration/matched-context-v062/design.json',
        'migration/matched-context-v062/byte-check-expected.json', 'analysis/matched-context-boss-v062/runner.diff',
        'contracts/matched-context-v062-runtime-PINNED.json', 'contracts/matched-context-v062-runtime-PINNED.json.sha256',
        'contracts/matched-context-v062-analysis-PINNED.json', 'contracts/matched-context-v062-analysis-PINNED.json.sha256',
        'sft/train_onpolicy_v064r2.py', 'contracts/onpolicy-training-v064r2-DRAFT.json', 'migration/onpolicy-v064/eligible.json',
        'tools/run_matched_context_v064a.py', 'tools/test_run_matched_context_v064a.py', 'migration/matched-context-v064a/design.json',
        'migration/matched-context-v064a/byte-check-expected.json', 'analysis/matched-context-boss-v064a/runner.diff',
        'contracts/matched-context-v064a-runtime-PINNED.json', 'contracts/matched-context-v064a-runtime-PINNED.json.sha256',
        'contracts/matched-context-v064a-analysis-PINNED.json', 'contracts/matched-context-v064a-analysis-PINNED.json.sha256',
        'contracts/onpolicy-training-v064r2-APPROVED.json', 'sft/train_onpolicy_s2_v064r2.py',
        'tools/run_matched_context_v064c.py', 'tools/test_run_matched_context_v064c.py',
        'migration/matched-context-v064c/design.json', 'migration/matched-context-v064c/tasks.json',
        'migration/matched-context-v064c/byte-check-expected.json', 'analysis/matched-context-boss-v064c/runner.diff']
    pins = {str(ROOT / name): sha(ROOT / name) for name in names}
    pins.update({str(path): sha(path) for path in FROZEN_CODE.values()})
    pins.update({leaf['path']: sha(leaf['path']) for leaf in PINNED_FILES.values()})
    pins[str(PREREG)] = sha(PREREG)
    pins.update(qualified['artifacts'])
    require(pins[str(TEMPLATE)] == TEMPLATE_SHA and pins[str(PREREG)] == PREREG_SHA and
            pins[str(PARENT_RUNTIME)] == PARENT_RUNTIME_SHA and pins[str(PARENT_ANALYSIS)] == PARENT_ANALYSIS_SHA and
            pins[str(WRAPPER)] == WRAPPER_SHA and pins[str(TRAINING_CONTRACT)] == qualified['training_contract_sha256'] and
            all(pins[leaf['path']] == leaf['sha256'] for leaf in PINNED_FILES.values()) and
            pins[str(PARENT_TRAINING_CONTRACT)] == PARENT_TRAINING_CONTRACT_SHA and
            pins[str(COMPOSITION)] == COMPOSITION_SHA and pins[str(SERVE_ORIGINAL)] == SERVE_ORIGINAL_SHA and
            pins[str(SERVE_PARENT)] == SERVE_PARENT_SHA and pins[str(RUNNER_PARENT)] == RUNNER_PARENT_SHA and
            pins[str(COMPARATOR_RUNNER)] == COMPARATOR_RUNNER_SHA and pins[str(COMPARATOR_RUNTIME)] == COMPARATOR_RUNTIME_SHA and
            pins[str(COMPARATOR_ANALYSIS)] == COMPARATOR_ANALYSIS_SHA and pins[str(COMPARATOR_PER_TASK)] == COMPARATOR_PER_TASK_SHA and
            all(pins[name] == digest for name, digest in qualified['artifacts'].items()), 'A pre-registered v064c input hash differs')
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
    runs = generation_runs()
    aliases = [run['adapter_alias'] for run in runs]
    config, populations = frozen['evaluation_config'], frozen['populations']
    comparator_runs = [run['name'].replace('v064_A_onpol_c2_', 'v055_M0qmc_', 1) for run in runs]
    differ = ('name', 'scientific_name', 'arm', 'adapter_alias', 'adapter_source', 'submission_dir', 'result_dir', 'trace_dir')
    require([{k: v for k, v in run.items() if k not in differ} for run in comparator['generation_runs']] ==
            [{k: v for k, v in run.items() if k not in differ} for run in runs] and
            [run['name'] for run in comparator['generation_runs']] == comparator_runs and
            comparator['memory_correction'] == MEMORY_CORRECTION, 'v064c runs differ from the v055 runs beyond names, aliases, adapter and paths')
    doc = dict(status='DRAFT_PREPARED_ONLY', root=str(ROOT), total_seconds=TOTAL, cleanup_seconds=RESERVE,
        generation='v064c', generation_runs=runs, memory_correction=MEMORY_CORRECTION,
        guard_source_sha256=GUARD_SOURCE_SHA, guard_wrapper_sha256=GUARD_WRAPPER_SHA,
        required_controls_receipt=str(CONTROLS_RECEIPT), required_replay_receipt=str(REPLAY_RECEIPT),
        readout_sha256=READOUT_SHA,
        pins=pins, directories={str(ROOT / 'models' / BASE): inventory(ROOT / 'models' / BASE)},
        server_environment=comparator['server_environment'], canary_task=CANARY, aliases=aliases,
        server_manifest=str(MANIFEST), run_order=[run['name'] for run in runs],
        policy='Launch byte check, then three routing/tool canaries (one per A_onpol alias), then three fixed48 runs; no efficacy stopping, analysis, submission or training',
        server_handoff='Verified TERM-only shutdown; qualification uses immutable recorded lifecycle, not server liveness',
        gpu_operations_during_preparation=False,
        preregistration=dict(path=str(PREREG), sha256=PREREG_SHA),
        chat_template=dict(path=str(TEMPLATE), sha256=TEMPLATE_SHA),
        server_script=str(SERVE), server_script_sha256=pins[str(SERVE)],
        serve_diff=dict(path=str(SERVE_DIFF), sha256=pins[str(SERVE_DIFF)], base=str(SERVE_PARENT), base_sha256=SERVE_PARENT_SHA,
                        original=str(SERVE_ORIGINAL), original_sha256=SERVE_ORIGINAL_SHA),
        runner=dict(path=str(ROOT / 'tools/run_matched_context_v064c.py'), sha256=pins[str(ROOT / 'tools/run_matched_context_v064c.py')],
                    diff=str(RUNNER_DIFF), diff_sha256=pins[str(RUNNER_DIFF)], base=str(RUNNER_PARENT), base_sha256=RUNNER_PARENT_SHA),
        adapter=dict(path=str(ADAPTER), sha256=qualified['artifacts'][str(ADAPTER / 'adapter_model.safetensors')],
                     meta_sha256=qualified['artifacts'][str(ADAPTER / 'meta.json')]), arm_training=qualified,
        composition_check=dict(path=str(COMPOSITION), sha256=COMPOSITION_SHA),
        expected_vllm_argv=expected_vllm_argv(comparator_vllm_argv(), aliases),
        byte_check=dict(expected=str(BYTE_EXPECTED), expected_sha256=pins[str(BYTE_EXPECTED)], trace=str(BYTE_TRACE),
                        calls=list(BYTE_CALLS), alias=BYTE_ALIAS, endpoint=TOKENIZE_URL, content_format='openai',
                        stop_rule='Any mismatch or error: byte-check.json, stopped.json, TERM cleanup, no retry'),
        evaluation=dict(effective_budget=config['effective_budget'], sampling=config['sampling'], compaction=config['compaction'],
                        seeds=[1, 2, 3], all48=populations['all48_sensitivity']['task_ids'],
                        primary40=populations['primary40']['task_ids'], clean36=populations['control_clean36_sensitivity']['task_ids']),
        comparator=dict(runtime=str(COMPARATOR_RUNTIME), runtime_sha256=COMPARATOR_RUNTIME_SHA, runner=str(COMPARATOR_RUNNER),
                        runner_sha256=COMPARATOR_RUNNER_SHA, runs=comparator_runs, pairing='task x decoding seed',
                        result_inventories={str(ROOT / 'results' / name): inventory(ROOT / 'results' / name) for name in comparator_runs},
                        analysis_report=str(COMPARATOR_ANALYSIS), analysis_report_sha256=COMPARATOR_ANALYSIS_SHA,
                        per_task=str(COMPARATOR_PER_TASK), per_task_sha256=COMPARATOR_PER_TASK_SHA),
        slot_binding=SLOT_BINDING,
        readings=dict(source=str(PREREG) + ': Arms (Evaluation line) and Readouts',
                      population='primary40, three-seed means (v042 prediction_inputs); all48 and clean36 sensitivities',
                      sign='A_onpol_c2 - A0mc (slot M1 - slot M0; slot M0 = the sealed v066 A0mc rows, bound by the analyzer)',
                      verbatim=prereg_readings()),
        readme=dict(deviations_from_v064a_runner=DEVIATIONS,
                    flag_source='vLLM 0.19.1 --help line 4 [--chat-template CHAT_TEMPLATE]; read at vllm/entrypoints/openai/api_server.py:346 and passed to chat (369) and tokenize (380-388) serving',
                    tokenize_source='vllm/entrypoints/serve/tokenize/protocol.py:50-145 (TokenizeChatRequest), serving.py:55-122 (same preprocess_chat as chat completions)'))
    same = ('total_seconds', 'cleanup_seconds', 'memory_correction', 'guard_source_sha256', 'guard_wrapper_sha256', 'required_controls_receipt',
            'required_replay_receipt', 'readout_sha256', 'server_environment', 'canary_task', 'chat_template', 'server_script',
            'server_script_sha256', 'composition_check', 'evaluation')
    require(all(doc[key] == comparator[key] for key in same), 'The v064c contract differs from the v055 contract beyond the declared changes')
    swap = lambda argv: [x.replace(str(MANIFEST), str(COMPARATOR_MANIFEST)).replace('v064_A_onpol_c2_s', 'v053_M0q_s') for x in argv]
    require(swap(doc['expected_vllm_argv']) == comparator['expected_vllm_argv'],
            'Expected vLLM argv differs from the v055 argv beyond manifest path and aliases')
    return doc


def contract_text(doc):
    return json.dumps(doc, indent=2, sort_keys=True) + chr(10)


def tasks_text(frozen):
    """tasks.json: the frozen populations of the readout, required equal to the lists of the v055 runtime contract."""
    populations, v055 = frozen['populations'], read(COMPARATOR_RUNTIME)['evaluation']
    lists = dict(primary40=populations['primary40']['task_ids'], all48=populations['all48_sensitivity']['task_ids'],
                 clean36=populations['control_clean36_sensitivity']['task_ids'])
    require(all(lists[key] == v055[key] for key in lists), 'Task lists differ from the v055 runtime contract')
    return json.dumps(dict(source=str(READOUT), source_sha256=READOUT_SHA, canary=CANARY, seeds=list(SEEDS), **lists,
                           equal_to_v055=dict(runtime=str(COMPARATOR_RUNTIME), sha256=COMPARATOR_RUNTIME_SHA, lists=sorted(lists))),
                      indent=2) + chr(10)


def prepare():
    """Writes the staged draft contract and its by-products once; a later call must re-derive identical bytes."""
    frozen = readout()
    require(sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == GUARD_SOURCE_SHA and
            sha(GUARD) == GUARD_WRAPPER_SHA, 'Guard sources are not the independently approved versions')
    require(serve_copy_is_one_flag() and sha(SERVE_PARENT) == SERVE_PARENT_SHA and SERVE.read_bytes() == SERVE_PARENT.read_bytes(),
            'Server script is not the v051 server script (serve_vllm.py plus the one template flag)')
    require(BYTE_ALIAS == generation_runs()[0]['adapter_alias'], 'Byte-check alias is not a v064c alias')
    PACKAGE.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    write_or_verify(SERVE_DIFF, unified_diff(SERVE_PARENT, SERVE))  # the existing, empty v055 serve.diff: nothing new is written
    write_or_verify(RUNNER_DIFF, unified_diff(RUNNER_PARENT, Path(__file__).resolve()))
    write_or_verify(BYTE_EXPECTED, byte_check_expected())
    write_or_verify(TASKS, tasks_text(frozen))
    with (ROOT / 'results/controls_train/controls.csv').open() as stream:
        controls = [row for row in csv.DictReader(stream) if row['instance_id'] == CANARY]
    require(len(controls) == 1 and controls[0]['gold_resolved'] == 'True' and
            controls[0]['empty_resolved'] == 'False' and controls[0]['empty_exit_code'] == '1', 'Fixed canary controls changed')
    write_or_verify(RUNTIME_DRAFT, contract_text(contract_doc(frozen, pending_qualified())))
    digest = sha(RUNTIME_DRAFT)
    write_or_verify(Path(str(RUNTIME_DRAFT) + '.sha256'), digest + '  ' + RUNTIME_DRAFT.name + chr(10))
    return digest


def verify_draft(expected):
    """Inspection of the staged draft, never a run input: its sidecar, and the contract re-derived now from the current files with the
    pending receipt equals the draft byte for byte. The adapter-dependent fields stay pending until the pin step."""
    require(Path(str(RUNTIME_DRAFT) + '.sha256').read_text().split()[0] == expected, 'Draft sidecar mismatch')
    require(RUNTIME_DRAFT.read_text() == contract_text(contract_doc(readout(), pending_qualified())),
            'The staged draft does not re-derive from the current files')
    return dict(runtime_draft_sha256=expected, pending_fields=len(pending_paths(read(RUNTIME_DRAFT))), arm_adapter=PENDING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--controls-receipt-sha')
    parser.add_argument('--replay-receipt-sha')
    args = parser.parse_args()
    require(sum((args.prepare, args.preflight, args.activate)) <= 1, 'Prepare, preflight and activate are separate modes')
    if args.prepare:
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(), gpu_operations=False)))
    elif args.preflight:
        require(args.runtime_sha, 'Preflight requires the runtime SHA')
        report = preflight(args.runtime_sha)
        print(json.dumps(report, indent=1))
        sys.exit(0 if report['ready'] else 3)
    elif args.activate:
        require(args.runtime_sha, 'Activation requires runtime SHA')
        activate(args.runtime_sha, args.controls_receipt_sha, args.replay_receipt_sha)
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        if RUNTIME_DRAFT.is_file() and sha(RUNTIME_DRAFT) == args.runtime_sha:
            print(json.dumps(dict(status='draft_verified', gpu_operations=False, **verify_draft(args.runtime_sha))))
            return
        doc, _ = verify_runtime(args.runtime_sha)
        require(qualified_arm() == doc['arm_training'], 'A_onpol training receipt changed since preparation')
        admitted = (admission_receipts(doc, args.controls_receipt_sha, args.replay_receipt_sha)['byte_compatibility']
                    if args.controls_receipt_sha or args.replay_receipt_sha else None)
        print(json.dumps(dict(status='prepared_only', gpu_operations=False, runtime_verified=True,
                              arm_adapter_sha256=doc['adapter']['sha256'], admission_receipts_verified=admitted)))


if __name__ == '__main__':
    main()
