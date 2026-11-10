"""Explicit v0351 retry generation; preserves the stopped v035 runtime and scientific inputs."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time

from prepare_v032_probes import load_recovery, require, sha, write_json

ROOT = Path('PROJECT_ROOT/gemma4')
DRAFT = ROOT / 'contracts/mask-control-v035-DRAFT.json'
APPROVED = ROOT / 'contracts/mask-control-v0351-APPROVED.json'
RUNTIME = ROOT / 'contracts/mask-control-v0351-runtime-DRAFT.json'
READOUT = ROOT / 'contracts/mask-control-v035-readout.json'
READOUT_SHA = '6319262b25c19ddcbb0e34a33f5e97325a7d6117a7b8faad8d4a4c7ff772c909'
EVIDENCE = ROOT / 'migration/mask-control-v0351/training'
PREVIOUS_FAILURE = ROOT / 'migration/mask-control-v035/training/stopped.json'
PREVIOUS_FAILURE_SHA = 'b9fdb007b43275422dab308d137ea5d49941cdfb582523178dc80e36949d9aab'
TRAIN_PY = Path('HOME_DIR/miniconda3/envs/train/bin/python')
E2_PID, E2_START = 2999341, '448738447'
ANALYZER_PID, ANALYZER_START = 3005824, '448806630'
E2_RUNTIME_SHA = 'e52bccac12d4b8dd4f680d38f24a90fdce49285b19b67f4720fe54ad4c8a58bc'
HOURS = 3600
ARM_SECONDS, TOTAL_SECONDS, CLEANUP = 6 * HOURS, 13 * HOURS, 600
HP = dict(rank=16, alpha=16, lr=1e-4, epochs=1, max_seq_len=16384, grad_accum=8,
          warmup_ratio=0.05, weight_decay=0.0, max_grad_norm=1.0, seed=20261103)


def module(path):
    spec = importlib.util.spec_from_file_location('v035_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def checked_json(path, expected):
    require(sha(path) == expected, 'Input hash mismatch: ' + str(path))
    result = json.loads(Path(path).read_text())
    require(sha(path) == expected, 'Input changed while reading')
    return result


def inputs(draft):
    require(draft['hyperparameters'] == HP, 'Original hyperparameters changed')
    require(draft['base'] == str(ROOT / 'models/gemma-4-e4b-it'), 'Wrong BF16 base')
    require(draft['fresh_outputs'] == {v: str(ROOT / ('sft/runs/A1_' + v + '_v035')) for v in ('M0', 'M1')},
            'Training output paths changed')
    for mapping in (draft['source_hashes'], draft['artifact_hashes']):
        for name, digest in mapping.items():
            require(sha(name) == digest, 'Frozen training input changed: ' + name)
    manifest = checked_json(draft['manifest_path'], draft['manifest_sha256'])
    for key in ('source_hashes', 'artifact_hashes', 'hyperparameters', 'fresh_outputs', 'base', 'probe_dir'):
        require(draft[key] == manifest[key], 'Draft/manifest mismatch: ' + key)
    original = json.loads((ROOT / 'sft/runs/A1.meta.json').read_text())['args']
    require(all(original[key] == value for key, value in HP.items()), 'Original A1 configuration mismatch')
    return manifest


def approval(path, expected, draft):
    require(Path(path).resolve() == APPROVED and APPROVED != DRAFT, 'Require canonical approved copy')
    contract = checked_json(path, expected)
    require(contract.get('status') == 'APPROVED' and contract.get('execute_training') is True,
            'Unapproved contract: no GPU operation allowed')
    for key in ('source_hashes', 'artifact_hashes', 'hyperparameters', 'fresh_outputs', 'base', 'probe_dir',
                'manifest_path', 'manifest_sha256', 'arm'):
        require(contract[key] == draft[key], 'Approved scientific/input fields changed: ' + key)
    require(not contract.get('unresolved_activation_gates'), 'Approved contract still has unresolved gates')
    require(contract['readout_path'] == str(READOUT) and contract['readout_sha256'] == READOUT_SHA,
            'Approved readout binding changed')
    require(contract['runtime_sha256'] == sha(RUNTIME) and
            contract['previous_failed_receipt_sha256'] == PREVIOUS_FAILURE_SHA and
            sha(PREVIOUS_FAILURE) == PREVIOUS_FAILURE_SHA, 'Retry runtime/previous failure binding changed')
    return contract


def verify_runtime(expected):
    doc = checked_json(RUNTIME, expected)
    require(RUNTIME.with_suffix('.json.sha256').read_text().split()[0] == expected, 'Runtime sidecar mismatch')
    require(doc['root'] == str(ROOT) and doc['total_seconds'] == TOTAL_SECONDS and
            doc['per_arm_seconds'] == ARM_SECONDS and doc['cleanup_seconds'] == CLEANUP, 'Runtime identity/bounds changed')
    for name, digest in doc['pins'].items():
        require(sha(name) == digest, 'Runtime source changed: ' + name)
    draft = checked_json(DRAFT, doc['draft_sha256'])
    inputs(draft)
    return doc, draft


def wait_exit(recovery, pid, start, queue, timeout):
    if queue:
        recovery.wait_parent(pid, start, timeout=timeout)
        return
    try:
        value = recovery.stat(pid)
        require(value['start'] == start and value['state'] == 'Z', 'Admission controller active or PID reused')
    except FileNotFoundError:
        pass


def handoff_identity_matches(recorded, live, recovery):
    """Allow only the verified exited E2 controller's top-server reparenting to init."""
    wait_exit(recovery, E2_PID, E2_START, False, 1)
    expected = json.loads(json.dumps(recorded))
    actual = json.loads(json.dumps(live))
    key = str(expected['pid'])
    if (key in expected['members'] and key in actual['members'] and
            expected['members'][key]['parent'] == E2_PID and actual['members'][key]['parent'] == 1):
        actual['members'][key]['parent'] = E2_PID
    return expected == actual


def qualified_e2(analyzer):
    analyzer.no_failed_completion(ROOT)
    require((ROOT / analyzer.EVIDENCE / 'completed.json').is_file(), 'E2 qualified completion unavailable')
    ledger = analyzer.Inputs(ROOT)
    plan, runtime, modules = analyzer.manifests(ROOT, ledger)
    done, runs = analyzer.qualify_runs(ROOT, plan, ledger, modules)
    ledger.verify()
    return done, runs


def qualified_analysis(analyzer, doc):
    path = ROOT / 'analysis/recovery-screen-v034/report.json'
    sidecar = Path(str(path) + '.sha256')
    expected = sidecar.read_text().split()[0]
    report = checked_json(path, expected)
    require(report['plan_sha256'] == analyzer.PLAN_SHA and report['runtime_sha256'] == E2_RUNTIME_SHA and
            report['source_sha256'] == doc['pins'][str(ROOT / 'tools/analyze_31b_recovery_screen_v034.py')] and
            report['tests_sha256'] == doc['pins'][str(ROOT / 'tools/test_analyze_31b_recovery_screen_v034.py')],
            'Analyzer report provenance mismatch')
    require(report['qualification'] == dict(full8=True, canary=True, route_drained=True, source_pins=True,
            trace_hashes_and_call_counts=True, inputs_stable_before_after=True), 'Analyzer qualification incomplete')
    require(report['automatic_expansion'] is False and report['submission'] is False and
            report['treatment_efficacy_measured'] is False, 'Unexpected analysis scope')
    for name, digest in report['inputs'].items():
        require(sha(name) == digest, 'Analyzer input changed: ' + name)
    # Deliberately no condition on opportunity_gate, task successes, or any scientific outcome.
    return dict(report_sha256=expected, sidecar_sha256=sha(sidecar))


def assert_idle(recovery):
    recovery.assert_idle((ROOT, Path('HOME_DIR/gemma4')))
    require(not any('train_mask_control_v035.py' in {Path(v).name for v in row['argv']}
                    for row in recovery.processes().values()), 'Another matched-control trainer is active')


def outputs_absent(contract):
    for value in contract['fresh_outputs'].values():
        for suffix in ('', '.meta.json', '.train_log.jsonl'):
            require(not Path(value + suffix).exists(), 'Refuse output overwrite: ' + value + suffix)
    for variant in ('M0', 'M1'):
        require(not (ROOT / ('logs/v0351_train_' + variant + '.log')).exists(), 'Training log collision')


def safe_header(path):
    with Path(path).open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        require(0 < count < Path(path).stat().st_size - 8, 'Invalid safetensors header size')
        header = json.loads(stream.read(count))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    require(tensors, 'Empty adapter')
    for entry in tensors.values():
        left, right = entry['data_offsets']
        require(type(left) is int and type(right) is int and 0 <= left < right <= Path(path).stat().st_size - count - 8,
                'Invalid adapter tensor data bounds')
    return tensors


def validate_training(contract, variant, contract_sha, started_ns):
    output = Path(contract['fresh_outputs'][variant])
    paths = [output / 'adapter_model.safetensors', output / 'adapter_config.json',
             Path(str(output) + '.meta.json'), Path(str(output) + '.train_log.jsonl')]
    require(all(p.is_file() and p.stat().st_mtime_ns >= started_ns for p in paths), 'Missing/stale training artifacts')
    meta = json.loads(paths[2].read_text())
    require(meta['contract_sha256'] == contract_sha and meta['names_match_probe'] is True,
            'Trainer completion contract/name mismatch')
    require(re.fullmatch('[0-9a-f]{64}', meta['initial_trainable_parameters_sha256']) is not None,
            'Missing common initialization hash')
    require(all(meta['args'][key] == value for key, value in HP.items()) and
            meta['args']['variant'] == variant and meta['args']['out'] == str(output) and
            meta['args']['base'] == contract['base'] and meta['args']['device'] == 'cuda', 'Trainer arguments mismatch')
    require(all(type(meta[k]) in (float, int) and math.isfinite(meta[k]) and meta[k] > 0
                for k in ('seconds', 'tokens')), 'Nonfinite/empty trainer summary')
    config = json.loads(paths[1].read_text())
    expected = json.loads((Path(contract['probe_dir']) / 'adapter_config.json').read_text())
    expected.update(r=16, lora_alpha=16, base_model_name_or_path='google/gemma-4-e4b-it-qat-w4a16-ct')
    require(config == expected, 'Rank/alpha/target module/config drift')
    actual = safe_header(paths[0])
    probe = safe_header(Path(contract['probe_dir']) / 'adapter_model.safetensors')
    require(set(actual) == set(probe) and meta['adapter_tensors'] == len(actual), 'Missing/unexpected adapter layers')
    for name, value in actual.items():
        shape = list(probe[name]['shape'])
        dimension = 0 if name.endswith('lora_A.weight') else 1
        require(name.endswith(('lora_A.weight', 'lora_B.weight')), 'Unexpected trainable tensor')
        shape[dimension] = 16
        require(value['shape'] == shape and value['dtype'] == 'BF16', 'Adapter tensor shape/dtype drift')
        require(value['data_offsets'][1] - value['data_offsets'][0] == math.prod(shape) * 2,
                'Adapter tensor byte count mismatch')
    logs = [json.loads(line) for line in paths[3].read_text().splitlines() if line.strip()]
    groups = json.loads((Path(contract['manifest_path']).parent / 'update_groups.json').read_text())
    require(len(logs) == len(groups) == 208, 'Training updates incomplete')
    for index, (row, group) in enumerate(zip(logs, groups), 1):
        require(row['step'] == index and row['loss_tokens'] == group['original_denominator'] and
                row['sequences'] == len(group['sequence_indices']) and
                row['remaining_loss_tokens'] == (group['original_denominator'] if variant == 'M0' else group['m1_remaining_targets']),
                'Frozen update exposure mismatch')
        for key in ('loss', 'lr', 'grad_norm', 'seconds', 'tokens_per_s', 'peak_alloc_gib'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]), 'Nonfinite training metric: ' + key)
    require(meta['tokens'] == sum(row['tokens'] for row in logs), 'Training token summary mismatch')
    return dict(variant=variant, initial_sha256=meta['initial_trainable_parameters_sha256'],
                artifacts={str(p): sha(p) for p in paths}, updates=len(logs), adapter_tensors=len(actual))


def activate(runtime_sha, approved_sha, queue, timeout):
    doc, draft = verify_runtime(runtime_sha)
    contract = approval(APPROVED, approved_sha, draft)
    recovery = load_recovery(ROOT)
    e2 = module(ROOT / 'tools/run_31b_recovery_screen_v034.py')
    analyzer = module(ROOT / 'tools/analyze_31b_recovery_screen_v034.py')
    wait_exit(recovery, E2_PID, E2_START, queue, timeout)
    done, runs = qualified_e2(analyzer)
    wait_exit(recovery, ANALYZER_PID, ANALYZER_START, queue, timeout)
    analysis = qualified_analysis(analyzer, doc)
    require(not EVIDENCE.exists(), 'Refuse training receipt overwrite')
    outputs_absent(contract)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'migration/recovery-screen-v034/evaluation.lock').open('a') as previous, \
         (EVIDENCE.parent / 'training.lock').open('a') as lease:
        fcntl.flock(previous, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        deadline = started + TOTAL_SECONDS
        EVIDENCE.mkdir(exist_ok=False)
        try:
            verify_runtime(runtime_sha)
            approval(APPROVED, approved_sha, draft)
            qualified_e2(analyzer)
            qualified_analysis(analyzer, doc)
            outputs_absent(contract)
            assert_idle(recovery)
            server = done['server_handoff']
            live_server = e2.server31_identity(recovery, server['pid'], server['start'])
            require(handoff_identity_matches(server, live_server, recovery), '31B handoff identity changed')
            server = live_server
            write_json(EVIDENCE / 'admission.json', dict(runtime_sha256=runtime_sha, approval_sha256=approved_sha,
                e2_completed_sha256=sha(ROOT / analyzer.EVIDENCE / 'completed.json'), analysis=analysis,
                server=server, total_seconds=TOTAL_SECONDS, per_arm_seconds=ARM_SECONDS, cleanup_seconds=CLEANUP))
            cleanup = e2.cleanup31(recovery, server, min(deadline, time.monotonic() + 60))
            require(not e2.reviewed.group_members(recovery, server['group']), '31B group remains active')
            gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
            require(not gpu.strip(), 'GPU compute processes remain; training refused')
            write_json(EVIDENCE / 'gpu-handoff.json', dict(server_cleanup=cleanup, compute_processes=[]))
            helper = recovery.load_helper(ROOT)
            env = e2.reviewed.environment(doc)
            env.update(PATH=str(TRAIN_PY.parent) + ':' + env.get('PATH', ''), TOKENIZERS_PARALLELISM='false')
            receipts = {}
            for variant in ('M0', 'M1'):
                verify_runtime(runtime_sha)
                approval(APPROVED, approved_sha, draft)
                assert_idle(recovery)
                require(time.monotonic() < deadline - CLEANUP, 'Total training allocation exhausted')
                wall_ns = time.time_ns()
                runner = e2.BoundedService(min(deadline, time.monotonic() + ARM_SECONDS + CLEANUP), EVIDENCE)
                log = ROOT / ('logs/v0351_train_' + variant + '.log')
                runner([TRAIN_PY, ROOT / 'sft/train_mask_control_v035.py', '--variant', variant,
                        '--execute', '--contract', APPROVED], log, env, ROOT)
                service_path = EVIDENCE / (log.name + '.service-end.json')
                service = json.loads(service_path.read_text())
                require(service['returncode'] == 0 and service['error'] is None and service['cgroup_empty'] is True,
                        'Trainer service failed or cgroup remains active')
                receipt = validate_training(contract, variant, approved_sha, wall_ns)
                verify_runtime(runtime_sha)
                approval(APPROVED, approved_sha, draft)
                receipt['service_receipt_sha256'] = sha(service_path)
                receipts[variant] = receipt
                if variant == 'M1':
                    require(receipt['initial_sha256'] == receipts['M0']['initial_sha256'], 'Fresh controls have different initialization')
                write_json(EVIDENCE / (variant + '-complete.json'), receipt)
            for receipt in receipts.values():
                for name, digest in receipt['artifacts'].items():
                    require(sha(name) == digest, 'Completed adapter artifact changed')
            require(time.monotonic() <= deadline, 'Total training deadline exceeded')
            write_json(EVIDENCE / 'completed.json', dict(status='training_only_qualified', controls=receipts,
                runtime_sha256=runtime_sha, approval_sha256=approved_sha, manifest_sha256=draft['manifest_sha256'],
                elapsed_seconds=time.monotonic() - started, evaluation_or_submission_started=False,
                handoff='Both fresh adapters ready for separate reviewed evaluation; no serving process started'))
        except BaseException as exc:
            residuals = {}
            for receipt in EVIDENCE.glob('*.service-end.json'):
                value = json.loads(receipt.read_text())
                residuals[receipt.name] = dict(unit=value['unit'], identity=value['identity'],
                    cgroup_empty=value['cgroup_empty'], state=value['state'])
            write_json(EVIDENCE / 'stopped.json', dict(type=type(exc).__name__, reason=str(exc),
                elapsed_seconds=time.monotonic() - started, owned_services=residuals, no_automatic_retry=True))
            raise


def prepare():
    require(not RUNTIME.exists() and not RUNTIME.with_suffix('.json.sha256').exists(), 'Runtime draft collision')
    draft = json.loads(DRAFT.read_text())
    require(draft['status'] == 'DRAFT_UNAPPROVED_NO_TRAINING' and draft['execute_training'] is False, 'Expected unapproved training draft')
    inputs(draft)
    names = ['tools/run_mask_control_v0351.py', 'tools/test_run_mask_control_v0351.py',
        'tools/run_mask_control_v035.py', 'tools/test_run_mask_control_v035.py',
        'contracts/mask-control-v035-runtime-DRAFT.json', 'contracts/mask-control-v035-runtime-DRAFT.json.sha256',
        'tools/run_31b_recovery_screen_v034.py', 'tools/test_run_31b_recovery_screen_v034.py',
        'tools/analyze_31b_recovery_screen_v034.py', 'tools/test_analyze_31b_recovery_screen_v034.py',
        'tools/prepare_v032_probes.py', 'tools/run_v032_probes.py', 'tools/recover_eval_v025.py',
        'tools/overnight_v02.py', 'contracts/recovery-screen-v034-runtime.json',
        'contracts/recovery-screen-v034-plan.json', 'contracts/mask-control-v035-readout.json',
        'contracts/mask-control-v035-readout.json.sha256']
    require(sha(READOUT) == READOUT_SHA and Path(str(READOUT) + '.sha256').read_text().split()[0] == READOUT_SHA,
            'Readout contract/sidecar mismatch')
    e2_doc = json.loads((ROOT / 'contracts/recovery-screen-v034-runtime.json').read_text())
    pins = {str(ROOT / name): sha(ROOT / name) for name in names}
    require(sha(PREVIOUS_FAILURE) == PREVIOUS_FAILURE_SHA, 'Previous failure receipt changed')
    pins[str(PREVIOUS_FAILURE)] = PREVIOUS_FAILURE_SHA
    pins.update(e2_doc['installed_pins'])
    pins[str(TRAIN_PY)] = sha(TRAIN_PY)
    document = dict(status='DRAFT_RUNTIME_NO_LAUNCH', root=str(ROOT), draft_sha256=sha(DRAFT), pins=pins,
        total_seconds=TOTAL_SECONDS, per_arm_seconds=ARM_SECONDS, cleanup_seconds=CLEANUP,
        server_environment=e2_doc['server_environment'], controls=['M0', 'M1'],
        queue=dict(e2_pid=E2_PID, e2_start=E2_START, analyzer_pid=ANALYZER_PID, analyzer_start=ANALYZER_START),
        approval_path=str(APPROVED), gpu_operations=False,
        previous_failed_receipt_path=str(PREVIOUS_FAILURE), previous_failed_receipt_sha256=PREVIOUS_FAILURE_SHA,
        generation='v0351 explicit retry; original v035 runtime/evidence preserved',
        containment='Reuse hash-pinned v034 BoundedService directly; inherited gemma-v034-eval UUID unit prefix is intentional; arm RuntimeMaxSec<=21600 plus bounded cleanup',
        handoff='Validate E2/analyzer qualified receipts without outcome selection; exact server31B TERM then empty group/GPU compute processes; training only')
    write_json(RUNTIME, document)
    digest = sha(RUNTIME)
    with RUNTIME.with_suffix('.json.sha256').open('x') as stream:
        stream.write(digest + '  ' + RUNTIME.name + '\n')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--approved-contract-sha')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--queue-after-e2', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=86400)
    args = parser.parse_args()
    require(math.isfinite(args.wait_seconds) and args.wait_seconds > 0, 'Invalid queue wait bound')
    if args.prepare:
        require(not args.activate, 'Preparation cannot activate')
        print(json.dumps(dict(status='prepared_only', runtime_sha256=prepare(), gpu_operations=False)))
    elif args.activate:
        require(args.runtime_sha and args.approved_contract_sha, 'Require exact runtime and approved contract hashes')
        activate(args.runtime_sha, args.approved_contract_sha, args.queue_after_e2, args.wait_seconds)
    else:
        require(args.runtime_sha, 'Inspection requires runtime SHA')
        verify_runtime(args.runtime_sha)
        print(json.dumps(dict(status='prepared_only', gpu_operations=False, controls=['M0', 'M1'])))


if __name__ == '__main__':
    main()
