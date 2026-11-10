"""Independent control readout; no model evaluation outcomes are accessed."""
from pathlib import Path
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path('PROJECT_ROOT/gemma4')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from sandbox_memory_guard_v043 import qualify_guard_receipt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = OUT / 'report.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite readout')
    selection = json.loads((OUT / 'selection.json').read_text())
    frozen = ROOT / 'contracts/mask-control-v035-readout.json'
    assert sha(frozen) == '6319262b25c19ddcbb0e34a33f5e97325a7d6117a7b8faad8d4a4c7ff772c909'
    readout = json.loads(frozen.read_text())
    primary = readout['populations']['primary40']['task_ids']
    all48 = readout['populations']['all48_sensitivity']['task_ids']
    clean36 = readout['populations']['control_clean36_sensitivity']['task_ids']
    rows = list(csv.DictReader((OUT / 'controls/controls.csv').open()))
    prior_rows = list(csv.DictReader((ROOT / 'results/controls_rich/controls.csv').open()))
    by_id = {r['instance_id']: r for r in rows}
    prior = {r['instance_id']: r for r in prior_rows}
    assert len(rows) == len(by_id) == 48 and set(by_id) == set(all48)
    assert set(primary) == set(selection['primary40_ids']) and len(primary) == 40
    assert set(all48) == set(selection['ids'])
    assert set(primary) == {r['instance_id'] for r in prior_rows if r['gold_resolved'] == 'True' and r['empty_resolved'] == 'False'}
    start = json.loads((OUT / 'service-start.json').read_text())
    service = json.loads((OUT / 'service-end.json').read_text())
    assert start['source_sha256']['tools/sandbox_memory_guard_v043.py'] == selection['expected_guard_sha256']
    assert sha(ROOT / 'tools/sandbox_memory_guard_v043.py') == selection['expected_guard_sha256']
    guard = qualify_guard_receipt(OUT / 'guard.json')
    observed = service['observed']
    assert observed and len({r['InvocationID'] for r in observed}) == 1
    for live in observed:
        assert int(live['MemoryMax']) == 24 * 1024**3 and int(live['MemorySwapMax']) == 0
        assert live['OOMPolicy'] == 'continue' and live['DelegateSubgroup'] == 'supervisor' and live['Delegate'] == 'yes'
    primary_pass = [i for i in primary if by_id[i]['gold_resolved'] == 'True' and by_id[i]['gold_exit_code'] == '0']
    clean_empty = [i for i in clean36 if by_id[i]['empty_resolved'] == 'False' and by_id[i]['empty_exit_code'] == '1']
    original8 = sorted(set(all48) - set(primary))
    host_oom = json.loads((OUT / 'host-oom-check.json').read_text())
    failures = []
    if not host_oom['no_global_oom_matches']:
        failures.append('Kernel global OOM check failed or found an event')
    if len(primary_pass) != 40:
        failures.append('Not all primary40 gold controls pass')
    if len(clean_empty) != 36:
        failures.append('Control-clean36 empty behavior changed')
    if any(r['empty_resolved'] != 'False' for r in rows):
        failures.append('Empty patch unexpectedly resolves a holdout task')
    if service['returncode'] != 0 or not service['systemd_run_success'] or not service['cgroup_empty']:
        failures.append('Control service did not succeed and drain')
    if int(guard['service_memory_max']) != 24 * 1024**3 or int(guard['service_memory_swap_max']) != 0:
        failures.append('Control service resource properties mismatch')
    def detail(i):
        r = by_id[i]
        return {'instance_id': i, 'prior_gold_resolved': prior[i]['gold_resolved'], 'prior_gold_exit_code': int(prior[i]['gold_exit_code']), 'gold_resolved': r['gold_resolved'], 'gold_exit_code': int(r['gold_exit_code']), 'prior_empty_exit_code': int(prior[i]['empty_exit_code']), 'empty_resolved': r['empty_resolved'], 'empty_exit_code': int(r['empty_exit_code'])}
    paths = [frozen, ROOT / 'controls/run_controls.py', ROOT / 'results/controls_rich/controls.csv', ROOT / 'tools/sandbox_memory_guard_v043.py', OUT / 'selection.json', OUT / 'run_controls_guarded.py', OUT / 'launch_controls.py', Path(__file__), OUT / 'service-start.json', OUT / 'service-end.json', OUT / 'service-observed.json', OUT / 'host-oom-check.json', OUT / 'guard.json', OUT / 'controls/controls.csv', OUT / 'controls/summary.json']
    paths += sorted((OUT / 'controls/test_outputs').glob('*.log'))
    paths += sorted(OUT.glob('*commands*.jsonl'))
    report = {'schema': 'oom-controls-v043', 'created_utc': datetime.now(timezone.utc).isoformat(), 'qualified': not failures, 'failures': failures, 'frozen_readout_sha256': sha(frozen), 'population_unchanged': True, 'primary40': {'ids': primary, 'expected_count': 40, 'gold_resolved_count': sum(by_id[i]['gold_resolved'] == 'True' for i in primary), 'gold_exit_zero_count': sum(by_id[i]['gold_exit_code'] == '0' for i in primary), 'all_gold_pass': len(primary_pass) == 40, 'nonpassing_ids': sorted(set(primary) - set(primary_pass))}, 'original8': {'ids': original8, 'rows': [detail(i) for i in original8]}, 'all48': {'ids': all48, 'row_count': len(rows), 'gold_resolved_count': sum(r['gold_resolved'] == 'True' for r in rows), 'empty_resolved_count': sum(r['empty_resolved'] == 'True' for r in rows)}, 'control_clean36': {'ids': clean36, 'expected_empty_exit_one_count': 36, 'actual_empty_exit_one_count': len(clean_empty)}, 'pathological_controls': [detail(i) for i in ['rich_3480', 'rich_3772', 'rich_3930', 'rich_4006']], 'memory': {'sandbox_memory_max': 4 * 1024**3, 'sandbox_swap_max': 0, 'service_memory_max': int(guard['service_memory_max']), 'service_memory_swap_max': int(guard['service_memory_swap_max']), 'service_result': 'success' if service['systemd_run_success'] else service['service']['Result'], 'service_invocation_id': observed[0]['InvocationID'], 'service_properties': observed[0], 'service_properties_last': observed[-1], 'service_returncode': service['returncode'], 'cgroup_empty': service['cgroup_empty'], 'guard_qualified': True, 'no_global_oom_matches': host_oom['no_global_oom_matches'], 'peak': int(guard['service_memory_peak']), 'oom_commands': guard['oom_command_count'], 'command_count': guard['command_count']}, 'service_observation_note': 'Active unit properties and InvocationID were observed during execution and retained separately from raw unloaded post-show. Exact final peak and memory limits corroborated by guard.close before unit exit; systemd-run --wait returned0 and log records success.', 'input_sha256': {str(path.relative_to(ROOT)): sha(path) for path in paths}}
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['qualified', 'failures', 'primary40', 'all48', 'control_clean36', 'pathological_controls', 'memory']}, indent=2))
    print('report_sha256', sha(target))
    raise SystemExit(0 if report['qualified'] else 1)


if __name__ == '__main__':
    main()
