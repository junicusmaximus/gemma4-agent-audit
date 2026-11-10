#!/usr/bin/env python3
"""Probe A v052 readout: A0cap (v052) against fresh M0 (v043) through the immutable v042 statistics."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path('PROJECT_ROOT/gemma4')
LEGACY = ROOT / 'tools/analyze_mask_control_v042.py'
LEGACY_SHA = '23915fb00c2836b00fe14e3adfeecb865bdf52eb34e1fe3a04c4093cffdaf376'
if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
    raise ValueError('Immutable v042 statistical implementation changed')
_spec = importlib.util.spec_from_file_location('v052_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
for _name in ('require', 'sha', 'read_json', 'Inputs', 'module', 'validate_populations', 'metrics_for_runs',
              'population_report', 'prediction_inputs', 'mean_three', 'write_fresh', 'SEEDS', 'READOUT', 'READOUT_SHA'):
    globals()[_name] = getattr(legacy, _name)

PRODUCER = ROOT / 'tools/mechanism_metrics_v032.py'
PRODUCER_SHA = '36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d'
VALIDATOR = ROOT / 'tools/analyze_mechanism_v031.py'
VALIDATOR_SHA = '821f2294661bf8928979d3df7da6c94fd8f662ee2ed17621a8c8dfaa06207478'
RESOLUTION = ROOT / 'tools/analyze_contract_v02_v025.py'
RESOLUTION_SHA = '20208c74cc921d9dd4975b1b6a7129b6a4b55825f17930429a4fa624935e229c'
SOURCES = {'v052': (ROOT / 'tools/run_a0_capped_v052.py', ROOT / 'contracts/a0-capped-v052-runtime-DRAFT.json'),
           'v043': (ROOT / 'tools/run_mask_evaluations_v043.py', ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json')}
V043_SOURCE_SHA = '9b39a600adb998117a694df5dfd61514f03e29b467795b58bc449b495f8ca590'
V043_RUNTIME_SHA = '7eaf5d53b56e02d6e45ef29a229748b2716bdee36ae7b4ebf0301450d4c57a63'
EVIDENCE = {'v052': ROOT / 'migration/a0-capped-v052/evaluation', 'v043': ROOT / 'migration/mask-evaluations-v043/evaluation'}
SEALED_V043 = {'per_task': ROOT / 'analysis/mask-control-v043/per_task.jsonl', 'report': ROOT / 'analysis/mask-control-v043/report.json'}
PRIMARY_IDS = ROOT / 'analysis/mechanism-v032/primary40_ids.txt'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-05-capped-A0-and-M1-nothink-probes.md'
PREREG_SHA = 'cc043e0394bbb586a2b242a3a8ef1b6a16bc655388aea56c7c5249c14a1464d3'
TESTS = ROOT / 'tools/test_analyze_a0_capped_v052.py'
MANIFEST = ROOT / 'contracts/a0-capped-v052-analysis-DRAFT.json'
OUTPUT = ROOT / 'analysis/a0-capped-v052'
# Frozen before any A0cap outcome exists: v042 slot -> (generation, source arm, output label).
SLOTS = {'M0': ('v043', 'M0', 'M0'), 'M1': ('v052', 'A0cap', 'A0cap')}
DIRECTION = 'A0cap minus M0 = v042 slot M1 minus slot M0 (sign +1); slot keys renamed only in the written output'
PROBE_A = dict(population='primary40', slot='M1', inert_min_pooled=4, inert_max_loop=.10,
               suppressive_max_pooled=1, denominator=120)
READINGS = {'A-i': 'environment inert: the cap does not drive the zero outcome of M0/M1; A0cap vs M0 is a within-environment, task-paired descriptive contrast (declared LoRA-off vs LoRA-on server-flag difference)',
            'A-ii': 'environment suppressive: the capped environment itself suppresses resolution; the M0/M1 resolution endpoint is uninformative and the paper rests on behavioural rates',
            'intermediate': 'reported descriptively; no reading',
            'UNDEFINED': 'a required aggregate is unavailable; no reading'}


def slot_runs(runs, arm, slot, generation):
    selected = [copy.deepcopy(run) for run in runs if run['arm'] == arm]
    require([run['decoding_seed'] for run in selected] == list(SEEDS), 'Expected decoding seeds 1-3: ' + generation + ' ' + arm)
    for run in selected:
        run.update(arm=slot, source_arm=arm, source_generation=generation)
    return selected


def slot_readout(readout, baseline_runs, treated_runs):
    effective = copy.deepcopy(readout)
    effective['future_runs'] = baseline_runs + treated_runs
    effective['future_analysis_dir'] = str(OUTPUT)
    return effective


def rename_slots(value):
    """Output-only relabelling: exact slot keys and per-run summary labels; metric names stay unchanged."""
    labels = {slot: spec[2] for slot, spec in SLOTS.items()}
    if isinstance(value, dict):
        return {labels.get(key, key): rename_slots(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_slots(item) for item in value]
    if isinstance(value, str) and re.fullmatch('M[01]_s[123]', value):
        return labels[value[:2]] + value[2:]
    return value


def probe_a_label(values, rule=PROBE_A):
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    counts = values.get('solved_by_seed')
    valid = isinstance(counts, (list, tuple)) and len(counts) == 3 and all(type(v) is int and 0 <= v <= 40 for v in counts)
    pooled = sum(counts) if valid else None
    loop = values.get('loop')
    loop_ok = (loop < rule['inert_max_loop'] or math.isclose(loop, rule['inert_max_loop'], rel_tol=0, abs_tol=1e-12)) if finite(loop) else None
    if pooled is None:
        label = 'UNDEFINED'
    elif pooled <= rule['suppressive_max_pooled']:
        label = 'A-ii'
    elif pooled >= rule['inert_min_pooled'] and loop_ok is None:
        label = 'UNDEFINED'
    elif pooled >= rule['inert_min_pooled'] and loop_ok:
        label = 'A-i'
    else:
        label = 'intermediate'
    return dict(label=label, reading=READINGS[label], pooled_resolved=pooled, denominator=rule['denominator'], loop=loop,
                predicates={'pooled_ge_4': None if pooled is None else pooled >= rule['inert_min_pooled'],
                            'loop_le_0.10': loop_ok,
                            'pooled_le_1': None if pooled is None else pooled <= rule['suppressive_max_pooled']},
                boundary_numerics='absolute tolerance 1e-12 only for floating-point equality at loop 0.10')


def qualify(generation, binding, ledger):
    source = ledger.pin(binding['source'], binding['source_sha256'])
    ledger.pin(binding['runtime'], binding['runtime_sha256'])
    runtime_doc, frozen, completed, runs = module(source).qualify_generation(binding['runtime_sha256'])
    require(sha(READOUT) == READOUT_SHA and frozen == read_json(READOUT), 'Original frozen readout changed')
    require(runtime_doc.get('generation') == completed.get('generation') == generation and
            completed.get('runtime_sha256') == binding['runtime_sha256'] and completed.get('readout_sha256') == READOUT_SHA and
            completed.get('submission_started') is False and completed.get('cleanup_completed') is True,
            'Completion binding/scope mismatch: ' + generation)
    require(runtime_doc['memory_correction'] == completed['memory_correction'], 'Memory correction differs: ' + generation)
    proof = completed['admission_receipts']
    for kind in ('controls', 'replay'):
        ledger.pin(proof[kind + '_path'], proof[kind + '_sha256'])
    for run in runtime_doc['generation_runs']:
        receipt = runs[run['name']]['receipt']
        require(receipt['generation'] == generation and receipt['scientific_name'] == run['scientific_name'] and
                receipt['result_dir'] == run['result_dir'], 'Cross-generation result receipt refused: ' + run['name'])
        ledger.pin(receipt['memory_guard']['path'], receipt['memory_guard']['sha256'])
    for path in sorted(EVIDENCE[generation].rglob('*.json')):
        ledger.pin(path)
    return runtime_doc, completed, runs


def sealed_records(path, effective_runs, slots):
    names = {run['name']: run for run in effective_runs if run['arm'] in slots}
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        run = None if row is None else names.get(row['run'])
        if run is not None:
            key = (run['arm'], run['decoding_seed'], row['instance_id'])
            require(key not in result, 'Duplicate sealed record: ' + str(key))
            result[key] = row
    return result


def compare_records(records, sealed, slots):
    mine = {key: json.loads(json.dumps(value, allow_nan=False)) for key, value in records.items() if key[0] in slots}
    require(set(mine) == set(sealed) and len(sealed) == 48 * len(SEEDS) * len(slots), 'Sealed record grid differs')
    mismatched = sorted(str(key) for key in mine if mine[key] != sealed[key])
    require(not mismatched, 'Recomputed records differ from sealed v043 rows: ' + ', '.join(mismatched[:5]))
    return len(mine)


def producer_cli_check(treated_runs, population, tokenizer_path, records):
    """Independent producer CLI pass over the treated slot on primary40, in a fresh temporary directory."""
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        command = [sys.executable, str(PRODUCER), '--task-ids', str(PRIMARY_IDS), '--tokenizer', tokenizer_path,
                   '--json', str(out / 'summary.json'), '--per-task', str(out / 'per_task.jsonl')]
        for run in treated_runs:
            command += ['--run-dir', run['result_dir']]
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        require(result.returncode == 0, 'Producer CLI failed: ' + result.stderr[-800:])
        files = {name: (out / name).read_bytes() for name in ('summary.json', 'per_task.jsonl', 'summary.provenance.json')}
    by_run = {run['name']: run for run in treated_runs}
    keys = ('n_tasks', 'missing_traces', 'think_rate_mean', 'loop_rate', 'submit_rate')
    summaries = {}
    for summary in json.loads(files['summary.json']):
        run = by_run[summary['run']]
        mine = population['per_run_summaries'][run['arm']][run['decoding_seed'] - 1]
        summaries[summary['run']] = {key: summary[key] == mine[key] for key in keys}
    rows = [json.loads(line) for line in files['per_task.jsonl'].decode().splitlines() if line.strip()]
    unequal = [row['run'] + '/' + row['instance_id'] for row in rows if row != json.loads(json.dumps(
        records[(by_run[row['run']]['arm'], by_run[row['run']]['decoding_seed'], row['instance_id'])], allow_nan=False))]
    require(set(summaries) == set(by_run) and all(all(v.values()) for v in summaries.values()),
            'Producer CLI did not reproduce think/loop/submit: ' + json.dumps(summaries))
    require(len(rows) == 40 * len(treated_runs) and not unequal, 'Producer CLI per-task rows differ: ' + ', '.join(unequal[:5]))
    return dict(command=[str(part) for part in command[1:]], compared_keys=list(keys), summaries_equal=summaries,
                per_task_rows_equal=len(rows)), files


def arm_means(population):
    return {slot: {key: mean_three([row[field] for row in population['per_run_summaries'][slot]])
                   for key, field in (('think', 'think_rate_mean'), ('loop', 'loop_rate'), ('submit', 'submit_rate'))}
            for slot in SLOTS}


def frozen_slots():
    return {slot: dict(generation=spec[0], source_arm=spec[1], output_label=spec[2]) for slot, spec in SLOTS.items()}


def bindings(v052_runtime_sha):
    (s48, r48), (s43, r43) = SOURCES['v052'], SOURCES['v043']
    return {'v052': dict(source=str(s48), source_sha256=sha(s48), runtime=str(r48), runtime_sha256=v052_runtime_sha),
            'v043': dict(source=str(s43), source_sha256=V043_SOURCE_SHA, runtime=str(r43), runtime_sha256=V043_RUNTIME_SHA)}


def statistics_modules(ledger):
    for path, digest in ((RESOLUTION, RESOLUTION_SHA), (VALIDATOR, VALIDATOR_SHA), (PRODUCER, PRODUCER_SHA)):
        ledger.pin(path, digest)
    return module(RESOLUTION), module(VALIDATOR), module(PRODUCER)


def prepare(v052_runtime_sha):
    """Freeze slots, sign, populations, Probe A rule and pins; reads no v052 outcome (none need exist)."""
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    populations = validate_populations(readout)
    require(sha(PREREG) == PREREG_SHA, 'Frozen Probe A prediction note changed')
    require(sorted(PRIMARY_IDS.read_text().split()) == populations['primary40'], 'primary40 id file differs from the readout')
    pins = {**readout['pins'], str(READOUT): READOUT_SHA, str(LEGACY): LEGACY_SHA, str(PRODUCER): PRODUCER_SHA,
            str(VALIDATOR): VALIDATOR_SHA, str(RESOLUTION): RESOLUTION_SHA, str(PREREG): PREREG_SHA,
            str(PRIMARY_IDS): sha(PRIMARY_IDS), str(Path(__file__).resolve()): sha(__file__), str(TESTS): sha(TESTS),
            **{str(path): sha(path) for path in SEALED_V043.values()}}
    for path, value in pins.items():
        require(sha(path) == value, 'Preparation source drift: ' + path)
    for path in SEALED_V043.values():
        require(Path(str(path) + '.sha256').read_text().split()[0] == pins[str(path)], 'Sealed v043 sidecar mismatch')
    binding = bindings(v052_runtime_sha)
    for item in binding.values():
        require(sha(item['source']) == item['source_sha256'] and sha(item['runtime']) == item['runtime_sha256'],
                'Evaluation binding changed')
    v052, v043 = read_json(binding['v052']['runtime']), read_json(binding['v043']['runtime'])
    require(v052['generation'] == 'v052' and v043['generation'] == 'v043' and
            v052['memory_correction'] == v043['memory_correction'], 'Generation/memory declaration mismatch')
    slot_readout(readout, slot_runs(v043['generation_runs'], 'M0', 'M0', 'v043'),
                 slot_runs(v052['generation_runs'], 'A0cap', 'M1', 'v052'))
    doc = {'version': 'v052-probe-a-a0cap-vs-m0', 'generation': 'v052', 'status': 'PREPARED_BEFORE_ACTIVATION_NO_OUTCOMES',
           'recorded_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA, 'prereg_sha256': PREREG_SHA,
           'slots': frozen_slots(), 'direction': DIRECTION, 'probe_a_rule': PROBE_A, 'readings': READINGS,
           'populations': sorted(populations), 'evaluation_bindings': binding, 'pins': pins, 'output': str(OUTPUT),
           'self_checks': ['recomputed M0 records equal the sealed analysis/mask-control-v043/per_task.jsonl rows',
                           'producer CLI with primary40_ids.txt on the A0cap result dirs reproduces think/loop/submit and per-task rows'],
           'server_flag_difference': 'M0 (v043) was served by a LoRA-on server with six adapters; A0cap (v052) by a LoRA-off base server as the sealed A0',
           'execution_authorized': False, 'requires': 'Qualified v052 completion and explicit --run with this manifest hash'}
    write_fresh(MANIFEST, json.dumps(doc, indent=2) + '\n')
    write_fresh(str(MANIFEST) + '.sha256', sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    return sha(MANIFEST)


def selfcheck():
    """Rebuild the sealed v043 M0/M1 readout through this wrapper's slot path; no v052 input, writes nothing."""
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    ledger = Inputs()
    v043, _, runs = qualify('v043', bindings(None)['v043'], ledger)
    effective = slot_readout(readout, slot_runs(v043['generation_runs'], 'M0', 'M0', 'v043'),
                             slot_runs(v043['generation_runs'], 'M1', 'M1', 'v043'))
    qualified = {run['name']: runs[run['name']] for run in effective['future_runs']}
    resolution, validator, producer = statistics_modules(ledger)
    outcomes, records, rows_by_run, _ = metrics_for_runs(effective, qualified, producer, validator, ledger)
    count = compare_records(records, sealed_records(SEALED_V043['per_task'], effective['future_runs'], ('M0', 'M1')), ('M0', 'M1'))
    populations = {name: population_report(effective, ids, outcomes, records, rows_by_run, producer, resolution, validator)
                   for name, ids in validate_populations(effective).items()}
    require(json.loads(json.dumps(populations, allow_nan=False)) == read_json(SEALED_V043['report'])['populations'],
            'Recomputed v043 populations differ from the sealed report')
    treated = [run for run in effective['future_runs'] if run['arm'] == 'M1']
    cli, _ = producer_cli_check(treated, populations['primary40'], readout['secondary_mechanisms']['tokenizer'], records)
    ledger.verify()
    return {'status': 'selfcheck_passed', 'records_equal_sealed': count, 'populations_equal_sealed': True,
            'producer_cli_on_v043_M1': cli, 'v052_inputs_read': False, 'outputs_written': False}


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha and Path(str(MANIFEST) + '.sha256').read_text().split()[0] == manifest_sha,
            'Analysis manifest mismatch')
    doc = read_json(MANIFEST)
    require(doc['root'] == str(ROOT) and doc['output'] == str(OUTPUT) and doc['slots'] == frozen_slots() and
            doc['direction'] == DIRECTION and doc['probe_a_rule'] == PROBE_A and doc['readings'] == READINGS,
            'Frozen analysis contract changed')
    ledger = Inputs()
    for path, value in doc['pins'].items():
        ledger.pin(path, value)
    ledger.pin(MANIFEST, manifest_sha)
    v052, done48, runs48 = qualify('v052', doc['evaluation_bindings']['v052'], ledger)
    v043, done43, runs43 = qualify('v043', doc['evaluation_bindings']['v043'], ledger)
    require(v052['memory_correction'] == v043['memory_correction'], 'Generations do not share the capped memory policy')
    readout = read_json(READOUT)
    effective = slot_readout(readout, slot_runs(v043['generation_runs'], 'M0', 'M0', 'v043'),
                             slot_runs(v052['generation_runs'], 'A0cap', 'M1', 'v052'))
    qualified = {run['name']: (runs43 if run['source_generation'] == 'v043' else runs48)[run['name']]
                 for run in effective['future_runs']}
    resolution, validator, producer = statistics_modules(ledger)
    outcomes, records, rows_by_run, tokenizer = metrics_for_runs(effective, qualified, producer, validator, ledger)
    m0_equal = compare_records(records, sealed_records(SEALED_V043['per_task'], effective['future_runs'], ('M0',)), ('M0',))
    populations = {name: population_report(effective, ids, outcomes, records, rows_by_run, producer, resolution, validator)
                   for name, ids in validate_populations(effective).items()}
    inputs = prediction_inputs(populations['primary40'])
    probe = probe_a_label(inputs[PROBE_A['slot']])
    treated = [run for run in effective['future_runs'] if run['arm'] == 'M1']
    cli, cli_files = producer_cli_check(treated, populations['primary40'], readout['secondary_mechanisms']['tokenizer'], records)
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_exploratory_probe_a_readout',
        'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
        'prereg_sha256': PREREG_SHA, 'slots': frozen_slots(), 'direction': DIRECTION,
        'bootstrap_roles': {'treated': 'A0cap', 'baseline': 'M0'}, 'generation_runs': effective['future_runs'],
        'populations': rename_slots(populations), 'arm_means': rename_slots({n: arm_means(p) for n, p in populations.items()}),
        'probe_a': {'inputs': rename_slots(inputs), **probe, 'rule': PROBE_A},
        'self_checks': {'m0_records_equal_sealed_v043': m0_equal, 'producer_cli': cli},
        'memory_correction': v052['memory_correction'], 'tokenizer': tokenizer, 'method': readout['bootstrap'],
        'limitations': readout['exposure_and_claim_limits'] + [
            'A0cap and M0 share the v043 memory policy but differ in the server LoRA flag (A0cap LoRA-off as the sealed A0; M0 LoRA-on with six adapters); interchangeability of base outputs across the flag is untested.',
            'Historical A0 runs were uncapped; A0cap versus historical A0 is descriptive only.',
            'Probe A thresholds are exploration-informed (prereg cc043e03, thresholds unchanged since 3f9fc19f); readings are descriptive, not confirmatory.'],
        'inputs': ledger.hashes, 'evaluation_completion': {'v052': done48, 'v043': done43},
        'automatic_training_or_submission': False}
    require(not OUTPUT.exists(), 'Fresh v052 analysis output required')
    OUTPUT.mkdir()
    write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_fresh(OUTPUT / 'per_task.jsonl', ''.join(json.dumps(v, allow_nan=False) + '\n' for _, v in sorted(records.items())))
    write_fresh(OUTPUT / 'report.md', markdown(report))
    (OUTPUT / 'producer-cli').mkdir()
    for name, data in cli_files.items():
        with (OUTPUT / 'producer-cli' / name).open('xb') as stream:
            stream.write(data)
    for path in sorted(p for p in OUTPUT.rglob('*') if p.is_file()):
        write_fresh(str(path) + '.sha256', sha(path) + '  ' + path.name + '\n')
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json')}


def markdown(report):
    pct = lambda x: 'undefined' if x is None else f'{x:.4f}'
    probe = report['probe_a']
    lines = ['# Probe A (v052): A0cap versus M0 in the capped v043 environment', '',
             'Exploratory known-holdout readout. A0cap is the untrained base on a LoRA-off server; M0 is the fresh v043 M0 adapter on a LoRA-on server.',
             'Effects are A0cap minus M0 (v042 slot M1 minus slot M0) with the 20,000-draw whole-task bootstrap; all intervals are descriptive.', '',
             f"Probe A label (primary40): {probe['label']}. Pooled resolved {probe['pooled_resolved']}/{probe['denominator']}, "
             f"loop {pct(probe['loop'])}; predicates {json.dumps(probe['predicates'])}.", '', 'Reading: ' + probe['reading'], '',
             '| Population | A0cap resolved by seed | M0 resolved by seed | A0cap - M0 (pp) | 95% CI (pp) |', '|---|---|---|---:|---|']
    for name, population in report['populations'].items():
        p = population['primary']
        lines.append(f"| {name} | {p['arms']['A0cap']['resolved_by_seed']} | {p['arms']['M0']['resolved_by_seed']} | "
                     f"{p['effect_pp']:.3f} | [{p['ci95_pp'][0]:.3f}, {p['ci95_pp'][1]:.3f}] |")
    lines += ['', '| Population | Arm | think | loop | submit |', '|---|---|---:|---:|---:|']
    for name, arms in report['arm_means'].items():
        for arm, values in arms.items():
            lines.append(f"| {name} | {arm} | {pct(values['think'])} | {pct(values['loop'])} | {pct(values['submit'])} |")
    lines += ['', '| Secondary metric, primary40 | A0cap - M0 | 95% CI | Units |', '|---|---:|---|---|']
    for metric, item in report['populations']['primary40']['secondary'].items():
        effect, interval = item.get('effect_pp', item['effect']), item.get('ci95_pp', item['ci95'])
        text = 'undefined' if interval is None else f'[{interval[0]:.3f}, {interval[1]:.3f}]'
        lines.append(f"| {metric} | {pct(effect)} | {text} | {'percentage points' if item['units'] == 'proportion' else item['units']} |")
    checks = report['self_checks']
    lines += ['', f"Self-checks: recomputed M0 records equal the sealed v043 rows ({checks['m0_records_equal_sealed_v043']}); "
              f"producer CLI reproduces think/loop/submit for {len(checks['producer_cli']['summaries_equal'])} runs and "
              f"{checks['producer_cli']['per_task_rows_equal']} per-task rows.", '', 'Limitations:']
    lines += ['- ' + item for item in report['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--runtime-sha')
    ap.add_argument('--manifest-sha')
    args = ap.parse_args()
    require(sum((args.prepare, args.selfcheck, args.run)) <= 1, 'Prepare, selfcheck and run are separate')
    if args.prepare:
        require(args.runtime_sha is not None, 'Preparation requires the v052 runtime SHA')
        print(json.dumps({'status': 'prepared_only', 'manifest_sha256': prepare(args.runtime_sha), 'outcomes_read': False}))
    elif args.selfcheck:
        print(json.dumps(selfcheck()))
    elif args.run:
        require(args.manifest_sha is not None, 'Explicit manifest SHA required')
        print(json.dumps(analyze(args.manifest_sha)))
    else:
        require(args.manifest_sha is not None and sha(MANIFEST) == args.manifest_sha, 'Inspection requires manifest SHA')
        doc = read_json(MANIFEST)
        print(json.dumps({'status': 'prepared_only', 'slots': doc['slots'], 'direction': doc['direction'], 'outcomes_read': False}))


if __name__ == '__main__':
    main()
