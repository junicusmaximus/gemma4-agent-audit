#!/usr/bin/env python3
"""v054 readout: M0q (v054) against the sealed v043 M0 through the immutable v042 statistics."""
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
_spec = importlib.util.spec_from_file_location('v054_immutable_statistics', LEGACY)
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
SOURCES = {'v054': (ROOT / 'tools/run_m0q_eval_v054.py', ROOT / 'contracts/m0q-eval-v054-runtime-DRAFT.json'),
           'v043': (ROOT / 'tools/run_mask_evaluations_v043.py', ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json')}
V043_SOURCE_SHA = '9b39a600adb998117a694df5dfd61514f03e29b467795b58bc449b495f8ca590'
V043_RUNTIME_SHA = '7eaf5d53b56e02d6e45ef29a229748b2716bdee36ae7b4ebf0301450d4c57a63'
EVIDENCE = {'v054': ROOT / 'migration/m0q-eval-v054/evaluation', 'v043': ROOT / 'migration/mask-evaluations-v043/evaluation'}
SEALED_V043 = {'per_task': ROOT / 'analysis/mask-control-v043/per_task.jsonl', 'report': ROOT / 'analysis/mask-control-v043/report.json'}
PRIMARY_IDS = ROOT / 'analysis/mechanism-v032/primary40_ids.txt'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-06-qat-base-lora-fixture-and-M0q.md'
TESTS = ROOT / 'tools/test_analyze_m0q_eval_v054.py'
MANIFEST = ROOT / 'contracts/m0q-eval-v054-analysis-DRAFT.json'
OUTPUT = ROOT / 'analysis/m0q-eval-v054'
# Frozen before any M0q outcome exists: v042 slot -> (generation, source arm, output label).
SLOTS = {'M0': ('v043', 'M0', 'M0'), 'M1': ('v054', 'M0q', 'M0q')}
DIRECTION = 'M0q minus M0 = v042 slot M1 minus slot M0 (sign +1); slot keys renamed only in the written output'
READING_RULE = dict(population='primary40', slot='M1', interval='95% whole-task paired bootstrap (v042 statistics)',
                    metrics={'resolution': 'primary', 'loop': 'M2.loop_rate', 'submit': 'M3.submit_rate',
                             'm5_error': 'M5.nonrecovery.error', 'repro_first': 'M4.opener.repro_script_opener'},
                    descriptive_only={'think': 'M1.observed_thinking_step_rate'}, notable_min_pooled=4, denominator=120)
READINGS = {'differs': 'M0q differs from M0 beyond the paired interval (the 95% interval excludes 0)',
            'does_not_differ': 'M0q does not differ from M0 beyond the paired interval',
            'undefined': 'paired interval undefined; no reading'}


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


def interval_reading(effect, interval):
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    if not (finite(effect) and isinstance(interval, (list, tuple)) and len(interval) == 2 and all(finite(v) for v in interval)):
        return dict(effect=effect, ci95=interval, reading='undefined')
    return dict(effect=effect, ci95=list(interval), reading='differs' if interval[0] > 0 or interval[1] < 0 else 'does_not_differ')


def contrast_readings(population, treated, rule=READING_RULE):
    """Prereg section 3: per metric only 'differs / does not differ beyond the paired interval'; resolution >= 4/120 notable."""
    result = {}
    for key, metric in rule['metrics'].items():
        item = population['primary'] if metric == 'primary' else population['secondary'][metric]
        result[key] = interval_reading(item.get('effect_pp', item.get('effect')), item.get('ci95_pp', item.get('ci95')))
    for key, metric in rule['descriptive_only'].items():
        item = population['secondary'][metric]
        result[key] = dict(effect=item.get('effect_pp', item['effect']), ci95=item.get('ci95_pp', item['ci95']), reading='descriptive')
    counts = treated.get('solved_by_seed')
    valid = isinstance(counts, (list, tuple)) and len(counts) == 3 and all(type(v) is int and 0 <= v <= 40 for v in counts)
    pooled = sum(counts) if valid else None
    result['resolution_notable'] = dict(pooled_resolved=pooled, denominator=rule['denominator'],
                                        notable_and_bounded=None if pooled is None else pooled >= rule['notable_min_pooled'])
    return result


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


def bindings(v054_runtime_sha):
    (s54, r54), (s43, r43) = SOURCES['v054'], SOURCES['v043']
    return {'v054': dict(source=str(s54), source_sha256=sha(s54), runtime=str(r54), runtime_sha256=v054_runtime_sha),
            'v043': dict(source=str(s43), source_sha256=V043_SOURCE_SHA, runtime=str(r43), runtime_sha256=V043_RUNTIME_SHA)}


def statistics_modules(ledger):
    for path, digest in ((RESOLUTION, RESOLUTION_SHA), (VALIDATOR, VALIDATOR_SHA), (PRODUCER, PRODUCER_SHA)):
        ledger.pin(path, digest)
    return module(RESOLUTION), module(VALIDATOR), module(PRODUCER)


def prepare(v054_runtime_sha, prereg_sha):
    """Freeze slots, sign, populations, reading rule and pins before activation; reads no M0q outcome (none exists)."""
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    populations = validate_populations(readout)
    require(sha(PREREG) == prereg_sha, 'Binding M0q prereg differs from the declared sha')
    require(sorted(PRIMARY_IDS.read_text().split()) == populations['primary40'], 'primary40 id file differs from the readout')
    pins = {**readout['pins'], str(READOUT): READOUT_SHA, str(LEGACY): LEGACY_SHA, str(PRODUCER): PRODUCER_SHA,
            str(VALIDATOR): VALIDATOR_SHA, str(RESOLUTION): RESOLUTION_SHA, str(PREREG): prereg_sha,
            str(PRIMARY_IDS): sha(PRIMARY_IDS), str(Path(__file__).resolve()): sha(__file__), str(TESTS): sha(TESTS),
            **{str(path): sha(path) for path in SEALED_V043.values()}}
    for path, value in pins.items():
        require(sha(path) == value, 'Preparation source drift: ' + path)
    for path in SEALED_V043.values():
        require(Path(str(path) + '.sha256').read_text().split()[0] == pins[str(path)], 'Sealed v043 sidecar mismatch')
    binding = bindings(v054_runtime_sha)
    for item in binding.values():
        require(sha(item['source']) == item['source_sha256'] and sha(item['runtime']) == item['runtime_sha256'],
                'Evaluation binding changed')
    v054, v043 = read_json(binding['v054']['runtime']), read_json(binding['v043']['runtime'])
    require(v054['generation'] == 'v054' and v043['generation'] == 'v043' and v054['pending'] is None and
            v054['memory_correction'] == v043['memory_correction'], 'Generation/memory declaration mismatch or adapter pending')
    slot_readout(readout, slot_runs(v043['generation_runs'], 'M0', 'M0', 'v043'),
                 slot_runs(v054['generation_runs'], 'M0q', 'M1', 'v054'))
    doc = {'version': 'v054-m0q-vs-sealed-v043-m0', 'generation': 'v054', 'status': 'PREPARED_BEFORE_ACTIVATION_NO_OUTCOMES',
           'recorded_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA, 'prereg_sha256': prereg_sha,
           'slots': frozen_slots(), 'direction': DIRECTION, 'reading_rule': READING_RULE, 'readings': READINGS,
           'populations': sorted(populations), 'evaluation_bindings': binding, 'pins': pins, 'output': str(OUTPUT),
           'self_checks': ['recomputed M0 records equal the sealed analysis/mask-control-v043/per_task.jsonl rows',
                           'producer CLI with primary40_ids.txt on the M0q result dirs reproduces think/loop/submit and per-task rows'],
           'contrast': 'M0q (dequantized-QAT base) versus the sealed v043 M0 (bf16 base): same v043 template, capped environment, '
                       'engine build and seeds; the serving manifest registers three M0q aliases (max_loras 3) instead of six',
           'execution_authorized': False, 'requires': 'Qualified v054 completion and explicit --run with this manifest hash'}
    write_fresh(MANIFEST, json.dumps(doc, indent=2) + '\n')
    write_fresh(str(MANIFEST) + '.sha256', sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    return sha(MANIFEST)


def selfcheck():
    """Rebuild the sealed v043 M0/M1 readout through this wrapper's slot path; no v054 input, writes nothing."""
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
    readings = contrast_readings(populations['primary40'], prediction_inputs(populations['primary40'])['M1'])
    ledger.verify()
    return {'status': 'selfcheck_passed', 'records_equal_sealed': count, 'populations_equal_sealed': True,
            'producer_cli_on_v043_M1': cli, 'reading_function_on_sealed_M1_minus_M0': readings,
            'v054_inputs_read': False, 'outputs_written': False}


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha and Path(str(MANIFEST) + '.sha256').read_text().split()[0] == manifest_sha,
            'Analysis manifest mismatch')
    doc = read_json(MANIFEST)
    require(doc['root'] == str(ROOT) and doc['output'] == str(OUTPUT) and doc['slots'] == frozen_slots() and
            doc['direction'] == DIRECTION and doc['reading_rule'] == READING_RULE and doc['readings'] == READINGS,
            'Frozen analysis contract changed')
    ledger = Inputs()
    for path, value in doc['pins'].items():
        ledger.pin(path, value)
    ledger.pin(MANIFEST, manifest_sha)
    v054, done54, runs54 = qualify('v054', doc['evaluation_bindings']['v054'], ledger)
    v043, done43, runs43 = qualify('v043', doc['evaluation_bindings']['v043'], ledger)
    require(v054['memory_correction'] == v043['memory_correction'], 'Generations do not share the capped memory policy')
    readout = read_json(READOUT)
    effective = slot_readout(readout, slot_runs(v043['generation_runs'], 'M0', 'M0', 'v043'),
                             slot_runs(v054['generation_runs'], 'M0q', 'M1', 'v054'))
    qualified = {run['name']: (runs43 if run['source_generation'] == 'v043' else runs54)[run['name']]
                 for run in effective['future_runs']}
    resolution, validator, producer = statistics_modules(ledger)
    outcomes, records, rows_by_run, tokenizer = metrics_for_runs(effective, qualified, producer, validator, ledger)
    m0_equal = compare_records(records, sealed_records(SEALED_V043['per_task'], effective['future_runs'], ('M0',)), ('M0',))
    populations = {name: population_report(effective, ids, outcomes, records, rows_by_run, producer, resolution, validator)
                   for name, ids in validate_populations(effective).items()}
    inputs = prediction_inputs(populations['primary40'])
    readings = contrast_readings(populations[READING_RULE['population']], inputs[READING_RULE['slot']])
    treated = [run for run in effective['future_runs'] if run['arm'] == 'M1']
    cli, cli_files = producer_cli_check(treated, populations['primary40'], readout['secondary_mechanisms']['tokenizer'], records)
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_exploratory_m0q_readout',
        'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
        'prereg_sha256': doc['prereg_sha256'], 'slots': frozen_slots(), 'direction': DIRECTION,
        'bootstrap_roles': {'treated': 'M0q', 'baseline': 'M0'}, 'generation_runs': effective['future_runs'],
        'populations': rename_slots(populations), 'arm_means': rename_slots({n: arm_means(p) for n, p in populations.items()}),
        'readings': {'primary40': readings, 'inputs': rename_slots(inputs), 'rule': READING_RULE, 'labels': READINGS},
        'self_checks': {'m0_records_equal_sealed_v043': m0_equal, 'producer_cli': cli},
        'm0q_training': v054['m0q_training'], 'composition_check': v054['composition_check'],
        'memory_correction': v054['memory_correction'], 'tokenizer': tokenizer, 'method': readout['bootstrap'],
        'limitations': readout['exposure_and_claim_limits'] + [
            'M0q and M0 differ in the training base weights (dequantized QAT versus bf16 unquantized); data, groups, hyper-parameters, seed and LoRA init are matched; loss trajectory and optimizer numerics unavoidably differ.',
            'The v054 server registers three M0q aliases (max_loras 3); the sealed v043 server registered six adapters. Template, engine build, flags and memory policy are otherwise the v043 ones.',
            'The comparator is the sealed v043 M0 (v043 template, capped environment) as set by the prereg re-freeze of 2026-11-05 18:3x UTC; the matched-context v051 M0 named in prereg section 3 is not part of this contrast.',
            'Readings are descriptive (prereg section 3): differs / does not differ beyond the paired interval; resolution >= 4/120 is notable and bounded.'],
        'inputs': ledger.hashes, 'evaluation_completion': {'v054': done54, 'v043': done43},
        'automatic_training_or_submission': False}
    require(not OUTPUT.exists(), 'Fresh v054 analysis output required')
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
    span = lambda v: 'undefined' if v is None else f'[{v[0]:.3f}, {v[1]:.3f}]'
    readings = report['readings']['primary40']
    lines = ['# M0q (v054) versus the sealed v043 M0 in the capped v043 environment', '',
             'Exploratory known-holdout readout. M0q is M0 retrained on the dequantized QAT base; both are served LoRA-on with the v043 template.',
             'Effects are M0q minus M0 (v042 slot M1 minus slot M0) with the 20,000-draw whole-task bootstrap; all intervals are descriptive.', '',
             '| Reading, primary40 | M0q - M0 | 95% CI | Reading |', '|---|---:|---|---|']
    for key in list(READING_RULE['metrics']) + list(READING_RULE['descriptive_only']):
        item = readings[key]
        lines.append(f"| {key} | {pct(item['effect'])} | {span(item['ci95'])} | {item['reading']} |")
    notable = readings['resolution_notable']
    lines += ['', f"M0q pooled resolved (primary40): {notable['pooled_resolved']}/{notable['denominator']}; "
              f"notable and bounded (>= 4/120): {notable['notable_and_bounded']}.", '',
              '| Population | M0q resolved by seed | M0 resolved by seed | M0q - M0 (pp) | 95% CI (pp) |', '|---|---|---|---:|---|']
    for name, population in report['populations'].items():
        p = population['primary']
        lines.append(f"| {name} | {p['arms']['M0q']['resolved_by_seed']} | {p['arms']['M0']['resolved_by_seed']} | "
                     f"{p['effect_pp']:.3f} | {span(p['ci95_pp'])} |")
    lines += ['', '| Population | Arm | think | loop | submit |', '|---|---|---:|---:|---:|']
    for name, arms in report['arm_means'].items():
        for arm, values in arms.items():
            lines.append(f"| {name} | {arm} | {pct(values['think'])} | {pct(values['loop'])} | {pct(values['submit'])} |")
    lines += ['', '| Secondary metric, primary40 | M0q - M0 | 95% CI | Units |', '|---|---:|---|---|']
    for metric, item in report['populations']['primary40']['secondary'].items():
        effect, interval = item.get('effect_pp', item['effect']), item.get('ci95_pp', item['ci95'])
        lines.append(f"| {metric} | {pct(effect)} | {span(interval)} | {'percentage points' if item['units'] == 'proportion' else item['units']} |")
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
    ap.add_argument('--prereg-sha')
    ap.add_argument('--manifest-sha')
    args = ap.parse_args()
    require(sum((args.prepare, args.selfcheck, args.run)) <= 1, 'Prepare, selfcheck and run are separate')
    if args.prepare:
        require(args.runtime_sha is not None and args.prereg_sha is not None, 'Preparation requires the v054 runtime and prereg SHAs')
        print(json.dumps({'status': 'prepared_only', 'manifest_sha256': prepare(args.runtime_sha, args.prereg_sha), 'outcomes_read': False}))
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
