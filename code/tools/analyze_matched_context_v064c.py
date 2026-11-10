#!/usr/bin/env python3
"""Matched-context v064c readout (prereg v064-r2, readings v069): v066 A0mc in slot M0, the A_onpol confirmation (second training seed,
LoRA init seed 20261104) in slot M1, immutable v042 statistics (M1 - M0); seed 1 (the sealed v064a report) beside it, never pooled."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path('PROJECT_ROOT/gemma4')
LEGACY = ROOT / 'tools/analyze_mask_control_v042.py'
LEGACY_SHA = '23915fb00c2836b00fe14e3adfeecb865bdf52eb34e1fe3a04c4093cffdaf376'
if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
    raise ValueError('Immutable v042 statistical implementation changed')
_spec = importlib.util.spec_from_file_location('v064c_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
for _name in ('require', 'sha', 'read_json', 'Inputs', 'module', 'validate_populations', 'metrics_for_runs',
              'population_report', 'prediction_inputs', 'resolution_report', 'ARMS', 'SEEDS', 'READOUT', 'READOUT_SHA'):
    globals()[_name] = getattr(legacy, _name)
# MANIFEST is the pinned analysis manifest this tool reads; the draft is the staged template with the runtime binding and the v069
# readings addendum still pending.
MANIFEST = ROOT / 'contracts/matched-context-v064c-analysis-PINNED.json'
MANIFEST_DRAFT = ROOT / 'contracts/matched-context-v064c-analysis-DRAFT.json'
RUNTIME_DRAFT = ROOT / 'contracts/matched-context-v064c-runtime-DRAFT.json'
OUTPUT = ROOT / 'analysis/matched-context-v064c'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md'
PREREG_SHA = '0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60'
# Templates of this generation (readings v069: the v064a machinery re-pinned): the v064a analysis tool and its pinned manifest, which is
# also the manifest of seed 1's sealed report.
PARENT_ANALYZER = ROOT / 'tools/analyze_matched_context_v064a.py'
PARENT_ANALYZER_SHA = 'b33958febff6f503a250f1d1f1a8d66ecf531be39b44c97b3a89bec6c2fea122'
PARENT_ANALYSIS = ROOT / 'contracts/matched-context-v064a-analysis-PINNED.json'
PARENT_ANALYSIS_SHA = 'ae461ea43882a09a3d2ef2e7adbc0121a772eb287c2ca291b44cf5cc82199cc4'
PENDING = 'PENDING_PIN'
BASE = 'gemma-4-e4b-it-qat-w4a16-ct'
RUNNER = ROOT / 'tools/run_matched_context_v064c.py'
RUNTIME = ROOT / 'contracts/matched-context-v064c-runtime-PINNED.json'
DATA_MANIFEST = ROOT / 'sft/data/onpolicy_v064/A_onpol/manifest.json'
# Serving-form reference named by the runtime contract's comparator block (the runner keeps the v055 M0qmc generation there).
REFERENCE_RUNTIME = ROOT / 'contracts/matched-context-v055-runtime-DRAFT.json'
REFERENCE_RUNTIME_SHA = '776c7f7b4810437e285003492be9cc6465cc7c54eb7d9ab59a76f2317a9038d7'
REFERENCE_ANALYSIS = ROOT / 'analysis/matched-context-v055/report.json'
REFERENCE_ANALYSIS_SHA = '163990653084e14e8feb08c1a04b920222d4900f18f8868cdda06d47966713c4'
REFERENCE_PER_TASK = ROOT / 'analysis/matched-context-v055/per_task.jsonl'
REFERENCE_PER_TASK_SHA = '682e3492b1ff45bb1b48f0bf53b06d96d93dcb9780a1ffd0b8fed624aeb57b1e'
# Slot M0 (prereg v064-r2: A0mc in M0): the sealed v066 A0mc generation, re-qualified by its own runner, rows equal to its sealed analysis.
COMPARATOR_RUNNER = ROOT / 'tools/run_matched_context_v066.py'
COMPARATOR_RUNNER_SHA = '0542edbf7d8e3ee57c605b75e15f0d8bd9a090d9fddd9d37fff7abc0c2f72e8a'
COMPARATOR_RUNTIME = ROOT / 'contracts/matched-context-v066-a0mc-runtime-PINNED.json'
COMPARATOR_RUNTIME_SHA = '0f5e04b257ffe1544633f90fba4e51e377dae82a82f97bb029d1b8c3d49944e5'
COMPARATOR_ANALYSIS = ROOT / 'analysis/matched-context-v066/report.json'
COMPARATOR_ANALYSIS_SHA = 'c448ed2e31ff77a93a5b9094dfbd33c08e21d3da1a03de66bf60b6295def5f8e'
COMPARATOR_PER_TASK = ROOT / 'analysis/matched-context-v066/per_task.jsonl'
COMPARATOR_PER_TASK_SHA = '72fc58d913a2c0f6258ac3719dabdfc6777cf2896fecb601dd0897e8191465b8'
AUDIT = ROOT / 'analysis/serialization-audit-boss-v046/audit.py'
AUDIT_SHA = '5d26f4cdc9aafa2e9ffc1f527fd88d1ccd58497770f6b4a626b04ef5078cee98'
PRODUCER = ROOT / 'tools/mechanism_metrics_v032.py'
PRODUCER_SHA = '36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d'
VALIDATOR = ROOT / 'tools/analyze_mechanism_v031.py'
VALIDATOR_SHA = '821f2294661bf8928979d3df7da6c94fd8f662ee2ed17621a8c8dfaa06207478'
RESOLUTION = ROOT / 'tools/analyze_contract_v02_v025.py'
RESOLUTION_SHA = '20208c74cc921d9dd4975b1b6a7129b6a4b55825f17930429a4fa624935e229c'
PRIMARY_IDS = ROOT / 'analysis/mechanism-v032/primary40_ids.txt'
PRIMARY_IDS_SHA = 'c71257437c5c518ab0077f2ee59d61d1b2653a1a4a1ba2ded36f682ccda0cba2'
EVIDENCE = {'v064c': ROOT / 'migration/matched-context-v064c/evaluation', 'v066': ROOT / 'migration/matched-context-v066/evaluation'}
STATUS = {'v064c': 'all_three_a_onpol_c2_evaluations_qualified', 'v066': 'all_three_a0mc_evaluations_qualified'}
SLOT_GENERATION = {'M0': 'v066', 'M1': 'v064c'}
SLOT_ARM = {'M0': 'A0mc', 'M1': 'A_onpol'}
LABELS = {'M0': 'v066_A0mc', 'M1': 'v064_A_onpol_c2'}
SLOTS = {'M0': 'v066_A0mc_s1..s3: comparator, the E4B QAT w4a16 base without adapter under the matched template 855bbbbb (sealed v066 generation)',
         'M1': 'v064_A_onpol_c2_s1..s3: the A_onpol confirmation adapter (second training seed, LoRA init seed 20261104, of the prereg v064-r2 '
               'A_onpol recipe on the dequantized QAT base), same matched template (v064c generation)'}
SIGN = 'M1 - M0 = v064_A_onpol_c2 - v066_A0mc (v042 resolution_report and paired_slots take M1 as treated)'
POPULATIONS = ('primary40', 'all48_sensitivity', 'control_clean36_sensitivity')
THRESHOLDS = {'NON-COLLAPSED': {'loop_max': .30, 'submit_min': .80}, 'COLLAPSED': {'loop_above': .40, 'submit_below': .70},
              'repair_minimum_pooled_primary40': 4, 'boundary_abs_tolerance': 1e-12}
NON_COLLAPSED = 'did not meet the pre-defined collapse criteria'
CI_WORDS = {'lower_bound_above_zero': 'improved over the matched-template base', 'upper_bound_below_zero': 'worse than the base',
            'otherwise': 'uncertain relative to the base'}
ZERO_CI = 'A [0, 0] bootstrap is not evidence of equivalence.'
DESCRIPTIVES = {
    'think_rate': 'v032 producer M1: share of agent steps with a non-blank thinking event, i.e. non-empty returned reasoning',
    'compaction_calls': 'audit.trace_items LLM calls; call k (1-based, k >= 2) is a compaction call when call k-1 prompt_tokens >= audit.THRESHOLD (14,336), the audit rule at audit.py:165-166; event availability is not re-evaluated',
    'content_without_reasoning_calls': 'trace-observable proxy for responses without <channel|>: calls that directly follow a tool response of a previous call without visible text (the matched template prefills the opener there) and carry visible text but no non-blank thought; an empty thought followed by text looks the same, and a missing <channel|> with a bare tool call leaves no trace (reasoning None and "" both drop), so this count is an upper bound for the visible kind only',
    'secondary': 'find_opener and repro_script_opener shares (M4), max repeat (M2.max_repeat), M5 silent and normal repeat rates: v042 paired effects on primary40, descriptive only',
    'descriptive_rows': 'v055 M0qmc (M0q adapter, matched template) primary40 three-seed means cited from its sealed report (prereg v064-r2: M0q cited only); no re-analysis, no contrast',
    'scope': 'Reported for both slots, all48, per run and per task; descriptive, no threshold'}
DESCRIPTIVE_ROWS = {
    'v055_M0qmc': (REFERENCE_ANALYSIS, REFERENCE_ANALYSIS_SHA, 'readings.inputs_primary40_three_seed_means.v055_M0qmc')}
# Readings addendum v069 (pre-written confirmation sentences), frozen by Boss on 2026-11-08: every preparation binds this sha, checks the
# FROZEN status line and that every sentence below occurs in it verbatim.
ADDENDUM = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-08-onpolicy-confirmation-readings-v069.md'
ADDENDUM_SHA = 'fabeddb2c62e9df81f68c86355ddec7d6caa083fda788dcee61030f84373385b'
MINUS = chr(0x2212)  # the addendum writes negative numbers with U+2212
TERMINATION_SENTENCES = {
    'NON-COLLAPSED': ('T1', "training on the base's own successful trajectories did not meet the pre-defined collapse criteria in two training "
                            'seeds (loop a1 and a2, submit b1 and b2; resolved 10/120 and x/120) on the known holdout', ('a1', 'a2', 'b1', 'b2', 'x')),
    'INTERMEDIATE': ('T2', 'the second training seed was intermediate (loop a2, submit b2; resolved x/120); no-collapse is not stated',
                     ('a2', 'b2', 'x')),
    'COLLAPSED': ('T3', 'seed-dependent: the second training seed met the collapse criteria (loop a2, submit b2; resolved x/120)', ('a2', 'b2', 'x'))}
INTERVAL_APPENDIX = '(x/120 vs 12/120; d pp [l, u])'
SEED1_INTERVAL = '(10/120 vs 12/120; ' + MINUS + '1.67 pp [' + MINUS + '6.67, +2.50])'
# Seed 1 = the sealed v064a readout of the first A_onpol training seed: printed beside the confirmation from its pinned files, never
# re-analysed and never pooled with it. Its slot M0 is the same sealed v066 A0mc generation as here.
SEED1 = {'report': (ROOT / 'analysis/matched-context-v064a/report.json', '8f32f58bc4c9ce3f5b59cdd107c395d64164e0f8e399acbcb4fc07e18430bc25'),
         'report_md': (ROOT / 'analysis/matched-context-v064a/report.md', 'dd855d04664362982daa4234f87d711fe9d00229ba5433d5db24fa43c980d3a4'),
         'per_task': (ROOT / 'analysis/matched-context-v064a/per_task.jsonl', '28015851d67cdcc134250c38b3231501c8abb54a421beff758f2802c5e73e95c'),
         'completion': (ROOT / 'migration/matched-context-v064a/evaluation/completed.json',
                        '3b5677363cbf09cdceb4faf5a4d05ca9ecc5c90cfdbc9204db99e002124ceed3')}
SEED1_STATUS = 'qualified_exploratory_matched_context_a_onpol_readout'
SEED1_PREWRITTEN = {'pooled_resolved': 10, 'a0mc_pooled_resolved': 12, 'denominator': 120, 'label': 'NON-COLLAPSED',
                    'interval_pp_2dp': (-1.67, -6.67, 2.5)}


def slot_runs(readout):
    """The six runs the unchanged v042 code sees: v066 A0mc seeds in slot M0, v064 A_onpol seeds in slot M1 (arm = slot)."""
    runs = []
    for slot, prefix, alias_prefix, source in (('M0', 'v066_A0mc_s', None, None), ('M1', 'v064_A_onpol_c2_s', 'v064_A_onpol_c2_s', 'A1_onpol_s2_v064')):
        for seed in SEEDS:
            name = prefix + str(seed)
            alias = BASE if alias_prefix is None else alias_prefix + str(seed)
            runs.append({'name': name, 'scientific_name': name if alias_prefix is None else alias, 'arm': slot, 'adapter_alias': alias,
                         'adapter_source': None if source is None else str(ROOT / 'sft/runs' / source), 'temperature': .2, 'decoding_seed': seed,
                         'baseline_config': str(ROOT / ('subs/e4b_budgetfit_s' + str(seed))),
                         'submission_dir': str(ROOT / 'subs' / name), 'result_dir': str(ROOT / 'results' / name),
                         'trace_dir': str(ROOT / 'results' / name / 'traces')})
    require([(run['arm'], run['decoding_seed']) for run in runs] == [(a, s) for a in ARMS for s in SEEDS], 'Slot grid changed')
    return runs


def effective_readout(readout):
    effective = copy.deepcopy(readout)
    effective['future_runs'] = slot_runs(readout)
    effective['future_analysis_dir'] = str(OUTPUT)
    return effective


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def termination(row):
    """Prereg v064-r2 termination label of one arm from three-seed means: NON-COLLAPSED (loop <= .30 and submit >= .80), COLLAPSED
    (loop > .40 or submit < .70), else INTERMEDIATE; UNDEFINED when loop or submit is not finite. A value within the tolerance of a
    bound counts as equal to it (inside a closed bound, outside an open one)."""
    close = lambda x, y: math.isclose(x, y, rel_tol=0, abs_tol=THRESHOLDS['boundary_abs_tolerance'])
    loop, submit, nc, c = row.get('loop'), row.get('submit'), THRESHOLDS['NON-COLLAPSED'], THRESHOLDS['COLLAPSED']
    if not (finite(loop) and finite(submit)):
        return 'UNDEFINED'
    if (loop < nc['loop_max'] or close(loop, nc['loop_max'])) and (submit > nc['submit_min'] or close(submit, nc['submit_min'])):
        return 'NON-COLLAPSED'
    if (loop > c['loop_above'] and not close(loop, c['loop_above'])) or (submit < c['submit_below'] and not close(submit, c['submit_below'])):
        return 'COLLAPSED'
    return 'INTERMEDIATE'


def ci_key(primary):
    """The prereg rule on the arm - A0mc paired CI (pp): lower > 0, upper < 0, otherwise; a bound within the tolerance of 0 counts as 0."""
    low, high = primary['ci95_pp']
    tolerance = THRESHOLDS['boundary_abs_tolerance']
    return 'lower_bound_above_zero' if low > tolerance else 'upper_bound_below_zero' if high < -tolerance else 'otherwise'


def arm_readings(populations):
    """Per population: the arm's termination label, think / repro-first / M5 (undefined stays undefined, never 0), pooled resolved of both
    slots and the arm - A0mc paired CI with its pre-written wording. primary40 is the reading; all48 and clean36 are sensitivities."""
    out = {}
    for name in POPULATIONS:
        inputs, primary = prediction_inputs(populations[name]), populations[name]['primary']
        key, label = ci_key(primary), termination(inputs['M1'])
        pooled = {slot: sum(primary['arms'][slot]['resolved_by_seed']) for slot in ARMS}
        out[name] = {
            'role': 'reading' if name == 'primary40' else 'sensitivity',
            'termination': {'label': label, 'loop': inputs['M1']['loop'], 'submit': inputs['M1']['submit'],
                            'wording': NON_COLLAPSED if label == 'NON-COLLAPSED' else None},
            'descriptive': {LABELS[slot]: {'loop': inputs[slot]['loop'], 'submit': inputs[slot]['submit'], 'think': inputs[slot]['think'],
                                           'repro_first': inputs[slot]['repro'],
                                           'm5_error': inputs[slot]['m5_error'] if finite(inputs[slot]['m5_error']) else 'undefined'}
                            for slot in ARMS},
            'repair': {'pooled_resolved': pooled['M1'], 'a0mc_pooled_resolved': pooled['M0'], 'denominator': 3 * primary['task_count'],
                       'resolved_by_seed': primary['arms']['M1']['resolved_by_seed'],
                       'a0mc_resolved_by_seed': primary['arms']['M0']['resolved_by_seed'],
                       'minimum_observation': pooled['M1'] >= THRESHOLDS['repair_minimum_pooled_primary40'] if name == 'primary40' else None,
                       'effect_pp': primary['effect_pp'], 'ci95_pp': primary['ci95_pp'], 'ci_key': key, 'ci_wording': CI_WORDS[key],
                       'zero_width_ci': [float(bound) for bound in primary['ci95_pp']] == [0.0, 0.0]}}
    return out


def relabel(value):
    """Output-only renaming: slot keys M0/M1 and per-run summary names; numbers and order unchanged."""
    if isinstance(value, dict):
        out = {LABELS.get(key, key): relabel(item) for key, item in value.items()}
        if isinstance(out.get('run'), str) and out['run'][:3] in ('M0_', 'M1_'):
            out['run'] = LABELS[out['run'][:2]] + out['run'][2:]
        return out
    if isinstance(value, list):
        return [relabel(item) for item in value]
    return value


def call_view(audit, trace):
    """Per-trace LLM calls rebuilt by the audit's own trace_items (audit.py:86-126)."""
    items, calls, bad = audit.trace_items(trace)
    compaction = [k + 1 for k in range(1, len(calls)) if (calls[k - 1]['prompt_tokens'] or 0) >= audit.THRESHOLD]
    after_tool = [k for k in range(1, len(calls))
                  if items[calls[k]['item'] - 1][0] == 'tool' and not calls[k - 1]['text'].strip()]
    leaks = [k + 1 for k in after_tool if calls[k]['text'].strip() and not calls[k]['thought'].strip()]
    return {'llm_calls': len(calls), 'compaction_calls': compaction, 'after_tool_response_calls': len(after_tool),
            'content_without_reasoning_calls': leaks, 'unmapped': dict(bad)}


def trace_descriptives(readout, audit, producer, ids):
    per_task, per_run = {}, {}
    for run in readout['future_runs']:
        rows = []
        for task in sorted(ids):
            trace, reason = producer.load_trace(Path(run['result_dir']), task)
            require(reason is None, 'Qualified trace became unavailable: ' + task)
            view = call_view(audit, trace)
            per_task.setdefault(task, {}).setdefault(run['arm'], {})[run['decoding_seed']] = view
            rows.append(view)
        per_run[run['name']] = {'slot': run['arm'], 'tasks': len(rows), 'llm_calls': sum(r['llm_calls'] for r in rows),
            'compaction_calls': sum(len(r['compaction_calls']) for r in rows),
            'tasks_with_compaction': sum(bool(r['compaction_calls']) for r in rows),
            'after_tool_response_calls': sum(r['after_tool_response_calls'] for r in rows),
            'content_without_reasoning_calls': sum(len(r['content_without_reasoning_calls']) for r in rows),
            'unmapped_steps': sum(sum(r['unmapped'].values()) for r in rows)}
    compaction_by_task = {task: {slot: [len(seeds[s]['compaction_calls']) for s in SEEDS] for slot, seeds in slots.items()}
                          for task, slots in per_task.items()}
    return {'definitions': DESCRIPTIVES, 'per_run': per_run, 'compaction_counts_by_task': compaction_by_task,
            'per_task': per_task}


def comparator_matches_sealed(records, ledger):
    """The M0 slot must reproduce the v066 analysis per-task rows of v066 A0mc exactly (same producer, same traces)."""
    ledger.pin(COMPARATOR_PER_TASK, COMPARATOR_PER_TASK_SHA)
    names = {'v066_A0mc_s' + str(s) for s in SEEDS}
    writer_fields = ('slot', 'decoding_seed', 'trace_view')  # added by the v066 writer beside the producer row
    sealed = [json.loads(line) for line in COMPARATOR_PER_TASK.read_text().splitlines() if line.strip()]
    expected = {(row['run'], row['instance_id']): {k: v for k, v in row.items() if k not in writer_fields}
                for row in sealed if row['run'] in names}
    actual = {(row['run'], row['instance_id']): json.loads(json.dumps(row, allow_nan=False))
              for (slot, _, _), row in records.items() if slot == 'M0'}
    require(len(expected) == len(actual) == 144 and actual == expected, 'v066 A0mc metrics differ from the v066 analysis rows')
    return {'rows_compared': len(actual), 'equal': True, 'sealed_per_task_sha256': COMPARATOR_PER_TASK_SHA}


def a0mc_matches_sealed(populations, ledger):
    """Slot M0 pooled resolved in every population equals the sealed v066 report (readings.descriptive_by_population.<population>.v066_A0mc)."""
    ledger.pin(COMPARATOR_ANALYSIS, COMPARATOR_ANALYSIS_SHA)
    sealed = read_json(COMPARATOR_ANALYSIS)['readings']['descriptive_by_population']
    got = {name: sum(populations[name]['primary']['arms']['M0']['resolved_by_seed']) for name in POPULATIONS}
    require(got == {name: sealed[name]['v066_A0mc']['pooled_resolved'] for name in POPULATIONS},
            'v066 A0mc resolved counts differ from the sealed v066 report')
    return {'a0mc_pooled_resolved': got, 'sealed_report_sha256': COMPARATOR_ANALYSIS_SHA}


def training_data(completed, ledger):
    """The arm's training facts for the report and the joint reading (n trajectories, t tasks, U updates): the data manifest pinned by the
    qualified training receipt of the completion, its update count equal to the receipt's."""
    training = completed['arm_training']
    digest = training['artifacts'][str(DATA_MANIFEST)]
    ledger.pin(DATA_MANIFEST, digest)
    counts = read_json(DATA_MANIFEST)['counts']
    require(type(training['updates']) is int and counts['updates'] == training['updates'] == training['summary']['updates'],
            'Training data counts differ from the qualified training receipt')
    return {'updates': training['updates'], 'counts': counts, 'data_manifest': str(DATA_MANIFEST), 'data_manifest_sha256': digest,
            'training_contract_sha256': training['training_contract_sha256'], 'summary': training['summary'],
            'adapter_sha256': training['artifacts'][str(ROOT / 'sft/runs/A1_onpol_s2_v064/adapter_model.safetensors')]}


def qualify_generations(manifest):
    binding = manifest.get('evaluation_binding')
    require(isinstance(binding, dict) and set(binding) == {'v064c', 'v066'} and all(entry['runtime_sha256'] != PENDING for entry in binding.values()),
            'Evaluation binding is pending; --run refused')
    ledger = Inputs()
    for path, value in manifest['pins'].items():
        ledger.pin(path, value)
    frozen = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Original frozen readout changed')
    completions, qualified = {}, {}
    for key in ('v064c', 'v066'):
        source = ledger.pin(binding[key]['source'], binding[key]['source_sha256'])
        ledger.pin(binding[key]['runtime'], binding[key]['runtime_sha256'])
        _, readout, completed, runs = module(source).qualify_generation(binding[key]['runtime_sha256'])
        require(readout == frozen and completed.get('status') == STATUS[key] and completed.get('generation') == key and
                completed.get('runtime_sha256') == binding[key]['runtime_sha256'] and completed.get('readout_sha256') == READOUT_SHA and
                completed.get('submission_started') is False and completed.get('cleanup_completed') is True,
                'Completion binding/scope mismatch: ' + key)
        completions[key], qualified[key] = completed, runs
    effective = effective_readout(frozen)
    runs = {}
    for run in effective['future_runs']:
        key = SLOT_GENERATION[run['arm']]
        require(run['name'] in qualified[key], 'Missing qualified run: ' + run['name'])
        receipt = qualified[key][run['name']]['receipt']
        require(receipt.get('generation') == key and receipt.get('scientific_name') == run['scientific_name'] and
                receipt.get('result_dir') == run['result_dir'] and receipt.get('alias') == run['adapter_alias'] and
                receipt.get('arm') == SLOT_ARM[run['arm']] and receipt.get('decoding_seed') == run['decoding_seed'], 'Slot binding refused: ' + run['name'])
        ledger.pin(receipt['memory_guard']['path'], receipt['memory_guard']['sha256'])
        runs[run['name']] = qualified[key][run['name']]
    for folder in EVIDENCE.values():
        for path in sorted(folder.rglob('*.json')):
            ledger.pin(path)
    ledger.verify()
    return effective, completions, runs, ledger


def descriptive_rows(ledger):
    """Primary40 three-seed means of the pre-declared descriptive rows, read from their pinned reports (no re-analysis)."""
    rows = {}
    for label, (path, digest, where) in DESCRIPTIVE_ROWS.items():
        ledger.pin(path, digest)
        doc = read_json(path)
        if where.startswith('prediction_inputs'):
            value = prediction_inputs(doc['populations']['primary40'])[where.rsplit('.', 1)[1]]
        else:
            value = doc
            for key in where.split('.'):
                value = value[key]
        rows[label] = dict(source=str(path), source_sha256=digest, location=where,
                           **{key: value.get(key) for key in ('loop', 'submit', 'm5_error', 'think', 'repro', 'solved_by_seed')})
    return rows


def write_or_verify(path, text):
    if Path(path).exists():
        require(Path(path).read_text() == text, 'Analysis preparation is not reproducible: ' + str(path))
        return
    with Path(path).open('x') as stream:
        stream.write(text)


def prereg_readings():
    """The Evaluation line of section Arms and the lines of section Readouts of the frozen pre-registration, verbatim."""
    text = PREREG.read_text(encoding='utf-8')
    section = lambda header: text.split(chr(10) + header, 1)[1].split(chr(10) + '## ', 1)[0].splitlines()[1:]
    return ([line for line in section('## Arms') if line.startswith('- Evaluation:')] +
            [line for line in section('## Readouts') if line.startswith('- ')])


def addendum_readings(addendum_sha):
    """The frozen v069 readings addendum: its sha, its FROZEN status line and the verbatim bullet lines of section Readings, in which every
    pre-written sentence this analyzer fills must occur verbatim (the termination, interval and CI sentences quoted, seed 1's interval)."""
    require(sha(ADDENDUM) == addendum_sha, 'v069 readings addendum differs from the given sha')
    text = ADDENDUM.read_text(encoding='utf-8')
    head = text.split(chr(10) + '---' + chr(10), 1)[0]
    require(text.startswith('---' + chr(10)) and (chr(10) + 'status: FROZEN') in head, 'v069 readings addendum is not frozen')
    section = text.split(chr(10) + '## Readings', 1)[1].split(chr(10) + '## ', 1)[0].splitlines()[1:]
    lines = [line for line in section if line.lstrip().startswith('- ')]
    joined = chr(10).join(lines)
    quoted = [template for _, template, _ in TERMINATION_SENTENCES.values()] + [INTERVAL_APPENDIX] + list(CI_WORDS.values())
    require(all('"' + sentence + '"' in joined for sentence in quoted) and SEED1_INTERVAL in joined,
            'v069 readings addendum does not carry the pre-written sentences verbatim')
    return lines


def seed1_readout(ledger):
    """Seed 1 from its sealed v064a files (pinned, read only): status, manifest and completion binding, and per population its termination
    and repair readings exactly as sealed. Nothing is recomputed and nothing is pooled with the confirmation."""
    for path, digest in SEED1.values():
        ledger.pin(path, digest)
    ledger.pin(PARENT_ANALYSIS, PARENT_ANALYSIS_SHA)
    report = read_json(SEED1['report'][0])
    completion, digest = SEED1['completion']
    require(report['status'] == SEED1_STATUS and report['analysis_manifest_sha256'] == PARENT_ANALYSIS_SHA and
            report['inputs'].get(str(completion)) == digest, 'Seed 1 report is not the sealed v064a readout')
    arm = report['readings']['arm']
    return {'source': {name: dict(path=str(path), sha256=value) for name, (path, value) in SEED1.items()},
            'analysis_manifest_sha256': PARENT_ANALYSIS_SHA, 'status': report['status'], 'label': 'v064_A_onpol (seed 1, sealed v064a)',
            'by_population': {name: {'termination': arm[name]['termination'], 'repair': arm[name]['repair']} for name in POPULATIONS}}


def fill(template, tokens, values):
    """A pre-written sentence with its placeholder words replaced by the read values; each placeholder occurs exactly once."""
    for token in tokens:
        pattern = '(?<![A-Za-z0-9])' + token + '(?![A-Za-z0-9])'
        require(len(re.findall(pattern, template)) == 1, 'Placeholder ' + token + ' does not occur exactly once')
        template = re.sub(pattern, values[token], template)
    return template


def confirmation_readings(readings, seed1):
    """The v069 sentences on primary40: the termination sentence of the confirmation's label (T1 NON-COLLAPSED, T2 INTERMEDIATE, T3
    COLLAPSED; UNDEFINED has none), the interval sentence (the v064-r2 CI words with the counts and interval appended) and the prose
    flags. Seed 1 must carry what the addendum pre-writes (10/120 vs 12/120, NON-COLLAPSED, -1.67 pp [-6.67, +2.50]) and share the A0mc slot."""
    mine, first = readings['primary40'], seed1['by_population']['primary40']
    two, one, expected = mine['repair'], first['repair'], SEED1_PREWRITTEN
    require(one['pooled_resolved'] == expected['pooled_resolved'] and first['termination']['label'] == expected['label'] and
            one['a0mc_pooled_resolved'] == two['a0mc_pooled_resolved'] == expected['a0mc_pooled_resolved'] and
            one['denominator'] == two['denominator'] == expected['denominator'] and
            tuple(round(value, 2) for value in [one['effect_pp']] + list(one['ci95_pp'])) == expected['interval_pp_2dp'],
            'Seed 1 or the A0mc count differs from what the v069 sentences pre-write')
    require(all(seed1['by_population'][name]['repair']['a0mc_pooled_resolved'] == readings[name]['repair']['a0mc_pooled_resolved'] and
                seed1['by_population'][name]['repair']['denominator'] == readings[name]['repair']['denominator'] for name in POPULATIONS),
            'Seed 1 and the confirmation do not share the sealed A0mc comparator')
    label = mine['termination']['label']
    values = dict(a1=fmt(first['termination']['loop']), b1=fmt(first['termination']['submit']), a2=fmt(mine['termination']['loop']),
                  b2=fmt(mine['termination']['submit']), x=str(two['pooled_resolved']), d='%+.2f' % two['effect_pp'],
                  l='%+.2f' % two['ci95_pp'][0], u='%+.2f' % two['ci95_pp'][1])
    sentence = None
    if label in TERMINATION_SENTENCES:
        key, template, tokens = TERMINATION_SENTENCES[label]
        sentence = dict(key=key, template=template, text=fill(template, tokens, values))
    return {'source': 'v069 readings addendum (reading_addendum of the analysis manifest)', 'population': 'primary40', 'label': label,
            'termination_sentence': sentence, 'interval_sentence': CI_WORDS[two['ci_key']] + ' ' + fill(INTERVAL_APPENDIX, ('x', 'd', 'l', 'u'), values),
            'ci_key': two['ci_key'], 'no_collapse_prose_allowed': label == 'NON-COLLAPSED',
            'improved_prose_allowed': two['ci_key'] == 'lower_bound_above_zero', 'seed1_interval': SEED1_INTERVAL,
            'seed1_interval_ascii': '(%d/120 vs %d/120; %+.2f pp [%+.2f, %+.2f])' % (
                one['pooled_resolved'], one['a0mc_pooled_resolved'], one['effect_pp'], one['ci95_pp'][0], one['ci95_pp'][1])}


def confirmation_lines(confirmation, seed1, reading, arm):
    """Report text of the v069 readings: the applicable sentences, the prose flags and both training seeds side by side (never pooled)."""
    sentence = confirmation['termination_sentence']
    lines = ['Confirmation readings (readings addendum v069, primary40):', '',
             'Termination sentence %s: "%s"' % (sentence['key'], sentence['text']) if sentence else
             'Termination ' + confirmation['label'] + ': no pre-written termination sentence applies.',
             'Interval sentence: "' + confirmation['interval_sentence'] + '"',
             'Prose flags: two-seed collapse-criteria prose %s (needs the confirmation NON-COLLAPSED); "improved" prose %s (needs the '
             "confirmation's lower bound > 0, then always beside seed 1's interval %s)." % (
                 'allowed' if confirmation['no_collapse_prose_allowed'] else 'not allowed',
                 'allowed' if confirmation['improved_prose_allowed'] else 'not allowed', confirmation['seed1_interval_ascii']), '',
             '| training seed | source | termination | loop | submit | pooled resolved (by seed) | arm - A0mc pp [95% CI] | CI wording |',
             '|---|---|---|---|---|---|---|---|']
    for seed, source, item in (('1: v064_A_onpol', 'sealed v064a report ' + seed1['source']['report']['sha256'][:8], seed1['by_population']['primary40']),
                               ('2: ' + arm, 'this readout', reading)):
        term, repair = item['termination'], item['repair']
        lines.append('| %s | %s | %s | %s | %s | %d/%d %s | %+.2f [%+.2f, %+.2f] | %s |' % (
            seed, source, term['label'], fmt(term['loop']), fmt(term['submit']), repair['pooled_resolved'], repair['denominator'],
            repair['resolved_by_seed'], repair['effect_pp'], repair['ci95_pp'][0], repair['ci95_pp'][1], repair['ci_wording']))
    return lines + ['', 'Seed 1 is printed from its sealed report and never pooled with seed 2; no pooled row exists.', '']


def prepare(runtime_sha=None):
    """No runtime sha: the staged draft with the v064c runtime binding pending. The pinned runtime contract's sha: the pinned manifest.
    Both bind the frozen v069 readings addendum (ADDENDUM_SHA)."""
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    populations = validate_populations(readout)
    require(sha(PREREG) == PREREG_SHA, 'Frozen v064-r2 pre-registration changed')
    verbatim = addendum_readings(ADDENDUM_SHA)
    pinned = runtime_sha is not None
    if pinned:
        require(sha(RUNTIME) == runtime_sha and Path(str(RUNTIME) + '.sha256').read_text().split()[0] == runtime_sha,
                'v064c runtime sidecar mismatch')
    runtime, comparator = read_json(RUNTIME if pinned else RUNTIME_DRAFT), read_json(COMPARATOR_RUNTIME)
    require(runtime.get('generation') == 'v064c' and runtime['slot_binding']['sign'] == 'M1 - M0 = A_onpol_c2 - A0mc' and
            runtime['comparator']['runtime_sha256'] == REFERENCE_RUNTIME_SHA and
            runtime['comparator']['analysis_report_sha256'] == REFERENCE_ANALYSIS_SHA and
            runtime['comparator']['per_task_sha256'] == REFERENCE_PER_TASK_SHA and
            runtime['preregistration']['sha256'] == PREREG_SHA and comparator.get('generation') == 'v066',
            'v064c runtime contract does not bind this analysis')
    runs = slot_runs(readout)
    keys = ('name', 'scientific_name', 'result_dir', 'adapter_alias', 'decoding_seed')
    for slot, source in (('M0', comparator), ('M1', runtime)):
        require([{k: run[k] for k in keys} for run in runs if run['arm'] == slot] ==
                [{k: run[k] for k in keys} for run in source['generation_runs']], 'Slot ' + slot + ' runs differ from their generation runs')
    require(sorted(PRIMARY_IDS.read_text().split()) == sorted(populations['primary40']), 'primary40 id file differs from the readout')
    runtime_pin = runtime_sha if pinned else PENDING
    pins = {**readout['pins'], str(READOUT): READOUT_SHA, str(LEGACY): LEGACY_SHA, str(PREREG): PREREG_SHA, str(AUDIT): AUDIT_SHA,
            str(PRODUCER): PRODUCER_SHA, str(VALIDATOR): VALIDATOR_SHA, str(RESOLUTION): RESOLUTION_SHA, str(PRIMARY_IDS): PRIMARY_IDS_SHA,
            str(COMPARATOR_ANALYSIS): COMPARATOR_ANALYSIS_SHA, str(COMPARATOR_PER_TASK): COMPARATOR_PER_TASK_SHA,
            str(COMPARATOR_RUNNER): COMPARATOR_RUNNER_SHA, str(COMPARATOR_RUNTIME): COMPARATOR_RUNTIME_SHA,
            str(REFERENCE_RUNTIME): REFERENCE_RUNTIME_SHA, str(REFERENCE_PER_TASK): REFERENCE_PER_TASK_SHA,
            str(PARENT_ANALYZER): PARENT_ANALYZER_SHA, str(PARENT_ANALYSIS): PARENT_ANALYSIS_SHA,
            str(ADDENDUM): ADDENDUM_SHA, **{str(path): digest for path, digest in SEED1.values()},
            **{str(path): digest for path, digest, _ in DESCRIPTIVE_ROWS.values()},
            str(RUNNER): sha(RUNNER), str(RUNTIME): runtime_pin, str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / 'tools/test_analyze_matched_context_v064c.py'): sha(ROOT / 'tools/test_analyze_matched_context_v064c.py')}
    for path, value in pins.items():
        require(value == PENDING or sha(path) == value, 'Preparation source drift: ' + path)
    binding = {'v064c': {'source': str(RUNNER), 'source_sha256': pins[str(RUNNER)], 'runtime': str(RUNTIME), 'runtime_sha256': runtime_pin},
               'v066': {'source': str(COMPARATOR_RUNNER), 'source_sha256': COMPARATOR_RUNNER_SHA,
                        'runtime': str(COMPARATOR_RUNTIME), 'runtime_sha256': COMPARATOR_RUNTIME_SHA}}
    doc = {'version': 'v064c-matched-context-a_onpol_c2-analysis', 'generation': 'v064c',
           'status': 'PREPARED_NOT_ACTIVATED' if pinned else 'PREPARED_PENDING_PIN', 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
           'preregistration': {'path': str(PREREG), 'sha256': PREREG_SHA, 'readings': 'section Arms (Evaluation line) and section Readouts',
                               'verbatim': prereg_readings()},
           'reading_addendum': {'path': str(ADDENDUM), 'sha256': ADDENDUM_SHA, 'readings': 'section Readings (bullet lines)', 'verbatim': verbatim,
                                'sentences': {'termination': {label: [key, text] for label, (key, text, _) in TERMINATION_SENTENCES.items()},
                                              'interval_appendix': INTERVAL_APPENDIX, 'seed1_interval': SEED1_INTERVAL}},
           'seed1': {name: dict(path=str(path), sha256=digest) for name, (path, digest) in SEED1.items()},
           'evaluation_binding': binding, 'pins': pins, 'execution_authorized': False, 'output': str(OUTPUT),
           'slots': SLOTS, 'sign': SIGN, 'labels': LABELS, 'slot_runs': runs, 'thresholds': THRESHOLDS, 'descriptives': DESCRIPTIVES,
           'reading_rules': {'non_collapsed_wording': NON_COLLAPSED, 'ci_words': CI_WORDS, 'zero_ci': ZERO_CI, 'populations': list(POPULATIONS),
                             'primary': 'primary40', 'repair_minimum': 'pooled resolved >= 4/120 on primary40: the pre-specified minimum repair observation only',
                             'descriptive': 'think, repro-first and M5 (undefined stays undefined); M0qmc cited from its sealed v055 report only'},
           'descriptive_rows': {label: dict(path=str(p), sha256=d, location=w) for label, (p, d, w) in DESCRIPTIVE_ROWS.items()},
           'producers': {'mechanism_metrics': PRODUCER_SHA, 'analyze_mechanism': VALIDATOR_SHA, 'resolution_bootstrap': RESOLUTION_SHA},
           'populations': sorted(populations),
           'comparator_check': 'M0-slot producer rows must equal the v066 analysis per_task.jsonl rows of v066_A0mc_s1..s3 (144 rows, 72fc58d9; the writer fields slot, decoding_seed and trace_view are not part of the producer row); M0-slot pooled resolved in every population must equal the sealed v066 report (c448ed2e)',
           'requires': 'Qualified v064c completion (all_three_a_onpol_c2_evaluations_qualified) and explicit --run with this manifest sha.'}
    target = MANIFEST if pinned else MANIFEST_DRAFT
    write_or_verify(target, json.dumps(doc, indent=2, sort_keys=True) + chr(10))
    write_or_verify(str(target) + '.sha256', sha(target) + '  ' + target.name + chr(10))
    return sha(target)


def fmt(value):
    """Three decimals for a finite float, the integer for an int, 'undefined' for a missing or non-finite value (never 0)."""
    return ('%.3f' % value if type(value) is float else str(value)) if finite(value) else 'undefined'


def ci_line(item):
    if item.get('effect') is None:
        return 'n/a'
    ci = item.get('ci95')
    return '%+.3f [%s]' % (item['effect'], 'undefined' if ci is None else '%+.3f, %+.3f' % tuple(ci))


def report_lines(populations, inputs, readings, descriptives, rows, training, confirmation, seed1):
    arm, base, reading, p = LABELS['M1'], LABELS['M0'], readings['primary40'], populations['primary40']
    term, repair, counts = reading['termination'], reading['repair'], training['counts']
    lines = ['# Matched-context readout v064c (prereg v064-r2, readings v069, A_onpol confirmation: LoRA init seed 20261104)', '',
             'Slots frozen before reading: M0 = ' + base + ' (sealed v066 generation: E4B QAT base without adapter, matched template), M1 = ' + arm +
             ' (A_onpol confirmation adapter, second training seed, matched template); every contrast is ' + arm + ' - ' + base + '. 120 rollouts = '
             '40 tasks x 3 decoding seeds, paired by task with the seeds grouped (never 120 independent tasks); known holdout; seed 1 (the sealed '
             'v064a report) is printed beside this second training seed and never pooled with it; the two v064 arms are never pooled.', '',
             'Termination on primary40 three-seed means: **%s**%s (loop %s, submit %s; NON-COLLAPSED: loop <= 0.30 and submit >= 0.80; '
             'COLLAPSED: loop > 0.40 or submit < 0.70; otherwise INTERMEDIATE).' % (
                 term['label'], ' ("' + term['wording'] + '")' if term['wording'] else '', fmt(term['loop']), fmt(term['submit'])), '',
             'Repair: %s pooled resolved %d/%d (by seed %s); the pre-specified minimum repair observation (>= 4/120) is %s.' % (
                 arm, repair['pooled_resolved'], repair['denominator'], repair['resolved_by_seed'], 'met' if repair['minimum_observation'] else 'not met'), '',
             '%s - %s on pooled resolved, paired task bootstrap (%d draws, seed %d): %+.2f pp [%+.2f, %+.2f]: **%s**.' % (
                 arm, base, legacy.DRAWS, legacy.RNG_SEED, repair['effect_pp'], repair['ci95_pp'][0], repair['ci95_pp'][1], repair['ci_wording']),
             ZERO_CI, ''] + confirmation_lines(confirmation, seed1, reading, arm) + [
             '| population | role | termination | loop | submit | %s pooled | %s pooled | effect pp [95%% CI] | CI wording |' % (arm, base),
             '|---|---|---|---|---|---|---|---|---|']
    for name in POPULATIONS:
        item = readings[name]
        lines.append('| %s | %s | %s | %s | %s | %d/%d | %d/%d | %+.2f [%+.2f, %+.2f] | %s |' % (
            name, item['role'], item['termination']['label'], fmt(item['termination']['loop']), fmt(item['termination']['submit']),
            item['repair']['pooled_resolved'], item['repair']['denominator'], item['repair']['a0mc_pooled_resolved'], item['repair']['denominator'],
            item['repair']['effect_pp'], item['repair']['ci95_pp'][0], item['repair']['ci95_pp'][1], item['repair']['ci_wording']))
    lines += ['', 'Descriptive rows, primary40 three-seed means (think, repro-first and M5 are descriptive; M5 undefined stays undefined):', '',
              '| row | loop | submit | m5_error | think | repro-first | resolved by seed |', '|---|---|---|---|---|---|---|']
    cited = rows['v055_M0qmc']
    for label, row in ((arm + ' (slot M1)', inputs['M1']), (base + ' (slot M0)', inputs['M0']),
                       ('v055_M0qmc (cited from its sealed report ' + cited['source_sha256'][:8] + ', not a slot)', cited)):
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (label, fmt(row.get('loop')), fmt(row.get('submit')), fmt(row.get('m5_error')),
                                                             fmt(row.get('think')), fmt(row.get('repro')), row.get('solved_by_seed')))
    lines += ['', '| v042 paired effect, primary40 (descriptive) | %s | %s | effect [95%% CI] |' % (base, arm), '|---|---|---|---|']
    for key, spec in (('loop', 'M2.loop_rate'), ('submit', 'M3.submit_rate'), ('m5_error', 'M5.nonrecovery.error'),
                      ('think', 'M1.observed_thinking_step_rate'), ('repro', 'M4.opener.repro_script_opener')):
        lines.append('| %s | %s | %s | %s |' % (key, fmt(inputs['M0'].get(key)), fmt(inputs['M1'].get(key)), ci_line(p['secondary'].get(spec, {}))))
    lines += ['', '| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |', '|---|---|---|---|---|---|']
    for name, row in descriptives['per_run'].items():
        lines.append('| %s | %s | %d | %d | %d | %d |' % (name, LABELS[row['slot']], row['llm_calls'], row['compaction_calls'],
                                                       row['tasks_with_compaction'], row['content_without_reasoning_calls']))
    return lines + ['', 'Training (%s): U = %d updates; %d trajectories, %d tasks, %d sequences (duplication m = %d); adapter %s; approved '
                    'training contract %s.' % (arm, training['updates'], counts['trajectories'], counts['tasks'], counts['sequences'],
                                               counts['duplication_m'], training['adapter_sha256'][:8], training['training_contract_sha256'][:8]),
                    'v055_M0qmc is cited from its sealed report only; no contrast is computed against it. Think rate is read from returned '
                    'reasoning (trace thinking events); compaction and content-without-reasoning counts are descriptive (definitions in report.json).']


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha, 'Analysis manifest mismatch')
    doc = read_json(MANIFEST)
    require(doc['root'] == str(ROOT) and doc['output'] == str(OUTPUT) and doc['slots'] == SLOTS and doc['sign'] == SIGN and
            doc['labels'] == LABELS and doc['thresholds'] == THRESHOLDS, 'Analysis root/output/slot freeze mismatch')
    addendum = doc.get('reading_addendum')
    require(isinstance(addendum, dict) and addendum.get('path') == str(ADDENDUM) and addendum.get('sha256') == ADDENDUM_SHA and
            addendum_readings(ADDENDUM_SHA) == addendum.get('verbatim'), 'v069 readings addendum binding changed after preparation')
    readout, completions, qualified, ledger = qualify_generations(doc)
    require(readout['future_runs'] == doc['slot_runs'], 'Slot runs changed after preparation')
    ledger.pin(MANIFEST, manifest_sha)
    resolution = module(ROOT / 'tools/analyze_contract_v02_v025.py')
    validator = module(ROOT / 'tools/analyze_mechanism_v031.py')
    producer = module(ROOT / 'tools/mechanism_metrics_v032.py')
    outcomes, records, rows_by_run, tokenizer = metrics_for_runs(readout, qualified, producer, validator, ledger)
    comparator = comparator_matches_sealed(records, ledger)
    populations = {name: population_report(readout, ids, outcomes, records, rows_by_run, producer, resolution, validator)
                   for name, ids in validate_populations(readout).items()}
    comparator.update(a0mc_matches_sealed(populations, ledger))
    inputs = prediction_inputs(populations['primary40'])
    readings = arm_readings(populations)
    seed1 = seed1_readout(ledger)
    confirmation = confirmation_readings(readings, seed1)
    training = training_data(completions['v064c'], ledger)
    audit = module(ledger.pin(AUDIT, AUDIT_SHA))
    require(audit.THRESHOLD == readout['evaluation_config']['compaction']['token_threshold'] == 14336, 'Compaction threshold changed')
    descriptives = trace_descriptives(readout, audit, producer, readout['populations']['all48_sensitivity']['task_ids'])
    rows = descriptive_rows(ledger)
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_exploratory_matched_context_a_onpol_c2_readout',
              'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
              'preregistration': doc['preregistration'], 'slots': relabel(SLOTS), 'sign': SIGN, 'labels': LABELS,
              'slot_runs': relabel(readout['future_runs']), 'populations': relabel(populations),
              'readings': {'inputs_primary40_three_seed_means': relabel(inputs), 'arm': readings, 'descriptive_rows': rows,
                           'confirmation': confirmation, 'seed1': seed1}, 'reading_addendum': addendum,
              'training': training, 'trace_descriptives': relabel(descriptives), 'comparator_check': comparator, 'tokenizer': tokenizer,
              'method': readout['bootstrap'], 'limitations': readout['exposure_and_claim_limits'] + [
                  'Prereg v064-r2 confounds apply: the declared bundle; A_onpol data from one repository (fastapi); per-task cap 3; duplication m '
                  'shared by both arms; two training seeds read side by side and never pooled (seed 1 = the sealed v064a report); small n; U >= 40 is not 208; equal updates are not equal compute; generation template '
                  '(default) differs from the evaluation template (matched); known holdout. The two v064 arms are reported side by side and never '
                  'pooled; M0qmc (v055) is cited from its sealed report only; no causal claim beyond the pre-written wording. ' + ZERO_CI],
              'inputs': ledger.hashes, 'evaluation_completion': completions, 'automatic_training_or_submission': False}
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v064c analysis output required')
    OUTPUT.mkdir(exist_ok=True)
    legacy.write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    legacy.write_fresh(OUTPUT / 'per_task.jsonl', ''.join(json.dumps(
        {**row, 'slot': LABELS[key[0]], 'decoding_seed': key[1], 'trace_view': descriptives['per_task'][key[2]][key[0]][key[1]]},
        allow_nan=False) + '\n' for key, row in sorted(records.items())))
    legacy.write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(populations, inputs, readings, descriptives, rows, training, confirmation, seed1)) + '\n')
    for path in list(OUTPUT.iterdir()):
        legacy.write_fresh(str(path) + '.sha256', sha(path) + '  ' + path.name + '\n')
    primary = readings['primary40']
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json'),
            'termination': primary['termination']['label'], 'pooled_resolved': primary['repair']['pooled_resolved'],
            'ci_wording': primary['repair']['ci_wording'], 'termination_sentence': (confirmation['termination_sentence'] or {}).get('key'),
            'improved_prose_allowed': confirmation['improved_prose_allowed']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--runtime-sha', help='with --prepare: sha of the pinned v064c runtime contract (omitted: the staged draft)')
    args = ap.parse_args()
    require(not (args.prepare and args.run), 'Prepare and run are separate')
    if args.prepare:
        print(json.dumps({'status': 'prepared_only', 'manifest_sha256': prepare(args.runtime_sha), 'pinned': args.runtime_sha is not None, 'outcomes_read': False}))
    elif args.run:
        require(args.manifest_sha is not None, 'Explicit manifest SHA required')
        print(json.dumps(analyze(args.manifest_sha)))
    else:
        require(args.manifest_sha is not None and sha(MANIFEST) == args.manifest_sha, 'Inspection requires manifest SHA')
        doc = read_json(MANIFEST)
        print(json.dumps({'status': 'prepared_only', 'generation': doc['generation'], 'slots': doc['slots'], 'sign': doc['sign'],
                          'outcomes_read': False}))


if __name__ == '__main__':
    main()
