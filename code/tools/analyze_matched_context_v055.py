#!/usr/bin/env python3
"""Matched-context v055 readout: v054 M0q in slot M0, v055 M0qmc in slot M1, immutable v042 statistics (M1 - M0)."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('PROJECT_ROOT/gemma4')
LEGACY = ROOT / 'tools/analyze_mask_control_v042.py'
LEGACY_SHA = '23915fb00c2836b00fe14e3adfeecb865bdf52eb34e1fe3a04c4093cffdaf376'
if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
    raise ValueError('Immutable v042 statistical implementation changed')
_spec = importlib.util.spec_from_file_location('v055_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
for _name in ('require', 'sha', 'read_json', 'Inputs', 'module', 'validate_populations', 'metrics_for_runs',
              'population_report', 'prediction_inputs', 'resolution_report', 'ARMS', 'SEEDS', 'READOUT', 'READOUT_SHA'):
    globals()[_name] = getattr(legacy, _name)
MANIFEST = ROOT / 'contracts/matched-context-v055-analysis-DRAFT.json'
OUTPUT = ROOT / 'analysis/matched-context-v055'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-06-matched-context-M0q-v055.md'
PREREG_SHA = '82ce678734c20fb756128fb819b157c3dd61f359b8b69473c865382227ca8e04'
RUNNER = ROOT / 'tools/run_matched_context_v055.py'
RUNTIME = ROOT / 'contracts/matched-context-v055-runtime-DRAFT.json'
COMPARATOR_RUNNER = ROOT / 'tools/run_m0q_eval_v054.py'
COMPARATOR_RUNNER_SHA = 'db26da294dbf2450e0ee58078332f2febdac0891ff8358f8cdcabdd7b8052522'
COMPARATOR_RUNTIME = ROOT / 'contracts/m0q-eval-v054-runtime-DRAFT.json'
COMPARATOR_RUNTIME_SHA = '75c82f37c3afdffdd436b4e6c402727ce784224d17d88863d8bdd07689d4f1ea'
COMPARATOR_ANALYSIS = ROOT / 'analysis/m0q-eval-v054/report.json'
COMPARATOR_ANALYSIS_SHA = 'fbbffc9f22c8a0babc0378e48da67947aa89325b5d384bd7edba12bb4db92ad9'
COMPARATOR_PER_TASK = ROOT / 'analysis/m0q-eval-v054/per_task.jsonl'
COMPARATOR_PER_TASK_SHA = '79427bb2f5cee0b95858e23c2d44f0812b395e7aa0ed0e55e96e6fa27ea046f5'
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
EVIDENCE = {'v055': ROOT / 'migration/matched-context-v055/evaluation', 'v054': ROOT / 'migration/m0q-eval-v054/evaluation'}
STATUS = {'v055': 'all_three_m0qmc_evaluations_qualified', 'v054': 'all_three_m0q_evaluations_qualified'}
SLOT_GENERATION = {'M0': 'v054', 'M1': 'v055'}
LABELS = {'M0': 'v054_M0q', 'M1': 'v055_M0qmc'}
SLOTS = {'M0': 'v054_M0q_s1..s3: comparator, the M0q adapter served with the v043 template (v054 generation)',
         'M1': 'v055_M0qmc_s1..s3: same adapter, matched template 855bbbbb (v055 generation)'}
SIGN = 'M1 - M0 = v055_M0qmc - v054_M0q (v042 resolution_report and paired_slots take M1 as treated)'
THRESHOLDS = {'MCq-i': {'loop_max': .30, 'submit_min': .80, 'm5_error_max': .20},
              'MCq-ii': {'abs_delta_max': .10, 'metrics': ['loop', 'submit', 'm5_error']},
              'resolution_notable_min_pooled_primary40': 4, 'boundary_abs_tolerance': 1e-12}
DESCRIPTIVES = {
    'think_rate': 'v032 producer M1: share of agent steps with a non-blank thinking event, i.e. non-empty returned reasoning',
    'compaction_calls': 'audit.trace_items LLM calls; call k (1-based, k >= 2) is a compaction call when call k-1 prompt_tokens >= audit.THRESHOLD (14,336), the audit rule at audit.py:165-166; event availability is not re-evaluated',
    'content_without_reasoning_calls': 'trace-observable proxy for responses without <channel|>: calls that directly follow a tool response of a previous call without visible text (the matched template prefills the opener there) and carry visible text but no non-blank thought; an empty thought followed by text looks the same, and a missing <channel|> with a bare tool call leaves no trace (reasoning None and "" both drop), so this count is an upper bound for the visible kind only',
    'secondary': 'find_opener and repro_script_opener shares (M4), max repeat (M2.max_repeat), M5 silent and normal repeat rates: v042 paired effects on primary40, descriptive only',
    'descriptive_rows': 'v051 M0mc, sealed v043 M0 and A0-capped v052 primary40 three-seed means read from their pinned reports; no re-analysis and no contrast',
    'scope': 'Reported for both slots, all48, per run and per task; descriptive, no threshold'}
DESCRIPTIVE_ROWS = {
    'v051_M0mc': (ROOT / 'analysis/matched-context-v051/report.json',
                  'ea0743c9c4bf2fce04c99d2f1adb72b8ec3034ef09e7b0c3c43c98e49f48a09f', 'readings.inputs_primary40_three_seed_means.v051_M0mc'),
    'v043_M0_sealed': (ROOT / 'analysis/mask-control-v043/report.json',
                       '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2', 'prediction_inputs(populations.primary40).M0'),
    'v052_A0cap': (ROOT / 'analysis/a0-capped-v052/report.json',
                   'd5850a7e521634e5a62b9d242d1923808b40b34c80fb2be139df561573eca1fd', 'probe_a.inputs.A0cap')}


def slot_runs(readout):
    """The six runs the unchanged v042 code sees: v054 M0q seeds in slot M0, v055 M0qmc seeds in slot M1 (arm = slot)."""
    runs = []
    for slot, prefix in (('M0', 'v054_M0q_s'), ('M1', 'v055_M0qmc_s')):
        for seed in SEEDS:
            name, alias = prefix + str(seed), 'v053_M0q_s' + str(seed)
            runs.append({'name': name, 'scientific_name': alias, 'arm': slot, 'adapter_alias': alias,
                         'adapter_source': str(ROOT / 'sft/runs/A1_M0q_v053'), 'temperature': .2, 'decoding_seed': seed,
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


def mcq_labels(inputs):
    """Prereg readings on primary40 three-seed means (v042 prediction_inputs): MCq-i, MCq-ii, else INTERMEDIATE."""
    b, t = inputs['M0'], inputs['M1']
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    close = lambda x, y: math.isclose(x, y, rel_tol=0, abs_tol=THRESHOLDS['boundary_abs_tolerance'])
    le, ge = (lambda x, y: x < y or close(x, y)), (lambda x, y: x > y or close(x, y))
    keys = THRESHOLDS['MCq-ii']['metrics']
    counts = t.get('solved_by_seed')
    pooled = sum(counts) if isinstance(counts, (list, tuple)) and len(counts) == 3 and all(type(v) is int for v in counts) else None
    result = {'values': {LABELS['M0']: {k: b.get(k) for k in keys}, LABELS['M1']: {k: t.get(k) for k in keys}},
              'thresholds': THRESHOLDS, 'resolution': {'pooled_resolved': pooled, 'denominator': 120,
              'notable': pooled is not None and pooled >= THRESHOLDS['resolution_notable_min_pooled_primary40'],
              'wording': 'pooled >= 4/120 is notable and bounded, never recovery; resolution is not predicted'},
              'both_regions_rule': 'MCq-i and MCq-ii cannot both hold for the frozen comparator (v054 M0q loop 0.792 needs delta <= -0.492 '
                                   'for MCq-i); if both hold for other inputs the label is UNDEFINED'}
    if not all(finite(b.get(k)) and finite(t.get(k)) for k in keys):
        return {**result, 'label': 'UNDEFINED', 'MCq-i': None, 'MCq-ii': None, 'delta': None}
    delta = {k: t[k] - b[k] for k in keys}
    mc_i = (le(t['loop'], THRESHOLDS['MCq-i']['loop_max']) and ge(t['submit'], THRESHOLDS['MCq-i']['submit_min']) and
            le(t['m5_error'], THRESHOLDS['MCq-i']['m5_error_max']))
    mc_ii = all(le(abs(delta[k]), THRESHOLDS['MCq-ii']['abs_delta_max']) for k in keys)
    label = 'UNDEFINED' if mc_i and mc_ii else 'MCq-i' if mc_i else 'MCq-ii' if mc_ii else 'INTERMEDIATE'
    return {**result, 'label': label, 'MCq-i': mc_i, 'MCq-ii': mc_ii, 'delta': delta}


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
    """The M0 slot must reproduce the v054 analysis per-task rows of v054 M0q exactly (same producer, same traces)."""
    ledger.pin(COMPARATOR_PER_TASK, COMPARATOR_PER_TASK_SHA)
    names = {'v054_M0q_s' + str(s) for s in SEEDS}
    sealed = [json.loads(line) for line in COMPARATOR_PER_TASK.read_text().splitlines() if line.strip()]
    expected = {(row['run'], row['instance_id']): row for row in sealed if row['run'] in names}
    actual = {(row['run'], row['instance_id']): json.loads(json.dumps(row, allow_nan=False))
              for (slot, _, _), row in records.items() if slot == 'M0'}
    require(len(expected) == len(actual) == 144 and actual == expected, 'v054 M0q metrics differ from the v054 analysis rows')
    return {'rows_compared': len(actual), 'equal': True, 'sealed_per_task_sha256': COMPARATOR_PER_TASK_SHA}


def qualify_generations(manifest):
    binding = manifest.get('evaluation_binding')
    require(isinstance(binding, dict) and set(binding) == {'v055', 'v054'}, 'Evaluation binding is pending; --run refused')
    ledger = Inputs()
    for path, value in manifest['pins'].items():
        ledger.pin(path, value)
    frozen = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Original frozen readout changed')
    completions, qualified = {}, {}
    for key in ('v055', 'v054'):
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
                receipt.get('arm') == 'M0q' and receipt.get('decoding_seed') == run['decoding_seed'], 'Slot binding refused: ' + run['name'])
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
            value = prediction_inputs(doc['populations']['primary40'])['M0']
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


def prepare():
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    populations = validate_populations(readout)
    require(sha(PREREG) == PREREG_SHA, 'Frozen v055 pre-registration changed')
    runtime_sha = sha(RUNTIME)
    require(Path(str(RUNTIME) + '.sha256').read_text().split()[0] == runtime_sha, 'v055 runtime sidecar mismatch')
    runtime, comparator = read_json(RUNTIME), read_json(COMPARATOR_RUNTIME)
    require(runtime.get('generation') == 'v055' and runtime['slot_binding']['sign'] == 'M1 - M0 = M0qmc - M0q' and
            runtime['comparator']['runtime_sha256'] == COMPARATOR_RUNTIME_SHA and
            runtime['comparator']['analysis_report_sha256'] == COMPARATOR_ANALYSIS_SHA and
            runtime['comparator']['per_task_sha256'] == COMPARATOR_PER_TASK_SHA and
            runtime['preregistration']['sha256'] == PREREG_SHA and comparator.get('generation') == 'v054',
            'v055 runtime contract does not bind this analysis')
    runs = slot_runs(readout)
    keys = ('name', 'scientific_name', 'result_dir', 'adapter_alias', 'decoding_seed')
    for slot, source in (('M0', comparator), ('M1', runtime)):
        require([{k: run[k] for k in keys} for run in runs if run['arm'] == slot] ==
                [{k: run[k] for k in keys} for run in source['generation_runs']], 'Slot ' + slot + ' runs differ from their generation runs')
    require(sorted(PRIMARY_IDS.read_text().split()) == sorted(populations['primary40']), 'primary40 id file differs from the readout')
    pins = {**readout['pins'], str(READOUT): READOUT_SHA, str(LEGACY): LEGACY_SHA, str(PREREG): PREREG_SHA, str(AUDIT): AUDIT_SHA,
            str(PRODUCER): PRODUCER_SHA, str(VALIDATOR): VALIDATOR_SHA, str(RESOLUTION): RESOLUTION_SHA, str(PRIMARY_IDS): PRIMARY_IDS_SHA,
            str(COMPARATOR_ANALYSIS): COMPARATOR_ANALYSIS_SHA, str(COMPARATOR_PER_TASK): COMPARATOR_PER_TASK_SHA,
            str(COMPARATOR_RUNNER): COMPARATOR_RUNNER_SHA, str(COMPARATOR_RUNTIME): COMPARATOR_RUNTIME_SHA,
            **{str(path): digest for path, digest, _ in DESCRIPTIVE_ROWS.values()},
            str(RUNNER): sha(RUNNER), str(RUNTIME): runtime_sha, str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / 'tools/test_analyze_matched_context_v055.py'): sha(ROOT / 'tools/test_analyze_matched_context_v055.py')}
    for path, value in pins.items():
        require(sha(path) == value, 'Preparation source drift: ' + path)
    binding = {'v055': {'source': str(RUNNER), 'source_sha256': pins[str(RUNNER)], 'runtime': str(RUNTIME), 'runtime_sha256': runtime_sha},
               'v054': {'source': str(COMPARATOR_RUNNER), 'source_sha256': COMPARATOR_RUNNER_SHA,
                        'runtime': str(COMPARATOR_RUNTIME), 'runtime_sha256': COMPARATOR_RUNTIME_SHA}}
    doc = {'version': 'v055-matched-context-m0q-analysis', 'generation': 'v055', 'status': 'PREPARED_NOT_ACTIVATED', 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
           'preregistration': {'path': str(PREREG), 'sha256': PREREG_SHA, 'readings': 'section Pre-declared readings'},
           'evaluation_binding': binding, 'pins': pins, 'execution_authorized': False, 'output': str(OUTPUT),
           'slots': SLOTS, 'sign': SIGN, 'labels': LABELS, 'slot_runs': runs, 'thresholds': THRESHOLDS, 'descriptives': DESCRIPTIVES,
           'descriptive_rows': {label: dict(path=str(p), sha256=d, location=w) for label, (p, d, w) in DESCRIPTIVE_ROWS.items()},
           'producers': {'mechanism_metrics': PRODUCER_SHA, 'analyze_mechanism': VALIDATOR_SHA, 'resolution_bootstrap': RESOLUTION_SHA},
           'populations': sorted(populations),
           'comparator_check': 'M0-slot producer rows must equal the v054 analysis per_task.jsonl rows of v054_M0q_s1..s3 (144 rows, 79427bb2)',
           'requires': 'Qualified v055 completion (all_three_m0qmc_evaluations_qualified) and explicit --run with this manifest sha.'}
    write_or_verify(MANIFEST, json.dumps(doc, indent=2, sort_keys=True) + '\n')
    write_or_verify(str(MANIFEST) + '.sha256', sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    return sha(MANIFEST)


def ci_line(item):
    if item.get('effect') is None:
        return 'n/a'
    ci = item.get('ci95')
    return '%+.3f [%s]' % (item['effect'], 'undefined' if ci is None else '%+.3f, %+.3f' % tuple(ci))


def report_lines(populations, inputs, labels, descriptives, rows):
    p = populations['primary40']
    lines = ['# Matched-context readout v055 (M0q)', '',
             'Slots frozen before reading: M0 = v054_M0q (served template), M1 = v055_M0qmc (matched template); '
             'every contrast is v055_M0qmc - v054_M0q. Exploratory; known holdout; one adapter and training seed.', '',
             'Pre-declared reading on primary40 three-seed means: **%s** (MCq-i %s, MCq-ii %s).' % (labels['label'], labels['MCq-i'], labels['MCq-ii']), '',
             '| metric (primary40) | v054_M0q | v055_M0qmc | paired task-bootstrap effect [95% CI] |', '|---|---|---|---|']
    for key, spec in (('loop', 'M2.loop_rate'), ('submit', 'M3.submit_rate'), ('m5_error', 'M5.nonrecovery.error'),
                      ('think', 'M1.observed_thinking_step_rate'), ('repro', 'M4.opener.repro_script_opener')):
        item = p['secondary'].get(spec, {})
        lines.append('| %s | %s | %s | %s |' % (key, inputs['M0'].get(key), inputs['M1'].get(key), ci_line(item)))
    lines += ['', '| descriptive secondary (primary40) | paired effect [95% CI] |', '|---|---|']
    for spec in ('M4.opener.find_opener', 'M4.opener.repro_script_opener', 'M2.max_repeat', 'M5.nonrecovery.silent', 'M5.nonrecovery.normal'):
        lines.append('| %s | %s |' % (spec, ci_line(p['secondary'].get(spec, {}))))
    lines += ['', 'Resolution (v042 statistics, effect in pp):']
    for name, value in populations.items():
        r = value['primary']
        lines.append('- %s: %.3f pp, 95%% CI %s; v054_M0q %s, v055_M0qmc %s.' % (
            name, r['effect_pp'], r['ci95_pp'], r['arms']['M0']['resolved_by_seed'], r['arms']['M1']['resolved_by_seed']))
    res = labels['resolution']
    lines += ['', 'Pooled v055_M0qmc resolved on primary40: %s/120 (notable: %s; never read as recovery).' % (res['pooled_resolved'], res['notable']),
              '', '| row (primary40 three-seed means) | loop | submit | m5_error | think | repro | resolved by seed |', '|---|---|---|---|---|---|---|']
    for label, row in [('v054_M0q (slot M0)', inputs['M0']), ('v055_M0qmc (slot M1)', inputs['M1'])] + list(rows.items()):
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (label, row.get('loop'), row.get('submit'), row.get('m5_error'),
                                                             row.get('think'), row.get('repro'), row.get('solved_by_seed')))
    lines += ['', '| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |', '|---|---|---|---|---|---|']
    for name, row in descriptives['per_run'].items():
        lines.append('| %s | %s | %d | %d | %d | %d |' % (name, LABELS[row['slot']], row['llm_calls'], row['compaction_calls'],
                                                       row['tasks_with_compaction'], row['content_without_reasoning_calls']))
    return lines + ['', 'Think-rate is read from returned reasoning (trace thinking events); it is about 0 by construction in both slots.',
                    'Content-without-reasoning is a trace proxy for responses with no <channel|> (definition in report.json). '
                    'Compaction counts are reported for both slots (v054 beside v055); rows other than the two slots are descriptive only.']


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha, 'Analysis manifest mismatch')
    doc = read_json(MANIFEST)
    require(doc['root'] == str(ROOT) and doc['output'] == str(OUTPUT) and doc['slots'] == SLOTS and doc['sign'] == SIGN and
            doc['labels'] == LABELS and doc['thresholds'] == THRESHOLDS, 'Analysis root/output/slot freeze mismatch')
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
    inputs = prediction_inputs(populations['primary40'])
    labels = mcq_labels(inputs)
    audit = module(ledger.pin(AUDIT, AUDIT_SHA))
    require(audit.THRESHOLD == readout['evaluation_config']['compaction']['token_threshold'] == 14336, 'Compaction threshold changed')
    descriptives = trace_descriptives(readout, audit, producer, readout['populations']['all48_sensitivity']['task_ids'])
    rows = descriptive_rows(ledger)
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_exploratory_matched_context_m0q_readout',
              'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
              'preregistration': doc['preregistration'], 'slots': relabel(SLOTS), 'sign': SIGN, 'labels': LABELS,
              'slot_runs': relabel(readout['future_runs']), 'populations': relabel(populations),
              'readings': {'inputs_primary40_three_seed_means': relabel(inputs), 'matched_context_m0q': labels, 'descriptive_rows': rows},
              'trace_descriptives': relabel(descriptives), 'comparator_check': comparator, 'tokenizer': tokenizer,
              'method': readout['bootstrap'], 'limitations': readout['exposure_and_claim_limits'] + [
                  'Prereg limits apply: one adapter, one training seed, known holdout; MCq-i attributes behaviour to the serving form '
                  'of this pipeline only and does not establish that matched serving makes the SFT beneficial; the matched template is a '
                  'serving choice, not the correct form; the descriptive rows (v051 M0mc, sealed v043 M0, A0-capped v052) are not contrasts.'],
              'inputs': ledger.hashes, 'evaluation_completion': completions, 'automatic_training_or_submission': False}
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v055 analysis output required')
    OUTPUT.mkdir(exist_ok=True)
    legacy.write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    legacy.write_fresh(OUTPUT / 'per_task.jsonl', ''.join(json.dumps(
        {**row, 'slot': LABELS[key[0]], 'decoding_seed': key[1], 'trace_view': descriptives['per_task'][key[2]][key[0]][key[1]]},
        allow_nan=False) + '\n' for key, row in sorted(records.items())))
    legacy.write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(populations, inputs, labels, descriptives, rows)) + '\n')
    for path in list(OUTPUT.iterdir()):
        legacy.write_fresh(str(path) + '.sha256', sha(path) + '  ' + path.name + '\n')
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json'), 'label': labels['label']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--manifest-sha')
    args = ap.parse_args()
    require(not (args.prepare and args.run), 'Prepare and run are separate')
    if args.prepare:
        print(json.dumps({'status': 'prepared_only', 'manifest_sha256': prepare(), 'outcomes_read': False}))
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
