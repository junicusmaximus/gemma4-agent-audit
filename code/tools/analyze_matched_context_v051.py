#!/usr/bin/env python3
"""Matched-context v051 readout: v043 M0 in slot M0, v051 M0mc in slot M1, immutable v042 statistics (M1 - M0)."""
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
_spec = importlib.util.spec_from_file_location('v051_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
for _name in ('require', 'sha', 'read_json', 'Inputs', 'module', 'validate_populations', 'metrics_for_runs',
              'population_report', 'prediction_inputs', 'resolution_report', 'ARMS', 'SEEDS', 'READOUT', 'READOUT_SHA'):
    globals()[_name] = getattr(legacy, _name)
MANIFEST = ROOT / 'contracts/matched-context-v051-analysis-DRAFT.json'
OUTPUT = ROOT / 'analysis/matched-context-v051'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-06-matched-context-eval.md'
PREREG_SHA = '86bc2ebbecc3a60363b8eed0b05a6575f48d1187fc8a1191fb2b301d918d6507'
RUNNER = ROOT / 'tools/run_matched_context_v051.py'
RUNTIME = ROOT / 'contracts/matched-context-v051-runtime-DRAFT.json'
COMPARATOR_RUNNER = ROOT / 'tools/run_mask_evaluations_v043.py'
COMPARATOR_RUNNER_SHA = '9b39a600adb998117a694df5dfd61514f03e29b467795b58bc449b495f8ca590'
COMPARATOR_RUNTIME = ROOT / 'contracts/mask-evaluations-v043-runtime-DRAFT.json'
COMPARATOR_RUNTIME_SHA = '7eaf5d53b56e02d6e45ef29a229748b2716bdee36ae7b4ebf0301450d4c57a63'
COMPARATOR_ANALYSIS = ROOT / 'analysis/mask-control-v043/report.json'
COMPARATOR_ANALYSIS_SHA = '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2'
COMPARATOR_PER_TASK = ROOT / 'analysis/mask-control-v043/per_task.jsonl'
COMPARATOR_PER_TASK_SHA = 'c4e51b4819351823f85ced2cf07c7f9132b0b53bdbb1b8c7084eef41f42b6a42'
AUDIT = ROOT / 'analysis/serialization-audit-boss-v046/audit.py'
AUDIT_SHA = '5d26f4cdc9aafa2e9ffc1f527fd88d1ccd58497770f6b4a626b04ef5078cee98'
EVIDENCE = {'v051': ROOT / 'migration/matched-context-v051/evaluation', 'v043': ROOT / 'migration/mask-evaluations-v043/evaluation'}
STATUS = {'v051': 'all_three_evaluations_qualified', 'v043': 'all_six_evaluations_qualified'}
SLOT_GENERATION = {'M0': 'v043', 'M1': 'v051'}
LABELS = {'M0': 'v043_M0', 'M1': 'v051_M0mc'}
SLOTS = {'M0': 'v043_M0_s1..s3: sealed comparator, served template (v043 generation)',
         'M1': 'v051_M0mc_s1..s3: same adapter, matched template 855bbbbb (v051 generation)'}
SIGN = 'M1 - M0 = v051_M0mc - v043_M0 (v042 resolution_report and paired_slots take M1 as treated)'
THRESHOLDS = {'MC-i': {'loop_max': .30, 'submit_min': .80, 'm5_error_max': .20},
              'MC-ii': {'abs_delta_max': .10, 'metrics': ['loop', 'submit', 'm5_error']},
              'resolution_notable_min_pooled_primary40': 4, 'boundary_abs_tolerance': 1e-12}
DESCRIPTIVES = {
    'think_rate': 'v032 producer M1: share of agent steps with a non-blank thinking event, i.e. non-empty returned reasoning',
    'compaction_calls': 'audit.trace_items LLM calls; call k (1-based, k >= 2) is a compaction call when call k-1 prompt_tokens >= audit.THRESHOLD (14,336), the audit rule at audit.py:165-166; event availability is not re-evaluated',
    'content_without_reasoning_calls': 'trace-observable proxy for responses without <channel|>: calls that directly follow a tool response of a previous call without visible text (the matched template prefills the opener there) and carry visible text but no non-blank thought; an empty thought followed by text looks the same, and a missing <channel|> with a bare tool call leaves no trace (reasoning None and "" both drop), so this count is an upper bound for the visible kind only',
    'scope': 'Reported for both slots, all48, per run and per task; descriptive, no threshold'}


def slot_runs(readout):
    """The six runs the unchanged v042 code sees: v043 M0 seeds in slot M0, v051 M0mc seeds in slot M1 (arm = slot)."""
    runs = []
    for slot, prefix in (('M0', 'v043_M0_s'), ('M1', 'v051_M0mc_s')):
        for original in [run for run in readout['future_runs'] if run['arm'] == 'M0']:
            name = prefix + str(original['decoding_seed'])
            runs.append({**copy.deepcopy(original), 'arm': slot, 'scientific_name': original['name'], 'name': name,
                         'submission_dir': str(ROOT / 'subs' / name), 'result_dir': str(ROOT / 'results' / name),
                         'trace_dir': str(ROOT / 'results' / name / 'traces')})
    require([(run['arm'], run['decoding_seed']) for run in runs] == [(a, s) for a in ARMS for s in SEEDS], 'Slot grid changed')
    return runs


def effective_readout(readout):
    effective = copy.deepcopy(readout)
    effective['future_runs'] = slot_runs(readout)
    effective['future_analysis_dir'] = str(OUTPUT)
    return effective


def mc_labels(inputs):
    """Prereg readings on primary40 three-seed means (v042 prediction_inputs): MC-i, MC-ii, else INTERMEDIATE."""
    b, t = inputs['M0'], inputs['M1']
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    close = lambda x, y: math.isclose(x, y, rel_tol=0, abs_tol=THRESHOLDS['boundary_abs_tolerance'])
    le, ge = (lambda x, y: x < y or close(x, y)), (lambda x, y: x > y or close(x, y))
    keys = THRESHOLDS['MC-ii']['metrics']
    counts = t.get('solved_by_seed')
    pooled = sum(counts) if isinstance(counts, (list, tuple)) and len(counts) == 3 and all(type(v) is int for v in counts) else None
    result = {'values': {LABELS['M0']: {k: b.get(k) for k in keys}, LABELS['M1']: {k: t.get(k) for k in keys}},
              'thresholds': THRESHOLDS, 'resolution': {'pooled_resolved': pooled, 'denominator': 120,
              'notable': pooled is not None and pooled >= THRESHOLDS['resolution_notable_min_pooled_primary40'],
              'wording': 'pooled >= 4/120 is notable and bounded, never recovery; resolution is not predicted'},
              'both_regions_rule': 'MC-i and MC-ii cannot both hold for the frozen comparator (loop 0.617 needs delta <= -0.317 '
                                   'for MC-i); if both hold for other inputs the label is UNDEFINED'}
    if not all(finite(b.get(k)) and finite(t.get(k)) for k in keys):
        return {**result, 'label': 'UNDEFINED', 'MC-i': None, 'MC-ii': None, 'delta': None}
    delta = {k: t[k] - b[k] for k in keys}
    mc_i = (le(t['loop'], THRESHOLDS['MC-i']['loop_max']) and ge(t['submit'], THRESHOLDS['MC-i']['submit_min']) and
            le(t['m5_error'], THRESHOLDS['MC-i']['m5_error_max']))
    mc_ii = all(le(abs(delta[k]), THRESHOLDS['MC-ii']['abs_delta_max']) for k in keys)
    label = 'UNDEFINED' if mc_i and mc_ii else 'MC-i' if mc_i else 'MC-ii' if mc_ii else 'INTERMEDIATE'
    return {**result, 'label': label, 'MC-i': mc_i, 'MC-ii': mc_ii, 'delta': delta}


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
    """The M0 slot must reproduce the sealed v044 per-task rows of v043 M0 exactly (same producer, same traces)."""
    ledger.pin(COMPARATOR_PER_TASK, COMPARATOR_PER_TASK_SHA)
    names = {'v043_M0_s' + str(s) for s in SEEDS}
    sealed = [json.loads(line) for line in COMPARATOR_PER_TASK.read_text().splitlines() if line.strip()]
    expected = {(row['run'], row['instance_id']): row for row in sealed if row['run'] in names}
    actual = {(row['run'], row['instance_id']): json.loads(json.dumps(row, allow_nan=False))
              for (slot, _, _), row in records.items() if slot == 'M0'}
    require(len(expected) == len(actual) == 144 and actual == expected, 'v043 M0 metrics differ from the sealed v044 rows')
    return {'rows_compared': len(actual), 'equal': True, 'sealed_per_task_sha256': COMPARATOR_PER_TASK_SHA}


def qualify_generations(manifest):
    binding = manifest.get('evaluation_binding')
    require(isinstance(binding, dict) and set(binding) == {'v051', 'v043'}, 'Evaluation binding is pending; --run refused')
    ledger = Inputs()
    for path, value in manifest['pins'].items():
        ledger.pin(path, value)
    frozen = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Original frozen readout changed')
    completions, qualified = {}, {}
    for key in ('v051', 'v043'):
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
                receipt.get('arm') == 'M0' and receipt.get('decoding_seed') == run['decoding_seed'], 'Slot binding refused: ' + run['name'])
        ledger.pin(receipt['memory_guard']['path'], receipt['memory_guard']['sha256'])
        runs[run['name']] = qualified[key][run['name']]
    for folder in EVIDENCE.values():
        for path in sorted(folder.rglob('*.json')):
            ledger.pin(path)
    ledger.verify()
    return effective, completions, runs, ledger


def write_or_verify(path, text):
    if Path(path).exists():
        require(Path(path).read_text() == text, 'Analysis preparation is not reproducible: ' + str(path))
        return
    with Path(path).open('x') as stream:
        stream.write(text)


def prepare():
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    validate_populations(readout)
    require(sha(PREREG) == PREREG_SHA, 'Frozen matched-context pre-registration changed')
    runtime_sha = sha(RUNTIME)
    require(Path(str(RUNTIME) + '.sha256').read_text().split()[0] == runtime_sha, 'v051 runtime sidecar mismatch')
    runtime = read_json(RUNTIME)
    require(runtime.get('generation') == 'v051' and runtime['slot_binding']['sign'] == 'M1 - M0 = M0mc - M0' and
            runtime['comparator']['runtime_sha256'] == COMPARATOR_RUNTIME_SHA and
            runtime['comparator']['analysis_report_sha256'] == COMPARATOR_ANALYSIS_SHA and
            runtime['preregistration']['sha256'] == PREREG_SHA, 'v051 runtime contract does not bind this analysis')
    runs = slot_runs(readout)
    require([{k: run[k] for k in ('name', 'result_dir', 'adapter_alias', 'decoding_seed')} for run in runs if run['arm'] == 'M1'] ==
            [{k: run[k] for k in ('name', 'result_dir', 'adapter_alias', 'decoding_seed')} for run in runtime['generation_runs']],
            'Slot M1 runs differ from the v051 generation runs')
    pins = {**readout['pins'], str(READOUT): READOUT_SHA, str(LEGACY): LEGACY_SHA, str(PREREG): PREREG_SHA,
            str(AUDIT): AUDIT_SHA, str(COMPARATOR_ANALYSIS): COMPARATOR_ANALYSIS_SHA, str(COMPARATOR_PER_TASK): COMPARATOR_PER_TASK_SHA,
            str(COMPARATOR_RUNNER): COMPARATOR_RUNNER_SHA, str(COMPARATOR_RUNTIME): COMPARATOR_RUNTIME_SHA,
            str(RUNNER): sha(RUNNER), str(RUNTIME): runtime_sha, str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / 'tools/test_analyze_matched_context_v051.py'): sha(ROOT / 'tools/test_analyze_matched_context_v051.py')}
    for path, value in pins.items():
        require(sha(path) == value, 'Preparation source drift: ' + path)
    binding = {'v051': {'source': str(RUNNER), 'source_sha256': pins[str(RUNNER)], 'runtime': str(RUNTIME), 'runtime_sha256': runtime_sha},
               'v043': {'source': str(COMPARATOR_RUNNER), 'source_sha256': COMPARATOR_RUNNER_SHA,
                        'runtime': str(COMPARATOR_RUNTIME), 'runtime_sha256': COMPARATOR_RUNTIME_SHA}}
    doc = {'version': 'v051-matched-context-analysis', 'generation': 'v051', 'status': 'PREPARED_NOT_ACTIVATED', 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
           'preregistration': {'path': str(PREREG), 'sha256': PREREG_SHA, 'readings': 'lines 20-25'},
           'evaluation_binding': binding, 'pins': pins, 'execution_authorized': False, 'output': str(OUTPUT),
           'slots': SLOTS, 'sign': SIGN, 'labels': LABELS, 'slot_runs': runs, 'thresholds': THRESHOLDS, 'descriptives': DESCRIPTIVES,
           'comparator_check': 'M0-slot producer rows must equal the sealed v044 per_task.jsonl rows of v043_M0_s1..s3',
           'requires': 'Qualified v051 completion (all_three_evaluations_qualified) and explicit --run with this manifest sha.'}
    write_or_verify(MANIFEST, json.dumps(doc, indent=2, sort_keys=True) + '\n')
    write_or_verify(str(MANIFEST) + '.sha256', sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    return sha(MANIFEST)


def ci_line(item):
    if item.get('effect') is None:
        return 'n/a'
    ci = item.get('ci95')
    return '%+.3f [%s]' % (item['effect'], 'undefined' if ci is None else '%+.3f, %+.3f' % tuple(ci))


def report_lines(populations, inputs, labels, descriptives):
    p = populations['primary40']
    lines = ['# Matched-context readout v051', '',
             'Slots frozen before reading: M0 = v043_M0 (served template), M1 = v051_M0mc (matched template); '
             'every contrast is v051_M0mc - v043_M0. Exploratory; known holdout; one adapter and training seed.', '',
             'Pre-declared reading on primary40 three-seed means: **%s** (MC-i %s, MC-ii %s).' % (labels['label'], labels['MC-i'], labels['MC-ii']), '',
             '| metric (primary40) | v043_M0 | v051_M0mc | paired task-bootstrap effect [95% CI] |', '|---|---|---|---|']
    for key, spec in (('loop', 'M2.loop_rate'), ('submit', 'M3.submit_rate'), ('m5_error', 'M5.nonrecovery.error'),
                      ('think', 'M1.observed_thinking_step_rate'), ('repro', 'M4.opener.repro_script_opener')):
        item = p['secondary'].get(spec, {})
        lines.append('| %s | %s | %s | %s |' % (key, inputs['M0'].get(key), inputs['M1'].get(key), ci_line(item)))
    lines += ['', 'Resolution (v042 statistics, effect in pp):']
    for name, value in populations.items():
        r = value['primary']
        lines.append('- %s: %.3f pp, 95%% CI %s; v043_M0 %s, v051_M0mc %s.' % (
            name, r['effect_pp'], r['ci95_pp'], r['arms']['M0']['resolved_by_seed'], r['arms']['M1']['resolved_by_seed']))
    res = labels['resolution']
    lines += ['', 'Pooled v051_M0mc resolved on primary40: %s/120 (notable: %s; never read as recovery).' % (res['pooled_resolved'], res['notable']),
              '', '| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |', '|---|---|---|---|---|---|']
    for name, row in descriptives['per_run'].items():
        lines.append('| %s | %s | %d | %d | %d | %d |' % (name, LABELS[row['slot']], row['llm_calls'], row['compaction_calls'],
                                                       row['tasks_with_compaction'], row['content_without_reasoning_calls']))
    return lines + ['', 'Think-rate is read from returned reasoning (trace thinking events) and is reported, not read, for M0.',
                    'Content-without-reasoning is a trace proxy (definition in report.json). Compaction fires earlier under the '
                    'matched template because prompts are longer; counts are descriptive.']


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
    labels = mc_labels(inputs)
    audit = module(ledger.pin(AUDIT, AUDIT_SHA))
    require(audit.THRESHOLD == readout['evaluation_config']['compaction']['token_threshold'] == 14336, 'Compaction threshold changed')
    descriptives = trace_descriptives(readout, audit, producer, readout['populations']['all48_sensitivity']['task_ids'])
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_exploratory_matched_context_readout',
              'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
              'preregistration': doc['preregistration'], 'slots': relabel(SLOTS), 'sign': SIGN, 'labels': LABELS,
              'slot_runs': relabel(readout['future_runs']), 'populations': relabel(populations),
              'readings': {'inputs_primary40_three_seed_means': relabel(inputs), 'matched_context': labels},
              'trace_descriptives': relabel(descriptives), 'comparator_check': comparator, 'tokenizer': tokenizer,
              'method': readout['bootstrap'], 'limitations': readout['exposure_and_claim_limits'] + [
                  'Prereg interpretation limits apply (lines 24-25): one adapter, one training seed, known holdout; MC-i attributes '
                  'behaviour to the serving form of this pipeline only; compaction fires earlier under the matched template.'],
              'inputs': ledger.hashes, 'evaluation_completion': completions, 'automatic_training_or_submission': False}
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v051 analysis output required')
    OUTPUT.mkdir(exist_ok=True)
    legacy.write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    legacy.write_fresh(OUTPUT / 'per_task.jsonl', ''.join(json.dumps(
        {**row, 'slot': LABELS[key[0]], 'decoding_seed': key[1], 'trace_view': descriptives['per_task'][key[2]][key[0]][key[1]]},
        allow_nan=False) + '\n' for key, row in sorted(records.items())))
    legacy.write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(populations, inputs, labels, descriptives)) + '\n')
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
