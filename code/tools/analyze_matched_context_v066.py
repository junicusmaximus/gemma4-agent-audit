#!/usr/bin/env python3
"""Matched-context v066 readout, descriptive: the A0mc base row (v066, slot M1) beside the reproduced v055 M0qmc reference (slot M0)."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('PROJECT_ROOT/gemma4')
LEGACY = ROOT / 'tools/analyze_mask_control_v042.py'
LEGACY_SHA = '23915fb00c2836b00fe14e3adfeecb865bdf52eb34e1fe3a04c4093cffdaf376'
if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
    raise ValueError('Immutable v042 statistical implementation changed')
_spec = importlib.util.spec_from_file_location('v066_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
for _name in ('require', 'sha', 'read_json', 'Inputs', 'module', 'validate_populations', 'metrics_for_runs',
              'population_report', 'prediction_inputs', 'resolution_report', 'ARMS', 'SEEDS', 'READOUT', 'READOUT_SHA'):
    globals()[_name] = getattr(legacy, _name)
# MANIFEST is the pinned analysis manifest this tool reads; the draft is the staged template with the runtime binding still pending.
MANIFEST = ROOT / 'contracts/matched-context-v066-a0mc-analysis-PINNED.json'
MANIFEST_DRAFT = ROOT / 'contracts/matched-context-v066-a0mc-analysis-DRAFT.json'
RUNTIME_DRAFT = ROOT / 'contracts/matched-context-v066-a0mc-runtime-DRAFT.json'
OUTPUT = ROOT / 'analysis/matched-context-v066'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md'
# Templates of this generation: the v062 analysis tool and its staged manifest.
PARENT_ANALYZER = ROOT / 'tools/analyze_matched_context_v062.py'
PARENT_ANALYZER_SHA = '78b96b54f3d7f648488ab20cf55260869f20776d26a81b08aedf92d830308843'
PARENT_ANALYSIS = ROOT / 'contracts/matched-context-v062-analysis-DRAFT.json'
PARENT_ANALYSIS_SHA = 'a68964bd4451b261cd01a808b581f7821e5b70d39a99d5ea89d34c2e4cb6eddf'
BASE = 'gemma-4-e4b-it-qat-w4a16-ct'
PENDING = 'PENDING_PIN'
RUNNER = ROOT / 'tools/run_matched_context_v066.py'
RUNTIME = ROOT / 'contracts/matched-context-v066-a0mc-runtime-PINNED.json'
COMPARATOR_RUNNER = ROOT / 'tools/run_matched_context_v055.py'
COMPARATOR_RUNNER_SHA = 'a24ebfdb154bc3034934d2df9b2a0123fbd038adaeaa105eba39d8d9340f76dd'
COMPARATOR_RUNTIME = ROOT / 'contracts/matched-context-v055-runtime-DRAFT.json'
COMPARATOR_RUNTIME_SHA = '776c7f7b4810437e285003492be9cc6465cc7c54eb7d9ab59a76f2317a9038d7'
COMPARATOR_ANALYSIS = ROOT / 'analysis/matched-context-v055/report.json'
COMPARATOR_ANALYSIS_SHA = '163990653084e14e8feb08c1a04b920222d4900f18f8868cdda06d47966713c4'
COMPARATOR_PER_TASK = ROOT / 'analysis/matched-context-v055/per_task.jsonl'
COMPARATOR_PER_TASK_SHA = '682e3492b1ff45bb1b48f0bf53b06d96d93dcb9780a1ffd0b8fed624aeb57b1e'
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
EVIDENCE = {'v066': ROOT / 'migration/matched-context-v066/evaluation', 'v055': ROOT / 'migration/matched-context-v055/evaluation'}
STATUS = {'v066': 'all_three_a0mc_evaluations_qualified', 'v055': 'all_three_m0qmc_evaluations_qualified'}
SLOT_GENERATION = {'M0': 'v055', 'M1': 'v066'}
SLOT_ARM = {'M0': 'M0q', 'M1': 'A0mc'}
LABELS = {'M0': 'v055_M0qmc', 'M1': 'v066_A0mc'}
SLOTS = {'M0': 'v055_M0qmc_s1..s3: the sealed v055 generation (M0q adapter, matched template 855bbbbb), reproduced row by row as the reference of the matched serving form; not a contrast',
         'M1': 'v066_A0mc_s1..s3: the E4B QAT w4a16 base with no adapter, same matched template (v066 generation)'}
SIGN = 'none: descriptive; the unchanged v042 code runs with A0mc in slot M1 and v055 M0qmc in slot M0, and no paired effect, CI or label is reported'
METRICS = ['resolved', 'loop', 'submit', 'm5_error', 'think', 'repro']
DESCRIPTIVES = {
    'think_rate': 'v032 producer M1: share of agent steps with a non-blank thinking event, i.e. non-empty returned reasoning',
    'compaction_calls': 'audit.trace_items LLM calls; call k (1-based, k >= 2) is a compaction call when call k-1 prompt_tokens >= audit.THRESHOLD (14,336), the audit rule at audit.py:165-166; event availability is not re-evaluated',
    'content_without_reasoning_calls': 'trace-observable proxy for responses without <channel|>: calls that directly follow a tool response of a previous call without visible text (the matched template prefills the opener there) and carry visible text but no non-blank thought; an empty thought followed by text looks the same, and a missing <channel|> with a bare tool call leaves no trace (reasoning None and "" both drop), so this count is an upper bound for the visible kind only',
    'secondary': 'find_opener and repro_script_opener shares (M4), max repeat (M2.max_repeat), M5 silent and normal repeat rates: v042 paired effects on primary40, descriptive only',
    'descriptive_rows': 'v051 M0mc, sealed v043 M0 and M1 (bf16 base) and A0-capped v052 primary40 three-seed means read from their pinned reports; no re-analysis, no contrast and no 2x2',
    'scope': 'Reported for both slots, all48, per run and per task; descriptive, no threshold'}
DESCRIPTIVE_ROWS = {
    'v051_M0mc': (ROOT / 'analysis/matched-context-v051/report.json',
                  'ea0743c9c4bf2fce04c99d2f1adb72b8ec3034ef09e7b0c3c43c98e49f48a09f', 'readings.inputs_primary40_three_seed_means.v051_M0mc'),
    'v043_M0_sealed': (ROOT / 'analysis/mask-control-v043/report.json',
                       '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2', 'prediction_inputs(populations.primary40).M0'),
    'v043_M1_sealed': (ROOT / 'analysis/mask-control-v043/report.json',
                       '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2', 'prediction_inputs(populations.primary40).M1'),
    'v052_A0cap': (ROOT / 'analysis/a0-capped-v052/report.json',
                   'd5850a7e521634e5a62b9d242d1923808b40b34c80fb2be139df561573eca1fd', 'probe_a.inputs.A0cap')}


def slot_runs(readout):
    """The six runs the unchanged v042 code sees: v055 M0qmc seeds in slot M0, v066 A0mc seeds in slot M1 (arm = slot)."""
    runs = []
    for slot, prefix, alias_prefix, source in (('M0', 'v055_M0qmc_s', 'v053_M0q_s', 'A1_M0q_v053'), ('M1', 'v066_A0mc_s', None, None)):
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


def seed_table(population, slot):
    """Per-seed resolved, loop, submit, M5-error, think and repro-first of one slot (v042 per-run summaries), with the v042
    three-seed means (undefined when a seed is undefined, never 0)."""
    arm = population['primary']['arms'][slot]
    seeds = [dict(decoding_seed=seed, resolved=count, denominator=arm['denominator_per_seed'], loop=row['loop_rate'],
                  submit=row['submit_rate'], m5_error=row['nonrecovery']['error']['mean_repeat_rate'],
                  m5_error_n=row['nonrecovery']['error']['n_tasks'], think=row['think_rate_mean'],
                  repro=row['opener_shares']['repro_script_opener'])
             for seed, count, row in zip(SEEDS, arm['resolved_by_seed'], population['per_run_summaries'][slot])]
    means = prediction_inputs(population)[slot]
    return dict(per_seed=seeds, three_seed_means={key: means[key] for key in METRICS[1:]},
                pooled_resolved=sum(arm['resolved_by_seed']), pooled_denominator=3 * arm['denominator_per_seed'])


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
    """The M0 slot must reproduce the v055 analysis per-task rows of v055 M0qmc exactly (same producer, same traces)."""
    ledger.pin(COMPARATOR_PER_TASK, COMPARATOR_PER_TASK_SHA)
    names = {'v055_M0qmc_s' + str(s) for s in SEEDS}
    writer_fields = ('slot', 'decoding_seed', 'trace_view')  # added by the v055 writer beside the producer row
    sealed = [json.loads(line) for line in COMPARATOR_PER_TASK.read_text().splitlines() if line.strip()]
    expected = {(row['run'], row['instance_id']): {k: v for k, v in row.items() if k not in writer_fields}
                for row in sealed if row['run'] in names}
    actual = {(row['run'], row['instance_id']): json.loads(json.dumps(row, allow_nan=False))
              for (slot, _, _), row in records.items() if slot == 'M0'}
    require(len(expected) == len(actual) == 144 and actual == expected, 'v055 M0qmc metrics differ from the v055 analysis rows')
    return {'rows_compared': len(actual), 'equal': True, 'sealed_per_task_sha256': COMPARATOR_PER_TASK_SHA}


def qualify_generations(manifest):
    binding = manifest.get('evaluation_binding')
    require(isinstance(binding, dict) and set(binding) == {'v066', 'v055'} and all(entry['runtime_sha256'] != PENDING for entry in binding.values()),
            'Evaluation binding is pending; --run refused')
    ledger = Inputs()
    for path, value in manifest['pins'].items():
        ledger.pin(path, value)
    frozen = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Original frozen readout changed')
    completions, qualified = {}, {}
    for key in ('v066', 'v055'):
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


def prepare(runtime_sha=None):
    """No runtime sha: the staged draft with the v066 runtime binding pending. The pinned runtime contract's sha: the pinned manifest."""
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    populations = validate_populations(readout)
    pinned = runtime_sha is not None
    if pinned:
        require(sha(RUNTIME) == runtime_sha and Path(str(RUNTIME) + '.sha256').read_text().split()[0] == runtime_sha,
                'v066 runtime sidecar mismatch')
    runtime, comparator = read_json(RUNTIME if pinned else RUNTIME_DRAFT), read_json(COMPARATOR_RUNTIME)
    require(runtime.get('generation') == 'v066' and runtime['slot_binding']['sign'] == 'none: descriptive A0mc row, no contrast and no PK reading' and runtime['aliases'] == [BASE] and
            runtime['comparator']['runtime_sha256'] == COMPARATOR_RUNTIME_SHA and
            runtime['comparator']['analysis_report_sha256'] == COMPARATOR_ANALYSIS_SHA and
            runtime['comparator']['per_task_sha256'] == COMPARATOR_PER_TASK_SHA and
            runtime['preregistration']['path'] == str(PREREG) and comparator.get('generation') == 'v055' and
            (not pinned or runtime['preregistration']['sha256'] == sha(PREREG)),
            'v066 runtime contract does not bind this analysis')
    runs = slot_runs(readout)
    keys = ('name', 'scientific_name', 'result_dir', 'adapter_alias', 'decoding_seed')
    for slot, source in (('M0', comparator), ('M1', runtime)):
        require([{k: run[k] for k in keys} for run in runs if run['arm'] == slot] ==
                [{k: run[k] for k in keys} for run in source['generation_runs']], 'Slot ' + slot + ' runs differ from their generation runs')
    require(sorted(PRIMARY_IDS.read_text().split()) == sorted(populations['primary40']), 'primary40 id file differs from the readout')
    runtime_pin = runtime_sha if pinned else PENDING
    prereg = dict(runtime['preregistration']) if pinned else dict(path=str(PREREG), sha256=PENDING, status=PENDING, comparator_e0=PENDING)
    pins = {**readout['pins'], str(READOUT): READOUT_SHA, str(LEGACY): LEGACY_SHA, str(PREREG): prereg['sha256'], str(AUDIT): AUDIT_SHA,
            str(PRODUCER): PRODUCER_SHA, str(VALIDATOR): VALIDATOR_SHA, str(RESOLUTION): RESOLUTION_SHA, str(PRIMARY_IDS): PRIMARY_IDS_SHA,
            str(COMPARATOR_ANALYSIS): COMPARATOR_ANALYSIS_SHA, str(COMPARATOR_PER_TASK): COMPARATOR_PER_TASK_SHA,
            str(COMPARATOR_RUNNER): COMPARATOR_RUNNER_SHA, str(COMPARATOR_RUNTIME): COMPARATOR_RUNTIME_SHA,
            str(PARENT_ANALYZER): PARENT_ANALYZER_SHA, str(PARENT_ANALYSIS): PARENT_ANALYSIS_SHA,
            **{str(path): digest for path, digest, _ in DESCRIPTIVE_ROWS.values()},
            str(RUNNER): sha(RUNNER), str(RUNTIME): runtime_pin, str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / 'tools/test_analyze_matched_context_v066.py'): sha(ROOT / 'tools/test_analyze_matched_context_v066.py')}
    for path, value in pins.items():
        require(value == PENDING or sha(path) == value, 'Preparation source drift: ' + path)
    binding = {'v066': {'source': str(RUNNER), 'source_sha256': pins[str(RUNNER)], 'runtime': str(RUNTIME), 'runtime_sha256': runtime_pin},
               'v055': {'source': str(COMPARATOR_RUNNER), 'source_sha256': COMPARATOR_RUNNER_SHA,
                        'runtime': str(COMPARATOR_RUNTIME), 'runtime_sha256': COMPARATOR_RUNTIME_SHA}}
    doc = {'version': 'v066-matched-context-a0mc-analysis', 'generation': 'v066',
           'status': 'PREPARED_NOT_ACTIVATED' if pinned else 'PREPARED_PENDING_PIN', 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
           'preregistration': dict(prereg, section='Comparator E0 - A0mc (shared with v065); descriptive, no readings'),
           'evaluation_binding': binding, 'pins': pins, 'execution_authorized': False, 'output': str(OUTPUT),
           'slots': SLOTS, 'sign': SIGN, 'labels': LABELS, 'slot_runs': runs, 'metrics': METRICS, 'descriptives': DESCRIPTIVES,
           'descriptive_rows': {label: dict(path=str(p), sha256=d, location=w) for label, (p, d, w) in DESCRIPTIVE_ROWS.items()},
           'producers': {'mechanism_metrics': PRODUCER_SHA, 'analyze_mechanism': VALIDATOR_SHA, 'resolution_bootstrap': RESOLUTION_SHA},
           'populations': sorted(populations),
           'comparator_check': 'M0-slot producer rows must equal the v055 analysis per_task.jsonl rows of v055_M0qmc_s1..s3 (144 rows, 682e3492; the writer fields slot, decoding_seed and trace_view are not part of the producer row)',
           'requires': 'Qualified v066 completion (all_three_a0mc_evaluations_qualified) and explicit --run with this manifest sha.'}
    target = MANIFEST if pinned else MANIFEST_DRAFT
    write_or_verify(target, json.dumps(doc, indent=2, sort_keys=True) + chr(10))
    write_or_verify(str(target) + '.sha256', sha(target) + '  ' + target.name + chr(10))
    return sha(target)


def cell(value):
    return 'undefined' if value is None else '%.3f' % value


def report_lines(tables, descriptives, rows):
    lines = ['# Matched-context readout v066 (A0mc, descriptive)', '',
             'A0mc = the E4B QAT w4a16 base with no adapter under the matched template 855bbbbb (v066 generation, 48 tasks x 3 decoding seeds, '
             'capped sandboxes). Descriptive base row: no PK reading, no contrast, no confidence interval. The v055_M0qmc rows are the sealed '
             'reference generation, reproduced row by row and shown for orientation only. Known holdout; decoding seeds are not training replicates.', '']
    for name, slots in tables.items():
        lines += ['## ' + name, '', '| run | resolved | loop | submit | M5-error | think | repro-first |', '|---|---|---|---|---|---|---|']
        a0 = slots[LABELS['M1']]
        for row in a0['per_seed']:
            lines.append('| %s_s%d | %d/%d | %s | %s | %s | %s | %s |' % (LABELS['M1'], row['decoding_seed'], row['resolved'], row['denominator'],
                         cell(row['loop']), cell(row['submit']), cell(row['m5_error']), cell(row['think']), cell(row['repro'])))
        for label, table in ((LABELS['M1'] + ' three-seed mean', a0), (LABELS['M0'] + ' three-seed mean (reference)', slots[LABELS['M0']])):
            means = table['three_seed_means']
            lines.append('| %s | %d/%d pooled | %s | %s | %s | %s | %s |' % (label, table['pooled_resolved'], table['pooled_denominator'],
                         cell(means['loop']), cell(means['submit']), cell(means['m5_error']), cell(means['think']), cell(means['repro'])))
        lines.append('')
    lines += ['| descriptive row (primary40 three-seed means) | loop | submit | m5_error | think | repro | resolved by seed |', '|---|---|---|---|---|---|---|']
    for label, row in rows.items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (label, row.get('loop'), row.get('submit'), row.get('m5_error'),
                                                             row.get('think'), row.get('repro'), row.get('solved_by_seed')))
    lines += ['', '| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |', '|---|---|---|---|---|---|']
    for name, row in descriptives['per_run'].items():
        lines.append('| %s | %s | %d | %d | %d | %d |' % (name, LABELS[row['slot']], row['llm_calls'], row['compaction_calls'],
                                                       row['tasks_with_compaction'], row['content_without_reasoning_calls']))
    return lines + ['', 'M5-error is undefined for a seed without error events and is never replaced by 0; a three-seed mean with an undefined seed is undefined.',
                    'Think-rate is read from returned reasoning (trace thinking events). v052 A0cap (default template) is cited as a descriptive row only; '
                    'A0mc vs A0cap is a template difference across separate runs and is never a claim.']


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha, 'Analysis manifest mismatch')
    doc = read_json(MANIFEST)
    require(doc['root'] == str(ROOT) and doc['output'] == str(OUTPUT) and doc['slots'] == SLOTS and doc['sign'] == SIGN and
            doc['labels'] == LABELS and doc['metrics'] == METRICS, 'Analysis root/output/slot freeze mismatch')
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
    tables = {name: {LABELS[slot]: seed_table(population, slot) for slot in ('M1', 'M0')} for name, population in populations.items()}
    audit = module(ledger.pin(AUDIT, AUDIT_SHA))
    require(audit.THRESHOLD == readout['evaluation_config']['compaction']['token_threshold'] == 14336, 'Compaction threshold changed')
    descriptives = trace_descriptives(readout, audit, producer, readout['populations']['all48_sensitivity']['task_ids'])
    rows = descriptive_rows(ledger)
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_descriptive_matched_context_a0mc_readout',
              'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
              'preregistration': doc['preregistration'], 'slots': relabel(SLOTS), 'sign': SIGN, 'labels': LABELS,
              'slot_runs': relabel(readout['future_runs']), 'populations': {name: relabel({'per_run_summaries': p['per_run_summaries'],
              'resolved_by_seed': {a: p['primary']['arms'][a]['resolved_by_seed'] for a in ARMS}}) for name, p in populations.items()},
              'readings': {'descriptive_by_population': tables, 'descriptive_rows': rows},
              'trace_descriptives': relabel(descriptives), 'comparator_check': comparator, 'tokenizer': tokenizer,
              'method': 'v042 producer per-run summaries and v042 three-seed means (mean_three); no bootstrap, paired effect, CI or label is reported', 'limitations': readout['exposure_and_claim_limits'] + [
                  'A0mc is a descriptive base row (prereg v064-r2, Comparator E0): no PK reading, no contrast; the v055 M0qmc rows are the reproduced '
                  'reference, not a comparator; A0mc vs A0cap (v052, default template) is a template difference across separate runs and never a claim; '
                  'the descriptive rows (v051 M0mc, sealed v043 M0 and M1, A0-capped v052) are not contrasts and form no 2x2; known holdout.'],
              'inputs': ledger.hashes, 'evaluation_completion': completions, 'automatic_training_or_submission': False}
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v066 analysis output required')
    OUTPUT.mkdir(exist_ok=True)
    legacy.write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    legacy.write_fresh(OUTPUT / 'per_task.jsonl', ''.join(json.dumps(
        {**row, 'slot': LABELS[key[0]], 'decoding_seed': key[1], 'trace_view': descriptives['per_task'][key[2]][key[0]][key[1]]},
        allow_nan=False) + '\n' for key, row in sorted(records.items())))
    legacy.write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(tables, descriptives, rows)) + '\n')
    for path in list(OUTPUT.iterdir()):
        legacy.write_fresh(str(path) + '.sha256', sha(path) + '  ' + path.name + '\n')
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json'),
            'a0mc_primary40_three_seed_means': tables['primary40'][LABELS['M1']]['three_seed_means']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--runtime-sha', help='with --prepare: sha of the pinned v066 runtime contract (omitted: the staged draft)')
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
