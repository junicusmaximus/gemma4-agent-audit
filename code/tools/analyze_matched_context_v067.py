#!/usr/bin/env python3
"""Matched-context v067 readout (prereg v067 part B): v055 M0qmc in slot M0, v067 M1qs2mc (M1q-s2) in slot M1, immutable v042
statistics (M1 - M0, descriptive); reading (3) M1qs2mc - A0mc by the part (A) rule; M1q (v062) beside M1q-s2, never pooled."""
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
_spec = importlib.util.spec_from_file_location('v067_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
for _name in ('require', 'sha', 'read_json', 'Inputs', 'module', 'validate_populations', 'metrics_for_runs',
              'population_report', 'prediction_inputs', 'resolution_report', 'ARMS', 'SEEDS', 'READOUT', 'READOUT_SHA'):
    globals()[_name] = getattr(legacy, _name)
# MANIFEST is the pinned analysis manifest this tool reads; the draft is the staged template with the runtime binding still pending.
MANIFEST = ROOT / 'contracts/matched-context-v067-analysis-PINNED.json'
MANIFEST_DRAFT = ROOT / 'contracts/matched-context-v067-analysis-DRAFT.json'
RUNTIME_DRAFT = ROOT / 'contracts/matched-context-v067-runtime-DRAFT.json'
OUTPUT = ROOT / 'analysis/matched-context-v067'
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-m1q-seed2-and-a0mc-branches-v067.md'
PREREG_SHA = '766c1e45386f93121fd29e8beca7de30526b9bb8f52482644608e49d17ce2b46'
PART_A = '## (A) M1q ' + chr(0x2212) + ' A0mc reading (before the A0mc report is read)'
PART_B = '## (B) Second training seed of M1q (M1q-s2)'
# Templates of this generation (prereg v067 part B: the v062 machinery re-pinned): the v062 analysis tool and its pinned manifest.
PARENT_ANALYZER = ROOT / 'tools/analyze_matched_context_v062.py'
PARENT_ANALYZER_SHA = '78b96b54f3d7f648488ab20cf55260869f20776d26a81b08aedf92d830308843'
PARENT_ANALYSIS = ROOT / 'contracts/matched-context-v062-analysis-PINNED.json'
PARENT_ANALYSIS_SHA = '91295f7cde58104928f6d72aef9f4124a2d2413323edb9005f007d0d0af27a0f'
PENDING = 'PENDING_PIN'
RUNNER = ROOT / 'tools/run_matched_context_v067.py'
RUNTIME = ROOT / 'contracts/matched-context-v067-runtime-PINNED.json'
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
EVIDENCE = {'v067': ROOT / 'migration/matched-context-v067/evaluation', 'v055': ROOT / 'migration/matched-context-v055/evaluation'}
STATUS = {'v067': 'all_three_m1qs2mc_evaluations_qualified', 'v055': 'all_three_m0qmc_evaluations_qualified'}
SLOT_GENERATION = {'M0': 'v055', 'M1': 'v067'}
SLOT_ARM = {'M0': 'M0q', 'M1': 'M1qs2'}
LABELS = {'M0': 'v055_M0qmc', 'M1': 'v067_M1qs2mc'}
SLOTS = {'M0': 'v055_M0qmc_s1..s3: comparator, the M0q adapter under the matched template 855bbbbb (v055 generation)',
         'M1': 'v067_M1qs2mc_s1..s3: the M1q-s2 adapter (second training seed of the M1q recipe, LoRA init seed 20261104), same matched template (v067 generation)'}
SIGN = 'M1 - M0 = v067_M1qs2mc - v055_M0qmc (v042 resolution_report and paired_slots take M1 as treated)'
THRESHOLDS = {'resolution_notable_min_pooled_primary40': 4, 'boundary_abs_tolerance': 1e-12,
              'a0mc_rule': 'primary40 CI of M1qs2mc - A0mc: lower > 0, upper < 0, otherwise (prereg v067 part A rule, applied by part B reading 3)',
              'sentence_B': 'pooled M1q-s2 resolved on primary40: >= 4, 1 to 3, or 0 (prereg v067 part B)'}
SEED1_POOLED = 10  # M1q (v062, seed 20261103) pooled primary40 resolved: read from its sealed report and required equal
SENTENCES_B = {'notable': 'observed in two training seeds (10/120 and x/120) on the known holdout',
               'below_notable': 'below the notable threshold in the second seed (x/120)',
               'zero': 'seed-dependent: one of two training seeds (10/120 and 0/120)'}
SENTENCES_A = {'lower_bound_above_zero': 'the masked arm on the QAT base resolved more tasks than the same-template base (x/120 vs y/120; +d pp [l, u]) '
                                         + chr(0x2014) + ' one training seed, known holdout',
               'upper_bound_below_zero': 'the masked arm resolved fewer tasks than the same-template base (x/120 vs y/120; d pp [l, u])',
               'otherwise': 'relative to the same-template base the difference is uncertain (x/120 vs y/120; d pp [l, u])'}
NEVER = ('recovery', 'partial recovery', 'closed the gap', 'masking restores / causes repair', 'no-collapse')
POPULATION_NAMES = ('primary40', 'all48_sensitivity', 'control_clean36_sensitivity')
# Sealed generations read by reading (3) and the side-by-side row; each re-qualified by its own runner's CPU-only interface.
A0MC = dict(generation='v066', arm='A0mc', run_prefix='v066_A0mc_s', status='all_three_a0mc_evaluations_qualified', label='v066_A0mc',
            runner=ROOT / 'tools/run_matched_context_v066.py', runner_sha='0542edbf7d8e3ee57c605b75e15f0d8bd9a090d9fddd9d37fff7abc0c2f72e8a',
            runtime=ROOT / 'contracts/matched-context-v066-a0mc-runtime-PINNED.json', runtime_sha='0f5e04b257ffe1544633f90fba4e51e377dae82a82f97bb029d1b8c3d49944e5',
            completed=ROOT / 'migration/matched-context-v066/evaluation/completed.json', completed_sha='fb1bf16f96bbb5002d77e0bda8b4a81a5a9c81a40bdd95b6616e72009f4f7500',
            report=ROOT / 'analysis/matched-context-v066/report.json', report_sha='c448ed2e31ff77a93a5b9094dfbd33c08e21d3da1a03de66bf60b6295def5f8e')
SEED1 = dict(generation='v062', arm='M1q', run_prefix='v062_M1qmc_s', status='all_three_m1qmc_evaluations_qualified', label='v062_M1qmc',
             runner=ROOT / 'tools/run_matched_context_v062.py', runner_sha='621cd9363e0523ed6391705c9329d003213c09f34eb8d3ff3ab78047d059da2b',
             runtime=ROOT / 'contracts/matched-context-v062-runtime-PINNED.json', runtime_sha='3717c8d036ae17cf2f4713458b5fe31a36cff95903dadf4e9175eb2efee9aefe',
             completed=ROOT / 'migration/matched-context-v062/evaluation/completed.json', completed_sha='5a41099813ddf91985ac21bc16de560cf9b49dc7903d3ce6ef595938d9dbb677',
             report=ROOT / 'analysis/matched-context-v062/report.json', report_sha='70f20fc0055cedbb0a4a0c1e2cc82dd1b3afa2d3430084b83e3cac664713ba1d')
V067A = dict(report=ROOT / 'analysis/m1q-vs-a0mc-v067/report.json', report_sha='a62b7f90b2720996aec38aedabfade765d12f28823fdd35be5dccff1b5b457c9',
             analyzer=ROOT / 'tools/analyze_m1q_vs_a0mc_v067.py', analyzer_sha='911e4ab9c959baaa2c0c6ff0acbb147a13f2c16873d1232124499bd0488e4e62')
DESCRIPTIVES = {
    'think_rate': 'v032 producer M1: share of agent steps with a non-blank thinking event, i.e. non-empty returned reasoning',
    'compaction_calls': 'audit.trace_items LLM calls; call k (1-based, k >= 2) is a compaction call when call k-1 prompt_tokens >= audit.THRESHOLD (14,336), the audit rule at audit.py:165-166; event availability is not re-evaluated',
    'content_without_reasoning_calls': 'trace-observable proxy for responses without <channel|>: calls that directly follow a tool response of a previous call without visible text (the matched template prefills the opener there) and carry visible text but no non-blank thought; an empty thought followed by text looks the same, and a missing <channel|> with a bare tool call leaves no trace (reasoning None and "" both drop), so this count is an upper bound for the visible kind only',
    'secondary': 'find_opener and repro_script_opener shares (M4), max repeat (M2.max_repeat), M5 silent and normal repeat rates: v042 paired effects on primary40, descriptive only',
    'descriptive_rows': 'v051 M0mc, sealed v043 M0 and M1 (bf16 base), A0-capped v052, M1q (v062, seed 1) and A0mc (v066) primary40 three-seed means read from their pinned reports; no re-analysis, no contrast and no 2x2',
    'scope': 'Reported for both slots, all48, per run and per task; descriptive, no threshold'}
DESCRIPTIVE_ROWS = {
    'v051_M0mc': (ROOT / 'analysis/matched-context-v051/report.json',
                  'ea0743c9c4bf2fce04c99d2f1adb72b8ec3034ef09e7b0c3c43c98e49f48a09f', 'readings.inputs_primary40_three_seed_means.v051_M0mc'),
    'v043_M0_sealed': (ROOT / 'analysis/mask-control-v043/report.json',
                       '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2', 'prediction_inputs(populations.primary40).M0'),
    'v043_M1_sealed': (ROOT / 'analysis/mask-control-v043/report.json',
                       '1b8d360b9ec246831deb04b46d8886fc69ca764f3791b229aaffee5e988256c2', 'prediction_inputs(populations.primary40).M1'),
    'v052_A0cap': (ROOT / 'analysis/a0-capped-v052/report.json',
                   'd5850a7e521634e5a62b9d242d1923808b40b34c80fb2be139df561573eca1fd', 'probe_a.inputs.A0cap'),
    'v062_M1qmc': (ROOT / 'analysis/matched-context-v062/report.json',
                   '70f20fc0055cedbb0a4a0c1e2cc82dd1b3afa2d3430084b83e3cac664713ba1d', 'readings.inputs_primary40_three_seed_means.v062_M1qmc'),
    'v066_A0mc': (ROOT / 'analysis/matched-context-v066/report.json',
                  'c448ed2e31ff77a93a5b9094dfbd33c08e21d3da1a03de66bf60b6295def5f8e', 'readings.descriptive_by_population.primary40.v066_A0mc.three_seed_means')}


def slot_runs(readout):
    """The six runs the unchanged v042 code sees: v055 M0qmc seeds in slot M0, v067 M1qs2mc seeds in slot M1 (arm = slot)."""
    runs = []
    for slot, prefix, alias_prefix, source in (('M0', 'v055_M0qmc_s', 'v053_M0q_s', 'A1_M0q_v053'), ('M1', 'v067_M1qs2mc_s', 'v067_M1qs2_s', 'A1_M1q_s2_v067')):
        for seed in SEEDS:
            name, alias = prefix + str(seed), alias_prefix + str(seed)
            runs.append({'name': name, 'scientific_name': alias, 'arm': slot, 'adapter_alias': alias,
                         'adapter_source': str(ROOT / 'sft/runs' / source), 'temperature': .2, 'decoding_seed': seed,
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


def part_b_readings(inputs):
    """Prereg v067 part B readings (1) and (2) on primary40 three-seed means (v042 prediction_inputs), descriptive: termination metrics and
    think rate vs M0qmc with their differences and no label, and the pooled resolution with the notable threshold (as v067)."""
    b, t = inputs['M0'], inputs['M1']
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    keys = ('loop', 'submit', 'm5_error', 'think', 'repro')
    counts = t.get('solved_by_seed')
    pooled = sum(counts) if isinstance(counts, (list, tuple)) and len(counts) == 3 and all(type(v) is int for v in counts) else None
    m5 = finite(b.get('m5_error')) and finite(t.get('m5_error'))
    return {'values': {LABELS['M0']: {k: b.get(k) for k in keys}, LABELS['M1']: {k: t.get(k) for k in keys}},
            'delta_m1_minus_m0': {k: (t[k] - b[k] if finite(b.get(k)) and finite(t.get(k)) else None) for k in keys},
            'm5_error': 'read' if m5 else 'undefined in an arm: reported as undefined, never substituted by 0',
            'wording': 'descriptive (prereg v067 part B reading 2): no PK label, no think label, no threshold',
            'resolution': {'pooled_resolved': pooled, 'by_seed': counts, 'denominator': 120,
                           'notable': pooled is not None and pooled >= THRESHOLDS['resolution_notable_min_pooled_primary40'],
                           'wording': 'reading 1: pooled >= 4/120 is notable (as v062); both training seeds side by side, never pooled; '
                                      'the independent-repository requirement of v062 is not lifted'}}


def rename(value, labels):
    """Output-only renaming of the slot keys M0/M1 with the given labels; numbers and order unchanged."""
    if isinstance(value, dict):
        return {labels.get(key, key): rename(item, labels) for key, item in value.items()}
    if isinstance(value, list):
        return [rename(item, labels) for item in value]
    return value


def qualified_slot_outcomes(spec, frozen):
    """Resolved per decoding seed (all48) of a sealed generation, re-qualified by its own runner (CPU-only interface)."""
    for key in ('runner', 'runtime', 'completed', 'report'):
        require(sha(spec[key]) == spec[key + '_sha'], 'Pinned input changed: ' + str(spec[key]))
    _, readout, completed, runs = module(spec['runner']).qualify_generation(spec['runtime_sha'])
    require(readout == frozen and completed == read_json(spec['completed']) and completed.get('status') == spec['status'] and
            completed.get('generation') == spec['generation'] and completed.get('runtime_sha256') == spec['runtime_sha'],
            'Completion binding mismatch: ' + spec['generation'])
    ids, outcomes = frozen['populations']['all48_sensitivity']['task_ids'], {}
    for seed in SEEDS:
        name = spec['run_prefix'] + str(seed)
        require(name in runs, 'Missing qualified run: ' + name)
        receipt, rows = runs[name]['receipt'], runs[name]['rows']
        require(receipt.get('arm') == spec['arm'] and receipt.get('decoding_seed') == seed and receipt.get('strict_canary') is False and
                receipt.get('generation') == spec['generation'], 'Run binding refused: ' + name)
        outcomes[seed] = {task: rows[task]['resolved'] for task in ids}
    return outcomes


def contrast_rows(m0, m1, populations, resolution):
    """The unchanged v042 resolution_report with the pinned v025 paired_bootstrap: slot M1 minus slot M0 per population."""
    runs = {**{('M0', seed): m0[seed] for seed in SEEDS}, **{('M1', seed): m1[seed] for seed in SEEDS}}
    return {name: resolution_report(runs, populations[name], resolution.paired_bootstrap) for name in POPULATION_NAMES}


def reproduce_v067a(seed1, a0mc, populations, resolution):
    """Code-path and row identity before reading (3): M1q (seed 1) - A0mc through the same function equals the sealed v067-A contrast."""
    require(sha(V067A['report']) == V067A['report_sha'], 'Sealed v067-A report changed')
    sealed = read_json(V067A['report'])['populations']
    again = contrast_rows(a0mc, seed1, populations, resolution)
    for name in POPULATION_NAMES:
        mine = rename(again[name], {'M0': A0MC['label'], 'M1': SEED1['label']})
        for key in ('effect_pp', 'ci95_pp', 'seed_effects_pp', 'task_count', 'arms', 'per_task'):
            require(mine[key] == sealed[name][key], 'Sealed v067-A contrast not reproduced: ' + name + ' ' + key)
    return {name: dict(effect_pp=again[name]['effect_pp'], ci95_pp=again[name]['ci95_pp']) for name in POPULATION_NAMES}


def a0mc_rule(primary):
    """Reading (3): exactly one part (A) sentence, chosen by the primary40 CI of M1qs2mc - A0mc (bounds within the tolerance of 0 count as 0)."""
    low, high = primary['ci95_pp']
    tol = THRESHOLDS['boundary_abs_tolerance']
    key = 'lower_bound_above_zero' if low > tol else 'upper_bound_below_zero' if high < -tol else 'otherwise'
    arms = primary['arms']
    require(arms['M1']['denominator_per_seed'] == arms['M0']['denominator_per_seed'] == 40 and primary['task_count'] == 40,
            'primary40 denominator changed')
    x, y, d = sum(arms['M1']['resolved_by_seed']), sum(arms['M0']['resolved_by_seed']), primary['effect_pp']
    text = SENTENCES_A[key].replace('(x/120 vs y/120; ', '(%d/120 vs %d/120; ' % (x, y), 1)
    if key == 'lower_bound_above_zero':
        require(d > 0, 'Positive branch with a non-positive effect')
        text = text.replace('; +d pp [l, u])', '; +%.2f pp [%+.2f, %+.2f])' % (d, low, high), 1)
    else:
        text = text.replace('; d pp [l, u])', '; %+.2f pp [%+.2f, %+.2f])' % (d, low, high), 1)
    require('x/120' not in text and 'y/120' not in text and '[l, u]' not in text and not any(p in text.lower() for p in NEVER),
            'Part (A) sentence not filled')
    return dict(sentence_key=key, sentence=text, x=x, y=y, denominator=120, d_pp=d, l_pp=low, u_pp=high)


def sentence_b(pooled):
    """The part (B) sentence chosen by the pooled M1q-s2 primary40 resolved: >= 4, 1 to 3, or 0."""
    require(type(pooled) is int and 0 <= pooled <= 120, 'Pooled M1q-s2 resolved out of range')
    key = ('notable' if pooled >= THRESHOLDS['resolution_notable_min_pooled_primary40'] else 'below_notable' if pooled >= 1 else 'zero')
    text = SENTENCES_B[key] if key == 'zero' else SENTENCES_B[key].replace('x/120', '%d/120' % pooled, 1)
    require('x/120' not in text, 'Part (B) sentence not filled')
    return dict(sentence_key=key, sentence=text, x=pooled, seed1_pooled=SEED1_POOLED, denominator=120)


def update1_identity():
    """Prereg v067 part B, descriptive: update 1 loss of M1q-s2 beside M1q's (seed-independent because LoRA B starts at zero)."""
    s2 = read_json(RUNTIME)['m1q_training']['summary']['loss_first']
    s1 = read_json(SEED1['runtime'])['m1q_training']['summary']['loss_first']
    return {SEED1['label']: s1, LABELS['M1']: s2, 'equal_at_5_decimals': round(s1, 5) == round(s2, 5),
            'wording': 'descriptive; a mismatch at 5 decimals is reported as a data/order/base difference (prereg v067 part B)'}


def part_b_spec():
    """The frozen part (B) reading specification; every pre-written sentence and the banned phrases are verbatim in prereg v067."""
    text = PREREG.read_text(encoding='utf-8')
    require(all(text.count('"' + s + '"') == 1 for s in list(SENTENCES_B.values()) + list(SENTENCES_A.values())) and
            all('"' + p + '"' in text for p in NEVER), 'A pre-written sentence is not verbatim in prereg v067')
    return {'sentences_B': SENTENCES_B, 'sentences_A': SENTENCES_A, 'never': list(NEVER), 'seed1_pooled': SEED1_POOLED,
            'a0mc': {k: str(v) for k, v in A0MC.items()}, 'seed1': {k: str(v) for k, v in SEED1.items()},
            'v067_A': {k: str(v) for k, v in V067A.items()}}


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
    require(isinstance(binding, dict) and set(binding) == {'v067', 'v055'} and all(entry['runtime_sha256'] != PENDING for entry in binding.values()),
            'Evaluation binding is pending; --run refused')
    ledger = Inputs()
    for path, value in manifest['pins'].items():
        ledger.pin(path, value)
    frozen = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Original frozen readout changed')
    completions, qualified = {}, {}
    for key in ('v067', 'v055'):
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
    """The verbatim reading lines of the frozen prereg v067: part (B) Readings and Pre-written sentences, then the part (A) rule that
    reading (3) applies (Contrast, the pre-written sentences with their three branches, Never)."""
    text = PREREG.read_text(encoding='utf-8')
    part = lambda header: text.split(chr(10) + header + chr(10), 1)[1].split(chr(10) + '## ', 1)[0].splitlines()
    return ([line for line in part(PART_B) if line.startswith(('- Readings', '- Pre-written sentences'))] +
            [line for line in part(PART_A) if line.startswith(('- Contrast', '- Pre-written sentences', '  - ', '- Never'))])


def prepare(runtime_sha=None):
    """No runtime sha: the staged draft with the v067 runtime binding pending. The pinned runtime contract's sha: the pinned manifest."""
    readout = read_json(READOUT)
    require(sha(READOUT) == READOUT_SHA, 'Frozen readout mismatch')
    populations = validate_populations(readout)
    require(sha(PREREG) == PREREG_SHA, 'Frozen v067 pre-registration changed')
    pinned = runtime_sha is not None
    if pinned:
        require(sha(RUNTIME) == runtime_sha and Path(str(RUNTIME) + '.sha256').read_text().split()[0] == runtime_sha,
                'v067 runtime sidecar mismatch')
    runtime, comparator = read_json(RUNTIME if pinned else RUNTIME_DRAFT), read_json(COMPARATOR_RUNTIME)
    require(runtime.get('generation') == 'v067' and runtime['slot_binding']['sign'] == 'M1 - M0 = M1qs2mc - M0qmc' and
            runtime['comparator']['runtime_sha256'] == COMPARATOR_RUNTIME_SHA and
            runtime['comparator']['analysis_report_sha256'] == COMPARATOR_ANALYSIS_SHA and
            runtime['comparator']['per_task_sha256'] == COMPARATOR_PER_TASK_SHA and
            runtime['preregistration']['sha256'] == PREREG_SHA and comparator.get('generation') == 'v055',
            'v067 runtime contract does not bind this analysis')
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
            str(PARENT_ANALYZER): PARENT_ANALYZER_SHA, str(PARENT_ANALYSIS): PARENT_ANALYSIS_SHA,
            **{str(path): digest for path, digest, _ in DESCRIPTIVE_ROWS.values()},
            **{str(spec[key]): spec[key + '_sha'] for spec in (SEED1, A0MC) for key in ('runner', 'runtime', 'completed', 'report')},
            str(V067A['report']): V067A['report_sha'], str(V067A['analyzer']): V067A['analyzer_sha'],
            str(RUNNER): sha(RUNNER), str(RUNTIME): runtime_pin, str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / 'tools/test_analyze_matched_context_v067.py'): sha(ROOT / 'tools/test_analyze_matched_context_v067.py')}
    for path, value in pins.items():
        require(value == PENDING or sha(path) == value, 'Preparation source drift: ' + path)
    binding = {'v067': {'source': str(RUNNER), 'source_sha256': pins[str(RUNNER)], 'runtime': str(RUNTIME), 'runtime_sha256': runtime_pin},
               'v055': {'source': str(COMPARATOR_RUNNER), 'source_sha256': COMPARATOR_RUNNER_SHA,
                        'runtime': str(COMPARATOR_RUNTIME), 'runtime_sha256': COMPARATOR_RUNTIME_SHA}}
    doc = {'version': 'v067-matched-context-m1q-analysis', 'generation': 'v067',
           'status': 'PREPARED_NOT_ACTIVATED' if pinned else 'PREPARED_PENDING_PIN', 'root': str(ROOT),
           'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
           'preregistration': {'path': str(PREREG), 'sha256': PREREG_SHA, 'readings': 'part (B) Readings and Pre-written sentences; part (A) rule (Contrast, sentences, Never)',
                               'verbatim': prereg_readings()},
           'evaluation_binding': binding, 'pins': pins, 'execution_authorized': False, 'output': str(OUTPUT),
           'slots': SLOTS, 'sign': SIGN, 'labels': LABELS, 'slot_runs': runs, 'thresholds': THRESHOLDS, 'part_B': part_b_spec(), 'descriptives': DESCRIPTIVES,
           'descriptive_rows': {label: dict(path=str(p), sha256=d, location=w) for label, (p, d, w) in DESCRIPTIVE_ROWS.items()},
           'producers': {'mechanism_metrics': PRODUCER_SHA, 'analyze_mechanism': VALIDATOR_SHA, 'resolution_bootstrap': RESOLUTION_SHA},
           'populations': sorted(populations),
           'comparator_check': 'M0-slot producer rows must equal the v055 analysis per_task.jsonl rows of v055_M0qmc_s1..s3 (144 rows, 682e3492; the writer fields slot, decoding_seed and trace_view are not part of the producer row)',
           'requires': 'Qualified v067 completion (all_three_m1qs2mc_evaluations_qualified) and explicit --run with this manifest sha.'}
    target = MANIFEST if pinned else MANIFEST_DRAFT
    write_or_verify(target, json.dumps(doc, indent=2, sort_keys=True) + chr(10))
    write_or_verify(str(target) + '.sha256', sha(target) + '  ' + target.name + chr(10))
    return sha(target)


def ci_line(item):
    if item.get('effect') is None:
        return 'n/a'
    ci = item.get('ci95')
    return '%+.3f [%s]' % (item['effect'], 'undefined' if ci is None else '%+.3f, %+.3f' % tuple(ci))


def report_lines(populations, inputs, labels, descriptives, rows, part3):
    p, res, b, a = populations['primary40'], labels['resolution'], part3['b'], part3['a']
    lines = ['# Matched-context readout v067 (M1q-s2, second training seed of the M1q recipe)', '',
             'Slots frozen before reading: M0 = v055_M0qmc (matched template, M0q adapter), M1 = v067_M1qs2mc (matched template, M1q-s2 adapter, '
             'LoRA init seed 20261104). Readings of prereg v067 part B: (1) pooled resolved, (2) termination metrics and think rate vs M0qmc '
             '(descriptive, no label), (3) M1qs2mc - A0mc by the part (A) rule. M1q (v062, seed 1) and M1q-s2 are reported side by side, never '
             'pooled. Exploratory; known holdout; the independent-repository requirement of v062 is not lifted.', '',
             'Pre-written part (B) sentence (pooled primary40 resolved %d/120): **%s**' % (b['x'], b['sentence_key']), '', '> ' + b['sentence'], '',
             'Reading (3), the pre-written part (A) sentence chosen by the primary40 CI of M1qs2mc - A0mc: **%s**' % a['sentence_key'], '',
             '> ' + a['sentence'], '',
             '| population | v067_M1qs2mc resolved by seed | v066_A0mc resolved by seed | M1qs2mc - A0mc pp [95% CI] |', '|---|---|---|---|']
    for name in POPULATION_NAMES:
        item = part3['contrast'][name]
        lines.append('| %s | %s | %s | %+.3f [%+.3f, %+.3f] |' % (name, item['arms']['M1']['resolved_by_seed'], item['arms']['M0']['resolved_by_seed'],
                                                               item['effect_pp'], item['ci95_pp'][0], item['ci95_pp'][1]))
    lines += ['', 'Reading (1), resolution (v042 statistics, M1qs2mc - M0qmc in pp, descriptive):']
    for name, value in populations.items():
        r = value['primary']
        lines.append('- %s: %.3f pp, 95%% CI %s; v055_M0qmc %s, v067_M1qs2mc %s.' % (
            name, r['effect_pp'], r['ci95_pp'], r['arms']['M0']['resolved_by_seed'], r['arms']['M1']['resolved_by_seed']))
    lines += ['', 'Pooled v067_M1qs2mc resolved on primary40: %s/120 (notable: %s); M1q (seed 1): %d/120; never pooled.' % (
                  res['pooled_resolved'], res['notable'], SEED1_POOLED), '',
              'Reading (2), termination metrics and think rate on primary40 (three-seed means; descriptive, no label; M5-error: %s):' % labels['m5_error'],
              '', '| metric (primary40) | v055_M0qmc | v067_M1qs2mc | paired task-bootstrap effect [95% CI] |', '|---|---|---|---|']
    for key, spec in (('loop', 'M2.loop_rate'), ('submit', 'M3.submit_rate'), ('m5_error', 'M5.nonrecovery.error'),
                      ('think', 'M1.observed_thinking_step_rate'), ('repro', 'M4.opener.repro_script_opener')):
        lines.append('| %s | %s | %s | %s |' % (key, inputs['M0'].get(key), inputs['M1'].get(key), ci_line(p['secondary'].get(spec, {}))))
    lines += ['', 'Side by side (primary40 three-seed means; the two training seeds are never pooled; rows other than the slots are descriptive):', '',
              '| row | loop | submit | m5_error | think | repro | resolved by seed |', '|---|---|---|---|---|---|---|']
    for label, row in [('v055_M0qmc (slot M0)', inputs['M0']), ('v067_M1qs2mc (slot M1, seed 2)', inputs['M1'])] + list(rows.items()):
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (label, row.get('loop'), row.get('submit'), row.get('m5_error'),
                                                             row.get('think'), row.get('repro'), row.get('solved_by_seed')))
    ident = part3['identity']
    lines += ['', 'Update 1 loss (descriptive): M1q %s, M1q-s2 %s; equal at 5 decimals: %s.' % (
                  ident[SEED1['label']], ident[LABELS['M1']], ident['equal_at_5_decimals']),
              'Reproduction check: the sealed v067-A contrast (M1q seed 1 - A0mc) was re-derived exactly in all three populations before reading (3).',
              '', '| run | slot | LLM calls | compaction calls | tasks with compaction | content without reasoning |', '|---|---|---|---|---|---|']
    for name, row in descriptives['per_run'].items():
        lines.append('| %s | %s | %d | %d | %d | %d |' % (name, LABELS[row['slot']], row['llm_calls'], row['compaction_calls'],
                                                       row['tasks_with_compaction'], row['content_without_reasoning_calls']))
    return lines + ['', 'Never written: ' + ', '.join('"' + phrase + '"' for phrase in NEVER) + '.',
                    'Think-rate is read from returned reasoning (trace thinking events).']


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha, 'Analysis manifest mismatch')
    doc = read_json(MANIFEST)
    require(doc['root'] == str(ROOT) and doc['output'] == str(OUTPUT) and doc['slots'] == SLOTS and doc['sign'] == SIGN and
            doc['labels'] == LABELS and doc['thresholds'] == THRESHOLDS and doc['part_B'] == part_b_spec(),
            'Analysis root/output/slot freeze mismatch')
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
    labels = part_b_readings(inputs)
    frozen, ids = read_json(READOUT), validate_populations(readout)
    seed1_report = read_json(SEED1['report'])
    require(sum(seed1_report['readings']['inputs_primary40_three_seed_means'][SEED1['label']]['solved_by_seed']) == SEED1_POOLED,
            'Sealed M1q (seed 1) pooled primary40 resolved is not 10/120')
    seed1, a0mc = qualified_slot_outcomes(SEED1, frozen), qualified_slot_outcomes(A0MC, frozen)
    reproduced = reproduce_v067a(seed1, a0mc, ids, resolution)
    contrast = contrast_rows(a0mc, {seed: outcomes['M1', seed] for seed in SEEDS}, ids, resolution)
    part3 = dict(b=sentence_b(labels['resolution']['pooled_resolved']), a=a0mc_rule(contrast['primary40']), contrast=contrast,
                 identity=update1_identity(), reproduced=reproduced)
    require(part3['a']['x'] == part3['b']['x'], 'Pooled M1q-s2 resolved differs between readings (1) and (3)')
    audit = module(ledger.pin(AUDIT, AUDIT_SHA))
    require(audit.THRESHOLD == readout['evaluation_config']['compaction']['token_threshold'] == 14336, 'Compaction threshold changed')
    descriptives = trace_descriptives(readout, audit, producer, readout['populations']['all48_sensitivity']['task_ids'])
    rows = descriptive_rows(ledger)
    ledger.verify()
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_exploratory_matched_context_m1qs2_readout',
              'analysis_manifest_sha256': manifest_sha, 'readout_sha256': READOUT_SHA, 'immutable_statistics_sha256': LEGACY_SHA,
              'preregistration': doc['preregistration'], 'slots': relabel(SLOTS), 'sign': SIGN, 'labels': LABELS,
              'slot_runs': relabel(readout['future_runs']), 'populations': relabel(populations),
              'readings': {'inputs_primary40_three_seed_means': relabel(inputs), 'part_B': labels, 'sentence_B': part3['b'],
                           'reading_3_a0mc': {'sentence': part3['a'], 'populations': rename(contrast, {'M0': A0MC['label'], 'M1': LABELS['M1']}),
                                              'reproduction_of_v067_A': reproduced},
                           'update1_loss_identity': part3['identity'], 'descriptive_rows': rows},
              'trace_descriptives': relabel(descriptives), 'comparator_check': comparator, 'tokenizer': tokenizer,
              'method': readout['bootstrap'], 'limitations': readout['exposure_and_claim_limits'] + [
                  'Prereg v067 limits apply: the second training seed of the M1q recipe (LoRA init seed 20261104), reported beside M1q (v062, '
                  'seed 1) and never pooled with it; one data source, known holdout; the independent-repository requirement of v062 is not lifted; '
                  'reading 2 is descriptive (no PK or think label); reading 3 applies the part (A) rule to M1q-s2 - A0mc; no mechanism and no '
                  'causal attribution to masking; the descriptive rows are not contrasts and form no 2x2.'],
              'inputs': ledger.hashes, 'evaluation_completion': completions, 'automatic_training_or_submission': False}
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v067 analysis output required')
    OUTPUT.mkdir(exist_ok=True)
    legacy.write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    legacy.write_fresh(OUTPUT / 'per_task.jsonl', ''.join(json.dumps(
        {**row, 'slot': LABELS[key[0]], 'decoding_seed': key[1], 'trace_view': descriptives['per_task'][key[2]][key[0]][key[1]]},
        allow_nan=False) + '\n' for key, row in sorted(records.items())))
    legacy.write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(populations, inputs, labels, descriptives, rows, part3)) + '\n')
    for path in list(OUTPUT.iterdir()):
        legacy.write_fresh(str(path) + '.sha256', sha(path) + '  ' + path.name + '\n')
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json'),
            'sentence_B': part3['b']['sentence'], 'reading_3': part3['a']['sentence']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--runtime-sha', help='with --prepare: sha of the pinned v067 runtime contract (omitted: the staged draft)')
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
