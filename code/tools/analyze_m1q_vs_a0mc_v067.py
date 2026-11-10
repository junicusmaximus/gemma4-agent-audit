#!/usr/bin/env python3
"""v067 part (A): M1qmc - A0mc on pooled resolved, the v062 statistics re-pinned (CPU only, write-once).

Slot M0 = the sealed v066 A0mc rows (results/v066_A0mc_s1..s3), slot M1 = the sealed v062 M1qmc rows (results/v062_M1qmc_s1..s3).
Rows are re-qualified by each generation's own CPU-only runner interface (qualify_generation); resolved is compared with the
unchanged v042 resolution_report and the pinned v025 paired_bootstrap (task-level, three decoding seeds grouped, 20,000 draws,
seed 20261103): primary40 first, all48 and clean36 as sensitivities. Exactly one pre-written sentence of prereg v067 (A) is chosen
by the primary40 CI. Termination metrics are copied verbatim from the two sealed reports (descriptive). No evaluation is rerun.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('PROJECT_ROOT/gemma4')
LEGACY = ROOT / 'tools/analyze_mask_control_v042.py'
LEGACY_SHA = '23915fb00c2836b00fe14e3adfeecb865bdf52eb34e1fe3a04c4093cffdaf376'
if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
    raise ValueError('Immutable v042 statistical implementation changed')
_spec = importlib.util.spec_from_file_location('v067_immutable_statistics', LEGACY)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
require, sha, read_json, module = legacy.require, legacy.sha, legacy.read_json, legacy.module
ARMS, SEEDS, DRAWS, RNG_SEED = legacy.ARMS, legacy.SEEDS, legacy.DRAWS, legacy.RNG_SEED

MANIFEST = ROOT / 'contracts/m1q-vs-a0mc-v067-analysis-DRAFT.json'
OUTPUT = ROOT / 'analysis/m1q-vs-a0mc-v067'
ANALYZER = Path(__file__).resolve()
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-m1q-seed2-and-a0mc-branches-v067.md'
PREREG_SHA = '766c1e45386f93121fd29e8beca7de30526b9bb8f52482644608e49d17ce2b46'
PREREG_SECTION = '## (A) M1q ' + chr(0x2212) + ' A0mc reading (before the A0mc report is read)'
RESOLUTION = ROOT / 'tools/analyze_contract_v02_v025.py'
RESOLUTION_SHA = '20208c74cc921d9dd4975b1b6a7129b6a4b55825f17930429a4fa624935e229c'
READOUT_SHA = '6319262b25c19ddcbb0e34a33f5e97325a7d6117a7b8faad8d4a4c7ff772c909'
LABELS = {'M0': 'v066_A0mc', 'M1': 'v062_M1qmc'}
V062_LABELS = {'M0': 'v055_M0qmc', 'M1': 'v062_M1qmc'}
SIGN = 'M1 - M0 = v062_M1qmc - v066_A0mc (the unchanged v042 resolution_report takes slot M1 as treated)'
POPULATIONS = ('primary40', 'all48_sensitivity', 'control_clean36_sensitivity')
TOLERANCE = 1e-12
SLOTS = {
    'M0': dict(generation='v066', arm='A0mc', run_prefix='v066_A0mc_s',
               runner='tools/run_matched_context_v066.py', runner_sha='0542edbf7d8e3ee57c605b75e15f0d8bd9a090d9fddd9d37fff7abc0c2f72e8a',
               runtime='contracts/matched-context-v066-a0mc-runtime-PINNED.json', runtime_sha='0f5e04b257ffe1544633f90fba4e51e377dae82a82f97bb029d1b8c3d49944e5',
               analysis='contracts/matched-context-v066-a0mc-analysis-PINNED.json', analysis_sha='a070f299535872e3231a079b76b8e17f352e83cf2fc2f1b143795b732ac19a67',
               completed='migration/matched-context-v066/evaluation/completed.json', completed_sha='fb1bf16f96bbb5002d77e0bda8b4a81a5a9c81a40bdd95b6616e72009f4f7500',
               status='all_three_a0mc_evaluations_qualified', report_dir='analysis/matched-context-v066',
               report_status='qualified_descriptive_matched_context_a0mc_readout'),
    'M1': dict(generation='v062', arm='M1q', run_prefix='v062_M1qmc_s',
               runner='tools/run_matched_context_v062.py', runner_sha='621cd9363e0523ed6391705c9329d003213c09f34eb8d3ff3ab78047d059da2b',
               runtime='contracts/matched-context-v062-runtime-PINNED.json', runtime_sha='3717c8d036ae17cf2f4713458b5fe31a36cff95903dadf4e9175eb2efee9aefe',
               analysis='contracts/matched-context-v062-analysis-PINNED.json', analysis_sha='91295f7cde58104928f6d72aef9f4124a2d2413323edb9005f007d0d0af27a0f',
               completed='migration/matched-context-v062/evaluation/completed.json', completed_sha='5a41099813ddf91985ac21bc16de560cf9b49dc7903d3ce6ef595938d9dbb677',
               status='all_three_m1qmc_evaluations_qualified', report_dir='analysis/matched-context-v062',
               report_status='qualified_exploratory_matched_context_m1q_readout',
               report_sha='70f20fc0055cedbb0a4a0c1e2cc82dd1b3afa2d3430084b83e3cac664713ba1d',
               per_task_sha='2f991e816fc6ce373c597f2a6431b0ea5bbec82a9f01ef7140924e80deb61927'),
}
SENTENCES = {
    'lower_bound_above_zero': 'the masked arm on the QAT base resolved more tasks than the same-template base (x/120 vs y/120; +d pp [l, u]) '
                              + chr(0x2014) + ' one training seed, known holdout',
    'upper_bound_below_zero': 'the masked arm resolved fewer tasks than the same-template base (x/120 vs y/120; d pp [l, u])',
    'otherwise': 'relative to the same-template base the difference is uncertain (x/120 vs y/120; d pp [l, u])',
}
NEVER = ('recovery', 'partial recovery', 'closed the gap', 'masking restores / causes repair', 'no-collapse')


def full(rel):
    return ROOT / rel


def relabel(value, labels=LABELS):
    """Output-only renaming of the slot keys M0/M1; numbers and order unchanged."""
    if isinstance(value, dict):
        return {labels.get(key, key): relabel(item, labels) for key, item in value.items()}
    if isinstance(value, list):
        return [relabel(item, labels) for item in value]
    return value


def write_fresh(path, text):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def write_or_verify(path, text):
    if Path(path).exists():
        require(Path(path).read_text(encoding='utf-8') == text, 'Existing file differs from the re-derived text: ' + str(path))
        return
    write_fresh(path, text)


def prereg_text():
    raw = PREREG.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PREREG_SHA, 'Prereg v067 changed')
    text = raw.decode('utf-8')
    require(text.count(PREREG_SECTION) == 1 and 'status: FROZEN' in text, 'Prereg v067 section or FROZEN status missing')
    for template in SENTENCES.values():
        require(text.count('"' + template + '"') == 1, 'Pre-written sentence not verbatim in prereg v067: ' + template[:40])
    for phrase in NEVER:
        require('"' + phrase + '"' in text, 'Banned phrase list changed: ' + phrase)
    return text


def pins():
    """Every bound input with the sha it must have. The v066 report is bound by rule at --run (it does not exist when this is written)."""
    table = {str(PREREG): PREREG_SHA, str(LEGACY): LEGACY_SHA, str(RESOLUTION): RESOLUTION_SHA, str(legacy.READOUT): READOUT_SHA}
    for slot in ('M0', 'M1'):
        for key in ('runner', 'runtime', 'analysis', 'completed'):
            table[str(full(SLOTS[slot][key]))] = SLOTS[slot][key + '_sha']
    folder = full(SLOTS['M1']['report_dir'])
    table[str(folder / 'report.json')] = SLOTS['M1']['report_sha']
    table[str(folder / 'per_task.jsonl')] = SLOTS['M1']['per_task_sha']
    return table


def verify_pins(table):
    for path, value in table.items():
        require(Path(path).is_file() and sha(Path(path)) == value, 'Pinned input missing or changed: ' + path)


def contract_doc():
    prereg_text()
    table = pins()
    verify_pins(table)
    paths = ('runner', 'runtime', 'analysis', 'completed', 'report_dir')
    return {
        'schema': 'm1q-vs-a0mc-v067-analysis-contract-v1', 'status': 'DRAFT', 'generation': 'v067-A', 'root': str(ROOT), 'output': str(OUTPUT),
        'analyzer': {'path': str(ANALYZER), 'sha256': sha(ANALYZER)},
        'preregistration': {'path': str(PREREG), 'sha256': PREREG_SHA, 'section': PREREG_SECTION},
        'labels': LABELS, 'sign': SIGN,
        'slots': {slot: {key: (str(full(value)) if key in paths else value) for key, value in spec.items()} for slot, spec in SLOTS.items()},
        'statistic': {'function': 'v042 resolution_report with the v025 paired_bootstrap (both unchanged and pinned)',
                      'unit': 'resolved from task_results, three decoding seeds per task', 'pairing': 'by task; seeds grouped as task means; task-level bootstrap',
                      'draws': DRAWS, 'seed': RNG_SEED, 'interval': '95% percentile, numpy quantile method linear', 'primary': 'primary40',
                      'sensitivities': ['all48_sensitivity', 'control_clean36_sensitivity'],
                      'populations_source': 'frozen readout ' + READOUT_SHA + ' via v042 validate_populations', 'boundary_abs_tolerance_pp': TOLERANCE},
        'sentences': SENTENCES,
        'sentence_rule': 'primary40 CI: lower > 0 -> lower_bound_above_zero; upper < 0 -> upper_bound_below_zero; otherwise -> otherwise; '
                         'bounds within 1e-12 pp of 0 count as 0; exactly one applies; chosen by the CI, never after inspection',
        'number_format': 'x, y = pooled resolved of M1 (v062_M1qmc) and M0 (v066_A0mc) on primary40, of 120; d = effect in pp, 2 decimals '
                         '(the + is the template literal in lower_bound_above_zero, otherwise signed); l, u = CI bounds in pp, signed, 2 decimals',
        'never': list(NEVER),
        'v066_report_rule': 'report.json, report.md and per_task.jsonl under ' + str(full(SLOTS['M0']['report_dir'])) + ' must exist with matching .sha256 '
                            'sidecars, status ' + SLOTS['M0']['report_status'] + ', analysis_manifest_sha256 ' + SLOTS['M0']['analysis_sha'] +
                            ' and evaluation_completion.v066 equal to the pinned completed.json; their shas are recorded in the output',
        'reproduction_check': 'before any v066 row is used, the sealed v062 per_task resolved arrays (v055_M0qmc, v062_M1qmc) run through the same '
                              'function must reproduce the sealed v062 effect, CI, seed effects and per-seed counts in all three populations; '
                              'the re-qualified v062 rows must equal the sealed v062_M1qmc arrays',
        'termination_metrics': 'descriptive only, copied verbatim: v062 readings.inputs_primary40_three_seed_means.v062_M1qmc and v066 '
                               'readings.descriptive_by_population.<population>.v066_A0mc; no label, threshold or contrast',
        'pins': table, 'outputs': ['report.json', 'report.md', 'report.json.sha256', 'report.md.sha256'],
    }


def prepare():
    text = json.dumps(contract_doc(), indent=2) + '\n'
    write_or_verify(MANIFEST, text)
    write_or_verify(MANIFEST.with_name(MANIFEST.name + '.sha256'), sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    return sha(MANIFEST)


def sealed_report(slot, names=('report.json', 'report.md', 'per_task.jsonl')):
    """The slot's sealed report: every file present with a matching sidecar, status, manifest and completion binding."""
    spec, folder = SLOTS[slot], full(SLOTS[slot]['report_dir'])
    for name in names:
        path, side = folder / name, folder / (name + '.sha256')
        require(path.is_file() and side.is_file(), spec['generation'] + ' report not sealed: ' + name + ' or its sidecar is missing')
        require(side.read_text().split() == [sha(path), name], spec['generation'] + ' report sidecar mismatch: ' + name)
    report = read_json(folder / 'report.json')
    require(report.get('status') == spec['report_status'] and report.get('analysis_manifest_sha256') == spec['analysis_sha'],
            spec['generation'] + ' report status/manifest binding mismatch')
    require((report.get('evaluation_completion') or {}).get(spec['generation']) == read_json(full(spec['completed'])),
            spec['generation'] + ' report completion binding mismatch')
    return report, {name: sha(folder / name) for name in names}


def qualified_outcomes(frozen, slots=('M0', 'M1')):
    """Resolved per (slot, seed) for all48, from rows re-qualified by each generation's own runner (CPU-only interface)."""
    ids = frozen['populations']['all48_sensitivity']['task_ids']
    outcomes, receipts = {}, {}
    for slot in slots:
        spec = SLOTS[slot]
        _, readout, completed, runs = module(full(spec['runner'])).qualify_generation(spec['runtime_sha'])
        require(readout == frozen, 'Readout of ' + spec['generation'] + ' differs from the frozen readout')
        require(completed == read_json(full(spec['completed'])) and completed.get('status') == spec['status'] and
                completed.get('generation') == spec['generation'] and completed.get('runtime_sha256') == spec['runtime_sha'] and
                completed.get('submission_started') is False and completed.get('cleanup_completed') is True,
                'Completion binding mismatch: ' + spec['generation'])
        for seed in SEEDS:
            name = spec['run_prefix'] + str(seed)
            require(name in runs, 'Missing qualified run: ' + name)
            receipt, rows = runs[name]['receipt'], runs[name]['rows']
            require(receipt.get('arm') == spec['arm'] and receipt.get('decoding_seed') == seed and receipt.get('generation') == spec['generation'] and
                    receipt.get('strict_canary') is False and receipt.get('task_rows') == len(ids) and
                    receipt.get('sha256') == sha(ROOT / 'results' / name / 'task_results.jsonl'), 'Run binding refused: ' + name)
            outcomes[slot, seed] = {task: rows[task]['resolved'] for task in ids}
            receipts[name] = {'task_results_sha256': receipt['sha256'], 'task_rows': receipt['task_rows']}
    return outcomes, receipts


def reproduction_check(report062, populations, resolution, outcomes=None):
    """Code-path identity on sealed, already-read v062 data; with outcomes, also row identity of the re-qualified M1 rows."""
    done = {}
    for pop in POPULATIONS:
        sealed = report062['populations'][pop]['primary']
        per_task = sealed['per_task']
        ids = [row['instance_id'] for row in per_task]
        require(ids == populations[pop], 'Sealed v062 population order/membership differs: ' + pop)
        runs = {(slot, seed): {row['instance_id']: row[V062_LABELS[slot]][i] for row in per_task}
                for slot in ARMS for i, seed in enumerate(SEEDS)}
        again = relabel(legacy.resolution_report(runs, ids, resolution.paired_bootstrap), V062_LABELS)
        for key in ('effect_pp', 'ci95_pp', 'seed_effects_pp', 'task_count', 'arms', 'per_task'):
            require(again[key] == sealed[key], 'Sealed v062 resolution statistics not reproduced: ' + pop + ' ' + key)
        if outcomes is not None:
            require(all([outcomes['M1', seed][row['instance_id']] for seed in SEEDS] == row['v062_M1qmc'] for row in per_task),
                    'Re-qualified v062 rows differ from the sealed v062_M1qmc arrays: ' + pop)
        done[pop] = {'effect_pp': again['effect_pp'], 'ci95_pp': again['ci95_pp'],
                     'resolved_by_seed': {label: again['arms'][label]['resolved_by_seed'] for label in V062_LABELS.values()}}
    return done


def contrast(outcomes, populations, bootstrap):
    return {pop: legacy.resolution_report(outcomes, populations[pop], bootstrap) for pop in POPULATIONS}


def choose(primary):
    low, high = primary['ci95_pp']
    if low > TOLERANCE:
        return 'lower_bound_above_zero'
    if high < -TOLERANCE:
        return 'upper_bound_below_zero'
    return 'otherwise'


def fill(key, primary):
    arms = primary['arms']
    require(arms['M1']['denominator_per_seed'] == arms['M0']['denominator_per_seed'] == 40 and primary['task_count'] == 40, 'primary40 denominator changed')
    x, y = sum(arms['M1']['resolved_by_seed']), sum(arms['M0']['resolved_by_seed'])
    d, (low, high) = primary['effect_pp'], primary['ci95_pp']
    text = SENTENCES[key].replace('(x/120 vs y/120; ', '(%d/120 vs %d/120; ' % (x, y), 1)
    if key == 'lower_bound_above_zero':
        require(d > 0, 'Positive branch with a non-positive effect')
        text = text.replace('; +d pp [l, u])', '; +%.2f pp [%+.2f, %+.2f])' % (d, low, high), 1)
    else:
        text = text.replace('; d pp [l, u])', '; %+.2f pp [%+.2f, %+.2f])' % (d, low, high), 1)
    require(text != SENTENCES[key] and 'x/120' not in text and 'y/120' not in text and '[l, u]' not in text, 'Sentence placeholders not filled')
    require(not any(phrase in text.lower() for phrase in NEVER), 'Banned phrase in the filled sentence')
    return {'sentence_key': key, 'sentence': text, 'x': x, 'y': y, 'denominator': 120, 'd_pp': d, 'l_pp': low, 'u_pp': high}


def termination(report062, report066):
    return {'v062_M1qmc_primary40_three_seed_means': report062['readings']['inputs_primary40_three_seed_means']['v062_M1qmc'],
            'v066_A0mc_by_population': {pop: report066['readings']['descriptive_by_population'][pop]['v066_A0mc'] for pop in POPULATIONS},
            'wording': 'descriptive only, copied from the sealed reports; no label, threshold or contrast (prereg v067 A reads resolution only)'}


def base_inputs():
    report062, sealed062 = sealed_report('M1')
    require(sealed062['report.json'] == SLOTS['M1']['report_sha'] and sealed062['per_task.jsonl'] == SLOTS['M1']['per_task_sha'],
            'Sealed v062 report shas changed')
    frozen = read_json(legacy.READOUT)
    require(sha(legacy.READOUT) == READOUT_SHA, 'Frozen readout changed')
    return report062, sealed062, frozen, legacy.validate_populations(frozen), module(RESOLUTION)


def self_check():
    """Reads no v066 row or report: contract derivation, pins, v062 report binding, reproduction, v062 row re-qualification."""
    doc = contract_doc()
    report062, _, frozen, populations, resolution = base_inputs()
    reproduced = reproduction_check(report062, populations, resolution)
    outcomes, receipts = qualified_outcomes(frozen, slots=('M1',))
    reproduction_check(report062, populations, resolution, outcomes)
    folder = full(SLOTS['M0']['report_dir'])
    return {'status': 'self_check_passed', 'analyzer_sha256': doc['analyzer']['sha256'], 'reproduced_v062': reproduced,
            'v062_rows_requalified': sorted(receipts), 'v066_report_files_present': sorted(p.name for p in folder.glob('*')) if folder.exists() else [],
            'v066_rows_read': False}


def report_lines(report):
    reading, pops = report['reading_primary40'], report['populations']
    lines = ['# M1q versus the matched-template base A0mc (prereg v067, part A)', '',
             'Slots: M0 = v066_A0mc (E4B QAT base, no adapter, matched template), M1 = v062_M1qmc (M1q adapter, matched template). Contrast M1 - M0 on '
             'pooled resolved, paired by task with the three decoding seeds grouped; ' + str(DRAWS) + ' task-level bootstrap draws (seed ' + str(RNG_SEED) +
             '); prereg v067 ' + PREREG_SHA[:8] + '. Exploratory; known holdout; one training seed.', '',
             'Pre-written reading on primary40 (exactly one sentence applies, chosen by the CI): **' + reading['sentence_key'] + '**', '',
             '> ' + reading['sentence'], '',
             '| population | v062_M1qmc resolved by seed | v066_A0mc resolved by seed | effect pp [95% CI] |', '|---|---|---|---|']
    for pop in POPULATIONS:
        item = pops[pop]
        lines.append('| %s | %s | %s | %+.3f [%+.3f, %+.3f] |' % (pop, item['arms']['v062_M1qmc']['resolved_by_seed'], item['arms']['v066_A0mc']['resolved_by_seed'],
                                                               item['effect_pp'], item['ci95_pp'][0], item['ci95_pp'][1]))
    terms = report['termination_metrics_descriptive']
    rows = (('v062_M1qmc', terms['v062_M1qmc_primary40_three_seed_means']), ('v066_A0mc', terms['v066_A0mc_by_population']['primary40']['three_seed_means']))
    lines += ['', 'Termination metrics on primary40, three-seed means (descriptive only, copied from the sealed reports; no label):', '',
              '| row | loop | submit | m5_error | think | repro |', '|---|---|---|---|---|---|']
    lines += ['| ' + name + ' | ' + ' | '.join(str(row.get(key)) for key in ('loop', 'submit', 'm5_error', 'think', 'repro')) + ' |' for name, row in rows]
    lines += ['', 'Reproduction check: the sealed v062 resolution statistics were re-derived exactly in all three populations before any v066 row was used.',
              'Never written: ' + ', '.join('"' + phrase + '"' for phrase in NEVER) + '.']
    return lines


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha, 'Analysis contract mismatch')
    doc = read_json(MANIFEST)
    require(doc == contract_doc(), 'Contract differs from the re-derived contract (an input or this analyzer changed)')
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v067-A output required')
    report066, sealed066 = sealed_report('M0')
    report062, sealed062, frozen, populations, resolution = base_inputs()
    reproduced = reproduction_check(report062, populations, resolution)
    outcomes, receipts = qualified_outcomes(frozen)
    reproduction_check(report062, populations, resolution, outcomes)
    results = contrast(outcomes, populations, resolution.paired_bootstrap)
    reading = fill(choose(results['primary40']), results['primary40'])
    a0mc = report066['readings']['descriptive_by_population']['primary40']['v066_A0mc']
    require(reading['y'] == a0mc['pooled_resolved'] and
            reading['x'] == sum(report062['readings']['inputs_primary40_three_seed_means']['v062_M1qmc']['solved_by_seed']),
            'Pooled resolved differs from the sealed reports')
    verify_pins(doc['pins'])
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_v067_A_m1q_minus_a0mc_reading',
              'contract_sha256': manifest_sha, 'analyzer_sha256': sha(ANALYZER), 'preregistration': doc['preregistration'], 'labels': LABELS,
              'sign': SIGN, 'statistic': doc['statistic'], 'sentence_rule': doc['sentence_rule'], 'reading_primary40': reading,
              'populations': relabel(results), 'termination_metrics_descriptive': termination(report062, report066),
              'reproduction_check_v062': reproduced, 'qualified_runs': receipts,
              'inputs': {'v066_report': sealed066, 'v062_report': sealed062, 'pins': doc['pins']},
              'not_claimed': 'prereg v067 Not claimed applies; never written: ' + ', '.join(NEVER)}
    OUTPUT.mkdir(exist_ok=True)
    write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(report)) + '\n')
    for name in ('report.json', 'report.md'):
        write_fresh(OUTPUT / (name + '.sha256'), sha(OUTPUT / name) + '  ' + name + '\n')
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json'),
            'sentence_key': reading['sentence_key'], 'sentence': reading['sentence']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true', help='write or verify the DRAFT contract; reads no outcome')
    ap.add_argument('--self-check', action='store_true', help='contract derivation, pins, v062 reproduction and re-qualification; reads no v066 row')
    ap.add_argument('--run', action='store_true', help='the v067 A reading; needs --manifest-sha and the sealed v066 report')
    ap.add_argument('--manifest-sha')
    args = ap.parse_args()
    require([args.prepare, args.self_check, args.run].count(True) == 1, 'Choose exactly one of --prepare, --self-check, --run')
    if args.prepare:
        print(json.dumps({'status': 'prepared_only', 'contract_sha256': prepare(), 'outcomes_read': False}))
    elif args.self_check:
        print(json.dumps(self_check()))
    else:
        require(args.manifest_sha is not None, 'Explicit contract SHA required')
        print(json.dumps(analyze(args.manifest_sha)))


if __name__ == '__main__':
    main()
