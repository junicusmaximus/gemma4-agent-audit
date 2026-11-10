#!/usr/bin/env python3
"""v064-r2 joint reading: the pre-written cell of prereg v064-r2 (Readouts: Cells) and the confirmation trigger, read from the two sealed
per-arm reports (v064a = A_onpol, v064t = T_sub) side by side and never pooled. CPU only, write-once; no evaluation or statistic is rerun.

Each per-arm report must be sealed (report.json, report.md, per_task.jsonl with matching .sha256 sidecars), carry its status and the frozen
prereg, be bound to its arm's PINNED analysis manifest (sha recorded in the report; the arm's analyzer, pinned here, pinned there) and name
the approved training contract read here. T_sub UNMATCHED in that contract (arms.T_sub.data): A_onpol is read alone and no cell is written
(a T_sub report must then not exist). Otherwise the cell is chosen by the two primary40 termination labels only: A NON-COLLAPSED and T_sub
COLLAPSED -> CONTRAST; both NON-COLLAPSED -> both_NC; both COLLAPSED -> both_C; any INTERMEDIATE, UNDEFINED or the reverse mix -> direction
only (the sign of A_onpol - T_sub on primary40 loop and submit, no sentence). The confirmation trigger fires when A_onpol is NON-COLLAPSED
with pooled resolved >= 4/120 on primary40. Both arms must report the same update count U.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('PROJECT_ROOT/gemma4')
PREREG = ROOT.parent / 'briefing/research/gemma-4-2026/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md'
PREREG_SHA = '0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60'
PREREG_SECTION = '## Readouts (primary40 three-seed means; all48 and clean36 sensitivities; v058/v062 thresholds)'
MANIFEST = ROOT / 'contracts/onpolicy-cells-v064-analysis-DRAFT.json'
OUTPUT = ROOT / 'analysis/onpolicy-cells-v064'
TRAINING_CONTRACT = ROOT / 'contracts/onpolicy-training-v064r2-APPROVED.json'
ANALYZER = Path(__file__).resolve()
ARMS = {'A_onpol': dict(generation='v064a', label='v064_A_onpol', analyzer='tools/analyze_matched_context_v064a.py',
                        manifest='contracts/matched-context-v064a-analysis-PINNED.json', report_dir='analysis/matched-context-v064a',
                        status='qualified_exploratory_matched_context_a_onpol_readout'),
        'T_sub': dict(generation='v064t', label='v064_T_sub', analyzer='tools/analyze_matched_context_v064t.py',
                      manifest='contracts/matched-context-v064t-analysis-PINNED.json', report_dir='analysis/matched-context-v064t',
                      status='qualified_exploratory_matched_context_t_sub_readout')}
CELLS = {
    'CONTRAST': 'under this recipe and update count, the self-generated-trajectory arm did not meet the collapse criteria while the '
                'teacher-subset arm did; the difference follows the declared data bundle, not a single member',
    'both_NC': 'neither arm met the collapse criteria at U updates with these short subset recipes; this establishes neither no change relative '
               'to the base (see the CI rule) nor that the 208-update collapse is caused by update count',
    'both_C': 'the collapse criteria were also met when training on self-generated successful trajectories under this condition (n, t tasks, U '
              'updates; loop a, submit b), as with an equal-update teacher subset (c, d)'}
NO_CELL = {'direction_only': 'INTERMEDIATE / mixed ' + chr(0x2192) + ' direction only',
           'T_sub_unmatched': 'T_sub unmatched ' + chr(0x2192) + ' A alone, no cell'}
CONFIRMATION = ('a pre-registered second training seed before any "no-collapse" prose; "improved" prose additionally needs the CI lower bound > 0 '
                'on the confirmation')
TRIGGER = 'A NC ' + chr(0x2227) + ' ' + chr(0x2265) + ' 4/120'
MINIMUM, TOLERANCE = 4, 1e-12
SENSITIVITIES = ('all48_sensitivity', 'control_clean36_sensitivity')
RULES = {
    'cell': 'primary40 termination labels of the two sealed per-arm reports: (NON-COLLAPSED, COLLAPSED) -> CONTRAST; both NON-COLLAPSED -> both_NC; '
            'both COLLAPSED -> both_C; any other pair (INTERMEDIATE, UNDEFINED or COLLAPSED with NON-COLLAPSED) -> direction_only; T_sub UNMATCHED '
            'in the approved training contract -> T_sub_unmatched (A_onpol alone, no cell)',
    'fill': 'both_NC: U = the update count both reports carry; both_C: n = A_onpol trajectories and t = A_onpol tasks (its data manifest counts), '
            'U, a and b = A_onpol primary40 loop and submit, c and d = T_sub primary40 loop and submit (three decimals); CONTRAST is verbatim',
    'direction': 'direction_only: per metric (primary40 loop, submit) the sign of A_onpol - T_sub (equal within 1e-12; undefined when a value is '
                 'not finite); no sentence and no CI',
    'trigger': TRIGGER + ' on primary40: A_onpol NON-COLLAPSED and pooled resolved >= 4 of 120; when it fires: ' + CONFIRMATION,
    'never': 'the arms are never pooled; no causal claim beyond the cell wording; no single-member attribution'}


def full(rel):
    return ROOT / rel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


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
    require(hashlib.sha256(raw).hexdigest() == PREREG_SHA, 'Prereg v064-r2 changed')
    text = raw.decode('utf-8')
    require(text.count(PREREG_SECTION) == 1 and 'status: FROZEN' in text, 'Prereg v064-r2 Readouts section or FROZEN status missing')
    for template in CELLS.values():
        require(text.count('"' + template + '"') == 1, 'Pre-written cell not verbatim in prereg v064-r2: ' + template[:40])
    for phrase in list(NO_CELL.values()) + [CONFIRMATION, TRIGGER]:
        require(text.count(phrase) == 1, 'Pre-written rule not verbatim in prereg v064-r2: ' + phrase[:40])
    return text


def pins():
    """The prereg and the two per-arm analyzers with the shas they must keep; the per-arm reports are bound by rule at --run."""
    table = {str(PREREG): PREREG_SHA}
    for spec in ARMS.values():
        table[str(full(spec['analyzer']))] = sha(full(spec['analyzer']))
    return table


def contract_doc():
    prereg_text()
    paths = ('analyzer', 'manifest', 'report_dir')
    return {'schema': 'onpolicy-cells-v064-analysis-contract-v1', 'status': 'DRAFT', 'generation': 'v064-cells', 'root': str(ROOT),
            'output': str(OUTPUT), 'analyzer': {'path': str(ANALYZER), 'sha256': sha(ANALYZER)},
            'preregistration': {'path': str(PREREG), 'sha256': PREREG_SHA, 'section': PREREG_SECTION},
            'arms': {arm: {key: (str(full(value)) if key in paths else value) for key, value in spec.items()} for arm, spec in ARMS.items()},
            'training_contract': str(TRAINING_CONTRACT), 'cells': CELLS, 'no_cell': NO_CELL, 'confirmation': CONFIRMATION, 'rules': RULES,
            'report_rule': 'per arm: report.json, report.md and per_task.jsonl with matching .sha256 sidecars, the arm status and prereg ' +
                           PREREG_SHA[:8] + ', analysis_manifest_sha256 equal to the arm PINNED analysis manifest, which pins the arm analyzer '
                           'listed here; training.training_contract_sha256 equal to the approved training contract read at --run',
            'pins': pins(), 'outputs': ['report.json', 'report.md', 'report.json.sha256', 'report.md.sha256']}


def prepare():
    write_or_verify(MANIFEST, json.dumps(contract_doc(), indent=2) + '\n')
    write_or_verify(MANIFEST.with_name(MANIFEST.name + '.sha256'), sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    return sha(MANIFEST)


def approved():
    """The approved v064-r2 training contract both arms were trained from, and whether it marks T_sub UNMATCHED."""
    require(TRAINING_CONTRACT.is_file(), 'Approved training contract unavailable: ' + str(TRAINING_CONTRACT))
    contract = read_json(TRAINING_CONTRACT)
    require(contract.get('status') == 'APPROVED' and contract.get('prereg') == {'path': str(PREREG), 'sha256': PREREG_SHA},
            'Training contract is not the approved v064-r2 contract')
    return sha(TRAINING_CONTRACT), contract['arms']['T_sub']['data'] == 'UNMATCHED'


def sealed_report(arm, approved_sha):
    """One arm's sealed report, bound to its PINNED analysis manifest, its analyzer and the approved training contract."""
    spec, folder = ARMS[arm], full(ARMS[arm]['report_dir'])
    names = ('report.json', 'report.md', 'per_task.jsonl')
    for name in names:
        path, side = folder / name, folder / (name + '.sha256')
        require(path.is_file() and side.is_file(), arm + ' report not sealed: ' + name + ' or its sidecar is missing')
        require(side.read_text().split() == [sha(path), name], arm + ' report sidecar mismatch: ' + name)
    report, manifest, analyzer = read_json(folder / 'report.json'), full(spec['manifest']), str(full(spec['analyzer']))
    require(report.get('status') == spec['status'] and report['preregistration']['sha256'] == PREREG_SHA,
            arm + ' report status or prereg mismatch')
    require(manifest.is_file() and sha(manifest) == report['analysis_manifest_sha256'], arm + ' report is not bound to its PINNED analysis manifest')
    doc = read_json(manifest)
    require(doc.get('generation') == spec['generation'] and doc.get('status') == 'PREPARED_NOT_ACTIVATED' and
            doc['pins'].get(analyzer) == sha(analyzer) and doc['labels']['M1'] == spec['label'],
            arm + ' analysis manifest does not pin this arm analyzer')
    require(report['training']['training_contract_sha256'] == approved_sha, arm + ' report names another training contract')
    return report, dict({name: sha(folder / name) for name in names}, manifest=sha(manifest))


def view(report):
    """The primary40 reading of one sealed per-arm report, with its sensitivities and training facts (copied, not recomputed)."""
    readings, training = report['readings']['arm'], report['training']
    primary = readings['primary40']
    return {'label': primary['termination']['label'], 'loop': primary['termination']['loop'], 'submit': primary['termination']['submit'],
            'pooled_resolved': primary['repair']['pooled_resolved'], 'denominator': primary['repair']['denominator'],
            'effect_pp': primary['repair']['effect_pp'], 'ci95_pp': primary['repair']['ci95_pp'], 'ci_wording': primary['repair']['ci_wording'],
            'minimum_observation': primary['repair']['minimum_observation'], 'updates': training['updates'],
            'trajectories': training['counts']['trajectories'], 'tasks': training['counts']['tasks'],
            'sensitivities': {name: {'label': readings[name]['termination']['label'], 'pooled_resolved': readings[name]['repair']['pooled_resolved'],
                                     'denominator': readings[name]['repair']['denominator'], 'ci_wording': readings[name]['repair']['ci_wording']}
                              for name in SENSITIVITIES}}


def choose(a, t):
    if t is None:
        return 'T_sub_unmatched'
    return {('NON-COLLAPSED', 'COLLAPSED'): 'CONTRAST', ('NON-COLLAPSED', 'NON-COLLAPSED'): 'both_NC',
            ('COLLAPSED', 'COLLAPSED'): 'both_C'}.get((a['label'], t['label']), 'direction_only')


def direction(x, y):
    if not (finite(x) and finite(y)):
        return 'undefined'
    if math.isclose(x, y, rel_tol=0, abs_tol=TOLERANCE):
        return 'equal'
    return 'A_onpol lower' if x < y else 'A_onpol higher'


def cell(a, t):
    """Exactly one outcome: a pre-written cell sentence (placeholders filled) or a verbatim no-cell rule (with directions when both arms exist)."""
    key = choose(a, t)
    if key in NO_CELL:
        out = {'key': key, 'sentence': None, 'rule': NO_CELL[key]}
        if t is not None:
            out['direction'] = {metric: {'A_onpol': a[metric], 'T_sub': t[metric], 'A_onpol_minus_T_sub': direction(a[metric], t[metric])}
                                for metric in ('loop', 'submit')}
        return out
    text = CELLS[key]
    if key == 'both_NC':
        text = text.replace(' at U updates ', ' at %d updates ' % a['updates'], 1)
    elif key == 'both_C':
        text = text.replace('(n, t tasks, U updates; loop a, submit b)', '(%d, %d tasks, %d updates; loop %.3f, submit %.3f)' % (
            a['trajectories'], a['tasks'], a['updates'], a['loop'], a['submit']), 1).replace('(c, d)', '(%.3f, %.3f)' % (t['loop'], t['submit']), 1)
    require((text == CELLS[key]) == (key == 'CONTRAST') and ' U updates' not in text and '(c, d)' not in text, 'Cell placeholders not filled')
    return {'key': key, 'sentence': text, 'rule': 'pre-written cell, prereg v064-r2 Readouts'}


def trigger(a):
    fires = a['label'] == 'NON-COLLAPSED' and a['pooled_resolved'] >= MINIMUM
    return {'rule': TRIGGER + ' (primary40)', 'fires': fires, 'A_onpol_label': a['label'], 'A_onpol_pooled_resolved': a['pooled_resolved'],
            'consequence': CONFIRMATION if fires else None}


def report_lines(report):
    arms, reading, trig = report['arms'], report['cell'], report['confirmation_trigger']
    lines = ['# v064-r2 joint reading: cell and confirmation trigger', '',
             'Arms side by side, never pooled: A_onpol (v064a) and T_sub (v064t), each against the sealed v066 A0mc base; primary40 three-seed '
             'means, all48 and clean36 as sensitivities; prereg v064-r2 ' + PREREG_SHA[:8] + '. One training seed per arm; known holdout.', '']
    if reading['sentence']:
        lines += ['Pre-written cell (chosen by the two termination labels): **' + reading['key'] + '**', '', '> ' + reading['sentence'], '']
    else:
        lines += ['No cell: "' + reading['rule'] + '".', '']
        for metric, item in sorted(reading.get('direction', {}).items()):
            lines.append('- %s: A_onpol %s, T_sub %s: %s' % (metric, item['A_onpol'], item['T_sub'], item['A_onpol_minus_T_sub']))
        lines += [''] if reading.get('direction') else []
    lines += ['Confirmation trigger (' + trig['rule'] + '): **' + ('fires' if trig['fires'] else 'does not fire') + '**' +
              ('; ' + trig['consequence'] + '.' if trig['fires'] else '.'), '',
              '| arm | termination | loop | submit | pooled resolved | arm - A0mc pp [95% CI] | CI wording | U | trajectories | tasks |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for arm, item in arms.items():
        if item is None:
            lines.append('| %s | not trained (UNMATCHED) | | | | | | | | |' % arm)
            continue
        lines.append('| %s | %s | %s | %s | %d/%d | %+.2f [%+.2f, %+.2f] | %s | %d | %d | %d |' % (
            arm, item['label'], item['loop'], item['submit'], item['pooled_resolved'], item['denominator'], item['effect_pp'],
            item['ci95_pp'][0], item['ci95_pp'][1], item['ci_wording'], item['updates'], item['trajectories'], item['tasks']))
    lines += ['', '| arm | population | termination | pooled resolved | CI wording |', '|---|---|---|---|---|']
    for arm, item in arms.items():
        for name, row in (item or {}).get('sensitivities', {}).items():
            lines.append('| %s | %s | %s | %d/%d | %s |' % (arm, name, row['label'], row['pooled_resolved'], row['denominator'], row['ci_wording']))
    return lines + ['', 'Sensitivities are descriptive and never choose the cell. ' + RULES['never'] + '.']


def analyze(manifest_sha):
    require(sha(MANIFEST) == manifest_sha, 'Joint contract mismatch')
    doc = read_json(MANIFEST)
    require(doc == contract_doc(), 'Contract differs from the re-derived contract (an input or this analyzer changed)')
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), 'Fresh v064 cells output required')
    approved_sha, unmatched = approved()
    a_report, a_files = sealed_report('A_onpol', approved_sha)
    if unmatched:
        require(not full(ARMS['T_sub']['report_dir']).exists(), 'T_sub is UNMATCHED in the approved contract but a T_sub report exists')
        t_report, t_files = None, None
    else:
        t_report, t_files = sealed_report('T_sub', approved_sha)
    a, t = view(a_report), (None if t_report is None else view(t_report))
    require(t is None or a['updates'] == t['updates'], 'The two arms report different update counts')
    report = {'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'qualified_v064_onpolicy_cells_reading',
              'contract_sha256': manifest_sha, 'analyzer_sha256': sha(ANALYZER), 'preregistration': doc['preregistration'],
              'training_contract': {'path': str(TRAINING_CONTRACT), 'sha256': approved_sha, 'T_sub_unmatched': unmatched},
              'arms': {'A_onpol': a, 'T_sub': t}, 'cell': cell(a, t), 'confirmation_trigger': trigger(a), 'rules': RULES,
              'inputs': {'A_onpol': a_files, 'T_sub': t_files, 'pins': doc['pins']}}
    OUTPUT.mkdir(exist_ok=True)
    write_fresh(OUTPUT / 'report.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_fresh(OUTPUT / 'report.md', '\n'.join(report_lines(report)) + '\n')
    for name in ('report.json', 'report.md'):
        write_fresh(OUTPUT / (name + '.sha256'), sha(OUTPUT / name) + '  ' + name + '\n')
    return {'status': report['status'], 'output': str(OUTPUT), 'report_sha256': sha(OUTPUT / 'report.json'), 'cell': report['cell']['key'],
            'sentence': report['cell']['sentence'], 'confirmation_trigger': report['confirmation_trigger']['fires']}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare', action='store_true', help='write or verify the DRAFT contract; reads no report')
    ap.add_argument('--run', action='store_true', help='the joint reading; needs --manifest-sha and the sealed per-arm reports')
    ap.add_argument('--manifest-sha')
    args = ap.parse_args()
    require([args.prepare, args.run].count(True) == 1, 'Choose exactly one of --prepare, --run')
    if args.prepare:
        print(json.dumps({'status': 'prepared_only', 'contract_sha256': prepare(), 'outcomes_read': False}))
    else:
        require(args.manifest_sha is not None, 'Explicit contract SHA required')
        print(json.dumps(analyze(args.manifest_sha)))


if __name__ == '__main__':
    main()
