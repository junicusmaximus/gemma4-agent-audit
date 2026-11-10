"""External-panel v070: analysis of one qualified generation with the readings of the frozen pre-registration b7b4d4be. Primary: the termination label of
each arm (A_onpol training seeds 1 and 2, T_sub) over the complete task set T_c with the frozen v064-r2 thresholds, and the one pre-written cell E1, E2 or
E3, which applies as a reading only when the planned paired set is complete. Repair is always reported, per arm and per seed, with the A0mc-ext arm of v068
beside it when the identity check passed and the sealed v068 outputs verify (attempt 2 of v068, the sole v068 reading). The cache state of the pin is reported
under the rule that the analysis manifest fixed at --prepare (k-disclose by default, or k-strict). Every row is reported, completed or not."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, 'PROJECT_ROOT/codex/experiments/v070-git-r2/runner')
import run_external_e4b_v070git2 as runner

OUTPUT = runner.WORK / 'analysis/readout'
MANIFEST = runner.WORK / 'contracts/external-e4b-v070git2-analysis-PINNED.json'
MANIFEST_DRAFT = runner.WORK / 'contracts/external-e4b-v070git2-analysis-DRAFT.json'
PRODUCER, PRODUCER_SHA = runner.ROOT / 'tools/mechanism_metrics_v032.py', '36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d'
RESOLUTION, RESOLUTION_SHA = runner.ROOT / 'tools/analyze_contract_v02_v025.py', '20208c74cc921d9dd4975b1b6a7129b6a4b55825f17930429a4fa624935e229c'
DRAWS, SEED = 20000, 20261103
COMPARATOR_ARM = 'A0mc'
ROW_FIELDS = ('resolved', 'error', 'test_exit_code', 'tool_calls', 'total_llm_calls', 'agent_patch_size', 'duration_seconds')
# The frozen v064-r2 thresholds; a value within the tolerance of a bound counts as equal to it (inside a closed bound, outside an open one).
THRESHOLDS = {'NON-COLLAPSED': {'loop_max': .30, 'submit_min': .80}, 'COLLAPSED': {'loop_above': .40, 'submit_below': .70}, 'boundary_abs_tolerance': 1e-12}
DOT = chr(0xb7)  # the middle dot of "9" + DOT + "n_e" in the pre-registration
# The pre-written sentences with the pre-registration's own symbols; the analysis refuses to run unless each of them is in the pre-registration file.
E1 = ('the termination contrast was also observed on an n_c-task external panel (A_onpol training seeds 1 and 2: loop a1 and a2, submit b1 and b2; '
      'T_sub, one training seed: loop c, submit d)')
E2 = ('no external contrast: the teacher subset did not meet the collapse criteria on the external panel (loop c, submit d; label L), while both A_onpol '
      'training seeds did not either (loop a1 and a2, submit b1 and b2)')
E3 = "the self-generated arm's non-collapse was not observed externally in training seed k (loop a, submit b; label L)"
# The Readings form is the only result sentence of a state without a cell. The Time-box sentence of the pre-registration ("... not completed before the
# result freeze") is the paper's rule for a report that was never sealed by the freeze; an analysis that ran cannot print it, so it is not here.
INCOMPLETE = 'incomplete (m of 9' + DOT + 'n_e rollouts; n_c of n_e complete tasks)'
LABEL_RULE = ('NON-COLLAPSED = loop ' + chr(0x2264) + ' 0.30 ' + chr(0x2227) + ' submit ' + chr(0x2265) + ' 0.80; COLLAPSED = loop > 0.40 ' + chr(0x2228) +
              ' submit < 0.70; otherwise INTERMEDIATE')
BOOTSTRAP_RULE = '20,000 draws, seed 20261103'
COMPLETE_RULE = 'The planned paired set is complete iff n_c = n_e and no row was excluded'
FROZEN_FORMS = (E1, E2, E3, INCOMPLETE, LABEL_RULE, BOOTSTRAP_RULE, COMPLETE_RULE)
NO_CLAIM = ('A panel statement only: no hidden-set estimate, no generalisation beyond these repositories, no mechanism, no pooling across seeds or arms; with '
            'small n_c an absence of contrast or repair is not evidence of absence. A termination cell is a statement about termination behaviour only and '
            'never implies repair transfer.',)
FORBIDDEN_WORDS = ('prevents collapse', 'on-policy learning', 'improved', 'equivalent', 'the contrast replicated', 'causes')
# The A0mc-ext comparator is a technical reference only (decision of the user, note 32f7625a, recorded 04:06Z before any attempt-2 outcome; wording of the lead).
# The label stands in report.md and report.json whether or not a comparator is shown. No sentence of an output other than the label itself contains one of the
# claim words (the label names the nouns to deny them). These words are not in the frozen pre-registration (FORBIDDEN_WORDS is), so they are a list of their own.
# The limitations are recorded in every report (same decision).
COMPARATOR_FORBIDDEN_WORDS = ('improved', 'improvement', 'recovered', 'recovery', 'recovers', 'non-inferior', 'equivalent', 'equivalence')
COMPARATOR_LABEL = 'A0mc-ext is withheld: v068 had an invalid submit path; no reused comparator rows or confidence intervals are reported.'
LIMITATIONS = ('Execution order: v070 ran after v068 attempts 1 and 2 on the same environment trees.',
               "Cache accumulation: bytecode caches written by the earlier attempts were present at v070's start; their differences from attempt 2's start are listed under (k).")
# The two pre-written rules for the cache state (v068 change record, section 7). The analysis manifest holds the one that applies, fixed at --prepare.
CACHE_RULES = {
    'k-disclose': 'cache state is not an identity leaf: the cache files that differ between the starting cache list of v070 and that of v068 attempt 2 are '
                  'disclosed as declared difference (k) and the comparator follows the frozen identity leaves only',
    'k-strict': 'the comparator is shown only if the starting cache list of v070 equals that of v068 attempt 2 (path, sha256); otherwise no comparator is shown '
                'and the differing cache files are listed'}
DEFAULT_CACHE_RULE = 'k-disclose'
DECLARED_DIFFERENCE_K = ('(k) v070 starts with the bytecode caches present after v068 attempt 2, so its starting cache list can differ from the starting list of '
                         'attempt 2. Python reads .pyc files: a module whose cache changed may load from the cache instead of being compiled from source. Leaving '
                         'cache files out of the dependency-tree hash is not a claim that caches cannot affect execution.')


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def verify_preregistration(doc):
    """The sentences and rules in this file are the ones in the frozen pre-registration file whose hash the contract pins."""
    pinned = doc['preregistration']
    runner.require(pinned == dict(path=str(runner.PREREG), sha256=runner.PREREG_SHA) and runner.sha(pinned['path']) == runner.PREREG_SHA,
                   'The pre-registration is not the frozen v070 file')
    text = Path(pinned['path']).read_text(encoding='utf-8')
    for form in FROZEN_FORMS:
        runner.require(form in text, 'Analysis sentence differs from the pre-registration: ' + form[:60])


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def termination(loop, submit):
    """The v064-r2 label of one arm from its pooled loop and submit rates: NON-COLLAPSED (loop <= .30 and submit >= .80), COLLAPSED (loop > .40 or
    submit < .70), otherwise INTERMEDIATE; UNDEFINED when a rate is not finite."""
    close = lambda x, y: math.isclose(x, y, rel_tol=0, abs_tol=THRESHOLDS['boundary_abs_tolerance'])
    nc, c = THRESHOLDS['NON-COLLAPSED'], THRESHOLDS['COLLAPSED']
    if not (finite(loop) and finite(submit)):
        return 'UNDEFINED'
    if (loop < nc['loop_max'] or close(loop, nc['loop_max'])) and (submit > nc['submit_min'] or close(submit, nc['submit_min'])):
        return 'NON-COLLAPSED'
    if (loop > c['loop_above'] and not close(loop, c['loop_above'])) or (submit < c['submit_below'] and not close(submit, c['submit_below'])):
        return 'COLLAPSED'
    return 'INTERMEDIATE'


def rate(value):
    return '%.3f' % value


def cell(labels, rates, n):
    """The one pre-written cell that applies when the planned paired set is complete. labels: arm -> label; rates: arm -> (loop, submit); n = n_c.
    E3 when an A_onpol seed is not NON-COLLAPSED (one clause per such seed, T_sub's values beside them), else E1 when T_sub is COLLAPSED, else E2."""
    first, second, teacher = rates['A_onpol_ts1'], rates['A_onpol_ts2'], rates['T_sub']
    beside = 'T_sub, one training seed: loop %s, submit %s, label %s' % (rate(teacher[0]), rate(teacher[1]), labels['T_sub'])
    bad = [(seed, arm) for seed, arm in ((1, 'A_onpol_ts1'), (2, 'A_onpol_ts2')) if labels[arm] != 'NON-COLLAPSED']
    if bad:
        clauses = ["the self-generated arm's non-collapse was not observed externally in training seed %d (loop %s, submit %s; label %s)" %
                   (seed, rate(rates[arm][0]), rate(rates[arm][1]), labels[arm]) for seed, arm in bad]
        return dict(cell='E3', sentence='; '.join(clauses + [beside]), tasks=n)
    seeds = 'A_onpol training seeds 1 and 2: loop %s and %s, submit %s and %s' % (rate(first[0]), rate(second[0]), rate(first[1]), rate(second[1]))
    if labels['T_sub'] == 'COLLAPSED':
        return dict(cell='E1', tasks=n, sentence='the termination contrast was also observed on an %d-task external panel (%s; T_sub, one training seed: '
                                                 'loop %s, submit %s)' % (n, seeds, rate(teacher[0]), rate(teacher[1])))
    return dict(cell='E2', tasks=n, sentence='no external contrast: the teacher subset did not meet the collapse criteria on the external panel (loop %s, '
                                             'submit %s; label %s), while both A_onpol training seeds did not either (loop %s and %s, submit %s and %s)' %
                                             (rate(teacher[0]), rate(teacher[1]), labels['T_sub'], rate(first[0]), rate(second[0]), rate(first[1]), rate(second[1])))


def row_of(task, run, item):
    """One completed invocation as a flat row: the harness result, the guard counts and the routing counts of its receipt."""
    receipt, result = item['receipt'], item['rows'][task['instance_id']]
    guard, routing = receipt['memory_guard'], receipt['routing']
    return dict(**{key: result[key] for key in ROW_FIELDS}, guard_command_count=guard['command_count'],
                guard_oom_command_count=guard['oom_command_count'], successful_requests=routing['successful_requests'],
                empty_actions=routing['empty_actions'], request_timeouts=routing['request_timeouts'],
                submission_diagnostics=receipt['git_setup']['submission'])


def state_of(item, task):
    """completed / excluded / not_run of one invocation from the qualified run (a run that never started has no receipt)."""
    receipt = None if item is None else item['receipt']
    if receipt is None:
        return 'not_run'
    states = receipt['states']
    return next(name for name in ('completed', 'excluded', 'not_run') if task['instance_id'] in states[name])


def per_task_rows(frozen, runs):
    """Every planned invocation of the panel runs, completed or not, in run then frozen task order. Training seed and decoding seed are separate fields."""
    family = {arm['arm']: arm['family'] for arm in frozen['arms']}
    rows = []
    for run in frozen['runs']:
        item = runs[run['name']]
        for task in frozen['tasks']:
            state = state_of(item, task)
            base = dict(instance_id=task['instance_id'], rank=task['rank'], repository=task['repository'], arm=run['arm'], family=family[run['arm']],
                        training_seed=run['training_seed'], decoding_seed=run['decoding_seed'], state=state)
            if state == 'completed':
                rows.append(dict(base, **row_of(task, run, item['invocations'][task['instance_id']])))
            elif state == 'excluded':
                record = item['exclusions'][task['instance_id']]['record']
                rows.append(dict(base, exclusion_kind=record['kind'], exclusion_reason=record['reason']))
            else:
                rows.append(base)
    return rows


def complete_tasks(frozen, rows):
    """T_c: the tasks with a completed row for every v070 arm at all three decoding seeds, in frozen order."""
    done = {(row['instance_id'], row['arm'], row['decoding_seed']) for row in rows if row['state'] == 'completed'}
    arms = [arm['arm'] for arm in frozen['arms']]
    return [task['instance_id'] for task in frozen['tasks'] if all((task['instance_id'], arm, seed) in done for arm in arms for seed in runner.SEEDS)]


def pooled(frozen, rows, complete):
    """Repair, always reported: per arm over T_c the resolved count of 3 n_c and its decoding seeds, the tasks resolved at least once; and over every row the
    states and the harness errors."""
    result = {}
    for arm in frozen['arms']:
        mine = [row for row in rows if row['arm'] == arm['arm']]
        inside = [row for row in mine if row['instance_id'] in complete and row['state'] == 'completed']
        per_task = Counter()
        for row in inside:
            per_task[row['instance_id']] += row['resolved']
        done = [row for row in mine if row['state'] == 'completed']
        result[arm['arm']] = dict(
            family=arm['family'], training_seed=arm['training_seed'],
            complete_tasks=dict(resolved=sum(row['resolved'] for row in inside), denominator=3 * len(complete),
                                per_decoding_seed={str(seed): sum(row['resolved'] for row in inside if row['decoding_seed'] == seed) for seed in runner.SEEDS},
                                tasks_resolved_at_least_once=sum(1 for count in per_task.values() if count > 0),
                                resolved_count_distribution=dict(sorted(Counter(per_task.values()).items()))),
            all_rows=dict(completed=len(done), excluded=sum(row['state'] == 'excluded' for row in mine), not_run=sum(row['state'] == 'not_run' for row in mine),
                          resolved_of_completed=sum(row['resolved'] for row in done), errors=dict(sorted(Counter(row['error'] or 'none' for row in done).items())),
                          guard_oom_command_count=sum(row['guard_oom_command_count'] for row in done)))
    return result


def canary_rows(frozen, runs):
    """Reported apart from the panel rows: the canary is a panel task and never replaces a panel row. A canary that did not complete is listed as such."""
    task, rows = frozen['tasks'][0], []
    for run in frozen['canaries']:
        item = runs[run['name']]
        base = dict(instance_id=task['instance_id'], arm=run['arm'], training_seed=run['training_seed'], decoding_seed=run['decoding_seed'])
        if state_of(item, task) == 'completed':
            rows.append(dict(base, state='completed', **row_of(task, run, item['invocations'][task['instance_id']])))
        else:
            rows.append(dict(base, state=state_of(item, task)))
    return rows


def outcomes(rows, tasks, arm):
    """Resolved flags of one arm over a task list: for every task the three decoding seeds in seed order."""
    flags = {(row['instance_id'], row['decoding_seed']): row['resolved'] for row in rows if row['arm'] == arm and row['state'] == 'completed'}
    return [[int(flags[(task, seed)]) for seed in runner.SEEDS] for task in tasks]


def mean_of(values):
    return None if any(value is None for value in values) else sum(values) / len(values)


def descriptive(frozen, complete, producer):
    """Loop, submit, M5 (error stratum) and think rate, as the Rich analyses read them from the v032 producer, over T_c: per run and as three-seed means."""
    ids, result = frozenset(complete), {}
    for arm in (arm['arm'] for arm in frozen['arms']):
        runs = []
        for run in (run for run in frozen['runs'] if run['arm'] == arm):
            summary = producer.analyze_run(Path(run['result_dir']), ids)[0]
            error = summary['nonrecovery']['error']
            runs.append(dict(run=run['name'], decoding_seed=run['decoding_seed'], tasks=summary['n_tasks'], missing_traces=summary['missing_traces'],
                             think=summary['think_rate_mean'], loop=summary['loop_rate'], submit=summary['submit_rate'],
                             repro=summary['opener_shares']['repro_script_opener'], m5_error=error['mean_repeat_rate'], m5_error_tasks=error['n_tasks']))
        result[arm] = dict(runs=runs, three_seed_mean={key: mean_of([item[key] for item in runs]) for key in ('think', 'loop', 'submit', 'repro', 'm5_error')})
    return result


def sealed_v068(doc):
    raise RuntimeError('v068 comparator reads are forbidden by the Git-repair amendment')


def cache_state(doc, rule):
    """The cache state that the pin step recorded (this pin's starting cache list against that of v068 attempt 2), with the rule of the analysis manifest. The
    difference file is read only if its sha256 is the one in the contract; environments without a difference are left out of the listing."""
    state = doc['pin']['cache_state']
    path = Path(state['difference']['path'])
    runner.require(path.is_file() and runner.sha(path) == state['difference']['sha256'], 'The cache-state difference file is missing or changed')
    listed = [item for item in json.loads(path.read_text())['environments'] if any(item[key] for key in ('added', 'deleted', 'changed', 'mtime_only'))]
    return dict(rule=rule, rule_text=CACHE_RULES[rule], declared_difference=DECLARED_DIFFERENCE_K, attempt2_ended_at_pin=state['attempt2_ended_at_pin'],
                equal_path_sha256=state['equal_path_sha256'], totals=state['totals'], difference=state['difference'], environments=listed)


def comparator(doc, arms, rows, complete, resolution, cache):
    return dict(shown=False, withheld=True, reason='v068 submit path was infrastructure-invalid; withheld by the amendment')


def termination_table(frozen, described):
    """Per arm the pooled loop and submit over T_c (three-seed means) and the label; training seed and decoding seeds are different columns."""
    table = {}
    for arm in frozen['arms']:
        mean = described[arm['arm']]['three_seed_mean']
        table[arm['arm']] = dict(family=arm['family'], training_seed=arm['training_seed'], loop=mean['loop'], submit=mean['submit'],
                                 label=termination(mean['loop'], mean['submit']), think=mean['think'], m5_error=mean['m5_error'], repro=mean['repro'])
    return table


def readings(frozen, rows, complete, completed, described, sealed_at):
    """The pre-registered readings. The planned paired set is complete iff n_c = n_e and no row was excluded (and the generation did not stop); only then,
    and only if the report is sealed before the result freeze, does a cell apply. Otherwise the result is the Readings form "incomplete (m of 9 n_e rollouts;
    n_c of n_e complete tasks)": after a stop or an exclusion (kind incomplete), and for a complete panel sealed at or after the freeze (kind late). The
    termination table over T_c is then descriptive and no cell sentence is a claim. The Time-box sentence belongs to the paper, for a report that was never
    sealed by the freeze; this analysis ran, so it never prints it."""
    n, total = len(complete), len(frozen['tasks'])
    planned = len(frozen['runs']) * total
    done = sum(row['state'] == 'completed' for row in rows)
    clean = n == total and not any(row['state'] == 'excluded' for row in rows) and completed['stop'] is None
    before = sealed_at < runner.FREEZE_UTC
    base = dict(complete_tasks=complete, n_c=n, n_e=total, planned_rollouts=planned, completed_rollouts=done, planned_paired_set_complete=clean,
                freeze=dict(utc=runner.FREEZE_UTC, sealed_at=sealed_at, before_freeze=before), termination=None, cell=None)
    if n:
        base['termination'] = termination_table(frozen, described)
    short = 'incomplete (%d of %d rollouts; %d of %d complete tasks)' % (done, planned, n, total)
    if not clean:
        return dict(base, kind='incomplete', sentence=short)
    if not before:
        return dict(base, kind='late', sentence=short)
    table = base['termination']
    undefined = [arm for arm, item in table.items() if item['label'] == 'UNDEFINED']
    if undefined:
        return dict(base, kind='undefined', sentence='no cell applies: the loop or submit rate is undefined for ' + ', '.join(undefined))
    chosen = cell({arm: item['label'] for arm, item in table.items()}, {arm: (item['loop'], item['submit']) for arm, item in table.items()}, n)
    return dict(base, kind='cell', sentence=chosen['sentence'], cell=chosen['cell'])


def feasibility_record():
    """The projection the runner logged after the canaries, as it was logged (descriptive), or None."""
    path = runner.EVIDENCE / 'feasibility.json'
    return json.loads(path.read_text()) if path.is_file() else None


def analyze(runtime_sha, rule=DEFAULT_CACHE_RULE):
    doc, frozen, completed, runs = runner.qualify_generation(runtime_sha)
    runner.require(rule in CACHE_RULES, 'Unknown cache rule: ' + str(rule))
    verify_preregistration(doc)
    cache = cache_state(doc, rule)
    rows = per_task_rows(frozen, runs)
    runner.require(len(rows) == len(frozen['runs']) * len(frozen['tasks']), 'Per-task row count differs from runs x eligible tasks')
    complete = complete_tasks(frozen, rows)
    described = descriptive(frozen, complete, runner.load_module(PRODUCER, PRODUCER_SHA)) if complete else None
    arms = [arm['arm'] for arm in frozen['arms']]
    record = runner.EVIDENCE / ('completed.json' if completed['status'] in (runner.STATUS_ALL, runner.STATUS_EXCLUSIONS) else 'stopped.json')
    if complete:
        beside = comparator(doc, arms, rows, complete, None, cache)
    else:
        beside = dict(shown=False, reason='the complete task set is empty')
    report = dict(schema='external-e4b-v070git2-report', runtime_sha256=runtime_sha, preregistration=doc['preregistration'], pin=doc['pin'], comparator_identity=doc['comparator'],
                  record=dict(file=record.name, sha256=runner.sha(record), status=completed['status'], stop=completed['stop'], excluded=completed['excluded']),
                  arms=[dict(arm=arm['arm'], alias=arm['alias'], family=arm['family'], training_seed=arm['training_seed']) for arm in frozen['arms']],
                  seeds=list(runner.SEEDS), eligible_tasks=[task['instance_id'] for task in frozen['tasks']], not_eligible=doc['panel']['not_eligible'],
                  canary_task=frozen['canary_task'], invocations=dict(Counter(row['state'] for row in rows), planned=len(rows)),
                  readings=readings(frozen, rows, complete, completed, described, utc_now()),
                  repair=dict(pooled=pooled(frozen, rows, complete), comparator=beside, comparator_label=COMPARATOR_LABEL), cache_state=cache, limitations=list(LIMITATIONS),
                  descriptive=described, canary=canary_rows(frozen, runs), feasibility=feasibility_record(), not_claimed=list(NO_CLAIM),
                  amendment=dict(path=str(runner.AMENDMENT), sha256=runner.AMENDMENT_SHA),
                  metric_definition=dict(submit='parsed submit_patch attempt', successful_capture='separate descriptive diagnostic'),
                  statement='primary forms retained under the Git-repair amendment; repair descriptive, A0mc withheld')
    return rows, report


def number(value):
    return 'n/a' if value is None else '%.3f' % value


def markdown(report):
    reading, repair, pin = report['readings'], report['repair'], report['pin']
    stop = report['record']['stop']
    lines = ['# External-repository panel v070 (termination contrast): analysis', '',
             'Generation: %s (stop: %s). Eligible tasks n_e = %d, complete tasks n_c = %d, completed rollouts %d of %d. Arms: %s.' % (
                 report['record']['status'], stop and stop['rule'], reading['n_e'], reading['n_c'], reading['completed_rollouts'], reading['planned_rollouts'],
                 ', '.join('%s (%s, training seed %d)' % (item['arm'], item['family'], item['training_seed']) for item in report['arms'])), '',
             '## Reading', '', reading['sentence'], '']
    lines += ['Cell %s applies: the planned paired set is complete and the report is sealed before the result freeze (%s).' % (reading['cell'], reading['freeze']['utc'])
              if reading['kind'] == 'cell' else 'No cell sentence is a claim here (%s). Planned paired set complete: %s; sealed at %s, result freeze %s.' % (
                  reading['kind'], reading['planned_paired_set_complete'], reading['freeze']['sealed_at'], reading['freeze']['utc']), '']
    if reading['termination']:
        lines += ['## Termination over the complete tasks (%s)' % ('reading' if reading['kind'] == 'cell' else 'descriptive'), '',
                  '| arm | family | training seed | loop | submit attempt | label | think | M5 error |', '|---|---|---|---|---|---|---|---|']
        lines += ['| %s | %s | %d | %s | %s | %s | %s | %s |' % (arm, item['family'], item['training_seed'], number(item['loop']), number(item['submit']),
                                                              item['label'], number(item['think']), number(item['m5_error']))
                  for arm, item in reading['termination'].items()] + ['']
    lines += ['## Repair, always reported (pooled resolved over the complete tasks, per decoding seed)', '', '| arm | training seed | resolved | of | decoding seeds 1, 2, 3 |',
              '|---|---|---|---|---|']
    lines += ['| %s | %d | %d | %d | %s |' % (arm, item['training_seed'], item['complete_tasks']['resolved'], item['complete_tasks']['denominator'],
                                              ', '.join(str(item['complete_tasks']['per_decoding_seed'][str(seed)]) for seed in report['seeds']))
              for arm, item in repair['pooled'].items()] + ['']
    beside = repair['comparator']
    lines += ['**%s**' % repair['comparator_label'], '']
    if beside['shown']:
        lines += ['A0mc-ext of v068 on the %d tasks of the complete set that have a completed A0mc-ext row at all three decoding seeds (left out: %s); '
                  'descriptive paired task-bootstrap, %d draws, seed %d, no wording rule:' % (beside['n_tasks'], ', '.join(beside['left_out']) or 'none', DRAWS, SEED), '']
        lines += ['- %s: %d/%d vs A0mc-ext %d/%d, %+.2f pp [%+.2f, %+.2f]' % (arm, item['resolved'], item['denominator'], item['a0mc_resolved'], item['denominator'],
                                                                           item['effect_pp'], item['ci95_pp'][0], item['ci95_pp'][1]) for arm, item in beside['arms'].items()] + ['']
    else:
        lines += ['No A0mc-ext comparator is shown: %s%s.' % (beside['reason'], ' (' + ', '.join(beside['differing_leaves']) + ')' if beside.get('differing_leaves') else ''), '']
    cache = report['cache_state']
    totals = cache['totals']
    lines += ['## Cache state (declared difference (k))', '', 'Rule fixed in the analysis manifest: %s (%s).' % (cache['rule'], cache['rule_text']), '', cache['declared_difference'], '',
              'Starting cache lists of v070 and of v068 attempt 2 (path, sha256): %s. Added %d, removed %d, changed %d; same content with another modification time %d.%s' % (
                  'equal' if cache['equal_path_sha256'] else 'different', totals['added'], totals['deleted'], totals['changed'], totals['mtime_only'],
                  '' if cache['attempt2_ended_at_pin'] else ' The pin ran before attempt 2 had ended, so its list is not the state at the start of v070.')]
    for item in cache['environments']:
        parts = (('added', ['%s (sha256 %s)' % (entry['path'], entry['sha256'][:12]) for entry in item['added']]),
                 ('removed', ['%s (sha256 %s)' % (entry['path'], entry['sha256'][:12]) for entry in item['deleted']]),
                 ('changed', ['%s (attempt 2 sha256 %s, v070 sha256 %s)' % (entry['path'], entry['v068_sha256'][:12], entry['v070_sha256'][:12]) for entry in item['changed']]),
                 ('modification time only', item['mtime_only']))
        lines += ['- %s: %s' % (item['environment_id'], '; '.join('%s %s' % (name, ', '.join(paths)) for name, paths in parts if paths))]
    lines += ['', '## Pin record', '', '- v068 contract (r21, identity leaves): %s' % pin['v068_runtime']['sha256'],
              '- v068 attempt-2 contract (r22): %s' % pin['v068_r22_runtime']['sha256'], '- comparator identity: %s' % (
              'withheld by amendment; identity mismatches: ' + (', '.join(report['comparator_identity']['differing']) or 'none'))]
    lines += ['- %s: adapter %s' % (item['arm'], item['adapter_model_sha256']) for item in pin['adapters']]
    lines += ['- %s: %d of %d training repositories are panel repositories%s' % (item['arm'], item['n_overlap'], item['n_training_repositories'],
                                                                              ': ' + ', '.join(item['overlap']) if item['overlap'] else '') for item in pin['training_overlap']] + ['']
    lines += ['## Rows that are not completed', '']
    lines += ['- not eligible at the pin: %s' % ', '.join('%s (%s)' % (item['instance_id'], item['status']) for item in report['not_eligible'])] if report['not_eligible'] else []
    lines += ['- excluded: %s' % json.dumps(report['record']['excluded'], sort_keys=True), '- invocations: %s' % json.dumps(report['invocations'], sort_keys=True), '']
    lines += ['## Canary rows (apart, never in the readings)', ''] + ['- %s decoding seed %d: %s' % (row['arm'], row['decoding_seed'], row['state']) for row in report['canary']]
    if report['feasibility']:
        lines += ['', '## Feasibility projection logged after the canaries (descriptive)', '', json.dumps(report['feasibility'], sort_keys=True)]
    return chr(10).join(lines + ['', '## Limitations', ''] + ['- ' + text for text in report['limitations']] +
                        ['', '## Not claimed', ''] + ['- ' + text for text in report['not_claimed']]) + chr(10)


def write(rows, report):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    files = {'per_task.jsonl': ''.join(json.dumps(row, sort_keys=True) + chr(10) for row in rows),
             'report.json': json.dumps(report, indent=1, sort_keys=True) + chr(10), 'report.md': markdown(report)}
    for name, text in files.items():
        with (OUTPUT / name).open('x', newline=chr(10)) as stream:
            stream.write(text)
    hashes = {name: runner.sha(OUTPUT / name) for name in files}
    for name, digest in hashes.items():
        with (OUTPUT / (name + '.sha256')).open('x', newline=chr(10)) as stream:
            stream.write(digest + '  ' + name + chr(10))
    return hashes


def manifest_doc(runtime_sha=None, cache_rule=DEFAULT_CACHE_RULE):
    """The analysis contract: the plan of the analysis fixed before the data are read. Without a runtime sha it is the staged draft with the runtime binding
    pending; with the sha of the pinned runtime contract it is the pinned manifest bound to that contract. cache_rule is the rule for the cache state that
    applies (k-disclose or k-strict, change record section 7), fixed here and read back from the manifest by the analysis."""
    runner.require(cache_rule in CACHE_RULES, 'Unknown cache rule: ' + str(cache_rule))
    pinned = runtime_sha is not None
    if pinned:
        runner.require(runner.sha(runner.RUNTIME) == runtime_sha and Path(str(runner.RUNTIME) + '.sha256').read_text().split()[0] == runtime_sha,
                       'v070 runtime sidecar mismatch')
    paths = [Path(__file__).resolve(), runner.ROOT / 'tools/test_analyze_external_e4b_v070.py', runner.ROOT / 'tools/run_external_e4b_v070.py',
             runner.ROOT / 'tools/test_run_external_e4b_v070.py', PRODUCER, RESOLUTION, runner.PREREG]
    paths = [Path(__file__).resolve(), runner.WORK / 'analysis/test_v070git2_analysis.py',
             runner.WORK / 'runner/run_external_e4b_v070git2.py', runner.WORK / 'runner/test_v070git2_runtime.py', PRODUCER, RESOLUTION, runner.PREREG, runner.AMENDMENT]
    pins = {str(path): runner.sha(path) for path in paths}
    runner.require(pins[str(PRODUCER)] == PRODUCER_SHA and pins[str(RESOLUTION)] == RESOLUTION_SHA and pins[str(runner.PREREG)] == runner.PREREG_SHA,
                   'A pre-registered analysis input hash differs')
    verify_preregistration(dict(preregistration=dict(path=str(runner.PREREG), sha256=runner.PREREG_SHA)))
    return dict(version='external-e4b-v070git2-analysis', generation=runner.GENERATION, status='PINNED' if pinned else 'DRAFT_PREPARED_ONLY', root=str(runner.ROOT),
                preregistration=dict(path=str(runner.PREREG), sha256=runner.PREREG_SHA, verbatim=list(FROZEN_FORMS)),
                runtime=dict(contract=str(runner.RUNTIME), sha256=runtime_sha if pinned else runner.PENDING_PIN, draft=str(runner.RUNTIME_DRAFT)),
                pins=pins, thresholds=THRESHOLDS, bootstrap=dict(draws=DRAWS, seed=SEED), result_freeze_utc=runner.FREEZE_UTC, output=str(OUTPUT),
                identity_leaves=[name for name, _ in runner.IDENTITY_LEAVES],
                amendment=dict(path=str(runner.AMENDMENT), sha256=runner.AMENDMENT_SHA),
                comparator=dict(arm=COMPARATOR_ARM, withheld=True, shown=False, reason='v068 path invalid; no reused rows or CIs'),
                cache_rule=cache_rule, cache_rules=CACHE_RULES, declared_difference_k=DECLARED_DIFFERENCE_K, comparator_label=COMPARATOR_LABEL, comparator_forbidden_words=list(COMPARATOR_FORBIDDEN_WORDS), limitations=list(LIMITATIONS),
                forbidden_words=list(FORBIDDEN_WORDS), execution_authorized=False,
                requires='A qualified v070 generation (finished or stopped by a rule) and the explicit --runtime-sha and --analysis-sha of the pinned contracts.')


def manifest_text(doc):
    return json.dumps(doc, indent=2, sort_keys=True) + chr(10)


def prepare(runtime_sha=None, cache_rule=DEFAULT_CACHE_RULE):
    doc = manifest_doc(runtime_sha, cache_rule)
    target = MANIFEST if runtime_sha is not None else MANIFEST_DRAFT
    runner.write_or_verify(target, manifest_text(doc))
    runner.write_or_verify(Path(str(target) + '.sha256'), runner.sha(target) + '  ' + target.name + chr(10))
    return runner.sha(target)


def verify_manifest(analysis_sha, runtime_sha):
    """The pinned analysis manifest: sidecar, binding to the pinned runtime contract, and the manifest re-derived from the current files with the cache rule
    that it holds. Returns that rule."""
    runner.require(runner.sha(MANIFEST) == analysis_sha and Path(str(MANIFEST) + '.sha256').read_text().split()[0] == analysis_sha, 'Analysis manifest sidecar mismatch')
    rule = json.loads(MANIFEST.read_text()).get('cache_rule')
    runner.require(rule in CACHE_RULES, 'The analysis manifest holds no known cache rule')
    runner.require(MANIFEST.read_text() == manifest_text(manifest_doc(runtime_sha, rule)), 'The analysis manifest does not re-derive from the current files')
    return rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--runtime-sha')
    parser.add_argument('--analysis-sha')
    parser.add_argument('--cache-rule', choices=sorted(CACHE_RULES), help='with --prepare: the rule for the cache state that the manifest fixes (default ' + DEFAULT_CACHE_RULE + ')')
    args = parser.parse_args()
    if args.prepare:
        rule = args.cache_rule or DEFAULT_CACHE_RULE
        print(json.dumps(dict(status='prepared_only', analysis_sha256=prepare(args.runtime_sha, rule), cache_rule=rule, gpu_operations=False)))
        return
    runner.require(args.cache_rule is None, 'The cache rule is fixed in the analysis manifest by --prepare')
    runner.require(args.runtime_sha and args.analysis_sha, 'The analysis needs --runtime-sha and --analysis-sha of the pinned contracts')
    rule = verify_manifest(args.analysis_sha, args.runtime_sha)
    rows, report = analyze(args.runtime_sha, rule)
    hashes = write(rows, report)
    print(json.dumps(dict(status='analysis_written', output=str(OUTPUT), hashes=hashes, cache_rule=rule, sentence=report['readings']['sentence'])))


if __name__ == '__main__':
    main()
