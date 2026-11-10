"""External-panel v068 (revision r22), analysis of one qualified generation with the readings of the frozen pre-registration 8061b917: the pooled counts
over the complete task set T_c, the four pre-written cells on (z = A0mc, x = M1q, y = M0q) with M1q-s2 read beside M1q, the paired task-bootstrap
intervals for M1q - M0q and M1q - A0mc, and the descriptive loop, submit, M5 and think rates. Every row is reported, completed or not."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import run_external_e4b_v068_r22 as runner

OUTPUT = runner.ROOT / 'analysis/external-e4b-v068'
PRODUCER, PRODUCER_SHA = runner.ROOT / 'tools/mechanism_metrics_v032.py', '36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d'
RESOLUTION, RESOLUTION_SHA = runner.ROOT / 'tools/analyze_contract_v02_v025.py', '20208c74cc921d9dd4975b1b6a7129b6a4b55825f17930429a4fa624935e229c'
DRAWS, SEED = 20000, 20261103
MIN_COMPLETE = 4
Z_ARM, X_ARM, Y_ARM, X2_ARM = 'A0mc', 'M1q', 'M0q', 'M1qs2'
ROW_FIELDS = ('resolved', 'error', 'test_exit_code', 'tool_calls', 'total_llm_calls', 'agent_patch_size', 'duration_seconds')
# The pre-written sentences of the pre-registration. {D} = 3 n, {n} = the number of tasks, {x}, {y}, {z} = the pooled resolved counts. FROZEN_FORMS
# are the same sentences with the pre-registration's own symbols; the analysis refuses to run unless each of them is in the pre-registration file.
CELLS = (
    ('z = 0', 'on the external panel the base resolved nothing (0/{D}), so the panel cannot inform the trained arms (M1q {x}, M0q {y})'),
    ('z > 0, x = 0', "M1q's repair was not observed on the external panel (0/{D}; base {z}/{D}; unmasked control {y}/{D})"),
    ('z > 0, x > 0, y = 0', "M1q's repair was also observed on an {n}-task external panel ({x}/{D} vs 0/{D} for the unmasked control; base {z}/{D})"),
    ('z > 0, x > 0, y > 0', 'both trained arms resolved external tasks (M1q {x}/{D}, unmasked control {y}/{D}; base {z}/{D})'))
INCOMPLETE = 'stopped/excluded, {c} of {e} tasks'
TOO_FEW = 'fewer than 4 complete tasks ({c} of {e})'
SYMBOLS = dict(D='3n_e', n='n_e', x='x', y='y', z='z')
FROZEN_FORMS = tuple(template.format(**SYMBOLS) for _, template in CELLS) + (
    INCOMPLETE.format(c='n_c', e='n_e'), TOO_FEW.format(c='n_c', e='n_e'), 'lower > 0 ' + chr(0x2192) + ' "more than", upper < 0 ' + chr(0x2192) + ' "fewer than", otherwise "uncertain"',
    '20,000 resamples, seed 20261103')
NOT_CLAIMED = (
    'A panel statement only: never a hidden-set estimate; no generalisation beyond these repositories; no mechanism; no pooling across seeds or arms; '
    'no recovery, restores or causes wording; with small n_e an absence of repair is not evidence of absence.',
    'After this read DEV14 is relabelled as a main-track development set (declared exposure); CONFIRM14 stays sealed.')


def verify_preregistration(doc):
    """The sentences and rules in this file are the ones in the frozen pre-registration file whose hash the contract pins."""
    pinned = doc['preregistration']
    runner.require(pinned == dict(path=str(runner.PREREG), sha256=runner.PREREG_SHA) and runner.sha(pinned['path']) == runner.PREREG_SHA,
                   'The pre-registration is not the frozen v068 file')
    text = Path(pinned['path']).read_text(encoding='utf-8')
    for form in FROZEN_FORMS:
        runner.require(form in text, 'Analysis sentence differs from the pre-registration: ' + form[:60])


def cell(z, x, y, n):
    """The one pre-written cell that applies: z = A0mc, x = M1q (or M1q-s2 beside it), y = M0q, pooled resolved counts over n tasks x 3 seeds."""
    index = 0 if z == 0 else 1 if x == 0 else 2 if y == 0 else 3
    label, template = CELLS[index]
    return dict(cell=label, sentence=template.format(D=3 * n, n=n, x=x, y=y, z=z), counts=dict(z=z, x=x, y=y), denominator=3 * n, tasks=n)


def reading_word(interval):
    """more than / fewer than / uncertain from the 95 % interval (prereg: lower > 0, upper < 0, otherwise)."""
    lower, upper = interval
    return 'more than' if lower > 0 else 'fewer than' if upper < 0 else 'uncertain'


def row_of(task, run, item):
    """One completed invocation as a flat row: the harness result, the guard counts and the routing counts of its receipt."""
    receipt, result = item['receipt'], item['rows'][task['instance_id']]
    guard, routing = receipt['memory_guard'], receipt['routing']
    return dict(instance_id=task['instance_id'], rank=task['rank'], repository=task['repository'], arm=run['arm'], decoding_seed=run['decoding_seed'],
                **{key: result[key] for key in ROW_FIELDS}, guard_command_count=guard['command_count'],
                guard_oom_command_count=guard['oom_command_count'], successful_requests=routing['successful_requests'],
                empty_actions=routing['empty_actions'], request_timeouts=routing['request_timeouts'])


def state_of(item, task):
    """completed / excluded / not_run of one invocation from the qualified run (a run that never started has no receipt)."""
    receipt = None if item is None else item['receipt']
    if receipt is None:
        return 'not_run'
    states = receipt['states']
    return next(name for name in ('completed', 'excluded', 'not_run') if task['instance_id'] in states[name])


def per_task_rows(frozen, runs):
    """Every planned invocation of the panel runs, completed or not, in run then frozen task order."""
    rows = []
    for run in frozen['runs']:
        item = runs[run['name']]
        for task in frozen['tasks']:
            state = state_of(item, task)
            base = dict(instance_id=task['instance_id'], rank=task['rank'], repository=task['repository'], arm=run['arm'],
                        decoding_seed=run['decoding_seed'], state=state)
            if state == 'completed':
                rows.append(dict(base, **row_of(task, run, item['invocations'][task['instance_id']])))
            elif state == 'excluded':
                record = item['exclusions'][task['instance_id']]['record']
                rows.append(dict(base, exclusion_kind=record['kind'], exclusion_reason=record['reason']))
            else:
                rows.append(base)
    return rows


def complete_tasks(frozen, rows):
    """T_c: the tasks with a completed row for every present arm at all three decoding seeds, in frozen order."""
    done = {(row['instance_id'], row['arm'], row['decoding_seed']) for row in rows if row['state'] == 'completed'}
    arms = [arm['arm'] for arm in frozen['arms']]
    return [task['instance_id'] for task in frozen['tasks'] if all((task['instance_id'], arm, seed) in done for arm in arms for seed in runner.SEEDS)]


def pooled(frozen, rows, complete):
    """Per arm over T_c: resolved of 3 n_c, per seed, tasks resolved at least once; and over every row: states and the harness errors."""
    result = {}
    for arm in (arm['arm'] for arm in frozen['arms']):
        mine = [row for row in rows if row['arm'] == arm]
        inside = [row for row in mine if row['instance_id'] in complete and row['state'] == 'completed']
        per_task = Counter()
        for row in inside:
            per_task[row['instance_id']] += row['resolved']
        done = [row for row in mine if row['state'] == 'completed']
        result[arm] = dict(
            complete_tasks=dict(resolved=sum(row['resolved'] for row in inside), denominator=3 * len(complete),
                                per_seed={str(seed): sum(row['resolved'] for row in inside if row['decoding_seed'] == seed) for seed in runner.SEEDS},
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
        if state_of(item, task) == 'completed':
            rows.append(dict(row_of(task, run, item['invocations'][task['instance_id']]), state='completed'))
        else:
            rows.append(dict(instance_id=task['instance_id'], arm=run['arm'], decoding_seed=run['decoding_seed'], state=state_of(item, task)))
    return rows


def outcomes(rows, complete, arm):
    """Resolved flags of one arm over T_c: for every task of T_c the three decoding seeds in seed order."""
    flags = {(row['instance_id'], row['decoding_seed']): row['resolved'] for row in rows if row['arm'] == arm and row['state'] == 'completed'}
    return [[int(flags[(task, seed)]) for seed in runner.SEEDS] for task in complete]


def contrast(rows, complete, treated, baseline, resolution):
    """Paired task bootstrap over T_c (the three seeds of a task stay together): 20,000 resamples, seed 20261103."""
    result = resolution.paired_bootstrap(outcomes(rows, complete, treated), outcomes(rows, complete, baseline), draws=DRAWS, seed=SEED)
    return dict(name=treated + ' - ' + baseline, treated=treated, baseline=baseline, tasks=len(complete), draws=DRAWS, seed=SEED,
                effect_pp=result['effect_pp'], ci95_pp=result['ci95_pp'], seed_effects_pp=result['seed_effects_pp'], reading=reading_word(result['ci95_pp']))


def readings(frozen, rows, complete, finished, resolution):
    """The pre-registered readings over T_c. A panel that is stopped or has an excluded row states it; fewer than 4 complete tasks give no cell and no interval."""
    n, total = len(complete), len(frozen['tasks'])
    arms = [arm['arm'] for arm in frozen['arms']]
    statement = None if finished else INCOMPLETE.format(c=n, e=total)
    base = dict(complete_tasks=complete, n_c=n, n_e=total, incomplete_statement=statement)
    if n < MIN_COMPLETE:
        return dict(base, sentence=TOO_FEW.format(c=n, e=total), cells=None, contrasts=None)
    count = lambda arm: sum(sum(flags) for flags in outcomes(rows, complete, arm))
    suffix = '' if statement is None else ' (' + statement + ')'
    z, x, y = count(Z_ARM), count(X_ARM), count(Y_ARM)
    primary = cell(z, x, y, n)
    beside = cell(z, count(X2_ARM), y, n) if X2_ARM in arms else None
    for item in (primary, beside):
        if item is not None:
            item['sentence'] += suffix
    return dict(base, sentence=primary['sentence'], cells=dict(primary=primary, m1qs2_beside=beside),
                contrasts=[contrast(rows, complete, X_ARM, Y_ARM, resolution), contrast(rows, complete, X_ARM, Z_ARM, resolution)])


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


def analyze(runtime_sha):
    doc, frozen, completed, runs = runner.qualify_generation(runtime_sha)
    verify_preregistration(doc)
    rows = per_task_rows(frozen, runs)
    runner.require(len(rows) == len(frozen['runs']) * len(frozen['tasks']), 'Per-task row count differs from runs x eligible tasks')
    complete = complete_tasks(frozen, rows)
    finished = completed['status'] == runner.STATUS_ALL
    record = runner.EVIDENCE / ('completed.json' if finished or completed['status'] == runner.STATUS_EXCLUSIONS else 'stopped.json')
    report = dict(schema='external-e4b-v068-r21-report', runtime_sha256=runtime_sha, preregistration=doc['preregistration'], pin=doc['pin'],
                  record=dict(file=record.name, sha256=runner.sha(record), status=completed['status'], stop=completed['stop'], excluded=completed['excluded']),
                  arms=[arm['arm'] for arm in frozen['arms']], seeds=list(runner.SEEDS), eligible_tasks=[task['instance_id'] for task in frozen['tasks']],
                  not_eligible=doc['panel']['not_eligible'], canary_task=frozen['canary_task'],
                  invocations=dict(Counter(row['state'] for row in rows), planned=len(rows)),
                  readings=readings(frozen, rows, complete, finished, runner.load_module(RESOLUTION, RESOLUTION_SHA)),
                  pooled=pooled(frozen, rows, complete),
                  descriptive=descriptive(frozen, complete, runner.load_module(PRODUCER, PRODUCER_SHA)) if complete else None,
                  canary=canary_rows(frozen, runs), not_claimed=list(NOT_CLAIMED),
                  statement='readings, contrasts and sentences are those of the frozen v068 pre-registration; descriptive rates are not contrasts')
    return rows, report


def markdown(report):
    reading, pin = report['readings'], report['pin']['m1qs2']
    lines = ['# External-repository panel v068 (M1q observation): analysis', '',
             'Generation: %s (stop: %s). Eligible tasks n_e = %d, complete tasks n_c = %d. Arms: %s. M1q-s2 %s at the pin.' % (
                 report['record']['status'], report['record']['stop'] and report['record']['stop']['rule'], reading['n_e'], reading['n_c'],
                 ', '.join(report['arms']), 'included' if pin['included'] else 'not included (gates_pass was not true)'), '',
             '## Reading', '', reading['sentence'], '']
    if reading['cells'] is not None:
        beside = reading['cells']['m1qs2_beside']
        lines += ['Cell: %s.' % reading['cells']['primary']['cell'], '']
        lines += ['Beside it, with x replaced by the M1q-s2 count (never pooled): %s' % beside['sentence'], ''] if beside is not None else []
        lines += ['## Paired task-bootstrap intervals (%d resamples, seed %d, over the complete tasks)' % (DRAWS, SEED), '']
        lines += ['- %s: %s (%+.2f pp [%+.2f, %+.2f])' % (item['name'], item['reading'], item['effect_pp'], item['ci95_pp'][0], item['ci95_pp'][1])
                  for item in reading['contrasts']] + ['']
    lines += ['## Pooled resolved over the complete tasks', '', '| arm | resolved | of | per seed |', '|---|---|---|---|']
    lines += ['| %s | %d | %d | %s |' % (arm, item['complete_tasks']['resolved'], item['complete_tasks']['denominator'], item['complete_tasks']['per_seed'])
              for arm, item in report['pooled'].items()]
    lines += ['', '## Rows that are not completed', '']
    lines += ['- not eligible at the pin: %s' % ', '.join('%s (%s)' % (item['instance_id'], item['status']) for item in report['not_eligible'])] if report['not_eligible'] else []
    lines += ['- excluded: %s' % json.dumps(report['record']['excluded'], sort_keys=True), '- invocations: %s' % json.dumps(report['invocations'], sort_keys=True), '']
    if report['descriptive']:
        lines += ['## Descriptive (three-seed means over the complete tasks)', '', '| arm | loop | submit | M5 error | think |', '|---|---|---|---|---|']
        lines += ['| %s | %s | %s | %s | %s |' % ((arm,) + tuple(item['three_seed_mean'][key] for key in ('loop', 'submit', 'm5_error', 'think')))
                  for arm, item in report['descriptive'].items()] + ['']
    lines += ['## Canary rows (apart, never in the readings)', ''] + ['- %s s%d: %s' % (row['arm'], row['decoding_seed'], row['state']) for row in report['canary']]
    return chr(10).join(lines + ['', '## Not claimed', ''] + ['- ' + text for text in report['not_claimed']]) + chr(10)


def write(rows, report):
    OUTPUT.mkdir(exist_ok=False)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-sha', required=True)
    args = parser.parse_args()
    rows, report = analyze(args.runtime_sha)
    print(json.dumps(dict(status='analyzed', written=write(rows, report), sentence=report['readings']['sentence']), indent=1))


if __name__ == '__main__':
    sys.exit(main())
