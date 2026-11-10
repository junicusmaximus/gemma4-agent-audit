"""v064 Stage 0 (build b): capped gold and empty controls for the 69 on-policy candidates, derived from the v056 capped controls runner (CPU only)."""
from __future__ import annotations

import argparse
import asyncio
import csv
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_temp_screen_31b_v056 as screen

ROOT = screen.ROOT
CONTROLS = ROOT / 'controls/run_controls.py'
TOOL = Path(__file__).resolve()
CURRENT = ContextVar('v064_controls_arm', default=None)
KEPT_ENVIRONMENT = ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS')
PACKAGE = ROOT / 'migration/onpolicy-v064'
WORK = PACKAGE / 'stage0'
CONTRACT = ROOT / 'contracts/onpolicy-controls-v064b.json'
ELIGIBLE = PACKAGE / 'eligible.json'
PREREG = Path('PROJECT_ROOT/briefing/research/gemma-4-2026/prereg-2026-11-07-onpolicy-vs-teacher-subset-v064-r2.md')
PREREG_SHA = '0893728c3d763990d5b3a4bd64c0d6b37b55eb0b555ecd2727f36a636c6f4b60'
ADDENDUM = Path('PROJECT_ROOT/briefing/research/gemma-4-2026/prereg-2026-11-07-addendum-confirmation-and-gate-rules-v064r2-v065.md')
ADDENDUM_SHA = 'b602a5d366002bc5335fcc7a94ada72cba71535b3a63551f74d28bd9a8f39a56'
PREVIOUS_BUILD = ROOT / 'tools/run_controls_capped_v064.py'
PREVIOUS_BUILD_SHA = '1b09d34a49609c2160257af36f2a492d69df31dfa6fec60c07243836d170a86b'
PREVIOUS_CONTRACT_SHA = '553e10134e4a0de9188fb0605b89f5668f301f0656faf9bbe163eb64718aa880'
SOURCE_TOOL = ROOT / 'tools/run_controls_capped_v056.py'
SOURCE_TOOL_SHA = 'a371e112e2bdabbe79f66bc08ffe45ed1808d28327fb86750eaad9d3b0636aea'
SPLIT_SHA = '9c99c412a0e094dca93f6867703e535b3a84c9d4878c5bc867bac8d8b5a59f8d'
SCHEMA, GENERATION, CANARY, MIN_ELIGIBLE = 'onpolicy-v064-controls-capped-1', 'v064', 'fastapi_11194', 10
CANDIDATES = (
    'fastapi_11355', 'fastapi_12942', 'fastapi_13207', 'fastapi_13537', 'fastapi_13713', 'fastapi_13786', 'fastapi_13920',
    'fastapi_14099', 'fastapi_14246', 'fastapi_14258', 'fastapi_14262', 'fastapi_14266', 'fastapi_14297', 'fastapi_14301',
    'fastapi_14303', 'fastapi_14306', 'fastapi_14349', 'fastapi_14360', 'fastapi_14361', 'fastapi_14371', 'fastapi_14372',
    'fastapi_14419', 'fastapi_14430', 'fastapi_14448', 'fastapi_14455', 'fastapi_14458', 'fastapi_14459', 'fastapi_14463',
    'fastapi_14479', 'fastapi_14482', 'fastapi_14485', 'fastapi_14487', 'fastapi_14492', 'fastapi_14512', 'fastapi_14583',
    'fastapi_14605', 'fastapi_14609', 'fastapi_14616', 'fastapi_14786', 'fastapi_14791', 'fastapi_14794', 'fastapi_14873',
    'fastapi_14953', 'fastapi_14962', 'fastapi_14964', 'fastapi_14978', 'fastapi_14986', 'fastapi_15023', 'fastapi_15030',
    'fastapi_15280', 'fastapi_15588', 'fastapi_15589', 'fastapi_15661', 'fastapi_15763', 'fastapi_15785', 'fastapi_15800',
    'fastapi_5077', 'fastapi_5624', 'fastapi_9425', 'fastapi_9555', 'fastapi_9753', 'httpx_3672', 'requests_6644',
    'requests_7205', 'requests_7309', 'requests_7315', 'requests_7328', 'requests_7427', 'requests_7505',
)
CAPPED = dict(concurrency=1, service_seconds=7200)
RULE = 'eligible = gold resolved and empty not resolved and no control error and no OOM kill (prereg v064-r2 Stage 0)'
PACE_RULE = dict(log=str(ROOT / 'analysis/m1q-boss-v062/m1q-v062-main.log'), unit='m1q-v062-main.service', baseline_steps=[2, 20],
                 window=3, factor=0.95, poll_seconds=30, reference_log=str(ROOT / 'sft/runs/A1_M0q_v053/train_log.jsonl'),
                 resume='never while the training unit is active; the job continues only after the unit is inactive')


def make_tagger(guarded, owners):
    """Wrap the guarded executor so each sandbox workspace is attributed to the (task, arm) that is running."""
    def tagged(*args, **kwargs):
        key = hashlib.sha256(str(Path(kwargs['cwd']).resolve()).encode()).hexdigest()
        owners.setdefault(key, CURRENT.get())
        return guarded(*args, **kwargs)
    return tagged


def attribute_peaks(guard, owners):
    """Guard sandbox peaks per (task, arm); a sandbox is keyed by the hash of its workspace path."""
    closed = {record['sandbox']: record for record in guard['sandboxes']}
    out = {}
    for key, owner in owners.items():
        if not owner:
            continue
        commands = [c for c in guard['commands'] if c['sandbox'] == key]
        slot = out.setdefault(tuple(owner), dict(peak_memory_bytes=0, commands=0, oom_kill_events=0, sandboxes=0))
        slot['peak_memory_bytes'] = max(slot['peak_memory_bytes'], closed[key]['memory_peak'] if key in closed else 0)
        slot['commands'] += len(commands)
        slot['oom_kill_events'] += sum(c['memory_events'].get('oom_kill', 0) for c in commands)
        slot['sandboxes'] += 1
    return out


def pace_state(rule):
    """v062 pace rule on the training unit's per-update stdout rows; always ok when that unit is not active."""
    state = subprocess.run(['systemctl', '--user', 'show', rule['unit'], '--property=ActiveState', '--value'],
                           capture_output=True, text=True, timeout=10).stdout.strip()
    if state != 'active':
        return dict(ok=True, unit_state=state)
    rows = [json.loads(line) for line in Path(rule['log']).read_text().splitlines() if line.startswith('{"step"')]
    rates = {row['step']: row['tokens_per_s'] for row in rows}
    first, last = rule['baseline_steps']
    if any(step not in rates for step in range(first, last + 1)) or len(rows) < rule['window']:
        return dict(ok=False, unit_state=state, reason='baseline not available yet')
    baseline = sum(rates[step] for step in range(first, last + 1)) / (last - first + 1)
    window = rows[-rule['window']:]
    rolling = sum(row['tokens_per_s'] for row in window) / rule['window']
    return dict(ok=rolling >= rule['factor'] * baseline, unit_state=state, step=rows[-1]['step'],
                baseline=round(baseline, 1), rolling=round(rolling, 1), same_step_ratio=same_step_ratio(rule, window))


def same_step_ratio(rule, window):
    """Addendum section 3: tokens/s over the same steps of the reference run; about 1.00 = sequence length, < 0.95 = contention."""
    try:
        reference = {row.get('step'): row.get('tokens_per_s') for row in
                     (json.loads(line) for line in Path(rule['reference_log']).read_text().splitlines() if line.strip())}
    except (OSError, ValueError):
        return None
    ratios = [row['tokens_per_s'] / reference[row['step']] for row in window if reference.get(row['step'])]
    return round(sum(ratios) / len(ratios), 3) if ratios else None


def journal_event(journal, event, state, task, arm):
    with Path(journal).open('a') as stream:
        stream.write(json.dumps(dict(state, event=event, task=task, arm=arm, at=screen.utc_now(), monotonic=time.monotonic()),
                                sort_keys=True) + chr(10))


async def pace_gate(rule, journal, task, arm, latch):
    """Checked before every control arm (addendum section 3): the first failing check pauses the job until the training unit is
    inactive, even if the pace recovers; a running arm always finishes; pause and resume are journaled."""
    while True:
        state = pace_state(rule)
        if state['unit_state'] != 'active':
            if latch.get('paused') and not latch.get('resumed'):
                latch['resumed'] = True
                journal_event(journal, 'resumed_after_training', state, task, arm)
            return state
        if not latch.get('paused'):
            if state['ok']:
                return state
            latch['paused'] = True
            journal_event(journal, 'paused', state, task, arm)
        await asyncio.sleep(rule['poll_seconds'])


class PacedGate:
    """The controls concurrency gate; once a slot is held, the arm starts only when the pace rule allows it."""
    def __init__(self, gate, rule, journal, task, arm, latch):
        self.gate, self.rule, self.journal, self.task, self.arm, self.latch = gate, rule, journal, task, arm, latch

    async def __aenter__(self):
        await self.gate.__aenter__()
        try:
            await pace_gate(self.rule, self.journal, self.task, self.arm, self.latch)
        except BaseException:
            await self.gate.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc):
        return await self.gate.__aexit__(*exc)


def payload(out_dir, contract_sha):
    """Runs inside the capped service: the original controls main under the v043 memory guard, paced by the v062 rule."""
    contract = verify_contract(contract_sha)
    ids = list(contract['candidate_ids'])
    out = Path(out_dir)
    spec = importlib.util.spec_from_file_location('original_controls_v064', CONTROLS)
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    controls.GEMMA, controls.DATA, controls.SPLITS_PATH = ROOT, ROOT / 'data/competition', ROOT / 'splits/v1.json'
    original, latch = controls.run_arm, {}

    async def tagged_arm(manager, config, task, arm, patch, gate, *rest):
        token = CURRENT.set([task.instance_id, arm])
        paced = PacedGate(gate, contract['pace_rule'], out / 'pace.jsonl', task.instance_id, arm, latch)
        try:
            return await original(manager, config, task, arm, patch, paced, *rest)
        finally:
            CURRENT.reset(token)

    controls.run_arm = tagged_arm
    sys.argv = [str(CONTROLS), '--ids', ','.join(ids), '--out', str(out / 'controls'),
                '--concurrency', str(contract['capped_controls']['concurrency'])]
    from sandbox_memory_guard_v043 import MemoryGuard, qualify_guard_receipt
    guard = MemoryGuard(out / 'guard.json', memory_max=contract['memory_correction']['per_sandbox']['memory_max'])
    guard.install()
    owners = {}
    guard.module.execute_subprocess_command = make_tagger(guard.module.execute_subprocess_command, owners)
    success, error = False, None
    try:
        controls.main()
        success = True
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        with (out / 'sandbox-owners.json').open('x') as stream:
            json.dump(owners, stream, indent=2, sort_keys=True)
            stream.write(chr(10))
        guard.close(completed=success, error=error)
    qualify_guard_receipt(out / 'guard.json', **contract['memory_correction']['per_sandbox'])


def read_if(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def build_receipt(out, contract, service_error):
    out = Path(out)
    ids = list(contract['candidate_ids'])
    csv_path = out / 'controls/controls.csv'
    rows = {r['instance_id']: r for r in csv.DictReader(csv_path.open())} if csv_path.is_file() else {}
    guard = read_if(out / 'guard.json')
    peaks = attribute_peaks(guard, read_if(out / 'sandbox-owners.json') or {}) if guard else {}
    entries = {}
    for task in ids:
        row = rows.get(task)
        if row is None:
            continue
        none = dict(peak_memory_bytes=None, commands=None, oom_kill_events=None, sandboxes=0)
        arms = {arm: dict(resolved=row[arm + '_resolved'] == 'True', exit_code=int(row[arm + '_exit_code']),
                          seconds=float(row[arm + '_seconds']), **peaks.get((task, arm), none)) for arm in ('gold', 'empty')}
        entries[task] = dict(arms, error=row['error'])
    start = read_if(out / 'service.log.service-start.json')
    end = read_if(out / 'service.log.service-end.json')
    guard_block = None
    if guard:
        guard_block = dict(path=str(out / 'guard.json'), sha256=screen.sha(out / 'guard.json'),
                           journal_path=str(out / 'guard.commands.jsonl'), journal_sha256=screen.sha(out / 'guard.commands.jsonl'),
                           command_count=guard['command_count'], oom_command_count=guard['oom_command_count'],
                           sandboxes=len(guard['sandboxes']), sandbox_peak_max=max((s['memory_peak'] for s in guard['sandboxes']), default=0),
                           service_memory_peak=guard['service_memory_peak'])
    return dict(
        schema=SCHEMA, generation=GENERATION, created_at=screen.utc_now(), directory=str(out),
        task_ids=ids, predicate=RULE, tasks=entries, contract_sha256=screen.sha(CONTRACT), pace=pace_summary(out, contract),
        memory=dict(per_sandbox=contract['memory_correction']['per_sandbox'],
                    service_memory_max=int(guard['service_memory_max']) if guard else None,
                    service_properties=(start or {}).get('memory_properties')),
        guard=guard_block,
        service=dict(error=service_error, unit=(start or {}).get('unit'), start_sha256=screen.sha(out / 'service.log.service-start.json') if start else None,
                     end_sha256=screen.sha(out / 'service.log.service-end.json') if end else None,
                     returncode=(end or {}).get('returncode'), cgroup_empty=(end or {}).get('cgroup_empty')),
        inputs={name: screen.sha(path) for name, path in (
            ('guard_source', ROOT / 'tools/sandbox_memory_guard_v043.py'), ('controls_runner', CONTROLS),
            ('tasks_jsonl', ROOT / 'data/competition/tasks.jsonl'), ('splits', ROOT / 'splits/v1.json'),
            ('this_tool', TOOL), ('screen_runner', ROOT / 'tools/run_temp_screen_31b_v056.py'),
            ('source_tool', SOURCE_TOOL), ('contract', CONTRACT), ('prereg', PREREG))},
        outputs={name: screen.sha(out / name) for name in ('controls/controls.csv', 'controls/summary.json', 'sandbox-owners.json')
                 if (out / name).is_file()})


def pace_summary(out, contract):
    path = Path(out) / 'pace.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    times = {row['event']: row['monotonic'] for row in rows}
    fields = ('event', 'at', 'step', 'baseline', 'rolling', 'same_step_ratio', 'unit_state', 'task', 'arm')
    return dict(rule=contract['pace_rule'], paused='paused' in times, events=[{key: row.get(key) for key in fields} for row in rows],
                paused_seconds=times['resumed_after_training'] - times['paused'] if {'paused', 'resumed_after_training'} <= set(times) else None,
                journal_sha256=screen.sha(path) if path.is_file() else None)


def eligibility(receipt):
    """Prereg v064-r2 Stage 0 rule per candidate; exit codes are recorded beside it, not judged."""
    run_oom_free = (receipt.get('guard') or {}).get('oom_command_count') == 0
    rows = {}
    for task in CANDIDATES:
        entry = (receipt.get('tasks') or {}).get(task)
        if entry is None:
            rows[task] = dict(eligible=False, missing=True)
            continue
        gold, empty = entry['gold'], entry['empty']
        kills = [gold['oom_kill_events'], empty['oom_kill_events']]
        checks = dict(gold_resolved=gold['resolved'] is True, empty_not_resolved=empty['resolved'] is False,
                      no_error=not entry['error'], no_oom_kill=all(k == 0 for k in kills) or (run_oom_free and all(k in (0, None) for k in kills)))
        rows[task] = dict(checks, eligible=all(checks.values()), gold_exit_code=gold['exit_code'], empty_exit_code=empty['exit_code'])
    return rows, [task for task in CANDIDATES if rows[task]['eligible']]


def receipt_problems(receipt, contract):
    """Infrastructure reasons a receipt cannot set eligibility; failing controls are outcomes, not problems."""
    memory, held, service = contract['memory_correction'], receipt.get('memory') or {}, receipt.get('service') or {}
    checks = [(receipt.get('schema') != SCHEMA or receipt.get('generation') != GENERATION, 'schema or generation'),
              (receipt.get('task_ids') != list(CANDIDATES), 'task set differs from the contract'),
              (held.get('per_sandbox') != memory['per_sandbox'] or held.get('service_memory_max') != memory['service']['MemoryMax'],
               'memory configuration differs from the contract'),
              (receipt.get('guard') is None, 'guard evidence missing'),
              (service.get('returncode') != 0 or service.get('cgroup_empty') is not True, 'capped service did not finish cleanly')]
    return [text for failed, text in checks if failed] + [t + ': missing' for t in CANDIDATES if t not in (receipt.get('tasks') or {})]


def write_eligible(receipt, receipt_path, contract_sha):
    """Written once; the generation contract pins this list and nothing outside it is generated on or kept."""
    rows, eligible = eligibility(receipt)
    listing = ''.join(task + chr(10) for task in eligible)
    doc = dict(schema='onpolicy-v064-eligible-1', generation=GENERATION, prereg=dict(path=str(PREREG), sha256=PREREG_SHA),
               contract_sha256=contract_sha, receipt=dict(path=str(receipt_path), sha256=screen.sha(receipt_path)), rule=RULE,
               candidate_count=len(CANDIDATES), eligible_task_ids=eligible, eligible_count=len(eligible),
               eligible_list_sha256=hashlib.sha256(listing.encode()).hexdigest(),
               list_sha256_rule='sha256 of the eligible ids in candidate order, each followed by a newline',
               per_repo={repo: sum(t.split('_')[0] == repo for t in eligible) for repo in sorted({t.split('_')[0] for t in eligible})},
               min_eligible=MIN_ELIGIBLE, stop=len(eligible) < MIN_ELIGIBLE, per_task=rows)
    with ELIGIBLE.open('x') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True)
        stream.write(chr(10))
    return dict(path=str(ELIGIBLE), sha256=screen.sha(ELIGIBLE), eligible_count=len(eligible), stop=doc['stop'],
                eligible_list_sha256=doc['eligible_list_sha256'])


def service_environment():
    env = {key: os.environ[key] for key in KEPT_ENVIRONMENT if key in os.environ}
    env.setdefault('XDG_RUNTIME_DIR', '/run/user/' + str(os.getuid()))
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=' + env['XDG_RUNTIME_DIR'] + '/bus')
    return env


def verify_contract(expected):
    """The Stage-0 contract by sha: identity, the 69 candidates, the frozen r2 prereg and every pin (snapshots included)."""
    screen.require(screen.sha(CONTRACT) == expected and Path(str(CONTRACT) + '.sha256').read_text().split()[0] == expected,
                   'Stage-0 contract hash mismatch')
    doc = screen.read(CONTRACT)
    screen.require(doc['generation'] == GENERATION and doc['candidate_ids'] == list(CANDIDATES) and doc['prereg']['sha256'] == PREREG_SHA and
                   doc['addendum'] == dict(path=str(ADDENDUM), sha256=ADDENDUM_SHA) and
                   doc['pace_rule'] == PACE_RULE and doc['capped_controls'] == CAPPED and doc['min_eligible'] == MIN_ELIGIBLE and
                   CANARY not in CANDIDATES and len(set(CANDIDATES)) == 69 and not any(t.startswith('rich_') for t in CANDIDATES),
                   'Stage-0 contract content changed')
    for name, digest in doc['pins'].items():
        screen.require(screen.sha(name) == digest, 'Stage-0 pin changed: ' + name)
    return doc


def prepare():
    """Writes the Stage-0 contract once (CPU only) and returns its sha."""
    screen.require(not CONTRACT.exists() and not Path(str(CONTRACT) + '.sha256').exists(), 'Stage-0 contract exists')
    screen.require(screen.sha(SOURCE_TOOL) == SOURCE_TOOL_SHA and screen.sha(PREREG) == PREREG_SHA and screen.sha(ADDENDUM) == ADDENDUM_SHA and
                   screen.sha(PREVIOUS_BUILD) == PREVIOUS_BUILD_SHA and screen.sha(ROOT / 'splits/v1.json') == SPLIT_SHA,
                   'Source tool, previous build, prereg, addendum or split changed')
    split = screen.read(ROOT / 'splits/v1.json')
    sound = [t for group in split['train_groups'] for t in group if split['soundness'][t]['sound'] == '1']
    snapshots = [ROOT / 'data/competition/snapshots' / (t + '.tgz') for t in sound]
    screen.require(len(sound) == 70 and [t for t in sound if t != CANARY] == list(CANDIDATES) and all(p.is_file() for p in snapshots) and
                   not any(t.startswith('rich_') for t in sound) and not set(sound) & set(split['holdout_rich']), 'Candidates differ from the split')
    memory = screen.read(screen.DESIGN)['memory_correction']
    screen.require(memory == screen.read(ROOT / 'contracts/a0-capped-v052-runtime-DRAFT.json')['memory_correction'],
                   'Controls memory settings differ from the v052 generation environment')
    names = [TOOL, ROOT / 'tools/test_run_controls_capped_v064b.py', PACKAGE / 'stage0-controls-b.diff', PACKAGE / 'stage0-controls-b-vs-a.diff',
             PREVIOUS_BUILD, ROOT / 'contracts/onpolicy-controls-v064.json', ADDENDUM, Path(PACE_RULE['reference_log']), SOURCE_TOOL,
             ROOT / 'tools/run_temp_screen_31b_v056.py', CONTROLS, ROOT / 'tools/sandbox_memory_guard_v043.py', screen.DESIGN,
             ROOT / 'data/competition/tasks.jsonl', ROOT / 'splits/v1.json', ROOT / 'contracts/a0-capped-v052-runtime-DRAFT.json', PREREG]
    names += [p for p in snapshots if p.name != CANARY + '.tgz']
    repos = lambda ids: {r: sum(t.split('_')[0] == r for t in ids) for r in sorted({t.split('_')[0] for t in ids})}
    doc = dict(schema='onpolicy-v064-stage0-contract-1', generation=GENERATION, stage='Stage 0 eligibility controls', status='STAGE0_READY',
               prereg=dict(path=str(PREREG), sha256=PREREG_SHA), source_tool=dict(path=str(SOURCE_TOOL), sha256=SOURCE_TOOL_SHA),
               candidate_ids=list(CANDIDATES), per_repo_candidates=repos(CANDIDATES), canary_excluded=CANARY,
               candidate_list_sha256=hashlib.sha256(''.join(t + chr(10) for t in CANDIDATES).encode()).hexdigest(),
               split=dict(path=str(ROOT / 'splits/v1.json'), sha256=SPLIT_SHA, sound_tasks=70, per_repo_sound=repos(sound),
                          snapshots_present='70/70', rich_ids=0, holdout_overlap=0),
               rule=RULE, min_eligible=MIN_ELIGIBLE, stop_rule='fewer than 10 eligible tasks: stop, A not trained (prereg v064-r2)',
               capped_controls=CAPPED, memory_correction=memory, pace_rule=PACE_RULE,
               pace_text='v062 rule as restated in the frozen addendum section 3: baseline = mean tokens/s of updates 2-20; the first check '
                         'with the rolling mean of the last 3 updates below 0.95 x baseline pauses the job, and no new control arm starts '
                         'until the training unit is inactive (a running arm finishes; checks every 30 s); pause and resume are journaled '
                         'with the same-step ratio to the M0q reference run (about 1.00 = sequence-length effect, below 0.95 = contention)',
               addendum=dict(path=str(ADDENDUM), sha256=ADDENDUM_SHA),
               previous_build=dict(tool=str(PREVIOUS_BUILD), sha256=PREVIOUS_BUILD_SHA, contract_sha256=PREVIOUS_CONTRACT_SHA,
                                   status='never launched; superseded because its pause resumed when the pace recovered'),
               timing='during a training under the pace rule or in an evaluation-free gap; never during a capped evaluation or the generation',
               leases_and_gpu='none: this job takes no lease, starts no server and uses no GPU',
               outputs=dict(work=str(WORK), eligible=str(ELIGIBLE)), pins={str(p): screen.sha(p) for p in names})
    with CONTRACT.open('x') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True)
        stream.write(chr(10))
    digest = screen.sha(CONTRACT)
    with Path(str(CONTRACT) + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + CONTRACT.name + chr(10))
    return digest


def check(contract_sha):
    """CPU-only dry run: the contract and every pin verify; nothing runs and no lease or GPU is touched."""
    contract = verify_contract(contract_sha)
    return dict(status='stage0_contract_verified', contract_sha256=contract_sha, prereg_sha256=contract['prereg']['sha256'],
                candidates=len(contract['candidate_ids']), candidate_list_sha256=contract['candidate_list_sha256'],
                per_repo=contract['per_repo_candidates'], pins=len(contract['pins']), eligible_written=ELIGIBLE.exists(),
                work=str(WORK), pace_now=pace_state(contract['pace_rule']), gpu_operations=False)


def launch(contract_sha):
    contract = verify_contract(contract_sha)
    screen.require(not ELIGIBLE.exists(), 'Eligible list already written: ' + str(ELIGIBLE))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = WORK / ('controls-capped-' + stamp)
    receipt_path = WORK / ('controls-capped-' + stamp + '.json')
    screen.require(not out.exists() and not receipt_path.exists(), 'Capped controls destination exists')
    out.mkdir(parents=True)
    at_launch = pace_state(contract['pace_rule'])
    service = screen.MemoryBoundedService(time.monotonic() + contract['capped_controls']['service_seconds'], out, 0,
                                          contract['memory_correction'])
    error = None
    try:
        service([sys.executable, str(TOOL), '--payload', str(out), '--contract-sha', contract_sha], out / 'service.log',
                env=service_environment(), cwd=ROOT)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    receipt = dict(build_receipt(out, contract, error), pace_at_launch=at_launch)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write(chr(10))
    problems = receipt_problems(receipt, contract) + ([] if error is None else ['service: ' + error])
    eligible = None if problems else write_eligible(receipt, receipt_path, contract_sha)
    print(json.dumps(dict(receipt=str(receipt_path), sha256=screen.sha(receipt_path), problems=problems, eligible=eligible,
                          tasks={t: [e['gold']['resolved'], e['gold']['exit_code'], e['empty']['resolved'], e['empty']['exit_code'],
                                     e['gold']['peak_memory_bytes'], e['empty']['peak_memory_bytes']]
                                 for t, e in receipt['tasks'].items()}), indent=1, sort_keys=True))
    return 0 if not problems else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--launch', action='store_true')
    group.add_argument('--payload', type=Path)
    parser.add_argument('--contract-sha')
    args = parser.parse_args(argv)
    if args.prepare:
        print(json.dumps(dict(status='stage0_contract_prepared', contract_sha256=prepare(), gpu_operations=False)))
        return 0
    screen.require(bool(args.contract_sha), 'The Stage-0 contract sha is required')
    if args.payload is not None:
        payload(args.payload, args.contract_sha)
        return 0
    if args.check:
        print(json.dumps(check(args.contract_sha), indent=1, sort_keys=True))
        return 0
    return launch(args.contract_sha)


if __name__ == '__main__':
    sys.exit(main())
