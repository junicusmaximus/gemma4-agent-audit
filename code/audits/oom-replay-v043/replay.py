#!/usr/bin/env python
"""Recorded-command compatibility and bounded aggregate-memory probes; no model execution."""
import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

ROOT=Path('PROJECT_ROOT/gemma4')
OUT=Path(__file__).resolve().parent
GUARD=ROOT/'tools/sandbox_memory_guard_v043.py'
SELECTION=OUT/'selection.json'
GIB=1024**3
FIELDS=('status','stdout','stderr','exit_code')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
def load_module(path):
    spec=importlib.util.spec_from_file_location('replay_memory_guard',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def raw_result(result):
    if dataclasses.is_dataclass(result): return dataclasses.asdict(result)
    if hasattr(result,'model_dump'): return result.model_dump()
    return {k:getattr(result,k) for k in (*FIELDS,'duration_seconds')}

def tree_digest(root):
    h=hashlib.sha256();count=0
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode()+b'\0');h.update(bytes.fromhex(sha(p)));count+=1
    return {'sha256':h.hexdigest(),'files':count}

def payload(guard_sha):
    import swegemma.sandbox.subprocess as sandbox
    from adk_eval_core.sandbox import SubprocessSandbox
    from adk_eval_core.sandbox import execute_subprocess_command
    m=load_module(GUARD)
    assert sha(GUARD)==guard_sha
    selected=json.loads(SELECTION.read_text())
    for path,value in selected['source_pins'].items(): assert sha(path)==value
    original_class=sandbox.SubprocessManager;original_adk=SubprocessSandbox
    # Environment setup occurs before installing the execution seam.
    work=OUT/'work';work.mkdir()
    managers=[]
    for arm in ('off','on','probe'):
        base=work/arm;base.mkdir()
        manager=original_class(base_dir=base)
        sid=manager.start();ws=Path(manager.work_dir)
        assert ws.parent.parent.resolve()==base.resolve()
        if arm!='probe':
            with tarfile.open(selected['snapshot']) as archive: archive.extractall(ws,filter='data')
        managers.append((manager,sid,ws))
    before=[tree_digest(item[2]) for item in managers[:2]]
    assert before[0]==before[1], 'Disposable repo copies differ before replay'
    results={};guard_paths=[];identity=[]
    for arm,(manager,sid,ws) in zip(('off','on'),managers[:2]):
        guard=None
        if arm=='on':
            receipt=OUT/'compat.guard.json';guard=m.MemoryGuard(receipt);guard.install();guard_paths.append(receipt)
        try:
            identity.append(sandbox.SubprocessManager is original_class and type(manager) is original_class)
            import adk_eval_core.sandbox as core
            identity.append(core.SubprocessSandbox is original_adk)
            records=[]
            for case in selected['commands']:
                result=manager.run_command(case['command'],timeout=20)
                records.append({'label':case['label'],'byte_comparable':case['byte_comparable'],
                    'raw_execution_result':raw_result(result),'workspace':str(ws)})
            results[arm]=records
        finally:
            assert Path(manager.work_dir).parent.parent.resolve()==(work/arm).resolve()
            manager.stop(sid)
            if guard:guard.close(completed=True)
    comparison=[]
    for off,on in zip(results['off'],results['on']):
        exact={k:off['raw_execution_result'][k].encode('utf8')==on['raw_execution_result'][k].encode('utf8')
            if isinstance(off['raw_execution_result'][k],str) else off['raw_execution_result'][k]==on['raw_execution_result'][k] for k in FIELDS}
        comparison.append({'label':off['label'],'byte_comparable':off['byte_comparable'],
            'equal_fields':exact,'all_semantic_fields_equal':all(exact.values()),
            'duration_seconds_equal':off['raw_execution_result']['duration_seconds']==on['raw_execution_result']['duration_seconds']})
    deterministic=[x for x in comparison if x['byte_comparable']]
    assert all(x['all_semantic_fields_equal'] for x in deterministic), 'Recorded command compatibility changed'
    for path in guard_paths:m.qualify_guard_receipt(path)
    probe,probe_sid,_=managers[2]
    leaf='import time; a=bytearray(32*1024**2); time.sleep(3)'
    middle='import subprocess,sys; p=[subprocess.Popen([sys.executable,"-B","-c",'+repr(leaf)+']) for _ in range(2)]; [x.wait() for x in p]'
    aggregate='import subprocess,sys; subprocess.run([sys.executable,"-B","-c",'+repr(middle)+'],check=True); print("aggregate_success")'
    allocation='a=bytearray(5*1024**3)\nfor i in range(0,len(a),4096): a[i]=1\nprint("allocated_5g")'
    probes={}
    for name,code,limit in [('aggregate_descendants',aggregate,64*1024**2),('allocation_5g',allocation,4*GIB)]:
        entries={}
        for arm in ('off','on'):
            guard=None;path=OUT/(name+'.guard.json')
            if arm=='on':guard=m.MemoryGuard(path,memory_max=limit);guard.install()
            try:
                entries[arm]=raw_result(probe.run_command('python3 -B -c '+shlex.quote(code),timeout=50))
            finally:
                if guard:guard.close(completed=True)
            if arm=='on':
                receipt=m.qualify_guard_receipt(path,memory_max=limit)
                entries['guard_receipt']=str(path)
                entries['guard_oom_command_count']=receipt['oom_command_count']
                entries['cgroups_empty']=receipt['all_command_cgroups_empty'] and receipt['all_sandbox_cgroups_removed']
        assert entries['off']['exit_code']==0
        assert entries['on']['exit_code']!=0 and entries['guard_oom_command_count']==1 and entries['cgroups_empty']
        probes[name]=entries
    probe.stop(probe_sid)
    assert sha(GUARD)==guard_sha, 'Guard source changed during replay'
    for path,value in selected['source_pins'].items():assert sha(path)==value
    identity.append(sandbox.SubprocessManager is original_class)
    assert all(identity)
    paths=[Path(inspect.getsourcefile(original_class)),Path(inspect.getsourcefile(execute_subprocess_command))]
    write(OUT/'payload.json',{'completed':True,'guard_source_sha256':guard_sha,'selection_sha256':sha(SELECTION),
        'class_identity_preserved':all(identity),'workspace_copies':before,'recorded_results':results,
        'deterministic_comparison':comparison,'probes':probes,
        'source_pins':{str(p):sha(p) for p in [Path(__file__),GUARD,SELECTION,*paths]},
        'limitations':['duration is retained and excluded from byte-equality gating',
            'path-emitting find is retained as a path-variant diagnostic, not normalized',
            'No model sampling, LLM decoding, scoring outcome or patch comparison is performed']})

def control(unit):
    p=subprocess.run(['systemctl','--user','show',unit,'--property=LoadState,InvocationID,ControlGroup,ActiveState,SubState,Result,MemoryMax,MemorySwapMax,RuntimeMaxUSec,MemoryPeak'],capture_output=True,text=True)
    data=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
    if p.returncode and data.get('LoadState')!='not-found':raise RuntimeError('Unable to inspect owned service: '+p.stderr[:300])
    return data

def launch(guard_sha):
    assert sha(GUARD)==guard_sha
    assert not (OUT/'payload.json').exists() and not (OUT/'report.json').exists()
    unit='gemma-oom-replay-v043-'+uuid.uuid4().hex+'.service'
    props=['Type=exec','Restart=no','RemainAfterExit=yes','KillMode=control-group','OOMPolicy=continue',
        'Delegate=memory','DelegateSubgroup=supervisor','MemoryAccounting=yes','MemoryMax=12G','MemorySwapMax=0','RuntimeMaxSec=180','TimeoutStopSec=15']
    argv=['systemd-run','--user','--unit',unit,'--wait','--pipe']
    for prop in props:argv+=['--property='+prop]
    argv += [sys.executable,'-B',str(Path(__file__).resolve()),'--payload','--guard-sha',guard_sha]
    log=OUT/'service.log'
    with log.open('xb') as stream:
        proc=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        began=time.monotonic();state={}
        try:
            for _ in range(100):
                state=control(unit)
                if state.get('InvocationID') and state.get('ControlGroup'):break
                if proc.poll() is not None:raise RuntimeError('Replay service exited before identity admission')
                time.sleep(.05)
            assert state.get('InvocationID') and state.get('ControlGroup')
            write(OUT/'service-start.json',{'unit':unit,'state':state,'argv':argv,'guard_source_sha256':guard_sha})
            deadline=time.monotonic()+175
            while True:
                end=control(unit)
                if (OUT/'payload.json').exists() and end.get('SubState')=='exited':break
                if proc.poll() is not None:raise RuntimeError('Replay service failed before completed payload: '+str(proc.returncode))
                if time.monotonic()>deadline:raise RuntimeError('Replay payload deadline exceeded')
                time.sleep(.1)
            assert end.get('Result')=='success'
            assert end.get('InvocationID')==state['InvocationID']
            assert end.get('MemoryMax')==str(12*GIB) and end.get('MemorySwapMax')=='0'
            cg=Path('/sys/fs/cgroup')/end['ControlGroup'].lstrip('/')
            assert 'populated 0' in (cg/'cgroup.events').read_text()
        finally:
            current=control(unit)
            if state.get('InvocationID') and current.get('InvocationID')==state['InvocationID']:
                subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=20)
            if proc.poll() is None:proc.wait(timeout=20)
    after=control(unit);cg=Path('/sys/fs/cgroup')/state['ControlGroup'].lstrip('/')
    empty=not cg.exists() or 'populated 0' in (cg/'cgroup.events').read_text()
    assert empty
    raw=json.loads((OUT/'payload.json').read_text())
    comparable=[x for x in raw['deterministic_comparison'] if x['byte_comparable']]
    report={'schema':'oom-replay-v043','completed':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'guard_source_sha256':guard_sha,'class_identity_preserved':raw['class_identity_preserved'],
        'deterministic_replays':{'count':len(comparable),'all_semantic_fields_equal':all(x['all_semantic_fields_equal'] for x in comparable),
            'compared_fields':list(FIELDS),'duration_excluded':True},
        'allocation_5g':{'off_success':raw['probes']['allocation_5g']['off']['exit_code']==0,
            'on_oom_contained':raw['probes']['allocation_5g']['guard_oom_command_count']==1},
        'aggregate_descendants':{'off_success':raw['probes']['aggregate_descendants']['off']['exit_code']==0,
            'on_oom_contained':raw['probes']['aggregate_descendants']['guard_oom_command_count']==1},
        'outer_services':{'MemoryMax':12*GIB,'MemorySwapMax':0,'RuntimeMaxSec':180,'all_exited':True,'all_cgroups_empty':empty,
            'unit':unit,'start':state,'end':end,'after':after,'client_returncode':proc.returncode,'seconds':time.monotonic()-began},
        'source_pins':raw['source_pins'],'source_snapshot_pins':json.loads(SELECTION.read_text())['source_pins'],
        'evidence_pins':{str(p):sha(p) for p in [OUT/'payload.json',OUT/'service-start.json',log,*sorted(OUT.glob('*.guard.json')),*sorted(OUT.glob('*.commands.jsonl'))]},
        'limitations':raw['limitations']}
    write(OUT/'report.json',report)
    (OUT/'report.json.sha256').write_text(sha(OUT/'report.json')+'  report.json\n')
    print(json.dumps({'completed':True,'report_sha256':sha(OUT/'report.json'),'deterministic_cases':len(comparable),'service_seconds':round(report['outer_services']['seconds'],3)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--guard-sha',required=True);ap.add_argument('--payload',action='store_true');args=ap.parse_args()
    if args.payload:payload(args.guard_sha)
    else:launch(args.guard_sha)
