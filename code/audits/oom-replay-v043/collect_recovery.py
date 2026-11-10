#!/usr/bin/env python
"""Collector-only recovery after successful bounded replay; executes no commands/models/probes."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import re
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
GUARD=ROOT/'tools/sandbox_memory_guard_v043.py'
EXPECTED='5ad9d40d3a1e4d48db3fbffed3164226e0857aee076482cd32ae0072b274ffd6'
GIB=1024**3
FIELDS=['status','stdout','stderr','exit_code']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
assert sha(GUARD)==EXPECTED
payload=json.loads((OUT/'payload.json').read_text())
selection=json.loads((OUT/'selection.json').read_text())
start=json.loads((OUT/'service-start.json').read_text())
log=(OUT/'service.log').read_text()
assert payload['completed'] and payload['class_identity_preserved']
assert payload['guard_source_sha256']==EXPECTED and payload['selection_sha256']==sha(OUT/'selection.json')
for mapping in [payload['source_pins'],selection['source_pins']]:
 for path,expected in mapping.items():assert sha(path)==expected, 'Input changed: '+path
assert start['guard_source_sha256']==EXPECTED and start['state']['InvocationID']
assert start['state']['MemoryMax']==str(12*GIB) and start['state']['MemorySwapMax']=='0'
assert start['state']['RuntimeMaxUSec']=='3min'
assert '--property=Delegate=memory' in start['argv'] and '--property=DelegateSubgroup=supervisor' in start['argv']
assert 'Finished with result: success' in log and 'code=exited, status=0/SUCCESS' in log
runtime=float(re.search(r'Service runtime:\s*([\d.]+)s',log)[1])
state=subprocess.run(['systemctl','--user','show',start['unit'],'--property=LoadState'],capture_output=True,text=True)
assert state.stdout.strip()=='LoadState=not-found', 'Owned unit still present; assess exact identity before action'
cg=Path('/sys/fs/cgroup')/start['state']['ControlGroup'].lstrip('/')
assert not cg.exists(), 'Captured owned cgroup remains present'
spec=importlib.util.spec_from_file_location('replay_qualifier',GUARD);guard=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guard;spec.loader.exec_module(guard)
guards={}
for name,limit in [('compat',4*GIB),('aggregate_descendants',64*1024**2),('allocation_5g',4*GIB)]:
 path=OUT/(name+'.guard.json');data=guard.qualify_guard_receipt(path,memory_max=limit)
 assert data['service_memory_max']==str(12*GIB) and data['service_memory_swap_max']==0
 assert data['service_memory_peak']<12*GIB
 guards[name]=data
comparable=[item for item in payload['deterministic_comparison'] if item['byte_comparable']]
assert len(comparable)==4 and all(item['all_semantic_fields_equal'] for item in comparable)
for item in comparable:
 a=next(v for v in payload['recorded_results']['off'] if v['label']==item['label'])['raw_execution_result']
 b=next(v for v in payload['recorded_results']['on'] if v['label']==item['label'])['raw_execution_result']
 assert set(a)==set(b) and all(a[k]==b[k] for k in FIELDS)
assert payload['workspace_copies'][0]==payload['workspace_copies'][1]
for name in ['aggregate_descendants','allocation_5g']:
 data=payload['probes'][name]
 assert data['off']['exit_code']==0 and data['on']['exit_code']==-9
 assert data['guard_oom_command_count']==guards[name]['oom_command_count']==1 and data['cgroups_empty']
variant=next(x for x in payload['deterministic_comparison'] if not x['byte_comparable'])
report={'schema':'oom-replay-v043','completed':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'guard_source_sha256':EXPECTED,'class_identity_preserved':True,
 'deterministic_replays':{'count':4,'all_semantic_fields_equal':True,'compared_fields':FIELDS,'duration_excluded':True,'all_original_fields_preserved':True},
 'path_variant':{'label':variant['label'],'declared_non_gating_before_execution':True,'all_raw_fields_preserved':True,'status_and_exit_code_equal':variant['equal_fields']['status'] and variant['equal_fields']['exit_code'],
   'stdout_stderr_equal':False,'observed_difference':'Absolute disposable workspace paths differ. The guarded find|head run also emitted Broken pipe diagnostics; the uncapped run did not. No output normalization was applied.'},
 'allocation_5g':{'off_success':True,'on_oom_contained':True,'on_exit_code':-9},
 'aggregate_descendants':{'off_success':True,'on_oom_contained':True,'on_exit_code':-9,'memory_max':64*1024**2},
 'outer_services':{'MemoryMax':12*GIB,'MemorySwapMax':0,'RuntimeMaxSec':180,'all_exited':True,'all_cgroups_empty':True,
   'unit':start['unit'],'captured_InvocationID':start['state']['InvocationID'],'captured_ControlGroup':start['state']['ControlGroup'],
   'start':start['state'],'end_evidence':{'source':'preserved systemd-run output','result':'success','main_exit_status':0,'seconds':runtime},
   'current_observation':{'LoadState':'not-found','captured_cgroup_exists':False},'memory_peak':max(v['service_memory_peak'] for v in guards.values())},
 'collector_recovery':{'needed':True,'reason':'Successful service had already released its empty ControlGroup; original collector formed a root cgroup path.',
   'original_collector_exit_code':1,'original_executed_source_sha256':sha(OUT/'replay.py'),
   'recovery_collector_sha256':sha(Path(__file__)),'exception_evidence':str(OUT/'collector-exception.txt'),
   'replayed_probes':False,'post_gc_properties_invented':False},
 'source_pins':{**payload['source_pins'],str(Path(__file__)):sha(Path(__file__)),str(OUT/'collector-exception.txt'):sha(OUT/'collector-exception.txt')},
 'source_snapshot_pins':selection['source_pins'],
 'evidence_pins':{str(p):sha(p) for p in [OUT/'payload.json',OUT/'service-start.json',OUT/'service.log',*sorted(OUT.glob('*.guard.json')),*sorted(OUT.glob('*.commands.jsonl'))]},
 'limitations':payload['limitations']+['Byte identity is demonstrated only for the four predeclared deterministic recorded commands.','No claim of bit-identical LLM decoding or universal shell-output equivalence.']}
write(OUT/'report.json',report)
(OUT/'report.json.sha256').write_text(sha(OUT/'report.json')+'  report.json\n')
print(json.dumps({'completed':True,'report_sha256':sha(OUT/'report.json'),'outer_runtime_seconds':runtime,'outer_peak_bytes':report['outer_services']['memory_peak'],'deterministic_cases':4,'probe_rerun':False}))
