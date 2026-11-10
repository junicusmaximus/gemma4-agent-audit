#!/usr/bin/env python3
"""Fresh v043 readout binding; immutable v042 statistics and v041 interpretation."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path('PROJECT_ROOT/gemma4')
LEGACY=ROOT/'tools/analyze_mask_control_v042.py'
LEGACY_SHA='23915fb00c2836b00fe14e3adfeecb865bdf52eb34e1fe3a04c4093cffdaf376'
if hashlib.sha256(LEGACY.read_bytes()).hexdigest()!=LEGACY_SHA:
    raise ValueError('Immutable v042 statistical implementation changed')
_spec=importlib.util.spec_from_file_location('v044_immutable_statistics',LEGACY)
legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(legacy)
for _name in ('require','sha','read_json','Inputs','module','validate_populations','validate_resolution_grid',
              'resolution_report','paired_slots','prediction_labels','validate_qualified_payload',
              'metrics_for_runs','secondary_report','population_report','prediction_inputs','write_fresh',
              'ARMS','SEEDS','DRAWS','RNG_SEED','READOUT','READOUT_SHA','PREDICTIONS','PREDICTIONS_SHA','INTERPRETATION'):
    globals()[_name]=getattr(legacy,_name)
MANIFEST=ROOT/'contracts/mask-control-analysis-v044-DRAFT.json'
OUTPUT=ROOT/'analysis/mask-control-v043'
MEMORY_CORRECTION={
    'version':'v043_sandbox_cgroup_v1',
    'scope':'Commands routed through SubprocessManager.exec: agent, verification pytest, and setup commands using that path',
    'exceptions':'Direct manager.start git/materialization and other direct subprocess host helpers remain under the outer 48GiB service cap only',
    'comparability':'Frozen numerical thresholds retained descriptively; this infrastructure correction is not evidence of historical exchangeability',
    'per_sandbox':{'memory_max':4294967296,'memory_swap_max':0},
    'service':{'MemoryMax':51539607552,'MemorySwapMax':0,'Delegate':'memory',
               'DelegateSubgroup':'supervisor','OOMPolicy':'continue'},
    'outer_memory_events_local_zero':['max','oom','oom_kill','oom_group_kill'],
    'host_min_available_bytes':68719476736,
    'global_oom_policy':'No kernel CONSTRAINT_NONE events after admission journal cursor through final cleanup'}


def bind_generation(readout,runtime):
    require(runtime.get('generation')=='v043','Expected fresh evaluation generation v043')
    runs=runtime.get('generation_runs')
    require(isinstance(runs,list) and len(runs)==len(readout['future_runs'])==6,'Expected all six fresh mappings')
    effective=copy.deepcopy(readout);effective['future_runs']=[]
    for original,new in zip(readout['future_runs'],runs):
        name=original['name'].replace('v035_','v043_',1)
        expected={**original,'scientific_name':original['name'],'name':name,
                  'submission_dir':str(ROOT/'subs'/name),'result_dir':str(ROOT/'results'/name),
                  'trace_dir':str(ROOT/'results'/name/'traces')}
        require(new==expected,'Fresh generation mapping/scientific identity changed: '+name)
        for key in ('submission_dir','result_dir','trace_dir'):
            require(Path(new[key]).resolve()==Path(new[key]),'Fresh path resolves outside declared generation: '+key)
        effective['future_runs'].append(copy.deepcopy(new))
    effective['future_analysis_dir']=str(OUTPUT)
    return effective


def validate_memory_correction(runtime,completed):
    require(runtime.get('memory_correction')==MEMORY_CORRECTION,'Missing or changed v043 memory correction')
    require(completed.get('memory_correction')==MEMORY_CORRECTION,'Completed memory correction differs')


def validate_admission(completed,ledger):
    proof=completed.get('admission_receipts')
    require(isinstance(proof,dict),'Missing memory recovery admission receipts')
    require(type(proof.get('gold_primary40')) is int and proof['gold_primary40']==40 and
            proof.get('original8_classified') is True and proof.get('byte_compatibility') is True,
            'Recovery controls/replay admission did not qualify')
    for kind in ('controls','replay'):
        require(isinstance(proof.get(kind+'_path'),str) and isinstance(proof.get(kind+'_sha256'),str),
                'Missing recovery admission proof path/hash')
        ledger.pin(proof[kind+'_path'],proof[kind+'_sha256'])


def qualify_evaluation(manifest,root=ROOT):
    binding=manifest.get('evaluation_binding')
    require(isinstance(binding,dict),'Evaluation binding is pending; --run refused')
    require(manifest.get('generation')=='v043' and manifest.get('memory_correction')==MEMORY_CORRECTION,
            'Analysis generation/memory declaration changed')
    ledger=Inputs()
    for path,value in manifest['pins'].items():ledger.pin(path,value)
    source=ledger.pin(binding['source'],binding['source_sha256'])
    ledger.pin(binding['runtime'],binding['runtime_sha256'])
    runtime_doc,frozen,completed,runs=module(source).qualify_generation(binding['runtime_sha256'])
    frozen_path=Path(root)/'contracts/mask-control-v035-readout.json'
    require(sha(frozen_path)==READOUT_SHA and frozen==read_json(frozen_path),'Original frozen readout changed')
    require(completed.get('runtime_sha256')==binding['runtime_sha256'] and
            completed.get('readout_sha256')==READOUT_SHA and completed.get('submission_started') is False,
            'Completion binding/scope mismatch')
    validate_memory_correction(runtime_doc,completed)
    validate_admission(completed,ledger)
    effective=bind_generation(frozen,runtime_doc)
    validate_qualified_payload(effective,completed,runs)
    for run in effective['future_runs']:
        receipt=runs[run['name']]['receipt']
        require(receipt.get('generation')=='v043' and receipt.get('scientific_name')==run['scientific_name'] and
                receipt.get('result_dir')==run['result_dir'],'Cross-generation result receipt refused')
        guard=receipt.get('memory_guard')
        require(isinstance(guard,dict) and 'path' in guard and 'sha256' in guard,'Missing qualified memory guard receipt')
        ledger.pin(guard['path'],guard['sha256'])
    for path in sorted((Path(root)/'migration/mask-evaluations-v043/evaluation').rglob('*.json')):ledger.pin(path)
    ledger.verify()
    return effective,completed,runs,ledger


def prepare(binding=None):
    readout=read_json(READOUT);require(sha(READOUT)==READOUT_SHA,'Frozen readout mismatch');validate_populations(readout)
    require(sha(PREDICTIONS)==PREDICTIONS_SHA,'Frozen prediction note changed')
    require(read_json(INTERPRETATION)==legacy.companion(),'Frozen interpretation changed')
    pins={**readout['pins'],str(READOUT):READOUT_SHA,str(LEGACY):LEGACY_SHA,
          str(PREDICTIONS):PREDICTIONS_SHA,str(INTERPRETATION):sha(INTERPRETATION),
          str(Path(__file__)):sha(__file__),str(ROOT/'tools/test_analyze_mask_control_v044.py'):sha(ROOT/'tools/test_analyze_mask_control_v044.py')}
    for path,value in pins.items():require(sha(path)==value,'Preparation source drift: '+path)
    if binding:
        for key,digest in [('source','source_sha256'),('runtime','runtime_sha256')]:
            require(sha(binding[key])==binding[digest],'Evaluation binding changed')
        runtime=read_json(binding['runtime']);bind_generation(readout,runtime)
        validate_memory_correction(runtime,{'memory_correction':runtime.get('memory_correction')})
    doc={'version':'v044-v043-memory-corrected-matched-mask-analysis','generation':'v043',
         'status':'PREPARED_NOT_ACTIVATED','recorded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
         'root':str(ROOT),'readout_sha256':READOUT_SHA,'immutable_statistics_sha256':LEGACY_SHA,
         'evaluation_binding':binding,'pins':pins,'execution_authorized':False,'output':str(OUTPUT),
         'memory_correction':MEMORY_CORRECTION,
         'historical_results_policy':'All six runs rerun in v043; no partial v039 results enter any estimate.',
         'requires':'Separately reviewed final v043 binding and explicit --run; all six guarded evaluations qualified.'}
    write_fresh(MANIFEST,json.dumps(doc,indent=2)+'\n')
    write_fresh(str(MANIFEST)+'.sha256',sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    return sha(MANIFEST)


def analyze(manifest_sha):
    require(sha(MANIFEST)==manifest_sha,'Analysis manifest mismatch')
    doc=read_json(MANIFEST);require(doc['root']==str(ROOT) and doc['output']==str(OUTPUT),'Analysis root/output mismatch')
    readout,completed,qualified,ledger=qualify_evaluation(doc)
    ledger.pin(MANIFEST,manifest_sha);ledger.pin(INTERPRETATION)
    resolution=module(ROOT/'tools/analyze_contract_v02_v025.py')
    validator=module(ROOT/'tools/analyze_mechanism_v031.py');producer=module(ROOT/'tools/mechanism_metrics_v032.py')
    outcomes,records,rows_by_run,tokenizer=metrics_for_runs(readout,qualified,producer,validator,ledger)
    populations={name:population_report(readout,ids,outcomes,records,rows_by_run,producer,resolution,validator)
                 for name,ids in validate_populations(readout).items()}
    inputs=prediction_inputs(populations['primary40']);labels=prediction_labels(inputs)
    interpretation=read_json(INTERPRETATION);ledger.verify()
    report={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'readout_sha256':READOUT_SHA,
        'analysis_manifest_sha256':manifest_sha,'immutable_statistics_sha256':LEGACY_SHA,
        'status':'qualified_exploratory_matched_mask_readout','evaluation_generation':'v043',
        'generation_runs':readout['future_runs'],'memory_correction':completed['memory_correction'],
        'historical_results_policy':doc['historical_results_policy'],'populations':populations,
        'auxiliary_predictions':{'inputs':inputs,'labels':labels,'interpretation':interpretation,
            'resource_context':{'legacy_training_seed_sensitive_flag':'Historical replication mismatch only; execution memory environment and initialization are confounded.',
                'matched_contrast':'Fresh M1-minus-M0 remains paired within the identical capped v043 generation.'}},
        'tokenizer':tokenizer,'method':readout['bootstrap'],'limitations':readout['exposure_and_claim_limits']+[
            'All paired results use the same v043 sandbox memory limits; earlier unguarded partial runs are excluded.',
            'Historical A0/A1/A2 runs were uncapped; fresh paired M0/M1 runs share the declared caps. Historical comparisons and fixed numerical thresholds remain descriptive, without matched historical exchangeability.'],
        'inputs':ledger.hashes,'evaluation_completion':completed,'automatic_training_or_submission':False}
    require(not OUTPUT.exists(),'Fresh v043 analysis output required');OUTPUT.mkdir()
    write_fresh(OUTPUT/'report.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_fresh(OUTPUT/'per_task.jsonl',''.join(json.dumps(v,allow_nan=False)+'\n' for _,v in sorted(records.items())))
    lines=['# Matched mask-control readout: v043','',
           'All six evaluations rerun with the declared memory guard; v042 statistical functions unchanged.','']
    for name,value in populations.items():
        p=value['primary'];lines.append(f"{name}: M1-M0 {p['effect_pp']:.3f} pp; 95% CI {p['ci95_pp']}; M0 {p['arms']['M0']['resolved_by_seed']}, M1 {p['arms']['M1']['resolved_by_seed']}.")
    lines+=['','Auxiliary numeric labels: '+json.dumps(labels),
            'The frozen P-A training_seed_sensitive flag denotes historical replication mismatch here; execution memory environment and initialization are confounded. The matched M1-minus-M0 contrast remains within the identical capped v043 generation.','',
            'Interpretations are descriptive; absolute recovery labels do not establish a positive paired effect.']
    write_fresh(OUTPUT/'report.md','\n'.join(lines)+'\n')
    for p in list(OUTPUT.iterdir()):write_fresh(str(p)+'.sha256',sha(p)+'  '+p.name+'\n')
    return {'status':report['status'],'output':str(OUTPUT),'report_sha256':sha(OUTPUT/'report.json')}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--run',action='store_true');ap.add_argument('--manifest-sha');args=ap.parse_args()
    require(not(args.prepare and args.run),'Prepare and run are separate')
    if args.prepare:print(json.dumps({'status':'prepared_only','manifest_sha256':prepare(),'outcomes_read':False}))
    elif args.run:
        require(args.manifest_sha is not None,'Explicit manifest SHA required');print(json.dumps(analyze(args.manifest_sha)))
    else:
        require(args.manifest_sha is not None and sha(MANIFEST)==args.manifest_sha,'Inspection requires manifest SHA')
        doc=read_json(MANIFEST);print(json.dumps({'status':'prepared_only','generation':doc['generation'],
            'evaluation_binding_ready':doc['evaluation_binding'] is not None,'outcomes_read':False}))


if __name__=='__main__':main()
