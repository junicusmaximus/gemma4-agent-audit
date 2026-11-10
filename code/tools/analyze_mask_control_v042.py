#!/usr/bin/env python3
"""Frozen matched-mask readout; no inference, submission, or conditional training."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True

ROOT = Path('PROJECT_ROOT/gemma4')
READOUT = ROOT/'contracts/mask-control-v035-readout.json'
READOUT_SHA = '6319262b25c19ddcbb0e34a33f5e97325a7d6117a7b8faad8d4a4c7ff772c909'
PREDICTIONS = ROOT.parent/'briefing/research/gemma-4-2026/prereg-2026-11-05-mask-control-mechanism-predictions.md'
PREDICTIONS_SHA = 'a24baec74fc5ddf899cfa63c89ab4fb211e000ae8faa7fc234ae8f1618ef8bfe'
MANIFEST = ROOT/'contracts/mask-control-analysis-v042-DRAFT.json'
INTERPRETATION = ROOT/'contracts/mask-control-mechanism-interpretation-v041.json'
ARMS, SEEDS, DRAWS, RNG_SEED = ('M0','M1'), (1,2,3), 20000, 20261103
STRATA = ('error','silent','normal')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result,'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=invalid)


class Inputs:
    def __init__(self): self.hashes = {}

    def pin(self,path,expected=None):
        path=Path(path).resolve();value=sha(path)
        require(expected is None or value==expected,'Input hash mismatch: '+str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)]==value,'Input changed: '+str(path))
        self.hashes[str(path)]=value
        return path

    def verify(self):
        for path,value in self.hashes.items():
            require(sha(path)==value,'Input changed after readout: '+path)


def module(path):
    spec=importlib.util.spec_from_file_location('v042_'+Path(path).stem,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def validate_populations(doc):
    sizes={'primary40':40,'all48_sensitivity':48,'control_clean36_sensitivity':36}
    require(set(doc['populations'])==set(sizes),'Population names changed')
    result={name:sorted(value['task_ids']) for name,value in doc['populations'].items()}
    for name,n in sizes.items():
        require(len(result[name])==len(set(result[name]))==n,'Population size/duplicates: '+name)
    require(set(result['control_clean36_sensitivity']) < set(result['primary40']) < set(result['all48_sensitivity']),
            'Population nesting changed')
    exclusions={v['task_id'] for v in doc['populations']['control_clean36_sensitivity']['excluded_from_primary40']}
    require(set(result['primary40'])-set(result['control_clean36_sensitivity'])==exclusions,'Clean36 exclusions changed')
    return result


def validate_resolution_grid(runs,ids):
    require(set(runs)=={(a,s) for a in ARMS for s in SEEDS},'Incomplete arm/seed grid')
    require(len(ids)==len(set(ids))>0,'Invalid task grid')
    for values in runs.values():
        require(set(values)==set(ids),'Incomplete task grid')
        require(all(type(v) is bool for v in values.values()),'Resolution must be literal boolean')


def resolution_report(runs,ids,bootstrap,draws=DRAWS,seed=RNG_SEED):
    ids=sorted(ids)
    arrays={a:np.array([[runs[a,s][task] for s in SEEDS] for task in ids],dtype=bool) for a in ARMS}
    result=bootstrap(arrays['M1'],arrays['M0'],draws=draws,seed=seed)
    result.update(task_count=len(ids),arms={a:{'resolved_by_seed':arrays[a].sum(axis=0).tolist(),
        'denominator_per_seed':len(ids),'three_seed_mean_resolution':float(arrays[a].mean())} for a in ARMS},
        per_task=[{'instance_id':task,**{a:arrays[a][i].tolist() for a in ARMS}} for i,task in enumerate(ids)])
    return result


def paired_slots(treated,baseline,bootstrap,draws=DRAWS,seed=RNG_SEED,include_sensitivity=True):
    require(set(treated)==set(baseline),'Task IDs differ')
    masks={};paired=[];tmeans=[];bmeans=[];details=[]
    for task in sorted(treated):
        t,b=treated[task],baseline[task]
        require(len(t)==len(b)==3,'Expected three seed slots')
        require(all(v is None or (type(v) in (int,float) and math.isfinite(v)) for v in [*t,*b]),'Invalid metric value')
        mask=[x is not None and y is not None for x,y in zip(t,b)];masks[task]=mask
        if any(mask):
            tv=[x if ok else None for x,ok in zip(t,mask)];bv=[x if ok else None for x,ok in zip(b,mask)]
            tm=float(np.mean([x for x in tv if x is not None]));bm=float(np.mean([x for x in bv if x is not None]))
            paired.append(task);tmeans.append(tm);bmeans.append(bm)
            details.append(dict(instance_id=task,treated_seed_values=tv,baseline_seed_values=bv,treated_task_mean=tm,baseline_task_mean=bm))
    result=bootstrap(tmeans,bmeans,draws=draws,seed=seed)
    result.update(paired_task_ids=paired,support_masks=masks,common_task_seed_pairs=sum(sum(v) for v in masks.values()),
                  per_task=details,seed_aggregation='paired common seed slots first, then unweighted task means',
                  per_arm_eligible_tasks={a:sum(any(v is not None for v in values) for values in arm.values())
                                          for a,arm in [('M1',treated),('M0',baseline)]})
    if include_sensitivity:
        complete=[k for k in masks if all(masks[k])]
        result['all_three_common_seed_sensitivity']=paired_slots({k:treated[k] for k in complete},
            {k:baseline[k] for k in complete},bootstrap,draws,seed,include_sensitivity=False)
    return result


def mean_defined(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def mean_three(values):
    return float(np.mean(values)) if len(values)==3 and all(v is not None for v in values) else None


def prediction_labels(values):
    b,t=values['M0'],values['M1']
    finite=lambda x:type(x) in (int,float) and math.isfinite(x)
    ge=lambda x,y:x>y or math.isclose(x,y,rel_tol=0,abs_tol=1e-12)
    le=lambda x,y:x<y or math.isclose(x,y,rel_tol=0,abs_tol=1e-12)
    lt=lambda x,y:x<y and not math.isclose(x,y,rel_tol=0,abs_tol=1e-12)
    valid_counts=lambda x:isinstance(x,(list,tuple)) and len(x)==3 and all(type(v) is int and 0<=v<=40 for v in x)
    a='UNDEFINED'
    if all(finite(b.get(k)) for k in ('think','loop','submit')) and valid_counts(b.get('solved_by_seed',[])):
        if lt(b['think'],.05) and ge(b['loop'],.40) and le(b['submit'],.60) and max(b['solved_by_seed'])<=1: a='PASS'
        elif b['think']>.50 and not math.isclose(b['think'],.50,rel_tol=0,abs_tol=1e-12) or sum(v>=3 for v in b['solved_by_seed'])>=2: a='FAIL'
        else: a='PARTIAL'
    pb='UNDEFINED' if not finite(t.get('think')) else 'HOLDS' if ge(t['think'],.50) else 'FAILS' if lt(t['think'],.10) else 'INTERMEDIATE'
    pc='C3' if pb=='FAILS' else 'NOT_READ'
    pooled=sum(t['solved_by_seed']) if valid_counts(t.get('solved_by_seed',[])) else None
    if pb=='HOLDS':
        c2=any(finite(t.get(k)) and ge(t[k],v) for k,v in [('loop',.30),('repro',.40)]) or (pooled is not None and pooled<=2)
        support=t.get('m5_error_n');supported=isinstance(support,(list,tuple)) and len(support)==3 and all(type(v) is int and v>=10 for v in support)
        defined=all(finite(t.get(k)) for k in ('loop','submit','repro','m5_error')) and pooled is not None and supported
        if c2: pc='C2'
        elif not defined: pc='UNDEFINED'
        elif le(t['loop'],.10) and ge(t['submit'],.90) and le(t['m5_error'],.10) and le(t['repro'],.20) and pooled>=7: pc='C1'
        else: pc='INTERMEDIATE'
    pd='UNDEFINED';delta=None
    if finite(b.get('repro')) and finite(t.get('repro')):
        delta=t['repro']-b['repro'];pd='HOLDS' if le(abs(delta),.15) else 'DROP_GT_0_30' if lt(delta,-.30) else 'NO_READING'
    return {'P-A':a,'P-B':pb,'P-C':pc,'P-D':pd,'P-D_delta':delta,'training_seed_sensitive':a=='FAIL',
            'replication_weak':a in ('PARTIAL','UNDEFINED'),'absolute_thresholds_do_not_require_contrast_improvement':True,
            'interpretation':{'C1':'descriptive recovery under absolute thresholds','C2':'descriptive partial recovery',
                'C3':'no observed channel restoration'}.get(pc,'inconclusive or not read'),
            'boundary_numerics':'absolute tolerance1e-12 only for floating-point equality at the unchanged thresholds'}


def validate_qualified_payload(readout,completed,runs):
    require(set(runs)=={r['name'] for r in readout['future_runs']},'Expected all six qualified runs')
    require(completed.get('cleanup_completed') is True,'Evaluation cleanup did not qualify')
    require(completed.get('analysis_pending') is True,'Unexpected analysis lifecycle')
    require(completed.get('status')=='all_six_evaluations_qualified','Unexpected qualification status')
    ids=readout['populations']['all48_sensitivity']['task_ids']
    values={(r['arm'],r['decoding_seed']):{task:row['resolved'] for task,row in runs[r['name']]['rows'].items()} for r in readout['future_runs']}
    validate_resolution_grid(values,ids)


def qualify_evaluation(manifest,root=ROOT):
    binding=manifest.get('evaluation_binding')
    require(isinstance(binding,dict),'Evaluation binding is pending; --run refused')
    ledger=Inputs()
    for path,value in manifest['pins'].items(): ledger.pin(path,value)
    source=ledger.pin(binding['source'],binding['source_sha256'])
    ledger.pin(binding['runtime'],binding['runtime_sha256'])
    runtime=module(source)
    runtime_doc,readout,completed,runs=runtime.qualify_generation(binding['runtime_sha256'])
    frozen_path=Path(root)/'contracts/mask-control-v035-readout.json'
    require(sha(frozen_path)==READOUT_SHA and readout==read_json(frozen_path),'Readout changed')
    require(completed.get('runtime_sha256')==binding['runtime_sha256'] and completed.get('readout_sha256')==READOUT_SHA and completed.get('submission_started') is False,'Completion binding/scope mismatch')
    validate_qualified_payload(readout,completed,runs)
    for path in sorted((Path(root)/'migration/mask-evaluations-v039/evaluation').rglob('*.json')): ledger.pin(path)
    ledger.verify()
    return readout,completed,runs,ledger


def metrics_for_runs(readout,qualified,producer,validator,ledger):
    tokenize=producer.load_tokenizer(readout['secondary_mechanisms']['tokenizer'])
    tokenizer=producer.tokenizer_provenance(readout['secondary_mechanisms']['tokenizer'])
    records={};outcomes={};rows_by_run={}
    ids=readout['populations']['all48_sensitivity']['task_ids']
    for run in readout['future_runs']:
        arm,seed=run['arm'],run['decoding_seed'];folder=Path(run['result_dir'])
        receipt=qualified[run['name']]['receipt']
        ledger.pin(folder/'task_results.jsonl',receipt['sha256']);rows=qualified[run['name']]['rows'];rows_by_run[arm,seed]=rows
        outcomes[arm,seed]={task:rows[task]['resolved'] for task in ids}
        for task in sorted(ids):
            ledger.pin(folder/'traces'/f'trace_{task}.json',receipt['traces'][task]['sha256'])
            trace,reason=producer.load_trace(folder,task)
            require(reason is None,'Qualified trace became unavailable: '+task)
            record={'run':run['name'],'instance_id':task,'trace_missing':False,'trace_error':None,
                    **producer.compute_task_metrics(trace,loop_threshold=5,max_output_tokens=8192,tokenize=tokenize)}
            validator.validate_row(record)
            require(record['n_tool_calls']==receipt['traces'][task]['parsed_tool_calls'],'Parsed call count changed')
            records[arm,seed,task]=record
    validate_resolution_grid(outcomes,ids)
    return outcomes,records,rows_by_run,tokenizer


def secondary_report(records,ids,validator,draws=DRAWS,seed=RNG_SEED):
    specs=validator.metric_specs()
    specs.update({'M2.distinct_ratio':('proportion',lambda r:r['distinct_ratio']),
                  **{f'M6.{k}':('tokens' if 'tokens' in k else 'characters',lambda r,k=k:r[k])
                     for k in ('thought_tokens_median','thought_tokens_max','thought_chars_median')}})
    result={}
    for name,(units,extract) in specs.items():
        values={a:{task:[extract(records[a,s,task]) for s in SEEDS] for task in sorted(ids)} for a in ARMS}
        item=paired_slots(values['M1'],values['M0'],validator.paired_task_bootstrap,draws,seed);item['units']=units
        if units=='proportion':
            item['effect_pp']=None if item['effect'] is None else item['effect']*100
            item['ci95_pp']=None if item['ci95'] is None else [v*100 for v in item['ci95']]
        if name.startswith('M5.'):
            stratum=name.rsplit('.',1)[1]
            item['eligibility_by_task']={task:{a:[records[a,s,task]['nonrecovery'][stratum] for s in SEEDS] for a in ARMS} for task in sorted(ids)}
            item['unknown_excluded_by_task']={task:{a:[records[a,s,task]['nonrecovery']['unknown_excluded'] for s in SEEDS] for a in ARMS} for task in sorted(ids)}
        result[name]=item
    return result


def population_report(readout,ids,outcomes,records,rows_by_run,producer,resolution,validator):
    summaries={a:[producer.summarize_run(f'{a}_s{s}',[rows_by_run[a,s][task] for task in ids],
                [records[a,s,task] for task in ids]) for s in SEEDS] for a in ARMS}
    return dict(primary=resolution_report(outcomes,ids,resolution.paired_bootstrap),
                secondary=secondary_report(records,ids,validator),per_run_summaries=summaries)


def prediction_inputs(population):
    result={}
    for arm in ARMS:
        summaries=population['per_run_summaries'][arm]
        result[arm]={key:mean_three([row[field] for row in summaries])
                     for key,field in [('think','think_rate_mean'),('loop','loop_rate'),('submit','submit_rate')]}
        result[arm].update(repro=mean_three([v['opener_shares']['repro_script_opener'] for v in summaries]),
            m5_error=mean_three([v['nonrecovery']['error']['mean_repeat_rate'] for v in summaries]),
            m5_error_n=[v['nonrecovery']['error']['n_tasks'] for v in summaries],
            solved_by_seed=population['primary']['arms'][arm]['resolved_by_seed'])
    return result


def write_fresh(path,text):
    with Path(path).open('x') as stream: stream.write(text)


def companion():
    return {'status':'PROSPECTIVE_INTERPRETATION_CLARIFICATION_NO_THRESHOLD_CHANGE',
        'predictions_path':str(PREDICTIONS),'predictions_sha256':PREDICTIONS_SHA,'readout_sha256':READOUT_SHA,
        'scope':'Exploration-informed prospective predictions; metadata mtime is not independent timestamp attestation.',
        'numeric_predicates':'Original P-A/P-B/P-C/P-D thresholds preserved; explicit null/inconclusive handling added.',
        'interpretation':{'C1':'Descriptive recovery; report M1-minus-M0 separately. Absolute thresholds do not require benefit.',
            'C2':'Descriptive partial recovery; no claim about the necessity of thinking or residual causes.',
            'C3':'No observed channel restoration under this masking intervention.'},
        'undefined_rule':'A required unavailable aggregate yields UNDEFINED; P-C not read without P-B HOLDS, except C3 when FAILS. M5 support below10 tasks in any seed prevents C1, while independently evaluable C2 remains readable.',
        'limits':['No dominant-cause, mediation, tolerated-orphaning, training-seed generality or external-generalization claim.',
                  'No conditional H1 training decision is made by this analyzer.']}


def prepare(binding=None):
    readout=read_json(READOUT);require(sha(READOUT)==READOUT_SHA,'Frozen readout mismatch');validate_populations(readout)
    require(sha(PREDICTIONS)==PREDICTIONS_SHA,'Frozen prediction note changed')
    if not INTERPRETATION.exists(): write_fresh(INTERPRETATION,json.dumps(companion(),indent=2)+'\n')
    require(read_json(INTERPRETATION)==companion(),'Interpretation companion changed')
    pins={**readout['pins'],str(READOUT):READOUT_SHA,str(PREDICTIONS):sha(PREDICTIONS),
          str(INTERPRETATION):sha(INTERPRETATION),str(Path(__file__)):sha(__file__),
          str(ROOT/'tools/test_analyze_mask_control_v042.py'):sha(ROOT/'tools/test_analyze_mask_control_v042.py')}
    for path,value in pins.items():require(sha(path)==value,'Preparation source drift: '+path)
    if binding:
        for key,digest in [('source','source_sha256'),('runtime','runtime_sha256')]:require(sha(binding[key])==binding[digest],'Evaluation binding changed')
    doc={'version':'v042-matched-mask-analysis','status':'PREPARED_NOT_ACTIVATED','root':str(ROOT),
         'recorded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'readout_sha256':READOUT_SHA,
         'evaluation_binding':binding,'pins':pins,'execution_authorized':False,
         'output':readout['future_analysis_dir'],'requires':'Separately reviewed final v039 binding and explicit --run with manifest hash; qualified complete six-run generation.'}
    write_fresh(MANIFEST,json.dumps(doc,indent=2)+'\n')
    write_fresh(str(MANIFEST)+'.sha256',sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    return sha(MANIFEST)


def analyze(manifest_sha):
    require(sha(MANIFEST)==manifest_sha,'Analysis manifest mismatch')
    doc=read_json(MANIFEST);require(doc['root']==str(ROOT),'Analysis root mismatch')
    readout,completed,qualified,ledger=qualify_evaluation(doc)
    ledger.pin(MANIFEST,manifest_sha);ledger.pin(INTERPRETATION)
    populations=validate_populations(readout)
    resolution=module(ROOT/'tools/analyze_contract_v02_v025.py')
    validator=module(ROOT/'tools/analyze_mechanism_v031.py');producer=module(ROOT/'tools/mechanism_metrics_v032.py')
    outcomes,records,rows_by_run,tokenizer=metrics_for_runs(readout,qualified,producer,validator,ledger)
    result={name:population_report(readout,ids,outcomes,records,rows_by_run,producer,resolution,validator)
            for name,ids in populations.items()}
    inputs=prediction_inputs(result['primary40']);labels=prediction_labels(inputs)
    interpretation=read_json(INTERPRETATION)
    ledger.verify()
    report={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'readout_sha256':READOUT_SHA,
        'analysis_manifest_sha256':manifest_sha,'status':'qualified_exploratory_matched_mask_readout',
        'populations':result,'auxiliary_predictions':{'inputs':inputs,'labels':labels,'interpretation':interpretation},
        'tokenizer':tokenizer,'method':readout['bootstrap'],'limitations':readout['exposure_and_claim_limits'],
        'inputs':ledger.hashes,'evaluation_completion':completed,'automatic_training_or_submission':False}
    output=Path(doc['output']);require(not output.exists(),'Fresh analysis output required');output.mkdir()
    write_fresh(output/'report.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_fresh(output/'per_task.jsonl',''.join(json.dumps(v,allow_nan=False)+'\n' for k,v in sorted(records.items())))
    lines=['# Matched mask-control readout','', 'Known-holdout exploratory intervention, one paired training seed.','']
    for name,value in result.items():
        p=value['primary'];lines.append(f"{name}: M1-M0 {p['effect_pp']:.3f} pp; 95% CI {p['ci95_pp']}; counts M0 {p['arms']['M0']['resolved_by_seed']}, M1 {p['arms']['M1']['resolved_by_seed']}.")
    lines+=['', 'Auxiliary numeric labels: '+json.dumps(labels), '', 'Interpretations are descriptive; absolute recovery labels do not establish a positive paired effect.']
    write_fresh(output/'report.md','\n'.join(lines)+'\n')
    for p in list(output.iterdir()):write_fresh(str(p)+'.sha256',sha(p)+'  '+p.name+'\n')
    return {'status':report['status'],'output':str(output),'report_sha256':sha(output/'report.json')}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--run',action='store_true');ap.add_argument('--manifest-sha')
    args=ap.parse_args();require(not(args.prepare and args.run),'Prepare and run are separate')
    if args.prepare: print(json.dumps({'status':'prepared_only','manifest_sha256':prepare(),'outcomes_read':False}))
    elif args.run:
        require(args.manifest_sha is not None,'Explicit manifest SHA required');print(json.dumps(analyze(args.manifest_sha)))
    else:
        require(args.manifest_sha is not None and sha(MANIFEST)==args.manifest_sha,'Inspection requires manifest SHA')
        doc=read_json(MANIFEST);print(json.dumps({'status':'prepared_only','evaluation_binding_ready':doc['evaluation_binding'] is not None,'outcomes_read':False}))


if __name__=='__main__': main()
