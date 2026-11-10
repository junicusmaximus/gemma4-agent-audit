#!/usr/bin/env python3
"""CPU inspection by default; future approved matched controls use frozen M0 denominators."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import train_lora as original
from prepare_mask_control_v035 import ROOT, OUT, CONTRACT, sha, digest, shifted, target_identity, make_groups


def verify_pair(contract):
    """Load every chunk, fail closed on identity/order/hash/count discrepancies; no model/GPU calls."""
    manifest = json.loads((OUT/'manifest.json').read_text())
    if sha(OUT/'manifest.json') != contract['manifest_sha256']:
        raise ValueError('manifest changed')
    for mapping in (manifest['source_hashes'], manifest['artifact_hashes']):
        for path, expected in mapping.items():
            if sha(path) != expected:
                raise ValueError('hash mismatch: '+path)
    ledgers = [json.loads(line) for line in open(OUT/'identity_ledger.jsonl')]
    seqs = [[], []]
    index, trajectory_ids = 0, []
    with open(OUT/'M0.jsonl') as f0, open(OUT/'M1.jsonl') as f1:
        from itertools import zip_longest
        for line0, line1 in zip_longest(f0, f1):
            if line0 is None or line1 is None:
                raise ValueError('record count mismatch')
            r0, r1 = json.loads(line0), json.loads(line1)
            if r0['traj_id'] != r1['traj_id'] or r0['source_line'] != r1['source_line']:
                raise ValueError('trajectory order mismatch')
            trajectory_ids.append(r0['traj_id'])
            if len(r0['chunks']) != len(r1['chunks']):
                raise ValueError('chunk boundary mismatch')
            for ci, (c0, c1) in enumerate(zip(r0['chunks'], r1['chunks'])):
                entry = ledgers[index]
                if (entry['sequence_index'], entry['traj_id'], entry['chunk_index'], entry['source_line']) != (index, r0['traj_id'], ci, r0['source_line']):
                    raise ValueError('source/chunk/order ledger mismatch')
                if {k:v for k,v in c0.items() if k not in ('labels','n_loss_tokens')} != {k:v for k,v in c1.items() if k not in ('labels','n_loss_tokens')}:
                    raise ValueError('inputs or chunk structure changed')
                ids, lab0, lab1 = c0['input_ids'], c0['labels'], c1['labels']
                if not len(ids) == len(lab0) == len(lab1) or len(ids) > 16384:
                    raise ValueError('length gate failed; skipping forbidden')
                reconstructed = list(lab1)
                for pos, token in entry['changed_original_positions']:
                    if lab0[pos] != token or lab1[pos] != -100 or token == -100:
                        raise ValueError('changed position mismatch')
                    reconstructed[pos] = token
                if reconstructed != lab0:
                    raise ValueError('exact target conservation failed')
                for key, value in [('input_sha256',ids),('m0_labels_sha256',lab0),('m1_labels_sha256',lab1)]:
                    if digest(value) != entry[key]:
                        raise ValueError('ledger identity mismatch: '+key)
                for variant, labs in enumerate((lab0,lab1)):
                    if shifted(labs) != entry['m%d_shifted_targets'%variant]:
                        raise ValueError('causal shifted count mismatch')
                    if target_identity(labs) != entry['m%d_target_identity_sha256'%variant]:
                        raise ValueError('target identity mismatch')
                    if any(y != -100 and y != x for x,y in zip(ids,labs)):
                        raise ValueError('input/target alignment mismatch')
                    seqs[variant].append((r0['traj_id'], np.asarray(ids,dtype=np.int32), np.asarray(labs,dtype=np.int32)))
                index += 1
    groups = json.loads((OUT/'update_groups.json').read_text())
    if index != len(ledgers) or groups != make_groups(ledgers):
        raise ValueError('update-group identity mismatch')
    if (len(trajectory_ids), index, sum(x['m0_shifted_targets'] for x in ledgers)) != (726,1661,4391649):
        raise ValueError('frozen population mismatch')
    return manifest, seqs, groups


def fixed_denominator_train(args, seqs, groups, model):
    """Original train_lora.train loop, with explicit frozen groups and M0 denominators only."""
    inner = model.base_model.model
    softcap = inner.config.get_text_config().final_logit_softcapping
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
    warm = max(1, round(args.warmup_ratio * len(groups)))
    logs, seen, started = [], 0, time.time()
    for step, frozen in enumerate(groups):
        group = [seqs[i] for i in frozen['sequence_indices']]
        denominator = frozen['original_denominator']
        if denominator <= 0:
            raise ValueError('zero original denominator')
        for g in opt.param_groups:
            g['lr'] = original.lr_at(step, len(groups), warm, args.lr)
        ts, loss_sum, tokens, remaining = time.time(), 0.0, 0, 0
        for _, ids, lab in group:
            value, count = original.sequence_loss(inner, torch.from_numpy(ids).long().to(args.device),
                                                  torch.from_numpy(lab).long().to(args.device), softcap)
            if count == 0:
                raise ValueError('zero-target sequence; skipping forbidden')
            (value / denominator).backward()
            loss_sum += float(value.detach()); tokens += len(ids); remaining += count
        expected = denominator if args.variant == 'M0' else frozen['m1_remaining_targets']
        if remaining != expected:
            raise ValueError('update exposure mismatch')
        norm = float(torch.nn.utils.clip_grad_norm_(params,args.max_grad_norm))
        opt.step(); opt.zero_grad(set_to_none=True)
        elapsed = time.time()-ts; seen += tokens
        row = {'step':step+1,'loss':round(loss_sum/denominator,5),'lr':opt.param_groups[0]['lr'],
               'grad_norm':round(norm,4),'sequences':len(group),'tokens':tokens,'loss_tokens':denominator,
               'remaining_loss_tokens':remaining,'seconds':round(elapsed,2),'tokens_per_s':round(tokens/elapsed,1)}
        if args.device == 'cuda':
            row['peak_alloc_gib'] = round(torch.cuda.max_memory_allocated()/original.GIB,2)
        logs.append(row);print(json.dumps(row),flush=True)
    return logs,time.time()-started,seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contract',type=Path,default=CONTRACT)
    ap.add_argument('--variant',choices=['M0','M1'],required=True)
    ap.add_argument('--execute',action='store_true',help='requires a separately reviewed approved contract')
    opts = ap.parse_args()
    contract = json.loads(opts.contract.read_text())
    manifest, sequences, groups = verify_pair(contract)
    print(json.dumps({'mode':'CPU_INSPECTION','variant':opts.variant,'chunks':len(sequences[0]),'updates':len(groups),
                      'm0_shifted_targets':manifest['m0_shifted_targets'], 'm1_shifted_targets':manifest['m1_shifted_targets'],
                      'all_identity_and_hash_gates':'PASS'}),flush=True)
    if not opts.execute:
        return
    if contract.get('status') != 'APPROVED' or contract.get('execute_training') is not True:
        raise SystemExit('REFUSED: contract is an unapproved draft; no model/GPU operation performed')
    for key in ('source_hashes','artifact_hashes','hyperparameters','fresh_outputs','base','probe_dir'):
        if contract[key] != manifest[key]:
            raise ValueError('approved contract drift: '+key)
    args = argparse.Namespace(**manifest['hyperparameters'],variant=opts.variant,base=manifest['base'],
                              out=manifest['fresh_outputs'][opts.variant],device='cuda',
                              served_base_name=original.SERVED_BASE)
    for suffix in ('','.meta.json','.train_log.jsonl'):
        if Path(args.out+suffix).exists():
            raise FileExistsError('fresh output required: '+args.out+suffix)
    original.PROBE_DIR = manifest['probe_dir']
    # The historical trainer seeded order only. These fresh controls explicitly seed
    # their common LoRA initialization; historical A1 initialization is unproven.
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    original.check_gpu(2.0)
    torch.cuda.manual_seed_all(args.seed)
    model = original.build_model(args)
    torch.cuda.reset_peak_memory_stats()
    initial = {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}
    import hashlib
    init_hash = hashlib.sha256()
    for name,tensor in sorted(initial.items()):
        init_hash.update(name.encode());init_hash.update(tensor.view(torch.uint8).numpy().tobytes())
    del initial
    other = 'M1' if opts.variant == 'M0' else 'M0'
    other_meta = Path(manifest['fresh_outputs'][other]+'.meta.json')
    if other_meta.exists() and json.loads(other_meta.read_text())['initial_trainable_parameters_sha256'] != init_hash.hexdigest():
        raise ValueError('matched initialization hash mismatch; training refused')
    logs,seconds,seen = fixed_denominator_train(args,sequences[0 if opts.variant=='M0' else 1],groups,model)
    n_tensors,names_match = original.save_adapter(args,model)
    Path(args.out+'.train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs))
    Path(args.out+'.meta.json').write_text(json.dumps({'args':vars(args),'contract_sha256':sha(opts.contract),
        'initial_trainable_parameters_sha256':init_hash.hexdigest(),'seconds':seconds,'tokens':seen,
        'adapter_tensors':n_tensors,'names_match_probe':names_match},indent=2)+'\n')


if __name__ == '__main__':
    main()
