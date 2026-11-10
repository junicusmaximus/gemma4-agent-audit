#!/usr/bin/env python3
"""Prepare unapproved matched A1 empty-scaffold controls, without model loading."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path('PROJECT_ROOT/gemma4')
sys.path.insert(0, str(ROOT / 'tools'))
import sft_example_audit as audit
ARM = 'arm_ordinary_tokmatched'
EXPECTED = (726, 1661, 4391649)
SEED = 20261103
DATA = ROOT / 'sft/data/trajectories.jsonl'
ARMS = ROOT / 'sft/data/arms.json'
OUT = ROOT / 'sft/data/mask_control_v035'
MARKERS = ROOT / 'analysis/sft-audit-v033/markers.json'
TOKENIZER = ROOT / 'models/gemma-4-e4b-it-qat-w4a16-ct'
CONTRACT = ROOT / 'contracts/mask-control-v035-DRAFT.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def dump(value):
    return json.dumps(value, separators=(',', ':'))


def shifted(labels):
    return sum(x != -100 for x in labels[1:])


def target_identity(labels, positions=None):
    positions = range(1, len(labels)) if positions is None else positions
    return digest([[i, labels[i]] for i in positions if i > 0 and labels[i] != -100])


def mask_chunk(chunk, row, markers, role_ids):
    """Only exact run-prefix empty scaffolds with proven prompt boundaries are eligible."""
    ids, original = chunk['input_ids'], chunk['labels']
    if len(ids) != len(original) or any(y != -100 and y != x for x, y in zip(ids, original)):
        raise ValueError('input/label identity mismatch')
    if len(ids) > 16384:
        raise ValueError('overlength chunk; sample skipping is forbidden')
    masked = list(original)
    changed, nonempty, decisions = [], [], []
    opener = list(markers.channel_open_ids)
    for run in row['runs']:
        start, length = run['start'], run['length']
        end = start + length
        stats = audit.run_stats(ids[start:end], markers)
        if stats['has_channel'] and not stats['empty_thought']:
            nonempty.extend(range(start, end))
        reason = 'nonempty_or_no_channel'
        span = []
        if stats['empty_thought']:
            n = len(opener) + 1 if stats['opener_in_run'] else 1
            expected = opener + [markers.channel_close] if stats['opener_in_run'] else [markers.channel_close]
            prefix = list(role_ids) if stats['opener_in_run'] else opener
            p = start - len(prefix)
            valid = p >= 0 and ids[p:start] == prefix and all(y == -100 for y in original[p:start])
            if not stats['opener_in_run']:
                # Later prompt opener must follow an actual no-loss tool-response close,
                # or a fresh assistant role prefix. Literal delimiters inside loss stay intact.
                before = p - 1
                role_before = p - len(role_ids)
                valid = valid and ((before >= 0 and ids[before] == markers.resp_close and original[before] == -100)
                    or (role_before >= 0 and ids[role_before:p] == list(role_ids)
                        and all(y == -100 for y in original[role_before:p])))
            if row['alignment_uncertain']:
                reason = 'unknown_alignment_retained'
            elif not valid or ids[start:start+n] != expected or n > length:
                reason = 'unsafe_boundary_retained'
            else:
                span = list(range(start, start+n))
                reason = 'generated_opener' if stats['opener_in_run'] else 'prompt_opener_close_only'
                for pos in span:
                    if original[pos] == -100:
                        raise ValueError('attempt to mask an original ignored position')
                    masked[pos] = -100
                    changed.append([pos, original[pos]])
        decisions.append({'start': start, 'length': length, 'reason': reason, 'masked_positions': span})
    if shifted(original) <= 0 or shifted(masked) <= 0:
        raise ValueError('zero-target chunk; skipping forbidden')
    changed_set = {i for i, _ in changed}
    if any((b != a and (i not in changed_set or a == -100 or b != -100))
           for i, (a, b) in enumerate(zip(original, masked))):
        raise ValueError('unapproved label change')
    if target_identity(original, nonempty) != target_identity(masked, nonempty):
        raise ValueError('nonempty target identities changed')
    if any(masked[i] != original[i] for i in range(len(ids)) if i not in changed_set):
        raise ValueError('target identity conservation failed')
    ledger = {'input_sha256': digest(ids), 'm0_labels_sha256': digest(original),
              'm1_labels_sha256': digest(masked), 'changed_original_positions': changed,
              'm0_shifted_targets': shifted(original), 'm1_shifted_targets': shifted(masked),
              'm0_target_identity_sha256': target_identity(original),
              'm1_target_identity_sha256': target_identity(masked),
              'nonempty_target_identity_sha256': target_identity(original, nonempty),
              'nonempty_targets_after_shift': sum(i > 0 for i in nonempty), 'runs': decisions}
    return masked, ledger


def make_groups(ledgers):
    order = list(range(len(ledgers)))
    random.Random(SEED).shuffle(order)
    groups = []
    for s in range(0, len(order), 8):
        group = order[s:s+8]
        original = sum(ledgers[i]['m0_shifted_targets'] for i in group)
        remaining = sum(ledgers[i]['m1_shifted_targets'] for i in group)
        if original <= 0 or remaining <= 0:
            raise ValueError('zero denominator/update loss is forbidden')
        groups.append({'update': len(groups)+1, 'sequence_indices': group,
                       'original_denominator': original, 'm1_remaining_targets': remaining})
    return groups


def prepare():
    if OUT.exists() or CONTRACT.exists():
        raise FileExistsError('fresh output and draft paths required')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER / 'tokenizer.json'))
    markers = audit.load_markers(str(MARKERS))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    if encode('<|channel>thought\n') != list(markers.channel_open_ids):
        raise ValueError('tokenizer/opener mismatch')
    if any(encode(text) != [token] for text, token in zip(audit.MARKER_KEYS, markers[:7])):
        raise ValueError('tokenizer/marker mismatch')
    role = encode('<|turn>model\n')
    wanted = json.loads(ARMS.read_text())[ARM]
    if len(wanted) != len(set(wanted)) or len(wanted) != EXPECTED[0]:
        raise ValueError('unexpected frozen arm membership')
    pins = [DATA, ARMS, MARKERS, ROOT/'tools/sft_example_audit.py', ROOT/'sft/train_lora.py',
            ROOT/'sft/build_sft.py', ROOT/'sft/runs/A1.meta.json', Path(__file__),
            ROOT/'sft/train_mask_control_v035.py', ROOT/'sft/tests/test_mask_control_v035.py',
            ROOT/'tools/test_sft_example_audit.py', ROOT/'tools/README-sft-audit.md',
            ROOT/'models/gemma-4-e4b-it/config.json', ROOT/'models/gemma-4-e4b-it/model.safetensors',
            ROOT/'subs/e4b_probe/adapters/probe_lora/adapter_config.json',
            ROOT/'subs/e4b_probe/adapters/probe_lora/adapter_model.safetensors']
    pins += sorted(p for p in TOKENIZER.iterdir() if p.is_file() and p.name in
                   ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'processor_config.json'))
    hashes = {str(p): sha(p) for p in pins}
    OUT.mkdir()
    ledgers, seen, reasons = [], [], collections.Counter()
    with open(DATA) as source, open(OUT/'M0.jsonl', 'x') as m0, open(OUT/'M1.jsonl', 'x') as m1, open(OUT/'identity_ledger.jsonl', 'x') as lf:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            if record['traj_id'] not in wanted:
                continue
            seen.append(record['traj_id'])
            _, rows = audit.audit_trajectory(record, {ARM: frozenset(wanted)}, markers)
            outputs = [{'traj_id': record['traj_id'], 'source_line': line_number, 'chunks': []} for _ in range(2)]
            for ci, (chunk, row) in enumerate(zip(record['chunks'], rows)):
                labels, ledger = mask_chunk(chunk, row, markers, role)
                ledger.update(source_line=line_number, traj_id=record['traj_id'], chunk_index=ci,
                              sequence_index=len(ledgers), chain=chunk['chain'], steps=chunk['steps'])
                ledgers.append(ledger)
                reasons.update(r['reason'] for r in ledger['runs'])
                lf.write(dump(ledger)+'\n')
                for output, labs in zip(outputs, (chunk['labels'], labels)):
                    output['chunks'].append({**chunk, 'labels': labs,
                                             'n_loss_tokens': sum(y != -100 for y in labs)})
            m0.write(dump(outputs[0])+'\n'); m1.write(dump(outputs[1])+'\n')
    totals = (len(seen), len(ledgers), sum(x['m0_shifted_targets'] for x in ledgers))
    if totals != EXPECTED or len(set(seen)) != len(seen) or set(seen) != set(wanted):
        raise ValueError('population identity gate failed: '+str(totals))
    groups = make_groups(ledgers)
    if len(groups) != 208:
        raise ValueError('expected exactly 208 updates')
    (OUT/'update_groups.json').write_text(json.dumps(groups, indent=2)+'\n')
    # Detect concurrent changes to any pinned input before declaring a prepared pair.
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError('pinned input changed while preparing')
    outputs = {str(p): sha(p) for p in sorted(OUT.iterdir())}
    manifest = {'status': 'DRAFT_UNAPPROVED_NO_TRAINING', 'arm': ARM, 'source_hashes': hashes,
                'artifact_hashes': outputs, 'trajectories': totals[0], 'chunks': totals[1],
                'm0_shifted_targets': totals[2], 'm1_shifted_targets': sum(x['m1_shifted_targets'] for x in ledgers),
                'masked_shifted_targets': sum(sum(pos > 0 for pos, _ in x['changed_original_positions']) for x in ledgers),
                'masked_unshifted_positions': sum(len(x['changed_original_positions']) for x in ledgers),
                'nonempty_targets_after_shift': sum(x['nonempty_targets_after_shift'] for x in ledgers),
                'decision_counts': dict(reasons), 'assistant_role_ids': role, 'update_groups': len(groups),
                'rule': 'audited aligned run + exact optional recognized opener then immediate channel close; verified ignored role/tool-response boundary; ambiguous retained; no whitespace widening',
                'loss': 'sum Mx shifted CE / same M0 shifted target count per original shuffled update',
                'hyperparameters': {'rank':16,'alpha':16,'lr':0.0001,'epochs':1,'max_seq_len':16384,
                    'grad_accum':8,'warmup_ratio':0.05,'weight_decay':0.0,'max_grad_norm':1.0,'seed':SEED},
                'base': str(ROOT/'models/gemma-4-e4b-it'),
                'probe_dir': str(ROOT/'subs/e4b_probe/adapters/probe_lora'),
                'fresh_outputs': {v:str(ROOT/('sft/runs/A1_'+v+'_v035')) for v in ('M0','M1')},
                'unresolved_activation_gates': ['Root + code reviewer + Momus approval',
                    'GPU lease released after E1/E2; fresh controls estimated 9 GPU hours',
                    'Historical A1 initialization unproven: original seed controlled order only; fresh pair explicitly seeds common initialization'],
                'legacy_source_equivalence': {str(Path('HOME_DIR/gemma4')/rel): sha(Path('HOME_DIR/gemma4')/rel)
                    for rel in ('sft/data/trajectories.jsonl','sft/data/arms.json','sft/train_lora.py')},
                'licenses': 'unchanged; local derivatives only; no external upload'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    contract = dict(manifest, manifest_path=str(OUT/'manifest.json'), manifest_sha256=sha(OUT/'manifest.json'),
                    approval_required=True, execute_training=False)
    CONTRACT.write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('trajectories','chunks','m0_shifted_targets','m1_shifted_targets',
          'masked_shifted_targets','masked_unshifted_positions','nonempty_targets_after_shift','decision_counts','update_groups')}))
    print('manifest_sha256='+sha(OUT/'manifest.json'))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
