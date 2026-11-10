#!/usr/bin/env python3
"""CPU validation-only full-vocabulary retrieval, streamed from safetensors.

No decoder or full embedding table is loaded. This is an exploratory diagnosis,
not the independent paper KNN protocol, and never estimates a client key.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256, read_text_jsonl
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.projection import iso_norm_project


@torch.no_grad()
def streamed_l2_topk(query, vocabulary_size, read_rows, *, k=10, candidate_chunk=1024, query_chunk=128):
    """Global exact candidate coverage; input dtype arithmetic, no tie guarantee."""
    if query.ndim != 2 or not torch.isfinite(query).all():
        raise ValueError('query must contain finite vectors')
    if min(candidate_chunk, query_chunk, k) < 1 or vocabulary_size < k:
        raise ValueError('invalid chunk, k or vocabulary size')
    best = torch.full((len(query), k), float('inf'), dtype=query.dtype)
    ids = torch.full((len(query), k), -1, dtype=torch.long)
    query_norm = query.square().sum(-1, keepdim=True)
    for start in range(0, vocabulary_size, candidate_chunk):
        part = read_rows(start, min(start + candidate_chunk, vocabulary_size)).to(query)
        if part.shape != (min(candidate_chunk, vocabulary_size - start), query.shape[1]) or not torch.isfinite(part).all():
            raise ValueError('invalid vocabulary chunk')
        norm = part.square().sum(-1)[None]
        for left in range(0, len(query), query_chunk):
            right = min(left + query_chunk, len(query))
            distance = (query_norm[left:right] + norm - 2 * query[left:right] @ part.T).clamp_min(0)
            values, local_ids = distance.topk(min(k, len(part)), largest=False, sorted=True)
            merged_values = torch.cat((best[left:right], values), dim=-1)
            merged_ids = torch.cat((ids[left:right], local_ids + start), dim=-1)
            order = merged_values.topk(k, largest=False, sorted=True).indices
            best[left:right] = merged_values.gather(-1, order)
            ids[left:right] = merged_ids.gather(-1, order)
    return ids


def conditional_counts(rankings, targets, cosines, epsilon):
    if len(rankings) != len(targets) or targets.shape != cosines.shape:
        raise ValueError('unaligned observations')
    result = {}
    for name, selected in [('all', torch.ones_like(targets, dtype=torch.bool)),
                           ('abs_cosine_le_epsilon', cosines.abs() <= epsilon),
                           ('abs_cosine_gt_epsilon', cosines.abs() > epsilon)]:
        positions = int(selected.sum())
        hits = {f'top{k}': int((rankings[selected, :k] == targets[selected, None]).any(-1).sum())
                for k in (1, 5, 10)}
        result[name] = {'positions': positions, 'hits': hits,
                        'rates': {name: count / positions if positions else None for name, count in hits.items()}}
    return result



def paired_top10_bootstrap(left, right, *, seed, resamples=5000):
    if not left or len(left) != len(right) or resamples < 2:
        raise ValueError('need nonempty aligned prompt rows and at least two resamples')
    for a,b in zip(left,right):
        if a['prompt_index'] != b['prompt_index'] or a['conditional_retrieval']['all']['positions'] != b['conditional_retrieval']['all']['positions']:
            raise ValueError('paired prompt IDs or counts disagree')
    counts=torch.tensor([row['conditional_retrieval']['all']['positions'] for row in left],dtype=torch.float64)
    if (counts<=0).any():
        raise ValueError('positive prompt counts required')
    a=torch.tensor([row['conditional_retrieval']['all']['hits']['top10'] for row in left],dtype=torch.float64)
    b=torch.tensor([row['conditional_retrieval']['all']['hits']['top10'] for row in right],dtype=torch.float64)
    indices=torch.randint(0,len(left),(resamples,len(left)),generator=torch.Generator().manual_seed(seed))
    estimates=(b[indices]-a[indices]).sum(-1)/counts[indices].sum(-1)
    return {'change':'right_minus_left_top10_recovery','seed':seed,'resamples':resamples,
            'n_prompts':len(left),'point_estimate':float((b-a).sum()/counts.sum()),
            'bootstrap_95ci':torch.quantile(estimates,torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
            'scope':'paired_validation_prompts_fixed_checkpoint_and_client_seed'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-jsonl', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-samples', type=int, default=128)
    p.add_argument('--max-tokens-per-prompt', type=int, default=16)
    p.add_argument('--candidate-chunk', type=int, default=1024)
    p.add_argument('--query-chunk', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--keys-per-prompt', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new report path')
    if min(args.max_samples, args.max_tokens_per_prompt, args.threads, args.candidate_chunk, args.query_chunk, args.keys_per_prompt) < 1:
        raise ValueError('positive sample and chunk limits required')
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    # Same tokenization as client geometry diagnostic, then restrict probe positions.
    ids = tokenizer(texts, truncation=True, max_length=args.max_tokens_per_prompt, padding=False)['input_ids']
    if any(not row for row in ids):
        raise ValueError('empty tokenized prompt')
    flat_ids = torch.tensor([token for row in ids for token in row])
    unique = sorted(set(flat_ids.tolist()))
    with safe_open(args.model_path / 'model.safetensors', framework='pt', device='cpu') as handle:
        table = handle.get_slice('model.embed_tokens.weight')
        vocabulary_size, width = table.get_shape()
        vectors = torch.cat([table[token:token+1].float() for token in unique])
        lookup = {token: index for index, token in enumerate(unique)}
        clean = vectors[[lookup[token] for token in flat_ids.tolist()]]
        rng = torch.Generator().manual_seed(args.seed)
        keys = torch.randn(len(ids), args.keys_per_prompt, int(config['key_dim']), generator=rng)
        queries = [clean]
        cosines = []
        checkpoints = []
        for checkpoint in args.checkpoint:
            digest = file_sha256(checkpoint)
            encoder = build_encryptor_from_config(config).eval()
            encoder.load_state_dict(load_encoder_state_dict(str(checkpoint), expected_config=config))
            if file_sha256(checkpoint) != digest:
                raise RuntimeError('checkpoint changed; use a frozen checkpoint')
            encrypted = []
            with torch.inference_mode():
                offset = 0
                for index, row in enumerate(ids):
                    h = clean[offset:offset+len(row)]
                    encrypted.append(iso_norm_project(encoder(h[None].expand(args.keys_per_prompt, -1, -1), keys[index]), h[None].expand(args.keys_per_prompt, -1, -1)))
                    offset += len(row)
            z = torch.cat(encrypted, dim=1)
            repeated_clean = clean.repeat(args.keys_per_prompt, 1)
            queries.append(z.reshape(-1, z.shape[-1]))
            cosines.append(F.cosine_similarity(repeated_clean, z.reshape(-1, z.shape[-1]), dim=-1))
            checkpoints.append({'checkpoint': str(checkpoint.resolve()), 'sha256': digest})
        print(json.dumps({'stage':'full_vocabulary_scan','queries':sum(map(len,queries)),
                          'vocabulary_size':vocabulary_size,'width':width}),flush=True)
        rankings = streamed_l2_topk(torch.cat(queries), vocabulary_size, lambda left,right: table[left:right],
                                    candidate_chunk=args.candidate_chunk, query_chunk=args.query_chunk)
    count = len(clean)
    positive = conditional_counts(rankings[:count], flat_ids, torch.ones(count), .3)['all']
    expanded_count = count * args.keys_per_prompt
    expanded_targets = flat_ids.repeat(args.keys_per_prompt)
    for index, item in enumerate(checkpoints):
        ranked = rankings[count + index*expanded_count:count + (index+1)*expanded_count]
        item['keys_per_prompt'] = args.keys_per_prompt
        item['conditional_retrieval'] = conditional_counts(ranked, expanded_targets, cosines[index], float(config['epsilon']))
        item['mean_cosine'] = float(cosines[index].mean())
        item['cosine_constraint_coverage'] = item['conditional_retrieval']['abs_cosine_le_epsilon']['positions'] / expanded_count
        item['key_summaries'] = []
        for key_index in range(args.keys_per_prompt):
            key_ranked = ranked[key_index*count:(key_index+1)*count]
            key_cosine = cosines[index][key_index*count:(key_index+1)*count]
            item['key_summaries'].append({'independent_draw_index':key_index,
                'conditional_retrieval':conditional_counts(key_ranked, flat_ids, key_cosine, float(config['epsilon']))})
        item['observations'] = []
        offset = 0
        for prompt_index, row in enumerate(ids):
            length = len(row)
            prompt_parts = []
            for key_index in range(args.keys_per_prompt):
                left = key_index*count + offset
                prompt_parts.append(conditional_counts(ranked[left:left+length], flat_ids[offset:offset+length],
                    cosines[index][left:left+length], float(config['epsilon'])))
            all_ranked = torch.cat([ranked[key_index*count+offset:key_index*count+offset+length] for key_index in range(args.keys_per_prompt)])
            all_targets = flat_ids[offset:offset+length].repeat(args.keys_per_prompt)
            all_cosine = torch.cat([cosines[index][key_index*count+offset:key_index*count+offset+length] for key_index in range(args.keys_per_prompt)])
            item['observations'].append({'prompt_index':prompt_index,
                'conditional_retrieval':conditional_counts(all_ranked, all_targets, all_cosine, float(config['epsilon'])),
                'keys':prompt_parts})
            offset += length
    paired = []
    for index in range(1,len(checkpoints)):
        paired.append({'left_checkpoint':checkpoints[0]['sha256'],'right_checkpoint':checkpoints[index]['sha256'],
                       **paired_top10_bootstrap(checkpoints[0]['observations'],checkpoints[index]['observations'],seed=args.seed)})
    report = {'format':'osnip_validation_conditional_retrieval_v1','scope':'exploratory_validation_only',
              'security_accepted':False,'key_attack_performed':False,
              'prompts':len(ids),'positions':count,'encrypted_token_key_observations':expanded_count,'seed':args.seed,
              'key_schedule':'independent_known_client_keys_per_prompt_shared_across_checkpoints',
              'keys_per_prompt':args.keys_per_prompt,
              'max_tokens_per_prompt':args.max_tokens_per_prompt,'full_vocabulary':True,
              'vocabulary_size':vocabulary_size,'epsilon':config['epsilon'],
              'model_path':str(args.model_path.resolve()),
              'model_manifest_sha256':file_sha256(args.model_path/'download_manifest.json'),
              'data_jsonl':str(args.data_jsonl.resolve()),'data_sha256':file_sha256(args.data_jsonl),
              'config_sha256':file_sha256(args.config),'device':'cpu','dtype':'float32',
              'positive_control':positive,'positive_control_valid':positive['rates']['top1']>=.95,
              'checkpoints':checkpoints,'paired_top10_bootstrap':paired,'elapsed_seconds':time.monotonic()-started,
              'source_hashes':{str(path.relative_to(ROOT)):file_sha256(path) for path in
                  [Path(__file__),ROOT/'src/osnip_repro/encoder.py',ROOT/'src/osnip_repro/projection.py']},
              'limitations':['Validation prefixes are used for diagnosis; this is not the independent 1000-prompt KNN protocol.',
                             'Conditional token counts are descriptive, not independent samples.',
                             'Each draw index groups independent keys across prompts, not one universal session key.',
                             'Full candidate coverage in float32; tied candidate order is unspecified.',
                             'No decoder utility, VMA or multi-seed security acceptance is performed.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'}),flush=True)
    for item in checkpoints:
        print(json.dumps({k:v for k,v in item.items() if k!='observations'}),flush=True)


if __name__ == '__main__':
    main()
