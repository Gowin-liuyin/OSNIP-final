import importlib.util
from pathlib import Path

import torch

from osnip_repro.encoder import build_encryptor
from osnip_repro.projection import iso_norm_project

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('streamed_knn',ROOT/'scripts/evaluate_streamed_knn.py')
knn=importlib.util.module_from_spec(spec);spec.loader.exec_module(knn)


def test_streamed_knn_matches_independent_per_prompt_key_and_distance_sorting():
    torch.manual_seed(71)
    table=torch.randn(21,7,dtype=torch.float32)
    encoder=build_encryptor(7,key_dim=3,hidden_dim=11).eval()
    token_rows=[[8,2,1],[10],[0,15,9,9]]
    clean,encrypted=knn.prepare_queries(token_rows,lambda a,b:table[a:b],encoder,key_dim=3,seed=42)
    expected_clean=[];expected_encrypted=[]
    rng=torch.Generator().manual_seed(42)
    for row in token_rows:
        ids=torch.tensor(row)[None]
        h=table[ids]
        key=torch.randn(3,generator=rng)
        z=iso_norm_project(encoder(h,key),h)
        expected_clean.append(h[0]);expected_encrypted.append(z[0])
    assert torch.equal(clean,torch.cat(expected_clean))
    assert torch.equal(encrypted,torch.cat(expected_encrypted))
    queries=torch.cat([clean,encrypted])
    ranked=knn.streamed_l2_topk(queries,len(table),lambda a,b:table[a:b],candidate_chunk=6,query_chunk=3)
    reference=(queries.double()[:,None]-table.double()[None]).square().sum(-1).argsort(-1)[:,:10]
    assert torch.equal(ranked,reference)
    rows,counts,rates=knn.score_rows(ranked,token_rows)
    assert [r['positions'] for r in rows]==[3,1,4]
    assert counts['clean']['top1']==8 and rates['clean']['top1']==1
    for side,points in [('clean',expected_clean),('encrypted',expected_encrypted)]:
        for row,(ids,query) in enumerate(zip(token_rows,points)):
            independent=(query.double()[:,None]-table.double()[None]).square().sum(-1).argsort(-1)
            for k in (1,5,10):
                assert rows[row][side][f'top{k}']==int((independent[:,:k]==torch.tensor(ids)[:,None]).any(-1).sum())


def test_multi_key_observations_use_independent_keys_per_prompt():
    torch.manual_seed(5)
    encoder = build_encryptor(2, key_dim=2, hidden_dim=8)
    table = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    reader = lambda a, b: table[a:b]
    doubled = [[0, 1], [0, 1]]
    clean, encrypted = knn.prepare_queries(doubled, reader, encoder, key_dim=2, seed=42)
    assert torch.equal(clean[:2], clean[2:])
    assert not torch.allclose(encrypted[:2], encrypted[2:])
    single, single_enc = knn.prepare_queries([[0, 1]], reader, encoder, key_dim=2, seed=42)
    assert torch.equal(encrypted[:2], single_enc)
