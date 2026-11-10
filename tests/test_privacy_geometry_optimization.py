"""Check the decoder-free control against the existing training losses."""
import importlib.util
from pathlib import Path

import torch

from osnip_repro.encoder import build_encryptor
from osnip_repro.losses import osnip_two_key_loss
from osnip_repro.projection import iso_norm_project

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('geometry_control', ROOT/'scripts/diagnose_privacy_optimization.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_geometry_control_has_exact_privacy_only_training_gradient():
    torch.manual_seed(41)
    encoder = build_encryptor(11, key_dim=3, hidden_dim=7)
    h, k1, k2 = torch.randn(13,11), torch.randn(3), torch.randn(3)
    actual, z1, z2, _, _ = control.geometry_loss(encoder,h,k1,k2,epsilon=.3,margin=1.)
    gradient = torch.autograd.grad(actual, tuple(encoder.parameters()))
    z1 = iso_norm_project(encoder(h,k1),h)
    z2 = iso_norm_project(encoder(h,k2),h)
    dummy_logits = torch.zeros(13,5)
    reference = osnip_two_key_loss(dummy_logits,dummy_logits,dummy_logits,h,z1,z2,
                    epsilon=.3,lambda_priv=1.,lambda_div=1.,diversity_margin=1.)['total']
    expected = torch.autograd.grad(reference,tuple(encoder.parameters()))
    assert torch.equal(actual,reference)
    assert all(torch.equal(a,b) for a,b in zip(gradient,expected))
    assert all(torch.isfinite(g).all() for g in gradient)
    assert any(torch.count_nonzero(g) for g in gradient)


def test_geometry_validation_uses_token_weights_and_both_fixed_keys():
    torch.manual_seed(99)
    encoder = build_encryptor(5,key_dim=2,hidden_dim=4)
    rows=[torch.randn(2,5),torch.randn(7,5)]
    keys=torch.randn(2,2,2)
    result=control.evaluate(encoder,rows,keys,epsilon=.3,margin=1.)
    expected=[]
    for h,pair in zip(rows,keys):
        for key in pair:
            z=iso_norm_project(encoder(h,key),h)
            expected.append(torch.nn.functional.cosine_similarity(h,z,dim=-1).abs())
    cosines=torch.cat(expected)
    assert result['token_key_observations']==18
    assert abs(result['mean_abs_cosine']-float(cosines.detach().mean()))<1e-6
    assert abs(result['constraint_coverage']-float((cosines.detach()<=.3).float().mean()))<1e-6
    assert result['iso_norm_max_abs_error']<1e-5


def test_streamed_geometry_reads_all_rows_with_bounded_cache(tmp_path):
    import json
    from safetensors.torch import save_file
    table=torch.arange(30,dtype=torch.float32).reshape(6,5)
    save_file({'model.embed_tokens.weight':table},str(tmp_path/'model.safetensors'))
    source=tmp_path/'train.jsonl'
    source.write_text('\n'.join(json.dumps({'text':row}) for row in ['0 1 0','2 3','4 5 1'])+'\n')
    def tokenizer(text,**kwargs):
        return {'input_ids':[int(x) for x in text.split()][:kwargs['max_length']]}
    rows=control.StreamingEmbeddingRows(source,tmp_path,tokenizer,max_length=3,cache_rows=2)
    assert len(rows)==3
    assert torch.equal(rows[0],table[[0,1,0]])
    assert torch.equal(rows[2],table[[4,5,1]])
    assert len(rows.cache)<=2
    assert rows.manifest()['unique_source_indices_seen']==2
    assert not rows.manifest()['entire_pool_seen']
    assert torch.equal(rows[1],table[[2,3]])
    assert torch.equal(rows[0],table[[0,1,0]])
    assert rows.manifest()['entire_pool_seen']
    assert rows.manifest()['prompt_reads']==4
    assert len(rows.cache)<=2
