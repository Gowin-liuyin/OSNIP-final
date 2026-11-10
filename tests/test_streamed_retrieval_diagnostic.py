import importlib.util
from pathlib import Path

import pytest
import torch

spec=importlib.util.spec_from_file_location('conditional_retrieval',Path(__file__).resolve().parents[1]/'scripts/diagnose_validation_retrieval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_streamed_full_ranking_matches_independent_float64_distance():
    rng=torch.Generator().manual_seed(89)
    candidates=torch.randn(31,11,generator=rng,dtype=torch.float64)
    query=torch.randn(17,11,generator=rng,dtype=torch.float64)
    calls=[]
    def read(left,right):
        calls.append((left,right))
        return candidates[left:right]
    actual=module.streamed_l2_topk(query,31,read,k=10,candidate_chunk=7,query_chunk=4)
    expected=torch.cdist(query,candidates).topk(10,largest=False).indices
    assert torch.equal(actual,expected)
    assert calls==[(0,7),(7,14),(14,21),(21,28),(28,31)]


def test_conditional_retrieval_counts_and_empty_stratum():
    rankings=torch.arange(10).repeat(3,1)
    targets=torch.tensor([0,4,9])
    cosine=torch.tensor([.1,.3,.9])
    counts=module.conditional_counts(rankings,targets,cosine,.3)
    assert counts['all']['hits']=={'top1':1,'top5':2,'top10':3}
    assert counts['abs_cosine_le_epsilon']['positions']==2
    assert counts['abs_cosine_gt_epsilon']['rates']['top5']==0.
    empty=module.conditional_counts(rankings,targets,torch.ones(3),.3)
    assert empty['abs_cosine_le_epsilon']['rates']['top1'] is None


def test_streamed_retrieval_rejects_bad_candidate_values():
    with pytest.raises(ValueError,match='vocabulary chunk'):
        module.streamed_l2_topk(torch.ones(2,3),10,lambda l,r:torch.full((r-l,3),float('nan')))


def test_paired_bootstrap_resamples_prompts_and_rejects_misalignment():
    def row(index,hits):
        return {'prompt_index':index,'conditional_retrieval':{'all':{'positions':4,'hits':{'top10':hits}}}}
    left=[row(0,4),row(1,4)]
    right=[row(0,2),row(1,2)]
    result=module.paired_top10_bootstrap(left,right,seed=9,resamples=100)
    assert result['point_estimate']==-.5
    assert result['bootstrap_95ci']==[-.5,-.5]
    with pytest.raises(ValueError,match='disagree'):
        module.paired_top10_bootstrap(left,[row(2,2),row(1,2)],seed=9)


@pytest.mark.parametrize('key_count', [1, 3])
def test_cli_multi_key_prompt_alignment_matches_direct_independent_retrieval(tmp_path, monkeypatch, key_count):
    import json
    import sys
    from safetensors.torch import save_file
    from osnip_repro.encoder import build_encryptor_from_config
    from osnip_repro.projection import iso_norm_project
    torch.manual_seed(51)
    table=torch.randn(19,4)
    config={'width':4,'key_dim':2,'hidden_dim':3,'epsilon':.3}
    encoder=build_encryptor_from_config(config).eval()
    with torch.no_grad():
        encoder.output_projection.weight.mul_(100)
    model=tmp_path/'model'
    model.mkdir()
    save_file({'model.embed_tokens.weight':table},model/'model.safetensors')
    (model/'download_manifest.json').write_text('{}')
    checkpoint=tmp_path/'encoder.pt'
    torch.save(encoder.state_dict(),checkpoint)
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    data=tmp_path/'valid.jsonl'
    data.write_text('{"text":"alpha"}\n{"text":"beta"}\n')
    token_rows=[[1,4],[7,2,11]]
    class Tokenizer:
        def __call__(self,*args,**kwargs):
            return {'input_ids':token_rows}
    monkeypatch.setattr(module.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['diagnostic','--model-path',str(model),
        '--config',str(config_path),'--data-jsonl',str(data),'--checkpoint',str(checkpoint),
        '--keys-per-prompt',str(key_count),'--candidate-chunk','7','--query-chunk','2',
        '--threads','1','--output',str(output),'--seed','17'])
    module.main()
    report=json.loads(output.read_text())
    item=report['checkpoints'][0]
    assert report['encrypted_token_key_observations']==5*key_count
    keys=torch.randn(2,key_count,2,generator=torch.Generator().manual_seed(17))
    expected_rows=[]
    with torch.no_grad():
        for i,ids in enumerate(token_rows):
            clean=table[ids]
            per_key=[]
            for j in range(key_count):
                z=iso_norm_project(encoder(clean,keys[i,j]),clean)
                ranks=torch.cdist(z.double(),table.double()).topk(10,largest=False).indices
                cosine=torch.nn.functional.cosine_similarity(clean,z,dim=-1)
                expected=module.conditional_counts(ranks,torch.tensor(ids),cosine,.3)
                assert item['observations'][i]['keys'][j]==expected
                per_key.append(expected)
            expected_rows.append(per_key)
    for stratum in ('all','abs_cosine_le_epsilon','abs_cosine_gt_epsilon'):
        assert item['conditional_retrieval'][stratum]['positions']==sum(
            x[stratum]['positions'] for row in expected_rows for x in row)
        for k in ('top1','top5','top10'):
            assert item['conditional_retrieval'][stratum]['hits'][k]==sum(
                x[stratum]['hits'][k] for row in expected_rows for x in row)
