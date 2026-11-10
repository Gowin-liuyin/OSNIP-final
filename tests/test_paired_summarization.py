import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paired_summary', ROOT/'scripts/evaluate_summarization_utility.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture(autouse=True)
def local_test_scorer_binding(monkeypatch):
    monkeypatch.setattr(summary,'resolve_scorer_snapshot',lambda name: ({'model_type':name,'num_layers':17},{'model_id':name,'test_only':True}))


def test_bertscore_pairs_clean_and_encrypted_in_one_actual_score_call(monkeypatch):
    import bert_score
    calls=[]

    def score(candidates, references, **kwargs):
        calls.append((candidates,references,kwargs))
        return (torch.tensor([.8,.6,.4,.2]),torch.tensor([.8,.6,.4,.2]),
                torch.tensor([.8,.6,.4,.2])), 'fixed-scorer-settings-hash'

    monkeypatch.setattr(bert_score,'score',score)
    report=summary.compute_paired_bertscore(['c1','c2'],['e1','e2'],['r1','r2'],'roberta-large',torch.device('cpu'))
    assert len(calls)==1
    assert calls[0][0]==['c1','c2','e1','e2']
    assert calls[0][1]==['r1','r2','r1','r2']
    assert calls[0][2]['return_hash'] is True
    assert report['clean']['f1']==pytest.approx(.7)
    assert report['encrypted']['f1']==pytest.approx(.3)
    assert report['f1_retention']==pytest.approx(3/7)
    assert report['rows'][1]['clean_f1']==pytest.approx(.6)
    assert report['rows'][1]['encrypted_f1']==pytest.approx(.2)
    assert report['scorer_hash']=='fixed-scorer-settings-hash'


def test_bertscore_refuses_nonfinite_output_and_undefined_positive_control(monkeypatch):
    import bert_score
    monkeypatch.setattr(bert_score,'score',lambda *a,**k: ((torch.zeros(2),torch.zeros(2),torch.tensor([float('nan'),.2])),'hash'))
    with pytest.raises(ValueError,match='non-finite'):
        summary.compute_paired_bertscore(['c'],['e'],['r'],'roberta-large',torch.device('cpu'))
    monkeypatch.setattr(bert_score,'score',lambda *a,**k: ((torch.zeros(2),torch.zeros(2),torch.zeros(2)),'hash'))
    report=summary.compute_paired_bertscore(['c'],['e'],['r'],'roberta-large',torch.device('cpu'))
    assert report['clean_positive'] is False and report['f1_retention'] is None


def test_suite_routes_formal_summary_and_disallows_skipping_required_scorer(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'selected_pareto.json').write_text(json.dumps({'selected':{'path':str(run/'best_pareto.pt')}}))
    command=[sys.executable,str(ROOT/'scripts/run_evaluation_suite.py'),'--model-path',str(tmp_path/'model'),
             '--run-dir',str(run),'--summarization-jsonl',str(tmp_path/'summary.jsonl'),'--max-summarization-samples','999',
             '--require-bertscore','--bertscore-model','roberta-large']
    subprocess.run(command,check=True,capture_output=True,text=True)
    plan=json.loads((run/'evaluation_suite_plan.json').read_text())
    assert '--require-bertscore' in plan['commands']['summarization']
    assert plan['commands']['summarization'][plan['commands']['summarization'].index('--max-samples')+1]=='999'
    bad=subprocess.run(command+['--skip-bertscore'],capture_output=True,text=True)
    assert bad.returncode!=0 and 'cannot be skipped' in bad.stderr


def test_summary_main_real_tiny_model_generates_both_branches_and_hashes_inputs(tmp_path,monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
    from osnip_repro.encoder import build_encryptor_from_config
    import bert_score
    torch.manual_seed(93)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=19,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,eos_token_id=2,pad_token_id=1)).eval()
    backend=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'a':3,'b':4,'Summary':5},unk_token='[UNK]'))
    backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]')
    model_dir=tmp_path/'model';model.save_pretrained(model_dir);tokenizer.save_pretrained(model_dir)
    config={'model_path':str(model_dir),'width':16,'key_dim':4,'hidden_dim':8}
    (tmp_path/'config.json').write_text(json.dumps(config))
    checkpoint=tmp_path/'encoder.pt';torch.save(build_encryptor_from_config(config).state_dict(),checkpoint)
    data=tmp_path/'summary.jsonl';data.write_text(json.dumps({'article':'a b','highlights':'a'})+'\n')
    out=tmp_path/'report.json'
    calls=[]
    def score(candidates,references,**kw):
        calls.append((candidates,references))
        return (torch.tensor([.8,.4]),torch.tensor([.8,.4]),torch.tensor([.8,.4])),'tiny-test-scorer'
    monkeypatch.setattr(bert_score,'score',score)
    monkeypatch.setattr(sys,'argv',['evaluate_summarization_utility.py','--model-path',str(model_dir),
        '--data-jsonl',str(data),'--checkpoint',str(checkpoint),'--output',str(out),
        '--max-samples','1','--max-input-length','16','--max-new-tokens','2','--require-bertscore'])
    summary.main()
    report=json.loads(out.read_text())
    assert len(calls)==1 and len(calls[0][0])==2
    assert report['records']==1 and len(report['rows'])==1
    assert report['retention']['bertscore_f1']==pytest.approx(.5)
    assert report['checkpoint_sha256']==summary.sha256(checkpoint)
    assert report['checkpoint_config_sha256']==summary.sha256(tmp_path/'config.json')
    assert report['data_sha256']==summary.sha256(data)
    assert report['model_file_sha256']['model.safetensors']==summary.sha256(model_dir/'model.safetensors')
    assert report['security_accepted'] is False and report['key_attack_performed'] is False


@pytest.mark.parametrize('family',['qwen','llama'])
def test_server_generation_receives_no_clear_prompt_ids_and_keeps_entire_continuation(family):
    from transformers import Qwen2Config,Qwen2ForCausalLM,LlamaConfig,LlamaForCausalLM
    torch.manual_seed(110)
    config_type,model_type=(Qwen2Config,Qwen2ForCausalLM) if family=='qwen' else (LlamaConfig,LlamaForCausalLM)
    model=model_type(config_type(vocab_size=19,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,eos_token_id=None,pad_token_id=0)).eval()
    prompt=torch.tensor([[3,4,5,6,7]])
    z=model.get_input_embeddings()(prompt)+.2
    mask=torch.ones_like(prompt)
    forward_calls=[]
    hook=model.register_forward_pre_hook(lambda mod,args,kwargs:forward_calls.append(kwargs.copy()),with_kwargs=True)
    class Tokenizer:
        pad_token_id=0
        eos_token_id=None
        def decode(self,ids,**kwargs):
            self.decoded=ids.clone()
            return 'generated continuation'
    tokenizer=Tokenizer()
    with torch.no_grad():
        text=summary.generate_summary(model,tokenizer,z,mask,max_new_tokens=2,num_beams=1)
    hook.remove()
    assert text=='generated continuation'
    assert forward_calls and forward_calls[0].get('input_ids') is None
    assert torch.equal(forward_calls[0]['inputs_embeds'],z)
    assert tokenizer.decoded.numel()==2  # no prompt prefix to slice away
    assert tokenizer.decoded.ndim==1
