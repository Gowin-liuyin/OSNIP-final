import importlib.util
import math
from pathlib import Path

import torch

spec=importlib.util.spec_from_file_location('diversity_pair',Path(__file__).resolve().parents[1]/'scripts/run_diversity_space_pair.py')
pair=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


def test_pair_resets_selection_without_touching_training_state():
    payload={'format':'osnip_training_checkpoint_v1','global_step':2000,
             'encoder':{'weight':torch.ones(3)},'optimizer':{'state':{1:torch.arange(3)}},
             'rng_state':{'torch_cpu':torch.arange(4)},
             'extra':{'best_utility':.2,'best_pareto':.4,'cursor':16000,'epoch':0,'key_rng_state':torch.arange(5)}}
    actual=pair.neutral_selection_checkpoint(payload)
    assert math.isinf(actual['extra']['best_pareto'])
    assert payload['extra']['best_pareto']==.4
    assert torch.equal(actual['encoder']['weight'],payload['encoder']['weight'])
    assert torch.equal(actual['optimizer']['state'][1],payload['optimizer']['state'][1])
    assert torch.equal(actual['extra']['key_rng_state'],payload['extra']['key_rng_state'])
    assert actual['extra']['cursor']==16000
    assert actual['global_step']==2000


def test_pair_commands_change_only_diversity_space_and_output_path(tmp_path):
    config={'_source_global_step':2800,'model_path':'model','train_jsonl':'train',
            'valid_jsonl':'valid','seed':20261018,'learning_rate':1e-5,'lambda_priv':1.,
            'max_train_samples':None,'encryptor_architecture':'mlp_residual_v1'}
    staged={'raw_r':tmp_path/'shared_start'/'raw_r'/'start.pt',
            'projected_z':tmp_path/'shared_start'/'projected_z'/'start.pt'}
    commands=pair.build_commands(config,staged=staged,output_root=tmp_path,
            additional_steps=200,validation_samples=128,interval=100,device='mps',dtype='float32')
    a,b=commands['raw_r'],commands['projected_z']
    assert a[a.index('--resume')+1]==str(staged['raw_r'])
    assert b[b.index('--resume')+1]==str(staged['projected_z'])
    changes=[(a[i-1],left,right) for i,(left,right) in enumerate(zip(a,b)) if left!=right]
    assert [row[0] for row in changes]==['--diversity-space','--resume','--output-dir']
    assert a[a.index('--steps')+1]=='3000'
    assert '--reset-optimizer' not in a
    assert '--max-train-samples' not in a
