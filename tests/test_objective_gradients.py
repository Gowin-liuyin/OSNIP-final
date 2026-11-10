import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('objective_gradients', Path(__file__).resolve().parents[1] / 'scripts/diagnose_objective_gradients.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_mps_gradient_transfer_is_split_before_dtype_cast():
    # MPS can corrupt a combined device transfer and dtype cast; cast on CPU.
    source = (Path(__file__).resolve().parents[1] / 'scripts/diagnose_objective_gradients.py').read_text()
    assert ".to('cpu').to(torch.float64)" in source
    assert ".to('cpu', torch.float64)" not in source


def test_conflicting_and_zero_weight_gradients():
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    losses = {'utility': p.sum(), 'privacy': -2 * p.sum(), 'diversity': p.square().sum()}
    losses['total'] = losses['utility'] + losses['privacy'] + 0 * losses['diversity']
    grads = diag.component_gradients(losses, [p])
    weighted = {'utility': grads['utility'], 'privacy': grads['privacy'], 'diversity': 0 * grads['diversity']}
    result = diag.gradient_summary(grads, weighted, grad_clip=1.)
    assert result['component_cosines']['utility:privacy'] == pytest.approx(-1.)
    assert result['weighted_component_norms']['diversity'] == 0
    assert result['first_order_loss_change_per_unit_sgd_lr']['utility'] > 0
    assert result['first_order_loss_change_per_unit_sgd_lr']['privacy'] < 0
    assert result['hypothetical_sgd_clip_scale'] < 1


def test_aligned_gradients_lower_both_objectives_and_unused_parameters():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    losses = {'utility': p.sum(), 'privacy': 2*p.sum(), 'diversity': 3*p.sum()}
    losses['total'] = sum(losses.values())
    grads = diag.component_gradients(losses, [p, unused])
    result = diag.gradient_summary(grads, {k: grads[k] for k in diag.COMPONENTS}, grad_clip=100.)
    assert result['component_cosines']['utility:privacy'] == pytest.approx(1.)
    assert all(v < 0 for v in result['first_order_loss_change_per_unit_sgd_lr'].values())
    assert torch.equal(grads['total'][2:], torch.zeros(3, dtype=torch.float64))


def test_accumulation_uses_per_microbatch_weights():
    first = {'utility': torch.tensor([1.], dtype=torch.float64), 'privacy': torch.tensor([4.], dtype=torch.float64), 'diversity': torch.tensor([0.], dtype=torch.float64)}
    second = {'utility': torch.tensor([3.], dtype=torch.float64), 'privacy': torch.tensor([2.], dtype=torch.float64), 'diversity': torch.tensor([0.], dtype=torch.float64)}
    average = {k: (first[k]+second[k])/2 for k in diag.COMPONENTS}
    weighted = {'utility': average['utility'], 'privacy': (0*first['privacy']+second['privacy'])/2, 'diversity': average['diversity']}
    average['total'] = sum(weighted.values())
    result = diag.gradient_summary(average, weighted, grad_clip=1.)
    assert result['weighted_component_norms']['privacy'] == 1.
    assert not torch.equal(weighted['privacy'], average['privacy'] * .5)
    assert result['component_cosines']['utility:diversity'] is None
    with pytest.raises(ValueError, match='disagree'):
        diag.gradient_summary(average, {**weighted, 'privacy': average['privacy'] * .5}, grad_clip=1.)


def test_wait_identity_is_bound_to_script_and_output(tmp_path):
    expected = tmp_path/'report.json'
    diag.validate_wait_identity(['python', 'verify_summary_scorer.py', '--output', str(expected)], 'verify_summary_scorer.py', '--output', expected)
    with pytest.raises(ValueError, match='identity'):
        diag.validate_wait_identity(['python', 'other.py', '--output', str(expected)], 'verify_summary_scorer.py', '--output', expected)
    with pytest.raises(ValueError, match='path mismatch'):
        diag.validate_wait_identity(['python', 'verify_summary_scorer.py', '--output', str(tmp_path/'other.json')], 'verify_summary_scorer.py', '--output', expected)


def test_real_tiny_llama_gradient_reconstruction_and_no_mutation():
    from transformers import LlamaConfig, LlamaForCausalLM
    from osnip_repro.encoder import KeyConditionedEncryptor
    torch.manual_seed(8)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    encoder = KeyConditionedEncryptor(16, key_dim=4, hidden_dim=8)
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {'input_ids': torch.tensor([[1, 3, 4, 2]]), 'attention_mask': torch.ones(1, 4, dtype=torch.long)}
    args = argparse.Namespace(max_length=8, key_dim=4, constraint_mode='gated', lambda_util=1.,
                              lambda_priv=1., lambda_div=1., warmup_steps=2, tau_low=0., tau_high=100.,
                              epsilon=.3, diversity_margin=1., diversity_space='projected_z', grad_clip=1.)
    before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    result, grads, weighted = diag.diagnose_one(model, encoder, Tokenizer(), 'fixed prompt', args, torch.device('cpu'), step=3, key_seed=7)
    assert result['total_gradient_reconstruction_absolute_error'] < 2e-5 * result['gradient_norms']['total']
    assert set(grads) == {'utility', 'privacy', 'diversity', 'total'}
    assert all(torch.equal(before[k], v) for k, v in encoder.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.isfinite(g).all() for g in weighted.values())


def test_cli_real_tiny_model_accumulates_a_complete_group(tmp_path, monkeypatch):
    import json
    import sys
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
    from osnip_repro.data import file_sha256
    from osnip_repro.encoder import build_encryptor_from_config
    model_dir = tmp_path/'model'; model_dir.mkdir()
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, eos_token_id=2, pad_token_id=1))
    model.save_pretrained(model_dir)
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'a': 3, 'b': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]').save_pretrained(model_dir)
    (model_dir/'download_manifest.json').write_text('{"test_only": true}\n')
    data = tmp_path/'valid.jsonl'
    data.write_text(json.dumps({'text': 'a b a'})+'\n'+json.dumps({'text': 'b a b'})+'\n')
    config = {'model_path': str(model_dir), 'valid_jsonl': str(data), 'valid_sha256': file_sha256(data),
              'width': 16, 'key_dim': 4, 'hidden_dim': 8, 'batch_size': 1, 'gradient_accumulation_steps': 2,
              'steps': 10, 'max_length': 8, 'constraint_mode': 'gated', 'lambda_util': 1., 'lambda_priv': 1.,
              'lambda_div': 1., 'warmup_steps': 2, 'tau_low': .0, 'tau_high': 100., 'epsilon': .3,
              'diversity_margin': 1., 'diversity_space': 'projected_z', 'grad_clip': 1., 'dtype': 'float32',
              'source_hashes': {}}
    run = tmp_path/'run'; run.mkdir(); (run/'config.json').write_text(json.dumps(config))
    checkpoint = tmp_path/'frozen.pt'
    torch.save({'format': 'osnip_training_checkpoint_v1', 'global_step': 3, 'encoder': build_encryptor_from_config(config).state_dict(), 'config': config}, checkpoint)
    before = file_sha256(checkpoint)
    out = tmp_path/'diagnostic'
    monkeypatch.setattr(sys, 'argv', ['diagnose', '--checkpoint', str(checkpoint), '--training-run', str(run),
                                     '--output-dir', str(out), '--max-samples', '2', '--device', 'cpu'])
    diag.main()
    report = json.loads((out/'report.json').read_text())
    assert report['status'] == 'diagnostic_completed'
    assert report['checkpoint_step'] == 3 and report['gate_step'] == 4
    assert report['prompts'] == 2 and len(report['rows']) == 2
    assert report['security_accepted'] is False and report['key_attack_performed'] is False
    assert file_sha256(checkpoint) == before == report['checkpoint_sha256']
    assert report['accumulated_gradient']['total_gradient_reconstruction_absolute_error'] < 1e-5
