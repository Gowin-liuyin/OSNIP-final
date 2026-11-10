"""Train/eval equivalence for scheduled perturbations and empty-resume rejection."""
import argparse
import json
from pathlib import Path

import pytest
import torch

from osnip_repro.checkpointing import load_encoder_state_dict, load_training_checkpoint, save_training_checkpoint
from osnip_repro.encoder import build_encryptor_from_config
from scripts import train_streaming


def recipe():
    return {"width": 8, "key_dim": 2, "hidden_dim": 4,
            "encryptor_architecture": "orthogonal_direction_v1",
            "alpha_init": 3.2, "alpha_ramp_steps": 1000}


@pytest.mark.parametrize("step", [0, 50, 300, 1000, 1200])
@pytest.mark.parametrize("legacy", [False, True])
def test_full_checkpoint_restores_actual_scale_and_outputs(tmp_path, step, legacy):
    config = recipe()
    encoder = build_encryptor_from_config(config)
    encoder.set_alpha_scale(min(1.0, step / config["alpha_ramp_steps"]))
    h, key = torch.randn(2, 5, 8), torch.randn(2, 2)
    expected = encoder(h, key)
    optimizer = torch.optim.AdamW(encoder.parameters())
    checkpoint = tmp_path / "step.pt"
    save_training_checkpoint(str(checkpoint), encoder=encoder, optimizer=optimizer,
                             global_step=step, config=config, metrics={})
    if legacy:
        payload = torch.load(checkpoint, weights_only=False)
        payload["encoder"].pop("alpha_scale")
        torch.save(payload, checkpoint)
    for training_loader in (False, True):
        restored = build_encryptor_from_config(config)
        if training_loader:
            load_training_checkpoint(str(checkpoint), encoder=restored, restore_rng=False)
        else:
            restored.load_state_dict(load_encoder_state_dict(str(checkpoint), expected_config=config))
        torch.testing.assert_close(restored(h, key), expected, rtol=0, atol=0)


def test_new_bare_export_preserves_scale_and_ambiguous_old_export_fails(tmp_path):
    config = recipe()
    encoder = build_encryptor_from_config(config)
    encoder.set_alpha_scale(0.3)
    checkpoint = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), checkpoint)
    restored = build_encryptor_from_config(config)
    restored.load_state_dict(load_encoder_state_dict(str(checkpoint), expected_config=config))
    assert restored.alpha_scale.item() == pytest.approx(0.3)
    legacy = dict(encoder.state_dict())
    legacy.pop("alpha_scale")
    torch.save(legacy, checkpoint)
    with pytest.raises(ValueError, match="legacy bare ramp"):
        load_encoder_state_dict(str(checkpoint), expected_config=config)
    with pytest.raises(ValueError, match="lacks ramp metadata"):
        load_encoder_state_dict(str(checkpoint))


def test_checkpoint_runtime_state_rejects_wrong_sidecar_ramp(tmp_path):
    config = recipe()
    encoder = build_encryptor_from_config(config)
    optimizer = torch.optim.AdamW(encoder.parameters())
    checkpoint = tmp_path / "step.pt"
    save_training_checkpoint(str(checkpoint), encoder=encoder, optimizer=optimizer,
                             global_step=50, config=config, metrics={})
    with pytest.raises(ValueError, match="mismatch for alpha_ramp_steps"):
        load_encoder_state_dict(str(checkpoint), expected_config={**config, "alpha_ramp_steps": 0})


@pytest.mark.parametrize("target_steps", [299, 300])
def test_noop_resume_fails_before_model_load_or_output_mutation(tmp_path, monkeypatch, target_steps):
    checkpoint = tmp_path / "last.pt"
    torch.save({"format": "osnip_training_checkpoint_v1", "global_step": 300}, checkpoint)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "config.json"
    sentinel.write_text('{"original": true}\n')
    argv = ["train_streaming.py", "--model-path", str(tmp_path / "model"),
            "--train-jsonl", str(tmp_path / "train.jsonl"),
            "--valid-jsonl", str(tmp_path / "valid.jsonl"), "--output-dir", str(output),
            "--resume", str(checkpoint), "--steps", str(target_steps)]
    monkeypatch.setattr("sys.argv", argv)
    def forbidden(*args, **kwargs):
        pytest.fail("a no-op resume must be rejected before loading the LLM")
    monkeypatch.setattr(train_streaming, "load_causal_lm", forbidden)
    with pytest.raises(ValueError, match="must exceed checkpoint global_step"):
        train_streaming.main()
    assert list(output.iterdir()) == [sentinel]
    assert json.loads(sentinel.read_text()) == {"original": True}


@pytest.mark.parametrize("gate_scope", ["microbatch", "effective_batch"])
def test_ramped_training_continuous_and_resumed_updates_match(tmp_path, monkeypatch, capsys, gate_scope):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    torch.manual_seed(8)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        eos_token_id=2, pad_token_id=1))
    model.save_pretrained(model_dir)
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'a': 3, 'b': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                            eos_token='[EOS]', unk_token='[UNK]').save_pretrained(model_dir)
    data = tmp_path / 'data.jsonl'
    data.write_text('{"text":"a b a b"}\n{"text":"b a b a"}\n')
    def train(directory, steps, resume=None):
        argv = ['train', '--model-path', str(model_dir), '--train-jsonl', str(data),
                '--valid-jsonl', str(data), '--output-dir', str(directory),
                '--steps', str(steps), '--max-valid-samples', '2', '--max-length', '8',
                '--key-dim', '2', '--hidden-dim', '4', '--gradient-accumulation-steps', '2',
                '--encryptor-architecture', 'orthogonal_direction_v1', '--alpha-init', '3.2',
                '--alpha-ramp-steps', '8', '--freeze-alpha', '--warmup-steps', '2',
                '--gate-scope', gate_scope, '--constraint-ramp-steps', '10', '--loss-warmup-steps', '2',
                '--evaluate-every', '1', '--checkpoint-every', '1', '--device', 'cpu']
        if resume:
            argv.extend(['--resume', str(resume)])
        monkeypatch.setattr('sys.argv', argv)
        train_streaming.main()
        return torch.load(directory / 'last.pt', weights_only=False)
    continuous = tmp_path / 'continuous'
    first = tmp_path / 'first'
    resumed = tmp_path / 'resumed'
    full = train(continuous, 2)
    train(first, 1)
    split = train(resumed, 2, first / 'last.pt')
    for name, value in full['encoder'].items():
        torch.testing.assert_close(split['encoder'][name], value, rtol=0, atol=0)
    assert full['global_step'] == split['global_step'] == 2
    assert split['encoder']['alpha_scale'].item() == 0.25
    initial = load_encoder_state_dict(str(first / 'initial_encoder.pt'))
    assert initial['alpha_scale'].item() == 0.0
    report = json.loads((resumed / 'report.json').read_text())
    assert report['baseline_step'] == 1 and report['steps_executed_this_run'] == 1
    rows = [json.loads(line) for line in (resumed / 'train_progress.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [2]
