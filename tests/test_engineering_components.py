import json
import random
from pathlib import Path

import numpy as np
import torch

from osnip_repro.checkpointing import load_encoder_state_dict, load_training_checkpoint, restore_rng_state, save_training_checkpoint
from osnip_repro.data import TextJsonlIndex, split_hashes
from osnip_repro.encoder import KeyConditionedEncryptor, build_encryptor_from_config


def test_seekable_jsonl_index_and_disjoint_hashes(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(json.dumps({"text": "alpha beta"}) + "\n" + json.dumps({"instruction": "gamma", "input": "delta"}) + "\n")
    valid.write_text(json.dumps({"text": "epsilon zeta"}) + "\n")
    index = TextJsonlIndex.build(train)
    assert len(index.offsets) == 2
    assert index.read_text(1) == "gamma\n\ndelta"
    report = split_hashes({"train": train, "valid": valid})
    assert report["status"] == "disjoint"


def test_checkpoint_roundtrip_restores_model_optimizer_and_rng(tmp_path: Path):
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = KeyConditionedEncryptor(4, key_dim=2, hidden_dim=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    h = torch.randn(2, 4)
    loss = model(h, torch.randn(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / "last.pt"
    save_training_checkpoint(str(path), encoder=model, optimizer=optimizer, global_step=3, config={"x": 1}, metrics={"loss": 2}, extra={"epoch": 4, "cursor": 9})

    expected_next = torch.rand(3)
    restored = KeyConditionedEncryptor(4, key_dim=2, hidden_dim=5)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_training_checkpoint(str(path), encoder=restored, optimizer=restored_optimizer)
    assert payload["global_step"] == 3
    assert payload["extra"] == {"epoch": 4, "cursor": 9}
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    assert torch.equal(expected_next, torch.rand(3))



def test_rng_restore_normalizes_device_moved_byte_tensor():
    if not torch.backends.mps.is_available():
        return
    torch.manual_seed(17)
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state().to("mps")}
    expected = torch.rand(3)
    restore_rng_state(state)
    assert torch.equal(expected, torch.rand(3))

def test_encoder_artifact_loader_accepts_full_and_bare_checkpoints(tmp_path: Path):
    model = KeyConditionedEncryptor(4, key_dim=2, hidden_dim=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    full_path = tmp_path / "best.pt"
    bare_path = tmp_path / "encoder.pt"
    save_training_checkpoint(str(full_path), encoder=model, optimizer=optimizer, global_step=1, config={}, metrics={})
    torch.save(model.state_dict(), bare_path)
    expected = model.state_dict()
    for path in (full_path, bare_path):
        loaded = load_encoder_state_dict(str(path))
        assert loaded.keys() == expected.keys()
        assert all(torch.equal(loaded[name], expected[name]) for name in expected)


def test_checkpoint_roundtrip_preserves_nondefault_architecture(tmp_path: Path):
    config = {"width": 4, "key_dim": 2, "hidden_dim": 5, "encryptor_architecture": "low_rank_residual_v1"}
    model = build_encryptor_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "variant.pt"
    save_training_checkpoint(str(path), encoder=model, optimizer=optimizer, global_step=2, config=config, metrics={})
    restored = build_encryptor_from_config(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    load_training_checkpoint(str(path), encoder=restored, optimizer=restored_optimizer)
    assert restored.architecture == "low_rank_residual_v1"
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
