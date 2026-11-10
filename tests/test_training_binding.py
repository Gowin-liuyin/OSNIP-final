import os

import pytest
import torch

from scripts.train_streaming import load_token_map, model_file_hashes, token_map_commitment, validate_resume_binding


def test_private_secret_and_tensor_share_commitment(tmp_path):
    perm = torch.tensor([2, 0, 1])
    path = tmp_path / "map.pt"
    torch.save({"perm_vocab": perm, "inv_perm_vocab": torch.tensor([1, 2, 0]), "random_master_key": b"private"}, path)
    assert token_map_commitment(load_token_map(path, 3)) == token_map_commitment(perm)
    assert token_map_commitment(perm) == "5237b8b02ae2d6b2e162c1634ee63c6460efc398b22ba79503fa33c5b28ee251"
    torch.save(perm, path)
    assert torch.equal(load_token_map(path, 3), perm)


@pytest.mark.parametrize("payload", [torch.tensor([0, 0, 1]), torch.tensor([0.0, 1.0, 2.0]),
                                      {"perm_vocab": torch.tensor([2, 0, 1]), "inv_perm_vocab": torch.tensor([0, 1, 2])}])
def test_invalid_map_rejected(tmp_path, payload):
    path = tmp_path / "map.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_token_map(path, 3)


def test_resume_detects_same_path_modified_in_place(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    path = model / "model.bin"
    path.write_bytes(b"original")
    prior = {"model_files_sha256": model_file_hashes(model), "token_id_map_sha256": "map", "token_map_commitment_sha256": "permutation"}
    validate_resume_binding(prior, model_file_hashes(model), "map", "permutation")
    stat = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="model contents"):
        validate_resume_binding(prior, model_file_hashes(model), "map", "permutation")
    with pytest.raises(ValueError, match="token map"):
        validate_resume_binding(prior, prior["model_files_sha256"], "map", "different")
