import importlib.util
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "decoder_recovery_test", Path(__file__).resolve().parents[1] / "scripts/evaluate_decoder_recovery.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_layer_cosine_uses_causal_valid_positions_only():
    clean_logits = torch.zeros(1, 4, 5)
    encrypted_logits = torch.zeros(1, 4, 5)
    clean_hidden = (torch.ones(1, 4, 3),)
    encrypted_hidden = (torch.ones(1, 4, 3) * 0.5,)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    metrics = recovery.row_metrics(clean_logits, encrypted_logits, clean_hidden,
                                   encrypted_hidden, (1,), input_ids, mask)
    # causal valid pairs: (t=0,1) and (t=1,2); position 3 is padding, position 0
    # has no padding predecessor but shifted index 0 pairs positions 0->1: valid.
    assert metrics["layers"]["1"]["tokens"] == 2
    assert metrics["tokens"] == 2
