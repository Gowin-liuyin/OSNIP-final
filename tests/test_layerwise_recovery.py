import pytest
import torch

from osnip_repro.modeling import forward_hidden_states, forward_logits_and_hidden_states


class TinyOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.logits = hidden_states[-1] + 10


class TinyModel:
    def __call__(self, **kwargs):
        x = kwargs["inputs_embeds"]
        return TinyOutput((x, x + 1, x + 2))


def test_forward_hidden_states_selects_huggingface_indices():
    x = torch.zeros(1, 2, 3)
    result = forward_hidden_states(TinyModel(), x, torch.ones(1, 2, dtype=torch.bool), (0, 2))
    assert len(result) == 2
    assert torch.equal(result[0], x)
    assert torch.equal(result[1], x + 2)


def test_forward_hidden_states_rejects_invalid_layer():
    with pytest.raises(ValueError):
        forward_hidden_states(TinyModel(), torch.zeros(1, 1, 2), torch.ones(1, 1), (-1,))
    with pytest.raises(ValueError):
        forward_hidden_states(TinyModel(), torch.zeros(1, 1, 2), torch.ones(1, 1), (3,))


def test_forward_logits_and_hidden_states_uses_one_output_contract():
    x = torch.zeros(1, 2, 3)
    logits, result = forward_logits_and_hidden_states(TinyModel(), x, torch.ones(1, 2, dtype=torch.bool), (1, 2))
    assert torch.equal(logits, x + 12)
    assert torch.equal(result[0], x + 1)
    assert torch.equal(result[1], x + 2)
