from types import SimpleNamespace

import pytest
import torch

from osnip_repro.modeling import causal_nll_stats, sliding_window_nll


class ConstantLM:
    def __call__(self, inputs_embeds, attention_mask, use_cache):
        return SimpleNamespace(logits=torch.zeros(*inputs_embeds.shape[:2], 7))


@pytest.mark.parametrize('length,window,stride', [(2, 8, 3), (19, 8, 3), (16, 8, 7), (9, 8, 1)])
def test_sliding_counts_every_target_exactly_once(length, window, stride):
    ids = torch.ones(2, length, dtype=torch.long)
    nll, count = sliding_window_nll(ConstantLM(), torch.zeros(2, length, 4), ids,
                                    max_length=window, stride=stride)
    assert count == 2 * (length - 1)
    assert nll / count == pytest.approx(torch.log(torch.tensor(7.)).item(), abs=1e-6)


def test_sliding_rejects_no_overlap_stride():
    with pytest.raises(ValueError, match='stride'):
        sliding_window_nll(ConstantLM(), torch.zeros(1, 16, 4), torch.ones(1, 16, dtype=torch.long),
                           max_length=8, stride=8)


def test_left_padding_has_no_spurious_first_target():
    ids = torch.tensor([[0, 0, 1, 2]])
    nll, count = causal_nll_stats(torch.zeros(1, 4, 7), ids, torch.tensor([[0, 0, 1, 1]]))
    assert count == 1
    assert nll.item() == pytest.approx(torch.log(torch.tensor(7.)).item())


@pytest.mark.parametrize("kind", ["llama", "qwen"])
def test_chunked_head_nll_matches_full_logits_for_real_decoder(kind):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from osnip_repro.modeling import selected_target_nll
    torch.manual_seed(53)
    cls, config_cls = (LlamaForCausalLM, LlamaConfig) if kind == "llama" else (Qwen2ForCausalLM, Qwen2Config)
    model = cls(config_cls(vocab_size=31, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    ids = torch.randint(0, 31, (2, 8))
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    embeddings = model.get_input_embeddings()(ids).detach()
    embeddings += torch.randn_like(embeddings) * 0.01
    positions = torch.tensor([0, 1, 2, 3, 9, 10, 11, 12, 13, 14])
    targets = torch.randint(0, 31, (len(positions),))
    with torch.no_grad():
        full = model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits
    expected = torch.nn.functional.cross_entropy(full.reshape(-1, 31)[positions], targets, reduction="sum")
    actual = selected_target_nll(model, embeddings, mask, positions, targets, head_chunk=3)
    assert actual.item() == pytest.approx(expected.item(), abs=1e-5)
