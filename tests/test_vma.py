import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from osnip_repro.vma import autoregressive_match, hidden_at_layer, model_candidates


def test_vma_uses_own_prefix_and_searches_tail_chunk():
    calls = []
    table = torch.arange(13, dtype=torch.float32)[:, None]
    def public_forward(prefix, ids):
        calls.append((prefix, ids.tolist()))
        return table[ids] + sum(prefix) * 100
    # First original token would be 3, but the observed first vector recovers 12.
    # Second is matched under predicted prefix (12,), not any true token prefix.
    obs = torch.tensor([[12.0], [1205.0]])
    result = autoregressive_match(obs, public_forward, vocab_size=13, chunk=5, topk=3)
    assert result['predictions'] == [12, 5]
    assert [c[0] for c in calls] == [(), (), (), (12,), (12,), (12,)]
    assert calls[2][1] == [10, 11, 12]
    assert result['all_candidates_searched']


def test_hidden_layers_match_hf_and_clean_full_vocab_inverts():
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=19, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    ids = torch.tensor([[18, 4, 16]])
    with torch.no_grad():
        expected = model(ids, output_hidden_states=True, use_cache=False).hidden_states
    # transformers 5.x registers its own persistent per-layer hooks during
    # output_hidden_states forwards; only our hooks must not accumulate.
    baseline = [len(layer._forward_hooks) for layer in model.model.layers]
    for layer in (0, 1, 2):
        actual = hidden_at_layer(model, layer=layer, input_ids=ids)
        assert torch.allclose(actual, expected[layer], atol=1e-6)
        result = autoregressive_match(actual[0], model_candidates(model, layer=layer),
                                      vocab_size=19, chunk=7, topk=10)
        assert result['predictions'] == ids[0].tolist()
    assert [len(layer._forward_hooks) for layer in model.model.layers] == baseline


def test_vma_ties_and_invalid_states():
    result = autoregressive_match(torch.zeros(1, 2), lambda p, ids: torch.zeros(len(ids), 2),
                                  vocab_size=12, chunk=5, topk=3)
    assert result['rankings'].tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError):
        autoregressive_match(torch.ones(1, 2), lambda p, ids: torch.full((len(ids), 2), float('nan')),
                             vocab_size=12, chunk=3)


def test_vma_supports_public_candidate_proposals():
    def public_forward(prefix, ids):
        return ids.float().unsqueeze(-1)
    proposals = lambda prefix: torch.tensor([5, 2, 9])
    result = autoregressive_match(torch.tensor([[5.0]]), public_forward, vocab_size=12, chunk=2, candidate_ids_fn=proposals)
    assert result["predictions"] == [5]
    assert result["all_candidates_searched"] is False
    assert result["searched_candidate_sizes"] == [3]


def test_vma_pads_small_reduced_candidate_pool():
    def public_forward(prefix, ids):
        return ids.float().unsqueeze(-1)
    result = autoregressive_match(
        torch.tensor([[4.0]]), public_forward, vocab_size=8, chunk=2, topk=5,
        candidate_ids_fn=lambda prefix: torch.tensor([4, 2]),
    )
    assert result["rankings"].shape == (1, 5)
    assert result["rankings"].tolist() == [[4, 2, -1, -1, -1]]


@pytest.mark.parametrize("kind", ["llama", "qwen"])
@pytest.mark.parametrize("layer", [1, 2])
def test_cached_candidates_match_full_prefix_and_isolate_chunks(kind, layer):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(83)
    cls, config_cls = (LlamaForCausalLM, LlamaConfig) if kind == "llama" else (Qwen2ForCausalLM, Qwen2Config)
    model = cls(config_cls(vocab_size=19, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    cached = model_candidates(model, layer=layer, use_prefix_cache=True)
    reference = model_candidates(model, layer=layer, use_prefix_cache=False)
    for prefix in ((), (3,), (3,), (3, 8, 1), (7,), ()):
        for ids in (torch.tensor([18, 0, 4]), torch.tensor([6, 11]), torch.tensor([18, 0, 4])):
            assert torch.allclose(cached(prefix, ids), reference(prefix, ids), atol=1e-6)
    observations = torch.randn(4, 16) * 0.1
    actual = autoregressive_match(observations, cached, vocab_size=19, chunk=7, topk=10)
    expected = autoregressive_match(observations, reference, vocab_size=19, chunk=7, topk=10)
    assert actual["predictions"] == expected["predictions"]
    assert torch.equal(actual["rankings"], expected["rankings"])
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
