import importlib.util
from pathlib import Path

import torch


def load_training_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_streaming.py"
    spec = importlib.util.spec_from_file_location("osnip_train_streaming", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mps_safe_packing_matches_boolean_mask_and_preserves_gradients():
    module = load_training_module()
    generator = torch.Generator().manual_seed(7)
    batch, length, width, vocab = 2, 6, 3, 5
    lengths = [6, 4]
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[0, :6] = True
    mask[1, :4] = True
    clean = torch.randn(batch, length, vocab, generator=generator, requires_grad=True)
    encrypted = torch.randn(batch, length, vocab, generator=generator, requires_grad=True)
    h = torch.randn(batch, length, width, generator=generator, requires_grad=True)
    z1 = torch.randn(batch, length, width, generator=generator, requires_grad=True)
    z2 = torch.randn(batch, length, width, generator=generator, requires_grad=True)

    causal = (mask[:, 1:] & mask[:, :-1]).reshape(-1)
    representation = mask.reshape(-1)
    expected = (
        clean[:, :-1].reshape(-1, vocab)[causal],
        encrypted[:, :-1].reshape(-1, vocab)[causal],
        h.reshape(-1, width)[representation],
        z1.reshape(-1, width)[representation],
        z2.reshape(-1, width)[representation],
    )
    actual = module.active_tensors(lengths, clean, encrypted, h, z1, z2)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    sum(value.square().sum() for value in actual).backward()
    assert clean.grad is not None and h.grad is not None


def test_tokenize_texts_falls_back_to_ones_without_pad_token():
    import torch
    from osnip_repro.data import tokenize_texts

    class DuckTokenizer:
        pad_token_id = None

        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.arange(6).reshape(len(texts), 3)}

    batch = tokenize_texts(DuckTokenizer(), ["a b", "b a"], max_length=8)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 1]]


def test_tokenize_texts_forces_right_padding_and_restores_the_tokenizer():
    from osnip_repro.data import tokenize_texts

    class LeftPaddingTokenizer:
        padding_side = "left"
        seen = None

        def __call__(self, texts, **kwargs):
            self.seen = self.padding_side
            return {"input_ids": torch.tensor([[1, 2], [3, 4]]),
                    "attention_mask": torch.ones(2, 2, dtype=torch.long)}

    tokenizer = LeftPaddingTokenizer()
    batch = tokenize_texts(tokenizer, ["a", "b"], max_length=8)
    assert tokenizer.seen == "right"
    assert tokenizer.padding_side == "left"
    assert batch["input_ids"].shape == (2, 2)
