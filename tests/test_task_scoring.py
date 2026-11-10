import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_task_utility import answer_nll, choice_nlls


class Tokenizer:
    def __call__(self, text, **kwargs):
        if text == "":
            return {"input_ids": []}
        if text.startswith(" "):
            return {"input_ids": [4] if text.strip() == "A" else [5] if text.strip() == "B" else [6, 7]}
        return {"input_ids": [1, 2, 3, 8, 9, 10]}


class Encryptor(torch.nn.Module):
    def forward(self, h, key):
        return h + 0.1 * key[:h.shape[-1]]


@pytest.mark.parametrize("kind", ["llama", "qwen"])
@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("labels", [["A", "B"], ["A", "multi"]])
def test_choice_scores_match_individual_teacher_forcing(kind, encrypted, labels):
    torch.manual_seed(42)
    cls, config_cls = (LlamaForCausalLM, LlamaConfig) if kind == "llama" else (Qwen2ForCausalLM, Qwen2Config)
    model = cls(config_cls(vocab_size=19, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    tokenizer, encoder, device = Tokenizer(), Encryptor(), torch.device("cpu")
    key = torch.randn(16) if encrypted else None
    expected = [answer_nll(model, tokenizer, encoder, "prompt", "", label, key=key,
                           max_length=6, prompt_token_budget=4, device=device) for label in labels]
    actual = choice_nlls(model, tokenizer, encoder, "prompt", "", labels, key=key,
                         max_length=6, prompt_token_budget=4, device=device)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_sciq_choices_shuffle_reproducibly_with_tracked_correct_index():
    from evaluate_task_utility import sciq_task
    record = {"question": "test question", "correct_answer": "right", "distractor1": "wrong1",
              "distractor2": "wrong2", "distractor3": "wrong3", "support": "context"}
    correct_positions = set()
    for seed in range(12):
        result = sciq_task(record, seed=seed)
        assert result == sciq_task(record, seed=seed)
        stem, options, labels, correct = result
        assert f"{labels[correct]}: right" in options
        assert "Context: context" in stem
        correct_positions.add(correct)
    assert len(correct_positions) == 4


def test_sciq_empty_correct_answer_is_rejected():
    from evaluate_task_utility import sciq_task
    record = {"question": "q", "correct_answer": "", "distractor1": "a",
              "distractor2": "b", "distractor3": "c"}
    assert sciq_task(record, seed=0) is None


def test_paired_task_bootstrap_preserves_pairing_and_reports_source_losses():
    from summarize_task_utility import summarize
    rows = [{'source': 'arc', 'correct': 'A', 'clean_correct': True, 'encrypted_correct': False},
            {'source': 'arc', 'correct': 'B', 'clean_correct': True, 'encrypted_correct': False},
            {'source': 'sciq', 'correct': 'C', 'clean_correct': True, 'encrypted_correct': True}]
    result = summarize(rows, seed=7, samples=100)
    assert result['arc']['retention'] == 0
    assert result['arc']['retention_bootstrap_95ci'] == [0., 0.]
    assert result['arc']['paired_accuracy_delta_bootstrap_95ci'] == [-1., -1.]
    assert result['sciq']['retention_bootstrap_95ci'] == [1., 1.]
    assert result['all']['correct_label_counts'] == {'A': 1, 'B': 1, 'C': 1}
