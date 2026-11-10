"""Task-prompt truncation must preserve the options and the Answer cue."""
from scripts.evaluate_task_utility import encode_prompt_answer


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": list(range(len(str(text).split())))}


def test_long_stem_is_truncated_but_options_and_answer_cue_survive():
    tokenizer = WordTokenizer()
    stem = " ".join(f"w{i}" for i in range(100))
    options = "A: alpha\nB: beta\nC: gamma\nAnswer:"
    encoded = encode_prompt_answer(tokenizer, stem, options, "A", max_length=12, prompt_token_budget=11)
    assert encoded is not None
    prompt_ids, answer_ids, truncated = encoded
    assert truncated is True
    assert len(prompt_ids) == 11
    assert len(answer_ids) == 1
    # the last tokens are exactly the options + Answer cue, never the stem tail
    option_ids = tokenizer(options, add_special_tokens=False)["input_ids"]
    assert prompt_ids[-len(option_ids):] == option_ids


def test_options_that_do_not_fit_are_refused_instead_of_scored():
    tokenizer = WordTokenizer()
    options = "A: alpha\nB: beta\nC: gamma\nAnswer:"
    assert encode_prompt_answer(tokenizer, "stem", options, "A", max_length=4, prompt_token_budget=3) is None


def test_short_stem_is_unchanged():
    tokenizer = WordTokenizer()
    encoded = encode_prompt_answer(tokenizer, "short question", "A: x\nAnswer:", "A", max_length=20, prompt_token_budget=19)
    assert encoded is not None
    prompt_ids, _, truncated = encoded
    assert truncated is False
