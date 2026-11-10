import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from osnip_repro.encoder import build_encryptor_from_config


def test_paper_protocol_knn_reuse_and_interrupted_vma_resume(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_paper_protocol.py'
    spec = importlib.util.spec_from_file_location('paper_protocol_resume_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(71)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=19, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'a': 3, 'b': 4, 'c': 5}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    config = {'model_path': str(model_dir), 'width': 16, 'key_dim': 4, 'hidden_dim': 8}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    checkpoint = tmp_path / 'encoder.pt'
    torch.save(build_encryptor_from_config(config).state_dict(), checkpoint)
    data = tmp_path / 'data.jsonl'
    data.write_text(json.dumps({'text': 'a b c'}) + '\n' + json.dumps({'text': 'c a'}) + '\n')
    args = argparse.Namespace(model_path=model_dir, knn_jsonl=data, vma_jsonl=data, checkpoint=checkpoint,
        output=tmp_path / 'knn.json', layers=[1, 2], keys_per_prompt=2, max_knn_samples=2,
        max_vma_prompts=2, max_tokens=16, vocab_chunk=7, candidate_chunk=7, candidate_mode='full',
        candidate_k=10, device='cpu', dtype='float32', seed=42, bootstrap_samples=20,
        knn_only=True, knn_report=None, resume_vma=False, progress_every=1)
    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'load_causal_lm', lambda *a, **kw: (tokenizer, model, torch.device('cpu')))
    monkeypatch.setattr(module, 'stopwords', lambda: (set(), 'test'))
    module.main()
    knn = json.loads(args.output.read_text())
    assert knn['samples'] == 2 and knn['rates']['clean']['top1'] == 1
    assert not knn['security_accepted']
    args.knn_only = False
    args.knn_report = args.output
    args.output = tmp_path / 'resumed.json'
    real_match = module.autoregressive_match
    calls = []

    def interrupted(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('interrupted test')
        return real_match(*a, **kw)

    monkeypatch.setattr(module, 'autoregressive_match', interrupted)
    with pytest.raises(RuntimeError, match='interrupted test'):
        module.main()
    assert not args.output.exists()
    assert len(args.output.with_suffix('.vma_rows.jsonl').read_text().splitlines()) == 2  # binding + clean observation
    args.resume_vma = True
    calls.clear()

    def tracked(*a, **kw):
        calls.append(1)
        return real_match(*a, **kw)

    monkeypatch.setattr(module, 'autoregressive_match', tracked)
    module.main()
    resumed = json.loads(args.output.read_text())
    assert len(calls) == 11  # 2 prompts * 2 layers * (clean + 2 keys), minus completed clean
    assert len(resumed['vma']['rows']) == 8
    assert len(resumed['vma']['clean_rows']) == 4
    assert len(resumed['knn']['rows']) == 2
    aggregate_spec = importlib.util.spec_from_file_location('actual_privacy_evidence', script.parent/'aggregate_reproduction_report.py')
    aggregate = importlib.util.module_from_spec(aggregate_spec); aggregate_spec.loader.exec_module(aggregate)
    evidence = aggregate.verify_privacy_observations(resumed)
    assert evidence['passed'], evidence
    assert evidence['vma_clean_observations_verified'] == 4
    assert evidence['vma_encrypted_observations_verified'] == 8
    assert resumed['vma']['full_vocabulary'] and not resumed['security_accepted']
    args.output = tmp_path / 'fresh.json'
    args.resume_vma = False
    module.main()
    fresh = json.loads(args.output.read_text())
    assert resumed['knn'] == fresh['knn']
    assert resumed['vma'] == fresh['vma']
    args.seed += 1
    args.output = tmp_path / 'invalid.json'
    with pytest.raises(ValueError, match='does not match'):
        module.main()


def test_clean_positive_control_is_enforced():
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_paper_protocol.py'
    spec = importlib.util.spec_from_file_location('paper_protocol_control_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.require_clean_positive_control({"clean": {"top1": 1.0}})
    with pytest.raises(ValueError, match="positive control"):
        module.require_clean_positive_control({"clean": {"top1": 0.4}})
    with pytest.raises(ValueError, match="positive control"):
        module.require_clean_positive_control({})


def test_stopword_mask_is_whole_word_from_source_text():
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_paper_protocol.py'
    spec = importlib.util.spec_from_file_location('paper_protocol_stopword_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = Tokenizer(WordLevel({"The,": 0, "cat": 1, "sat": 2, "[UNK]": 99}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    text = "The, cat sat"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    # Whitespace pre-tokenization splits the comma into its own piece; every
    # piece inherits its source word's verdict, so "The," and "," are both stop.
    assert module.word_stopword_mask(tokenizer, text, ids, {"the"}) == [True, True, False, False]
    with pytest.raises(ValueError, match="offset mapping"):
        module.word_stopword_mask(tokenizer, text, ids + [1], {"the"})


def test_stopword_mask_length_matches_truncated_ids():
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_paper_protocol.py'
    spec = importlib.util.spec_from_file_location('paper_protocol_trunc_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = Tokenizer(WordLevel({"The,": 0, "cat": 1, "sat": 2, "[UNK]": 99}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    text = "The, cat sat"
    full = tokenizer(text, add_special_tokens=False)["input_ids"]
    mask = module.word_stopword_mask(tokenizer, text, full[:2], {"the"})
    assert mask == [True, True]


class ByteLevelOffsetTokenizer:
    """Minimal byte-level stand-in: word-initial offsets include the leading space."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re
        ids, offsets = [], []
        for index, match in enumerate(re.finditer(r"\S+", text)):
            start = match.start()
            if start > 0 and text[start - 1].isspace():
                start -= 1
            ids.append(index)
            offsets.append((start, match.end()))
        return {"input_ids": ids, "offset_mapping": offsets}


def test_stopword_mask_handles_byte_level_leading_space_offsets():
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_paper_protocol.py'
    spec = importlib.util.spec_from_file_location('paper_protocol_bytelevel_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = ByteLevelOffsetTokenizer()
    text = "The cat sat on the mat"
    ids = ["The", "cat", "sat", "on", "the", "mat"]
    mask = module.word_stopword_mask(tokenizer, text, ids, {"the", "on"})
    assert mask == [True, False, False, True, True, False]
