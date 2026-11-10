"""SciQ training text must not always put the correct answer first."""
from scripts.prepare_osnip_datasets import source_text

RECORD = {"question": "q", "correct_answer": "right", "distractor1": "wrong1",
          "distractor2": "wrong2", "distractor3": "wrong3", "support": "ctx"}


def test_sciq_training_text_has_answer_cue_and_shuffled_options():
    text = source_text(RECORD, "sciq", 0, sciq_answer_cue=True)
    assert "Answer: right" in text
    assert "Options:" in text
    assert text == source_text(RECORD, "sciq", 0, sciq_answer_cue=True)


def test_legacy_sciq_format_has_no_answer_cue():
    text = source_text(RECORD, "sciq", 0)
    assert "Answer:" not in text
    assert text.startswith("q\nContext: ctx\nOptions: right | wrong1")


def test_correct_answer_is_not_always_first_across_records():
    option_lines = []
    for index in range(20):
        text = source_text(RECORD, "sciq", index, sciq_answer_cue=True)
        line = next(line for line in text.splitlines() if line.startswith("Options:"))
        option_lines.append(line)
    assert len(set(option_lines)) > 1
    assert any(line.split("Options: ")[1].split(" | ")[0] != "right" for line in option_lines)


def test_empty_gold_answer_never_gets_a_fabricated_answer_cue():
    from scripts.prepare_osnip_datasets import source_text
    record = {"question": "q", "correct_answer": "", "distractor1": "a",
              "distractor2": "b", "distractor3": "c"}
    text = source_text(record, "sciq", 0, sciq_answer_cue=True)
    assert "Answer:" not in text
    assert "Options:" in text


def test_sciq_shuffle_honors_the_seed():
    from scripts.prepare_osnip_datasets import source_text
    a = source_text(RECORD, "sciq", 3, sciq_answer_cue=True, sciq_seed=7)
    b = source_text(RECORD, "sciq", 3, sciq_answer_cue=True, sciq_seed=7)
    assert a == b
    orders = {source_text(RECORD, "sciq", 3, sciq_answer_cue=True, sciq_seed=s)
              for s in range(8)}
    assert len(orders) > 1


def test_load_source_excludes_official_test_split(tmp_path):
    import json
    from scripts.prepare_osnip_datasets import load_source
    path = tmp_path / "sciq.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (
        {"question": "train q", "correct_answer": "a", "_source_split": "train"},
        {"question": "test q", "correct_answer": "b", "_source_split": "test"},
    )) + "\n")
    rows = load_source(path, "sciq", 1, exclude_official_test=True)
    assert len(rows) == 1
    assert rows[0]["source_index"] == 0
    assert len(load_source(path, "sciq", 1)) == 2
