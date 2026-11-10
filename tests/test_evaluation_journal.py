import pytest
from osnip_repro.evaluation_journal import EvaluationJournal


def test_journal_resumes_exact_observations_and_rejects_rebinding(tmp_path):
    path = tmp_path / 'vma.jsonl'
    binding = {'checkpoint': 'first', 'seed': 4, 'full_vocabulary': True}
    journal = EvaluationJournal(path, binding)
    row = {'prompt': 0, 'layer': 5, 'key': -1, 'positions': 3, 'total_top1_hits': 3}
    journal.append(row)
    resumed = EvaluationJournal(path, binding, resume=True)
    assert resumed.get(0, 5, -1) == row
    assert resumed.get(0, 5, 0) is None
    with pytest.raises(FileExistsError):
        EvaluationJournal(path, binding)
    with pytest.raises(ValueError, match='binding mismatch'):
        EvaluationJournal(path, {**binding, 'checkpoint': 'second'}, resume=True)
    with pytest.raises(ValueError, match='duplicate'):
        resumed.append(row)
    assert len(EvaluationJournal(path, binding, resume=True).rows) == 1


def test_resume_reports_truncated_tail_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = EvaluationJournal(path, {"seed": 1})
    journal.append({"prompt": 0, "layer": 1, "key": 0, "positions": 2})
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"prompt": 1, "lay')
    with pytest.raises(ValueError, match="truncated tail line"):
        EvaluationJournal(path, {"seed": 1}, resume=True)


def test_bounds_refuse_out_of_protocol_rows(tmp_path):
    path = tmp_path / "bounded.jsonl"
    bounds = {"prompts": 2, "layers": (1,), "key_min": -1, "keys_per_prompt": 2}
    journal = EvaluationJournal(path, {"seed": 1}, bounds=bounds)
    journal.append({"prompt": 1, "layer": 1, "key": -1, "positions": 1})
    with pytest.raises(ValueError, match="outside declared protocol bounds"):
        journal.append({"prompt": 5, "layer": 1, "key": 0, "positions": 1})
    with pytest.raises(ValueError, match="outside declared protocol bounds"):
        journal.append({"prompt": 0, "layer": 9, "key": 0, "positions": 1})
    resumed = EvaluationJournal(path, {"seed": 1}, resume=True, bounds=bounds)
    assert (1, 1, -1) in resumed.rows


def test_append_serializes_before_writing(tmp_path):
    import numpy as np
    path = tmp_path / "serialize.jsonl"
    journal = EvaluationJournal(path, {"seed": 2})
    with pytest.raises(TypeError):
        journal.append({"prompt": 0, "layer": 1, "key": 0, "positions": np.int64(3)})
    assert len(path.read_text().splitlines()) == 1
    assert journal.rows == {}
