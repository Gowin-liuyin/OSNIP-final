"""Official test-split task set builder keeps only _source_split=test rows."""
import json
import sys
from pathlib import Path
from unittest import mock

import scripts.prepare_task_test_set as builder


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_builder_writes_only_official_test_rows(tmp_path):
    arc_easy = tmp_path / "arc_easy.jsonl"
    arc_c = tmp_path / "arc_challenge.jsonl"
    sciq = tmp_path / "sciq.jsonl"
    write_jsonl(arc_easy, [{"question": "train", "_source_split": "train"},
                           {"question": "test", "_source_split": "test"}])
    write_jsonl(arc_c, [{"question": "c", "_source_split": "test"}])
    write_jsonl(sciq, [{"question": "s", "_source_split": "test"}])
    out = tmp_path / "out"
    argv = ["prepare_task_test_set", "--arc-easy", str(arc_easy),
            "--arc-challenge", str(arc_c), "--sciq", str(sciq), "--output-dir", str(out)]
    with mock.patch.object(sys, "argv", argv):
        builder.main()
    rows = [json.loads(line) for line in (out / "task_test.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0] == {"source": "arc_easy", "source_index": 1}
    manifest = json.loads((out / "task_test.manifest.json").read_text())
    assert manifest["split"] == "test"
    assert manifest["source_counts"] == {"arc_easy": 1, "arc_challenge": 1, "sciq": 1}
