from __future__ import annotations

import hashlib
import json

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def _text_from_record(record: dict[str, Any]) -> str:
    if isinstance(record.get("text"), str):
        return record["text"]
    parts: list[str] = []
    for key in ("instruction", "input", "question", "context", "output", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not parts:
        raise ValueError("record has no supported text field")
    return "\n\n".join(parts)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_json_values(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL records must be objects")
                yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for index, record in enumerate(value):
            if not isinstance(record, dict):
                raise ValueError(f"{path}[{index}] must be an object")
            yield record
    elif isinstance(value, dict):
        yield value
    else:
        raise ValueError(f"{path} must contain an object or list of objects")


def iter_text_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[str]:
    count = 0
    for record in _iter_json_values(Path(path)):
        yield _text_from_record(record)
        count += 1
        if limit is not None and count >= limit:
            break


def read_text_jsonl(path: str | Path, *, limit: int | None = None) -> list[str]:
    rows = list(iter_text_jsonl(path, limit=limit))
    if not rows:
        raise ValueError(f"no text records found in {path}")
    return rows


@dataclass(frozen=True)
class TextJsonlIndex:
    """A seekable index that stores offsets, not tokenized examples."""

    path: Path
    offsets: tuple[int, ...]

    @classmethod
    def build(cls, path: str | Path, *, limit: int | None = None) -> "TextJsonlIndex":
        path = Path(path)
        offsets: list[int] = []
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                offsets.append(offset)
                if limit is not None and len(offsets) >= limit:
                    break
        if not offsets:
            raise ValueError(f"no JSONL records found in {path}")
        return cls(path.resolve(), tuple(offsets))

    def read_text(self, index: int) -> str:
        if not 0 <= index < len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            line = stream.readline().decode("utf-8")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"record at offset {self.offsets[index]} is not an object")
        return _text_from_record(record)

    def read_texts(self, indices: Sequence[int]) -> list[str]:
        return [self.read_text(index) for index in indices]

    def manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "records": len(self.offsets),
            "file_sha256": file_sha256(self.path),
        }


def split_hashes(paths: dict[str, str | Path]) -> dict[str, Any]:
    rows = {name: list(iter_text_jsonl(path)) for name, path in paths.items()}
    hashes = {name: [text_sha256(text) for text in values] for name, values in rows.items()}
    overlaps: dict[str, list[str]] = {}
    names = tuple(paths)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            shared = sorted(set(hashes[left]) & set(hashes[right]))
            if shared:
                overlaps[f"{left}__{right}"] = shared
    return {
        "splits": {
            name: {"path": str(Path(paths[name]).resolve()), "records": len(rows[name]),
                   "file_sha256": file_sha256(paths[name]), "text_sha256": hashes[name]}
            for name in names
        },
        "overlaps": overlaps,
        "status": "overlap_detected" if overlaps else "disjoint",
        "normalization": "collapse runs of whitespace before text hashing",
    }


def tokenize_texts(tokenizer: Any, texts: Iterable[str], *, max_length: int) -> dict[str, Any]:
    # Llama-3.2 sets padding_side="left"; the trainer packs valid rows from
    # the start of each row, so left padding would silently include pad rows
    # and drop the last real token whenever batch_size > 1. Force right padding
    # for training/validation and restore the tokenizer's setting afterwards.
    previous_side = getattr(tokenizer, "padding_side", None)
    if previous_side is not None:
        tokenizer.padding_side = "right"
    try:
        batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        if previous_side is not None:
            tokenizer.padding_side = previous_side
    if "attention_mask" not in batch:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            # Tokenizers without a pad token never pad: every position is real.
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        else:
            batch["attention_mask"] = batch["input_ids"].ne(pad_id).long()
    return batch
