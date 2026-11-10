"""Bound, append-only journal of completed independent VMA observations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class EvaluationJournal:
    """Append-only journal of completed observations.

    ``bounds`` (optional) declares the protocol envelope
    ``{"prompts": n, "layers": (…), "key_min": k0, "keys_per_prompt": k}``;
    rows outside it are refused on append and on resume load so a corrupted or
    mis-bound journal cannot silently grow past the declared protocol.
    """

    def __init__(self, path: Path, binding: dict[str, Any], *, resume: bool = False,
                 bounds: dict[str, Any] | None = None):
        self.path = path
        self.rows: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.bounds = bounds
        self.binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        if path.exists():
            if not resume:
                raise FileExistsError(f"VMA journal already exists; use --resume-vma: {path}")
            lines = path.read_text().splitlines()
            if not lines or json.loads(lines[0]).get("binding_sha256") != self.binding:
                raise ValueError("VMA journal binding mismatch: model/checkpoint/data/protocol/source changed")
            for index, line in enumerate(lines[1:]):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if index == len(lines) - 2:
                        raise ValueError(
                            "truncated tail line: the journal may have crashed mid-append; "
                            "manually remove the incomplete final line before using --resume-vma"
                        ) from exc
                    raise
                self.check_bounds(row)
                key = self.key(row)
                if key in self.rows:
                    raise ValueError(f"duplicate completed VMA observation: {key}")
                self.rows[key] = row
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(json.dumps({"format": "osnip_vma_journal_v1", "binding_sha256": self.binding,
                                         "binding": binding}) + "\n")

    @staticmethod
    def key(row: dict[str, Any]) -> tuple[int, int, int]:
        return int(row["prompt"]), int(row["layer"]), int(row["key"])

    def check_bounds(self, row: dict[str, Any]) -> None:
        if not self.bounds:
            return
        prompt, layer, key = self.key(row)
        inside = (0 <= prompt < int(self.bounds["prompts"])
                  and layer in tuple(self.bounds["layers"])
                  and int(self.bounds["key_min"]) <= key < int(self.bounds["keys_per_prompt"]))
        if not inside:
            raise ValueError(f"journal row outside declared protocol bounds: {(prompt, layer, key)}")

    def get(self, prompt: int, layer: int, key: int) -> dict[str, Any] | None:
        return self.rows.get((prompt, layer, key))

    def append(self, row: dict[str, Any]) -> None:
        self.check_bounds(row)
        key = self.key(row)
        if key in self.rows:
            raise ValueError(f"duplicate completed VMA observation: {key}")
        # Serialize before touching the file: a non-serializable value must
        # never leave a half-written row behind while memory stays clean.
        line = json.dumps(row, allow_nan=False)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows[key] = row
