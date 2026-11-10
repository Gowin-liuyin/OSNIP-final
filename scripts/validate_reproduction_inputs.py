#!/usr/bin/env python3
"""Validate all fixed inputs before a Llama-3.2-1B OSNIP run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/llama32_1b_question_disjoint_v2.json")
    ap.add_argument("--data-manifest", type=Path, default=ROOT / "data/osnip_65k_question_disjoint_v2/data_manifest.json")
    ap.add_argument("--attack-manifest", type=Path, default=ROOT / "data/paper_attack_sets/attack_manifest.json")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def fail(checks: list[dict[str, Any]], name: str, detail: str) -> None:
    checks.append({"name": name, "status": "failed", "detail": detail})


def passed(checks: list[dict[str, Any]], name: str, **details: Any) -> None:
    checks.append({"name": name, "status": "passed", **details})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def check_expected_hash(checks: list[dict[str, Any]], name: str, path: Path, expected: str | None) -> None:
    if not path.is_file():
        fail(checks, name, f"missing: {path}")
        return
    actual = sha256(path)
    if expected and expected != actual:
        fail(checks, name, f"sha256 mismatch: expected {expected}, got {actual}")
    else:
        passed(checks, name, path=str(path.resolve()), sha256=actual)



def validate_question_disjoint_policy(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    attack_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    """Bind raw-source verification and fixed exclusions to the run's config."""
    from verify_question_disjoint_training import verify_dataset

    data = config["data"]
    if data.get("dataset_policy") != "question_disjoint_v2" or data.get("fresh_training_required") is not True:
        raise ValueError("v2 requires explicit question_disjoint_v2 policy and fresh_training_required=true")
    if manifest.get("status") != "ready_for_fresh_training_only":
        raise ValueError("v2 dataset is not ready for fresh training")
    for name in ("manifest_sha256", "raw_sources_manifest_sha256", "attack_manifest_sha256"):
        value = data.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("v2 input fingerprint is not pinned: " + name)
    pinned = data.get("exclusion_sha256", {})
    required = {"knn_reserved", "knn", "vma", "sciq_confirmatory"}
    if set(pinned) != required or set(manifest.get("exclusions", {})) != required:
        raise ValueError("v2 must pin all four fixed exclusion sets")
    for name in sorted(required):
        spec = manifest["exclusions"][name]
        if len(str(pinned[name])) != 64 or pinned[name] != spec.get("file_sha256"):
            raise ValueError("v2 exclusion fingerprint mismatch: " + name)
        if name in {"knn", "vma"} and Path(spec["path"]).resolve() != (attack_dir / (name + ".jsonl")).resolve():
            raise ValueError("v2 exclusion does not reference the evaluated attack file: " + name)
    if config["objective"].get("diversity_space") != "raw_r" or config["training"].get("diversity_space") != "raw_r":
        raise ValueError("v2 main reproduction requires raw_r Eq.19 diversity")
    for source in ("alpaca", "arc_easy", "arc_challenge", "sciq"):
        expected = data.get("source_revisions", {}).get(source)
        if not expected or manifest["sources"][source].get("revision") != expected:
            raise ValueError("v2 unpinned or different source revision: " + source)
    if set(manifest["splits"]) != {"train", "valid", "test"}:
        raise ValueError("v2 requires exactly train/valid/test")
    for name, spec in manifest["splits"].items():
        expected = data.get(name + "_sha256")
        if not expected or len(expected) != 64 or spec.get("file_sha256") != expected:
            raise ValueError("v2 split fingerprint missing or mismatched: " + name)
        if Path(spec["path"]).resolve() != (data_dir / (name + ".jsonl")).resolve():
            raise ValueError("v2 split path differs from the validated training input: " + name)
    fraction = manifest["construction"]["general_fraction_actual"]
    if data.get("general_fraction") != fraction or data.get("general_fraction_is_paper_exact") is not False:
        raise ValueError("v2 must disclose actual general fraction and paper-ratio deviation")
    science_fraction = data.get("science_fraction")
    if isinstance(science_fraction, bool) or not isinstance(science_fraction, (int, float)) or not math.isclose(science_fraction, 1.0 - fraction, rel_tol=0, abs_tol=1e-12):
        raise ValueError("v2 science fraction must match the actual non-Alpaca fraction")
    for name in sorted(required):
        rows = read_jsonl(Path(manifest["exclusions"][name]["path"]))
        expected_count = 200 if name == "vma" else 1000
        if len(rows) != expected_count:
            raise ValueError("v2 fixed exclusion has wrong record count: " + name)
        if name in {"knn", "knn_reserved"}:
            limits = data["knn"]
            if any(not limits["min_chars"] <= len(" ".join(row["text"].split())) <= limits["max_chars"] for row in rows):
                raise ValueError("v2 KNN character filter mismatch")
    return verify_dataset(manifest)


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    config = json.loads(args.config.read_text())
    if not args.model_path.is_dir():
        fail(checks, "model_directory", f"missing: {args.model_path}")
    else:
        model_config_path = args.model_path / "config.json"
        if not model_config_path.is_file():
            fail(checks, "model_files", f"missing: {model_config_path}")
        else:
            model_config = json.loads(model_config_path.read_text())
            architectures = model_config.get("architectures") or []
            if model_config.get("model_type") != "llama" or "LlamaForCausalLM" not in architectures:
                fail(checks, "model_architecture", f"expected LlamaForCausalLM/llama, got model_type={model_config.get('model_type')!r}, architectures={architectures}")
            else:
                passed(checks, "model_architecture", model_type=model_config.get("model_type"), architectures=architectures)
            passed(checks, "model_directory", model_id=config["model"]["model_id"], requested_model_revision=config["model"]["revision"])
            for name, key in (("config.json", "config_sha256"), ("model.safetensors", "weight_sha256"), ("tokenizer.json", "tokenizer_sha256")):
                expected = config["model"].get(key)
                if not expected or len(expected) != 64:
                    fail(checks, key, "model fingerprint is not pinned")
                else:
                    check_expected_hash(checks, key, args.model_path / name, expected)

    data_manifest: dict[str, Any] | None = None
    attack_manifest: dict[str, Any] | None = None
    is_question_disjoint = False
    if not args.data_manifest.is_file():
        fail(checks, "data_manifest", f"missing: {args.data_manifest}")
    else:
        data_manifest = json.loads(args.data_manifest.read_text())
        is_question_disjoint = data_manifest.get("format") == "osnip_question_disjoint_dataset_manifest_v1"
        expected_status = "ready_for_fresh_training_only" if is_question_disjoint else "ready"
        if data_manifest.get("status") != expected_status:
            fail(checks, "data_manifest", f"status is {data_manifest.get('status')!r}")
        else:
            passed(checks, "data_manifest", sha256=sha256(args.data_manifest))
        check_expected_hash(checks, "data_manifest_hash", args.data_manifest, config["data"].get("manifest_sha256"))
        raw_manifest = ROOT / "data/osnip_sources/raw_sources_manifest.json"
        check_expected_hash(checks, "raw_sources_manifest_hash", raw_manifest, config["data"].get("raw_sources_manifest_sha256"))

    if not args.attack_manifest.is_file():
        fail(checks, "attack_manifest", f"missing: {args.attack_manifest}")
    else:
        attack_manifest = json.loads(args.attack_manifest.read_text())
        if attack_manifest.get("status") != "ready":
            fail(checks, "attack_manifest", f"status is {attack_manifest.get('status')!r}")
        else:
            passed(checks, "attack_manifest", sha256=sha256(args.attack_manifest))
        check_expected_hash(checks, "attack_manifest_hash", args.attack_manifest, config["data"].get("attack_manifest_sha256"))

    if data_manifest is not None:
        counts = {name: info["records"] for name, info in data_manifest.get("splits", {}).items()}
        if counts != {"train": 65000, "valid": 1000, "test": 1000}:
            fail(checks, "data_counts", f"expected 65000/1000/1000, got {counts}")
        elif data_manifest.get("overlaps"):
            fail(checks, "data_overlap", f"overlap keys: {sorted(data_manifest['overlaps'])}")
        else:
            passed(checks, "data_counts_and_overlap", counts=counts)

        split_expected = {
            "train": config["data"].get("train_sha256"),
            "valid": config["data"].get("valid_sha256"),
            "test": config["data"].get("test_sha256"),
        }
        split_ok = True
        for name, expected in split_expected.items():
            info = data_manifest.get("splits", {}).get(name, {})
            actual = info.get("file_sha256")
            check_expected_hash(checks, f"{name}_actual_file_hash", args.data_manifest.parent / f"{name}.jsonl", expected or actual)
            if expected and actual != expected:
                split_ok = False
                fail(checks, f"{name}_hash", f"manifest hash mismatch: expected {expected}, got {actual}")
        if split_ok:
            passed(checks, "split_hashes", hashes={name: data_manifest["splits"][name]["file_sha256"] for name in split_expected})

        expected_revisions = config["data"].get("source_revisions", {})
        revision_failures = []
        for source, expected in expected_revisions.items():
            actual = data_manifest.get("sources", {}).get(source, {}).get("revision")
            if actual is not None and actual != expected:
                revision_failures.append(f"{source}: expected {expected}, got {actual}")
        if revision_failures:
            fail(checks, "source_revisions", "; ".join(revision_failures))
        else:
            passed(checks, "source_revisions", revisions={source: data_manifest.get("sources", {}).get(source, {}).get("revision") for source in expected_revisions if source in data_manifest.get("sources", {})})

        if is_question_disjoint:
            try:
                verified = validate_question_disjoint_policy(config, data_manifest, attack_dir=args.attack_manifest.parent, data_dir=args.data_manifest.parent)
            except (ValueError, KeyError, OSError, AssertionError) as exc:
                fail(checks, "question_disjoint_policy", str(exc))
            else:
                passed(checks, "question_disjoint_policy", **verified)
        else:
            if config["data"].get("dataset_policy") == "question_disjoint_v2":
                fail(checks, "question_disjoint_policy", "v2 config cannot use a legacy data manifest")
            holdout = data_manifest.get("knn_holdout", {})
            reserve_path = Path(holdout.get("path", ""))
            if holdout.get("records") != 1000:
                fail(checks, "knn_holdout_count", f"expected 1000, got {holdout.get('records')}")
            elif not reserve_path.is_file():
                fail(checks, "knn_holdout_file", f"missing: {reserve_path}")
            else:
                reserve_rows = read_jsonl(reserve_path)
                train_rows = read_jsonl(Path(data_manifest["splits"]["train"]["path"]))
                train_indices = {row.get("source_index") for row in train_rows if row.get("source") == "alpaca"}
                reserve_indices = {row.get("source_index") for row in reserve_rows}
                if len(reserve_rows) != 1000 or train_indices & reserve_indices:
                    fail(checks, "knn_holdout_disjoint", f"reserve_rows={len(reserve_rows)}, source_index_overlap={len(train_indices & reserve_indices)}")
                else:
                    passed(checks, "knn_holdout_disjoint", records=len(reserve_rows), source_index_overlap=0)
                check_expected_hash(checks, "knn_holdout_hash", reserve_path, holdout.get("file_sha256"))

            knn_filter = data_manifest.get("construction", {}).get("knn_filter", {})
            expected_knn = config["data"].get("knn", {})
            if {"min_chars": knn_filter.get("min_chars"), "max_chars": knn_filter.get("max_chars")} != {"min_chars": expected_knn.get("min_chars"), "max_chars": expected_knn.get("max_chars")}:
                fail(checks, "knn_filter", f"expected {expected_knn.get('min_chars')}-{expected_knn.get('max_chars')}, got {knn_filter}")
            else:
                passed(checks, "knn_filter", min_chars=knn_filter.get("min_chars"), max_chars=knn_filter.get("max_chars"))

    if attack_manifest is not None:
        knn = attack_manifest.get("knn", {})
        vma = attack_manifest.get("vma", {})
        if knn.get("records") != 1000 or vma.get("records") != 200:
            fail(checks, "attack_counts", f"expected KNN=1000,VMA=200, got {knn.get('records')},{vma.get('records')}")
        else:
            passed(checks, "attack_counts", knn=1000, vma=200)
        if not knn.get("reserved_holdout"):
            fail(checks, "attack_holdout_policy", "KNN attack set is not marked as reserved holdout")
        else:
            passed(checks, "attack_holdout_policy", reserved_holdout=True)
        source_aliases = {
            "Alpaca": "alpaca",
            "ARC-Easy": "arc_easy",
            "ARC-Challenge": "arc_challenge",
            "SciQ": "sciq",
            "WikiText-103": "wikitext",
        }
        expected_sources = {source_aliases.get(value, value) for value in config["data"].get("vma", {}).get("sources", [])}
        actual_sources = set(vma.get("sources", []))
        if not expected_sources.issubset(actual_sources):
            fail(checks, "vma_sources", f"missing sources: {sorted(expected_sources - actual_sources)}")
        else:
            passed(checks, "vma_sources", sources=sorted(actual_sources))
        if vma.get("keys_per_prompt") != config["data"].get("vma", {}).get("keys_per_prompt"):
            fail(checks, "vma_keys_per_prompt", f"expected {config['data']['vma'].get('keys_per_prompt')}, got {vma.get('keys_per_prompt')}")
        else:
            passed(checks, "vma_keys_per_prompt", keys_per_prompt=vma.get("keys_per_prompt"))
        for name in ("knn", "vma"):
            path = args.attack_manifest.parent / f"{name}.jsonl"
            expected = attack_manifest.get(name, {}).get("file_sha256")
            check_expected_hash(checks, f"{name}_file_hash", path, expected)

    if config["model"]["model_id"] != "meta-llama/Llama-3.2-1B":
        fail(checks, "model_identity", config["model"]["model_id"])
    else:
        passed(checks, "model_identity", model_id=config["model"]["model_id"])
    status = "ready" if checks and all(item["status"] == "passed" for item in checks) else "failed"
    report = {"format": "osnip_reproduction_input_validation_v2", "status": status, "model_path": str(args.model_path.resolve()), "config": str(args.config.resolve()), "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "output": str(args.output.resolve()), "failed_checks": [x for x in checks if x["status"] == "failed"]}, ensure_ascii=False, indent=2))
    if status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
