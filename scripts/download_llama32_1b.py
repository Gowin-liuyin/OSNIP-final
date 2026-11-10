#!/usr/bin/env python3
"""Download and lock the Llama-3.2-1B base model.

The preferred source is the gated Hugging Face repository.  When Hugging Face
is unavailable (for example, the local account has no Meta approval),
``--source auto`` falls back to the ModelScope mirror used by this
reproduction.  The local directory is always checked as a LlamaForCausalLM
and every downloaded file is locked by SHA-256.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.data import file_sha256

MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
MODELSCOPE_MODEL_ID = "AI-ModelScope/Llama-3.2-1B"
MODELSCOPE_REVISION = "master"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--source", choices=("auto", "huggingface", "modelscope"), default="auto")
    ap.add_argument("--modelscope-model-id", default=MODELSCOPE_MODEL_ID)
    ap.add_argument("--modelscope-revision", default=MODELSCOPE_REVISION)
    ap.add_argument("--token-env", default="HF_TOKEN")
    return ap.parse_args()


def download_huggingface(args: argparse.Namespace) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    token = os.environ.get(args.token_env)
    auth_mode = "environment_token" if token else "cached_huggingface_login"
    download_token: str | bool = token if token else True
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(args.output_dir),
        token=download_token,
    )
    return {
        "source": "huggingface",
        "repository": args.model_id,
        "revision": args.revision,
        "auth_mode": auth_mode,
    }


def download_modelscope(args: argparse.Namespace) -> dict[str, str]:
    from modelscope import snapshot_download

    snapshot_download(
        model_id=args.modelscope_model_id,
        revision=args.modelscope_revision,
        local_dir=str(args.output_dir),
    )
    return {
        "source": "modelscope",
        "repository": args.modelscope_model_id,
        "revision": args.modelscope_revision,
        "auth_mode": "public_mirror",
    }


def validate_and_manifest(args: argparse.Namespace, provenance: dict[str, str], attempts: list[dict[str, str]]) -> Path:
    config_path = args.output_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"download completed without config.json: {config_path}")
    model_config = json.loads(config_path.read_text())
    architectures = model_config.get("architectures") or []
    if model_config.get("model_type") != "llama" or "LlamaForCausalLM" not in architectures:
        raise RuntimeError(
            "downloaded repository is not a LlamaForCausalLM: "
            f"model_type={model_config.get('model_type')!r}, architectures={architectures}"
        )

    files = {
        str(path.relative_to(args.output_dir)): file_sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "download_manifest.json"
    }
    manifest: dict[str, Any] = {
        "format": "osnip_llama_model_download_manifest_v3",
        "status": "ready",
        "model_id": args.model_id,
        "requested_huggingface_revision": args.revision,
        "provenance": provenance,
        "attempts": attempts,
        "local_dir": str(args.output_dir.resolve()),
        "model_type": model_config.get("model_type"),
        "architectures": architectures,
        "config_sha256": file_sha256(config_path),
        "files": files,
        "mirror_equivalence": "validated_by_local_file_hash_only",
    }
    output = args.output_dir / "download_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    args = parse_args()
    if args.model_id != MODEL_ID:
        raise ValueError(f"this reproduction is locked to {MODEL_ID}, got {args.model_id}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, str]] = []
    provenance: dict[str, str] | None = None
    sources = (("huggingface", download_huggingface), ("modelscope", download_modelscope)) if args.source == "auto" else ((args.source, download_huggingface if args.source == "huggingface" else download_modelscope),)
    for source, downloader in sources:
        try:
            provenance = downloader(args)
            break
        except Exception as exc:
            attempts.append({"source": source, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            if args.source != "auto":
                raise RuntimeError(f"unable to download Llama-3.2-1B from {source}: {exc}") from exc

    if provenance is None:
        details = "; ".join(f"{item['source']}: {item['error']}" for item in attempts)
        raise RuntimeError(f"unable to download {MODEL_ID} from Hugging Face or ModelScope: {details}")

    attempts.append({"source": provenance["source"], "status": "ready", "repository": provenance["repository"], "revision": provenance["revision"]})
    manifest = validate_and_manifest(args, provenance, attempts)
    print(json.dumps({"status": "ready", "model_id": args.model_id, "source": provenance["source"], "repository": provenance["repository"], "revision": provenance["revision"], "manifest": str(manifest.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
