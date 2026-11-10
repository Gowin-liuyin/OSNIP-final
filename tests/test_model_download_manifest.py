import importlib.util
import json
from argparse import Namespace
from pathlib import Path


def load_downloader():
    path = Path(__file__).resolve().parents[1] / "scripts/download_llama32_1b.py"
    spec = importlib.util.spec_from_file_location("download_llama32_1b", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_modelscope_provenance_manifest_is_hash_locked(tmp_path: Path):
    module = load_downloader()
    output = tmp_path / "model"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    (output / "model.safetensors").write_bytes(b"weights")
    args = Namespace(model_id=module.MODEL_ID, revision=module.MODEL_REVISION, output_dir=output)
    manifest_path = module.validate_and_manifest(
        args,
        {"source": "modelscope", "repository": module.MODELSCOPE_MODEL_ID, "revision": "master", "auth_mode": "public_mirror"},
        [{"source": "modelscope", "status": "ready"}],
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "ready"
    assert manifest["provenance"]["source"] == "modelscope"
    assert manifest["files"]["model.safetensors"]
    assert manifest["mirror_equivalence"] == "validated_by_local_file_hash_only"
