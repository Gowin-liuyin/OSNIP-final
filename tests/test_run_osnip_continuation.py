import json
import pytest
from scripts.run_osnip_continuation import verify_inputs,write_json


def test_bound_continuation_refuses_input_drift(tmp_path,monkeypatch):
    import scripts.run_osnip_continuation as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    source=tmp_path/'train.py';source.write_text('locked')
    checkpoint=tmp_path/'parent.pt';checkpoint.write_bytes(b'checkpoint')
    from osnip_repro.data import file_sha256
    manifest={'bound_inputs':{'train.py':file_sha256(source)},
              'parent_checkpoint':str(checkpoint),
              'parent_checkpoint_sha256':file_sha256(checkpoint)}
    verify_inputs(manifest)
    source.write_text('changed')
    with pytest.raises(ValueError,match='input changed'):
        verify_inputs(manifest)


def test_json_writer_replaces_status_atomically_and_rejects_nan(tmp_path):
    path=tmp_path/'status.json'
    write_json(path,{'state':'created','security_accepted':False})
    with pytest.raises(ValueError):write_json(path,{'bad':float('nan')})
    assert json.loads(path.read_text())=={'state':'created','security_accepted':False}
