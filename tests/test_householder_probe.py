import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('house',Path(__file__).resolve().parents[1]/'scripts/run_householder_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_householder_preserves_norm_and_is_key_conditioned():
 torch.manual_seed(1);e=m.HouseholderKeyEncryptor(16,4,8);h=torch.randn(2,5,16);k1=torch.randn(2,4);k2=torch.randn(2,4);z1=e(h,k1);z2=e(h,k2)
 assert torch.allclose(z1.norm(dim=-1),h.norm(dim=-1),atol=2e-5)
 assert not torch.allclose(z1,z2)
