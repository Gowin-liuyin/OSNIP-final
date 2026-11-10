"""Signed-cosine buckets must not merge negative cosines into [0, 0.1)."""
from scripts.diagnose_knn_coverage_buckets import bucket_of


def test_negative_cosines_get_their_own_bucket():
    assert bucket_of(-0.4) == "<0.00"
    assert bucket_of(-1e-6) == "<0.00"


def test_bucket_boundaries_are_half_open():
    assert bucket_of(0.0) == "0.00-0.10"
    assert bucket_of(0.1) == "0.10-0.20"
    assert bucket_of(0.25) == "0.20-0.30"
    assert bucket_of(0.3) == "0.30-0.40"
    assert bucket_of(0.99) == "0.80-1.01"
    assert bucket_of(1.0) == "0.80-1.01"
