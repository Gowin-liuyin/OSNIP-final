"""The four targets build different observations and refuse a missing OSNIP checkpoint."""
from __future__ import annotations

import unittest

import torch

from attack_ready.observe import observe, require_checkpoint
from osnip_repro.encoder import build_encryptor


class FourTargetTests(unittest.TestCase):
    def test_osnip_and_ab_refuse_without_a_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            require_checkpoint("osnip", None)
        with self.assertRaises(ValueError):
            require_checkpoint("a+b", None)
        require_checkpoint("plaintext", None)
        require_checkpoint("aloepri", None)

    def test_plaintext_is_the_table_row_and_osnip_is_not(self) -> None:
        generator = torch.Generator().manual_seed(0)
        table = torch.randn(12, 8, generator=generator)
        ids = torch.tensor([1, 4, 7])
        plain, _ = observe(table, ids, "plaintext")
        self.assertTrue(torch.equal(plain, table[ids]))
        encoder = build_encryptor(8, key_dim=4, hidden_dim=8).eval()
        encrypted, info = observe(table, ids, "osnip", encoder=encoder, key_dim=4, seed=1)
        self.assertEqual(encrypted.shape, plain.shape)
        self.assertFalse(torch.allclose(encrypted, plain))
        self.assertTrue(info["osnip"])
        composed, composed_info = observe(table, ids, "a+b", encoder=encoder, key_dim=4, seed=1)
        self.assertIn("composition", composed_info)
        self.assertFalse(torch.allclose(composed, encrypted))
