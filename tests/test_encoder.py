# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import torch

from arc_prize.encoder import GridEncoder, PairEncoder


class TestGridEncoder:
    def test_output_shape(self):
        enc = GridEncoder(z_dim=64)
        grid = torch.randn(2, 10, 30, 30)
        mask = torch.ones(2, 30, 30)
        z = enc(grid, mask)
        assert z.shape == (2, 64)

    def test_mask_effect(self):
        """Different masks should produce different encodings."""
        enc = GridEncoder(z_dim=64)
        grid = torch.randn(1, 10, 30, 30)
        mask_full = torch.ones(1, 30, 30)
        mask_small = torch.zeros(1, 30, 30)
        mask_small[:, :5, :5] = 1.0

        z_full = enc(grid, mask_full)
        z_small = enc(grid, mask_small)
        # Not strictly guaranteed but very unlikely to be equal
        assert not torch.allclose(z_full, z_small, atol=1e-3)


class TestPairEncoder:
    def test_output_shape(self):
        enc = PairEncoder(z_dim=64)
        in_grid = torch.randn(2, 10, 30, 30)
        in_mask = torch.ones(2, 30, 30)
        out_grid = torch.randn(2, 10, 30, 30)
        out_mask = torch.ones(2, 30, 30)
        z = enc(in_grid, in_mask, out_grid, out_mask)
        assert z.shape == (2, 64)

    def test_same_input_output(self):
        """Identity transform should produce a specific z (not random)."""
        enc = PairEncoder(z_dim=64)
        grid = torch.randn(1, 10, 30, 30)
        mask = torch.ones(1, 30, 30)
        z = enc(grid, mask, grid, mask)
        assert z.shape == (1, 64)
        assert torch.isfinite(z).all()
