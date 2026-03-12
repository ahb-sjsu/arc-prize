# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import torch

from arc_prize.decoder import GridDecoder, RuleConditioner


class TestRuleConditioner:
    def test_output_shape(self):
        cond = RuleConditioner(z_dim=64, n_channels=128)
        feat = torch.randn(2, 128, 10, 10)
        z = torch.randn(2, 64)
        out = cond(feat, z)
        assert out.shape == (2, 128, 10, 10)


class TestGridDecoder:
    def test_output_shape(self):
        dec = GridDecoder(z_dim=64, hidden=128)
        z_rule = torch.randn(2, 64)
        grid = torch.randn(2, 10, 30, 30)
        mask = torch.ones(2, 30, 30)
        logits = dec(z_rule, grid, mask)
        assert logits.shape == (2, 10, 30, 30)

    def test_predict_grid(self):
        dec = GridDecoder(z_dim=64, hidden=128)
        z_rule = torch.randn(1, 64)
        grid = torch.randn(1, 10, 30, 30)
        mask = torch.ones(1, 30, 30)
        pred = dec.predict_grid(z_rule, grid, mask)
        assert pred.shape == (1, 30, 30)
        assert pred.min() >= 0
        assert pred.max() <= 9

    def test_mask_zeros_padding(self):
        dec = GridDecoder(z_dim=64, hidden=128)
        z_rule = torch.randn(1, 64)
        grid = torch.randn(1, 10, 30, 30)
        mask = torch.zeros(1, 30, 30)
        mask[:, :5, :5] = 1.0
        logits = dec(z_rule, grid, mask)
        # Padding region should be zero
        assert logits[:, :, 10:, 10:].abs().sum() == 0
