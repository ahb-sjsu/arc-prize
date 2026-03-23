# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import torch

from arc_prize.geometric import HyperbolicRuleEncoder, PoincareBall


class TestPoincareBall:
    def test_mobius_add_zero(self):
        """Adding zero should return the other point."""
        x = torch.tensor([[0.1, 0.2, 0.3]])
        zero = torch.zeros_like(x)
        result = PoincareBall.mobius_add(zero, x)
        assert torch.allclose(result, x, atol=1e-4)

    def test_distance_to_self(self):
        x = torch.tensor([[0.1, 0.2, 0.3]])
        d = PoincareBall.distance(x, x)
        assert d.item() < 0.01

    def test_distance_symmetric(self):
        x = torch.tensor([[0.1, 0.2]])
        y = torch.tensor([[0.3, -0.1]])
        d_xy = PoincareBall.distance(x, y)
        d_yx = PoincareBall.distance(y, x)
        assert torch.allclose(d_xy, d_yx, atol=1e-4)

    def test_project_inside_ball(self):
        """Points inside the ball should not change."""
        x = torch.tensor([[0.1, 0.2]])
        p = PoincareBall.project(x)
        assert torch.allclose(x, p, atol=1e-6)

    def test_project_outside_ball(self):
        """Points outside should be projected to boundary."""
        x = torch.tensor([[5.0, 5.0]])
        p = PoincareBall.project(x)
        assert p.norm() <= 0.96  # max_norm=0.95


class TestHyperbolicRuleEncoder:
    def test_forward_shape(self):
        enc = HyperbolicRuleEncoder(z_dim=64, hyp_dim=16)
        z = torch.randn(4, 64)
        h = enc(z)
        assert h.shape == (4, 16)

    def test_points_inside_ball(self):
        enc = HyperbolicRuleEncoder(z_dim=64, hyp_dim=16)
        z = torch.randn(10, 64)
        h = enc(z)
        norms = h.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_rule_similarity(self):
        enc = HyperbolicRuleEncoder(z_dim=64, hyp_dim=16)
        z1 = torch.randn(1, 64)
        z2 = torch.randn(1, 64)
        h1, h2 = enc(z1), enc(z2)
        sim = enc.rule_similarity(h1, h2)
        assert 0.0 <= sim.item() <= 1.0

    def test_rule_depth(self):
        enc = HyperbolicRuleEncoder(z_dim=64, hyp_dim=16)
        z = torch.randn(3, 64)
        h = enc(z)
        depth = enc.rule_depth(h)
        assert depth.shape == (3,)
        assert (depth >= 0).all()
