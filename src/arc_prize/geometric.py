# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Hyperbolic geometry for hierarchical rule representation.

ARC rules are hierarchical — "tile the pattern" contains sub-rules like
"repeat", "mirror", "offset". The Poincaré ball naturally represents this
hierarchy: general rules near the origin, specific sub-rules near the boundary.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoincareBall:
    """Operations in the Poincaré ball model of hyperbolic space."""

    EPS = 1e-5

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Möbius addition in the Poincaré ball with curvature c."""
        x_sq = (x * x).sum(dim=-1, keepdim=True).clamp(max=1 - PoincareBall.EPS)
        y_sq = (y * y).sum(dim=-1, keepdim=True).clamp(max=1 - PoincareBall.EPS)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c**2 * x_sq * y_sq
        return num / denom.clamp(min=PoincareBall.EPS)

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Poincaré distance between points."""
        diff = PoincareBall.mobius_add(-x, y, c)
        norm = diff.norm(dim=-1).clamp(min=PoincareBall.EPS, max=1 - PoincareBall.EPS)
        return (2.0 / (c**0.5)) * torch.atanh(c**0.5 * norm)

    @staticmethod
    def exp_map(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Exponential map: move from x in direction v on the manifold."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=PoincareBall.EPS)
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp(max=1 - PoincareBall.EPS)
        coeff = torch.tanh(c**0.5 * (1 - c * x_norm_sq) * v_norm / 2)
        return PoincareBall.mobius_add(x, coeff * v / v_norm, c)

    @staticmethod
    def project(x: torch.Tensor, c: float = 1.0, max_norm: float = 0.95) -> torch.Tensor:
        """Project back into the ball if norms exceed boundary."""
        norm = x.norm(dim=-1, keepdim=True)
        max_r = max_norm / (c**0.5)
        return torch.where(norm > max_r, x * max_r / norm, x)


class HyperbolicRuleEncoder(nn.Module):
    """Map z-space rule representations into the Poincaré ball.

    General/abstract rules cluster near the origin.
    Specific/concrete sub-rules live near the boundary.
    Hierarchy emerges naturally from hyperbolic distance.
    """

    def __init__(self, z_dim: int = 128, hyp_dim: int = 32, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
        self.proj = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, hyp_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map from Euclidean z to Poincaré ball."""
        h = self.proj(z)
        # Project into the open ball
        return PoincareBall.project(h, self.curvature)

    def rule_similarity(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Similarity between rules in hyperbolic space (higher = more similar)."""
        dist = PoincareBall.distance(h1, h2, self.curvature)
        return torch.exp(-dist)

    def rule_depth(self, h: torch.Tensor) -> torch.Tensor:
        """How specific/concrete a rule is (distance from origin)."""
        return h.norm(dim=-1)
