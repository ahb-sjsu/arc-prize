# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Grid decoder: generates output grids from latent representations.

Given z_rule (the inferred transformation rule) and an input grid encoding,
produce the output grid cell-by-cell.  Uses a small CNN decoder with
cross-attention to the rule representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arc_prize.grid import NUM_COLORS, MAX_GRID_SIZE


class RuleConditioner(nn.Module):
    """Condition spatial features on the rule representation via FiLM.

    Feature-wise Linear Modulation: given rule z, compute per-channel
    scale γ and shift β, then apply γ * feat + β.
    """

    def __init__(self, z_dim: int, n_channels: int):
        super().__init__()
        self.gamma = nn.Linear(z_dim, n_channels)
        self.beta = nn.Linear(z_dim, n_channels)

    def forward(self, feat: torch.Tensor, z_rule: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W] spatial features
            z_rule: [B, z_dim] rule representation

        Returns:
            Conditioned features [B, C, H, W]
        """
        gamma = self.gamma(z_rule).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta(z_rule).unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class GridDecoder(nn.Module):
    """Decode latent rule + input encoding → output grid.

    Architecture:
    1. Broadcast z_rule to spatial dims
    2. Concatenate with input grid features
    3. CNN with FiLM conditioning at each layer
    4. Per-cell 10-way classification (10 ARC colors)
    """

    def __init__(self, z_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim

        # Input processing: re-encode the test input grid
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Combine input features + broadcasted rule
        self.combine = nn.Conv2d(64 + z_dim, hidden, 1)

        # Decoder layers with FiLM conditioning
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.conditioners = nn.ModuleList([RuleConditioner(z_dim, hidden) for _ in range(4)])

        # Output: per-cell color logits
        self.output = nn.Conv2d(hidden, NUM_COLORS, 1)

    def forward(
        self,
        z_rule: torch.Tensor,
        test_grid: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_rule:    [B, z_dim] — inferred transformation rule
            test_grid: [B, 10, 30, 30] — one-hot encoded test input
            test_mask: [B, 30, 30] — binary mask for real cells

        Returns:
            logits: [B, 10, 30, 30] — per-cell color logits
        """
        b = z_rule.shape[0]

        # Encode test input
        in_feat = self.input_conv(test_grid)  # [B, 64, 30, 30]

        # Broadcast rule to spatial dims
        z_spatial = (
            z_rule.unsqueeze(-1).unsqueeze(-1).expand(b, self.z_dim, MAX_GRID_SIZE, MAX_GRID_SIZE)
        )  # [B, z_dim, 30, 30]

        # Combine
        combined = torch.cat([in_feat, z_spatial], dim=1)  # [B, 64+z_dim, 30, 30]
        h = self.combine(combined)  # [B, hidden, 30, 30]

        # Decoder layers with FiLM conditioning
        for layer, conditioner in zip(self.layers, self.conditioners):
            h = layer(h)
            h = conditioner(h, z_rule)

        # Output logits
        logits = self.output(h)  # [B, 10, 30, 30]

        # Mask padding positions
        mask_expanded = test_mask.unsqueeze(1)  # [B, 1, 30, 30]
        logits = logits * mask_expanded

        return logits

    def predict_grid(
        self,
        z_rule: torch.Tensor,
        test_grid: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the output grid (argmax per cell).

        Returns: [B, 30, 30] integer grid.
        """
        logits = self.forward(z_rule, test_grid, test_mask)
        return logits.argmax(dim=1)  # [B, 30, 30]
