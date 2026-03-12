# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Grid encoder: maps ARC grids to latent z-space.

Uses a small CNN + attention pooling. The encoder must handle variable-size
grids (1x1 to 30x30) padded to a fixed size with masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_prize.grid import NUM_COLORS


class GridEncoder(nn.Module):
    """Encode a padded ARC grid [10, 30, 30] + mask → z [z_dim]."""

    def __init__(self, z_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim

        # CNN backbone — small but deep enough for 30x30 grids
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )

        # Attention pooling over spatial dims (handles variable sizes via mask)
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden))
        self.attn_proj = nn.Linear(hidden, 1)

        self.proj = nn.Sequential(
            nn.Linear(hidden, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, 10, 30, 30] one-hot encoded grid
            mask: [B, 30, 30] binary mask (1 = real cell, 0 = padding)

        Returns:
            z: [B, z_dim] latent representation
        """
        # CNN features: [B, hidden, 30, 30]
        feat = self.conv(grid)

        # Reshape for attention: [B, 900, hidden]
        b, c, h, w = feat.shape
        feat_flat = feat.view(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]

        # Attention scores with masking
        scores = self.attn_proj(feat_flat).squeeze(-1)  # [B, HW]
        mask_flat = mask.view(b, h * w)  # [B, HW]
        scores = scores.masked_fill(mask_flat == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, HW, 1]

        # Weighted sum
        pooled = (feat_flat * weights).sum(dim=1)  # [B, C]

        return self.proj(pooled)


class PairEncoder(nn.Module):
    """Encode an input-output pair → z_pair.

    Concatenates input and output encodings with a learned difference signal.
    This captures the *transformation* between input and output.
    """

    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.grid_enc = GridEncoder(z_dim=z_dim)
        self.pair_proj = nn.Sequential(
            nn.Linear(z_dim * 3, z_dim),
            nn.LayerNorm(z_dim),
            nn.GELU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(
        self,
        in_grid: torch.Tensor,
        in_mask: torch.Tensor,
        out_grid: torch.Tensor,
        out_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a (input, output) pair → z_pair [B, z_dim]."""
        z_in = self.grid_enc(in_grid, in_mask)
        z_out = self.grid_enc(out_grid, out_mask)
        z_diff = z_out - z_in  # Transformation signal
        return self.pair_proj(torch.cat([z_in, z_out, z_diff], dim=-1))
