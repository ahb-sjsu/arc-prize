# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Adversarial training components for ARC solver.

Gradient reversal forces the encoder to learn representations that are
INVARIANT to surface-level features (color palette, position) while
remaining SENSITIVE to structural features (pattern, symmetry, count).

This is the neural complement to the probe-based fuzzing in fuzzer.py:
- fuzzer.py: external probing → discovers what the model has learned
- adversarial.py: internal training signal → shapes what the model learns
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Reverse gradients during backward pass.

    Forward:  identity
    Backward: negate and scale gradients by λ

    This forces the encoder to produce features that *cannot* be used
    by the adversary to predict the surface property — i.e., the encoder
    learns to be invariant to whatever the adversary is trying to detect.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scaling factor λ."""
    return GradientReversal.apply(x, lambd)


class AdversarialHead(nn.Module):
    """Adversarial classifier head with gradient reversal.

    Tries to predict a surface property (e.g., dominant color, grid size)
    from the latent z.  Gradient reversal ensures the encoder learns to
    make this prediction impossible → invariance to that property.
    """

    def __init__(self, z_dim: int, n_classes: int, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Classify the reversed-gradient representation."""
        z_rev = grad_reverse(z, self.lambd)
        return self.classifier(z_rev)


class MultiHeadAdversary(nn.Module):
    """Multiple adversarial heads for different surface properties.

    Each head forces invariance to a different surface feature:
    - color_dist: dominant color distribution class
    - grid_size:  binned grid dimensions
    - density:    fraction of non-background cells
    - position:   spatial centroid quadrant of objects

    The combined adversarial loss shapes the encoder to focus on
    abstract structural patterns, not surface statistics.
    """

    def __init__(self, z_dim: int = 128, lambd: float = 1.0):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                # 10 colors
                "color_dist": AdversarialHead(z_dim, 10, lambd),
                # Binned into 6 size categories
                "grid_size": AdversarialHead(z_dim, 6, lambd),
                # Binary: sparse vs dense
                "density": AdversarialHead(z_dim, 2, lambd),
                # 4 quadrants
                "position": AdversarialHead(z_dim, 4, lambd),
            }
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run all adversarial heads. Returns dict of logits."""
        return {name: head(z) for name, head in self.heads.items()}

    def compute_loss(
        self,
        z: torch.Tensor,
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Total adversarial loss across all heads.

        The encoder's gradient from this loss is REVERSED — it learns
        to make these properties unpredictable from z.
        """
        logits = self.forward(z)
        total = torch.tensor(0.0, device=z.device)
        for name, pred in logits.items():
            if name in labels:
                total = total + nn.functional.cross_entropy(pred, labels[name])
        return total


# ---------------------------------------------------------------------------
# Helpers to extract adversarial labels from grids
# ---------------------------------------------------------------------------


def extract_adversarial_labels(
    grid_np,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract surface property labels from a raw grid for adversarial training.

    Args:
        grid_np: numpy array [H, W] with integer colors 0-9
        device: torch device

    Returns:
        Dict mapping head name → label tensor [1]
    """
    import numpy as np

    h, w = grid_np.shape

    # Dominant non-background color
    flat = grid_np.flatten()
    non_bg = flat[flat > 0]
    if len(non_bg) > 0:
        color_label = int(np.bincount(non_bg).argmax())
    else:
        color_label = 0

    # Grid size bin: <5, <10, <15, <20, <25, >=25
    max_dim = max(h, w)
    if max_dim < 5:
        size_label = 0
    elif max_dim < 10:
        size_label = 1
    elif max_dim < 15:
        size_label = 2
    elif max_dim < 20:
        size_label = 3
    elif max_dim < 25:
        size_label = 4
    else:
        size_label = 5

    # Density: sparse (< 30% non-background) vs dense
    density = np.mean(grid_np > 0)
    density_label = 0 if density < 0.3 else 1

    # Position: centroid quadrant of non-background cells
    rows, cols = np.where(grid_np > 0)
    if len(rows) > 0:
        cr, cc = rows.mean() / h, cols.mean() / w
        position_label = int(cr >= 0.5) * 2 + int(cc >= 0.5)
    else:
        position_label = 0

    return {
        "color_dist": torch.tensor([color_label], device=device),
        "grid_size": torch.tensor([size_label], device=device),
        "density": torch.tensor([density_label], device=device),
        "position": torch.tensor([position_label], device=device),
    }
