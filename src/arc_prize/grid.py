# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Grid encoding and manipulation for ARC-AGI tasks.

ARC grids are 2D arrays of integers 0-9, max 30x30. We represent them
as tensors with one-hot color channels for the neural encoder.
"""

from __future__ import annotations

import numpy as np
import torch


# ARC uses 10 colors (0-9)
NUM_COLORS = 10
MAX_GRID_SIZE = 30


def grid_to_tensor(grid: list[list[int]] | np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert an ARC grid to a one-hot tensor [10, H, W]."""
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    one_hot = np.zeros((NUM_COLORS, h, w), dtype=np.float32)
    for c in range(NUM_COLORS):
        one_hot[c] = (arr == c).astype(np.float32)
    return torch.from_numpy(one_hot).to(device)


def tensor_to_grid(tensor: torch.Tensor) -> list[list[int]]:
    """Convert a [10, H, W] tensor back to a grid (argmax per cell)."""
    arr = tensor.detach().cpu().numpy()
    grid = np.argmax(arr, axis=0).astype(int)
    return grid.tolist()


def pad_grid(tensor: torch.Tensor, size: int = MAX_GRID_SIZE) -> torch.Tensor:
    """Pad a [10, H, W] tensor to [10, size, size] with zeros."""
    c, h, w = tensor.shape
    if h >= size and w >= size:
        return tensor[:, :size, :size]
    padded = torch.zeros(c, size, size, dtype=tensor.dtype, device=tensor.device)
    padded[:, :h, :w] = tensor
    return padded


def grid_size_mask(h: int, w: int, size: int = MAX_GRID_SIZE, device: str = "cpu") -> torch.Tensor:
    """Create a [size, size] mask that is 1 inside the actual grid, 0 in padding."""
    mask = torch.zeros(size, size, device=device)
    mask[:h, :w] = 1.0
    return mask


def extract_objects(grid: np.ndarray, background: int = 0) -> list[dict]:
    """Extract connected components (objects) from a grid.

    Returns list of dicts with keys: color, cells, bbox, size.
    Uses flood-fill to find connected regions of the same color.
    """
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == background:
                continue
            # BFS flood fill
            color = grid[r, c]
            cells = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= h or cc < 0 or cc >= w:
                    continue
                if visited[cr, cc] or grid[cr, cc] != color:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                stack.extend([(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)])

            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            objects.append(
                {
                    "color": int(color),
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "size": len(cells),
                }
            )

    return objects
