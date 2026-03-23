# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Grid augmentation for ARC training.

ARC tasks are invariant to certain symmetry transforms (rotations,
reflections, color permutations).  Augmenting the training set with
these transforms is essentially free data.
"""

from __future__ import annotations

import numpy as np


def rotate_grid(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate grid by k * 90 degrees counter-clockwise."""
    return np.rot90(grid, k)


def reflect_grid(grid: np.ndarray, axis: int = 0) -> np.ndarray:
    """Reflect grid.  axis=0 → vertical flip, axis=1 → horizontal flip."""
    if axis == 0:
        return np.flipud(grid).copy()
    return np.fliplr(grid).copy()


def permute_colors(grid: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomly permute non-background colors (1-9)."""
    rng = np.random.RandomState(seed)
    perm = list(range(10))
    non_bg = perm[1:]
    rng.shuffle(non_bg)
    perm[1:] = non_bg
    result = grid.copy()
    for old, new in enumerate(perm):
        if old != new:
            result[grid == old] = new
    return result


def all_dihedral(grid: np.ndarray) -> list[np.ndarray]:
    """Generate all 8 dihedral group transforms of a grid.

    Returns [identity, rot90, rot180, rot270, flip_h, flip_v, diag1, diag2].
    """
    results = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        results.append(rotated.copy())
    flipped = np.fliplr(grid)
    for k in range(4):
        results.append(np.rot90(flipped, k).copy())
    return results


def augment_pair(
    in_grid: np.ndarray,
    out_grid: np.ndarray,
    *,
    n_augments: int = 4,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Augment an input-output pair with consistent transforms.

    Applies the SAME transform to both input and output so the
    transformation rule is preserved.

    Returns list of (augmented_input, augmented_output) pairs.
    """
    rng = np.random.RandomState(seed)
    pairs = []

    for _ in range(n_augments):
        # Random rotation
        k = rng.randint(0, 4)
        aug_in = np.rot90(in_grid, k).copy()
        aug_out = np.rot90(out_grid, k).copy()

        # Random reflection
        if rng.random() > 0.5:
            aug_in = np.fliplr(aug_in).copy()
            aug_out = np.fliplr(aug_out).copy()
        if rng.random() > 0.5:
            aug_in = np.flipud(aug_in).copy()
            aug_out = np.flipud(aug_out).copy()

        # Color permutation (same mapping for both)
        color_seed = rng.randint(0, 2**31)
        aug_in = permute_colors(aug_in, seed=color_seed)
        aug_out = permute_colors(aug_out, seed=color_seed)

        pairs.append((aug_in, aug_out))

    return pairs
