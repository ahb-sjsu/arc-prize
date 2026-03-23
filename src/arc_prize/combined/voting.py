# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""AIRV-style voting — augmented inference with inverse transforms and
majority voting.  Core technique from MindsAI's 3rd-place solution.

The key insight: if you solve a task under multiple augmentations and
un-augment each answer back to the original frame, correct solutions
will agree while errors will scatter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from arc_prize.augment import permute_colors, rotate_grid, reflect_grid


@dataclass
class AugmentedCandidate:
    """A prediction produced under a specific augmentation."""

    grid: np.ndarray  # predicted output in ORIGINAL coordinate frame
    augmentation_id: str  # identifier for the augmentation used
    confidence: float = 1.0  # optional confidence weight


def _grid_hash(grid: np.ndarray) -> tuple:
    """Convert grid to hashable tuple for voting."""
    return tuple(map(tuple, grid.tolist()))


def _inverse_rotation(grid: np.ndarray, k: int) -> np.ndarray:
    """Inverse of np.rot90(grid, k) is np.rot90(grid, -k)."""
    return np.rot90(grid, -k).copy()


def _inverse_hflip(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()


def _inverse_vflip(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid).copy()


def _inverse_color_perm(grid: np.ndarray, seed: int) -> np.ndarray:
    """Reverse a color permutation by computing the inverse mapping."""
    rng = np.random.RandomState(seed)
    perm = list(range(10))
    non_bg = perm[1:]
    rng.shuffle(non_bg)
    perm[1:] = non_bg
    # Build inverse: inv_perm[perm[i]] = i
    inv_perm = [0] * 10
    for old, new in enumerate(perm):
        inv_perm[new] = old
    result = grid.copy()
    for new_val, old_val in enumerate(inv_perm):
        if new_val != old_val:
            result[grid == new_val] = old_val
    return result


@dataclass
class AugmentationSpec:
    """Specifies a single augmentation and how to invert it."""

    rotation_k: int = 0  # 0-3 quarter turns
    flip_h: bool = False
    flip_v: bool = False
    color_seed: int | None = None  # None = no color permutation

    @property
    def aug_id(self) -> str:
        parts = [f"r{self.rotation_k}"]
        if self.flip_h:
            parts.append("fh")
        if self.flip_v:
            parts.append("fv")
        if self.color_seed is not None:
            parts.append(f"c{self.color_seed}")
        return "_".join(parts)


def apply_augmentation(grid: np.ndarray, spec: AugmentationSpec) -> np.ndarray:
    """Apply an augmentation to a grid."""
    g = grid.copy()
    if spec.rotation_k:
        g = np.rot90(g, spec.rotation_k).copy()
    if spec.flip_h:
        g = np.fliplr(g).copy()
    if spec.flip_v:
        g = np.flipud(g).copy()
    if spec.color_seed is not None:
        g = permute_colors(g, seed=spec.color_seed)
    return g


def invert_augmentation(grid: np.ndarray, spec: AugmentationSpec) -> np.ndarray:
    """Invert an augmentation — undo transforms in reverse order."""
    g = grid.copy()
    # Undo in reverse order of application
    if spec.color_seed is not None:
        g = _inverse_color_perm(g, spec.color_seed)
    if spec.flip_v:
        g = _inverse_vflip(g)
    if spec.flip_h:
        g = _inverse_hflip(g)
    if spec.rotation_k:
        g = _inverse_rotation(g, spec.rotation_k)
    return g


def generate_augmentation_specs(
    n_geometric: int = 8,
    n_color: int = 4,
    seed: int = 42,
) -> list[AugmentationSpec]:
    """Generate a diverse set of augmentation specs for AIRV.

    Includes the identity (no augmentation) plus random combinations
    of rotations, flips, and color permutations.
    """
    rng = np.random.RandomState(seed)
    specs = [AugmentationSpec()]  # identity always included

    for i in range(n_geometric - 1):
        spec = AugmentationSpec(
            rotation_k=rng.randint(0, 4),
            flip_h=bool(rng.random() > 0.5),
            flip_v=bool(rng.random() > 0.5),
        )
        specs.append(spec)

    # Add color permutation variants
    for i in range(n_color):
        base = specs[i % len(specs)]
        color_spec = AugmentationSpec(
            rotation_k=base.rotation_k,
            flip_h=base.flip_h,
            flip_v=base.flip_v,
            color_seed=rng.randint(1, 2**31),
        )
        specs.append(color_spec)

    return specs


def augment_task_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    spec: AugmentationSpec,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply the same augmentation to all input-output pairs in a task."""
    return [
        (apply_augmentation(inp, spec), apply_augmentation(out, spec))
        for inp, out in pairs
    ]


def augment_inputs(
    inputs: list[np.ndarray],
    spec: AugmentationSpec,
) -> list[np.ndarray]:
    """Apply augmentation to test inputs."""
    return [apply_augmentation(inp, spec) for inp in inputs]


def vote_on_candidates(
    candidates: list[AugmentedCandidate],
    top_k: int = 2,
) -> list[np.ndarray]:
    """Majority voting across augmented candidates.

    Returns the top-k most-voted unique grids.

    This is the core AIRV mechanism: correct predictions cluster
    together across augmentations, while incorrect ones scatter.
    """
    if not candidates:
        return []

    # Count votes for each unique grid
    vote_counts: dict[tuple, float] = Counter()
    grid_lookup: dict[tuple, np.ndarray] = {}

    for cand in candidates:
        h = _grid_hash(cand.grid)
        vote_counts[h] += cand.confidence
        grid_lookup[h] = cand.grid

    # Sort by vote count (descending)
    ranked = sorted(vote_counts.items(), key=lambda x: -x[1])

    results = []
    for grid_hash, _count in ranked[:top_k]:
        results.append(grid_lookup[grid_hash])

    return results


def vote_statistics(candidates: list[AugmentedCandidate]) -> dict:
    """Compute voting statistics for confidence estimation."""
    if not candidates:
        return {"n_candidates": 0, "n_unique": 0, "top_votes": 0, "confidence": 0.0}

    vote_counts: dict[tuple, float] = Counter()
    for cand in candidates:
        vote_counts[_grid_hash(cand.grid)] += cand.confidence

    sorted_counts = sorted(vote_counts.values(), reverse=True)
    total = sum(sorted_counts)
    top = sorted_counts[0] if sorted_counts else 0

    return {
        "n_candidates": len(candidates),
        "n_unique": len(vote_counts),
        "top_votes": top,
        "confidence": top / max(total, 1),
        "margin": (sorted_counts[0] - sorted_counts[1]) / max(total, 1)
        if len(sorted_counts) > 1
        else 1.0,
    }
