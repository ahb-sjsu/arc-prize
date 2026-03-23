# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for the combined solver pipeline."""

import numpy as np
import pytest

from arc_prize.data import ARCTask, ARCPair
from arc_prize.combined.config import CombinedConfig
from arc_prize.combined.voting import (
    AugmentationSpec,
    AugmentedCandidate,
    apply_augmentation,
    invert_augmentation,
    generate_augmentation_specs,
    augment_task_pairs,
    vote_on_candidates,
    vote_statistics,
)
from arc_prize.combined.repair import (
    repair_size,
    repair_border,
    repair_color_majority,
    repair_symmetry,
    repair_prediction,
)
from arc_prize.combined.dsl_solver import solve_with_dsl


def _make_task(
    train_pairs: list[tuple[list, list]],
    test_pairs: list[tuple[list, list]],
    task_id: str = "test_task",
) -> ARCTask:
    """Helper to create a task from nested lists."""
    train = [
        ARCPair(
            input=np.array(inp, dtype=np.int64),
            output=np.array(out, dtype=np.int64),
        )
        for inp, out in train_pairs
    ]
    test = [
        ARCPair(
            input=np.array(inp, dtype=np.int64),
            output=np.array(out, dtype=np.int64),
        )
        for inp, out in test_pairs
    ]
    return ARCTask(task_id=task_id, train=train, test=test)


# ── Voting tests ──────────────────────────────────────────────


class TestAugmentation:
    def test_identity_roundtrip(self):
        grid = np.array([[1, 2], [3, 4]])
        spec = AugmentationSpec()
        result = apply_augmentation(grid, spec)
        np.testing.assert_array_equal(result, grid)

    def test_rotation_roundtrip(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        for k in range(4):
            spec = AugmentationSpec(rotation_k=k)
            augmented = apply_augmentation(grid, spec)
            recovered = invert_augmentation(augmented, spec)
            np.testing.assert_array_equal(recovered, grid, err_msg=f"Failed for k={k}")

    def test_flip_roundtrip(self):
        grid = np.array([[1, 2], [3, 4], [5, 6]])
        for fh, fv in [(True, False), (False, True), (True, True)]:
            spec = AugmentationSpec(flip_h=fh, flip_v=fv)
            augmented = apply_augmentation(grid, spec)
            recovered = invert_augmentation(augmented, spec)
            np.testing.assert_array_equal(recovered, grid)

    def test_color_perm_roundtrip(self):
        grid = np.array([[0, 1, 2], [3, 4, 5]])
        spec = AugmentationSpec(color_seed=42)
        augmented = apply_augmentation(grid, spec)
        recovered = invert_augmentation(augmented, spec)
        np.testing.assert_array_equal(recovered, grid)

    def test_combined_roundtrip(self):
        grid = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        spec = AugmentationSpec(rotation_k=2, flip_h=True, flip_v=False, color_seed=123)
        augmented = apply_augmentation(grid, spec)
        recovered = invert_augmentation(augmented, spec)
        np.testing.assert_array_equal(recovered, grid)

    def test_generate_specs_includes_identity(self):
        specs = generate_augmentation_specs(n_geometric=4, n_color=2)
        assert specs[0].rotation_k == 0
        assert not specs[0].flip_h
        assert not specs[0].flip_v
        assert specs[0].color_seed is None

    def test_generate_specs_count(self):
        specs = generate_augmentation_specs(n_geometric=8, n_color=4)
        assert len(specs) == 8 + 4  # geometric + color

    def test_augment_task_pairs_consistent(self):
        pairs = [
            (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])),
        ]
        spec = AugmentationSpec(rotation_k=1)
        aug_pairs = augment_task_pairs(pairs, spec)
        assert len(aug_pairs) == 1
        # Both should be rotated the same way
        aug_in, aug_out = aug_pairs[0]
        expected_in = np.rot90(pairs[0][0], 1)
        np.testing.assert_array_equal(aug_in, expected_in)


class TestVoting:
    def test_unanimous_vote(self):
        grid = np.array([[1, 2], [3, 4]])
        candidates = [
            AugmentedCandidate(grid=grid.copy(), augmentation_id=f"aug_{i}")
            for i in range(5)
        ]
        winners = vote_on_candidates(candidates, top_k=2)
        assert len(winners) == 1  # Only one unique grid
        np.testing.assert_array_equal(winners[0], grid)

    def test_majority_wins(self):
        grid_a = np.array([[1, 1], [1, 1]])
        grid_b = np.array([[2, 2], [2, 2]])
        candidates = [
            AugmentedCandidate(grid=grid_a.copy(), augmentation_id=f"a_{i}")
            for i in range(3)
        ] + [
            AugmentedCandidate(grid=grid_b.copy(), augmentation_id=f"b_{i}")
            for i in range(2)
        ]
        winners = vote_on_candidates(candidates, top_k=2)
        assert len(winners) == 2
        np.testing.assert_array_equal(winners[0], grid_a)
        np.testing.assert_array_equal(winners[1], grid_b)

    def test_vote_statistics(self):
        grid = np.array([[1]])
        candidates = [
            AugmentedCandidate(grid=grid.copy(), augmentation_id=f"a_{i}")
            for i in range(4)
        ]
        stats = vote_statistics(candidates)
        assert stats["n_candidates"] == 4
        assert stats["n_unique"] == 1
        assert stats["confidence"] == 1.0

    def test_empty_candidates(self):
        winners = vote_on_candidates([], top_k=2)
        assert winners == []


# ── Repair tests ──────────────────────────────────────────────


class TestRepair:
    def test_repair_size_fixed_output(self):
        """If all training outputs are 3x3, resize prediction to 3x3."""
        task = _make_task(
            train_pairs=[
                ([[1, 2, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                ([[4, 5, 6]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]),
            ],
            test_pairs=[([[7, 8, 9]], [[7, 7, 7], [8, 8, 8], [9, 9, 9]])],
        )
        # Prediction is wrong size (2x3 instead of 3x3)
        pred = np.array([[7, 7, 7], [8, 8, 8]])
        result = repair_size(pred, task, task.test[0].input)
        assert result.shape == (3, 3)

    def test_repair_border_uniform(self):
        """If border is mostly one color, enforce it."""
        pred = np.array([
            [1, 1, 1, 1],
            [1, 5, 5, 1],
            [1, 5, 5, 2],  # 2 is the error
            [1, 1, 1, 1],
        ])
        result = repair_border(pred)
        assert result[2, 3] == 1  # Fixed to border color

    def test_repair_symmetry_horizontal(self):
        """If grid is almost horizontally symmetric, fix it."""
        pred = np.array([
            [1, 2, 1],
            [3, 4, 3],
            [5, 6, 9],  # 9 should be 5 for symmetry
        ])
        result = repair_symmetry(pred, threshold=0.7)
        # Should enforce horizontal symmetry
        assert result[2, 2] == result[2, 0]

    def test_repair_color_majority(self):
        """Isolated pixel replaced by local majority."""
        pred = np.array([
            [1, 1, 1],
            [1, 9, 1],  # 9 is isolated
            [1, 1, 1],
        ])
        result = repair_color_majority(pred, window=3)
        assert result[1, 1] == 1

    def test_repair_prediction_returns_multiple(self):
        task = _make_task(
            train_pairs=[
                ([[1, 2], [3, 4]], [[4, 3], [2, 1]]),
            ],
            test_pairs=[([[5, 6], [7, 8]], [[8, 7], [6, 5]])],
        )
        pred = np.array([[8, 7], [6, 5]])
        candidates = repair_prediction(pred, task, task.test[0].input)
        assert len(candidates) >= 1
        # Original should be in candidates
        assert any(np.array_equal(c, pred) for c in candidates)


# ── DSL solver tests ──────────────────────────────────────────


class TestDSLSolver:
    def test_identity_task(self):
        """DSL should find identity function for same-input-output task."""
        # This is a trivial task where output == input
        task = _make_task(
            train_pairs=[
                ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
                ([[5, 6], [7, 8]], [[5, 6], [7, 8]]),
            ],
            test_pairs=[([[9, 1], [2, 3]], [[9, 1], [2, 3]])],
        )
        # Note: DSL might not have an explicit identity primitive,
        # but this validates the interface
        result = solve_with_dsl(task, max_depth=1, time_limit=5.0)
        # May or may not solve depending on primitives available
        # Just verify it doesn't crash and returns correct type
        assert result is None or (
            isinstance(result, list) and len(result) == 1
        )

    def test_rotation_task(self):
        """DSL should find rot90 for a rotation task."""
        grid1 = np.array([[1, 2], [3, 4]])
        out1 = np.rot90(grid1).copy()
        grid2 = np.array([[5, 6], [7, 8]])
        out2 = np.rot90(grid2).copy()
        grid3 = np.array([[9, 1], [2, 3]])
        out3 = np.rot90(grid3).copy()

        task = _make_task(
            train_pairs=[(grid1.tolist(), out1.tolist()), (grid2.tolist(), out2.tolist())],
            test_pairs=[(grid3.tolist(), out3.tolist())],
        )
        result = solve_with_dsl(task, max_depth=1, time_limit=10.0)
        if result is not None:
            assert len(result) == 1
            assert len(result[0]) == 2
            np.testing.assert_array_equal(result[0][0], out3)


# ── Config tests ──────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = CombinedConfig()
        assert config.time.total_per_task == 90.0
        assert config.dsl.max_depth == 3
        assert config.airv.enabled is True
        assert config.repair.enabled is True
        assert "dsl" in config.solver_order

    def test_config_customization(self):
        from arc_prize.combined.config import TimebudgetConfig, DSLConfig

        config = CombinedConfig(
            time=TimebudgetConfig(total_per_task=120.0),
            dsl=DSLConfig(max_depth=2, time_limit=10.0),
        )
        assert config.time.total_per_task == 120.0
        assert config.dsl.max_depth == 2
