# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np

from arc_prize.fuzzer import (
    TransformChain,
    _add_noise,
    _color_permute,
    _rotate,
    make_transform_suite,
)


class TestParametricTransform:
    def test_zero_intensity_noop(self):
        grid = np.array([[1, 2], [3, 4]])
        result = _color_permute(grid, 0.0)
        np.testing.assert_array_equal(grid, result)

    def test_rotation(self):
        grid = np.array([[1, 2], [3, 4]])
        rot90 = _rotate(grid, 0.33)  # ~1 quarter turn
        assert rot90.shape[0] == 2 and rot90.shape[1] == 2
        # 1 quarter turn of [[1,2],[3,4]] → [[2,4],[1,3]]
        expected = np.rot90(grid, 1)
        np.testing.assert_array_equal(rot90, expected)

    def test_noise_at_zero(self):
        grid = np.array([[1, 2], [3, 4]])
        result = _add_noise(grid, 0.0)
        np.testing.assert_array_equal(grid, result)

    def test_noise_changes_grid(self):
        grid = np.ones((10, 10), dtype=int) * 5
        noisy = _add_noise(grid, 1.0)
        assert not np.array_equal(grid, noisy)


class TestTransformChain:
    def test_chain_applies_all(self):
        suite = make_transform_suite()
        chain = TransformChain(
            [
                (suite[0], 1.0),  # color_permute
                (suite[1], 0.33),  # rotate
            ]
        )
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        result = chain(grid)
        assert isinstance(result, np.ndarray)

    def test_random_chains(self):
        suite = make_transform_suite()
        chains = TransformChain.random_chains(suite, n_chains=5)
        assert len(chains) == 5


class TestTransformSuite:
    def test_suite_has_both_types(self):
        suite = make_transform_suite()
        invariants = [t for t in suite if t.is_structural_invariant]
        stresses = [t for t in suite if not t.is_structural_invariant]
        assert len(invariants) >= 3
        assert len(stresses) >= 2

    def test_all_transforms_produce_valid_grids(self):
        suite = make_transform_suite()
        grid = np.random.randint(0, 10, (5, 5))
        for transform in suite:
            result = transform(grid, 0.5)
            assert isinstance(result, np.ndarray)
            assert result.min() >= 0
            assert result.max() <= 9
