# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np
import torch

from arc_prize.grid import (
    extract_objects,
    grid_size_mask,
    grid_to_tensor,
    pad_grid,
    tensor_to_grid,
)


class TestGridToTensor:
    def test_basic(self):
        grid = [[0, 1], [2, 3]]
        t = grid_to_tensor(grid)
        assert t.shape == (10, 2, 2)
        assert t[0, 0, 0] == 1.0  # color 0 at (0,0)
        assert t[1, 0, 1] == 1.0  # color 1 at (0,1)
        assert t[2, 1, 0] == 1.0  # color 2 at (1,0)

    def test_roundtrip(self):
        grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        t = grid_to_tensor(grid)
        recovered = tensor_to_grid(t)
        assert recovered == grid

    def test_numpy_input(self):
        arr = np.array([[1, 2], [3, 4]])
        t = grid_to_tensor(arr)
        assert t.shape == (10, 2, 2)


class TestPadGrid:
    def test_small_grid(self):
        t = torch.zeros(10, 5, 5)
        padded = pad_grid(t)
        assert padded.shape == (10, 30, 30)
        assert padded[:, :5, :5].sum() == 0  # original region
        assert padded[:, 5:, :].sum() == 0  # padding

    def test_already_max_size(self):
        t = torch.ones(10, 30, 30)
        padded = pad_grid(t)
        assert padded.shape == (10, 30, 30)
        assert torch.equal(padded, t)


class TestGridSizeMask:
    def test_mask_shape(self):
        mask = grid_size_mask(5, 10)
        assert mask.shape == (30, 30)
        assert mask[:5, :10].sum() == 50
        assert mask[5:, :].sum() == 0
        assert mask[:, 10:].sum() == 0


class TestExtractObjects:
    def test_single_object(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[1:3, 1:3] = 1
        objects = extract_objects(grid)
        assert len(objects) == 1
        assert objects[0]["color"] == 1
        assert objects[0]["size"] == 4

    def test_two_colors(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[0, 0] = 1
        grid[4, 4] = 2
        objects = extract_objects(grid)
        assert len(objects) == 2

    def test_empty_grid(self):
        grid = np.zeros((3, 3), dtype=int)
        objects = extract_objects(grid)
        assert len(objects) == 0
