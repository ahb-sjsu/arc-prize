# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np

from arc_prize.evaluate import cell_accuracy, exact_match, score_task


class TestExactMatch:
    def test_identical(self):
        a = np.array([[1, 2], [3, 4]])
        assert exact_match(a, a.copy())

    def test_different(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 5]])
        assert not exact_match(a, b)

    def test_different_shape(self):
        a = np.array([[1, 2]])
        b = np.array([[1], [2]])
        assert not exact_match(a, b)


class TestScoreTask:
    def test_correct_first_attempt(self):
        target = np.array([[1, 2]])
        preds = [[np.array([[1, 2]]), np.array([[0, 0]])]]
        assert score_task(preds, [target])

    def test_correct_second_attempt(self):
        target = np.array([[1, 2]])
        preds = [[np.array([[0, 0]]), np.array([[1, 2]])]]
        assert score_task(preds, [target])

    def test_both_wrong(self):
        target = np.array([[1, 2]])
        preds = [[np.array([[0, 0]]), np.array([[9, 9]])]]
        assert not score_task(preds, [target])


class TestCellAccuracy:
    def test_perfect(self):
        a = np.array([[1, 2], [3, 4]])
        assert cell_accuracy(a, a) == 1.0

    def test_half_correct(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [0, 0]])
        assert cell_accuracy(a, b) == 0.5
