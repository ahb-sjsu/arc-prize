# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Evaluation metrics for ARC-AGI-2.

ARC scoring is strict: exact grid match only. No partial credit.
Each task allows 2 attempts — if either matches, the task is scored correct.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def exact_match(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Check if two grids are exactly identical."""
    if predicted.shape != target.shape:
        return False
    return bool(np.array_equal(predicted, target))


def score_task(
    predictions: List[List[np.ndarray]],
    targets: List[np.ndarray],
) -> bool:
    """Score a single task.  Each test input gets 2 attempts.

    Args:
        predictions: list of [candidate_1, candidate_2] per test input
        targets: list of ground truth output grids

    Returns:
        True if ALL test inputs have at least one correct candidate.
    """
    for preds, target in zip(predictions, targets):
        if not any(exact_match(p, target) for p in preds):
            return False
    return True


def score_submission(
    all_predictions: Dict[str, List[List[np.ndarray]]],
    all_targets: Dict[str, List[np.ndarray]],
) -> Tuple[int, int, float]:
    """Score a full submission.

    Args:
        all_predictions: {task_id: [[cand1, cand2], ...]} per test input
        all_targets: {task_id: [target_grid, ...]}

    Returns:
        (n_correct, n_total, accuracy)
    """
    n_correct = 0
    n_total = 0

    for task_id, targets in all_targets.items():
        n_total += 1
        if task_id in all_predictions:
            if score_task(all_predictions[task_id], targets):
                n_correct += 1

    accuracy = n_correct / max(n_total, 1)
    return n_correct, n_total, accuracy


def cell_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    """Per-cell accuracy (useful for training monitoring, not official scoring).

    Handles grids of different sizes by comparing the overlapping region.
    """
    h = min(predicted.shape[0], target.shape[0])
    w = min(predicted.shape[1], target.shape[1])
    if h == 0 or w == 0:
        return 0.0
    overlap_pred = predicted[:h, :w]
    overlap_tgt = target[:h, :w]
    correct = np.sum(overlap_pred == overlap_tgt)

    # Penalize size mismatch
    total = target.shape[0] * target.shape[1]
    return float(correct / max(total, 1))
