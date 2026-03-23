# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Generate ARC-AGI-2 Kaggle submission file.

Format: JSON with {task_id: [{"attempt_1": grid, "attempt_2": grid}]}
where each grid is a list of lists of integers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def format_grid(grid: np.ndarray) -> list[list[int]]:
    """Convert numpy grid to JSON-serializable list of lists."""
    return grid.astype(int).tolist()


def make_submission(
    predictions: dict[str, list[list[np.ndarray]]],
    output_path: str | Path = "submission.json",
) -> Path:
    """Create a Kaggle submission JSON file.

    Args:
        predictions: {task_id: [[attempt1, attempt2], ...]}
            Each task can have multiple test inputs.
            Each test input has 2 candidate grids.
        output_path: Where to write the JSON.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    submission = {}

    for task_id, test_preds in predictions.items():
        task_attempts = []
        for candidates in test_preds:
            entry = {
                "attempt_1": format_grid(candidates[0]),
                "attempt_2": format_grid(candidates[1])
                if len(candidates) > 1
                else format_grid(candidates[0]),
            }
            task_attempts.append(entry)
        submission[task_id] = task_attempts

    with open(output_path, "w") as f:
        json.dump(submission, f)

    return output_path
