# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""DSL solver wrapper — adapts the dsl_v2 search engine to the
combined pipeline interface.

DSL is the fastest solver and most reliable when it works (exact match).
Always run first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add repo root so dsl_v2 is importable
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dsl_v2 import ARCTask as DSLTask, ARCPair as DSLPair, search_programs, apply_program
from arc_prize.data import ARCTask, ARCPair


def _to_dsl_task(task: ARCTask) -> DSLTask:
    """Convert arc_prize.data.ARCTask to dsl_v2.ARCTask."""
    train = [DSLPair(p.input, p.output) for p in task.train]
    test = [DSLPair(p.input, p.output) for p in task.test]
    return DSLTask(task.task_id, train, test)


def solve_with_dsl(
    task: ARCTask,
    max_depth: int = 3,
    time_limit: float = 20.0,
) -> list[list[np.ndarray]] | None:
    """Try to solve a task using DSL program search.

    Returns:
        List of [candidate_1, candidate_2] per test input if solved,
        or None if no program found.
    """
    dsl_task = _to_dsl_task(task)
    result = search_programs(dsl_task, max_depth=max_depth, time_limit=time_limit)

    if result is None:
        return None

    names, fns = result
    predictions = []

    for test_pair in task.test:
        output = apply_program(test_pair.input, fns)
        if output is None:
            return None  # Program failed on test input
        # DSL gives exact answer — use same prediction for both attempts
        predictions.append([output, output])

    return predictions


def solve_batch_dsl(
    tasks: list[ARCTask],
    max_depth: int = 3,
    time_per_task: float = 20.0,
) -> dict[str, list[list[np.ndarray]]]:
    """Solve a batch of tasks with DSL, return dict of solved tasks.

    Returns:
        {task_id: [[cand1, cand2], ...]} for successfully solved tasks.
    """
    solved = {}

    for task in tasks:
        t0 = time.time()
        result = solve_with_dsl(task, max_depth=max_depth, time_limit=time_per_task)
        elapsed = time.time() - t0

        if result is not None:
            solved[task.task_id] = result

    return solved
