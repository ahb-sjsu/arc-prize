# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Combined orchestrator — runs DSL, neural AIRV, and repair in sequence
with time-budget management.

Architecture (inspired by NVARC 1st place + MindsAI 3rd place):

    ┌─────────────────────────────────────────────────────────────┐
    │                   Combined Orchestrator                      │
    │                                                              │
    │  ┌──────────┐   ┌───────────────┐   ┌──────────────────┐   │
    │  │   DSL    │──▶│  Neural + AIRV │──▶│  Near-Miss Repair│   │
    │  │ (exact)  │   │  (TTT + vote)  │   │   (post-proc)   │   │
    │  └──────────┘   └───────────────┘   └──────────────────┘   │
    │       │                │                      │              │
    │       ▼                ▼                      ▼              │
    │  ┌─────────────────────────────────────────────────┐        │
    │  │            Final Ensemble / Selector             │        │
    │  │  (pick best 2 candidates per test input)        │        │
    │  └─────────────────────────────────────────────────┘        │
    └─────────────────────────────────────────────────────────────┘

Time budget per task (default 90s on Kaggle T4):
  - DSL:    20s  (fast search, exact matches)
  - Neural: 45s  (TTT + AIRV voting)
  - Repair: 15s  (fix near-misses)
  - Buffer: 10s  (overhead)
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from arc_prize.data import ARCTask
from arc_prize.evaluate import cell_accuracy, exact_match
from arc_prize.solver import ARCSolver, SolverConfig
from arc_prize.combined.config import CombinedConfig
from arc_prize.combined.dsl_solver import solve_with_dsl
from arc_prize.combined.neural_solver import solve_with_neural_airv
from arc_prize.combined.repair import repair_prediction, select_best_repair
from arc_prize.combined.voting import vote_statistics

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from solving a single task."""

    task_id: str
    predictions: list[list[np.ndarray]]  # [test_idx][candidate_idx] = grid
    solver_used: str  # which solver produced the winning candidates
    time_elapsed: float
    confidence: float = 0.0
    stats: dict = field(default_factory=dict)


class CombinedOrchestrator:
    """Orchestrates multiple solvers with time-budget management.

    Usage:
        config = CombinedConfig()
        orch = CombinedOrchestrator(config)
        orch.load_neural_model(checkpoint_path)

        results = orch.solve_all(tasks)
        # or per-task:
        result = orch.solve_task(task)
    """

    def __init__(self, config: CombinedConfig | None = None):
        self.config = config or CombinedConfig()
        self.neural_solver: ARCSolver | None = None
        self._stats = {
            "dsl_solved": 0,
            "neural_solved": 0,
            "repaired": 0,
            "total": 0,
            "time_total": 0.0,
        }

    def load_neural_model(
        self,
        checkpoint_path: str | None = None,
        solver: ARCSolver | None = None,
    ) -> None:
        """Load or set the neural solver.

        Either provide a checkpoint path to load from disk, or pass
        a pre-initialized ARCSolver instance.
        """
        if solver is not None:
            self.neural_solver = solver
            return

        nc = self.config.neural
        solver_config = SolverConfig(
            z_dim=nc.z_dim,
            hyp_dim=nc.hyp_dim,
            hidden=nc.hidden,
            curvature=nc.curvature,
            refine_steps=nc.refine_steps,
            refine_lr=nc.refine_lr,
            n_augments=nc.n_augments,
            device=self.config.device,
        )
        self.neural_solver = ARCSolver(solver_config)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.config.device)
            self.neural_solver.load_state_dict(state)

        self.neural_solver.to(self.config.device)
        self.neural_solver.eval()

    def solve_task(self, task: ARCTask) -> TaskResult:
        """Solve a single ARC task using the combined pipeline.

        Tries solvers in priority order, stopping early when confident.
        """
        t0 = time.time()
        self._stats["total"] += 1
        n_tests = len(task.test)

        # Stage 1: DSL (fast, exact)
        dsl_result = self._try_dsl(task)
        if dsl_result is not None:
            self._stats["dsl_solved"] += 1
            elapsed = time.time() - t0
            self._stats["time_total"] += elapsed
            return TaskResult(
                task_id=task.task_id,
                predictions=dsl_result,
                solver_used="dsl",
                time_elapsed=elapsed,
                confidence=1.0,
                stats={"method": "dsl_exact"},
            )

        # Stage 2: Neural + AIRV
        neural_result = self._try_neural_airv(task, time_limit=self.config.time.neural_ttt_time)

        # Stage 3: Repair pass on neural output
        if neural_result is not None and self.config.repair.enabled:
            repaired_result = self._try_repair(task, neural_result)
            if repaired_result is not None:
                self._stats["repaired"] += 1
                neural_result = repaired_result

        if neural_result is not None:
            self._stats["neural_solved"] += 1

        elapsed = time.time() - t0
        self._stats["time_total"] += elapsed

        # Fallback: if neural failed entirely, return blank grids
        if neural_result is None:
            neural_result = []
            for test_pair in task.test:
                h, w = test_pair.input.shape
                blank = np.zeros((h, w), dtype=int)
                neural_result.append([blank, blank])

        return TaskResult(
            task_id=task.task_id,
            predictions=neural_result,
            solver_used="neural_airv" if neural_result else "fallback",
            time_elapsed=elapsed,
            confidence=0.5,
        )

    def _try_dsl(self, task: ARCTask) -> list[list[np.ndarray]] | None:
        """Stage 1: Try DSL program search."""
        try:
            return solve_with_dsl(
                task,
                max_depth=self.config.dsl.max_depth,
                time_limit=self.config.dsl.time_limit,
            )
        except Exception as e:
            logger.warning(f"DSL failed on {task.task_id}: {e}")
            return None

    def _try_neural_airv(
        self,
        task: ARCTask,
        time_limit: float = 45.0,
    ) -> list[list[np.ndarray]] | None:
        """Stage 2: Neural solver with AIRV voting."""
        if self.neural_solver is None:
            logger.warning("Neural solver not loaded, skipping")
            return None

        try:
            return solve_with_neural_airv(
                task,
                solver=self.neural_solver,
                neural_config=self.config.neural,
                airv_config=self.config.airv,
                time_limit=time_limit,
            )
        except Exception as e:
            logger.warning(f"Neural AIRV failed on {task.task_id}: {e}")
            return None

    def _try_repair(
        self,
        task: ARCTask,
        predictions: list[list[np.ndarray]],
    ) -> list[list[np.ndarray]] | None:
        """Stage 3: Repair near-misses."""
        rc = self.config.repair

        repaired = []
        any_changed = False

        for t_idx, (test_pair, preds) in enumerate(zip(task.test, predictions)):
            test_input = test_pair.input
            new_preds = []

            for pred in preds:
                candidates = repair_prediction(
                    pred,
                    task,
                    test_input,
                    test_idx=t_idx,
                    use_size=rc.use_size_correction,
                    use_border=rc.use_border_fix,
                    use_majority=rc.use_color_majority,
                    use_symmetry=rc.use_symmetry_completion,
                )
                best = select_best_repair(candidates, task)
                if not np.array_equal(best, pred):
                    any_changed = True
                new_preds.append(best)

            repaired.append(new_preds)

        return repaired if any_changed else predictions

    def solve_all(
        self,
        tasks: list[ARCTask],
        progress_callback=None,
    ) -> dict[str, TaskResult]:
        """Solve all tasks and return results dict.

        Args:
            tasks: list of ARC tasks
            progress_callback: optional fn(task_idx, task_id, result) called after each

        Returns:
            {task_id: TaskResult}
        """
        results = {}

        for i, task in enumerate(tasks):
            result = self.solve_task(task)
            results[task.task_id] = result

            if progress_callback:
                progress_callback(i, task.task_id, result)

            # Log progress periodically
            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                n = i + 1
                dsl_pct = self._stats["dsl_solved"] / max(n, 1) * 100
                neural_pct = self._stats["neural_solved"] / max(n, 1) * 100
                avg_time = self._stats["time_total"] / max(n, 1)
                logger.info(
                    f"[{n}/{len(tasks)}] DSL: {self._stats['dsl_solved']} ({dsl_pct:.1f}%) | "
                    f"Neural: {self._stats['neural_solved']} ({neural_pct:.1f}%) | "
                    f"Repaired: {self._stats['repaired']} | "
                    f"Avg time: {avg_time:.1f}s"
                )

        return results

    @property
    def stats(self) -> dict:
        """Return current solver statistics."""
        return dict(self._stats)

    def format_submission(
        self,
        results: dict[str, TaskResult],
    ) -> dict[str, list[list[list[int]]]]:
        """Format results as an ARC submission JSON.

        Output format: {task_id: [[attempt1, attempt2], ...]}
        where each attempt is a 2D list of ints.
        """
        submission = {}
        for task_id, result in results.items():
            task_preds = []
            for test_preds in result.predictions:
                attempts = []
                for pred in test_preds[:2]:  # Max 2 attempts
                    attempts.append(pred.tolist())
                # Pad to exactly 2 attempts
                while len(attempts) < 2:
                    attempts.append(attempts[-1] if attempts else [[0]])
                task_preds.append(attempts)
            submission[task_id] = task_preds
        return submission
