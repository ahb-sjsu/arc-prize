# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Neural solver with AIRV — wraps the base ARCSolver with MindsAI-style
augmented inference and voting.

Enhancement over base solver:
1. LR search during TTT (borrowed from MindsAI's mini_lr_grid)
2. AIRV: run inference under multiple augmentations, un-augment, vote
3. Multiple candidates from voting consensus
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from arc_prize.data import ARCTask
from arc_prize.solver import ARCSolver, SolverConfig
from arc_prize.combined.voting import (
    AugmentedCandidate,
    AugmentationSpec,
    augment_task_pairs,
    augment_inputs,
    generate_augmentation_specs,
    invert_augmentation,
    vote_on_candidates,
    vote_statistics,
)
from arc_prize.combined.config import AIRVConfig, NeuralConfig


def _lr_search(
    solver: ARCSolver,
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    device: str,
    n_trials: int = 4,
    lr_min: float = 2e-6,
    lr_max: float = 5e-4,
    steps_per_trial: int = 15,
) -> float:
    """Find the best learning rate for TTT on this task.

    Runs short training trials at log-spaced learning rates and picks
    the one with lowest leave-one-out loss.  Borrowed from MindsAI's
    mini_lr_grid approach.
    """
    import math

    lrs = [lr_min * (lr_max / lr_min) ** (i / max(n_trials - 1, 1)) for i in range(n_trials)]
    best_lr = lrs[len(lrs) // 2]  # default to middle
    best_loss = float("inf")

    original_state = copy.deepcopy(solver.state_dict())

    for lr in lrs:
        solver.load_state_dict(copy.deepcopy(original_state))
        solver.config.refine_lr = lr
        solver.config.refine_steps = steps_per_trial

        try:
            z_rule = solver.refine_on_task(train_pairs, device)

            # Evaluate leave-one-out loss
            total_loss = 0.0
            with torch.no_grad():
                for in_grid, out_grid in train_pairs:
                    pred = solver.predict(z_rule, in_grid, device)
                    # Simple cell accuracy as proxy
                    h = min(pred.shape[0], out_grid.shape[0])
                    w = min(pred.shape[1], out_grid.shape[1])
                    if h > 0 and w > 0:
                        acc = np.mean(pred[:h, :w] == out_grid[:h, :w])
                        total_loss += 1.0 - acc
                    else:
                        total_loss += 1.0

            if total_loss < best_loss:
                best_loss = total_loss
                best_lr = lr
        except RuntimeError:
            continue  # OOM or other error, skip this LR

    # Restore original state for the real training run
    solver.load_state_dict(original_state)
    return best_lr


def solve_with_neural_airv(
    task: ARCTask,
    solver: ARCSolver,
    neural_config: NeuralConfig | None = None,
    airv_config: AIRVConfig | None = None,
    time_limit: float = 45.0,
) -> list[list[np.ndarray]]:
    """Solve a task using neural TTT with AIRV voting.

    Pipeline:
    1. Optional LR search
    2. For each augmentation spec:
       a. Augment training pairs + test inputs
       b. TTT: refine solver on augmented training pairs
       c. Predict augmented test inputs
       d. Un-augment predictions back to original frame
    3. Vote across all un-augmented predictions
    4. Return top-2 candidates per test input

    Returns:
        List of [candidate_1, candidate_2] per test input.
    """
    nc = neural_config or NeuralConfig()
    ac = airv_config or AIRVConfig()
    device = solver.config.device
    t0 = time.time()

    train_pairs = [(p.input, p.output) for p in task.train]
    test_inputs = [p.input for p in task.test]
    n_tests = len(test_inputs)

    # Save original model state — we restore between augmentations
    original_state = copy.deepcopy(solver.state_dict())

    # Step 1: LR search (on original task, no augmentation)
    if nc.lr_search_enabled and time.time() - t0 < time_limit * 0.3:
        best_lr = _lr_search(
            solver,
            train_pairs,
            device,
            n_trials=nc.lr_search_trials,
            lr_min=nc.lr_search_min,
            lr_max=nc.lr_search_max,
        )
        solver.config.refine_lr = best_lr
        solver.load_state_dict(copy.deepcopy(original_state))

    # Step 2: Generate augmentation specs
    if ac.enabled:
        aug_specs = generate_augmentation_specs(
            n_geometric=ac.n_augmentations,
            n_color=ac.n_color_perms,
        )
    else:
        aug_specs = [AugmentationSpec()]  # identity only

    # Step 3: AIRV loop — collect candidates per test input
    per_test_candidates: list[list[AugmentedCandidate]] = [[] for _ in range(n_tests)]

    for spec in aug_specs:
        if time.time() - t0 > time_limit * 0.9:
            break  # respect time budget

        # Restore fresh model state for each augmentation
        solver.load_state_dict(copy.deepcopy(original_state))

        # Augment training pairs and test inputs
        aug_train = augment_task_pairs(train_pairs, spec)
        aug_test = augment_inputs(test_inputs, spec)

        # TTT on augmented task
        solver.config.refine_steps = nc.refine_steps
        try:
            z_rule = solver.refine_on_task(aug_train, device)
        except RuntimeError:
            continue  # OOM, skip this augmentation

        # Predict each augmented test input
        solver.eval()
        with torch.no_grad():
            for t_idx, aug_input in enumerate(aug_test):
                pred_aug = solver.predict(z_rule, aug_input, device)
                # Un-augment prediction back to original frame
                pred_original = invert_augmentation(pred_aug, spec)
                per_test_candidates[t_idx].append(
                    AugmentedCandidate(
                        grid=pred_original,
                        augmentation_id=spec.aug_id,
                    )
                )

    # Step 4: Vote and select top-2 per test input
    results = []
    for t_idx in range(n_tests):
        candidates = per_test_candidates[t_idx]
        top_grids = vote_on_candidates(candidates, top_k=2)

        if len(top_grids) == 0:
            # Fallback: direct prediction without augmentation
            solver.load_state_dict(original_state)
            z_rule = solver.infer_rule(train_pairs, device)
            fallback = solver.predict(z_rule, test_inputs[t_idx], device)
            top_grids = [fallback, fallback]
        elif len(top_grids) == 1:
            top_grids = [top_grids[0], top_grids[0]]

        results.append(top_grids)

    # Restore original state
    solver.load_state_dict(original_state)
    return results
