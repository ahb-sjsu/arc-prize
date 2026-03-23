# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Configuration for the combined ARC solver pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimebudgetConfig:
    """Per-task time allocation across solvers."""

    total_per_task: float = 90.0  # seconds total budget per task
    dsl_time: float = 20.0  # DSL search (fast, try first)
    neural_ttt_time: float = 45.0  # Neural TTT with AIRV
    repair_time: float = 15.0  # Near-miss repair pass
    # Reserve remaining time for overhead + safety margin


@dataclass
class DSLConfig:
    """DSL solver configuration."""

    max_depth: int = 3
    time_limit: float = 20.0  # per-task search time


@dataclass
class AIRVConfig:
    """Augmented Inference with Randomized Voting — borrowed from MindsAI."""

    enabled: bool = True
    n_augmentations: int = 8  # number of geometric augmentations per inference
    n_color_perms: int = 4  # number of color permutations per augmentation
    vote_threshold: int = 2  # minimum votes for a candidate to be considered
    use_beam_search: bool = True
    beam_width: int = 2  # candidates per augmentation pass


@dataclass
class NeuralConfig:
    """Neural solver configuration."""

    z_dim: int = 128
    hyp_dim: int = 32
    hidden: int = 256
    curvature: float = 1.0
    refine_steps: int = 50
    refine_lr: float = 1e-4
    n_augments: int = 4
    # LR search (borrowed from MindsAI)
    lr_search_enabled: bool = True
    lr_search_trials: int = 4
    lr_search_min: float = 2e-6
    lr_search_max: float = 5e-4


@dataclass
class RepairConfig:
    """Near-miss repair pass configuration."""

    enabled: bool = True
    cell_accuracy_threshold: float = 0.85  # minimum cell accuracy to attempt repair
    max_repair_attempts: int = 50  # random local perturbations to try
    # Repair strategies
    use_border_fix: bool = True
    use_color_majority: bool = True
    use_size_correction: bool = True
    use_symmetry_completion: bool = True


@dataclass
class CombinedConfig:
    """Full configuration for the combined ARC solver."""

    time: TimebudgetConfig = field(default_factory=TimebudgetConfig)
    dsl: DSLConfig = field(default_factory=DSLConfig)
    airv: AIRVConfig = field(default_factory=AIRVConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    device: str = "cuda"
    # Solver priority: try these in order, stop early on confidence
    solver_order: list[str] = field(
        default_factory=lambda: ["dsl", "neural_airv", "repair"]
    )
