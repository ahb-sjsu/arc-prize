# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Radar-like parametric structure probing for ARC grids.

Core idea: apply controllable-intensity transforms to grids and observe
how the model's latent representation changes.  Transforms that DON'T
change the representation reveal invariances (the model "sees through" them).
Transforms that DO change the representation reveal what structural features
the model uses to solve the puzzle.

This is the ARC adaptation of the DEME/Bond-Index parametric fuzzing
framework — instead of ethical reframes, we use geometric grid transforms.
Instead of measuring verdict flip, we measure latent-space displacement.

The "radar" analogy: we send known perturbations (pulses) and measure
how the model's internal state reflects them back.  The response profile
tells us what the model has learned about each task's structure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Parametric grid transforms — each maps (grid, intensity) → grid
# ---------------------------------------------------------------------------


class ParametricTransform:
    """A grid transform with controllable intensity ∈ [0, 1].

    Mirrors the ErisML ParametricTransform API but operates on numpy grids.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[np.ndarray, float], np.ndarray],
        *,
        is_structural_invariant: bool = True,
        description: str = "",
    ):
        self.name = name
        self.fn = fn
        self.is_structural_invariant = is_structural_invariant
        self.description = description

    def __call__(self, grid: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        return self.fn(grid, intensity)

    def at_intensity(self, intensity: float) -> Callable[[np.ndarray], np.ndarray]:
        return lambda g: self(g, intensity)


class TransformChain:
    """Compose multiple (transform, intensity) pairs sequentially."""

    def __init__(self, steps: list[tuple[ParametricTransform, float]]):
        self.steps = steps

    def __call__(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for transform, intensity in self.steps:
            result = transform(result, intensity)
        return result

    @staticmethod
    def random_chains(
        transforms: list[ParametricTransform],
        *,
        n_chains: int = 30,
        max_length: int = 3,
        intensities: list[float] = (0.3, 0.6, 1.0),
        seed: int = 42,
    ) -> list[TransformChain]:
        rng = np.random.RandomState(seed)
        chains = []
        for _ in range(n_chains):
            length = rng.randint(1, max_length + 1)
            steps = []
            for _ in range(length):
                t = transforms[rng.randint(len(transforms))]
                i = intensities[rng.randint(len(intensities))]
                steps.append((t, i))
            chains.append(TransformChain(steps))
        return chains


# ---------------------------------------------------------------------------
# Built-in grid transforms
# ---------------------------------------------------------------------------


def _color_permute(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Permute non-background colors.  Structural invariant — the pattern
    doesn't change, only which colors are used."""
    if intensity < 0.01:
        return grid.copy()
    rng = np.random.RandomState(hash(grid.tobytes()) % (2**31))
    result = grid.copy()
    colors = list(range(1, 10))  # 0 is background
    n_swaps = max(1, int(intensity * len(colors)))
    for _ in range(n_swaps):
        i, j = rng.choice(len(colors), 2, replace=False)
        colors[i], colors[j] = colors[j], colors[i]
    mapping = {0: 0}
    for orig, perm in zip(range(1, 10), colors, strict=True):
        mapping[orig] = perm
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            result[r, c] = mapping[result[r, c]]
    return result


def _rotate(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Rotate grid.  Structural invariant (pattern preserved up to orientation)."""
    k = int(round(intensity * 3))  # 0, 1, 2, or 3 quarter turns
    return np.rot90(grid, k)


def _reflect_h(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Horizontal flip.  Applied when intensity > 0.5."""
    if intensity < 0.5:
        return grid.copy()
    return np.fliplr(grid).copy()


def _reflect_v(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Vertical flip.  Applied when intensity > 0.5."""
    if intensity < 0.5:
        return grid.copy()
    return np.flipud(grid).copy()


def _add_noise(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Randomly flip cells.  NOT a structural invariant — this destroys pattern."""
    if intensity < 0.01:
        return grid.copy()
    rng = np.random.RandomState(hash(grid.tobytes()) % (2**31))
    result = grid.copy()
    prob = intensity * 0.3  # At max intensity, flip 30% of cells
    mask = rng.random(grid.shape) < prob
    new_colors = rng.randint(0, 10, grid.shape)
    result[mask] = new_colors[mask]
    return result


def _crop_border(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Remove border cells (set to background).  Stress test — destroys border info."""
    if intensity < 0.01:
        return grid.copy()
    result = grid.copy()
    border = max(1, int(intensity * min(grid.shape) // 3))
    h, w = grid.shape
    result[:border, :] = 0
    result[h - border :, :] = 0
    result[:, :border] = 0
    result[:, w - border :] = 0
    return result


def _scale_up(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Scale grid by repeating each cell.  Structural invariant (pattern preserved)."""
    if intensity < 0.3:
        return grid.copy()
    factor = 2 if intensity < 0.7 else 3
    scaled = np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
    # Clip to 30x30 max
    return scaled[:30, :30].copy()


def _cell_dropout(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Randomly set cells to background.  Stress test."""
    if intensity < 0.01:
        return grid.copy()
    rng = np.random.RandomState(hash(grid.tobytes()) % (2**31))
    result = grid.copy()
    prob = intensity * 0.4
    mask = rng.random(grid.shape) < prob
    result[mask] = 0
    return result


def _translate(grid: np.ndarray, intensity: float) -> np.ndarray:
    """Shift grid contents, wrapping around.  Structural invariant (w/ periodic boundary)."""
    if intensity < 0.01:
        return grid.copy()
    h, w = grid.shape
    shift_r = int(intensity * h * 0.5) % h
    shift_c = int(intensity * w * 0.5) % w
    return np.roll(np.roll(grid, shift_r, axis=0), shift_c, axis=1)


def make_transform_suite() -> list[ParametricTransform]:
    """Standard suite of ARC grid transforms for structure probing."""
    return [
        # Structural invariants — model should be robust to these
        ParametricTransform(
            "color_permute",
            _color_permute,
            is_structural_invariant=True,
            description="Permute non-background colors",
        ),
        ParametricTransform(
            "rotate", _rotate, is_structural_invariant=True, description="Rotate grid 0-270 degrees"
        ),
        ParametricTransform(
            "reflect_h",
            _reflect_h,
            is_structural_invariant=True,
            description="Horizontal reflection",
        ),
        ParametricTransform(
            "reflect_v", _reflect_v, is_structural_invariant=True, description="Vertical reflection"
        ),
        ParametricTransform(
            "translate", _translate, is_structural_invariant=True, description="Cyclic translation"
        ),
        ParametricTransform(
            "scale_up",
            _scale_up,
            is_structural_invariant=True,
            description="Integer scaling of grid",
        ),
        # Stress transforms — these destroy structure, model should be sensitive
        ParametricTransform(
            "add_noise",
            _add_noise,
            is_structural_invariant=False,
            description="Random cell color flips",
        ),
        ParametricTransform(
            "crop_border",
            _crop_border,
            is_structural_invariant=False,
            description="Remove border cells",
        ),
        ParametricTransform(
            "cell_dropout",
            _cell_dropout,
            is_structural_invariant=False,
            description="Random cell dropout to background",
        ),
    ]


# ---------------------------------------------------------------------------
# Structure Probe — the "radar" system
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of probing a single transform at one intensity."""

    transform_name: str
    intensity: float
    latent_displacement: float  # L2 distance in z-space
    cosine_similarity: float
    is_structural_invariant: bool


@dataclass
class StructureProbeReport:
    """Full probe report for one task, analogous to AdvancedBondIndexResult."""

    task_id: str
    probes: list[ProbeResult] = field(default_factory=list)
    sensitivity_profile: dict[str, float] = field(default_factory=dict)
    invariance_profile: dict[str, float] = field(default_factory=dict)
    robustness_index: float = 0.0  # Analogous to Bond Index

    def compute_profiles(self):
        """Aggregate per-transform statistics."""
        from collections import defaultdict

        displacements = defaultdict(list)
        for p in self.probes:
            displacements[p.transform_name].append(p.latent_displacement)

        for name, vals in displacements.items():
            arr = np.array(vals)
            self.sensitivity_profile[name] = float(arr.mean())
            # Invariance = 1 - normalized displacement (higher = more invariant)
            self.invariance_profile[name] = float(1.0 - np.clip(arr.mean(), 0, 1))

        # Robustness index: how well does the model distinguish structural
        # invariants (low displacement) from stress tests (high displacement)?
        inv_disps = [
            v for p in self.probes if p.is_structural_invariant for v in [p.latent_displacement]
        ]
        stress_disps = [
            v for p in self.probes if not p.is_structural_invariant for v in [p.latent_displacement]
        ]
        if inv_disps and stress_disps:
            inv_mean = np.mean(inv_disps)
            stress_mean = np.mean(stress_disps)
            # Good model: low inv_mean, high stress_mean → index near 1
            if stress_mean + inv_mean > 0:
                self.robustness_index = float((stress_mean - inv_mean) / (stress_mean + inv_mean))


class StructureProbe:
    """Radar-like structure prober for ARC models.

    Given an encoder and a grid, applies parametric transforms at multiple
    intensities and measures how the latent representation changes.

    Usage:
        probe = StructureProbe(encoder, transforms=make_transform_suite())
        report = probe.scan(task_id, input_grid, device="cuda")

    The report reveals:
    - Which transforms the model is invariant to (learned symmetries)
    - Which transforms the model is sensitive to (structural features it uses)
    - The robustness index: separation between invariants and sensitivities
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        transforms: list[ParametricTransform] | None = None,
        intensities: list[float] = (0.1, 0.3, 0.5, 0.7, 1.0),
    ):
        self.encoder = encoder
        self.transforms = transforms or make_transform_suite()
        self.intensities = list(intensities)

    @torch.no_grad()
    def _encode_grid(self, grid: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Encode a raw grid to latent z using the grid encoder."""
        from arc_prize.grid import grid_size_mask, grid_to_tensor, pad_grid

        tensor = grid_to_tensor(grid.tolist(), device=device)
        padded = pad_grid(tensor).unsqueeze(0)  # [1, 10, 30, 30]
        h, w = grid.shape
        mask = grid_size_mask(h, w, device=device).unsqueeze(0)  # [1, 30, 30]
        return self.encoder(padded, mask)  # [1, z_dim]

    @torch.no_grad()
    def scan(
        self,
        task_id: str,
        grid: np.ndarray,
        device: str = "cpu",
    ) -> StructureProbeReport:
        """Run full probe scan on a single grid.

        Returns a StructureProbeReport with per-transform, per-intensity
        displacement measurements and aggregated profiles.
        """
        self.encoder.eval()
        z_original = self._encode_grid(grid, device)

        report = StructureProbeReport(task_id=task_id)

        for transform in self.transforms:
            for intensity in self.intensities:
                transformed_grid = transform(grid, intensity)
                # Handle shape changes from transforms like rotate/scale
                z_transformed = self._encode_grid(transformed_grid, device)

                displacement = float((z_original - z_transformed).norm(dim=-1).item())
                cos_sim = float(
                    torch.nn.functional.cosine_similarity(z_original, z_transformed, dim=-1).item()
                )

                report.probes.append(
                    ProbeResult(
                        transform_name=transform.name,
                        intensity=intensity,
                        latent_displacement=displacement,
                        cosine_similarity=cos_sim,
                        is_structural_invariant=transform.is_structural_invariant,
                    )
                )

        report.compute_profiles()
        return report

    @torch.no_grad()
    def find_adversarial_threshold(
        self,
        grid: np.ndarray,
        transform: ParametricTransform,
        threshold: float = 0.5,
        tolerance: float = 0.01,
        device: str = "cpu",
    ) -> float:
        """Binary search for minimal intensity that causes displacement > threshold.

        Analogous to AdvancedFuzzer.find_adversarial_threshold.
        Returns the intensity ∈ [0, 1] at which the model's representation
        shifts by more than `threshold` L2 distance.
        """
        self.encoder.eval()
        z_original = self._encode_grid(grid, device)

        lo, hi = 0.0, 1.0
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            transformed = transform(grid, mid)
            z_t = self._encode_grid(transformed, device)
            disp = float((z_original - z_t).norm(dim=-1).item())
            if disp > threshold:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
