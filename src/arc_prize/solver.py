# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Main ARC solver with iterative refinement loop.

The dominant pattern from ARC Prize 2025 winners: per-task test-time
training (TTT) that iteratively refines the model's understanding of
each specific task.

Pipeline:
1. Encode all training pairs → z_pairs
2. Infer rule in hyperbolic space (consensus across training pairs)
3. Generate candidate output for test input
4. Iterative refinement: use augmented training pairs as self-supervision
5. Return top-2 candidate outputs (ARC allows 2 attempts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from arc_prize.grid import (
    grid_to_tensor,
    pad_grid,
    grid_size_mask,
)
from arc_prize.encoder import PairEncoder, GridEncoder
from arc_prize.decoder import GridDecoder
from arc_prize.geometric import HyperbolicRuleEncoder, PoincareBall


@dataclass
class SolverConfig:
    """Configuration for the ARC solver."""

    z_dim: int = 128
    hyp_dim: int = 32
    hidden: int = 256
    curvature: float = 1.0
    # Refinement loop
    refine_steps: int = 50
    refine_lr: float = 1e-4
    # Augmentation during refinement
    n_augments: int = 4
    # Number of candidates to return
    n_candidates: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ARCSolver(nn.Module):
    """End-to-end ARC solver.

    Combines encoder, hyperbolic rule inference, decoder, and
    iterative refinement into a single module.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__()
        self.config = config or SolverConfig()
        c = self.config

        self.pair_encoder = PairEncoder(z_dim=c.z_dim)
        self.grid_encoder = GridEncoder(z_dim=c.z_dim, hidden=c.hidden)
        self.rule_encoder = HyperbolicRuleEncoder(
            z_dim=c.z_dim,
            hyp_dim=c.hyp_dim,
            curvature=c.curvature,
        )
        self.decoder = GridDecoder(z_dim=c.z_dim, hidden=c.hidden)

        # Rule aggregation: weighted combination of training pair rules
        self.rule_attn = nn.Sequential(
            nn.Linear(c.hyp_dim, 1),
            nn.Softmax(dim=0),
        )

        # Map aggregated hyperbolic rule back to z-space for decoder
        self.rule_to_z = nn.Sequential(
            nn.Linear(c.hyp_dim, c.z_dim),
            nn.LayerNorm(c.z_dim),
            nn.GELU(),
            nn.Linear(c.z_dim, c.z_dim),
        )

    def _encode_grid(
        self,
        grid: np.ndarray,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single grid to padded tensor + mask."""
        t = grid_to_tensor(grid.tolist(), device=device)
        padded = pad_grid(t)
        h, w = grid.shape
        mask = grid_size_mask(h, w, device=device)
        return padded, mask

    def infer_rule(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        device: str,
    ) -> torch.Tensor:
        """Infer transformation rule from training pairs.

        Encodes each pair → z_pair → hyperbolic rule → weighted aggregate.
        Returns z_rule [1, z_dim] for the decoder.
        """
        z_pairs = []
        for in_grid, out_grid in train_pairs:
            in_t, in_m = self._encode_grid(in_grid, device)
            out_t, out_m = self._encode_grid(out_grid, device)
            z_pair = self.pair_encoder(
                in_t.unsqueeze(0),
                in_m.unsqueeze(0),
                out_t.unsqueeze(0),
                out_m.unsqueeze(0),
            )  # [1, z_dim]
            z_pairs.append(z_pair.squeeze(0))

        # Stack and map to hyperbolic space
        z_stack = torch.stack(z_pairs)  # [N, z_dim]
        h_rules = self.rule_encoder(z_stack)  # [N, hyp_dim]

        # Attention-weighted aggregation in hyperbolic space
        weights = self.rule_attn(h_rules)  # [N, 1]
        h_agg = (weights * h_rules).sum(dim=0, keepdim=True)  # [1, hyp_dim]
        h_agg = PoincareBall.project(h_agg, self.config.curvature)

        # Map back to z-space for decoder
        z_rule = self.rule_to_z(h_agg)  # [1, z_dim]
        return z_rule

    def predict(
        self,
        z_rule: torch.Tensor,
        test_input: np.ndarray,
        device: str,
    ) -> np.ndarray:
        """Generate output grid for a test input given the inferred rule."""
        in_t, in_m = self._encode_grid(test_input, device)
        logits = self.decoder(
            z_rule,
            in_t.unsqueeze(0),
            in_m.unsqueeze(0),
        )  # [1, 10, 30, 30]

        # Extract actual grid dimensions from the output
        pred = logits.argmax(dim=1).squeeze(0)  # [30, 30]
        # Trim to a reasonable output size — use input dims as initial guess
        h, w = test_input.shape
        return pred[:h, :w].cpu().numpy().astype(int)

    def refine_on_task(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        device: str,
    ) -> torch.Tensor:
        """Test-time training: refine the model on a specific task.

        Creates a copy of task-relevant parameters and fine-tunes on
        the training pairs, using augmented versions for more signal.
        """
        from arc_prize.augment import augment_pair

        # Augment training pairs
        all_pairs = list(train_pairs)
        for in_grid, out_grid in train_pairs:
            for aug_in, aug_out in augment_pair(
                in_grid,
                out_grid,
                n_augments=self.config.n_augments,
            ):
                all_pairs.append((aug_in, aug_out))

        # Fine-tune decoder + rule aggregation on this task
        params = (
            list(self.decoder.parameters())
            + list(self.rule_attn.parameters())
            + list(self.rule_to_z.parameters())
        )
        optimizer = optim.Adam(params, lr=self.config.refine_lr)

        self.train()
        for step in range(self.config.refine_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for in_grid, out_grid in all_pairs:
                # Use all OTHER pairs to infer rule, predict THIS pair's output
                # (leave-one-out for more robust rule inference)
                other_pairs = [
                    (i, o)
                    for (i, o) in all_pairs
                    if not (np.array_equal(i, in_grid) and np.array_equal(o, out_grid))
                ]
                if not other_pairs:
                    other_pairs = all_pairs  # Fallback if only 1 pair

                z_rule = self.infer_rule(other_pairs, device)

                in_t, in_m = self._encode_grid(in_grid, device)
                out_t, out_m = self._encode_grid(out_grid, device)

                logits = self.decoder(
                    z_rule,
                    in_t.unsqueeze(0),
                    in_m.unsqueeze(0),
                )  # [1, 10, 30, 30]

                # Cross-entropy loss on actual cells only
                target = out_t.argmax(dim=0).unsqueeze(0)  # [1, 30, 30]
                loss = nn.functional.cross_entropy(
                    logits,
                    target.long(),
                    reduction="none",
                )  # [1, 30, 30]
                # Mask to real cells
                loss = (loss * out_m.unsqueeze(0)).sum() / out_m.sum().clamp(min=1)
                total_loss = total_loss + loss

            total_loss = total_loss / len(all_pairs)
            total_loss.backward()
            optimizer.step()

        self.eval()
        return self.infer_rule(train_pairs, device)

    @torch.no_grad()
    def solve_task(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        *,
        refine: bool = True,
    ) -> List[List[np.ndarray]]:
        """Solve a complete ARC task.

        Args:
            train_pairs: list of (input_grid, output_grid) numpy arrays
            test_inputs: list of test input grids
            refine: whether to do test-time training

        Returns:
            List of [candidate_1, candidate_2] for each test input.
        """
        device = self.config.device
        self.to(device)

        if refine:
            # Test-time training
            z_rule = self.refine_on_task(train_pairs, device)
        else:
            z_rule = self.infer_rule(train_pairs, device)

        results = []
        for test_input in test_inputs:
            # Generate primary candidate
            candidate_1 = self.predict(z_rule, test_input, device)

            # Generate second candidate with augmented rule inference
            # (use augmented training pairs for a different perspective)
            from arc_prize.augment import augment_pair

            aug_pairs = []
            for in_g, out_g in train_pairs:
                augs = augment_pair(in_g, out_g, n_augments=2, seed=99)
                aug_pairs.extend(augs)
            z_rule_alt = self.infer_rule(aug_pairs, device)
            candidate_2 = self.predict(z_rule_alt, test_input, device)

            results.append([candidate_1, candidate_2])

        return results
