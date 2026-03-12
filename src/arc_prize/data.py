# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Data loading for ARC-AGI-2 tasks.

Handles loading from arckit format and the official ARC JSON files.
Each task has 2-5 training pairs and 1-2 test pairs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from arc_prize.grid import grid_to_tensor, pad_grid, grid_size_mask


@dataclass
class ARCPair:
    """A single input-output grid pair."""

    input: np.ndarray  # [H, W] integers 0-9
    output: np.ndarray  # [H, W] integers 0-9


@dataclass
class ARCTask:
    """A complete ARC task with training and test pairs."""

    task_id: str
    train: List[ARCPair]
    test: List[ARCPair]

    @property
    def n_train(self) -> int:
        return len(self.train)


def load_task_from_json(path: Path) -> ARCTask:
    """Load a single ARC task from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    train = [
        ARCPair(
            input=np.array(p["input"], dtype=np.int64),
            output=np.array(p["output"], dtype=np.int64),
        )
        for p in data["train"]
    ]
    test = [
        ARCPair(
            input=np.array(p["input"], dtype=np.int64),
            output=np.array(p["output"], dtype=np.int64),
        )
        for p in data["test"]
    ]
    return ARCTask(task_id=path.stem, train=train, test=test)


def load_tasks_from_dir(task_dir: Path) -> List[ARCTask]:
    """Load all ARC tasks from a directory of JSON files."""
    tasks = []
    for path in sorted(task_dir.glob("*.json")):
        tasks.append(load_task_from_json(path))
    return tasks


def load_arckit_tasks() -> Tuple[List[ARCTask], List[ARCTask]]:
    """Load ARC-AGI-1 tasks via the arckit package.

    Returns (training_tasks, evaluation_tasks).
    """
    try:
        import arckit
    except ImportError:
        raise ImportError("Install arckit: pip install arckit")

    train_set, eval_set = arckit.load_data()

    def convert(task) -> ARCTask:
        train = [
            ARCPair(
                input=np.array(inp, dtype=np.int64),
                output=np.array(out, dtype=np.int64),
            )
            for inp, out in zip(task.train_inputs, task.train_outputs)
        ]
        test = [
            ARCPair(
                input=np.array(inp, dtype=np.int64),
                output=np.array(out, dtype=np.int64),
            )
            for inp, out in zip(task.test_inputs, task.test_outputs)
        ]
        return ARCTask(task_id=task.id, train=train, test=test)

    train_tasks = [convert(t) for t in train_set]
    eval_tasks = [convert(t) for t in eval_set]
    return train_tasks, eval_tasks


class ARCDataset(Dataset):
    """PyTorch dataset over ARC training pairs.

    Each item yields tensors ready for the encoder:
    - in_grid:  [10, 30, 30] one-hot padded input
    - in_mask:  [30, 30] binary mask
    - out_grid: [10, 30, 30] one-hot padded output
    - out_mask: [30, 30] binary mask
    - task_id:  string identifier
    """

    def __init__(
        self,
        tasks: List[ARCTask],
        augment: bool = False,
        n_augments: int = 4,
        device: str = "cpu",
    ):
        self.device = device
        self.items: List[Tuple[ARCPair, str]] = []
        for task in tasks:
            for pair in task.train:
                self.items.append((pair, task.task_id))

        if augment:
            from arc_prize.augment import augment_pair

            augmented = []
            for pair, tid in self.items:
                aug_pairs = augment_pair(pair.input, pair.output, n_augments=n_augments)
                for aug_in, aug_out in aug_pairs:
                    augmented.append(
                        (
                            ARCPair(input=aug_in, output=aug_out),
                            tid,
                        )
                    )
            self.items.extend(augmented)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        pair, task_id = self.items[idx]

        in_t = grid_to_tensor(pair.input.tolist(), device=self.device)
        in_padded = pad_grid(in_t)
        ih, iw = pair.input.shape
        in_mask = grid_size_mask(ih, iw, device=self.device)

        out_t = grid_to_tensor(pair.output.tolist(), device=self.device)
        out_padded = pad_grid(out_t)
        oh, ow = pair.output.shape
        out_mask = grid_size_mask(oh, ow, device=self.device)

        return {
            "in_grid": in_padded,
            "in_mask": in_mask,
            "out_grid": out_padded,
            "out_mask": out_mask,
            "task_id": task_id,
        }
