# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Near-miss repair pass — fix predictions that are almost correct.

Analysis of ARC-AGI-2 eval showed 110/400 tasks with >90% cell accuracy
and 58/400 with >95%.  These near-misses can often be fixed by:

1. Size correction: adjust grid dimensions to match expected output size
2. Border repair: fix border cells using pattern from interior
3. Color majority: replace rare disagreeing cells with local majority
4. Symmetry completion: if grid is almost symmetric, enforce it

Each strategy is lightweight and runs as a post-processing step.
"""

from __future__ import annotations

import numpy as np
from collections import Counter

from arc_prize.data import ARCTask, ARCPair
from arc_prize.evaluate import cell_accuracy


def _detect_symmetry(grid: np.ndarray) -> dict[str, float]:
    """Detect how close a grid is to various symmetries."""
    h, w = grid.shape
    scores = {}

    # Horizontal symmetry (left-right)
    if w > 1:
        flipped = np.fliplr(grid)
        scores["horizontal"] = float(np.mean(grid == flipped))

    # Vertical symmetry (top-bottom)
    if h > 1:
        flipped = np.flipud(grid)
        scores["vertical"] = float(np.mean(grid == flipped))

    # 180-degree rotational
    rotated = np.rot90(grid, 2)
    if rotated.shape == grid.shape:
        scores["rot180"] = float(np.mean(grid == rotated))

    # Diagonal (transpose)
    if h == w:
        scores["diagonal"] = float(np.mean(grid == grid.T))

    return scores


def _infer_output_size(task: ARCTask) -> tuple[int, int] | None:
    """Try to infer the expected output size from training pairs.

    Common patterns:
    - Output same size as input
    - Output is fixed size across all pairs
    - Output is a consistent function of input size (e.g., 2x, half)
    """
    train = task.train

    # Check if all outputs have the same size
    out_shapes = [p.output.shape for p in train]
    if len(set(out_shapes)) == 1:
        return out_shapes[0]

    # Check if output size = input size
    if all(p.input.shape == p.output.shape for p in train):
        return None  # Same-size rule, but varies per input

    # Check ratio patterns
    ratios_h = [p.output.shape[0] / max(p.input.shape[0], 1) for p in train]
    ratios_w = [p.output.shape[1] / max(p.input.shape[1], 1) for p in train]

    if len(set(ratios_h)) == 1 and len(set(ratios_w)) == 1:
        return None  # Ratio rule, applied per-input in caller

    return None


def repair_size(
    prediction: np.ndarray,
    task: ARCTask,
    test_input: np.ndarray,
) -> np.ndarray:
    """Fix grid dimensions if we can infer the correct output size."""
    expected = _infer_output_size(task)

    if expected is not None:
        eh, ew = expected
    else:
        # Check if same-size rule
        if all(p.input.shape == p.output.shape for p in task.train):
            eh, ew = test_input.shape
        else:
            # Check ratio rule
            ratios_h = [p.output.shape[0] / max(p.input.shape[0], 1) for p in task.train]
            ratios_w = [p.output.shape[1] / max(p.input.shape[1], 1) for p in task.train]
            if len(set(ratios_h)) == 1 and len(set(ratios_w)) == 1:
                eh = int(round(test_input.shape[0] * ratios_h[0]))
                ew = int(round(test_input.shape[1] * ratios_w[0]))
            else:
                return prediction  # Can't determine size

    ph, pw = prediction.shape
    if (ph, pw) == (eh, ew):
        return prediction  # Already correct size

    # Resize: crop or pad
    result = np.zeros((eh, ew), dtype=prediction.dtype)
    copy_h = min(ph, eh)
    copy_w = min(pw, ew)
    result[:copy_h, :copy_w] = prediction[:copy_h, :copy_w]
    return result


def repair_border(prediction: np.ndarray) -> np.ndarray:
    """Fix border cells using the dominant border color from training outputs."""
    h, w = prediction.shape
    if h < 3 or w < 3:
        return prediction

    # Collect border colors
    border = []
    border.extend(prediction[0, :].tolist())
    border.extend(prediction[-1, :].tolist())
    border.extend(prediction[1:-1, 0].tolist())
    border.extend(prediction[1:-1, -1].tolist())

    interior = prediction[1:-1, 1:-1]
    if interior.size == 0:
        return prediction

    # Check if border should be uniform
    border_counter = Counter(border)
    dominant, count = border_counter.most_common(1)[0]
    if count / len(border) > 0.8:
        # Border is mostly one color — enforce it
        result = prediction.copy()
        result[0, :] = dominant
        result[-1, :] = dominant
        result[:, 0] = dominant
        result[:, -1] = dominant
        return result

    return prediction


def repair_color_majority(
    prediction: np.ndarray,
    reference: np.ndarray | None = None,
    window: int = 3,
) -> np.ndarray:
    """Replace isolated disagreeing cells with local majority color.

    If reference is provided, only fix cells that differ from reference
    and where the local majority agrees with the reference.
    """
    h, w = prediction.shape
    result = prediction.copy()
    pad = window // 2

    for r in range(h):
        for c in range(w):
            if reference is not None:
                if r < reference.shape[0] and c < reference.shape[1]:
                    if prediction[r, c] == reference[r, c]:
                        continue  # Already agrees

            # Get local neighborhood
            r_lo = max(0, r - pad)
            r_hi = min(h, r + pad + 1)
            c_lo = max(0, c - pad)
            c_hi = min(w, c + pad + 1)
            neighborhood = prediction[r_lo:r_hi, c_lo:c_hi].flatten()

            counter = Counter(neighborhood.tolist())
            majority_color, majority_count = counter.most_common(1)[0]
            if majority_count > len(neighborhood) * 0.6 and majority_color != prediction[r, c]:
                result[r, c] = majority_color

    return result


def repair_symmetry(
    prediction: np.ndarray,
    threshold: float = 0.85,
) -> np.ndarray:
    """If grid is almost symmetric, enforce the symmetry."""
    sym_scores = _detect_symmetry(prediction)

    best_sym = max(sym_scores, key=sym_scores.get) if sym_scores else None
    if best_sym is None or sym_scores[best_sym] < threshold:
        return prediction  # Not close enough to any symmetry

    h, w = prediction.shape
    result = prediction.copy()

    if best_sym == "horizontal" and w > 1:
        # Enforce left-right symmetry using left half
        for r in range(h):
            for c in range(w // 2):
                mirror_c = w - 1 - c
                # Use whichever cell has more local agreement
                result[r, mirror_c] = result[r, c]

    elif best_sym == "vertical" and h > 1:
        for r in range(h // 2):
            mirror_r = h - 1 - r
            result[mirror_r, :] = result[r, :]

    elif best_sym == "rot180":
        for r in range(h // 2):
            for c in range(w):
                mr, mc = h - 1 - r, w - 1 - c
                result[mr, mc] = result[r, c]

    elif best_sym == "diagonal" and h == w:
        for r in range(h):
            for c in range(r + 1, w):
                result[c, r] = result[r, c]

    return result


def repair_prediction(
    prediction: np.ndarray,
    task: ARCTask,
    test_input: np.ndarray,
    test_idx: int = 0,
    *,
    use_size: bool = True,
    use_border: bool = True,
    use_majority: bool = True,
    use_symmetry: bool = True,
) -> list[np.ndarray]:
    """Apply all repair strategies and return improved candidates.

    Returns a list of repaired grids (one per strategy that changed
    something), plus the original.
    """
    candidates = [prediction]

    if use_size:
        sized = repair_size(prediction, task, test_input)
        if not np.array_equal(sized, prediction):
            candidates.append(sized)
            prediction = sized  # Use corrected size for further repairs

    if use_border:
        bordered = repair_border(prediction)
        if not np.array_equal(bordered, prediction):
            candidates.append(bordered)

    if use_majority:
        majored = repair_color_majority(prediction)
        if not np.array_equal(majored, prediction):
            candidates.append(majored)

    if use_symmetry:
        symmed = repair_symmetry(prediction)
        if not np.array_equal(symmed, prediction):
            candidates.append(symmed)

    # Also try chained repairs: size → border → majority → symmetry
    chained = prediction.copy()
    if use_size:
        chained = repair_size(chained, task, test_input)
    if use_border:
        chained = repair_border(chained)
    if use_majority:
        chained = repair_color_majority(chained)
    if use_symmetry:
        chained = repair_symmetry(chained)

    if not any(np.array_equal(chained, c) for c in candidates):
        candidates.append(chained)

    return candidates


def select_best_repair(
    candidates: list[np.ndarray],
    task: ARCTask,
) -> np.ndarray:
    """Select the best repair candidate by checking consistency with training pairs.

    The candidate whose pattern is most consistent with the training outputs
    (color distribution, symmetry properties, size) is preferred.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Score each candidate by similarity to training output statistics
    train_stats = _compute_output_stats(task)

    best_score = -1.0
    best = candidates[0]

    for cand in candidates:
        score = _score_candidate(cand, train_stats)
        if score > best_score:
            best_score = score
            best = cand

    return best


def _compute_output_stats(task: ARCTask) -> dict:
    """Compute statistical properties of training outputs."""
    outputs = [p.output for p in task.train]

    # Color distribution
    all_colors = []
    for out in outputs:
        all_colors.extend(out.flatten().tolist())
    color_dist = Counter(all_colors)
    total = sum(color_dist.values())
    color_freq = {c: count / total for c, count in color_dist.items()}

    # Symmetry scores
    sym_scores = [_detect_symmetry(out) for out in outputs]
    avg_sym = {}
    if sym_scores:
        all_keys = set()
        for s in sym_scores:
            all_keys.update(s.keys())
        for key in all_keys:
            vals = [s.get(key, 0.0) for s in sym_scores]
            avg_sym[key] = sum(vals) / len(vals)

    return {"color_freq": color_freq, "symmetry": avg_sym}


def _score_candidate(grid: np.ndarray, stats: dict) -> float:
    """Score a candidate grid against training output statistics."""
    score = 0.0

    # Color distribution similarity
    flat = grid.flatten().tolist()
    total = len(flat)
    if total > 0:
        cand_freq = Counter(flat)
        cand_freq = {c: count / total for c, count in cand_freq.items()}
        ref_freq = stats["color_freq"]
        all_colors = set(cand_freq) | set(ref_freq)
        color_sim = 1.0 - sum(
            abs(cand_freq.get(c, 0) - ref_freq.get(c, 0)) for c in all_colors
        ) / 2.0
        score += color_sim

    # Symmetry similarity
    cand_sym = _detect_symmetry(grid)
    ref_sym = stats["symmetry"]
    if ref_sym:
        sym_sim = 0.0
        n = 0
        for key in ref_sym:
            if key in cand_sym:
                sym_sim += 1.0 - abs(cand_sym[key] - ref_sym[key])
                n += 1
        if n > 0:
            score += sym_sim / n

    return score
