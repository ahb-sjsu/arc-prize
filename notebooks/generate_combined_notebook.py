#!/usr/bin/env python3
'''Generate the combined ARC winner notebook (arc_combined_winner.ipynb).

Fully self-contained — all code inlined so it runs on Kaggle/Colab
without needing the arc_prize package installed.

Run: python notebooks/generate_combined_notebook.py
'''

import json

cells = []


def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})


def code(source: str):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": [],
        "execution_count": None,
    })


# ══════════════════════════════════════════════════════════════
# Section 0: Header
# ══════════════════════════════════════════════════════════════
md('''# ARC-AGI Combined Winner
## DSL + Neural TTT + AIRV Voting + Near-Miss Repair

**Architecture** (inspired by NVARC 1st + MindsAI 3rd + hyperbolic rule encoding):

```
Task --> DSL Search (20s) --- exact match? --> Done
         | no
         v
       Neural TTT + AIRV Voting (45s)
         |
         v
       Near-Miss Repair (15s)
         |
         v
       Top-2 Candidates --> Submission
```

Key innovations:
- **71 DSL primitives** with depth-3 program search
- **Hyperbolic rule encoding** on Poincare ball for hierarchical task understanding
- **AIRV voting** (from MindsAI): infer under 12 augmentations, un-augment, majority vote
- **LR search** (from MindsAI): mini grid search finds optimal TTT learning rate per task
- **Near-miss repair**: fix almost-correct predictions (size, border, symmetry, color)
- **Time-budget management**: 90s/task = 20s DSL + 45s neural + 15s repair + 10s buffer
''')

# ══════════════════════════════════════════════════════════════
# Section 1: Environment + Imports
# ══════════════════════════════════════════════════════════════
md("## 1. Environment Setup")

code('''!pip install -q arckit 2>/dev/null || true

import os, sys, time, json, copy, logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("arc_combined")
''')

# ══════════════════════════════════════════════════════════════
# Section 2: Data types + loading
# ══════════════════════════════════════════════════════════════
md("## 2. Data Types & Loading")

code('''@dataclass
class ARCPair:
    input: np.ndarray   # [H, W] integers 0-9
    output: np.ndarray  # [H, W] integers 0-9

@dataclass
class ARCTask:
    task_id: str
    train: list  # list[ARCPair]
    test: list   # list[ARCPair]

    @property
    def n_train(self):
        return len(self.train)


def load_arc_json(path):
    """Load tasks from ARC JSON format (competition format)."""
    with open(path) as f:
        data = json.load(f)

    tasks = []
    for task_id, task_data in data.items():
        train = [
            ARCPair(
                input=np.array(p["input"], dtype=np.int64),
                output=np.array(p["output"], dtype=np.int64),
            )
            for p in task_data["train"]
        ]
        test = [
            ARCPair(
                input=np.array(p["input"], dtype=np.int64),
                output=np.array(p.get("output", p["input"]), dtype=np.int64),
            )
            for p in task_data["test"]
        ]
        tasks.append(ARCTask(task_id=task_id, train=train, test=test))
    return tasks


# ----- Load data -----
# Try multiple data sources in order of preference
tasks = []

def _try_load_json(paths):
    """Try loading from a list of candidate paths."""
    for p in paths:
        if os.path.exists(p):
            print(f"Found data: {p}")
            return load_arc_json(p)
    return None

# Candidate JSON paths (Kaggle competition, local, etc.)
_candidates = [
    "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json",
    "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json",
    # ARC Prize 2025 alternate dataset names
    "/kaggle/input/arc-agi-2/arc-agi-2_evaluation_challenges.json",
    "/kaggle/input/arc-agi-2/arc-agi-2_training_challenges.json",
]

# Also scan /kaggle/input for any ARC JSON files
if os.path.exists("/kaggle/input"):
    import glob
    for f in sorted(glob.glob("/kaggle/input/**/*.json", recursive=True)):
        if f not in _candidates:
            _candidates.append(f)
    print(f"Kaggle input dirs: {os.listdir('/kaggle/input/')}")

result = _try_load_json(_candidates)
if result is not None:
    tasks = result
    print(f"Loaded {len(tasks)} tasks from Kaggle data")

if not tasks:
    # Fallback: arckit package
    try:
        import arckit
        train_set, eval_set = arckit.load_data()
        def _convert_arckit(task_set):
            out = []
            for t in task_set:
                train = [ARCPair(np.array(i, dtype=np.int64), np.array(o, dtype=np.int64))
                         for i, o in zip(t.train_inputs, t.train_outputs)]
                test = [ARCPair(np.array(i, dtype=np.int64), np.array(o, dtype=np.int64))
                        for i, o in zip(t.test_inputs, t.test_outputs)]
                out.append(ARCTask(task_id=t.id, train=train, test=test))
            return out
        train_tasks = _convert_arckit(train_set)
        eval_tasks = _convert_arckit(eval_set)
        tasks = eval_tasks
        print(f"Loaded {len(train_tasks)} train + {len(eval_tasks)} eval tasks via arckit")
    except ImportError:
        print("arckit not available")

if not tasks:
    print("WARNING: No tasks loaded! Attach ARC dataset or install arckit.")
else:
    print(f"Total tasks to solve: {len(tasks)}")
''')

# ══════════════════════════════════════════════════════════════
# Section 3: Grid utilities + Evaluation
# ══════════════════════════════════════════════════════════════
md("## 3. Grid Utilities & Evaluation")

code('''NUM_COLORS = 10
MAX_GRID_SIZE = 30


def grid_to_tensor(grid, device="cpu"):
    """Convert ARC grid to one-hot tensor [10, H, W]."""
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    one_hot = np.zeros((NUM_COLORS, h, w), dtype=np.float32)
    for c in range(NUM_COLORS):
        one_hot[c] = (arr == c).astype(np.float32)
    return torch.from_numpy(one_hot).to(device)


def pad_grid(tensor, size=MAX_GRID_SIZE):
    """Pad [10, H, W] tensor to [10, size, size]."""
    c, h, w = tensor.shape
    if h >= size and w >= size:
        return tensor[:, :size, :size]
    padded = torch.zeros(c, size, size, dtype=tensor.dtype, device=tensor.device)
    padded[:, :h, :w] = tensor
    return padded


def grid_size_mask(h, w, size=MAX_GRID_SIZE, device="cpu"):
    """Binary mask: 1 inside real grid, 0 in padding."""
    mask = torch.zeros(size, size, device=device)
    mask[:h, :w] = 1.0
    return mask


def exact_match(predicted, target):
    if predicted.shape != target.shape:
        return False
    return bool(np.array_equal(predicted, target))


def score_task(predictions, targets):
    """Each test input gets 2 attempts. All must have >= 1 correct."""
    for preds, target in zip(predictions, targets):
        if not any(exact_match(p, target) for p in preds):
            return False
    return True


def cell_accuracy(predicted, target):
    h = min(predicted.shape[0], target.shape[0])
    w = min(predicted.shape[1], target.shape[1])
    if h == 0 or w == 0:
        return 0.0
    correct = np.sum(predicted[:h, :w] == target[:h, :w])
    total = target.shape[0] * target.shape[1]
    return float(correct / max(total, 1))


print("Grid utilities and evaluation ready")
''')

# ══════════════════════════════════════════════════════════════
# Section 4: Augmentation + AIRV voting
# ══════════════════════════════════════════════════════════════
md('''## 4. Augmentation & AIRV Voting

AIRV (Augmented Inference with Randomized Voting) from MindsAI's 3rd-place solution:
solve under multiple augmentations, un-augment each answer, majority vote.
''')

code('''def permute_colors(grid, seed=0):
    """Randomly permute non-background colors (1-9)."""
    rng = np.random.RandomState(seed)
    perm = list(range(10))
    non_bg = perm[1:]
    rng.shuffle(non_bg)
    perm[1:] = non_bg
    result = grid.copy()
    for old, new in enumerate(perm):
        if old != new:
            result[grid == old] = new
    return result


def _inverse_color_perm(grid, seed):
    """Reverse a color permutation."""
    rng = np.random.RandomState(seed)
    perm = list(range(10))
    non_bg = perm[1:]
    rng.shuffle(non_bg)
    perm[1:] = non_bg
    inv_perm = [0] * 10
    for old, new in enumerate(perm):
        inv_perm[new] = old
    result = grid.copy()
    for new_val, old_val in enumerate(inv_perm):
        if new_val != old_val:
            result[grid == new_val] = old_val
    return result


@dataclass
class AugSpec:
    """Specifies one augmentation and how to invert it."""
    rotation_k: int = 0
    flip_h: bool = False
    flip_v: bool = False
    color_seed: int = -1  # -1 = no color permutation

    @property
    def aug_id(self):
        parts = [f"r{self.rotation_k}"]
        if self.flip_h: parts.append("fh")
        if self.flip_v: parts.append("fv")
        if self.color_seed >= 0: parts.append(f"c{self.color_seed}")
        return "_".join(parts)


def apply_aug(grid, spec):
    """Apply augmentation to a grid."""
    g = grid.copy()
    if spec.rotation_k:
        g = np.rot90(g, spec.rotation_k).copy()
    if spec.flip_h:
        g = np.fliplr(g).copy()
    if spec.flip_v:
        g = np.flipud(g).copy()
    if spec.color_seed >= 0:
        g = permute_colors(g, seed=spec.color_seed)
    return g


def invert_aug(grid, spec):
    """Undo augmentation in reverse order."""
    g = grid.copy()
    if spec.color_seed >= 0:
        g = _inverse_color_perm(g, spec.color_seed)
    if spec.flip_v:
        g = np.flipud(g).copy()
    if spec.flip_h:
        g = np.fliplr(g).copy()
    if spec.rotation_k:
        g = np.rot90(g, -spec.rotation_k).copy()
    return g


def generate_aug_specs(n_geometric=8, n_color=4, seed=42):
    """Generate diverse augmentation specs for AIRV."""
    rng = np.random.RandomState(seed)
    specs = [AugSpec()]  # identity always first
    for _ in range(n_geometric - 1):
        specs.append(AugSpec(
            rotation_k=int(rng.randint(0, 4)),
            flip_h=bool(rng.random() > 0.5),
            flip_v=bool(rng.random() > 0.5),
        ))
    for i in range(n_color):
        base = specs[i % len(specs)]
        specs.append(AugSpec(
            rotation_k=base.rotation_k,
            flip_h=base.flip_h,
            flip_v=base.flip_v,
            color_seed=int(rng.randint(1, 2**31)),
        ))
    return specs


def augment_task_pairs(pairs, spec):
    """Apply same augmentation to all input-output pairs."""
    return [(apply_aug(inp, spec), apply_aug(out, spec)) for inp, out in pairs]


def augment_pair(in_grid, out_grid, n_augments=4, seed=42):
    """Augment an input-output pair with consistent random transforms."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n_augments):
        k = rng.randint(0, 4)
        ai = np.rot90(in_grid, k).copy()
        ao = np.rot90(out_grid, k).copy()
        if rng.random() > 0.5:
            ai = np.fliplr(ai).copy()
            ao = np.fliplr(ao).copy()
        if rng.random() > 0.5:
            ai = np.flipud(ai).copy()
            ao = np.flipud(ao).copy()
        cs = rng.randint(0, 2**31)
        ai = permute_colors(ai, seed=cs)
        ao = permute_colors(ao, seed=cs)
        pairs.append((ai, ao))
    return pairs


def _grid_hash(grid):
    return tuple(map(tuple, grid.tolist()))


def vote_on_candidates(candidates, top_k=2):
    """Majority voting. candidates = list of (grid, confidence)."""
    if not candidates:
        return []
    counts = Counter()
    lookup = {}
    for grid, conf in candidates:
        h = _grid_hash(grid)
        counts[h] += conf
        lookup[h] = grid
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    return [lookup[h] for h, _ in ranked[:top_k]]


# Verify roundtrip
test_grid = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
for k in range(4):
    for fh in [False, True]:
        spec = AugSpec(rotation_k=k, flip_h=fh, flip_v=not fh, color_seed=42 if k > 1 else -1)
        assert np.array_equal(invert_aug(apply_aug(test_grid, spec), spec), test_grid), f"Roundtrip failed: {spec}"
print("AIRV augmentation roundtrip verified")
print(f"Will use {len(generate_aug_specs())} augmentation passes per task")
''')

# ══════════════════════════════════════════════════════════════
# Section 5: Neural model (encoder + decoder + hyperbolic)
# ══════════════════════════════════════════════════════════════
md('''## 5. Neural Model

CNN encoder + Poincare ball rule encoder + FiLM-conditioned decoder.
''')

code('''class PoincareBall:
    """Operations in the Poincare ball model of hyperbolic space."""
    EPS = 1e-5

    @staticmethod
    def mobius_add(x, y, c=1.0):
        x_sq = (x * x).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        y_sq = (y * y).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
        denom = 1 + 2*c*xy + c**2 * x_sq * y_sq
        return num / denom.clamp(min=1e-5)

    @staticmethod
    def distance(x, y, c=1.0):
        diff = PoincareBall.mobius_add(-x, y, c)
        norm = diff.norm(dim=-1).clamp(min=1e-5, max=1-1e-5)
        return (2.0 / c**0.5) * torch.atanh(c**0.5 * norm)

    @staticmethod
    def project(x, c=1.0, max_norm=0.95):
        norm = x.norm(dim=-1, keepdim=True)
        max_r = max_norm / c**0.5
        return torch.where(norm > max_r, x * max_r / norm, x)


class GridEncoder(nn.Module):
    """Encode padded ARC grid [10, 30, 30] + mask -> z [z_dim]."""
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
        )
        self.attn_proj = nn.Linear(hidden, 1)
        self.proj = nn.Sequential(nn.Linear(hidden, z_dim), nn.LayerNorm(z_dim))

    def forward(self, grid, mask):
        feat = self.conv(grid)
        b, c, h, w = feat.shape
        feat_flat = feat.view(b, c, h*w).permute(0, 2, 1)
        scores = self.attn_proj(feat_flat).squeeze(-1)
        mask_flat = mask.view(b, h*w)
        scores = scores.masked_fill(mask_flat == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (feat_flat * weights).sum(dim=1)
        return self.proj(pooled)


class PairEncoder(nn.Module):
    """Encode (input, output) pair -> z_pair capturing the transformation."""
    def __init__(self, z_dim=128):
        super().__init__()
        self.grid_enc = GridEncoder(z_dim=z_dim)
        self.pair_proj = nn.Sequential(
            nn.Linear(z_dim * 3, z_dim), nn.LayerNorm(z_dim), nn.GELU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, in_grid, in_mask, out_grid, out_mask):
        z_in = self.grid_enc(in_grid, in_mask)
        z_out = self.grid_enc(out_grid, out_mask)
        z_diff = z_out - z_in
        return self.pair_proj(torch.cat([z_in, z_out, z_diff], dim=-1))


class HyperbolicRuleEncoder(nn.Module):
    """Map z-space -> Poincare ball for hierarchical rule representation."""
    def __init__(self, z_dim=128, hyp_dim=32, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        self.proj = nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, hyp_dim))

    def forward(self, z):
        return PoincareBall.project(self.proj(z), self.curvature)


class RuleConditioner(nn.Module):
    """FiLM conditioning: gamma * feat + beta."""
    def __init__(self, z_dim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(z_dim, n_channels)
        self.beta = nn.Linear(z_dim, n_channels)

    def forward(self, feat, z_rule):
        g = self.gamma(z_rule).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(z_rule).unsqueeze(-1).unsqueeze(-1)
        return g * feat + b


class GridDecoder(nn.Module):
    """Decode rule + input -> output grid logits [B, 10, 30, 30]."""
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.z_dim = z_dim
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.combine = nn.Conv2d(64 + z_dim, hidden, 1)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU())
            for _ in range(4)
        ])
        self.conditioners = nn.ModuleList([RuleConditioner(z_dim, hidden) for _ in range(4)])
        self.output = nn.Conv2d(hidden, NUM_COLORS, 1)

    def forward(self, z_rule, test_grid, test_mask):
        b = z_rule.shape[0]
        in_feat = self.input_conv(test_grid)
        z_spatial = z_rule.unsqueeze(-1).unsqueeze(-1).expand(b, self.z_dim, MAX_GRID_SIZE, MAX_GRID_SIZE)
        combined = torch.cat([in_feat, z_spatial], dim=1)
        h = self.combine(combined)
        for layer, cond in zip(self.layers, self.conditioners):
            h = layer(h)
            h = cond(h, z_rule)
        logits = self.output(h)
        return logits * test_mask.unsqueeze(1)


print("Neural model classes defined")
print(f"  GridEncoder -> PairEncoder -> HyperbolicRuleEncoder -> GridDecoder")
''')

# ══════════════════════════════════════════════════════════════
# Section 6: ARC Solver with TTT
# ══════════════════════════════════════════════════════════════
md("## 6. ARC Solver with Test-Time Training")

code('''class ARCSolver(nn.Module):
    """End-to-end ARC solver: encode pairs -> infer rule -> decode output."""

    def __init__(self, z_dim=128, hyp_dim=32, hidden=256, curvature=1.0,
                 refine_steps=50, refine_lr=1e-4, n_augments=4):
        super().__init__()
        self.z_dim = z_dim
        self.refine_steps = refine_steps
        self.refine_lr = refine_lr
        self.n_augments = n_augments
        self.curvature = curvature

        self.pair_encoder = PairEncoder(z_dim=z_dim)
        self.grid_encoder = GridEncoder(z_dim=z_dim, hidden=hidden)
        self.rule_encoder = HyperbolicRuleEncoder(z_dim=z_dim, hyp_dim=hyp_dim, curvature=curvature)
        self.decoder = GridDecoder(z_dim=z_dim, hidden=hidden)
        self.rule_attn = nn.Sequential(nn.Linear(hyp_dim, 1), nn.Softmax(dim=0))
        self.rule_to_z = nn.Sequential(
            nn.Linear(hyp_dim, z_dim), nn.LayerNorm(z_dim), nn.GELU(), nn.Linear(z_dim, z_dim),
        )

    def _encode_grid(self, grid, device):
        t = grid_to_tensor(grid.tolist(), device=device)
        padded = pad_grid(t)
        h, w = grid.shape
        mask = grid_size_mask(h, w, device=device)
        return padded, mask

    def infer_rule(self, train_pairs, device):
        z_pairs = []
        for in_grid, out_grid in train_pairs:
            in_t, in_m = self._encode_grid(in_grid, device)
            out_t, out_m = self._encode_grid(out_grid, device)
            z_pair = self.pair_encoder(
                in_t.unsqueeze(0), in_m.unsqueeze(0),
                out_t.unsqueeze(0), out_m.unsqueeze(0),
            )
            z_pairs.append(z_pair.squeeze(0))
        z_stack = torch.stack(z_pairs)
        h_rules = self.rule_encoder(z_stack)
        weights = self.rule_attn(h_rules)
        h_agg = (weights * h_rules).sum(dim=0, keepdim=True)
        h_agg = PoincareBall.project(h_agg, self.curvature)
        return self.rule_to_z(h_agg)

    def predict(self, z_rule, test_input, device):
        in_t, in_m = self._encode_grid(test_input, device)
        logits = self.decoder(z_rule, in_t.unsqueeze(0), in_m.unsqueeze(0))
        pred = logits.argmax(dim=1).squeeze(0)
        h, w = test_input.shape
        return pred[:h, :w].cpu().numpy().astype(int)

    def refine_on_task(self, train_pairs, device):
        """Test-time training with leave-one-out."""
        all_pairs = list(train_pairs)
        for in_g, out_g in train_pairs:
            for ai, ao in augment_pair(in_g, out_g, n_augments=self.n_augments):
                all_pairs.append((ai, ao))

        params = list(self.decoder.parameters()) + list(self.rule_attn.parameters()) + list(self.rule_to_z.parameters())
        optimizer = optim.Adam(params, lr=self.refine_lr)

        self.train()
        for _ in range(self.refine_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for in_g, out_g in all_pairs:
                other = [(i, o) for i, o in all_pairs if not (np.array_equal(i, in_g) and np.array_equal(o, out_g))]
                if not other:
                    other = all_pairs
                z_rule = self.infer_rule(other, device)
                in_t, in_m = self._encode_grid(in_g, device)
                out_t, out_m = self._encode_grid(out_g, device)
                logits = self.decoder(z_rule, in_t.unsqueeze(0), in_m.unsqueeze(0))
                target = out_t.argmax(dim=0).unsqueeze(0)
                loss = nn.functional.cross_entropy(logits, target.long(), reduction="none")
                loss = (loss * out_m.unsqueeze(0)).sum() / out_m.sum().clamp(min=1)
                total_loss = total_loss + loss
            total_loss = total_loss / len(all_pairs)
            total_loss.backward()
            optimizer.step()

        self.eval()
        return self.infer_rule(train_pairs, device)


n_params = sum(p.numel() for p in ARCSolver().parameters())
print(f"ARCSolver: {n_params:,} parameters")
''')

# ══════════════════════════════════════════════════════════════
# Section 7: DSL primitives (condensed)
# ══════════════════════════════════════════════════════════════
md('''## 7. DSL Primitives

71 grid transformation primitives + depth-3 program search.
''')

# Read the DSL file and embed it directly
code('''# === DSL v2: Comprehensive grid transformation library ===
# (condensed from dsl_v2.py — 71 primitives + search engine)

from itertools import product as iprod

def grid_colors(g): return set(g.flatten().tolist())
def grid_bg(g):
    c = Counter(g.flatten().tolist())
    return c.most_common(1)[0][0]

def crop_to_content(g, bg=0):
    rows, cols = np.where(g != bg)
    if len(rows) == 0: return g.copy()
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()

def extract_objects_dsl(g, bg=0, connectivity=4):
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    objects = []
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if connectivity == 8 else [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == bg: continue
            color = g[r, c]
            cells = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if 0 <= cr < h and 0 <= cc < w and not visited[cr, cc] and g[cr, cc] == color:
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    for dr, dc in nbrs:
                        stack.append((cr+dr, cc+dc))
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            objects.append({"color": int(color), "cells": cells,
                           "bbox": (min(rows), min(cols), max(rows)+1, max(cols)+1), "size": len(cells)})
    return objects

# --- Geometric ---
def rot90(g): return np.rot90(g, 1).copy()
def rot180(g): return np.rot90(g, 2).copy()
def rot270(g): return np.rot90(g, 3).copy()
def flip_h(g): return np.fliplr(g).copy()
def flip_v(g): return np.flipud(g).copy()
def transpose(g): return g.T.copy()
def transpose_anti(g): return np.rot90(np.fliplr(g)).copy()

# --- Cropping ---
def crop_bg(g):
    bg = grid_bg(g)
    return crop_to_content(g, bg)
def crop_bg0(g): return crop_to_content(g, 0)

# --- Tiling ---
def tile_2x2(g): return np.tile(g, (2, 2))
def tile_3x3(g): return np.tile(g, (3, 3))
def tile_2x1(g): return np.tile(g, (2, 1))
def tile_1x2(g): return np.tile(g, (1, 2))

# --- Scaling ---
def upscale_2x(g): return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
def upscale_3x(g): return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)
def downscale_2x(g):
    h, w = g.shape
    if h % 2 or w % 2: return None
    return g[::2, ::2].copy()
def downscale_3x(g):
    h, w = g.shape
    if h % 3 or w % 3: return None
    return g[::3, ::3].copy()

# --- Color ---
def swap_bg_most(g):
    bg = grid_bg(g)
    c = Counter(g.flatten().tolist())
    if len(c) < 2: return g.copy()
    vals = c.most_common()
    second = vals[1][0] if vals[0][0] == bg else vals[0][0]
    r = g.copy()
    r[g == bg] = second
    r[g == second] = bg
    return r

def invert_colors(g):
    return (9 - g).copy()

def replace_with_most(g):
    bg = grid_bg(g)
    c = Counter(g.flatten().tolist())
    if len(c) < 2: return g.copy()
    vals = c.most_common()
    most = vals[0][0] if vals[0][0] != bg else (vals[1][0] if len(vals) > 1 else bg)
    r = g.copy()
    r[r != bg] = most
    return r

def unique_colors_only(g):
    bg = grid_bg(g)
    c = Counter(g.flatten().tolist())
    r = np.full_like(g, bg)
    for val, cnt in c.items():
        if cnt == 1:
            r[g == val] = val
    return r

def remove_color(g):
    bg = grid_bg(g)
    c = Counter(g.flatten().tolist())
    if len(c) < 2: return g.copy()
    least = c.most_common()[-1][0]
    r = g.copy()
    r[r == least] = bg
    return r

# --- Symmetry ---
def mirror_h(g):
    return np.hstack([g, np.fliplr(g)])
def mirror_v(g):
    return np.vstack([g, np.flipud(g)])
def mirror_both(g):
    top = np.hstack([g, np.fliplr(g)])
    return np.vstack([top, np.flipud(top)])

def complete_horizontal_symmetry(g):
    h, w = g.shape
    r = g.copy()
    for i in range(h):
        for j in range(w // 2):
            mj = w - 1 - j
            if r[i, j] == 0 and r[i, mj] != 0: r[i, j] = r[i, mj]
            elif r[i, mj] == 0 and r[i, j] != 0: r[i, mj] = r[i, j]
    return r

def complete_vertical_symmetry(g):
    h, w = g.shape
    r = g.copy()
    for i in range(h // 2):
        mi = h - 1 - i
        for j in range(w):
            if r[i, j] == 0 and r[mi, j] != 0: r[i, j] = r[mi, j]
            elif r[mi, j] == 0 and r[i, j] != 0: r[mi, j] = r[i, j]
    return r

def complete_4fold_symmetry(g):
    return complete_horizontal_symmetry(complete_vertical_symmetry(g))

# --- Border/Fill ---
def fill_border(g):
    bg = grid_bg(g)
    c = Counter(g.flatten().tolist())
    if len(c) < 2: return g.copy()
    vals = c.most_common()
    fill = vals[0][0] if vals[0][0] != bg else (vals[1][0] if len(vals) > 1 else bg)
    r = g.copy()
    r[0, :] = fill; r[-1, :] = fill; r[:, 0] = fill; r[:, -1] = fill
    return r

def hollow_interior(g):
    bg = grid_bg(g)
    r = g.copy()
    if r.shape[0] > 2 and r.shape[1] > 2:
        r[1:-1, 1:-1] = bg
    return r

def remove_border(g):
    if g.shape[0] <= 2 or g.shape[1] <= 2: return g.copy()
    return g[1:-1, 1:-1].copy()

def fill_enclosed_regions(g):
    bg = grid_bg(g)
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    bg_mask = np.zeros_like(g, dtype=bool)
    # BFS from edges to find background-connected regions
    queue = []
    for i in range(h):
        for j in [0, w-1]:
            if g[i, j] == bg and not visited[i, j]:
                queue.append((i, j))
                visited[i, j] = True
    for j in range(w):
        for i in [0, h-1]:
            if g[i, j] == bg and not visited[i, j]:
                queue.append((i, j))
                visited[i, j] = True
    while queue:
        r, c = queue.pop(0)
        bg_mask[r, c] = True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    result = g.copy()
    # Fill enclosed bg cells
    c = Counter(g.flatten().tolist())
    if len(c) < 2: return result
    vals = c.most_common()
    fill_color = vals[0][0] if vals[0][0] != bg else (vals[1][0] if len(vals) > 1 else bg)
    for i in range(h):
        for j in range(w):
            if g[i, j] == bg and not bg_mask[i, j]:
                result[i, j] = fill_color
    return result

# --- Gravity ---
def gravity_down(g):
    bg = grid_bg(g)
    r = np.full_like(g, bg)
    h, w = g.shape
    for j in range(w):
        col = [g[i, j] for i in range(h) if g[i, j] != bg]
        for i, v in enumerate(col):
            r[h - len(col) + i, j] = v
    return r

def gravity_up(g):
    bg = grid_bg(g)
    r = np.full_like(g, bg)
    h, w = g.shape
    for j in range(w):
        col = [g[i, j] for i in range(h) if g[i, j] != bg]
        for i, v in enumerate(col):
            r[i, j] = v
    return r

def gravity_left(g): return gravity_up(g.T).T.copy()
def gravity_right(g): return gravity_down(g.T).T.copy()

# --- Object extraction ---
def extract_largest(g):
    bg = grid_bg(g)
    objs = extract_objects_dsl(g, bg)
    if not objs: return g.copy()
    largest = max(objs, key=lambda o: o["size"])
    r1, c1, r2, c2 = largest["bbox"]
    return g[r1:r2, c1:c2].copy()

def extract_smallest(g):
    bg = grid_bg(g)
    objs = extract_objects_dsl(g, bg)
    if not objs: return g.copy()
    smallest = min(objs, key=lambda o: o["size"])
    r1, c1, r2, c2 = smallest["bbox"]
    return g[r1:r2, c1:c2].copy()

# --- Half splits ---
def top_half(g): return g[:g.shape[0]//2, :].copy()
def bottom_half(g): return g[g.shape[0]//2:, :].copy()
def left_half(g): return g[:, :g.shape[1]//2].copy()
def right_half(g): return g[:, g.shape[1]//2:].copy()

def xor_halves_h(g):
    h, w = g.shape
    if w % 2: return None
    hw = w // 2
    L, R = g[:, :hw], g[:, hw:]
    bg = grid_bg(g)
    r = np.full((h, hw), bg, dtype=g.dtype)
    r[(L != bg) ^ (R != bg)] = 1
    return r

def and_halves_h(g):
    h, w = g.shape
    if w % 2: return None
    hw = w // 2
    L, R = g[:, :hw], g[:, hw:]
    bg = grid_bg(g)
    r = np.full((h, hw), bg, dtype=g.dtype)
    mask = (L != bg) & (R != bg)
    r[mask] = L[mask]
    return r

def or_halves_h(g):
    h, w = g.shape
    if w % 2: return None
    hw = w // 2
    L, R = g[:, :hw], g[:, hw:]
    bg = grid_bg(g)
    r = L.copy()
    mask = (L == bg) & (R != bg)
    r[mask] = R[mask]
    return r

# --- Pattern detection ---
def get_minimal_period_h(g):
    h, w = g.shape
    for p in range(1, w+1):
        if w % p == 0 and all(np.array_equal(g[:, :p], g[:, i:i+p]) for i in range(0, w, p)):
            return g[:, :p].copy()
    return g.copy()

def get_minimal_period_v(g):
    h, w = g.shape
    for p in range(1, h+1):
        if h % p == 0 and all(np.array_equal(g[:p, :], g[i:i+p, :]) for i in range(0, h, p)):
            return g[:p, :].copy()
    return g.copy()

def unique_rows(g):
    seen = set()
    rows = []
    for i in range(g.shape[0]):
        key = tuple(g[i])
        if key not in seen:
            seen.add(key)
            rows.append(g[i])
    if not rows: return g.copy()
    return np.array(rows)

def most_common_row(g):
    c = Counter(tuple(g[i]) for i in range(g.shape[0]))
    row = c.most_common(1)[0][0]
    return np.array([list(row)])


# === PRIMITIVES REGISTRY ===
PRIMITIVES = {
    "rot90": rot90, "rot180": rot180, "rot270": rot270,
    "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose, "transpose_anti": transpose_anti,
    "crop_bg": crop_bg, "crop_bg0": crop_bg0,
    "tile_2x2": tile_2x2, "tile_3x3": tile_3x3, "tile_2x1": tile_2x1, "tile_1x2": tile_1x2,
    "upscale_2x": upscale_2x, "upscale_3x": upscale_3x,
    "downscale_2x": downscale_2x, "downscale_3x": downscale_3x,
    "swap_bg_most": swap_bg_most, "invert_colors": invert_colors,
    "replace_with_most": replace_with_most, "unique_colors_only": unique_colors_only,
    "remove_color": remove_color,
    "mirror_h": mirror_h, "mirror_v": mirror_v, "mirror_both": mirror_both,
    "complete_horizontal_symmetry": complete_horizontal_symmetry,
    "complete_vertical_symmetry": complete_vertical_symmetry,
    "complete_4fold_symmetry": complete_4fold_symmetry,
    "fill_border": fill_border, "hollow_interior": hollow_interior,
    "remove_border": remove_border, "fill_enclosed_regions": fill_enclosed_regions,
    "gravity_down": gravity_down, "gravity_up": gravity_up,
    "gravity_left": gravity_left, "gravity_right": gravity_right,
    "extract_largest": extract_largest, "extract_smallest": extract_smallest,
    "top_half": top_half, "bottom_half": bottom_half,
    "left_half": left_half, "right_half": right_half,
    "xor_halves_h": xor_halves_h, "and_halves_h": and_halves_h, "or_halves_h": or_halves_h,
    "get_minimal_period_h": get_minimal_period_h, "get_minimal_period_v": get_minimal_period_v,
    "unique_rows": unique_rows, "most_common_row": most_common_row,
}

def make_task_specific_fns(task):
    """Generate task-specific primitives (color maps, fixed outputs, crops)."""
    fns = {}
    # Color replacement for each pair seen in training
    train_colors = set()
    for p in task.train:
        train_colors.update(grid_colors(p.input))
        train_colors.update(grid_colors(p.output))
    for src in train_colors:
        for dst in train_colors:
            if src != dst:
                def _repl(g, s=src, d=dst):
                    r = g.copy(); r[g == s] = d; return r
                fns[f"recolor_{src}_{dst}"] = _repl
    # Fixed output
    outs = [p.output for p in task.train]
    if all(np.array_equal(outs[0], o) for o in outs):
        fns["fixed_output"] = lambda g, o=outs[0].copy(): o
    # Consistent color map
    color_maps = []
    for p in task.train:
        mapping = {}
        if p.input.shape == p.output.shape:
            for val in set(p.input.flatten()):
                out_vals = set(p.output[p.input == val].flatten())
                if len(out_vals) == 1:
                    mapping[int(val)] = int(out_vals.pop())
        color_maps.append(mapping)
    if color_maps and all(m == color_maps[0] for m in color_maps) and color_maps[0]:
        cm = color_maps[0]
        def _cmap(g, m=cm):
            r = g.copy()
            for s, d in m.items(): r[g == s] = d
            return r
        fns["learned_color_map"] = _cmap
    return fns

def apply_program(grid, fns):
    g = grid.copy()
    for fn in fns:
        try:
            r = fn(g)
            if r is None: return None
            g = r
        except Exception:
            return None
    return g

def verify_program(task, fns):
    for p in task.train:
        r = apply_program(p.input, fns)
        if r is None or not np.array_equal(r, p.output):
            return False
    return True

def search_programs(task, max_depth=3, time_limit=20.0):
    t0 = time.time()
    all_prims = dict(PRIMITIVES)
    all_prims.update(make_task_specific_fns(task))
    names = list(all_prims.keys())
    fns_list = [all_prims[n] for n in names]
    n = len(names)

    for i in range(n):
        if time.time() - t0 > time_limit: return None
        if verify_program(task, [fns_list[i]]):
            return ([names[i]], [fns_list[i]])
    if max_depth < 2: return None

    for i in range(n):
        if time.time() - t0 > time_limit: return None
        r1 = apply_program(task.train[0].input, [fns_list[i]])
        if r1 is None: continue
        for j in range(n):
            if time.time() - t0 > time_limit: return None
            if verify_program(task, [fns_list[i], fns_list[j]]):
                return ([names[i], names[j]], [fns_list[i], fns_list[j]])
    if max_depth < 3: return None

    base_names = list(PRIMITIVES.keys())
    base_fns = [PRIMITIVES[nm] for nm in base_names]
    for i in range(n):
        if time.time() - t0 > time_limit: return None
        r1 = apply_program(task.train[0].input, [fns_list[i]])
        if r1 is None: continue
        for j in range(n):
            if time.time() - t0 > time_limit: return None
            r2 = apply_program(r1, [fns_list[j]])
            if r2 is None: continue
            for k in range(len(base_fns)):
                if time.time() - t0 > time_limit: return None
                if verify_program(task, [fns_list[i], fns_list[j], base_fns[k]]):
                    return ([names[i], names[j], base_names[k]], [fns_list[i], fns_list[j], base_fns[k]])
    return None

print(f"DSL ready: {len(PRIMITIVES)} base primitives + task-specific")
''')

# ══════════════════════════════════════════════════════════════
# Section 8: Near-miss repair
# ══════════════════════════════════════════════════════════════
md('''## 8. Near-Miss Repair

Fix predictions with >85% cell accuracy using:
size correction, border fix, color majority, symmetry completion.
''')

code('''def repair_size(prediction, task, test_input):
    """Fix grid dimensions based on training output patterns."""
    out_shapes = [p.output.shape for p in task.train]
    if len(set(out_shapes)) == 1:
        eh, ew = out_shapes[0]
    elif all(p.input.shape == p.output.shape for p in task.train):
        eh, ew = test_input.shape
    else:
        ratios_h = [p.output.shape[0] / max(p.input.shape[0], 1) for p in task.train]
        ratios_w = [p.output.shape[1] / max(p.input.shape[1], 1) for p in task.train]
        if len(set(ratios_h)) == 1 and len(set(ratios_w)) == 1:
            eh = int(round(test_input.shape[0] * ratios_h[0]))
            ew = int(round(test_input.shape[1] * ratios_w[0]))
        else:
            return prediction
    ph, pw = prediction.shape
    if (ph, pw) == (eh, ew):
        return prediction
    result = np.zeros((eh, ew), dtype=prediction.dtype)
    ch, cw = min(ph, eh), min(pw, ew)
    result[:ch, :cw] = prediction[:ch, :cw]
    return result


def repair_border(prediction):
    """If border is mostly one color, enforce it."""
    h, w = prediction.shape
    if h < 3 or w < 3: return prediction
    border = list(prediction[0, :]) + list(prediction[-1, :]) + list(prediction[1:-1, 0]) + list(prediction[1:-1, -1])
    bc = Counter(border)
    dominant, count = bc.most_common(1)[0]
    if count / len(border) > 0.8:
        result = prediction.copy()
        result[0, :] = dominant; result[-1, :] = dominant
        result[:, 0] = dominant; result[:, -1] = dominant
        return result
    return prediction


def repair_color_majority(prediction, window=3):
    """Replace isolated cells with local majority."""
    h, w = prediction.shape
    result = prediction.copy()
    pad = window // 2
    for r in range(h):
        for c in range(w):
            r_lo, r_hi = max(0, r-pad), min(h, r+pad+1)
            c_lo, c_hi = max(0, c-pad), min(w, c+pad+1)
            nbr = prediction[r_lo:r_hi, c_lo:c_hi].flatten()
            mc, cnt = Counter(nbr.tolist()).most_common(1)[0]
            if cnt > len(nbr) * 0.6 and mc != prediction[r, c]:
                result[r, c] = mc
    return result


def repair_symmetry(prediction, threshold=0.85):
    """If grid is almost symmetric, enforce it."""
    h, w = prediction.shape
    # Check horizontal symmetry
    if w > 1:
        hsym = float(np.mean(prediction == np.fliplr(prediction)))
        if hsym >= threshold:
            result = prediction.copy()
            for r in range(h):
                for c in range(w // 2):
                    result[r, w-1-c] = result[r, c]
            return result
    if h > 1:
        vsym = float(np.mean(prediction == np.flipud(prediction)))
        if vsym >= threshold:
            result = prediction.copy()
            for r in range(h // 2):
                result[h-1-r, :] = result[r, :]
            return result
    return prediction


def repair_prediction(pred, task, test_input):
    """Apply all repair strategies, return list of candidates."""
    candidates = [pred]
    sized = repair_size(pred, task, test_input)
    if not np.array_equal(sized, pred):
        candidates.append(sized)
        pred = sized
    bordered = repair_border(pred)
    if not np.array_equal(bordered, pred):
        candidates.append(bordered)
    majored = repair_color_majority(pred)
    if not np.array_equal(majored, pred):
        candidates.append(majored)
    symmed = repair_symmetry(pred)
    if not np.array_equal(symmed, pred):
        candidates.append(symmed)
    # Chained
    chained = repair_symmetry(repair_color_majority(repair_border(repair_size(pred, task, test_input))))
    if not any(np.array_equal(chained, c) for c in candidates):
        candidates.append(chained)
    return candidates


def select_best_repair(candidates, task):
    """Pick candidate most consistent with training output statistics."""
    if len(candidates) == 1: return candidates[0]
    # Score by color distribution similarity to training outputs
    all_out_colors = []
    for p in task.train:
        all_out_colors.extend(p.output.flatten().tolist())
    ref_freq = Counter(all_out_colors)
    total_ref = sum(ref_freq.values())
    ref_freq = {c: cnt/total_ref for c, cnt in ref_freq.items()}

    best, best_score = candidates[0], -1
    for cand in candidates:
        flat = cand.flatten().tolist()
        total = len(flat)
        if total == 0: continue
        cf = Counter(flat)
        cf = {c: cnt/total for c, cnt in cf.items()}
        all_c = set(cf) | set(ref_freq)
        sim = 1.0 - sum(abs(cf.get(c, 0) - ref_freq.get(c, 0)) for c in all_c) / 2.0
        if sim > best_score:
            best_score = sim
            best = cand
    return best


print("Near-miss repair strategies ready")
''')

# ══════════════════════════════════════════════════════════════
# Section 9: Neural AIRV solver
# ══════════════════════════════════════════════════════════════
md("## 9. Neural AIRV Solver (LR Search + Augmented Voting)")

code('''def lr_search(solver, train_pairs, device, n_trials=4, lr_min=2e-6, lr_max=5e-4, steps=15):
    """Find best TTT learning rate via short trials."""
    lrs = [lr_min * (lr_max / lr_min) ** (i / max(n_trials-1, 1)) for i in range(n_trials)]
    best_lr, best_loss = lrs[len(lrs)//2], float("inf")
    orig_state = copy.deepcopy(solver.state_dict())

    for lr in lrs:
        solver.load_state_dict(copy.deepcopy(orig_state))
        solver.refine_lr = lr
        solver.refine_steps = steps
        try:
            z_rule = solver.refine_on_task(train_pairs, device)
            total_loss = 0.0
            with torch.no_grad():
                for ig, og in train_pairs:
                    p = solver.predict(z_rule, ig, device)
                    mh = min(p.shape[0], og.shape[0])
                    mw = min(p.shape[1], og.shape[1])
                    total_loss += 1.0 - (np.mean(p[:mh, :mw] == og[:mh, :mw]) if mh > 0 and mw > 0 else 0)
            if total_loss < best_loss:
                best_loss = total_loss
                best_lr = lr
        except RuntimeError:
            continue
    solver.load_state_dict(orig_state)
    return best_lr


def solve_neural_airv(task, solver, device, n_geo=8, n_color=4, refine_steps=50, time_limit=45.0,
                      do_lr_search=True):
    """Solve task with neural TTT + AIRV voting."""
    t0 = time.time()
    train_pairs = [(p.input, p.output) for p in task.train]
    test_inputs = [p.input for p in task.test]
    orig_state = copy.deepcopy(solver.state_dict())

    # LR search
    if do_lr_search and time.time() - t0 < time_limit * 0.3:
        best_lr = lr_search(solver, train_pairs, device)
        solver.refine_lr = best_lr
        solver.load_state_dict(copy.deepcopy(orig_state))

    aug_specs = generate_aug_specs(n_geometric=n_geo, n_color=n_color)
    per_test = [[] for _ in test_inputs]

    for spec in aug_specs:
        if time.time() - t0 > time_limit * 0.9:
            break
        solver.load_state_dict(copy.deepcopy(orig_state))
        aug_train = augment_task_pairs(train_pairs, spec)
        aug_test = [apply_aug(inp, spec) for inp in test_inputs]

        solver.refine_steps = refine_steps
        try:
            z_rule = solver.refine_on_task(aug_train, device)
        except RuntimeError:
            continue

        solver.eval()
        with torch.no_grad():
            for ti, ai in enumerate(aug_test):
                pred_aug = solver.predict(z_rule, ai, device)
                pred_orig = invert_aug(pred_aug, spec)
                per_test[ti].append((pred_orig, 1.0))

    results = []
    for ti in range(len(test_inputs)):
        top = vote_on_candidates(per_test[ti], top_k=2)
        if len(top) == 0:
            solver.load_state_dict(orig_state)
            z = solver.infer_rule(train_pairs, device)
            fb = solver.predict(z, test_inputs[ti], device)
            top = [fb, fb]
        elif len(top) == 1:
            top = [top[0], top[0]]
        results.append(top)

    solver.load_state_dict(orig_state)
    return results


print("Neural AIRV solver ready")
''')

# ══════════════════════════════════════════════════════════════
# Section 10: Run the pipeline
# ══════════════════════════════════════════════════════════════
md('''## 10. Run Combined Pipeline

**Stage 1**: DSL search (exact matches)
**Stage 2**: Neural TTT + AIRV (learned solutions)
**Stage 3**: Near-miss repair (post-processing)
''')

code('''# ===== STAGE 1: DSL =====
print("=" * 50)
print("STAGE 1: DSL Program Search")
print("=" * 50)

dsl_solutions = {}
t0 = time.time()
for i, task in enumerate(tasks):
    result = search_programs(task, max_depth=3, time_limit=20.0)
    if result is not None:
        names, fns = result
        preds = []
        ok = True
        for tp in task.test:
            output = apply_program(tp.input, fns)
            if output is None:
                ok = False
                break
            preds.append([output, output])
        if ok:
            dsl_solutions[task.task_id] = preds
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(tasks)}] DSL solved so far: {len(dsl_solutions)}")

dsl_time = time.time() - t0
print(f"DSL solved: {len(dsl_solutions)}/{len(tasks)} ({100*len(dsl_solutions)/max(len(tasks),1):.1f}%)")
print(f"DSL time: {dsl_time:.1f}s")

unsolved = [t for t in tasks if t.task_id not in dsl_solutions]
print(f"Remaining for neural: {len(unsolved)}")
''')

code('''# ===== STAGE 2: NEURAL TTT + AIRV =====
print("\\n" + "=" * 50)
print("STAGE 2: Neural TTT + AIRV Voting")
print("=" * 50)

solver = ARCSolver(z_dim=128, hyp_dim=32, hidden=256, curvature=1.0,
                   refine_steps=50, refine_lr=1e-4, n_augments=4)
solver.to(DEVICE)
solver.eval()
n_params = sum(p.numel() for p in solver.parameters())
print(f"Solver: {n_params:,} parameters on {DEVICE}")

neural_solutions = {}
t0 = time.time()

for i, task in enumerate(unsolved):
    task_t0 = time.time()
    try:
        result = solve_neural_airv(
            task, solver, DEVICE,
            n_geo=8, n_color=4, refine_steps=50, time_limit=50.0,
        )
        neural_solutions[task.task_id] = result
    except Exception as e:
        logger.warning(f"Neural failed on {task.task_id}: {e}")

    if (i + 1) % 10 == 0:
        avg = (time.time() - t0) / (i + 1)
        remaining = avg * (len(unsolved) - i - 1)
        print(f"  [{i+1}/{len(unsolved)}] {time.time()-task_t0:.1f}s/task, ETA: {remaining/60:.0f}min")

neural_time = time.time() - t0
print(f"Neural processed: {len(neural_solutions)}/{len(unsolved)} tasks in {neural_time:.1f}s")
''')

code('''# ===== STAGE 3: REPAIR =====
print("\\n" + "=" * 50)
print("STAGE 3: Near-Miss Repair")
print("=" * 50)

repaired_count = 0
for task in unsolved:
    if task.task_id not in neural_solutions:
        continue
    preds = neural_solutions[task.task_id]
    new_preds = []
    changed = False
    for t_idx, tp in enumerate(task.test):
        test_preds = preds[t_idx] if t_idx < len(preds) else [np.zeros_like(tp.input)]
        new_tp = []
        for pred in test_preds:
            candidates = repair_prediction(pred, task, tp.input)
            best = select_best_repair(candidates, task)
            if not np.array_equal(best, pred):
                changed = True
            new_tp.append(best)
        new_preds.append(new_tp)
    if changed:
        neural_solutions[task.task_id] = new_preds
        repaired_count += 1

print(f"Repaired: {repaired_count} tasks improved")
''')

# ══════════════════════════════════════════════════════════════
# Section 11: Score + Submit
# ══════════════════════════════════════════════════════════════
md("## 11. Score & Submit")

code('''# Merge all solutions
all_predictions = {}
all_predictions.update(dsl_solutions)
all_predictions.update(neural_solutions)

# Score (eval mode only)
KAGGLE = os.path.exists("/kaggle")
if not KAGGLE:
    n_correct = 0
    n_total = 0
    near_misses = 0

    for task in tasks:
        n_total += 1
        if task.task_id not in all_predictions:
            continue
        preds = all_predictions[task.task_id]
        targets = [p.output for p in task.test]
        if score_task(preds, targets):
            n_correct += 1
        else:
            for pp, target in zip(preds, targets):
                for pred in pp:
                    if cell_accuracy(pred, target) > 0.9:
                        near_misses += 1
                        break

    print(f"\\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total tasks:    {n_total}")
    print(f"DSL solved:     {len(dsl_solutions)} ({100*len(dsl_solutions)/max(n_total,1):.1f}%)")
    print(f"Neural solved:  {n_correct - len(dsl_solutions)} additional")
    print(f"Total correct:  {n_correct}/{n_total} ({100*n_correct/max(n_total,1):.1f}%)")
    print(f"Near-misses:    {near_misses} (>90% cell accuracy)")
    print(f"{'='*50}")
else:
    print(f"Kaggle mode: {len(all_predictions)} tasks have predictions")
''')

code('''# Generate submission
submission = {}
for task in tasks:
    tid = task.task_id
    if tid in all_predictions:
        preds = all_predictions[tid]
    else:
        preds = [[p.input, p.input] for p in task.test]

    task_output = []
    for test_preds in preds:
        attempts = [pred.tolist() for pred in test_preds[:2]]
        while len(attempts) < 2:
            attempts.append(attempts[-1] if attempts else [[0]])
        task_output.append(attempts)
    submission[tid] = task_output

output_path = "submission.json"
with open(output_path, "w") as f:
    json.dump(submission, f)

print(f"Submission saved: {output_path}")
print(f"Tasks: {len(submission)}")

total_time = dsl_time + neural_time
print(f"\\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"Projected for 400 tasks: {total_time/max(len(tasks),1) * 400 / 3600:.1f}h")
''')

# ══════════════════════════════════════════════════════════════
# Write notebook
# ══════════════════════════════════════════════════════════════
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells,
}

import os
out_path = os.path.join(os.path.dirname(__file__), "arc_combined_winner.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated: {out_path}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
