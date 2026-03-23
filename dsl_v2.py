"""
ARC DSL v2 — Comprehensive grid transformation library.

Goal: maximize solve rate through rich primitives + smart search.
"""
import time
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Callable, Set
from dataclasses import dataclass
from itertools import product as iprod

# ============================================================
# Data types
# ============================================================
@dataclass
class ARCPair:
    input: np.ndarray
    output: np.ndarray

@dataclass
class ARCTask:
    task_id: str
    train: List[ARCPair]
    test: List[ARCPair]

def load_arckit_tasks():
    import arckit
    train_set, eval_set = arckit.load_data()
    def convert(tasks):
        result = {}
        for t in tasks:
            train = [ARCPair(np.array(i), np.array(o)) for i, o in t.train]
            test = [ARCPair(np.array(i), np.array(o)) for i, o in t.test]
            result[t.id] = ARCTask(t.id, train, test)
        return result
    return convert(train_set), convert(eval_set)


# ============================================================
# Grid utilities
# ============================================================
def grid_colors(g: np.ndarray) -> set:
    return set(g.flatten().tolist())

def grid_bg(g: np.ndarray) -> int:
    c = Counter(g.flatten().tolist())
    return c.most_common(1)[0][0]

def crop_to_content(g: np.ndarray, bg: int = 0) -> np.ndarray:
    rows, cols = np.where(g != bg)
    if len(rows) == 0: return g.copy()
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()

def extract_objects(g: np.ndarray, bg: int = 0, connectivity: int = 4) -> List[dict]:
    """Flood-fill connected components. connectivity=4 or 8."""
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    objects = []
    if connectivity == 8:
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == bg:
                continue
            color = g[r, c]
            cells = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if 0 <= cr < h and 0 <= cc < w and not visited[cr, cc] and g[cr, cc] == color:
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    for dr, dc in neighbors:
                        stack.append((cr+dr, cc+dc))
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            objects.append({
                'color': int(color), 'cells': cells,
                'bbox': (min(rows), min(cols), max(rows)+1, max(cols)+1),
                'size': len(cells),
            })
    return objects

def extract_objects_multicolor(g: np.ndarray, bg: int = 0) -> List[dict]:
    """Flood-fill connected components allowing any non-bg color."""
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == bg:
                continue
            cells = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if 0 <= cr < h and 0 <= cc < w and not visited[cr, cc] and g[cr, cc] != bg:
                    visited[cr, cc] = True
                    cells.append((cr, cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            bbox = (min(rows), min(cols), max(rows)+1, max(cols)+1)
            subg = g[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
            objects.append({
                'cells': cells, 'bbox': bbox, 'size': len(cells),
                'subgrid': subg,
                'colors': set(g[r,c] for r,c in cells),
            })
    return objects

def get_object_subgrid(g: np.ndarray, obj: dict) -> np.ndarray:
    r0, c0, r1, c1 = obj['bbox']
    return g[r0:r1, c0:c1].copy()

def analyze_task(task: ARCTask) -> dict:
    info = {}
    out_shapes = set()
    in_shapes = set()
    same_shape = True
    for p in task.train:
        if p.input.shape != p.output.shape:
            same_shape = False
        out_shapes.add(p.output.shape)
        in_shapes.add(p.input.shape)
    info['same_shape'] = same_shape
    info['fixed_output_size'] = len(out_shapes) == 1
    info['fixed_input_size'] = len(in_shapes) == 1
    info['output_shape'] = out_shapes.pop() if len(out_shapes) == 1 else None
    # Ratio analysis
    ratios_h = set()
    ratios_w = set()
    for p in task.train:
        ih, iw = p.input.shape
        oh, ow = p.output.shape
        if ih > 0: ratios_h.add(oh / ih)
        if iw > 0: ratios_w.add(ow / iw)
    info['h_ratio'] = ratios_h.pop() if len(ratios_h) == 1 else None
    info['w_ratio'] = ratios_w.pop() if len(ratios_w) == 1 else None
    return info


# ============================================================
# PRIMITIVES — Atomic transforms
# ============================================================

# --- Identity / Geometric ---
def identity(g): return g.copy()
def rot90(g): return np.rot90(g, 1).copy()
def rot180(g): return np.rot90(g, 2).copy()
def rot270(g): return np.rot90(g, 3).copy()
def flip_h(g): return np.fliplr(g).copy()
def flip_v(g): return np.flipud(g).copy()
def transpose(g): return g.T.copy()
def transpose_anti(g): return np.rot90(np.fliplr(g)).copy()

# --- Cropping ---
def crop_bg(g): return crop_to_content(g, grid_bg(g))
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
    c = Counter(g.flatten().tolist())
    mc = c.most_common(2)
    if len(mc) < 2: return g.copy()
    bg, fg = mc[0][0], mc[1][0]
    r = g.copy(); r[g == bg] = fg; r[g == fg] = bg; return r

def invert_colors(g):
    r = g.copy(); mask = r > 0; r[mask] = 9 - r[mask]; return r

def replace_with_most(g):
    bg = grid_bg(g); non_bg = g[g != bg]
    if len(non_bg) == 0: return g.copy()
    mc = Counter(non_bg.tolist()).most_common(1)[0][0]
    r = g.copy(); r[r != bg] = mc; return r

def unique_colors_only(g):
    """Keep only cells whose color appears exactly once."""
    bg = grid_bg(g)
    c = Counter(g.flatten().tolist())
    r = np.full_like(g, bg)
    for color, cnt in c.items():
        if cnt == 1 and color != bg:
            r[g == color] = color
    return r

def remove_color(g):
    """Remove the least common non-bg color."""
    bg = grid_bg(g)
    non_bg = g[g != bg]
    if len(non_bg) == 0: return g.copy()
    c = Counter(non_bg.tolist())
    least = c.most_common()[-1][0]
    r = g.copy(); r[r == least] = bg; return r

# --- Mirroring / symmetry ---
def mirror_h(g): return np.concatenate([g, np.fliplr(g)], axis=1)
def mirror_v(g): return np.concatenate([g, np.flipud(g)], axis=0)
def mirror_both(g):
    top = np.concatenate([g, np.fliplr(g)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)

# --- Border / fill ---
def fill_border(g):
    bg = grid_bg(g); non_bg = g[g != bg]
    if len(non_bg) == 0: return g.copy()
    fc = Counter(non_bg.tolist()).most_common(1)[0][0]
    r = g.copy(); r[0,:] = fc; r[-1,:] = fc; r[:,0] = fc; r[:,-1] = fc; return r

def hollow_interior(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    r[0,:] = g[0,:]; r[-1,:] = g[-1,:]; r[:,0] = g[:,0]; r[:,-1] = g[:,-1]; return r

def remove_border(g):
    """Remove 1-cell border."""
    if g.shape[0] <= 2 or g.shape[1] <= 2: return g.copy()
    return g[1:-1, 1:-1].copy()

# --- Gravity ---
def gravity_down(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]; vals = col[col != bg]
        if len(vals) > 0: r[-len(vals):, c] = vals
    return r

def gravity_up(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]; vals = col[col != bg]
        if len(vals) > 0: r[:len(vals), c] = vals
    return r

def gravity_left(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    for row in range(g.shape[0]):
        vals = g[row][g[row] != bg]
        if len(vals) > 0: r[row, :len(vals)] = vals
    return r

def gravity_right(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    for row in range(g.shape[0]):
        vals = g[row][g[row] != bg]
        if len(vals) > 0: r[row, -len(vals):] = vals
    return r

# --- Extract objects ---
def extract_largest(g):
    bg = grid_bg(g); objs = extract_objects(g, bg)
    if not objs: return g.copy()
    return get_object_subgrid(g, max(objs, key=lambda o: o['size']))

def extract_smallest(g):
    bg = grid_bg(g); objs = extract_objects(g, bg)
    if not objs: return g.copy()
    return get_object_subgrid(g, min(objs, key=lambda o: o['size']))

def extract_largest_mc(g):
    """Extract largest multi-color object."""
    bg = grid_bg(g); objs = extract_objects_multicolor(g, bg)
    if not objs: return g.copy()
    largest = max(objs, key=lambda o: o['size'])
    return largest['subgrid']

def extract_2nd_largest(g):
    bg = grid_bg(g); objs = extract_objects(g, bg)
    if len(objs) < 2: return g.copy()
    objs.sort(key=lambda o: o['size'], reverse=True)
    return get_object_subgrid(g, objs[1])

# --- Binary / mask ---
def to_binary(g):
    bg = grid_bg(g); return (g != bg).astype(int)

def count_colors_grid(g):
    c = Counter(g.flatten().tolist())
    r = np.zeros_like(g)
    for color, cnt in c.items():
        r[g == color] = cnt % 10
    return r

# --- Sort ---
def sort_rows(g): return np.sort(g, axis=1)
def sort_cols(g): return np.sort(g, axis=0)

# --- Grid splitting ---
def top_half(g):
    h = g.shape[0]
    if h < 2: return g.copy()
    return g[:h//2, :].copy()

def bottom_half(g):
    h = g.shape[0]
    if h < 2: return g.copy()
    return g[h//2:, :].copy()

def left_half(g):
    w = g.shape[1]
    if w < 2: return g.copy()
    return g[:, :w//2].copy()

def right_half(g):
    w = g.shape[1]
    if w < 2: return g.copy()
    return g[:, w//2:].copy()

def top_left_quarter(g):
    h, w = g.shape
    return g[:h//2, :w//2].copy()

def top_right_quarter(g):
    h, w = g.shape
    return g[:h//2, w//2:].copy()

def bottom_left_quarter(g):
    h, w = g.shape
    return g[h//2:, :w//2].copy()

def bottom_right_quarter(g):
    h, w = g.shape
    return g[h//2:, w//2:].copy()

# --- Boolean / mask operations on halves ---
def xor_halves_h(g):
    """XOR top and bottom halves (non-bg = 1)."""
    h, w = g.shape
    if h % 2 != 0: return None
    bg = grid_bg(g)
    top = g[:h//2, :]
    bot = g[h//2:, :]
    t_mask = (top != bg)
    b_mask = (bot != bg)
    xor_mask = t_mask ^ b_mask
    # Use top colors where top has content, else bottom
    r = np.full((h//2, w), bg, dtype=g.dtype)
    r[t_mask & ~b_mask] = top[t_mask & ~b_mask]
    r[b_mask & ~t_mask] = bot[b_mask & ~t_mask]
    return r

def xor_halves_v(g):
    """XOR left and right halves."""
    h, w = g.shape
    if w % 2 != 0: return None
    bg = grid_bg(g)
    left = g[:, :w//2]
    right = g[:, w//2:]
    l_mask = (left != bg)
    r_mask = (right != bg)
    result = np.full((h, w//2), bg, dtype=g.dtype)
    result[l_mask & ~r_mask] = left[l_mask & ~r_mask]
    result[r_mask & ~l_mask] = right[r_mask & ~l_mask]
    return result

def and_halves_h(g):
    """AND top and bottom halves."""
    h, w = g.shape
    if h % 2 != 0: return None
    bg = grid_bg(g)
    top = g[:h//2, :]
    bot = g[h//2:, :]
    mask = (top != bg) & (bot != bg)
    r = np.full((h//2, w), bg, dtype=g.dtype)
    r[mask] = top[mask]
    return r

def and_halves_v(g):
    h, w = g.shape
    if w % 2 != 0: return None
    bg = grid_bg(g)
    left = g[:, :w//2]
    right = g[:, w//2:]
    mask = (left != bg) & (right != bg)
    r = np.full((h, w//2), bg, dtype=g.dtype)
    r[mask] = left[mask]
    return r

def or_halves_h(g):
    """OR top and bottom halves (overlay, bottom priority)."""
    h, w = g.shape
    if h % 2 != 0: return None
    bg = grid_bg(g)
    top = g[:h//2, :].copy()
    bot = g[h//2:, :]
    mask = (bot != bg)
    top[mask] = bot[mask]
    return top

def or_halves_v(g):
    h, w = g.shape
    if w % 2 != 0: return None
    bg = grid_bg(g)
    left = g[:, :w//2].copy()
    right = g[:, w//2:]
    mask = (right != bg)
    left[mask] = right[mask]
    return left

def diff_halves_h(g):
    """Difference: cells in top but NOT in bottom."""
    h, w = g.shape
    if h % 2 != 0: return None
    bg = grid_bg(g)
    top = g[:h//2, :]
    bot = g[h//2:, :]
    r = np.full((h//2, w), bg, dtype=g.dtype)
    mask = (top != bg) & (bot == bg)
    r[mask] = top[mask]
    return r

# --- Symmetry completion ---
def complete_horizontal_symmetry(g):
    """Make grid horizontally symmetric by mirroring left half over right."""
    h, w = g.shape
    r = g.copy()
    for i in range(h):
        for j in range(w // 2):
            mirror_j = w - 1 - j
            if r[i, j] != 0 and r[i, mirror_j] == 0:
                r[i, mirror_j] = r[i, j]
            elif r[i, mirror_j] != 0 and r[i, j] == 0:
                r[i, j] = r[i, mirror_j]
    return r

def complete_vertical_symmetry(g):
    """Make grid vertically symmetric by mirroring top half over bottom."""
    h, w = g.shape
    r = g.copy()
    for i in range(h // 2):
        mirror_i = h - 1 - i
        for j in range(w):
            if r[i, j] != 0 and r[mirror_i, j] == 0:
                r[mirror_i, j] = r[i, j]
            elif r[mirror_i, j] != 0 and r[i, j] == 0:
                r[i, j] = r[mirror_i, j]
    return r

def complete_4fold_symmetry(g):
    """Make grid 4-fold symmetric."""
    r = complete_horizontal_symmetry(g)
    r = complete_vertical_symmetry(r)
    return r

# --- Grid dividers ---
def split_by_color_line_h(g):
    """Split grid at horizontal line of single color, return top part."""
    bg = grid_bg(g)
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        if len(set(row.tolist())) == 1 and row[0] != bg:
            if r > 0:
                return g[:r, :].copy()
    return None

def split_by_color_line_h_bottom(g):
    """Split grid at horizontal line of single color, return bottom part."""
    bg = grid_bg(g)
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        if len(set(row.tolist())) == 1 and row[0] != bg:
            if r < h - 1:
                return g[r+1:, :].copy()
    return None

def split_by_color_line_v(g):
    """Split grid at vertical line of single color, return left part."""
    bg = grid_bg(g)
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        if len(set(col.tolist())) == 1 and col[0] != bg:
            if c > 0:
                return g[:, :c].copy()
    return None

def split_by_color_line_v_right(g):
    """Split grid at vertical line of single color, return right part."""
    bg = grid_bg(g)
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        if len(set(col.tolist())) == 1 and col[0] != bg:
            if c < w - 1:
                return g[:, c+1:].copy()
    return None

# --- Overlay operations for split grids ---
def overlay_split_h(g):
    """Split at horizontal divider, overlay top on bottom."""
    bg = grid_bg(g)
    h, w = g.shape
    for r in range(1, h - 1):
        row = g[r, :]
        if len(set(row.tolist())) == 1 and row[0] != bg:
            top = g[:r, :]
            bot = g[r+1:, :]
            if top.shape == bot.shape:
                result = bot.copy()
                mask = top != bg
                result[mask] = top[mask]
                return result
    return None

def overlay_split_v(g):
    """Split at vertical divider, overlay left on right."""
    bg = grid_bg(g)
    h, w = g.shape
    for c in range(1, w - 1):
        col = g[:, c]
        if len(set(col.tolist())) == 1 and col[0] != bg:
            left = g[:, :c]
            right = g[:, c+1:]
            if left.shape == right.shape:
                result = right.copy()
                mask = left != bg
                result[mask] = left[mask]
                return result
    return None

def xor_split_h(g):
    """Split at horizontal divider, XOR top and bottom."""
    bg = grid_bg(g)
    h, w = g.shape
    for r in range(1, h - 1):
        row = g[r, :]
        if len(set(row.tolist())) == 1 and row[0] != bg:
            top = g[:r, :]
            bot = g[r+1:, :]
            if top.shape == bot.shape:
                t_mask = top != bg
                b_mask = bot != bg
                result = np.full_like(top, bg)
                xor = t_mask ^ b_mask
                result[xor & t_mask] = top[xor & t_mask]
                result[xor & b_mask] = bot[xor & b_mask]
                return result
    return None

def and_split_h(g):
    """Split at horizontal divider, AND top and bottom."""
    bg = grid_bg(g)
    h, w = g.shape
    for r in range(1, h - 1):
        row = g[r, :]
        if len(set(row.tolist())) == 1 and row[0] != bg:
            top = g[:r, :]
            bot = g[r+1:, :]
            if top.shape == bot.shape:
                mask = (top != bg) & (bot != bg)
                result = np.full_like(top, bg)
                result[mask] = top[mask]
                return result
    return None

# --- Flood fill enclosed regions ---
def fill_enclosed_regions(g):
    """Fill enclosed background regions with the surrounding color."""
    bg = grid_bg(g)
    h, w = g.shape
    # Find bg cells not connected to border
    visited = np.zeros_like(g, dtype=bool)
    # BFS from all border bg cells
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if g[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if g[r, c] == bg and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    # Fill unvisited bg cells
    r = g.copy()
    for i in range(h):
        for j in range(w):
            if g[i, j] == bg and not visited[i, j]:
                # Find nearest non-bg neighbor color
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = i+dr, j+dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != bg:
                        r[i, j] = g[nr, nc]
                        break
    return r

# --- Repeating pattern detection ---
def detect_and_tile(g):
    """If grid is a repeated pattern, extract the minimal tile."""
    h, w = g.shape
    for th in range(1, h//2 + 1):
        if h % th != 0: continue
        for tw in range(1, w//2 + 1):
            if w % tw != 0: continue
            tile = g[:th, :tw]
            tiled = np.tile(tile, (h//th, w//tw))
            if np.array_equal(tiled, g):
                return tile
    return None

def get_minimal_period_h(g):
    """Find minimal vertical period."""
    h, w = g.shape
    for p in range(1, h):
        if h % p != 0: continue
        tile = g[:p, :]
        if np.array_equal(np.tile(tile, (h//p, 1)), g):
            return tile
    return g.copy()

def get_minimal_period_v(g):
    """Find minimal horizontal period."""
    h, w = g.shape
    for p in range(1, w):
        if w % p != 0: continue
        tile = g[:, :p]
        if np.array_equal(np.tile(tile, (1, w//p)), g):
            return tile
    return g.copy()

# --- Row/column deduplication ---
def unique_rows(g):
    """Keep only unique rows (first occurrence)."""
    seen = []
    result = []
    for r in range(g.shape[0]):
        row = tuple(g[r].tolist())
        if row not in seen:
            seen.append(row)
            result.append(g[r])
    if not result: return g.copy()
    return np.array(result)

def unique_cols(g):
    return unique_rows(g.T).T

# --- Diagonal operations ---
def extract_main_diagonal(g):
    """Extract main diagonal as a row."""
    n = min(g.shape)
    return np.array([g[i, i] for i in range(n)]).reshape(1, -1)

def extract_anti_diagonal(g):
    n = min(g.shape)
    w = g.shape[1]
    return np.array([g[i, w-1-i] for i in range(n)]).reshape(1, -1)

# --- Most common subgrid ---
def most_common_row(g):
    """Return the most repeated row, tiled to original height."""
    rows = [tuple(g[r].tolist()) for r in range(g.shape[0])]
    c = Counter(rows)
    mc = c.most_common(1)[0][0]
    return np.tile(np.array(mc), (g.shape[0], 1))

def most_common_col(g):
    return most_common_row(g.T).T


# ============================================================
# Build primitive catalog
# ============================================================
PRIMITIVES = {}
_g = dict(globals())
for name, obj in _g.items():
    if callable(obj) and not name.startswith('_') and name not in (
        'grid_colors', 'grid_bg', 'crop_to_content', 'extract_objects',
        'extract_objects_multicolor', 'get_object_subgrid', 'analyze_task',
        'load_arckit_tasks', 'apply_program', 'verify_program', 'verify_on_test',
        'search_programs', 'make_task_specific_fns', 'evaluate_dsl',
        'ARCPair', 'ARCTask', 'iprod',
    ):
        # Only include functions that take a single ndarray arg
        import inspect
        try:
            sig = inspect.signature(obj)
            params = [p for p in sig.parameters.values()
                     if p.default is inspect.Parameter.empty]
            if len(params) == 1:
                PRIMITIVES[name] = obj
        except (ValueError, TypeError):
            pass

# Remove non-primitive entries that slipped through
for k in list(PRIMITIVES.keys()):
    if k in ('dataclass', 'field', 'convert', 'time', 'Counter',
             'defaultdict', 'np', 'List', 'Dict', 'Optional', 'Tuple',
             'Callable', 'Set'):
        del PRIMITIVES[k]

print(f'DSL v2 primitives: {len(PRIMITIVES)}')


# ============================================================
# Task-specific primitives
# ============================================================
def make_task_specific_fns(task: ARCTask) -> Dict[str, Callable]:
    fns = {}

    # Color replacements
    all_colors = set()
    for p in task.train:
        all_colors |= grid_colors(p.input) | grid_colors(p.output)
    for src in sorted(all_colors):
        for dst in sorted(all_colors):
            if src == dst: continue
            def make_fn(s, d):
                def fn(g):
                    r = g.copy(); r[g == s] = d; return r
                return fn
            fns[f'recolor_{src}_to_{dst}'] = make_fn(src, dst)

    # Learned color mapping: if there's a consistent mapping across all train pairs
    # (for same-shape tasks)
    if all(p.input.shape == p.output.shape for p in task.train):
        # Try to find a global color permutation
        mappings = defaultdict(set)
        for p in task.train:
            for i in range(p.input.shape[0]):
                for j in range(p.input.shape[1]):
                    mappings[int(p.input[i,j])].add(int(p.output[i,j]))
        # Check if each input color maps to exactly one output color
        if all(len(v) == 1 for v in mappings.values()):
            cmap = {k: v.pop() for k, v in mappings.items()}
            def color_map_fn(g, cm=dict(cmap)):
                r = g.copy()
                for src, dst in cm.items():
                    r[g == src] = dst
                return r
            fns['learned_color_map'] = color_map_fn

    # Fixed output
    outputs = [p.output for p in task.train]
    if all(np.array_equal(outputs[0], o) for o in outputs):
        fixed = outputs[0].copy()
        fns['fixed_output'] = lambda g, f=fixed: f.copy()

    # Size-based crops
    pair = task.train[0]
    oh, ow = pair.output.shape
    ih, iw = pair.input.shape
    if oh <= ih and ow <= iw:
        def crop_tl(g, h=oh, w=ow): return g[:h, :w].copy()
        fns[f'crop_tl_{oh}x{ow}'] = crop_tl
        def crop_br(g, h=oh, w=ow): return g[-h:, -w:].copy()
        fns[f'crop_br_{oh}x{ow}'] = crop_br
        def crop_center(g, h=oh, w=ow):
            gh, gw = g.shape
            r0 = (gh - h) // 2; c0 = (gw - w) // 2
            return g[r0:r0+h, c0:c0+w].copy()
        fns[f'crop_center_{oh}x{ow}'] = crop_center

    # If output is always a specific number of rows/cols from input
    if len(task.train) >= 2:
        # Check row deletion pattern
        row_diffs = set()
        col_diffs = set()
        for p in task.train:
            row_diffs.add(p.input.shape[0] - p.output.shape[0])
            col_diffs.add(p.input.shape[1] - p.output.shape[1])
        if len(row_diffs) == 1 and len(col_diffs) == 1:
            rd = row_diffs.pop(); cd = col_diffs.pop()
            if rd > 0 or cd > 0:
                def trim(g, r=rd, c=cd):
                    gh, gw = g.shape
                    r0 = r // 2; r1 = gh - (r - r // 2)
                    c0 = c // 2; c1 = gw - (c - c // 2)
                    return g[r0:r1, c0:c1].copy()
                fns[f'trim_{rd}r_{cd}c'] = trim

    return fns


# ============================================================
# Search engine
# ============================================================
def apply_program(g: np.ndarray, program: list) -> Optional[np.ndarray]:
    result = g
    for fn in program:
        try:
            result = fn(result)
            if result is None or not isinstance(result, np.ndarray): return None
            if result.size == 0 or result.ndim != 2: return None
            if max(result.shape) > 30: return None
        except: return None
    return result

def verify_program(task: ARCTask, program: list) -> bool:
    for pair in task.train:
        result = apply_program(pair.input, program)
        if result is None or not np.array_equal(result, pair.output):
            return False
    return True

def verify_on_test(task: ARCTask, program: list) -> bool:
    for pair in task.test:
        if pair.output is None: return False
        result = apply_program(pair.input, program)
        if result is None or not np.array_equal(result, pair.output):
            return False
    return True

def search_programs(task: ARCTask, max_depth: int = 3, time_limit: float = 30.0):
    start = time.time()
    all_prims = dict(PRIMITIVES)
    all_prims.update(make_task_specific_fns(task))
    prim_names = list(all_prims.keys())
    prim_fns = [all_prims[n] for n in prim_names]
    n = len(prim_names)

    # Depth 1
    for i in range(n):
        if time.time() - start > time_limit: return None
        if verify_program(task, [prim_fns[i]]):
            return ([prim_names[i]], [prim_fns[i]])
    if max_depth < 2: return None

    # Depth 2
    for i in range(n):
        if time.time() - start > time_limit: return None
        test_r = apply_program(task.train[0].input, [prim_fns[i]])
        if test_r is None: continue
        for j in range(n):
            if time.time() - start > time_limit: return None
            if verify_program(task, [prim_fns[i], prim_fns[j]]):
                return ([prim_names[i], prim_names[j]], [prim_fns[i], prim_fns[j]])
    if max_depth < 3: return None

    # Depth 3: use base primitives only for the middle step to limit explosion
    for i in range(n):
        if time.time() - start > time_limit: return None
        r1 = apply_program(task.train[0].input, [prim_fns[i]])
        if r1 is None: continue
        for j in range(n):
            if time.time() - start > time_limit: return None
            r2 = apply_program(r1, [prim_fns[j]])
            if r2 is None: continue
            # Only try base primitives at depth 3 to limit search space
            base_names = list(PRIMITIVES.keys())
            base_fns = [PRIMITIVES[nm] for nm in base_names]
            for k in range(len(base_fns)):
                if time.time() - start > time_limit: return None
                if verify_program(task, [prim_fns[i], prim_fns[j], base_fns[k]]):
                    return ([prim_names[i], prim_names[j], base_names[k]],
                            [prim_fns[i], prim_fns[j], base_fns[k]])
    return None


# ============================================================
# Evaluation
# ============================================================
def evaluate_dsl(tasks: Dict[str, ARCTask], time_per_task: float = 30.0,
                 max_depth: int = 2, label: str = 'EVAL'):
    task_list = list(tasks.values())
    solved_train = 0
    solved_test = 0
    examples = []
    t0 = time.time()

    for i, task in enumerate(task_list):
        result = search_programs(task, max_depth=max_depth, time_limit=time_per_task)
        if result is not None:
            names, fns = result
            solved_train += 1
            if verify_on_test(task, fns):
                solved_test += 1
            if len(examples) < 15:
                examples.append((task.task_id, names))
        if (i + 1) % 100 == 0 or i + 1 == len(task_list):
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(task_list)}] Train-verified: {solved_train}, '
                  f'Test-verified: {solved_test}, Time: {elapsed:.0f}s')

    elapsed = time.time() - t0
    rate_train = solved_train / len(task_list) if task_list else 0
    rate_test = solved_test / len(task_list) if task_list else 0
    print(f'\n{label} Results:')
    print(f'  Train-verified: {solved_train}/{len(task_list)} = {rate_train:.1%}')
    print(f'  Test-verified:  {solved_test}/{len(task_list)} = {rate_test:.1%}')
    print(f'  Time: {elapsed:.0f}s ({elapsed/len(task_list):.1f}s/task avg)')
    if examples:
        print(f'  Solved examples:')
        for tid, names in examples:
            print(f'    {tid}: {" -> ".join(names)}')
    return solved_train, solved_test, len(task_list)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Loading data...')
    train_tasks, eval_tasks = load_arckit_tasks()
    print(f'Train: {len(train_tasks)}, Eval: {len(eval_tasks)}')

    print('\n=== TRAIN (depth 2, 30s/task) ===')
    evaluate_dsl(train_tasks, time_per_task=30.0, max_depth=2, label='TRAIN')

    print('\n=== EVAL (depth 2, 30s/task) ===')
    evaluate_dsl(eval_tasks, time_per_task=30.0, max_depth=2, label='EVAL')
