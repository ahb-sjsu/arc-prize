"""Quick test: DSL solve rate on ARC training + eval tasks."""
import time
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass

# ---- Data loading ----
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

# ---- Grid utilities ----
def grid_colors(g): return set(g.flatten().tolist())
def grid_bg(g):
    c = Counter(g.flatten().tolist())
    return c.most_common(1)[0][0]

def extract_objects(g, bg=0):
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    objects = []
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
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            objects.append({
                'color': int(color), 'cells': cells,
                'bbox': (min(rows), min(cols), max(rows)+1, max(cols)+1),
                'size': len(cells),
            })
    return objects

def crop_to_content(g, bg=0):
    rows, cols = np.where(g != bg)
    if len(rows) == 0: return g.copy()
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()

def get_object_subgrid(g, obj):
    r0, c0, r1, c1 = obj['bbox']
    return g[r0:r1, c0:c1].copy()

def analyze_task(task):
    info = {'same_shape': True, 'fixed_output_size': True}
    out_shapes = set()
    for p in task.train:
        if p.input.shape != p.output.shape:
            info['same_shape'] = False
        out_shapes.add(p.output.shape)
    info['fixed_output_size'] = len(out_shapes) == 1
    info['output_shape'] = out_shapes.pop() if len(out_shapes) == 1 else None
    return info

# ---- DSL Primitives ----
def identity(g): return g.copy()
def rot90(g): return np.rot90(g, 1).copy()
def rot180(g): return np.rot90(g, 2).copy()
def rot270(g): return np.rot90(g, 3).copy()
def flip_h(g): return np.fliplr(g).copy()
def flip_v(g): return np.flipud(g).copy()
def transpose(g): return g.T.copy()
def transpose_anti(g): return np.rot90(g.T, 2).copy()
def crop_bg(g): return crop_to_content(g, grid_bg(g))
def crop_bg0(g): return crop_to_content(g, 0)
def tile_2x2(g): return np.tile(g, (2, 2))
def tile_3x3(g): return np.tile(g, (3, 3))
def tile_2x1(g): return np.tile(g, (2, 1))
def tile_1x2(g): return np.tile(g, (1, 2))
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
def swap_bg_most(g):
    c = Counter(g.flatten().tolist())
    mc = c.most_common(2)
    if len(mc) < 2: return g.copy()
    bg, fg = mc[0][0], mc[1][0]
    r = g.copy(); r[g == bg] = fg; r[g == fg] = bg
    return r
def invert_colors(g):
    r = g.copy(); mask = r > 0; r[mask] = 9 - r[mask]; return r
def replace_with_most(g):
    bg = grid_bg(g); non_bg = g[g != bg]
    if len(non_bg) == 0: return g.copy()
    mc = Counter(non_bg.tolist()).most_common(1)[0][0]
    r = g.copy(); r[r != bg] = mc; return r
def mirror_h(g): return np.concatenate([g, np.fliplr(g)], axis=1)
def mirror_v(g): return np.concatenate([g, np.flipud(g)], axis=0)
def mirror_both(g):
    top = np.concatenate([g, np.fliplr(g)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)
def fill_border(g):
    bg = grid_bg(g); non_bg = g[g != bg]
    if len(non_bg) == 0: return g.copy()
    fc = Counter(non_bg.tolist()).most_common(1)[0][0]
    r = g.copy(); r[0,:] = fc; r[-1,:] = fc; r[:,0] = fc; r[:,-1] = fc; return r
def hollow_interior(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    r[0,:] = g[0,:]; r[-1,:] = g[-1,:]; r[:,0] = g[:,0]; r[:,-1] = g[:,-1]; return r
def gravity_down(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]; non_bg = col[col != bg]
        if len(non_bg) > 0: r[-len(non_bg):, c] = non_bg
    return r
def gravity_left(g):
    bg = grid_bg(g); r = np.full_like(g, bg)
    for row in range(g.shape[0]):
        vals = g[row][g[row] != bg]
        if len(vals) > 0: r[row, :len(vals)] = vals
    return r
def extract_largest(g):
    bg = grid_bg(g); objs = extract_objects(g, bg)
    if not objs: return g.copy()
    return get_object_subgrid(g, max(objs, key=lambda o: o['size']))
def extract_smallest(g):
    bg = grid_bg(g); objs = extract_objects(g, bg)
    if not objs: return g.copy()
    return get_object_subgrid(g, min(objs, key=lambda o: o['size']))
def to_binary(g):
    bg = grid_bg(g); return (g != bg).astype(int)
def sort_rows(g): return np.sort(g, axis=1)
def sort_cols(g): return np.sort(g, axis=0)

PRIMITIVES = {
    'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270,
    'flip_h': flip_h, 'flip_v': flip_v, 'transpose': transpose, 'transpose_anti': transpose_anti,
    'crop_bg': crop_bg, 'crop_bg0': crop_bg0,
    'tile_2x2': tile_2x2, 'tile_3x3': tile_3x3, 'tile_2x1': tile_2x1, 'tile_1x2': tile_1x2,
    'upscale_2x': upscale_2x, 'upscale_3x': upscale_3x,
    'downscale_2x': downscale_2x, 'downscale_3x': downscale_3x,
    'swap_bg_most': swap_bg_most, 'invert_colors': invert_colors, 'replace_with_most': replace_with_most,
    'mirror_h': mirror_h, 'mirror_v': mirror_v, 'mirror_both': mirror_both,
    'fill_border': fill_border, 'hollow_interior': hollow_interior,
    'gravity_down': gravity_down, 'gravity_left': gravity_left,
    'extract_largest': extract_largest, 'extract_smallest': extract_smallest,
    'to_binary': to_binary, 'sort_rows': sort_rows, 'sort_cols': sort_cols,
}

# ---- Parameterized primitives ----
def make_color_replace_fns(task):
    fns = {}
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
    return fns

def make_input_output_ops(task):
    fns = {}
    pair = task.train[0]
    ih, iw = pair.input.shape
    oh, ow = pair.output.shape
    if oh <= ih and ow <= iw:
        def crop_tl(g, h=oh, w=ow): return g[:h, :w].copy()
        fns[f'crop_to_{oh}x{ow}'] = crop_tl
        def crop_br(g, h=oh, w=ow): return g[-h:, -w:].copy()
        fns[f'crop_br_{oh}x{ow}'] = crop_br
    outputs = [p.output for p in task.train]
    if all(np.array_equal(outputs[0], o) for o in outputs):
        fixed = outputs[0].copy()
        fns['fixed_output'] = lambda g, f=fixed: f.copy()
    return fns

# ---- Search engine ----
def apply_program(g, program):
    result = g
    for fn in program:
        try:
            result = fn(result)
            if result is None or not isinstance(result, np.ndarray): return None
            if result.size == 0 or result.ndim != 2: return None
            if max(result.shape) > 30: return None
        except: return None
    return result

def verify_program(task, program):
    for pair in task.train:
        result = apply_program(pair.input, program)
        if result is None or not np.array_equal(result, pair.output):
            return False
    return True

def verify_on_test(task, program):
    """Check if program also works on test pairs (for scoring)."""
    for pair in task.test:
        if pair.output is None: return False
        result = apply_program(pair.input, program)
        if result is None or not np.array_equal(result, pair.output):
            return False
    return True

def search_programs(task, max_depth=2, time_limit=30.0):
    start = time.time()
    all_prims = dict(PRIMITIVES)
    all_prims.update(make_color_replace_fns(task))
    all_prims.update(make_input_output_ops(task))
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

    # Depth 3 (base prims only)
    base_names = list(PRIMITIVES.keys())
    base_fns = [PRIMITIVES[n] for n in base_names]
    nb = len(base_names)
    for i in range(nb):
        if time.time() - start > time_limit: return None
        r1 = apply_program(task.train[0].input, [base_fns[i]])
        if r1 is None: continue
        for j in range(nb):
            r2 = apply_program(r1, [base_fns[j]])
            if r2 is None: continue
            if time.time() - start > time_limit: return None
            for k in range(nb):
                if time.time() - start > time_limit: return None
                if verify_program(task, [base_fns[i], base_fns[j], base_fns[k]]):
                    return ([base_names[i], base_names[j], base_names[k]],
                            [base_fns[i], base_fns[j], base_fns[k]])
    return None

# ---- Main ----
if __name__ == '__main__':
    print('Loading data...')
    train_tasks, eval_tasks = load_arckit_tasks()
    print(f'Train: {len(train_tasks)}, Eval: {len(eval_tasks)}')

    for label, tasks in [('TRAIN', train_tasks), ('EVAL', eval_tasks)]:
        print(f'\n=== {label} ({len(tasks)} tasks) ===')
        solved_train = 0  # verified on training pairs
        solved_test = 0   # also correct on test pairs
        task_list = list(tasks.values())
        examples = []
        t0 = time.time()

        for i, task in enumerate(task_list):
            result = search_programs(task, max_depth=2, time_limit=15.0)
            if result is not None:
                names, fns = result
                solved_train += 1
                if verify_on_test(task, fns):
                    solved_test += 1
                if len(examples) < 10:
                    examples.append((task.task_id, names))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f'  [{i+1}/{len(task_list)}] Train-verified: {solved_train}, '
                      f'Test-verified: {solved_test}, Time: {elapsed:.0f}s')

        elapsed = time.time() - t0
        print(f'{label} Results:')
        print(f'  Train-verified: {solved_train}/{len(task_list)} = {solved_train/len(task_list):.1%}')
        print(f'  Test-verified:  {solved_test}/{len(task_list)} = {solved_test/len(task_list):.1%}')
        print(f'  Time: {elapsed:.0f}s ({elapsed/len(task_list):.1f}s/task)')
        if examples:
            print(f'  Examples:')
            for tid, names in examples[:5]:
                print(f'    {tid}: {" -> ".join(names)}')
