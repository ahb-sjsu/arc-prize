"""Analyze eval tasks to understand what patterns they need."""
import numpy as np
from collections import Counter
from dsl_v2 import load_arckit_tasks, grid_bg, grid_colors, extract_objects, analyze_task

_, eval_tasks = load_arckit_tasks()

print(f'Eval tasks: {len(eval_tasks)}')
print()

for tid, task in list(eval_tasks.items())[:20]:
    info = analyze_task(task)
    p = task.train[0]
    bg = grid_bg(p.input)
    in_colors = grid_colors(p.input)
    out_colors = grid_colors(p.output)
    in_objs = extract_objects(p.input, bg)
    out_objs = extract_objects(p.output, bg)

    print(f'--- {tid} ---')
    print(f'  Train pairs: {len(task.train)}, Test pairs: {len(task.test)}')
    print(f'  Input shape: {p.input.shape}, Output shape: {p.output.shape}')
    print(f'  Same shape: {info["same_shape"]}')
    print(f'  H ratio: {info["h_ratio"]}, W ratio: {info["w_ratio"]}')
    print(f'  In colors: {sorted(in_colors)}, Out colors: {sorted(out_colors)}')
    print(f'  New colors in output: {sorted(out_colors - in_colors)}')
    print(f'  In objects: {len(in_objs)} (sizes: {sorted([o["size"] for o in in_objs], reverse=True)[:5]})')
    print(f'  Out objects: {len(out_objs)} (sizes: {sorted([o["size"] for o in out_objs], reverse=True)[:5]})')

    # Show grid content for first pair
    print(f'  Input:')
    for r in range(min(p.input.shape[0], 8)):
        print(f'    {p.input[r].tolist()}')
    if p.input.shape[0] > 8:
        print(f'    ... ({p.input.shape[0]} rows total)')
    print(f'  Output:')
    for r in range(min(p.output.shape[0], 8)):
        print(f'    {p.output[r].tolist()}')
    if p.output.shape[0] > 8:
        print(f'    ... ({p.output.shape[0]} rows total)')
    print()
