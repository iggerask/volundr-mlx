"""Per-prompt breakdown of compare_logits.py: top-1 and KL(p_ref || p_var) per prompt (rows in prompts.json order).

usage: python3 kl_by_prompt.py ../parity/prompts.json ../parity/mlx.bin ../parity/mlx_w4a8.bin
"""
import json
import sys

import numpy as np

V = 248077
prompts = json.load(open(sys.argv[1]))
a, b = (np.memmap(f, dtype=np.uint16, mode="r") for f in sys.argv[2:4])
f = lambda x: (x.astype(np.uint32) << 16).view(np.float32).astype(np.float64)  # noqa: E731
row = 0
for p in prompts:
    n = len(p["force"])
    kls, top1 = [], 0
    for i in range(row, row + n):
        x, y = f(a[i * V:(i + 1) * V]), f(b[i * V:(i + 1) * V])
        lx = x.max() + np.log(np.exp(x - x.max()).sum())
        ly = y.max() + np.log(np.exp(y - y.max()).sum())
        px = np.exp(x - lx)
        kls.append(float((px * ((x - lx) - (y - ly))).sum()))
        top1 += int(x.argmax() == y.argmax())
    k = np.array(kls)
    print(f"{p['name']:16s} {len(p['ids']):6d} prompt tokens: top-1 {top1}/{n}, KL mean {k.mean():.2e} "
          f"p50 {np.median(k):.2e} max {k.max():.2e}; first row (prefill) {k[0]:.2e}")
    row += n
