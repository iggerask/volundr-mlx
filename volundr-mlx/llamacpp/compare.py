#!/usr/bin/env python3
"""Compare greedy replies from two chat.py result files: identical or not, and where they first differ.

usage: compare.py A.json[,A2.json] B.json
"""
import json, sys
from pathlib import Path


def load(arg):
    out = {}
    for f in arg.split(","):
        out.update(json.loads(Path(f).read_text()))
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
for name in [n for n in a if n in b]:
    x, y = a[name][-1]["content"] or "", b[name][-1]["content"] or ""
    i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    words = len(x[:i].split())
    if x == y:
        print(f"{name}: identical ({len(x)} chars)")
    else:
        print(f"{name}: first difference at char {i} of {len(x)}/{len(y)} (~{words} words in)")
        print(f"   A: ...{x[max(0, i - 60):i + 80]!r}")
        print(f"   B: ...{y[max(0, i - 60):i + 80]!r}")
