"""Per category: decode tok/s, ms per verify step, tokens per step, for each spec run; tokens identical between runs?
usage: stepms.py base.jsonl runA.jsonl [runB.jsonl ...]"""
import json
import sys
from collections import defaultdict


def read(p):
    return {r["name"]: r for r in map(json.loads, open(p))}


base = read(sys.argv[1])
runs = [(p.split("/")[-1], read(p)) for p in sys.argv[2:]]
cats = sorted({r["category"] for r in base.values()})
print(f"{'category':14s} {'base':>6s} " + " ".join(f"| {n[:26]:>26s}" for n, _ in runs))
for c in cats + ["ALL", "long+agent>4K"]:
    if c == "ALL":
        names = list(base)
    elif c == "long+agent>4K":
        names = [n for n, r in base.items() if r["category"] in ("long", "agent") and r["prompt_tokens"] > 4096]
    else:
        names = [n for n, r in base.items() if r["category"] == c]
    b_tok = sum(base[n]["generated"] - 1 for n in names) / sum(base[n]["decode_s"] for n in names)
    line = f"{c:14s} {b_tok:6.2f} "
    for _, run in runs:
        ns = [n for n in names if n in run]
        if not ns:
            line += "| " + " " * 26
            continue
        gen = sum(run[n]["generated"] - 1 for n in ns)
        t = sum(run[n]["decode_s"] for n in ns)
        steps = sum(run[n]["steps"] for n in ns)
        acc = sum(sum(run[n]["acc"]) for n in ns) / steps
        bb = sum(base[n]["generated"] - 1 for n in ns) / sum(base[n]["decode_s"] for n in ns)
        line += f"| {gen / t:5.2f} {gen / t / bb:4.2f}x {1e3 * t / steps:5.1f}ms {acc:4.2f} "
    print(line)
# token identity between consecutive runs
for (na, a), (nb, b) in zip(runs, runs[1:]):
    same = sum(a[n]["tokens"] == b[n]["tokens"] for n in a if n in b)
    print(f"{na} vs {nb}: identical tokens on {same}/{len([n for n in a if n in b])} prompts")
for n_, run in runs:
    div = sum(run[n]["tokens"] != base[n]["tokens"][: len(run[n]["tokens"])] for n in run if n in base)
    print(f"{n_}: differs from greedy on {div}/{len(run)} prompts")
