"""Summarize bench runs against the greedy baseline: per category and pooled.

tok/s per category = generated tokens (after the first) / decode seconds, summed over the category's prompts.
Speed-up = spec tok/s / baseline tok/s on the same prompts. Pooled: the same over all prompts, plus the geometric mean
of per-prompt speed-ups (every prompt weighs the same). acc = tokens committed per verify step. diverge = prompts whose
spec tokens differ from the baseline's (greedy ties: the first index where they differ).

usage: python analyze.py runs/tune_base.jsonl runs/tune_x.jsonl [runs/tune_y.jsonl ...]
"""
import json
import math
import sys
from collections import defaultdict


def read(path):
    return {r["name"]: r for r in map(json.loads, open(path))}


def rate(rs):
    n = sum(r["generated"] - 1 for r in rs)
    t = sum(r["decode_s"] for r in rs)
    return n / t if t else float("nan")


def main():
    base = read(sys.argv[1])
    runs = [(p, read(p)) for p in sys.argv[2:]]
    cats = sorted({r["category"] for r in base.values()})
    hdr = f"{'category':14s} {'base':>6s}" + "".join(f" | {p.split('/')[-1][:22]:>22s}" for p, _ in runs)
    print(hdr)
    print(f"{'':14s} {'tok/s':>6s}" + " | tok/s   x     acc  div" * len(runs))
    for c in cats + ["ALL"]:
        names = [n for n, r in base.items() if c in ("ALL", r["category"])]
        b = rate([base[n] for n in names])
        line = f"{c:14s} {b:6.1f}"
        for _, run in runs:
            ns = [n for n in names if n in run]
            if not ns:
                line += " | " + " " * 22
                continue
            s = rate([run[n] for n in ns])
            bb = rate([base[n] for n in ns])
            acc = sum(sum(run[n]["acc"]) for n in ns) / max(1, sum(len(run[n]["acc"]) for n in ns))
            div = sum(run[n]["tokens"] != base[n]["tokens"][: len(run[n]["tokens"])] for n in ns)
            line += f" | {s:5.1f} {s / bb:4.2f}x {acc:4.2f} {div:2d}/{len(ns):<2d}"
        print(line)
    for p, run in runs:
        ns = [n for n in base if n in run]
        g = math.exp(sum(math.log(run[n]["tok_s"] / base[n]["tok_s"]) for n in ns) / len(ns))
        divs = []
        for n in ns:
            a, b = run[n]["tokens"], base[n]["tokens"]
            d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if d is not None:
                divs.append(d)
        lk = sum(sum(1 for s in run[n].get("src", []) if s) for n in ns)
        st = sum(len(run[n]["acc"]) for n in ns)
        print(f"{p}: geo-mean per-prompt speed-up {g:.2f}x over {len(ns)} prompts; lookup steps {lk}/{st}; "
              f"first divergence at token {sorted(divs)}")


if __name__ == "__main__":
    main()
