"""Summarise the interleaved suite A/Bs (spec/ab_suite.py): per config pooled decode tok/s and the geometric mean of
per-prompt speed-ups against the first config, tokens per verify step (spec), and token agreement against a reference
config (greedy: first divergence; spec: identical outputs).

usage: python analyze_ab.py <prefix> <configs> [--ref fused] [--old runs/tune2_base.jsonl]
  prefix: e.g. ../spec/runs/decode-quick-wins_tune_greedy (reads <prefix>_<config>.jsonl)
"""
import argparse
import json
import math
from collections import defaultdict


def read(path):
    return {r["name"]: r for r in map(json.loads, open(path))}


def rate(rs):
    n = sum(r["generated"] - 1 for r in rs)
    t = sum(r["decode_s"] for r in rs)
    return n / t if t else float("nan")


def first_div(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if d is None and len(a) != len(b):
        d = min(len(a), len(b))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("configs")
    ap.add_argument("--ref", default="fused", help="token-agreement reference config")
    ap.add_argument("--old", default=None, help="an older specdec-format run of the same prompts (token check)")
    args = ap.parse_args()
    cfgs = args.configs.split(",")
    runs = {c: read(f"{args.prefix}_{c}.jsonl") for c in cfgs}
    names = [n for n in runs[cfgs[0]] if all(n in runs[c] for c in cfgs)]
    cats = sorted({runs[cfgs[0]][n]["category"] for n in names})
    spec = "acc" in runs[cfgs[0]][names[0]]
    print(f"{args.prefix}: {len(names)} prompts in all of {cfgs}")
    hdr = f"{'category':14s}" + "".join(f" | {c[:20]:>20s}" for c in cfgs)
    print(hdr)
    for cat in cats + ["ALL"]:
        ns = [n for n in names if cat in ("ALL", runs[cfgs[0]][n]["category"])]
        b = rate([runs[cfgs[0]][n] for n in ns])
        line = f"{cat:14s}"
        for c in cfgs:
            r = rate([runs[c][n] for n in ns])
            cell = f"{r:5.2f} {r / b:5.3f}x"
            if spec:
                acc = sum(sum(runs[c][n]["acc"]) for n in ns) / sum(len(runs[c][n]["acc"]) for n in ns)
                cell += f" {acc:4.2f}"
            line += f" | {cell:>20s}"
        print(line)
    for c in cfgs:
        g = math.exp(sum(math.log(runs[c][n]["tok_s"] / runs[cfgs[0]][n]["tok_s"]) for n in names) / len(names))
        dec = sum(runs[c][n]["decode_s"] for n in names)
        # ms per decode step: verify steps (spec) or generated tokens after the first (greedy)
        steps = sum(runs[c][n]["steps"] if spec else runs[c][n]["generated"] - 1 for n in names)
        print(f"  {c:16s} pooled {rate([runs[c][n] for n in names]):6.2f} tok/s, geo-mean per-prompt x{g:.3f} vs "
              f"{cfgs[0]}, decode {dec:7.1f} s, {sum(runs[c][n]['generated'] for n in names)} tokens, "
              f"{1e3 * dec / steps:6.2f} ms/step")
    ref = runs.get(args.ref)
    if ref:
        for c in cfgs:
            if c == args.ref:
                continue
            divs = [first_div(runs[c][n]["tokens"], ref[n]["tokens"]) for n in names]
            same = sum(d is None for d in divs)
            dd = sorted(d for d in divs if d is not None)
            tot = sum(len(ref[n]["tokens"]) for n in names)
            pref = sum(len(ref[n]["tokens"]) if d is None else d for n, d in zip(names, divs))
            line = (f"  tokens {c} vs {args.ref}: identical on {same}/{len(names)} prompts; first divergence at "
                    f"{dd}; {pref}/{tot} ref tokens before the first divergence")
            if spec:
                eq_acc = sum(runs[c][n]["acc"] == ref[n]["acc"] for n in names)
                line += f"; per-step acceptance identical on {eq_acc}/{len(names)}"
            print(line)
    if args.old:
        old = read(args.old)
        for c in cfgs:
            ns = [n for n in names if n in old]
            same = sum(runs[c][n]["tokens"] == old[n]["tokens"] for n in ns)
            line = f"  tokens {c} vs {args.old}: identical on {same}/{len(ns)}"
            if spec and "acc" in old[ns[0]]:
                line += f"; acceptance identical on {sum(runs[c][n]['acc'] == old[n]['acc'] for n in ns)}/{len(ns)}"
            print(line)


if __name__ == "__main__":
    main()
