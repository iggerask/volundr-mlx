"""Summary of the combined-test spec runs: per config and prompt group, pooled decode tok/s, median and pooled ms per
verify step, tokens per step, ratios against the first config; token identity and per-step acceptance against the
first config and against older reference runs.

usage: python analyze.py <prefix> <configs> [--ref name=path ...]
  prefix: e.g. ../spec/runs/combined-test_tune_spec (reads <prefix>_<config>.jsonl)
  --ref config=path: compare that config's tokens / acceptance with an older run of the same prompts
"""
import argparse
import json
import statistics


def read(path):
    return {r["name"]: r for r in map(json.loads, open(path))}


def stats(rs):
    gen = sum(r["generated"] - 1 for r in rs)
    t = sum(r["decode_s"] for r in rs)
    steps = sum(r["steps"] for r in rs)
    return {"tok_s": gen / t, "ms_step": 1e3 * t / steps,
            "ms_step_med": statistics.median(1e3 * r["decode_s"] / r["steps"] for r in rs),
            "tok_step": sum(sum(r["acc"]) for r in rs) / sum(len(r["acc"]) for r in rs)}


def first_div(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return min(len(a), len(b)) if d is None and len(a) != len(b) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("configs")
    ap.add_argument("--ref", action="append", default=[])
    args = ap.parse_args()
    cfgs = args.configs.split(",")
    runs = {c: read(f"{args.prefix}_{c}.jsonl") for c in cfgs}
    b = cfgs[0]
    names = [n for n in runs[b] if all(n in runs[c] for c in cfgs)]
    print(f"{args.prefix}: {len(names)} prompts in all of {cfgs}")
    for c in cfgs:
        flags = {(r.get("fused"), r.get("views"), r.get("bcsa_verify")) for r in runs[c].values()}
        print(f"  {c}: (fused, views, bcsa_verify) = {sorted(flags, key=str)}")
    groups = {
        "all": names,
        "short (<4K)": [n for n in names if runs[b][n]["prompt_tokens"] < 4096],
        "long+agent": [n for n in names if runs[b][n]["category"] in ("long", "agent")],
        "long": [n for n in names if runs[b][n]["category"] == "long"],
        "agent": [n for n in names if runs[b][n]["category"] == "agent"],
    }
    print("\n| group | prompts | config | tok/s | x baseline | median ms/step | pooled ms/step | tokens/step |")
    print("|---|---|---|---|---|---|---|---|")
    for g, ns in groups.items():
        s0 = stats([runs[b][n] for n in ns])
        for c in cfgs:
            s = stats([runs[c][n] for n in ns])
            print(f"| {g} | {len(ns)} | {c} | {s['tok_s']:.2f} | {s['tok_s'] / s0['tok_s']:.3f} | "
                  f"{s['ms_step_med']:.1f} | {s['ms_step']:.1f} | {s['tok_step']:.2f} |")
    print()
    for c in cfgs[1:]:
        divs = [first_div(runs[c][n]["tokens"], runs[b][n]["tokens"]) for n in names]
        same = sum(d is None for d in divs)
        acc = sum(runs[c][n]["acc"] == runs[b][n]["acc"] for n in names)
        print(f"  {c} vs {b}: tokens identical on {same}/{len(names)}, per-step acceptance identical on "
              f"{acc}/{len(names)}; first divergences at {sorted(d for d in divs if d is not None)}")
    for c in cfgs[1:]:
        for g in ("short (<4K)", "long+agent"):
            ns = groups[g]
            same = sum(runs[c][n]["tokens"] == runs[b][n]["tokens"] for n in ns)
            print(f"  {c} vs {b}, {g}: tokens identical on {same}/{len(ns)}")
    for spec in args.ref:
        c, path = spec.split("=", 1)
        old = read(path)
        ns = [n for n in names if n in old]
        same = sum(runs[c][n]["tokens"] == old[n]["tokens"] for n in ns)
        acc = sum(runs[c][n]["acc"] == old[n]["acc"] for n in ns) if "acc" in old[ns[0]] else None
        print(f"  {c} vs {path}: tokens identical on {same}/{len(ns)}"
              + ("" if acc is None else f", per-step acceptance identical on {acc}/{len(ns)}"))


if __name__ == "__main__":
    main()
