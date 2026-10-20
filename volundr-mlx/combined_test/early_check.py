"""Progress check of a running combined-test A/B: for the prompts done so far, does baseline reproduce main's run
(tokens and per-step acceptance), and how do the configs compare in tok/s.
usage: python early_check.py <prefix> <ref.jsonl> [configs]"""
import json
import sys


def read(p):
    try:
        return {r["name"]: r for r in map(json.loads, open(p))}
    except FileNotFoundError:
        return {}


prefix, refp = sys.argv[1], sys.argv[2]
cfgs = (sys.argv[3] if len(sys.argv) > 3 else "baseline,exact,exact+q8g64").split(",")
runs = {c: read(f"{prefix}_{c}.jsonl") for c in cfgs}
ref = read(refp)
b = runs[cfgs[0]]
same_t = sum(b[n]["tokens"] == ref[n]["tokens"] for n in b if n in ref)
same_a = sum(b[n]["acc"] == ref[n]["acc"] for n in b if n in ref)
print(f"{cfgs[0]}: {len(b)} prompts; vs {refp.split('/')[-1]}: tokens identical {same_t}/{len(b)}, "
      f"acceptance identical {same_a}/{len(b)}")
names = [n for n in b if all(n in runs[c] for c in cfgs)]
for c in cfgs:
    rs = [runs[c][n] for n in names]
    if rs:
        gen = sum(r["generated"] - 1 for r in rs)
        t = sum(r["decode_s"] for r in rs)
        print(f"  {c}: {len(rs)} prompts, {gen / t:.2f} tok/s, {1e3 * t / sum(r['steps'] for r in rs):.1f} ms/step")
