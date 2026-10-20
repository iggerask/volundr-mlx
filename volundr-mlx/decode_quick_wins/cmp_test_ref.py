"""Partial held-out sample against a reference run from another session on the same prompts (NOT an A/B: different
code and session). Pooled tok/s, tokens per verify step, and the speed-up against a reference greedy run.
usage (from spec/): python ../decode_quick_wins/cmp_test_ref.py runs/decode-quick-wins_test_frozen.jsonl \
    runs/bcsa-verify-8q_test_8q.jsonl runs/bcsa-verify-8q_test_base.jsonl"""
import json
import sys


def read(p):
    return {r["name"]: r for r in map(json.loads, open(p))}


def rate(rs):
    return sum(r["generated"] - 1 for r in rs) / sum(r["decode_s"] for r in rs)


mine, ref, base = (read(p) for p in sys.argv[1:4])
names = [n for n in mine if n in ref and n in base]
cats = sorted({mine[n]["category"] for n in names})
m, r, b = ([d[n] for n in names] for d in (mine, ref, base))
acc = lambda rs: sum(sum(x["acc"]) for x in rs) / sum(len(x["acc"]) for x in rs)
ms = lambda rs: 1e3 * sum(x["decode_s"] for x in rs) / sum(x["steps"] for x in rs)
print(f"{len(names)} common prompts ({', '.join(cats)})")
print(f"  this run  {rate(m):6.2f} tok/s, {acc(m):.2f} tokens/step, {ms(m):6.2f} ms/step, x{rate(m) / rate(b):.2f} vs ref greedy")
print(f"  ref spec  {rate(r):6.2f} tok/s, {acc(r):.2f} tokens/step, {ms(r):6.2f} ms/step, x{rate(r) / rate(b):.2f} vs ref greedy")
print(f"  ref greedy {rate(b):6.2f} tok/s; this run / ref spec x{rate(m) / rate(r):.3f}")
print(f"  tokens identical to ref spec on {sum(mine[n]['tokens'] == ref[n]['tokens'] for n in names)}/{len(names)}, "
      f"to ref greedy on {sum(mine[n]['tokens'] == base[n]['tokens'] for n in names)}/{len(names)}; "
      f"acceptance identical to ref spec on {sum(mine[n]['acc'] == ref[n]['acc'] for n in names)}/{len(names)}")
