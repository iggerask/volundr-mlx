"""Per-prompt differences between an interleaved A/B config run and an older specdec run of the same prompts.
usage (from spec/): python ../decode_quick_wins/cmp_old.py runs/decode-quick-wins_tune_spec_main.jsonl runs/tune2_d4h4_lk.jsonl"""
import json
import sys


def read(p):
    return {x["name"]: x for x in map(json.loads, open(p))}


new, old = read(sys.argv[1]), read(sys.argv[2])
print(f"{len(new)} new prompts; categories {sorted({x['category'] for x in new.values()})}")
for n, r in new.items():
    if n not in old:
        continue
    o = old[n]
    if r["tokens"] == o["tokens"] and r.get("acc") == o.get("acc"):
        continue
    td = next((i for i, (x, y) in enumerate(zip(r["tokens"], o["tokens"])) if x != y), None)
    line = f"{n}: tokens {'equal' if r['tokens'] == o['tokens'] else f'differ at {td}'}"
    if "acc" in r:
        a, b = r["acc"], o["acc"]
        ad = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        line += f"; tokens/step {sum(a) / len(a):.3f} vs {sum(b) / len(b):.3f}, first step differing {ad} of {len(a)}/{len(b)}"
    print(line)
