"""Per-prompt ms/step, tok/s, tokens/step and peak memory of the long / agent prompts across configs.
usage: python per_prompt.py <prefix> <configs> [categories]"""
import json
import sys

prefix, cfgs = sys.argv[1], sys.argv[2].split(",")
cats = (sys.argv[3] if len(sys.argv) > 3 else "agent,long").split(",")
runs = {c: {r["name"]: r for r in map(json.loads, open(f"{prefix}_{c}.jsonl"))} for c in cfgs}
for n, r0 in runs[cfgs[0]].items():
    if r0["category"] not in cats:
        continue
    cells = []
    for c in cfgs:
        r = runs[c][n]
        cells.append(f"{c}: {1e3 * r['decode_s'] / r['steps']:6.1f} ms {r['tok_s']:5.1f} t/s {sum(r['acc']) / len(r['acc']):4.2f} "
                     f"pk {r.get('peak_gb')}")
    print(f"{n[:30]:30s} {r0['prompt_tokens']:6d} | " + " | ".join(cells))
