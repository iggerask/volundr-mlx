"""Per-prompt ms/step of the interleaved A/B configs for one category (outlier check).
usage (from spec/): python ../decode_quick_wins/per_prompt.py runs/decode-quick-wins_tune_spec coding main,fused,fused+q8g64"""
import json
import sys

prefix, cat, cfgs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
runs = {c: {r["name"]: r for r in map(json.loads, open(f"{prefix}_{c}.jsonl"))} for c in cfgs}
for n, r in runs[cfgs[0]].items():
    if r["category"] != cat:
        continue
    cells = []
    for c in cfgs:
        x = runs[c][n]
        steps = x.get("steps", x["generated"] - 1)
        cells.append(f"{c} {1e3 * x['decode_s'] / steps:6.1f}")
    print(f"{n:40s} " + " | ".join(cells))
