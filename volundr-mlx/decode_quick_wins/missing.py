"""Which tune prompts / configs the interleaved A/B (tune_chain.sh) still lacks, per category group.
usage (from volundr-mlx): python3 decode_quick_wins/missing.py"""
import json

s = [e for e in json.load(open("spec/suite.json")) if e["split"] == "tune"]
groups = ["coding,humanities,json,math,multilingual,qa,rag", "reasoning,roleplay,stem,summarization,writing", "agent,long"]
cfgs = ["main", "fused", "fused+q8g64", "fused+q8g128"]
for mode in ("spec", "greedy"):
    have = {}
    for c in cfgs:
        try:
            with open(f"spec/runs/decode-quick-wins_tune_{mode}_{c}.jsonl") as f:
                have[c] = {json.loads(line)["name"] for line in f if line.strip()}
        except FileNotFoundError:
            have[c] = set()
    for gi, g in enumerate(groups, 1):
        names = [e["name"] for e in s if e["category"] in g.split(",")]
        miss = {c: [n for n in names if n not in have[c]] for c in cfgs}
        tot = sum(len(v) for v in miss.values())
        if tot == 0:
            print(f"{mode} group {gi} ({g}): complete, {len(names)} prompts x 4 configs")
        elif tot == 4 * len(names):
            print(f"{mode} group {gi} ({g}): not started, {len(names)} prompts")
        else:
            print(f"{mode} group {gi} ({g}): missing " + "; ".join(f"{c}: {v}" for c, v in miss.items() if v))
