"""Extra columns for a spec A/B (spec/ab_suite.py records) next to combined_test/analyze.py's table: per config the
host segments per step (CPU before the first submit, the submit (+ commit build in the early loop), the verify build),
the wait, the share of lookup / long-copy steps, the peak memory, and token / acceptance identity against the first
config.
usage: python ab_report.py <prefix> <configs>"""
import json
import sys


def read(path):
    return {r["name"]: r for r in map(json.loads, open(path))}


prefix, cfgs = sys.argv[1], sys.argv[2].split(",")
runs = {c: read(f"{prefix}_{c}.jsonl") for c in cfgs}
names = [n for n in runs[cfgs[0]] if all(n in runs[c] for c in cfgs)]
print(f"{len(names)} prompts")
print("| config | CPU before submit ms | submit (+commit build) ms | verify build ms | wait ms | lookup steps | "
      "16-row steps | max peak GB | tokens = first | acceptance = first |")
print("|---|---|---|---|---|---|---|---|---|---|")
b = cfgs[0]
for c in cfgs:
    rs = [runs[c][n] for n in names]
    steps = sum(r["steps"] for r in rs)
    seg = [sum(r["seg_s"][i] for r in rs if "seg_s" in r) * 1e3 / steps for i in range(3)]
    wait = sum(r["wait_s"] for r in rs) * 1e3 / steps
    src = [s for r in rs for s in r["src"]]
    same = sum(runs[c][n]["tokens"] == runs[b][n]["tokens"] for n in names)
    acc = sum(runs[c][n]["acc"] == runs[b][n]["acc"] for n in names)
    print(f"| {c} | {seg[0]:.2f} | {seg[1]:.2f} | {seg[2]:.2f} | {wait:.2f} | "
          f"{100 * sum(s >= 7 for s in src) / len(src):.1f}% | {100 * sum(s > 7 for s in src) / len(src):.1f}% | "
          f"{max(r['peak_gb'] for r in rs):.2f} | {same}/{len(names)} | {acc}/{len(names)} |")
