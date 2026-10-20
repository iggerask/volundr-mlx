"""agent/02-03 repeat against the A/B: identical tokens / acceptance (determinism + Lookup replay), and the agent and
long+agent rows with the repeat's timings substituted (sensitivity to the noisy first run).
usage: python rerun_check.py"""
import json

R = "/Users/ingemarrask/personal-dev/llms/volundr-mlx/spec/runs/combined-test_tune_spec"
CF = ("baseline", "exact", "exact+q8g64")
main = {c: {r["name"]: r for r in map(json.loads, open(f"{R}_{c}.jsonl"))} for c in CF}
rep = {c: {r["name"]: r for r in map(json.loads, open(f"{R}_agentrerun_{c}.jsonl"))} for c in CF}
for c in CF:
    for n in ("agent/02-buriy__python-readabilit-think", "agent/03-pyupio__safety"):
        a, b = main[c][n], rep[c][n]
        print(f"{c:12s} {n[:24]:24s} tokens equal {a['tokens'] == b['tokens']}, acc equal {a['acc'] == b['acc']}; "
              f"ms/step A/B {1e3 * a['decode_s'] / a['steps']:.1f} repeat {1e3 * b['decode_s'] / b['steps']:.1f}")


def rate(rs):
    return sum(r["generated"] - 1 for r in rs) / sum(r["decode_s"] for r in rs)


for label, cats in (("agent", ("agent",)), ("long+agent", ("agent", "long"))):
    out = []
    for c in CF:
        sub = {**main[c], **{n: r for n, r in rep[c].items() if n.startswith(("agent/02", "agent/03"))}}
        rs = [r for r in sub.values() if r["category"] in cats]
        out.append((c, rate(rs)))
    b = out[0][1]
    print(f"{label} with the repeat substituted: " + ", ".join(f"{c} {v:.2f} tok/s (x{v / b:.3f})" for c, v in out))
