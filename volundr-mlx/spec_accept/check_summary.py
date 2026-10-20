"""Summary of specdec.py check runs (spec against plain greedy, 128 tokens): per run, prompts identical to greedy and,
for the others, the greedy top-2 margin at the first divergence (0.0 = an exact bf16 tie).
usage: python check_summary.py <check.jsonl> ..."""
import json
import sys

for path in sys.argv[1:]:
    rs = [json.loads(x) for x in open(path) if x.strip()]
    same = sum(r["match"] for r in rs)
    divs = [(r["name"], r["first_diverge"], r["margin_at_div"]) for r in rs if not r["match"]]
    print(f"{path}: {same}/{len(rs)} identical to greedy; divergences (prompt, token, greedy margin there):")
    for d in divs:
        print(f"  {d[0]:40s} {d[1]:4d}  {d[2]}")
    print(f"  mean tokens/step {sum(r['mean_acc'] for r in rs) / len(rs):.2f}")
