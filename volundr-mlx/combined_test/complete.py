"""Exit 0 when a combined-test stage has every record: tune = 56 tune prompts x 3 configs; test = the 18 held-out
coding / long / agent prompts in both runs.
usage: python complete.py tune|test"""
import json
import sys
from pathlib import Path

R = Path("/Users/ingemarrask/personal-dev/llms/volundr-mlx/spec/runs")
suite = json.loads((R.parent / "suite.json").read_text())
stage = sys.argv[1]
if stage == "tune":
    want = {e["name"] for e in suite if e["split"] == "tune"}
    files = [R / f"combined-test_tune_spec_{c}.jsonl" for c in ("baseline", "exact", "exact+q8g64")]
else:
    want = {e["name"] for e in suite if e["split"] == "test" and e["category"] in ("coding", "long", "agent")}
    files = [R / f"combined-test_test_{c}.jsonl" for c in ("baseline", "exact+q8g64")]
ok = True
for f in files:
    have = {json.loads(ln)["name"] for ln in f.read_text().splitlines() if ln.strip()} if f.exists() else set()
    print(f"{f.name}: {len(have & want)}/{len(want)}")
    ok &= want <= have
sys.exit(0 if ok else 1)
