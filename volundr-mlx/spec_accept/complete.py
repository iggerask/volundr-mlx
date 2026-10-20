"""Exit 0 when every config of an A/B has all its records.
usage: python complete.py <prefix under spec/runs> <configs> tune|test"""
import json
import sys
from pathlib import Path

R = Path("/Users/ingemarrask/personal-dev/llms/volundr-mlx/spec/runs")
suite = json.loads((R.parent / "suite.json").read_text())
prefix, configs, stage = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
if stage == "tune":
    want = {e["name"] for e in suite if e["split"] == "tune"}
else:
    want = {e["name"] for e in suite if e["split"] == "test" and e["category"] in ("coding", "long", "agent")}
ok = True
for c in configs:
    f = R / f"{prefix}_{c}.jsonl"
    have = {json.loads(ln)["name"] for ln in f.read_text().splitlines() if ln.strip()} if f.exists() else set()
    print(f"{f.name}: {len(have & want)}/{len(want)}")
    ok &= want <= have
sys.exit(0 if ok else 1)
