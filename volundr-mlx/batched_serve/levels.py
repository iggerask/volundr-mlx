"""Print the levels of a server.jsonl so far (optionally waiting until it has N lines).
usage: python batched_serve/levels.py FILE [N] [timeout_s]"""
import json
import sys
import time
from pathlib import Path

f = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end = time.time() + (float(sys.argv[3]) if len(sys.argv) > 3 else 0)
while n and time.time() < end and (not f.exists() or len([x for x in f.read_text().splitlines() if x.strip()]) < n):
    time.sleep(5)
for ln in f.read_text().splitlines() if f.exists() else []:
    if not ln.strip():
        continue
    r = json.loads(ln)
    print({k: r.get(k) for k in ("kind", "concurrency", "agg_tok_s", "per_request_tok_s_mean", "ttft_p50_s",
                                  "ttft_p95_s", "peak_gb", "modes", "mean_batch", "canary_before", "canary_after",
                                  "wall_s", "schedule", "agent_ttft_s")})
