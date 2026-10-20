"""Long-copy steps of an A/B run by the Lookup's match-length bucket: replays each config's Lookup over its recorded run
(the A/B's process groups, in suite order) to recover the bucket of every copy step, then prints per bucket the copy
steps, their mean accepted count n and the share with n > 8 (more than an 8-row verify can accept).
usage: python long_buckets.py <prefix> <config> <loop flags>   e.g. spec-accept_tune exact+q8g64@early@cv@long16 long16
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spec"))
import lookup as LK  # noqa: E402

R = Path("/Users/ingemarrask/personal-dev/llms/volundr-mlx/spec/runs")
GROUPS = [("coding", "humanities", "json", "math", "multilingual", "qa", "rag"),
          ("reasoning", "roleplay", "stem", "summarization", "writing"), ("agent", "long")]


def main():
    prefix, cfg, flags = sys.argv[1], sys.argv[2], sys.argv[3]
    long = int(flags[4:].rstrip("c").split("g")[0])
    cap = "c" in flags[4:]
    suite = [e for e in json.loads((R.parent / "suite.json").read_text()) if e["split"] == "tune"]
    recs = {r["name"]: r for r in map(json.loads, open(R / f"{prefix}_{cfg}.jsonl"))}
    eos = {248046, 248044}  # the model config's eos ids
    by = defaultdict(list)
    for grp in GROUPS:
        look = LK.Lookup()
        for e in [e for e in suite if e["category"] in grp]:
            rec = recs[e["name"]]
            out = rec["tokens"]
            look.reset(list(e["ids"]) + out[:1])
            pos = 1
            for n, s in zip(rec["acc"], rec["src"]):
                hint = look.propose(long - 1)
                b = look.pending[1] if look.pending and look.pending[0] else None
                k = long - 1 if len(hint) >= long - 1 else (7 if len(hint) >= 7 else len(hint))
                assert k == s, (e["name"], k, s)
                if s >= 7:
                    by[(b, s)].append(n)
                emitted = out[pos : pos + n]
                if any(t in eos for t in emitted):  # the run stops before recording this step
                    break
                look.record(emitted + [-1] * (n - len(emitted)), cap=7 if cap else None)
                pos += n
    print(f"{cfg}: copy steps by (bucket, drafts): count, mean n, share n > 8")
    for (b, s), ns in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  bucket {b} (match >= {LK.BUCKETS[b]}), {s} drafts: {len(ns):4d} steps, mean n {sum(ns) / len(ns):.2f}, "
              f"n > 8 on {100 * sum(x > 8 for x in ns) / len(ns):.0f}%")


if __name__ == "__main__":
    main()
