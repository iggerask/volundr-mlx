"""Tables for NOTES_BATCHED.md from the batched-bench run files.

usage (from volundr-mlx): python batched/report.py spec/runs/batched-bench_decode_200.jsonl [...]
"""
import json
import sys
from collections import defaultdict


def main():
    for path in sys.argv[1:]:
        recs = [json.loads(line) for line in open(path)]
        setup = [r for r in recs if r["kind"] == "setup"]
        refs = [r for r in recs if r["kind"] == "ref"]
        runs = [r for r in recs if r["kind"] == "batch"]
        print(f"## {path}")
        for s in setup:
            print({k: v for k, v in s.items() if k != "kind"})
        if refs:
            ms = [r["ms_step"] for r in refs]
            print(f"refs: {len(refs)} streams, B = 1 ms/step {min(ms):.2f}-{max(ms):.2f}, "
                  f"min greedy margin per stream {[r['min_margin'] for r in refs]}")
        print("| B | NAX | ms/step | aggregate tok/s | per-stream tok/s | peak GB | peak less snapshot | identical | "
              "first divergences (stream@token: ref margin / batched margin) |")
        print("|---|---|---|---|---|---|---|---|---|")
        by = defaultdict(list)
        for r in runs:
            by[(r["B"], r["nax"])].append(r)
        for (B, nax), rs in sorted(by.items()):
            for r in rs:
                div = ", ".join(f"{d['stream']}@{d['diverge_at']}: {d['ref_margin']:.3f} / {d['margin']:.3f}"
                                for d in r.get("diverged", []))
                print(f"| {B} | {'on' if nax else 'off'} | {r['ms_step']:.2f} | {r['agg_tok_s']:.1f} | "
                      f"{r['stream_tok_s']:.2f} | {r['peak_gb']:.2f} | {r['peak_less_snapshot_gb']:.2f} | "
                      f"{r.get('identical', '-')}/{B} | {div or '-'} |")
        print()


if __name__ == "__main__":
    main()
