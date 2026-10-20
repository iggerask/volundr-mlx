"""Pooled per-stream acceptance and step time of a spec run file (aggregates only), for the batched spec estimate.

usage (from volundr-mlx): python batched/accept_summary.py "spec/runs/spec-accept_tune_exact+q8g64@early@cv.jsonl"
"""
import json
import sys


def main():
    recs = [json.loads(line) for line in open(sys.argv[1])]
    groups = {"all": recs, "short (<4K)": [r for r in recs if r["prompt_tokens"] < 4096],
              ">=4K (long+agent)": [r for r in recs if r["prompt_tokens"] >= 4096]}
    for name, rs in groups.items():
        steps = sum(r["steps"] for r in rs)
        toks = sum(sum(r["acc"]) for r in rs)
        dec = sum(r["decode_s"] for r in rs)
        gen = sum(r["generated"] for r in rs)
        print(f"{name}: prompts {len(rs)}, tokens/step {toks / steps:.3f}, ms/step {1e3 * dec / steps:.2f}, "
              f"tok/s {gen / dec:.2f}, context {min(r['prompt_tokens'] for r in rs)}-{max(r['prompt_tokens'] for r in rs)}")


if __name__ == "__main__":
    main()
