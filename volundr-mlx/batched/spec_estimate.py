"""ESTIMATE of batched speculative decoding (B independent sequences, T = 8) from measured parts (NOTES_BATCHED.md).

step(B) = step(1) + [verify(B) - verify(1)] + [drafter(B) - drafter(1)] + (B - 1) * commit
- step(1), tokens/step: the measured B = 1 spec loop (spec-accept tune run, pooled; aggregates only);
- verify(B), drafter(B): bench_verify.py, from an idle GPU, same session (verify: B x 8 rows, NAX, more than 16 rows as
  16-row NAX calls; drafter: B proposals submitted together);
- commit = commit + drafter-context update of one stream, 2.4 + 0.8 ms (NOTES_SPEC_ACCEPT.md serialized profile).
The low end assumes a batched drafter and commit that cost what B = 1 costs (drafter(B) = drafter(1), no commit term).

usage (from volundr-mlx): python batched/spec_estimate.py spec/runs/batched-bench_verify.jsonl
"""
import json
import sys

COMMIT = 2.4 + 0.8
# measured B = 1 spec loop, exact+q8g64@early@cv on the tune split (batched/accept_summary.py)
B1 = {200: {"step": 93.19, "tps": 2.617, "group": "short (<4K), 48 prompts"},
      8192: {"step": 97.45, "tps": 2.516, "group": ">=4K (4.5-12.5K), 8 prompts"}}


def main():
    recs = [json.loads(line) for line in open(sys.argv[1])]
    get = {}
    for r in recs:  # the last record of each (context, name) wins
        get[(r["context"], r["name"])] = r["ms_median"]
    print("| context | B | verify rows | verify ms (B=1) | drafter ms | est. step ms | est. per-stream tok/s | "
          "est. aggregate tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for ctx, ref in B1.items():
        v1, d1 = get[(ctx, "verify_rows B=1 nax")], get[(ctx, "drafter B=1")]
        for B in (1, 2, 4):
            v = get.get((ctx, f"verify_rows B={B} split"), get[(ctx, f"verify_rows B={B} nax")])
            d = get[(ctx, f"drafter B={B}")]
            hi = ref["step"] + (v - v1) + (d - d1) + (B - 1) * COMMIT
            lo = ref["step"] + (v - v1)
            per = (ref["tps"] * 1e3 / hi, ref["tps"] * 1e3 / lo)
            print(f"| {ctx} | {B} | {8 * B} | {v:.1f} ({v1:.1f}) | {d:.1f} | {lo:.0f}-{hi:.0f} | "
                  f"{per[0]:.1f}-{per[1]:.1f} | {B * per[0]:.1f}-{B * per[1]:.1f} |")


if __name__ == "__main__":
    main()
