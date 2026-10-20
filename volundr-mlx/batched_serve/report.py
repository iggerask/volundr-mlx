"""Tables for NOTES_BATCHED_SERVE.md from a serve_bench.sh run directory.

usage: python batched_serve/report.py batched_serve/run
"""
import json
import sys
from pathlib import Path

UPSTREAM = {1: 87.7, 2: 129.3, 4: 155.1}


def rows(p):
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()] if Path(p).exists() else []


def main():
    d = Path(sys.argv[1])
    levels = [r for r in rows(d / "server.jsonl") if r.get("kind") == "level"]
    cmp_ = {r["concurrency"]: r for r in rows(d / "compare.jsonl")}
    ties = rows(d / "ties.jsonl")
    print("| concurrency | aggregate tok/s | upstream | all c streaming tok/s | per-request tok/s (mean / p50) | "
          "TTFT p50 / p95 / max s | wall s | mean batch | modes | peak GB | identical to c=1 | canary before / after |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    prev = None
    for lv in levels:
        c = lv["concurrency"]
        cm = cmp_.get(c)
        ident = f"{cm['identical_to_c1']}/{cm['requests']}" if cm else "-"
        modes = ", ".join(f"{k} {v}" for k, v in sorted(lv["modes"].items(), key=str))
        can = f"{(lv.get('canary_before') or '-').replace('canary ', '')} / {(lv.get('canary_after') or '-').replace('canary ', '')}"
        print(f"| {c} | **{lv['agg_tok_s']}** | {UPSTREAM.get(c, '-')} | {lv.get('agg_full_tok_s', '-')} | "
              f"{lv['per_request_tok_s_mean']} / "
              f"{lv['per_request_tok_s_p50']} | {lv['ttft_p50_s']} / {lv['ttft_p95_s']} / {lv['ttft_max_s']} | "
              f"{lv['wall_s']} | {lv['mean_batch']} | {modes} | {lv['peak_gb']} | {ident} | {can} |")
        e = lv.get("engine") or {}
        if prev is not None:
            delta = {k: round(e.get(k, 0) - prev.get(k, 0), 3) for k in ("steps_batch", "steps_spec", "rows_batch",
                                                                        "prefill_slices")}
            print(f"<!-- c={c} engine deltas {delta} -->")
        prev = e
    if cmp_:
        print()
        for c, r in sorted(cmp_.items()):
            div = sorted(x["diverge_at"] for x in r["diverged"] if x["diverge_at"] is not None)
            print(f"- c={c}: {r['identical_to_c1']}/{r['requests']} identical to c=1; first differences at {div}")
    if ties:
        print()
        print("| concurrency | first difference | modes (c=1, c) | logit of c=1's token | of the other | top-2 margin |")
        print("|---|---|---|---|---|---|")
        for t in ties:
            print(f"| {t['concurrency']} | {t['diverge_at']} | {t['modes'][0]}, {t['modes'][1]} | "
                  f"{t['logit_c1_token']} | {t['logit_other']} | {t['top2_margin']} |")
    mixed = rows(d / "mixed.jsonl")
    if mixed:
        print()
        for m in mixed:
            if m["kind"] != "mixed":
                print(json.dumps(m))
                continue
            aw = m.get("agent_worker") or {}
            print(f"### {m['schedule']} (slice {m['slice_ms']} ms): agent TTFT {m['agent_ttft_s']} s, prompt "
                  f"{aw.get('prompt_tokens')} (cached {aw.get('cached_tokens')}, prefilled {aw.get('prefilled_tokens')}),"
                  f" agent decode {aw.get('decode_tok_s')} tok/s, mean batch {aw.get('mean_batch')}")
            print("| stream | tok/s before | tokens / tok/s during the agent prefill | longest gap overlapping it s | "
                  "tok/s after the agent joins | gaps > 250 ms (s after the agent's send, length s) |")
            print("|---|---|---|---|---|---|")
            for x in m["short"]:
                print(f"| {x['stream']} | {x['tok_s_before']} | {x['tokens_during_prefill']} / "
                      f"{x['tok_s_during_prefill']} | {x['max_gap_during_prefill_s']} | {x['tok_s_after_join']} | "
                      f"{x['gaps_over_250ms']} |")


if __name__ == "__main__":
    main()
