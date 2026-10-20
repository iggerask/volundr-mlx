"""Acceptance upside from spec/accept_log.py records: tokens per step of the run, and what alternative candidates
(measured on the same steps against the run's greedy output) would have given.

usage: python acc_analyze.py <acclog.jsonl>
Each "policy" picks, per step, a second chain verified next to the run's chain (batched verify, B = 2), or replaces the
run's chain; tokens/step = mean over steps of (accepted drafts of the better chain + 1). Steps are the run's own (the
trajectory a policy would take differs after its first gain, so this is a per-step estimate)."""
import json
import sys
from collections import Counter


def load(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def groups(recs):
    return {
        "all": recs,
        "short (<4K)": [r for r in recs if r["prompt_tokens"] < 4096],
        "long+agent": [r for r in recs if r["category"] in ("long", "agent")],
    }


def tps(steps, f):
    return sum(f(s) + 1 for s in steps) / len(steps)


def main():
    recs = load(sys.argv[1])
    print(f"{sys.argv[1]}: {len(recs)} prompts")
    for g, rs in groups(recs).items():
        st = [s for r in rs for s in r["steps"]]
        n = len(st)
        has_lk = [s for s in st if s["lookup"]]
        print(f"\n## {g}: {len(rs)} prompts, {n} steps; lookup chain available on {len(has_lk)} "
              f"({100 * len(has_lk) / n:.0f}%), used fully on {sum(s['src'] >= 7 for s in st)}, partially on "
              f"{sum(0 < s['src'] < 7 for s in st)}")
        hist = Counter(s["n"] for s in st)
        print("  n histogram: " + ", ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
        base = tps(st, lambda s: s["acc_used"])
        rows = [("run (as verified)", base)]
        rows.append(("DFlash chain only", tps(st, lambda s: s["acc_dflash"])))
        rows.append(("lookup copy whenever available, else DFlash",
                     tps(st, lambda s: s["acc_lookup7"] if s["lookup"] else s["acc_dflash"])))
        rows.append(("oracle choice DFlash | lookup (1 chain)",
                     tps(st, lambda s: max(s["acc_dflash"], s["acc_lookup7"] or 0, s["acc_used"]))))
        rows.append(("B=2: run + other of {DFlash, lookup}",
                     tps(st, lambda s: max(s["acc_used"], s["acc_dflash"], s["acc_lookup7"] or 0))))
        for j in range(7):
            rows.append((f"B=2: run + DFlash alt at slot {j}", tps(st, lambda s, j=j: max(s["acc_used"], s["acc_alts"][j]))))
        rows.append(("B=2: run + DFlash alt at min-margin slot",
                     tps(st, lambda s: max(s["acc_used"], s["acc_alts"][min(range(7), key=lambda j: s["margins"][j])]))))
        rows.append(("B=2: run + DFlash alt at the first rejected slot (oracle)",
                     tps(st, lambda s: max(s["acc_used"], s["acc_alts"][min(s["acc_dflash"], 6)]))))
        rows.append(("B=2: run + (lookup or DFlash if the run used the other, else alt slot 0)",
                     tps(st, lambda s: max(s["acc_used"], (max(s["acc_dflash"], s["acc_lookup7"] or 0)
                                                           if s["lookup"] else s["acc_alts"][0])))))
        rows.append(("T=16 lookup copies (15 drafts) where the run used a full copy",
                     tps(st, lambda s: s["acc_lookup15"] if s["src"] >= 7 and s["acc_lookup15"] is not None
                         else s["acc_used"])))
        for name, v in rows:
            print(f"  {name:72s} {v:.3f}  ({100 * (v / base - 1):+.1f}%)")
        fc = [s for s in st if s["src"] >= 7]
        if fc:
            l15 = [s for s in fc if len(s["lookup"]) >= 15]
            print(f"  full-copy steps: {len(fc)} ({100 * len(fc) / n:.1f}%); copy runs on to 15 tokens on {len(l15)}; "
                  f"7-draft acceptance there {sum(s['acc_used'] for s in l15) / max(1, len(l15)):.2f}, with 15 drafts "
                  f"{sum(s['acc_lookup15'] for s in l15) / max(1, len(l15)):.2f}")
            by_m = {}
            for s in l15:
                by_m.setdefault(min(s["lookup_m"], 16) // 4 * 4, []).append(s)
            print("    by match length (>=): " + "; ".join(
                f"{m}: {len(v)} steps, 7 -> {sum(s['acc_used'] for s in v) / len(v):.2f}, 15 -> "
                f"{sum(s['acc_lookup15'] for s in v) / len(v):.2f}" for m, v in sorted(by_m.items())))
            for extra_ms in (7.5,):
                base_ms = 94.0
                t_b = len(st) * base_ms
                t_l = t_b + len(l15) * extra_ms
                tok_b = sum(s["acc_used"] + 1 for s in st)
                tok_l = tok_b + sum(s["acc_lookup15"] - s["acc_used"] for s in l15)
                print(f"    est. tok/s gain of 16-row copies at +{extra_ms} ms each on a {base_ms} ms step: "
                      f"{100 * ((tok_l / t_l) / (tok_b / t_b) - 1):+.1f}%")
        full = [s for s in st if s["src"] >= 7 and s["acc_used"] == 7]
        print(f"  full copies fully accepted: {len(full)} steps; with 15 drafts they would accept "
              f"{sum(s['acc_lookup15'] for s in full) / max(1, len(full)):.2f} on average")
        miss = Counter(s["dflash_miss_rank"] for s in st if s["dflash_miss_rank"] is not None)
        tot = sum(miss.values())
        print("  DFlash first miss: rank of the target token among the 16 candidates: " +
              ", ".join(f"{k}:{100 * v / tot:.0f}%" for k, v in sorted(miss.items(), key=lambda kv: kv[0])))
        b2 = sum(max(s["acc_dflash"], s["acc_lookup7"] or 0) > s["acc_used"] for s in st)
        print(f"  steps where the other of DFlash / lookup beats the run's chain: {b2} ({100 * b2 / n:.1f}%); where the run's "
              f"chain beats both alternatives' max: {sum(s['acc_used'] > max(s['acc_dflash'], s['acc_lookup7'] or 0) for s in st)}")


if __name__ == "__main__":
    main()
