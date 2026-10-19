"""Teacher-forced check of the verify / commit path against plain decode steps.

Per prompt: greedy-decode N tokens g_0..g_{N-1} keeping each step's logits D_k (predicting g_{k+1}). Then, from a
fresh cache, run verify forwards over [g_i .. g_{i+T-1}] and commit a random n in 1..T each time (so every rollback
kind runs: KDA replay, BCSA partial append, dense-KV trim, Engram ids). Row j of a verify forward should reproduce
D_{i+j}. A wrong rollback shows as a large KL from the next step on; rounding alone stays near the parity noise
floor (the exact kernels vs phase 1: mean KL 8e-4).

usage: python check_verify.py [--suite suite.json] [--names a,b] [--cats long,agent] [--new 64] [--T 8]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402

F32 = mx.float32


def kl(p_logits, q_logits):
    lp = p_logits.astype(F32) - mx.logsumexp(p_logits.astype(F32), axis=-1, keepdims=True)
    lq = q_logits.astype(F32) - mx.logsumexp(q_logits.astype(F32), axis=-1, keepdims=True)
    return (mx.exp(lp) * (lp - lq)).sum(-1)


def run(model, ids, N, T, rng, nax_rows=False):
    vv = model.args.vocab_valid
    cache = model.make_cache()
    last = S.G.prefill(model, cache, ids)
    D, g, logits = [], [], last
    for _ in range(N + T):
        y = mx.argmax(logits[:vv]).astype(mx.int32)
        mx.eval(y)
        g.append(y.item())
        logits = model(y.reshape(1, 1), cache=cache, last_only=True)[0, -1]
        D.append(logits[:vv])
        mx.eval(logits, [c.state for c in cache])
    del cache
    mx.clear_cache()
    cache = model.make_cache()
    S.G.prefill(model, cache, ids)
    i, rows = 0, []
    while i + T <= N:
        inp = mx.array(g[i : i + T], dtype=mx.int32)[None]
        lg, _, rec = VF.forward(model, inp, cache, verify=True)
        lg = lg[:, :vv]
        ref = mx.stack(D[i : i + T])
        k = kl(ref, lg)
        top = mx.argmax(lg, axis=-1)
        mx.eval(k, top, [c.state for c in cache])
        for j in range(T):
            rows.append({"pos": i + j, "j": j, "kl": k[j].item(), "agree": top[j].item() == g[i + j + 1]})
        n = rng.randint(1, T)
        VF.commit(cache, rec, n)
        i += n
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--cats", default="")
    ap.add_argument("--per-cat", type=int, default=1)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--nax", action="store_true")
    args = ap.parse_args()
    model, _, tok = S.load(nax=args.nax)
    rng = random.Random(7)
    allk = []
    for e in S.suite(args.suite, args.split, args.per_cat, [c for c in args.cats.split(",") if c]):
        rows = run(model, e["ids"], args.new, args.T, rng)
        ks = sorted(r["kl"] for r in rows)
        allk += ks
        print(json.dumps({"name": e["name"], "prompt_tokens": len(e["ids"]), "rows": len(rows),
                          "agree": sum(r["agree"] for r in rows), "kl_mean": sum(ks) / len(ks),
                          "kl_p99": ks[int(0.99 * (len(ks) - 1))], "kl_max": ks[-1],
                          "kl_mean_j0": sum(r["kl"] for r in rows if r["j"] == 0) / max(1, sum(r["j"] == 0 for r in rows)),
                          "kl_mean_j7": sum(r["kl"] for r in rows if r["j"] == args.T - 1) /
                          max(1, sum(r["j"] == args.T - 1 for r in rows))}), flush=True)
        mx.clear_cache()
    allk.sort()
    print(json.dumps({"all_rows": len(allk), "kl_mean": sum(allk) / len(allk), "kl_p99": allk[int(0.99 * (len(allk) - 1))],
                      "kl_max": allk[-1]}))


if __name__ == "__main__":
    main()
