"""Teacher-forced check of the batched (B = 2) verify / commit path (verify.forward_batch / commit_batch), as
check_verify.py does for B = 1.

Per prompt: greedy-decode N tokens g_0..g_{N-1}, keeping each step's logits D_k (predicting g_{k+1}). Then from a fresh
cache, verify forwards over two chains: the true block [g_i .. g_{i+T-1}] and a decoy that equals it up to a random
position and then has random tokens; the true chain sits in a random batch row. Commit a random n in 1..T of the true
chain (so the KDA replay / n = T state pick, BCSA partial append, dense-KV append and Engram ids run from either row).
Rows of the true chain should reproduce D_{i+j} (KL at the B = 1 verify's noise floor); the decoy's rows before its
first random token see the same prefix and must agree with the true chain's to rounding. Also, per step, the B = 1
verify of the same true block from the same state (run on a second cache kept in lockstep) for a B = 2 vs B = 1 KL.

usage: python check_verify_b2.py [--cats ...] [--per-cat 1] [--new 64] [--T 8] --nax
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
from check_verify import kl  # noqa: E402


def run(model, ids, N, T, rng):
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
    cache, cache1 = model.make_cache(), model.make_cache()
    S.G.prefill(model, cache, ids)
    S.G.prefill(model, cache1, ids)
    i, rows = 0, []
    while i + T <= N:
        true = g[i : i + T]
        cut = rng.randint(1, T - 1)
        decoy = true[:cut] + [rng.randrange(1000, 200000) for _ in range(T - cut)]
        b = rng.randint(0, 1)
        chains = [true, decoy] if b == 0 else [decoy, true]
        lg, _, rec = VF.forward_batch(model, mx.array(chains, dtype=mx.int32), cache)
        lg = lg[..., :vv]
        lg1, _, rec1 = VF.forward(model, mx.array([true], dtype=mx.int32), cache1, verify=True)
        lg1 = lg1[:, :vv]
        ref = mx.stack(D[i : i + T])
        k = kl(ref, lg[b])
        k21 = kl(lg1, lg[b])
        kd = kl(lg[b, :cut], lg[1 - b, :cut])
        top = mx.argmax(lg[b], axis=-1)
        mx.eval(k, k21, kd, top, [c.state for c in cache], [c.state for c in cache1])
        for j in range(T):
            rows.append({"pos": i + j, "j": j, "kl": k[j].item(), "kl_b2_vs_b1": k21[j].item(),
                         "agree": top[j].item() == g[i + j + 1],
                         "kl_decoy_prefix": kd[j].item() if j < cut else None})
        n = rng.randint(1, T)
        VF.commit_batch(cache, rec, b, n)
        VF.commit(cache1, rec1, n)
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
    allk, all21 = [], []
    for e in S.suite(args.suite, args.split, args.per_cat, [c for c in args.cats.split(",") if c]):
        rows = run(model, e["ids"], args.new, args.T, rng)
        ks = sorted(r["kl"] for r in rows)
        k21 = sorted(r["kl_b2_vs_b1"] for r in rows)
        kd = [r["kl_decoy_prefix"] for r in rows if r["kl_decoy_prefix"] is not None]
        allk += ks
        all21 += k21
        print(json.dumps({"name": e["name"], "prompt_tokens": len(e["ids"]), "rows": len(rows),
                          "agree": sum(r["agree"] for r in rows), "kl_mean": sum(ks) / len(ks),
                          "kl_p99": ks[int(0.99 * (len(ks) - 1))], "kl_max": ks[-1],
                          "kl_b2_vs_b1_mean": sum(k21) / len(k21), "kl_b2_vs_b1_max": k21[-1],
                          "kl_decoy_prefix_max": max(kd)}), flush=True)
        mx.clear_cache()
    allk.sort()
    print(json.dumps({"all_rows": len(allk), "kl_mean": sum(allk) / len(allk),
                      "kl_p99": allk[int(0.99 * (len(allk) - 1))], "kl_max": allk[-1],
                      "kl_b2_vs_b1_mean": sum(all21) / len(all21), "kl_b2_vs_b1_max": max(all21)}))


if __name__ == "__main__":
    main()
