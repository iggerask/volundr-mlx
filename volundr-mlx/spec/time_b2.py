"""GPU cost of the batched verify against the B = 1 verify on a real context: verify forward (+ argmax) and the
commit of n tokens, each timed from an idle GPU, the variants interleaved per rep (median ms). The context advances by
n after each timed call (both variants commit the same tokens), so every call sees a similar context length.

usage (from volundr-mlx/spec, under ../with_gpu_lock.sh): python time_b2.py --name long/01-long1 [--reps 8] [--n 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="coding/00-12623519-think,long/01-long1")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()
    model, drafter, tok = S.load(4, 4, True)
    suite = {e["name"]: e for e in json.loads((HERE / "suite.json").read_text())}
    gen = [int(t) for t in tok.encode(" the quick brown fox jumps over the lazy dog and runs far away" * 40)]
    vv = model.args.vocab_valid
    for name in args.names.split(","):
        ids = suite[name]["ids"]
        cache, dcache = model.make_cache(), drafter.make_cache()
        S.spec_prefill(model, drafter, cache, dcache, ids)
        T, n, pos = args.T, args.n, 0
        res = {"b1": [], "b2": [], "b1_commit": [], "b2_commit": []}
        for rep in range(args.reps + 1):
            for var in ("b1", "b2"):
                blk = gen[pos : pos + T]
                mx.synchronize()
                t0 = time.perf_counter()
                if var == "b1":
                    lg, cap, rec = VF.forward(model, mx.array([blk], dtype=mx.int32), cache,
                                              capture=drafter.capture_layers, verify=True)
                    top = mx.argmax(lg[:, :vv], axis=-1)
                else:
                    alt = blk[:2] + gen[pos + 100 : pos + 100 + T - 2]
                    lg, cap, rec = VF.forward_batch(model, mx.array([blk, alt], dtype=mx.int32), cache,
                                                    capture=drafter.capture_layers)
                    top = mx.argmax(lg[..., :vv], axis=-1)
                mx.eval(top, cap, [c.state for c in cache])
                t1 = time.perf_counter()
                if var == "b1":
                    VF.commit(cache, rec, n)
                else:
                    VF.commit_batch(cache, rec, 0, n)
                mx.eval([c.state for c in cache])
                t2 = time.perf_counter()
                pos += n
                if rep:
                    res[var].append(1e3 * (t1 - t0))
                    res[var + "_commit"].append(1e3 * (t2 - t1))
        out = {"name": name, "prompt_tokens": len(ids), "T": T, "n": n, "reps": args.reps}
        for k, v in res.items():
            v.sort()
            out[k + "_ms"] = round(v[len(v) // 2], 2)
        print(json.dumps(out), flush=True)
        del cache, dcache
        mx.clear_cache()


if __name__ == "__main__":
    main()
