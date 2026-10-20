"""CPU cost of building the verify graph (verify.forward, verify=True), by function (cProfile) and in total, with and
without the GPU busy (the build overlaps the draft in the spec loop). No GPU work is timed.

usage (from volundr-mlx/spec, under ../with_gpu_lock.sh): python prof_vbuild.py [--reps 20] [--T 8]
"""
import argparse
import cProfile
import pstats
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
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()
    model, drafter, tok = S.load(4, 4, True)
    ids = tok.encode("Write a short story about a lighthouse keeper who finds a message in a bottle." * 3)
    cache, dcache = model.make_cache(), drafter.make_cache()
    S.spec_prefill(model, drafter, cache, dcache, ids)
    gen = [int(t) for t in tok.encode(" the quick brown fox jumps over the lazy dog" * 40)]
    w = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    mx.eval(w)
    prof = cProfile.Profile()
    res = {"idle": [], "busy": []}
    pos = 0
    for rep in range(2 * args.reps + 2):
        busy = rep % 2 == 1
        inp = mx.array([gen[pos : pos + args.T]], dtype=mx.int32)
        if busy:  # keep the GPU streaming (a ~15 ms matmul chain) while the CPU builds
            x = mx.ones((16, 4096), mx.bfloat16)
            for _ in range(200):
                x = (x @ w) * 1e-3
            mx.async_eval(x)
        if rep >= 2 and not busy:
            prof.enable()
        t0 = time.perf_counter()
        lg, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
        tk = mx.argmax(lg, axis=-1)
        t1 = time.perf_counter()
        if rep >= 2 and not busy:
            prof.disable()
        if rep >= 2:
            res["busy" if busy else "idle"].append(1e3 * (t1 - t0))
        mx.eval(tk, cap, [c.state for c in cache])
        VF.commit(cache, rec, 3)
        mx.eval([c.state for c in cache])
        pos += 3
    for k, v in res.items():
        v.sort()
        print(f"verify build T={args.T}, GPU {k}: median {v[len(v) // 2]:.2f} ms (of {len(v)})")
    pstats.Stats(prof).sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
