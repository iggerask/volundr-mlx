"""CPU cost of building one speculative step's lazy graph (no GPU work timed): commit, materialize, propose, verify.

usage: python prof_build.py [--nax] [--drafter-bits 4] [--profile]
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
    ap.add_argument("--nax", action="store_true")
    ap.add_argument("--drafter-bits", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    model, drafter, tok = S.load(args.drafter_bits, 0, args.nax)
    ids = tok.encode("Write a short story about a lighthouse keeper who finds a message in a bottle." * 3)
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = S.spec_prefill(model, drafter, cache, dcache, ids)
    bonus = mx.argmax(last).astype(mx.int32)
    P = len(ids)
    times = {k: [] for k in ("propose", "verify", "commit", "materialize")}
    prof = cProfile.Profile() if args.profile else None
    for rep in range(args.reps + 1):
        t0 = time.perf_counter()
        drafts = drafter.propose(dcache, bonus, P)
        inp = mx.concatenate([bonus.reshape(1), drafts])[None]
        t1 = time.perf_counter()
        if prof and rep == args.reps:
            prof.enable()
        logits, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
        tok_ = mx.argmax(logits, axis=-1).astype(mx.int32)
        if prof and rep == args.reps:
            prof.disable()
        t2 = time.perf_counter()
        mx.eval(tok_, cap, [c.state for c in cache], dcache.state)
        n = 3
        t3 = time.perf_counter()
        VF.commit(cache, rec, n)
        t4 = time.perf_counter()
        drafter.materialize(dcache, cap[:n], P)
        t5 = time.perf_counter()
        mx.eval([c.state for c in cache], dcache.state)
        bonus, P = tok_[n - 1], P + n
        if rep:
            for k, v in zip(times, (t1 - t0, t2 - t1, t4 - t3, t5 - t4)):
                times[k].append(1e3 * v)
    for k, v in times.items():
        v.sort()
        print(f"{k:12s} build {v[len(v) // 2]:6.2f} ms (median of {len(v)})")
    if prof:
        pstats.Stats(prof).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
