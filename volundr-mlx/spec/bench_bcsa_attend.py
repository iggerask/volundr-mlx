"""Micro-benchmark of the verify BCSA attention kernels on synthetic data shaped like the model's (H 24, KVH 4,
D 256, W 4096, C 4, top-512), 17 calls per eval (one per BCSA layer).

usage (from spec/, under ../with_gpu_lock.sh): python bench_bcsa_attend.py [--P 23632] [--union 1565] [--T 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bcsa_verify  # noqa: E402
import kernels  # noqa: E402

H, KVH, D, W, C, TOPK, NL = 24, 4, 256, 4096, 4, 512, 17


def timed(f, reps=15):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return round(1e3 * ts[len(ts) // 2] / NL, 4)  # ms per layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=23632)
    ap.add_argument("--union", type=int, default=1565)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--ntg", default="8,16,32")
    args = ap.parse_args()
    P, T = args.P, args.T
    rng = np.random.default_rng(0)
    bf = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)
    q, kn, vn = bf(H, T, D), bf(KVH, T, D), bf(KVH, T, D)
    kr, vr = bf(KVH, W + TOPK, D), bf(KVH, W + TOPK, D)
    nv_max = max(0, (P + T - 1 - W + 1) // C)
    cap = (nv_max + 1023) // 1024 * 1024 + 1024
    pk, pv = bf(KVH, cap, D), bf(KVH, cap, D)
    rec = {"P": P, "T": T, "nv_max": nv_max}
    if nv_max:
        pool = rng.permutation(nv_max)[: min(args.union, nv_max)]
        sel = np.stack([np.sort(rng.choice(pool, min(TOPK, len(pool)), replace=False)) for _ in range(T)]).astype(np.int32)
        sel = mx.array(sel)
        uidx, ucnt, umask = kernels.far_union(sel, nv_max, 32)
        mx.eval(uidx, ucnt, umask)
        rec["union"] = int(ucnt[0].item())
    mx.eval(q, kn, vn, kr, vr, pk, pv)
    far = (pk, pv, uidx, ucnt, umask) if nv_max else ()
    for ntg in map(int, args.ntg.split(",")):
        rec[f"attend_ntg{ntg}"] = timed(lambda: mx.eval([bcsa_verify.attend(q, kn, vn, kr, vr, P, W, D**-0.5, *far, ntg=ntg)
                                                         for _ in range(NL)]))
        rec[f"window_ntg{ntg}"] = timed(lambda: mx.eval([bcsa_verify.attend(q, kn, vn, kr, vr, P, W, D**-0.5, ntg=ntg)
                                                         for _ in range(NL)]))
    if nv_max:
        rec["far_union"] = timed(lambda: mx.eval([kernels.far_union(sel, nv_max, 32)[0] for _ in range(NL)]))
        # the prefill-kernel path (window gathered + concatenated, then bcsa_attend)
        nprev = min(P, W - 1)
        idx = mx.array((np.arange(P - nprev, P) % W).astype(np.int32))

        def old():
            keys = mx.concatenate([mx.take(kr, idx, axis=1), kn], axis=1)
            vals = mx.concatenate([mx.take(vr, idx, axis=1), vn], axis=1)
            return kernels.bcsa_attend(q, keys, vals, pk, pv, sel, nv_max, D**-0.5, nprev, W)
        rec["old_total"] = timed(lambda: mx.eval([old() for _ in range(NL)]))
    # bandwidth floor: one pass over the window and union K/V
    n_keys = min(P, W) + rec.get("union", 0)
    rec["kv_mb"] = round(KVH * n_keys * D * 2 * 2 / 1e6, 2)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
