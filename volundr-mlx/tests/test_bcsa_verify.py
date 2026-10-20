"""bcsa_verify.attend (the T-query verify attention) against fp64 and against the prefill-kernel path it replaces.

run: python tests/test_bcsa_verify.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bcsa_verify  # noqa: E402
import kernels  # noqa: E402

RNG = np.random.default_rng(4321)


def bf(*sh, sc=1.0):
    return (mx.array(RNG.standard_normal(sh).astype(np.float32)) * sc).astype(mx.bfloat16)


def case(H, KVH, D, T, P, W, C, nb, topk, extra=8, p_empty=0.0, ntg=None):
    """Ring of W slots (+ extra far slots) holding positions < P, T new queries at P.., nb pooled blocks of which
    each query selects up to topk among its valid ones (or all of them when it has at most topk)."""
    G = H // KVH
    q, kn, vn = bf(H, T, D, sc=2.0), bf(KVH, T, D), bf(KVH, T, D)
    # the sequence's keys / values by position, and the ring as the cache holds them (slot = position % W)
    ks, vs = bf(KVH, P, D), bf(KVH, P, D)
    kr = np.zeros((KVH, W + extra, D), np.float32)
    vr = np.zeros((KVH, W + extra, D), np.float32)
    ksf, vsf = np.array(ks.astype(mx.float32)), np.array(vs.astype(mx.float32))
    for p in range(max(0, P - W), P):
        kr[:, p % W], vr[:, p % W] = ksf[:, p], vsf[:, p]
    kr[:, W:], vr[:, W:] = RNG.standard_normal((KVH, extra, D)), RNG.standard_normal((KVH, extra, D))  # stale far slots
    krm, vrm = mx.array(kr).astype(mx.bfloat16), mx.array(vr).astype(mx.bfloat16)
    cap = max(32, (nb + 31) // 32 * 32)
    pk, pv = bf(KVH, cap, D), bf(KVH, cap, D)
    nv_max = max(0, (P + T - 1 - W + 1) // C)
    assert nv_max <= nb
    sel = np.full((T, topk), -1, np.int32)
    for t in range(T):
        nv = max(0, (P + t - W + 1) // C)
        if nv and RNG.uniform() >= p_empty:
            pick = np.arange(nv) if nv <= topk else np.sort(RNG.choice(nv, topk, replace=False))
            sel[t, : len(pick)] = pick
    scale = D**-0.5
    if nv_max:
        uidx, ucnt, umask = kernels.far_union(mx.array(sel), nv_max, 32)
        got = bcsa_verify.attend(q, kn, vn, krm, vrm, P, W, scale, pk, pv, uidx, ucnt, umask, ntg=ntg)
    else:
        got = bcsa_verify.attend(q, kn, vn, krm, vrm, P, W, scale, ntg=ntg)
    # fp64 reference
    qd, knd, vnd, pkd, pvd = (np.array(a.astype(mx.float32), np.float64) for a in (q, kn, vn, pk, pv))
    want = np.zeros((H, T, D))
    for h in range(H):
        g = h // G
        for t in range(T):
            lo = max(0, P + t - W + 1)
            kk = np.concatenate([ksf[g, lo:P], knd[g, : t + 1], pkd[g][sel[t][sel[t] >= 0]]])
            vv = np.concatenate([vsf[g, lo:P], vnd[g, : t + 1], pvd[g][sel[t][sel[t] >= 0]]])
            s = kk @ qd[h, t] * scale
            p = np.exp(s - s.max())
            want[h, t] = p @ vv / p.sum()
    err = np.abs(np.array(got.astype(mx.float32)) - want)
    ok = (err <= 2**-7 * np.abs(want) + 2e-3).all()
    assert ok, (H, D, T, P, W, nv_max, err.max())
    # against the prefill kernels (window gathered out of the ring + concatenated, then bcsa_attend / bcsa_window)
    nprev = min(P, W - 1)
    if nprev:
        idx = mx.array((np.arange(P - nprev, P) % W).astype(np.int32))
        keys = mx.concatenate([mx.take(krm, idx, axis=1), kn], axis=1)
        vals = mx.concatenate([mx.take(vrm, idx, axis=1), vn], axis=1)
        if nv_max:
            old = kernels.bcsa_attend(q, keys, vals, pk, pv, mx.array(sel), nv_max, scale, nprev, W)
        else:
            old = kernels.bcsa_window(q, keys, vals, scale, nprev, W)
        d = mx.abs(got.astype(mx.float32) - old.astype(mx.float32))
        return float(d.max().item()), float((got != old).astype(mx.float32).mean().item())
    return 0.0, 0.0


def test_attend():
    cases = [
        # H, KVH, D, T, P, W, C, nb, topk
        (24, 4, 256, 8, 20000 // 40, 128, 4, 200, 16),   # far regime, ring wrapped
        (24, 4, 256, 8, 300, 128, 4, 50, 64),            # every valid block selected (nv <= topk)
        (24, 4, 256, 8, 90, 128, 4, 8, 16),              # ring not full, no far blocks
        (24, 4, 256, 8, 124, 128, 4, 8, 16),             # ring fills inside the verify; first far block at t >= 4
        (24, 4, 256, 1, 1000, 128, 4, 300, 32),
        (24, 4, 256, 3, 1000, 128, 4, 300, 32),
        (24, 4, 256, 16, 777, 256, 4, 200, 24),
        (8, 2, 64, 8, 5000, 512, 4, 1200, 128),          # many batches per split, unions over several words
        (8, 2, 64, 5, 40, 64, 4, 4, 8),
    ]
    for c in cases:
        for ntg in (None, 3):
            dmax, frac = case(*c, ntg=ntg)
            print(f"    {c} ntg={ntg}: vs prefill kernels max |diff| {dmax:.3g}, {frac:.2%} of outputs differ")
    # selections with empty rows (queries whose top-k is all -1 cannot occur, but the union must not assume more)
    case(24, 4, 256, 8, 600, 128, 4, 150, 32, p_empty=0.3)


if __name__ == "__main__":
    test_attend()
    print("ok")
