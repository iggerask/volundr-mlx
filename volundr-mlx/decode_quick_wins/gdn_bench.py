"""Decode-shape KDA recurrence: 54 layers' worth of independent states, chained through the outputs like a decode step
(each call waits for the previous one), per-call time for variants."""
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx
import numpy as np

import decode_fused as DF
import kernels

B, T, Hk, Hv, Dk = 1, 1, 16, 48, 128
NL = 54
rng = np.random.default_rng(0)
f = lambda *s, sc=1.0: mx.array(rng.standard_normal(s).astype(np.float32) * sc)
states = [f(B, Hv, Dk, Dk, sc=0.1) for _ in range(NL)]
q, k = f(B, T, Hk, Dk, sc=0.1), f(B, T, Hk, Dk, sc=0.1)
v = f(B, T, Hv, Dk).astype(mx.bfloat16)
g = mx.sigmoid(f(B, T, Hv, Dk) + 2)
beta = mx.sigmoid(f(B, T, Hv))
z = f(B, T, Hv, Dk).astype(mx.bfloat16)
w = (f(Dk) * 0.1 + 1).astype(mx.bfloat16)
mx.eval(states, q, k, v, g, beta, z, w)


def chain(variant):
    vv = v
    outs = []
    for s in states:
        if variant == "vec+astype+gnorm":
            y, st = kernels.gated_delta_vec(q, k, vv, g, beta, s)
            o = kernels.gated_norm(y.astype(mx.bfloat16), w, z, 1e-6)
        elif variant == "vec(bf16)+gnorm":
            y, st = kernels.gated_delta_vec(q, k, vv, g, beta, s, out_dtype=mx.bfloat16)
            o = kernels.gated_norm(y, w, z, 1e-6)
        elif variant == "vec only":
            o, st = kernels.gated_delta_vec(q, k, vv, g, beta, s, out_dtype=mx.bfloat16)
        elif variant == "fused gdn_norm":
            o, st = DF.gated_delta_norm(q, k, vv, g, beta, s, w, z, 1e-6)
        vv = o.reshape(vv.shape)  # the next call depends on this one
        outs.append(st)
    return vv, outs


variants = ["vec+astype+gnorm", "vec(bf16)+gnorm", "vec only"] + (["fused gdn_norm"] if hasattr(DF, "gated_delta_norm") else [])
best = {}
for rep in range(15):
    for var in variants:
        y, sts = chain(var)
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(y, sts)
        mx.synchronize()
        dt = (time.perf_counter() - t) / NL
        best[var] = min(best.get(var, 1e9), dt)
for var in variants:
    print(f"{var:20s} {best[var] * 1e6:6.1f} us per layer ({6.29e6 / best[var] / 1e9:5.0f} GB/s state r+w)")
# exactness of the fused kernel
if "fused gdn_norm" in variants:
    y, st = kernels.gated_delta_vec(q, k, v, g, beta, states[0])
    o = kernels.gated_norm(y.astype(mx.bfloat16), w, z, 1e-6)
    o2, st2 = DF.gated_delta_norm(q, k, v, g, beta, states[0], w, z, 1e-6)
    print("fused == vec + gnorm:", mx.array_equal(o, o2).item(), mx.array_equal(st, st2).item())
