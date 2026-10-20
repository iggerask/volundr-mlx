"""GPU latency of the decode small-op sequences, old vs fused, in dependency chains over 72 (mHC) / 54 (KDA) layers
with real-size weights, N = 1. Interleaved, best of reps."""
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import compute_g

import decode_fused as DF
import kernels

rng = np.random.default_rng(0)
BF, F32 = mx.bfloat16, mx.float32
bf = lambda *s, sc=1.0: (mx.array(rng.standard_normal(s).astype(np.float32)) * sc).astype(BF)
D, NL = 5120, 72
dyn = [nn.Linear(D, 24, bias=False) for _ in range(NL)]
for l in dyn:
    l.weight = bf(24, D, sc=0.01)
stat = [(bf(4) * 0.3, bf(4) * 0.3 + 1, bf(4, 4) + 4 * mx.eye(4).astype(BF)) for _ in range(NL)]
wn = [bf(D, sc=0.1) + 1 for _ in range(NL)]
wi = [bf(D, sc=0.1) for _ in range(NL)]
mx.eval([l.weight for l in dyn], stat, wn, wi)
H0 = bf(1, 4, D)
x0 = bf(1, D)
_, post0, res0 = kernels.mhc_params(bf(1, 24), *stat[0], 10)
mx.eval(H0, x0, post0, res0)


def mhc_old():
    H, post, res, y, x_in = H0, post0, res0, x0, x0
    for i in range(NL):
        H, mean = kernels.mhc_combine(H, res, post, y, x_in)
        hn = kernels.rms_norm(mean, wn[i], 1e-6, offset=False)[1]
        d = dyn[i](hn)
        pre, post, res = kernels.mhc_params(d, *stat[i], 10)
        x_in = kernels.mhc_mix(H, pre)
        y = kernels.rms_norm(x_in, wi[i], 1e-6, offset=True)[1]
    return [H, y]


def mhc_new():
    H, post, res, y, x_in = H0, post0, res0, x0, x0
    for i in range(NL):
        H, hn = DF.combine_norm(H, res, post, y, x_in, wn[i])
        d = dyn[i](hn)
        x_in, y, post, res = DF.mix_norm(d, *stat[i], H, 10, w=wi[i])
    return [H, y]


def norm_only():  # 2 x 72 single-threadgroup norms (the old mhc and input norms alone)
    y = x0
    for i in range(NL):
        y = kernels.rms_norm(y, wn[i], 1e-6, offset=False)[1]
        y = kernels.rms_norm(y, wi[i], 1e-6, offset=True)[1]
    return [y]


def params_only():  # 72 serial-Sinkhorn kernels chained through d
    d = bf(1, 24)
    for i in range(NL):
        pre, post, res = kernels.mhc_params(d, *stat[i], 10)
        d = mx.concatenate([pre, post, res.reshape(1, 16)], axis=1)
    return [d]


def params_new_only():
    d = bf(1, 24)
    for i in range(NL):
        x, _, post, res = DF.mix_norm(d, *stat[i], H0, 10)
        d = mx.concatenate([x[:, :4], post, res.reshape(1, 16)], axis=1)
    return [d]


def concat_only():
    d = bf(1, 24)
    for i in range(NL):
        d = mx.concatenate([d[:, :4], d[:, 4:8], d[:, 8:]], axis=1)
    return [d]


# KDA small-op sequence
Hk, Dk, HV = 16, 128, 48
KD = Hk * Dk
C = 2 * KD + HV * Dk
NK = 54
cw = [mx.array(rng.standard_normal((C, 4)).astype(np.float32) * 0.5) for _ in range(NK)]
Al = [mx.array(rng.standard_normal(HV).astype(np.float32)) for _ in range(NK)]
dtb = [mx.array(rng.standard_normal(HV * Dk).astype(np.float32)) for _ in range(NK)]
gw = [bf(Dk, sc=0.1) + 1 for _ in range(NK)]
st0 = [mx.array(rng.standard_normal((1, HV, Dk, Dk)).astype(np.float32) * 0.1) for _ in range(NK)]
qkv0, cs0, a0, b0, z0 = bf(1, 1, C), bf(1, 3, C), bf(1, 1, HV * Dk), bf(1, 1, HV), bf(1, 1, HV, Dk)
mx.eval(cw, Al, dtb, gw, st0, qkv0, cs0, a0, b0, z0)


def kda_old():
    qkv, outs = qkv0, []
    for i in range(NK):
        q, k, v = kernels.kda_conv_qk4(qkv, cs0, cw[i], KD, Dk)
        cs = mx.concatenate([cs0, qkv], axis=1)[:, -3:]
        g = compute_g(Al[i].reshape(HV, 1), a0.reshape(1, 1, HV, Dk), dtb[i].reshape(HV, Dk))
        beta = mx.sigmoid(b0.astype(F32))
        y, s = kernels.gated_delta_vec(q.reshape(1, 1, Hk, Dk), k.reshape(1, 1, Hk, Dk), v.reshape(1, 1, HV, Dk), g,
                                       beta, st0[i])
        o = kernels.gated_norm(y.astype(BF), gw[i], z0, 1e-6)
        qkv = mx.concatenate([o.reshape(1, 1, -1), qkv[..., HV * Dk:]], axis=-1)  # next layer depends on this one
        outs += [s, cs]
    return outs + [qkv]


def kda_new():
    qkv, outs = qkv0, []
    for i in range(NK):
        q, k, v, cs, g, beta = DF.kda_conv_gates(qkv, cs0, cw[i], KD, Al[i], a0, dtb[i], b0)
        y, s = kernels.gated_delta_vec(q.reshape(1, 1, Hk, Dk), k.reshape(1, 1, Hk, Dk), v.reshape(1, 1, HV, Dk),
                                       g.reshape(1, 1, HV, Dk), beta, st0[i], out_dtype=BF)
        o = kernels.gated_norm(y, gw[i], z0, 1e-6)
        qkv = mx.concatenate([o.reshape(1, 1, -1), qkv[..., HV * Dk:]], axis=-1)
        outs += [s, cs]
    return outs + [qkv]


sys.path.insert(0, WT + "/tests")
import test_decode_fused as TD

_old = mx.fast.metal_kernel(name="chain_old_norm", input_names=["x", "w"], output_names=["y"],
                            source=TD._OLD_NORM_SRC.replace("EPS_VAL", "1e-06f").replace("LOAD", kernels._NORM_LOAD))


def old_norm(x, w, offset):
    return _old(inputs=[x, w], template=[("D", D), ("NT", 256), ("OFFSET", int(offset))], grid=(256, 1, 1),
                threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[BF])[0]


def norm_only_old():
    y = x0
    for i in range(NL):
        y = old_norm(y, wn[i], False)
        y = old_norm(y, wi[i], True)
    return [y]


def with_wide(f, nw):
    def g():
        kernels.NORM_WIDE, DF.WIDE = nw, nw
        try:
            return f()
        finally:
            kernels.NORM_WIDE, DF.WIDE = 1024, 1024
    return g


V = {"2 single-TG norms (strided loop)": (norm_only_old, NL),
     "2 single-TG norms (unrolled, NT=256)": (with_wide(norm_only, 0), NL),
     "2 single-TG norms (wide 512)": (with_wide(norm_only, 512), NL),
     "2 single-TG norms (wide 1024)": (with_wide(norm_only, 1024), NL),
     "mhc old sequence (NT=256 norms)": (with_wide(mhc_old, 0), NL),
     "mhc fused (NT=256)": (with_wide(mhc_new, 0), NL),
     "mhc fused (wide 512)": (with_wide(mhc_new, 512), NL),
     "mhc fused (wide 1024)": (with_wide(mhc_new, 1024), NL),
     "kda old": (kda_old, NK), "kda fused": (kda_new, NK)}
best = {}
for rep in range(12):
    for name, (f, n) in V.items():
        outs = f()
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(outs)
        mx.synchronize()
        best[name] = min(best.get(name, 1e9), (time.perf_counter() - t) / n)
for name in V:
    print(f"{name:36s} {best[name] * 1e6:7.1f} us per layer")
