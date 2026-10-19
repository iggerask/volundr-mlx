"""Non-matmul pieces of one 2048-token prefill chunk, each timed over its layer count."""
import time, math
import mlx.core as mx
import volundr_synth as V
from mlx_lm.models.gated_delta import gated_delta_kernel
BF, T = mx.bfloat16, 2048
def bench(f, K, iters=3):
    mx.eval([f() for _ in range(K)]); mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters): mx.eval([f() for _ in range(K)])
    mx.synchronize(); return (time.perf_counter() - t) / iters      # seconds for K calls
r = lambda *s, dt=BF: (mx.random.normal(s) * 0.02).astype(dt)
H, hm, x, y = r(T, 4, 5120), r(T, 5120), r(T, 5120), r(T, 5120)
Wd, wn = r(24, 5120), r(5120)
ps, pp, rs = r(4, dt=mx.float32), r(4, dt=mx.float32), r(16, dt=mx.float32)
mx.eval(H, hm, x, y, Wd, wn, ps, pp, rs)
C = V.compiled
def mhc():
    d = (mx.fast.rms_norm(hm, wn, 1e-6) @ Wd.T).astype(mx.float32)
    x_in, post, res = C(V._mhc_pre)(d, H, ps, pp, rs)
    M = V.sinkhorn_kernel(res)
    return C(V._mhc_post)(M, H, post, y, x_in)
res = {}
res["mHC pre+Sinkhorn+post (x72)"] = bench(mhc, 72)
res["2x Gemma RMSNorm + residual adds (x72)"] = bench(lambda: mx.fast.rms_norm(x + y, wn, 1e-6) + mx.fast.rms_norm(y, wn, 1e-6) + x, 72)
gu = r(T, 2 * 17408); mx.eval(gu)
res["SwiGLU on gate_up output (x72)"] = bench(lambda: C(V._swiglu)(gu), 72)
xc = r(1, T + 3, 10240); cw = r(10240, 4, 1); mx.eval(xc, cw)
res["KDA depthwise conv1d k=4 + SiLU (x54)"] = bench(lambda: V.nn_silu(mx.conv1d(xc, cw, groups=10240)), 54)
f = r(1, T, 48, 128, dt=mx.float32); dtb = r(48, 128, dt=mx.float32); an = -mx.exp(r(48, 1, dt=mx.float32)); b = r(1, T, 48)
q = r(1, T, 16, 128); k = r(1, T, 16, 128); v = r(1, T, 48, 128); z = r(1, T, 48, 128); on = r(128)
mx.eval(f, dtb, an, b, q, k, v, z, on)
def kda_small():
    g, beta = C(V._kda_gates)(f, dtb, an, b)
    qq = mx.fast.rms_norm(q, None, 1e-6) * (1 / 128); kk = mx.fast.rms_norm(k, None, 1e-6) * (1 / math.sqrt(128))
    o = C(V._gated_out)(mx.fast.rms_norm(v, on, 1e-6), z)
    return g, beta, qq, kk, o
res["KDA gates, q/k L2 norm, gated output norm (x54)"] = bench(kda_small, 54)
st = mx.zeros((1, 48, 128, 128), dtype=mx.float32); g, beta = C(V._kda_gates)(f, dtb, an, b); mx.eval(st, g, beta)
qq = mx.fast.rms_norm(q, None, 1e-6) * (1 / 128); kk = mx.fast.rms_norm(k, None, 1e-6) * (1 / math.sqrt(128)); mx.eval(qq, kk)
res["KDA recurrence kernel, T=2048 (x54)"] = bench(lambda: gated_delta_kernel(qq, kk, v, g.astype(BF), beta, st)[0], 54)
qa = r(1, T, 24, 256); ka = r(1, T, 4, 256); qn = r(256); mx.eval(qa, ka, qn)
def attn_pre():
    q2 = mx.fast.rope(mx.fast.rms_norm(qa, qn, 1e-6).transpose(0, 2, 1, 3), 64, traditional=False, base=1e7, scale=1.0, offset=0)
    k2 = mx.fast.rope(mx.fast.rms_norm(ka, qn, 1e-6).transpose(0, 2, 1, 3), 64, traditional=False, base=1e7, scale=1.0, offset=0)
    return q2, k2
res["attn q/k norm + RoPE (x18)"] = bench(attn_pre, 18)
qs = r(1, 24, T, 256); ks = r(1, 4, T, 256); vs = r(1, 4, T, 256); mx.eval(qs, ks, vs)
res["first-chunk causal SDPA (x18)"] = bench(lambda: mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=1 / 16, mask="causal"), 18)
tot = 0
for kname, t in res.items():
    tot += t
    print(f"{kname:50s} {t*1e3:7.1f} ms per chunk")
print(f"{'sum of non-matmul pieces':50s} {tot*1e3:7.1f} ms per chunk")
