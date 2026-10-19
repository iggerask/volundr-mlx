"""Correctness of the fused prep kernels (prep_nax.py) and the 8-bit kernel (qmm_nax_u8.py).

Each prep variant is compared with the stock MLX ops followed by qmm_nax.prep_x(); the u8 kernel
with mx.quantized_matmul and an fp32 dequantized reference. Errors are relative to the reference's
max magnitude (max) and RMS (rms)."""
import mlx.core as mx
import qmm_nax as Q
import prep_nax as P
from qmm_nax_u8 import qmm_nax_u8_prepped
from common import wait_gpu_idle

BF = mx.bfloat16


def errs(y, ref):
    y, ref = y.astype(mx.float32), ref.astype(mx.float32)
    return ((mx.abs(y - ref).max() / mx.maximum(mx.abs(ref).max(), 1e-30)).item(),
            (mx.sqrt(((y - ref) ** 2).mean()) / mx.maximum(mx.sqrt((ref ** 2).mean()), 1e-30)).item())


def same(name, px, ref_x):
    M, xp, xs, _ = px
    _, rp, rs, _ = Q.prep_x(ref_x)
    assert xp.shape == rp.shape and xs.shape == rs.shape, (name, xp.shape, rp.shape)
    ex, es = errs(xp, rp), errs(xs, rs)
    pad = mx.abs(xp[M:].astype(mx.float32)).max().item() if M < 16 else 0.0
    print(f"{name:28s} x max {ex[0]:.1e} rms {ex[1]:.1e} | sums max {es[0]:.1e} rms {es[1]:.1e}"
          f" | pad rows {pad:.0e}")
    assert ex[1] < 4e-3 and es[1] < 4e-3 and pad == 0.0, name


def silu(x):
    return x * mx.sigmoid(x)


wait_gpu_idle()
mx.random.seed(0)
for M in (2, 8, 16):
    for K in (1024, 5120, 17408):
        x = mx.random.normal((M, K)).astype(BF)
        w = (1 + 0.1 * mx.random.normal((K,))).astype(BF)
        same(f"plain M={M} K={K}", P.prep(x), x)
        same(f"rms M={M} K={K}", P.rms_prep(x, w), mx.fast.rms_norm(x, w, 1e-6))
        gu = mx.random.normal((M, 2 * K)).astype(BF)
        same(f"swiglu M={M} K={K}", P.swiglu_prep(gu), silu(gu[:, :K]) * gu[:, K:])
    xs = mx.random.normal((M, 2 * 5120)).astype(BF)
    same(f"plain strided M={M}", P.prep(xs[:, :5120]), xs[:, :5120])
    # attention: SDPA output (1, H, T, D) transposed, gate = second half of each q head
    o = mx.random.normal((1, 24, M, 256)).astype(BF)
    qg = mx.random.normal((1, M, 24, 512)).astype(BF)
    ot, gate = o.transpose(0, 2, 1, 3)[0], qg[0, :, :, 256:]
    same(f"sgate M={M}", P.sgate_prep(ot, gate), (ot * mx.sigmoid(gate)).reshape(M, -1))
    # KDA: per-head rms_norm then o * silu(z)
    ok = mx.random.normal((M, 48, 128)).astype(BF)
    wn = (1 + 0.1 * mx.random.normal((128,))).astype(BF)
    z = mx.random.normal((M, 48 * 128)).astype(BF)
    ref = (mx.fast.rms_norm(ok, wn, 1e-6) * silu(z.reshape(M, 48, 128))).reshape(M, -1)
    same(f"kdaout M={M}", P.kda_out_prep(ok, wn, z), ref)

# end to end through the 4-bit kernel: rms_prep + qmm_nax vs rms_norm + stock
n, k = 10240, 5120
wq, sq, bq = mx.quantize(mx.random.normal((n, k)).astype(BF) * 0.02, group_size=128, bits=4)
for M in (2, 8, 16):
    x = mx.random.normal((M, k)).astype(BF)
    w = (1 + 0.1 * mx.random.normal((k,))).astype(BF)
    ref = mx.quantized_matmul(mx.fast.rms_norm(x, w, 1e-6), wq, sq, bq, transpose=True, group_size=128, bits=4)
    e = errs(Q.qmm_nax_prepped(P.rms_prep(x, w), wq, sq, bq), ref)
    print(f"rms_prep + qmm_nax vs stock M={M}: max {e[0]:.1e} rms {e[1]:.1e}")
    assert e[1] < 5e-3

# 8-bit g64 lm_head-shaped kernel
n, k = 8192, 5120
w8, s8, b8 = mx.quantize(mx.random.normal((n, k)).astype(BF) * 0.02, group_size=64, bits=8)
wd = mx.dequantize(w8, s8, b8, group_size=64, bits=8).astype(mx.float32)
for M in (1, 2, 8, 16):
    x = mx.random.normal((M, k)).astype(BF)
    ref = x.astype(mx.float32) @ wd.T
    es = errs(mx.quantized_matmul(x, w8, s8, b8, transpose=True, group_size=64, bits=8), ref)
    en = errs(qmm_nax_u8_prepped(P.prep(x, gs=64), w8, s8, b8), ref)
    print(f"u8 g64 M={M:2d}: stock max {es[0]:.1e} rms {es[1]:.1e} | nax max {en[0]:.1e} rms {en[1]:.1e}")
    assert en[1] < 2 * es[1] + 1e-3
print("all prep/u8 checks passed")
