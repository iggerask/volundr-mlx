"""Prefill-shaped matmul and attention throughput. Run as is (NAX on) and with
MLX_METAL_GPU_ARCH=applegpu_g16s (NAX off: MLX then believes the GPU predates M5)."""
import os, time
import mlx.core as mx

print("arch override:", os.environ.get("MLX_METAL_GPU_ARCH", "(none)"))


def bench(f, K=2, iters=4):
    mx.eval([f() for _ in range(K)]); mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval([f() for _ in range(K)])
    mx.synchronize()
    return (time.perf_counter() - t) / (iters * K)


for dt in (mx.bfloat16, mx.float16):
    for (M, N, K) in ((8192, 8192, 8192), (2048, 34816, 5120), (2048, 5120, 17408), (4096, 34816, 5120)):
        a = mx.random.normal((M, K)).astype(dt)
        b = mx.random.normal((N, K)).astype(dt)
        mx.eval(a, b)
        t = bench(lambda: a @ b.T)
        print(f"[gemm] {str(dt):14s} M={M:5d} N={N:5d} K={K:5d}: {2*M*N*K/t/1e12:5.1f} TFLOP/s")

for (M, N, K) in ((2048, 34816, 5120), (2048, 5120, 17408), (4096, 34816, 5120), (8192, 34816, 5120)):
    w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.02
    wq, s, bb = mx.quantize(w, group_size=128, bits=4)
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    mx.eval(wq, s, bb, x)
    t = bench(lambda: mx.quantized_matmul(x, wq, s, bb, transpose=True, group_size=128, bits=4))
    print(f"[qmm ] w4g128 bf16 M={M:5d} N={N:5d} K={K:5d}: {2*M*N*K/t/1e12:5.1f} TFLOP/s")

# prefill attention: 24 q heads, 4 kv heads, D=256 (Volundr's gated attention)
for (qL, kL, kind) in ((2048, 6144, "bcsa window (bool mask)"), (2048, 2048, "causal self"), (2048, 65536, "dense, 64K ctx")):
    q = mx.random.normal((1, 24, qL, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, kL, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 4, kL, 256)).astype(mx.bfloat16)
    if kind.startswith("bcsa"):
        qi = mx.arange(qL)[:, None] + (kL - qL)
        ki = mx.arange(kL)[None, :]
        mask = (ki <= qi) & (ki > qi - 4096)
        mm = mask
    elif kind.startswith("causal"):
        mm = "causal"
    else:
        qi = mx.arange(qL)[:, None] + (kL - qL)
        mm = mx.arange(kL)[None, :] <= qi
    mx.eval(q, k, v)
    if not isinstance(mm, str):
        mx.eval(mm)
    t = bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 16, mask=mm))
    # useful flops: QK and PV over the unmasked keys
    if kind.startswith("bcsa"):
        keys = 4096 - 0  # each query sees ~4096 keys (window)
        useful = 2 * 2 * 24 * qL * keys * 256
    elif kind.startswith("causal"):
        useful = 2 * 2 * 24 * qL * (qL / 2) * 256
    else:
        useful = 2 * 2 * 24 * qL * (kL - qL / 2) * 256
    print(f"[sdpa] {kind:24s} q={qL} k={kL}: {t*1e3:7.1f} ms  {useful/t/1e12:5.1f} useful TFLOP/s")
