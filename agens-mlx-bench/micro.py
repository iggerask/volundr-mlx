"""Microbenchmarks for the ops a Volundr decode/prefill step needs, on this Mac (MLX).

Every timing queues K calls before one eval, as a real forward pass does, so the per-eval
GPU round trip is amortised and the number is the kernel's throughput cost.
"""
import sys, time
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel

BF = mx.bfloat16
SEC = set(sys.argv[1:]) or {"bw", "qmv", "qmm", "kda", "sdpa", "topk", "disp", "sync"}


def bench(f, K=1, iters=10, warmup=2):
    """Seconds per call of f, with K calls queued per eval."""
    def run():
        mx.eval([f() for _ in range(K)])
    for _ in range(warmup):
        run()
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        run()
    mx.synchronize()
    return (time.perf_counter() - t) / (iters * K)


def rot(items):
    i = [0]
    def nxt():
        i[0] = (i[0] + 1) % len(items)
        return items[i[0]]
    return nxt


def qweight(n, k, bits, group):
    w = mx.random.randint(0, 2**31 - 1, (n, k * bits // 32)).astype(mx.uint32)
    s = (mx.random.uniform(shape=(n, k // group)) * 0.01).astype(BF)
    b = (-s * 8).astype(BF)
    mx.eval(w, s, b)
    return w, s, b


def qbytes(n, k, bits, group):
    return n * k * bits / 8 + 2 * (n * k // group) * 2


if "sync" in SEC:
    x = mx.ones((16,))
    mx.eval(x)
    t = bench(lambda: x + 1, K=1, iters=200, warmup=20)
    print(f"[sync] one tiny op + eval round trip: {t*1e6:.0f} us")

if "bw" in SEC:
    xs = [mx.random.randint(0, 2**31 - 1, (2**27,)).astype(mx.uint32) for _ in range(4)]  # 4 x 512 MiB
    mx.eval(xs)
    nxt = rot(xs)
    t = bench(lambda: mx.sum(nxt()), K=4)
    print(f"[bw] streaming read (sum over 4 x 512 MiB): {2**29 / t / 1e9:.0f} GB/s")
    del xs

SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408), "kda_qkv": (10240, 5120), "out": (5120, 6144)}
CFGS = [(4, 128), (4, 64), (3, 128), (2, 128), (8, 64)]
MS = [1, 2, 4, 5, 8, 16]

if "qmv" in SEC:
    print("[qmv] effective GB/s at M=1, ms, then time(M)/time(1) for M in", MS[1:])
    for bits, group in CFGS:
        for name, (n, k) in SHAPES.items():
            copies = max(4, int(1.5e9 // qbytes(n, k, bits, group)))
            ws = [qweight(n, k, bits, group) for _ in range(copies)]
            nxt = rot(ws)
            ts = []
            for m in MS:
                x = mx.random.normal((m, k)).astype(BF)
                mx.eval(x)
                def f():
                    w, s, b = nxt()
                    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits)
                ts.append(bench(f, K=copies))
            gbs = qbytes(n, k, bits, group) / ts[0] / 1e9
            print(f"  w{bits}g{group:<3} {name:8s} {gbs:5.0f} GB/s {ts[0]*1e3:6.3f} ms  ratios " +
                  " ".join(f"{t/ts[0]:.2f}" for t in ts[1:]))
            del ws

if "qmm" in SEC:
    M = 2048
    for bits, group in [(4, 128), (3, 128), (2, 128), (8, 64)]:
        for name in ("gate_up", "down"):
            n, k = SHAPES[name]
            ws = [qweight(n, k, bits, group) for _ in range(2)]
            nxt = rot(ws)
            x = mx.random.normal((M, k)).astype(BF)
            mx.eval(x)
            def f():
                w, s, b = nxt()
                return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits)
            t = bench(f, K=2, iters=5)
            print(f"[qmm] w{bits}g{group} {name} M={M}: {2*M*n*k/t/1e12:5.1f} TFLOP/s")
            del ws
    n, k = SHAPES["gate_up"]
    w = mx.random.normal((n, k)).astype(BF)
    x = mx.random.normal((M, k)).astype(BF)
    mx.eval(w, x)
    t = bench(lambda: x @ w.T, K=2, iters=5)
    print(f"[qmm] bf16 gate_up M={M}: {2*M*n*k/t/1e12:5.1f} TFLOP/s")
    del w

if "kda" in SEC:
    Hk, Hv, D = 16, 48, 128
    sts = [mx.zeros((1, Hv, D, D), dtype=mx.float32) for _ in range(54)]
    mx.eval(sts)
    for T in (1, 8, 2048):
        q = mx.random.normal((1, T, Hk, D)).astype(BF) * 0.08
        k = mx.random.normal((1, T, Hk, D)).astype(BF) * 0.08
        v = mx.random.normal((1, T, Hv, D)).astype(BF)
        g = mx.random.uniform(0.9, 1.0, (1, T, Hv, D)).astype(BF)
        beta = mx.random.uniform(shape=(1, T, Hv)).astype(BF)
        mx.eval(q, k, v, g, beta)
        nxt = rot(sts)
        t = bench(lambda: gated_delta_kernel(q, k, v, g, beta, nxt())[0], K=54, iters=5 if T > 8 else 20)
        ideal = 2 * 54 * Hv * D * D * 4 / 248e9
        print(f"[kda] fused per-channel kernel T={T}: {t*1e3:7.3f} ms/layer  x54 = {54*t*1e3:7.2f} ms"
              + (f"  ({T/(54*t):.0f} tok/s if alone)" if T > 8 else f"  (state-traffic floor {ideal*1e3:.2f} ms)"))

if "sdpa" in SEC:
    for L in (4096 + 512, 65536, 219000):
        kk = mx.random.normal((1, 4, L, 256)).astype(BF)
        vv = mx.random.normal((1, 4, L, 256)).astype(BF)
        mx.eval(kk, vv)
        kvb = 2 * L * 4 * 256 * 2
        for T in (1, 8):
            q = mx.random.normal((1, 24, T, 256)).astype(BF)
            mx.eval(q)
            t = bench(lambda: mx.fast.scaled_dot_product_attention(q, kk, vv, scale=1 / 16), K=17)
            print(f"[sdpa] q={T} keys={L:6d}: {t*1e3:6.3f} ms  ({kvb/t/1e9:4.0f} GB/s KV)")
        qf = mx.random.normal((1, 24 * 8, 1, 256)).astype(BF)
        mx.eval(qf)
        t = bench(lambda: mx.fast.scaled_dot_product_attention(qf, kk, vv, scale=1 / 16), K=17)
        print(f"[sdpa] q=8 folded into heads (1 x 192 heads) keys={L:6d}: {t*1e3:6.3f} ms  ({kvb/t/1e9:4.0f} GB/s KV)")
        del kk, vv

if "topk" in SEC:
    for nb in (1024, 16384, 55000):
        for rows in (1, 8):
            bk = mx.random.normal((nb, 128)).astype(BF)
            iq = mx.random.normal((rows, 4, 128)).astype(BF)
            kv = mx.random.normal((nb * 4, 4, 256)).astype(BF)
            mx.eval(bk, iq, kv)
            def f():
                s = (iq @ bk.T).mean(axis=1)
                idx = mx.argpartition(-s, 512 - 1, axis=-1)[:, :512]
                rowsel = (idx[..., None] * 4 + mx.arange(4)).reshape(-1)
                return mx.take(kv, rowsel, axis=0).reshape(rows, 512, 4, 4, 256).mean(axis=2)
            t = bench(f, K=17)
            print(f"[bcsa-far] blocks={nb:6d} queries={rows}: score+top512+gather+pool {t*1e3:6.3f} ms/layer")

if "disp" in SEC:
    x = mx.ones((5120,), dtype=BF)
    mx.eval(x)
    def chain(y, n=1000):
        for _ in range(n):
            y = y * 1.0001
        return y
    t = bench(lambda: chain(x))
    print(f"[disp] dependent tiny elementwise ops, eager: {t/1000*1e6:.2f} us/op")
    cf = mx.compile(chain)
    t = bench(lambda: cf(x))
    print(f"[disp] same chain under mx.compile (fused): {t/1000*1e6:.2f} us/op-equivalent")
    W = mx.random.normal((24, 5120)).astype(BF)
    mx.eval(W)
    def chain2(y, n=250):
        for _ in range(n):
            y = y + (W @ y).sum() * 1e-6  # matvec, sum, mul, add
        return y
    t = bench(lambda: chain2(x))
    print(f"[disp] dependent tiny matvec/reduce/elementwise mix: {t/250/4*1e6:.2f} us/op")
