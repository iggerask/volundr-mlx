"""Per-kernel cost of tiny dependent ops on this GPU."""
import time
import mlx.core as mx

x0 = mx.random.normal((5120,)).astype(mx.bfloat16)
w = mx.random.normal((5120,)).astype(mx.float32)
mx.eval(x0, w)


def chain(n, kind):
    x = x0
    for i in range(n):
        if kind == "add":
            x = x + 1
        elif kind == "astype2":
            x = x.astype(mx.float32).astype(mx.bfloat16)
        elif kind == "norm":
            x = mx.fast.rms_norm(x, None, 1e-6)
    return x


def bench(n, kind, reps=20):
    best = 1e9
    for _ in range(reps):
        y = chain(n, kind)
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(y)
        mx.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


for kind in ("add", "astype2", "norm"):
    for n in (100, 500, 1000):
        k = n * (2 if kind == "astype2" else 1)
        dt = bench(n, kind)
        print(f"{kind:8s} n={n:5d}: {dt * 1e3:7.2f} ms total, {dt / k * 1e6:5.2f} us per kernel", flush=True)

# independent ops (no dependency): many adds of x0
for n in (500, 1000):
    best = 1e9
    for _ in range(10):
        ys = [x0 + i for i in range(n)]
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(ys)
        mx.synchronize()
        best = min(best, time.perf_counter() - t)
    print(f"indep add n={n}: {best * 1e3:.2f} ms, {best / n * 1e6:.2f} us per kernel")
