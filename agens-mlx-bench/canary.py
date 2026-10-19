import time, mlx.core as mx
a = mx.random.normal((8192, 8192)).astype(mx.bfloat16); b = mx.random.normal((8192, 8192)).astype(mx.bfloat16)
mx.eval(a, b); mx.eval(a @ b); mx.synchronize()
t = time.perf_counter(); mx.eval([a @ b for _ in range(8)]); mx.synchronize()
print(f"canary {8*2*8192**3/(time.perf_counter()-t)/1e12:.1f} TFLOP/s", flush=True)
