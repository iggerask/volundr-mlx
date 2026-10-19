"""Long-context prefill pieces for one 2048-token chunk at ~219K context (per layer)."""
import time, mlx.core as mx
BF = mx.bfloat16
def bench(f, K=1, iters=3):
    mx.eval([f() for _ in range(K)]); mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters): mx.eval([f() for _ in range(K)])
    mx.synchronize(); return (time.perf_counter() - t) / (iters * K)
T, NB, CTX = 2048, 54750, 219000
iq = mx.random.normal((T, 4, 128)).astype(BF)
bk = mx.random.normal((NB, 128)).astype(BF)
pk = mx.random.normal((NB, 4, 256)).astype(BF)
pv = mx.random.normal((NB, 4, 256)).astype(BF)
q = mx.random.normal((1, 24, T, 256)).astype(BF)
mx.eval(iq, bk, pk, pv, q)
def sel():
    s = iq.mean(axis=1) @ bk.T                      # mean of head dots == dot with mean query
    return mx.argpartition(-s, 511, axis=-1)[:, :512]
t_sel = bench(sel)
print(f"[far] score {T}x{NB} + per-query top-512: {t_sel*1e3:6.1f} ms/layer")
idx = sel(); mx.eval(idx)
def far_attend(tile=256):
    outs = []
    for t0 in range(0, T, tile):
        ii = idx[t0:t0 + tile].reshape(-1)
        K = mx.take(pk, ii, axis=0).reshape(tile, 512, 4, 256).transpose(0, 2, 1, 3)
        V = mx.take(pv, ii, axis=0).reshape(tile, 512, 4, 256).transpose(0, 2, 1, 3)
        qq = q[0, :, t0:t0 + tile, :].transpose(1, 0, 2)[:, :, None, :]   # [tile,24,1,256]
        outs.append(mx.fast.scaled_dot_product_attention(qq, K, V, scale=1 / 16))
    return mx.concatenate(outs, axis=0)
t_far = bench(far_attend)
print(f"[far] materialised gather + attend, 256-query tiles: {t_far*1e3:6.1f} ms/layer "
      f"(moves {2*T*512*4*256*2*2/1e9:.1f} GB)")
# window attention: one call over 6144 keys vs 512-query tiles over 4608 keys
kw = mx.random.normal((1, 4, 4096 + T, 256)).astype(BF); vw = mx.random.normal((1, 4, 4096 + T, 256)).astype(BF)
qi = mx.arange(T)[:, None] + 4096; ki = mx.arange(4096 + T)[None, :]
band = (ki <= qi) & (ki > qi - 4096)
mx.eval(kw, vw, band)
t_w1 = bench(lambda: mx.fast.scaled_dot_product_attention(q, kw, vw, scale=1 / 16, mask=band))
def wtiles(tile=512):
    outs = []
    for t0 in range(0, T, tile):
        k0, k1 = t0, t0 + 4096 + tile
        outs.append(mx.fast.scaled_dot_product_attention(q[:, :, t0:t0 + tile], kw[:, :, k0:k1], vw[:, :, k0:k1],
                                                         scale=1 / 16, mask=band[t0:t0 + tile, k0:k1]))
    return mx.concatenate(outs, axis=2)
t_w2 = bench(wtiles)
print(f"[win] band-masked SDPA, one call over 6144 keys: {t_w1*1e3:6.1f} ms/layer; 512-query tiles: {t_w2*1e3:6.1f} ms/layer")
kd = mx.random.normal((1, 4, CTX, 256)).astype(BF); vd = mx.random.normal((1, 4, CTX, 256)).astype(BF)
dm = mx.arange(CTX)[None, :] <= (mx.arange(T)[:, None] + CTX - T)
mx.eval(kd, vd, dm)
t_d = bench(lambda: mx.fast.scaled_dot_product_attention(q, kd, vd, scale=1 / 16, mask=dm), iters=2)
print(f"[dense] one layer, 2048 queries over {CTX} keys: {t_d*1e3:6.1f} ms "
      f"({2*2*24*T*(CTX-T/2)*256/t_d/1e12:.1f} TFLOP/s)")
