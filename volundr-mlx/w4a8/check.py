"""Correctness of the W4A8 kernels: (a) bit-level arithmetic against a CPU fp32 emulation of the same formula,
(b) end error against the fp32 product x @ dequant(w).T (CPU), next to the stock bf16 path's error.

usage: python check.py
"""
import sys
import mlx.core as mx
import w4a8

GS = 128


def emulate(x, w, s, b, ag=128):
    """The kernel's formula in fp32 ops on the CPU: sum_g s * (sum_j sx_j * D_j + d8) + b * xs, with activation
    groups j of ag (128 or 32) inside each 128-group g."""
    with mx.stream(mx.cpu):
        M, K = x.shape
        G, J = K // GS, GS // ag
        xf = x.astype(mx.float32).reshape(M, G, J, ag)
        amax = mx.abs(xf).max(-1)
        sx = amax / 127.0
        inv = mx.where(amax > 0, 127.0 / amax, 0.0)
        xq = mx.clip(mx.round(xf * inv[..., None]), -127, 127)
        xs = xf.sum((-2, -1))
        d8 = 8.0 * (xs - (sx * xq.sum(-1)).sum(-1))
        q = mx.dequantize(w, mx.ones_like(s), mx.zeros_like(b), group_size=GS, bits=4).astype(mx.float32)
        D = mx.einsum("mgjk,ngjk->gjmn", xq, q.reshape(-1, G, J, ag))  # exact in fp32 (|D| < 2^24)
        sxD = (sx.transpose(1, 2, 0)[..., None] * D).sum(1)  # [G, M, N]
        sf, bf = s.astype(mx.float32).T[:, None, :], b.astype(mx.float32).T[:, None, :]
        y = (sf * (sxD + d8.T[:, :, None]) + bf * xs.T[:, :, None]).sum(0)
        return mx.eval(y) or y


def ref_fp32(x, w, s, b):
    with mx.stream(mx.cpu):
        y = x.astype(mx.float32) @ mx.dequantize(w, s, b, group_size=GS, bits=4).astype(mx.float32).T
        return mx.eval(y) or y


def rel(a, ref):
    with mx.stream(mx.cpu):
        a, ref = a.astype(mx.float32), ref.astype(mx.float32)
        err = a - ref
        return (f"rms {(mx.sqrt(mx.mean(err * err)) / mx.sqrt(mx.mean(ref * ref))).item():.2e} "
                f"max {(mx.abs(err).max() / mx.abs(ref).max()).item():.2e}")


def main():
    mx.random.seed(0)
    cfgs = [dict(ag=5), dict(ag=5, wm=4, wn=1), dict(ag=4), dict(ag=4, wm=4, wn=1), dict(ag=3), dict(ag=1), dict(ag=2), dict(ag=1, wm=4, wn=1), dict(ag=2, sm=64), dict(plain=True), dict(plain=True, brow=True), dict(plain=True, sm=64), dict(plain=True, sn=64),
            dict(plain=True, sm=64, brow=True), dict(plain=True, sm=64, sn=64), dict(), dict(sbt=True), dict(wm=4, wn=1), dict(brow=False), dict(brow=False, pf=2), dict(brow=False, sbt=True), dict(pf=False, bvec=False), dict(bvec=False, sbt=True), dict(btg=True, pf=False),
            dict(sm=64, sn=32, wm=2, wn=2, sbt=True), dict(sm=32, sn=64, wm=4, wn=1, sbt=True)]
    for (M, N, K, outl) in [(256, 512, 1024, 0), (200, 256, 384, 1), (2048, 512, 5120, 0)]:
        x = mx.random.normal((M, K))
        if outl:
            x = x * (1 + 4 * (mx.random.uniform(shape=(1, K)) > 0.98))
        x = x.astype(mx.bfloat16)
        wf = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)  # bf16 scales / biases, as in the model
        w, s, b = mx.quantize(wf, group_size=GS, bits=4)
        ref = ref_fp32(x, w, s, b)
        emu = emulate(x, w, s, b)
        emu32 = emulate(x, w, s, b, ag=32)
        emu64 = emulate(x, w, s, b, ag=64)
        stock = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=4)
        print(f"M={M} N={N} K={K} outliers={outl}: stock vs fp32 {rel(stock, ref)}; W4A8 emulation vs fp32 {rel(emu, ref)}"
              f"; 32-wide emulation vs fp32 {rel(emu32, ref)}")
        for c in cfgs:
            y = w4a8.qmm(x, w, s, b, **c)
            e = emu64 if c.get("ag") == 5 else emu32 if c.get("ag") else emu
            print(f"   {c}: kernel vs emulation {rel(y, e)}; vs fp32 {rel(y, ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
