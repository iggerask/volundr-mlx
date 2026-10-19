"""Correctness of the 2/3-bit kernel against mx.dequantize + matmul (fp32)."""
import sys
import mlx.core as mx
import qmm_nax_lowbit as L

BF = mx.bfloat16
for bits in (2, 3):
    for (n, k) in [(5120, 6144), (10240, 5120)]:
        w, s, b = mx.quantize(mx.random.normal((n, k)).astype(BF) * 0.02, group_size=128, bits=bits)
        wd = mx.dequantize(w, s, b, group_size=128, bits=bits).astype(mx.float32)
        wr = L.repack(w, bits, k)
        for m in (1, 8, 16):
            x = mx.random.normal((m, k)).astype(BF)
            ref = x.astype(mx.float32) @ wd.T
            st = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=bits).astype(mx.float32)
            out = [("stock", st)]
            for dt in ("bf16", "i8"):
                for nt in (32, 64):
                    y = L.qmm_lowbit_prepped(L.prep_x(x, dt), wr, s, L.centred_bias(s, b, bits), bits, dt, nt=nt).astype(mx.float32)
                    out.append((f"{dt}/nt{nt}", y))
            sc = mx.abs(ref).max()
            print(f"{bits}-bit {n}x{k} M={m:2d} " + "  ".join(
                f"{nm} {(mx.abs(y - ref).max() / sc).item():.1e}" for nm, y in out), flush=True)
