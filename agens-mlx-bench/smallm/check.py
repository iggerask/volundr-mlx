"""Correctness of qmm_nax against mx.dequantize + matmul in fp32, for M = 1..16.

Reports max and RMS error relative to the reference, next to stock mx.quantized_matmul."""
import sys
import mlx.core as mx
from qmm_nax import qmm_nax, to_int4

BF = mx.bfloat16
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408), "kda_qkv": (10240, 5120), "out": (5120, 6144)}


def errs(y, ref):
    y = y.astype(mx.float32)
    return ((mx.abs(y - ref).max() / mx.abs(ref).max()).item(),
            (mx.sqrt(((y - ref) ** 2).mean()) / mx.sqrt((ref ** 2).mean())).item())


worst = {"stock": [0, 0], "bf16": [0, 0], "i8": [0, 0]}
for name in sys.argv[1:] or list(SHAPES):
    n, k = SHAPES[name]
    w, s, b = mx.quantize(mx.random.normal((n, k)).astype(BF) * 0.02, group_size=128, bits=4)
    w4 = to_int4(w, s, b)
    wd = mx.dequantize(w, s, b, group_size=128, bits=4).astype(mx.float32)
    for m in range(1, 17):
        x = mx.random.normal((m, k)).astype(BF)
        ref = x.astype(mx.float32) @ wd.T
        r = {"stock": errs(mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=4), ref),
             "bf16": errs(qmm_nax(x, w, s, b), ref),
             "i8": errs(qmm_nax(x, *w4, dtype="i8"), ref)}
        assert qmm_nax(x, w, s, b).shape == (m, n)
        for key, (mx_e, rms_e) in r.items():
            worst[key][0] = max(worst[key][0], mx_e)
            worst[key][1] = max(worst[key][1], rms_e)
        if m in (1, 8, 16):
            print(f"{name:8s} M={m:2d} " + "  ".join(f"{key} max {v[0]:.1e} rms {v[1]:.1e}" for key, v in r.items()))
print("worst over shapes and M=1..16: " +
      "  ".join(f"{key} max {v[0]:.1e} rms {v[1]:.1e}" for key, v in worst.items()))
