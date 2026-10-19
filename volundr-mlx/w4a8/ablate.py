"""Which part of the W4A8 GEMM costs the time? Ablation switches in w4a8.ABLATE (results are wrong, only timing).

usage: python ablate.py [--configs 32x32x2x2,64x32x2x2] [--N 17408] [--K 5120]
"""
import argparse
import mlx.core as mx
import w4a8
from bench_w4a8 import parse
from w4common import compare, spin_up

COMBOS = [()]
ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="32x32x2x2p,32x32x2x2q,32x32x2x2o")
ap.add_argument("--N", type=int, default=17408)
ap.add_argument("--K", type=int, default=5120)
ap.add_argument("--M", type=int, default=2048)
ap.add_argument("--ag", type=int, default=128, help="act_quant group width: 32 for the p / q / o / l configs, 64 for h")
args = ap.parse_args()
M, N, K = args.M, args.N, args.K
w, s, b = mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=128, bits=4)
x = mx.random.normal((M, K)).astype(mx.bfloat16)
xq, prm = w4a8.act_quant(x, mpad=256, ag=args.ag)
mx.eval(w, s, b, xq, prm)
xq0, prm0 = w4a8.act_quant(x, mpad=256)
fns = {"stock": lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=4),
       "128-wide default": lambda: w4a8.w4a8_matmul(xq0, prm0, M, w, s, b)}
for c in args.configs.split(","):
    for ab in COMBOS:
        kw = parse(c)
        kw["sbt"] = w4a8.sb_transpose(s, b) if kw["sbt"] else None
        fns[f"{c} {'+'.join(ab) or 'full'}"] = lambda kw=kw, ab=ab: w4a8.w4a8_matmul(xq, prm, M, w, s, b, ablate=ab, **kw)
for f in fns.values():
    mx.eval(f())
spin_up()
t = compare(fns, K=4, reps=4, iters=2, warmup=1)
for k, v in t.items():
    print(f"  {k:44s} {v * 1e3:7.3f} ms  {2 * M * N * K / v / 1e12:5.1f} TFLOP/s", flush=True)
