"""Where the nax verify step's time goes beyond the matmul kernels (volundr_synth.py, short ctx).

Variants, each T=8 and T=16 unless noted, timed interleaved (best of REPS):
  stock / nax8          the real steps
  nax8 small-N stock    linears with N <= 1024 (k, v, iq, ik, fdown) back on mx.quantized_matmul
  *-nomm                every quantized matmul replaced by one trivial broadcast op on its input,
                        so the step keeps the same op count and dependencies but no weight traffic;
                        the nomm difference between nax8 and stock is the prep kernels' cost net of
                        the ops they fused away
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import volundr_synth as V
from common import wait_gpu_idle

wait_gpu_idle()
L, head = V.build()
real_qmm = V.qmm


def nomm(x, p):
    xa = V.P.unprep(x) if isinstance(x, tuple) else x
    return mx.zeros((xa.shape[0], p[0].shape[0]), dtype=V.BF) + xa[:, :1]


def small_n_stock(x, p):
    if isinstance(x, tuple) and p[0].shape[0] <= 1024:
        x = V.P.unprep(x)
    return real_qmm(x, p)


VARIANTS = [("1", "tuned", "", real_qmm, "stock"), ("1", "tuned", "", nomm, "stock-nomm"),
            ("1", "linears", "", nomm, "stock-nomm")]
for T in ("8", "16"):
    for mode in ("tuned", "linears"):
        VARIANTS += [(T, mode, "", real_qmm, "stock"), (T, mode, "nax8", real_qmm, "nax8"),
                     (T, mode, "nax8", small_n_stock, "nax8 small-N stock"),
                     (T, mode, "", nomm, "stock-nomm"), (T, mode, "nax8", nomm, "nax8-nomm")]
best = {}
for _ in range(int(os.environ.get("REPS", "3"))):
    for T, mode, nax, f, name in VARIANTS:
        V.qmm = f
        print(f"[{name}] ", end="")
        dt = V.run(L, head, int(T), "short", mode, nax)
        best[(T, mode, name)] = min(best.get((T, mode, name), 1e9), dt)
V.qmm = real_qmm
print("best of reps (ms/step):")
for (T, mode, name), dt in best.items():
    print(f"  T={T:>2s} {mode:7s} {name:20s} {dt * 1e3:6.1f}")
