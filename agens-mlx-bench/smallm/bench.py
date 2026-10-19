"""time(M)/time(stock M=1) for M = 1..16: the NAX small-M kernel vs mx.quantized_matmul.

Weights rotate through enough copies (>= 1.5 GB) that every call streams from DRAM; K calls are
queued per eval, as in ../micro.py. All candidates are timed interleaved, best of --reps, and
only while no other GPU job runs (see common.wait_gpu_idle).

Rows:
  stock      mx.quantized_matmul(x[M])
  nax        qmm_nax kernel alone, on a prep_x() result made once (shared by several matmuls)
  nax+prep   prep_x (pad, group sums) + kernel, per call
  nax-i8     int8-activation variant, kernel alone
"""
import argparse
import mlx.core as mx
import qmm_nax as Q
from common import compare, spin_up, qweight, qbytes, SHAPES, BF

ap = argparse.ArgumentParser()
ap.add_argument("shapes", nargs="*", default=list(SHAPES))
ap.add_argument("--ms", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", default=None, help="append the markdown table to this file")
args = ap.parse_args()
MS = [int(v) for v in args.ms.split(",")]

lines = []
spin_up()
for name in args.shapes:
    n, k = SHAPES[name]
    copies = max(4, int(1.5e9 // qbytes(n, k)))
    ws = [qweight(n, k) for _ in range(copies)]
    w4s = [Q.to_int4(*t) for t in ws]
    mx.eval(w4s)
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) % copies
        return i[0]
    x16 = mx.random.normal((16, k)).astype(BF)
    mx.eval(x16)
    fns = {}
    for m in MS:
        x = x16[:m]
        px = Q.prep_x(x)
        p8 = Q.prep_x(x, "i8")
        mx.eval(x, *px[1:], *p8[1:])
        fns[("stock", m)] = lambda x=x: mx.quantized_matmul(x, *ws[nxt()], transpose=True, group_size=128, bits=4)
        fns[("nax", m)] = lambda px=px: Q.qmm_nax_prepped(px, *ws[nxt()])
        fns[("nax+prep", m)] = lambda x=x: Q.qmm_nax(x, *ws[nxt()])
        fns[("nax-i8", m)] = lambda p8=p8: Q.qmm_nax_prepped(p8, *w4s[nxt()], dtype="i8")
    t = compare(fns, K=copies, reps=args.reps)
    base = t[("stock", 1)]
    head = (f"{name} ({n}x{k}, {qbytes(n, k)/1e6:.1f} MB): stock M=1 = {base*1e3:.3f} ms "
            f"({qbytes(n, k)/base/1e9:.0f} GB/s); nax M=16 = {t[('nax', MS[-1])]*1e3:.3f} ms "
            f"({qbytes(n, k)/t[('nax', MS[-1])]/1e9:.0f} GB/s)")
    lines.append(head)
    lines.append("")
    lines.append("| M | " + " | ".join(str(m) for m in MS) + " |")
    lines.append("|---|" + "---|" * len(MS))
    for row in ("stock", "nax", "nax+prep", "nax-i8"):
        lines.append(f"| {row} | " + " | ".join(f"{t[(row, m)]/base:.2f}" for m in MS) + " |")
    lines.append("")
    print("\n".join(lines[-8:]), flush=True)
    del ws, w4s
if args.out:
    with open(args.out, "a") as f:
        f.write("\n".join(lines) + "\n")
