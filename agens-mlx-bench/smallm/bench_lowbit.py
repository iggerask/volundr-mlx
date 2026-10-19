"""2/3-bit: NAX low-bit kernel at M=16 vs stock mx.quantized_matmul at M=1 and M=16 (same bits).

python bench_lowbit.py <bits> [shapes...] [--cfgs bf16:32:8,i8:64:8,...]
"""
import argparse
import mlx.core as mx
import qmm_nax_lowbit as L
from common import compare, spin_up, qweight, qbytes, SHAPES, BF

ap = argparse.ArgumentParser()
ap.add_argument("bits", type=int)
ap.add_argument("shapes", nargs="*", default=list(SHAPES))
ap.add_argument("--cfgs", default="bf16:32:8,bf16:64:8,bf16:32:4,i8:32:8,i8:64:8,i8:64:4")
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()
spin_up()
for name in a.shapes:
    n, k = SHAPES[name]
    nb = qbytes(n, k, a.bits)
    copies = max(4, int(1.5e9 // nb))
    ws = [qweight(n, k, a.bits) for _ in range(copies)]
    wrs = [(L.repack(w, a.bits, k), s, L.centred_bias(s, b, a.bits)) for w, s, b in ws]
    mx.eval(wrs)
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) % copies
        return i[0]
    x16 = mx.random.normal((16, k)).astype(BF)
    x1 = x16[:1]
    mx.eval(x16, x1)
    fns = {"stock M=1": lambda: mx.quantized_matmul(x1, *ws[nxt()], transpose=True, group_size=128, bits=a.bits),
           "stock M=16": lambda: mx.quantized_matmul(x16, *ws[nxt()], transpose=True, group_size=128, bits=a.bits)}
    for cfg in a.cfgs.split(","):
        dt, nt, ks = cfg.split(":")
        px = L.prep_x(x16, dt)
        mx.eval(*px[1:])
        fns[f"nax {dt} nt={nt} ks={ks} M=16"] = (lambda px=px, dt=dt, nt=int(nt), ks=int(ks):
                                                 L.qmm_lowbit_prepped(px, *wrs[nxt()], a.bits, dt, nt=nt, ks=ks))
    t = compare(fns, K=copies, reps=a.reps)
    t1 = t["stock M=1"]
    print(f"{a.bits}-bit {name} ({n}x{k}, {nb/1e6:.1f} MB), best of {a.reps} interleaved:")
    for nm, v in t.items():
        print(f"  {nm:30s} {v*1e3:7.3f} ms {nb/v/1e9:5.0f} GB/s  {v/t1:5.2f}x stock M=1", flush=True)
    del ws, wrs
