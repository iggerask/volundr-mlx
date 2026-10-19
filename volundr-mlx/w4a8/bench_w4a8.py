"""W4A8 GEMM vs mx.quantized_matmul at prefill shapes (M = 2048 by default), interleaved best-of-reps.

usage: python bench_w4a8.py [--shapes gate,down] [--configs 64x32x2x2,32x64x2x2] [--M 2048] [--sweep]
Weights rotate through enough copies (>= 256 MB) that every call streams them from DRAM, as in the model.
"""
import argparse

import mlx.core as mx

import w4a8
from w4common import SHAPES, compare, spin_up

DEFAULT = ["32x32x2x2RV", "32x32x2x2", "32x32x4x1"]
SWEEP = ["32x32x2x2RV", "32x32x2x2", "32x32x2x2s", "32x32x4x1", "32x32x4x2", "32x32x8x1", "32x32x2x1", "32x32x1x1",
         "32x32x2x4", "64x32x2x2", "64x32x4x1", "32x64x2x2", "32x32x2x22"]
OLD_SWEEP = ["32x32x2x2PV", "32x32x2x2V", "32x32x2x2", "32x32x2x2s", "32x32x4x2s", "32x32x2x4s", "32x32x4x4s",
         "32x32x4x1s", "32x32x8x1s", "32x32x1x4s", "64x32x2x2s", "64x32x4x1s", "32x64x4x1s", "32x64x2x2s"]


def parse(c):
    """'SMxSNxWMxWN' + flags: R = per-lane B gathers (act_quant perm 0) instead of row chunks, t = B staged in
    threadgroup memory, P = no prefetch, V = scalar B gathers (with R), s = transposed (s, b) pairs for the epilogue,
    n = N-fastest grid, p / q / o = 32-wide activation groups (p: fp32 partial, q: straight into acc, o: q with
    two result tiles overlapping scaling and the next op, l: q with the fewest live registers; h = 64-wide groups on
    the lean path; implies R);
    a trailing digit = prefetch depth (1)."""
    parts = c.split("x")
    sm, sn, wm = (int(v) for v in parts[:3])
    wn = int("".join(ch for ch in parts[3] if ch.isdigit())[:1])
    f = parts[3][1:]
    depth = int(f[-1]) if f and f[-1].isdigit() else 1
    if "L" in f:  # plain-register kernel: R = gathers (perm 0), else row chunks
        return dict(sm=sm, sn=sn, wm=wm, wn=wn, mfast="n" not in f, brow="R" not in f, plain=True)
    ag = 1 if "p" in f else 2 if "q" in f else 3 if "o" in f else 4 if "l" in f else 5 if "h" in f else 0
    return dict(sm=sm, sn=sn, wm=wm, wn=wn, mfast="n" not in f, btg="t" in f,
                pf=(depth if "P" not in f and "t" not in f else 0), bvec="V" not in f and "R" in f, sbt="s" in f,
                brow="R" not in f and "t" not in f and not ag, ag=ag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="gate_up,gate,down,q_proj,kda_qkv,z,out")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    cfgs = args.configs.split(",") if args.configs else (SWEEP if args.sweep else DEFAULT)
    M = args.M
    want = args.shapes.split(",")
    spin_up()
    for name, N, K in SHAPES:
        if name not in want:
            continue
        ncopy = max(1, -(-256 * 2**20 // (N * K // 2)))
        ws = []
        for _ in range(ncopy):
            wf = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
            ws.append(mx.quantize(wf, group_size=128, bits=4))
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(ws, x)
        it = {"i": 0}

        def nxt():
            it["i"] = (it["i"] + 1) % ncopy
            return ws[it["i"]]

        fns = {"stock": lambda: mx.quantized_matmul(x, *nxt(), transpose=True, group_size=128, bits=4)}
        xq_cache = {}
        for c in cfgs:
            kw = parse(c)
            bm, bn = kw["sm"] * kw["wm"], kw["sn"] * kw["wn"]
            if N % bn:
                continue
            plain = kw.pop("plain", False)
            perm = int(kw["brow"] and not kw.get("btg", False))
            agw = 64 if kw.get("ag") == 5 else 32 if kw.get("ag") else 128
            if (bm, perm, agw) not in xq_cache:
                xq_cache[(bm, perm, agw)] = w4a8.act_quant(x, mpad=bm, perm=perm, ag=agw)
                mx.eval(xq_cache[(bm, perm, agw)])
            xq, prm = xq_cache[(bm, perm, agw)]
            if plain:
                fn = w4a8.w4a8_matmul_plain
                try:
                    mx.eval(fn(xq, prm, M, *ws[0], **kw))
                except Exception as e:  # noqa: BLE001
                    print(f"  {c}: failed: {str(e).splitlines()[0][:120]}")
                    continue
                fns[c] = lambda xq=xq, prm=prm, kw=kw: w4a8.w4a8_matmul_plain(xq, prm, M, *nxt(), **kw)
                continue
            if kw["sbt"]:
                if "sbt" not in xq_cache:
                    xq_cache["sbt"] = [w4a8.sb_transpose(w[1], w[2]) for w in ws]
                    mx.eval(xq_cache["sbt"])
                kw["sbt"] = xq_cache["sbt"]
            else:
                kw["sbt"] = None
            try:
                mx.eval(w4a8.w4a8_matmul(xq, prm, M, *ws[0], **{**kw, "sbt": kw["sbt"][0] if kw["sbt"] else None}))
            except Exception as e:  # noqa: BLE001
                print(f"  {c}: failed: {str(e).splitlines()[0][:120]}")
                continue
            def f(xq=xq, prm=prm, kw=kw):
                w = nxt()
                return w4a8.w4a8_matmul(xq, prm, M, *w, **{**kw, "sbt": kw["sbt"][it["i"]] if kw["sbt"] else None})
            fns[c] = f
        fns["act_quant"] = lambda: w4a8.act_quant(x, mpad=128)[0]
        fns["act_quant32"] = lambda: w4a8.act_quant(x, mpad=128, ag=32)[0]
        fns["sb_transpose"] = lambda: w4a8.sb_transpose(*nxt()[1:])
        t = compare(fns, K=4, reps=args.reps, iters=3, warmup=2)
        flop = 2 * M * N * K
        print(f"{name} N={N} K={K} M={M}:")
        for k, v in t.items():
            tf = "" if k in ("act_quant", "act_quant32", "sb_transpose") else f"{flop / v / 1e12:5.1f} TFLOP/s  {t['stock'] / v:4.2f}x"
            print(f"  {k:12s} {v * 1e3:7.3f} ms  {tf}", flush=True)


if __name__ == "__main__":
    main()
