"""Per-linear error of the W4A8 path on real activations: one 2048-token prefill chunk through the bf16 model, the
inputs of every 4-bit linear in a few layers captured, then stock (bf16 activations) and W4A8 compared against
an fp32 CPU product on a row subset.

usage: python error_real.py --model ../models/volundr-int4-mlx [--layers 0,4,35,40,70,71] [--rows 256]
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from generate import load, set_wired_limit  # noqa: E402
import w4a8  # noqa: E402
import w4a8_hook as hook  # noqa: E402


def emulate(x, q, s, b, ag):
    """The kernel's arithmetic (centred codes, exact group sums for the bias term) with activation groups of ag."""
    M, K = x.shape
    N, G = s.shape
    xf = x.astype(mx.float32)
    xg = xf.reshape(M, K // ag, ag)
    amax = mx.abs(xg).max(-1, keepdims=True)
    xh = (mx.clip(mx.round(xg * mx.where(amax > 0, 127.0 / amax, 0.0)), -127, 127) * (amax / 127.0)).reshape(M, K)
    sf, bf = s.astype(mx.float32), b.astype(mx.float32)
    wc = ((q.reshape(N, G, 128) - 8.0) * sf[..., None]).reshape(N, K)
    xs = xf.reshape(M, G, 128).sum(-1)
    return xh @ wc.T + xs @ (bf + 8.0 * sf).T


def rel_rms(a, ref):
    e = a - ref
    return (mx.sqrt(mx.mean(e * e)) / mx.sqrt(mx.mean(ref * ref))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="0,4,35,40,70,71")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()
    model, tok = load(args.model)
    set_wired_limit()
    layers = [int(v) for v in args.layers.split(",")]
    names = {}
    for li in layers:
        for nm, m in model.model.layers[li].named_modules():
            if isinstance(m, nn.QuantizedLinear):
                names[id(m)] = (f"L{li}.{model.model.layers[li].mixer}.{nm}", m)
    cap = {}
    orig = nn.QuantizedLinear.__call__

    def rec(self, x):
        e = names.get(id(self))
        if e is not None and e[0] not in cap:
            cap[e[0]] = (x, e[1])
        return orig(self, x)

    nn.QuantizedLinear.__call__ = rec
    src = (HERE.parent / "ref/agens-engine/kernels/m1/m1_model.cu").read_text()
    ids = tok.encode(src)[: args.chunk]
    out = model(mx.array([ids]), cache=model.make_cache(), last_only=True)
    mx.eval(out, [v[0] for v in cap.values()])
    nn.QuantizedLinear.__call__ = orig

    sel = mx.arange(0, args.chunk, args.chunk // args.rows)
    print(f"{'linear':34s} {'N':>6s} {'K':>6s}  {'stock':>8s}  {'w4a8':>8s}  ratio  {'w4a8 vs stock':>13s}"
          f"  {'emu a128':>8s}  {'emu a64':>8s}  {'emu a32':>8s}")
    tot = {"stock": [], "w4a8": [], "emu a128": [], "emu a64": [], "emu a32": []}
    for nm, (x, m) in cap.items():
        x2 = x.reshape(-1, x.shape[-1])
        N, K = m.weight.shape[0], x2.shape[1]
        if N % hook._bn():
            continue
        stock = mx.quantized_matmul(x2, m.weight, m.scales, m.biases, transpose=True, group_size=128, bits=4)
        y = w4a8.qmm(x2, m.weight, m.scales, m.biases, **hook.CFG)
        mx.eval(stock, y)
        with mx.stream(mx.cpu):
            # exact dequantization in fp32 (mx.dequantize rounds s * q + b in bf16)
            q = mx.dequantize(m.weight, mx.ones_like(m.scales), mx.zeros_like(m.biases), group_size=128, bits=4)
            G = m.scales.shape[1]
            wd = (q.astype(mx.float32).reshape(N, G, 128) * m.scales.astype(mx.float32)[..., None]
                  + m.biases.astype(mx.float32)[..., None]).reshape(N, -1)
            ref = x2[sel].astype(mx.float32) @ wd.T
            s32, y32 = stock[sel].astype(mx.float32), y[sel].astype(mx.float32)
            es, ew, ews = rel_rms(s32, ref), rel_rms(y32, ref), rel_rms(y32, s32)
            em = {ag: rel_rms(emulate(x2[sel], q.astype(mx.float32), m.scales, m.biases, ag), ref) for ag in (128, 64, 32)}
        tot["stock"].append(es)
        tot["w4a8"].append(ew)
        for ag, v in em.items():
            tot[f"emu a{ag}"].append(v)
        print(f"{nm:34s} {N:6d} {K:6d}  {es:8.2e}  {ew:8.2e}  {ew / es:5.2f}  {ews:13.2e}"
              f"  {em[128]:8.2e}  {em[64]:8.2e}  {em[32]:8.2e}", flush=True)
    for k, v in tot.items():
        print(f"mean rel RMS error vs fp32, {k}: {sum(v) / len(v):.2e} over {len(v)} linears")


if __name__ == "__main__":
    main()
