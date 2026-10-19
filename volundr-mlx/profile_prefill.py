"""Per-component timing of one prefill chunk on the real model (sync after every piece; for attribution, not speed).

usage: python profile_prefill.py --model models/volundr-int4-mlx [--offset 0] [--chunk 2048] [--chunks 2]
--offset P prefills P tokens first (untimed) so the chunks run at positions [P, P + chunks * chunk).
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volundr as V  # noqa: E402
from generate import load, prefill, set_wired_limit  # noqa: E402

T = defaultdict(float)


def tick(key, *arrs):
    mx.eval(*arrs)
    mx.synchronize()
    now = time.perf_counter()
    T[key] += now - tick.t
    tick.t = now


def forward(model, ids, cache):
    """The model's own code paths, synced per mHC mix / layer / combine (attribution by layer kind)."""
    vm = model.model
    a = vm.args
    B, L = ids.shape
    x = vm.embed_tokens(ids)
    st = cache[-1]
    prev = st.prev if st.prev is not None else mx.full((B, st.keep), -1, dtype=mx.int32)
    full = mx.concatenate([prev, ids], axis=1)
    st.prev = full[:, -st.keep:]
    rows = [V.engram_rows(full, n, L, a.engram_rows) for n in a.engram_orders]
    H = mx.broadcast_to(x[:, :, None, :], (B, L, a.mhc_streams, a.hidden_size))
    hmean = x
    tick("embed+hash", x, *rows)
    for layer, mhc, c in zip(vm.layers, vm.mhc, cache):
        x_in, post, res = mhc.mix(H, hmean)
        tick("mhc.mix", x_in, post, res)
        y = layer(x_in, c, rows)
        tick(f"layer.{layer.mixer}" + ("+engram" if "engram" in layer else ""), y)
        H, hmean = mhc.combine(H, post, res, y, x_in)
        tick("mhc.combine", H, hmean)
    out = model.lm_head(vm.norm(H[:, -1:, 0]))
    tick("head", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=2, help="consecutive chunks to profile; the first one at a new "
                    "offset also pays one-time kernel compilation, so read the last")
    args = ap.parse_args()
    model, tok = load(args.model)
    set_wired_limit()
    src = (Path(__file__).parent / "ref/agens-engine/kernels/m1/m1_model.cu").read_text() * 8
    ids = np.array(tok.encode(src)[: args.offset + (args.chunks + 1) * args.chunk], dtype=np.int32)
    cache = model.make_cache()
    if args.offset:
        prefill(model, cache, ids[: args.offset].tolist(), 2048)
    for i in range(args.chunks):
        p0 = args.offset + i * args.chunk
        mx.clear_cache()
        T.clear()
        chunk = mx.array(ids[None, p0 : p0 + args.chunk])
        t0 = time.perf_counter()
        tick.t = t0
        forward(model, chunk, cache)
        total = time.perf_counter() - t0
        print(f"chunk {args.chunk} at offset {p0}: {total:.2f} s ({args.chunk / total:.0f} tok/s with syncs)")
        for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
            print(f"  {k:28s} {v * 1000:8.1f} ms  {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
