"""End-to-end check of volundr_synth's nax path: one verify step through the first 4 layers
(KDA, KDA, KDA + Engram, BCSA), stock vs nax vs nax8, from identical cache state and RNG.

Every linear's output is recorded and compared with the stock run's, so each fused prep is checked
in place, with the real producer feeding it. Random weights: the nax paths should differ from stock
by bf16 rounding only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import volundr_synth as V
from common import wait_gpu_idle

wait_gpu_idle()
mx.random.seed(0)
L, head = V.build(4)
_qmm, rec = V.qmm, []


def recording_qmm(x, p):
    y = _qmm(x, p)
    rec.append((isinstance(x, tuple), p[3], y))
    return y


V.qmm = recording_qmm


def one(T, nax, mode):
    mx.random.seed(1)
    c = V.Cache(T, False)
    c.pos = 0  # H starts from the embeddings; at pos > 0 the synthetic H starts as zeros
    rec.clear()
    mx.random.seed(2)
    y = V.step(L, head, c, mx.arange(T).astype(mx.uint32) * 37, mode, nax)
    mx.eval(y, [r[2] for r in rec])
    return y, list(rec)


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.sqrt(((a - b) ** 2).mean()) / mx.maximum(mx.sqrt((b ** 2).mean()), 1e-30)).item()


for mode in ("tuned", "linears"):
    for T in (2, 8, 16):
        yr, ref = one(T, "", mode)
        for nax in ("nax", "nax8"):
            y, got = one(T, nax, mode)
            assert len(got) == len(ref)
            n4 = sum(1 for g in got if g[0] and g[1] == 4)
            n8 = sum(1 for g in got if g[0] and g[1] == 8)
            d = [rel(g[2], r[2]) for g, r in zip(got, ref)]
            print(f"{mode:7s} T={T:2d} {nax:4s}: {n4}/{len(got) - 1} 4-bit + {n8} 8-bit linears on NAX; "
                  f"output rel rms diff vs stock: max {max(d):.1e}, mean {sum(d) / len(d):.1e}; "
                  f"argmax agreement {(y == yr).astype(mx.float32).mean().item():.2f}", flush=True)
