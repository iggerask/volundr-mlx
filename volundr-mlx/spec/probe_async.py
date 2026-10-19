"""Which part of the drafter graph makes mx.async_eval block? Times async_eval (call only) and the following eval."""
import sys
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash as DF  # noqa: E402
import specdec as S  # noqa: E402


def probe(name, build, reps=4):
    res = []
    for _ in range(reps + 1):
        x = build()
        t0 = time.perf_counter()
        mx.async_eval(x)
        t1 = time.perf_counter()
        mx.eval(x)
        t2 = time.perf_counter()
        res.append((1e3 * (t1 - t0), 1e3 * (t2 - t1)))
    res = sorted(res[1:])
    a, w = res[len(res) // 2]
    print(f"{name:28s} async_eval call {a:6.2f} ms   then eval {w:6.2f} ms", flush=True)


def main():
    model, drafter, tok = S.load(0, 0, True)
    ids = tok.encode("Write a short story about a lighthouse keeper." * 4)
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = S.spec_prefill(model, drafter, cache, dcache, ids)
    bonus = mx.argmax(last).astype(mx.int32)
    mx.eval(bonus)
    P = len(ids)
    w = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    probe("matmul chain", lambda: (mx.ones((8, 4096), mx.bfloat16) @ w @ w @ w @ w @ w @ w))
    probe("drafter hidden", lambda: drafter.hidden(dcache, bonus, P))
    hs = drafter.hidden(dcache, bonus, P)[1:]
    mx.eval(hs)
    probe("head", lambda: drafter.head(hs))
    lg = drafter.head(hs)[:, : drafter.vocab_valid]
    mx.eval(lg)
    probe("argpartition", lambda: mx.argpartition(-lg, DF.TOPK - 1, axis=-1)[:, : DF.TOPK])
    probe("propose (all)", lambda: drafter.propose(dcache, bonus, P))
    probe("embed only", lambda: drafter.embed(mx.array([1, 2, 3], dtype=mx.int32)))


if __name__ == "__main__":
    main()
