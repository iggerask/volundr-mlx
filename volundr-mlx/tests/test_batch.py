"""batch.py on the tiny model (GPU): batched decode with per-row positions.

- A batch of one row is session.decode_step bit for bit (logits, features), in every BCSA regime (inside the first
  window, the window full, far blocks <= top-k, far top-k selection), and capture_row gives the snapshot a B = 1 cache
  gives (restored, every array identical; continuing from it identical).
- Rows of different lengths decoding together, joining and leaving (swap-remove) at different steps, give their own
  B = 1 runs' logits bit for bit. (The tiny model's bf16 matmuls are row-invariant at its sizes, so any difference
  would be the batching itself; the real model's matmuls change kernels with the row count, NOTES_BATCHED_SERVE.md.
  An fp32 model is no tighter check: MLX's fp32 matmul with several rows rounds like a reduced-precision product.)

run: python tests/test_batch.py [substring ...]
"""
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import batch as BT  # noqa: E402
import session as S  # noqa: E402
from tiny import tiny_model  # noqa: E402
from test_session import feed, ids_for, same  # noqa: E402

# tiny: W = 16, C = 4, top-k 3. Lengths: inside the first window, at the window, nv <= top-k, nv > top-k (selection)
LENGTHS = (5, 15, 16, 19, 27, 40, 71)


def arrays(c):
    """The arrays a cache holds (KVCache: the rows up to its offset)."""
    if hasattr(c, "cache"):
        return [a for a in c.cache if a is not None]
    if hasattr(c, "keys") and hasattr(c, "trim"):
        return [c.keys[..., : c.offset, :], c.values[..., : c.offset, :]]
    return list(c.state)


def prefilled(m, ids, chunk=7):
    c = m.make_cache()
    feed(m, c, ids, chunk=chunk)
    return c


def test_batch_of_one_is_decode_step():
    """B = 1 through batch.forward == session.decode_step, bit for bit; capture_row == capture."""
    m = tiny_model(0)
    capl = (1, 4)
    for n in LENGTHS:
        ids = ids_for(n + 30, n)
        c1 = prefilled(m, ids[:n])
        c2, _ = S.restore(S.capture(c1))
        bc = BT.BatchCache(m)
        bc.apply(join=[("r", c2)])
        for i, t in enumerate(ids[n:]):
            a, fa = S.decode_step(m, c1, int(t), capture=capl)
            b, fb = BT.forward(m, bc, mx.array([int(t)]), capture=capl)
            mx.eval(a, fa, b, fb, [c.state for c in c1], bc.state)
            same(a, b[0], f"n={n} step {i}: logits")
            same(fa, fb, f"n={n} step {i}: features")
            assert bc.pos == [S.cache_pos(c1)], (bc.pos, S.cache_pos(c1))
        # the row's snapshot is the B = 1 cache's: restored, every array equal; continuing, the same logits
        r1, _ = S.restore(S.capture(c1))
        r2, _ = S.restore(bc.capture_row("r"))
        for x, y in zip(r1, r2):
            assert type(x) is type(y)
            for p, q in zip(arrays(x), arrays(y)):
                same(p, q, f"n={n}: restored array")
        tail = ids_for(12, 99)
        same(feed(m, r1, tail), feed(m, r2, tail), f"n={n}: continued from capture_row")


def _schedule(m, prompts, joins, leaves, steps, rng):
    """Teacher-forced batched decode: prompt i joins at step joins[i] and leaves after leaves[i] steps. Returns
    {i: [logits per step]} and the tokens fed, plus the B = 1 reference logits of the same feeds."""
    feeds = {i: rng.integers(0, 280, size=steps).tolist() for i in range(len(prompts))}
    bc = BT.BatchCache(m)
    got = {i: [] for i in range(len(prompts))}
    fed = {i: 0 for i in range(len(prompts))}
    for t in range(steps):
        join = [(i, prefilled(m, prompts[i])) for i in range(len(prompts)) if joins[i] == t]
        leave = [i for i in list(bc.keys) if fed[i] >= leaves[i]]
        bc.apply(leave=leave, join=join)
        if not bc.B:
            continue
        toks = mx.array([feeds[i][fed[i]] for i in bc.keys], dtype=mx.int32)
        lg, _ = BT.forward(m, bc, toks)
        mx.eval(lg, bc.state)
        for r, i in enumerate(bc.keys):
            got[i].append(lg[r])
            fed[i] += 1
    ref = {}
    for i, p in enumerate(prompts):
        c = prefilled(m, p)
        ref[i] = [S.decode_step(m, c, t)[0] for t in feeds[i][: fed[i]]]
        mx.eval(ref[i])
    return got, ref


def test_rows_join_leave():
    """Rows of every regime at different lengths, joining and leaving at different steps (swap-remove of the
    leavers, slots reused), two schedules: each row's logits equal its own B = 1 run's, bit for bit."""
    for seed, joins, leaves in ((1, [0, 0, 2, 3, 3, 6, 9], [20, 7, 25, 3, 30, 12, 18]),
                                (2, [0, 1, 1, 2, 4, 4, 5], [25, 22, 9, 30, 15, 25, 20])):
        m = tiny_model(seed)
        prompts = [ids_for(n, 10 * seed + n) for n in LENGTHS]
        got, ref = _schedule(m, prompts, joins, leaves, 40, np.random.default_rng(seed))
        for i in got:
            assert len(got[i]) == leaves[i], (i, len(got[i]))
            for s, (a, b) in enumerate(zip(ref[i], got[i])):
                same(a, b, f"schedule {seed}: row {i} (prompt {LENGTHS[i]}) step {s}")
    print(f"    {sum(len(v) for v in got.values())} row-steps in the second schedule")


def test_buffer_growth():
    """The ring widths (real model: 256 ... 2048, then W + F) and the pooled-block / dense K/V capacity steps scaled
    down to the tiny model's W = 16, so the batch buffers grow while rows decode: still B = 1's logits bit for bit."""
    saved = BT.RING_SIZES, BT.BLOCK_STEP, BT.KV_STEP
    BT.RING_SIZES, BT.BLOCK_STEP, BT.KV_STEP = (4, 8, 12), 2, 8
    try:
        m = tiny_model(3)
        lens = (2, 3, 6, 10, 14)
        prompts = [ids_for(n, 40 + n) for n in lens]
        got, ref = _schedule(m, prompts, [0, 0, 3, 5, 6], [30, 12, 25, 20, 28], 36, np.random.default_rng(3))
        for i in got:
            for s, (a, b) in enumerate(zip(ref[i], got[i])):
                same(a, b, f"row {i} (prompt {lens[i]}) step {s}")
    finally:
        BT.RING_SIZES, BT.BLOCK_STEP, BT.KV_STEP = saved


def test_prebuilt_step_and_rollback():
    """Pipelining: the next step built ahead (prebuild) and then used gives the same logits as building it then; a
    prebuilt step rolled back (a row leaves, a snapshot is taken) leaves the batch exactly as before it was built."""
    m = tiny_model(5)
    prompts = [ids_for(n, 60 + n) for n in (6, 19, 40)]

    def greedy_run(pipe, steps=24, leave_at=10):
        bc = BT.BatchCache(m)
        bc.apply(join=[(i, prefilled(m, p)) for i, p in enumerate(prompts)])
        toks = mx.array([1, 2, 3], dtype=mx.int32)
        got, snaps = {0: [], 1: [], 2: []}, None
        for t in range(steps):
            key = tuple(bc.keys)
            pre = bc.take_prebuilt(key) if pipe else None
            if pre is None:
                lg, _ = BT.forward(m, bc, toks)
                pre = (lg, mx.argmax(lg, axis=-1).astype(mx.int32), bc.state)
            lg, g, st = pre
            if pipe:
                mx.async_eval(lg, g, st)
                bc.prebuild(key, lambda g=g: (lambda lg2: (lg2, mx.argmax(lg2, axis=-1).astype(mx.int32),
                                                           bc.state))(BT.forward(m, bc, g)[0]))
            mx.eval(lg, g, st)
            for r, k in enumerate(bc.keys):
                got[k].append(lg[r])
            toks = g
            if t == leave_at:  # a snapshot, then row 1 leaves (rolls a prebuilt step back first)
                snaps = bc.capture_row(2)
                b = bc.keys.index(1)
                bc.apply(leave=[1])
                toks = mx.array([int(x) for i, x in enumerate(g.tolist()) if i != b], dtype=mx.int32)
        return got, snaps

    a, sa = greedy_run(False)
    b, sb = greedy_run(True)
    for k in a:
        assert len(a[k]) == len(b[k]), (k, len(a[k]), len(b[k]))
        for s, (x, y) in enumerate(zip(a[k], b[k])):
            same(x, y, f"row {k} step {s}")
    for x, y in zip(S.restore(sa)[0], S.restore(sb)[0]):
        for p, q in zip(arrays(x), arrays(y)):
            same(p, q, "snapshot taken with a step prebuilt")


def test_slots():
    """S doubles as rows join, swap-remove keeps every row's state, and an emptied batch frees its buffers."""
    m = tiny_model(4)
    bc = BT.BatchCache(m)
    caches = [prefilled(m, ids_for(n, n)) for n in (9, 33, 50)]
    bc.apply(join=[("a", caches[0])])
    assert bc.slots_cap == 1
    bc.apply(join=[("b", caches[1]), ("c", caches[2])])
    assert bc.slots_cap == 4 and bc.keys == ["a", "b", "c"] and bc.pos == [9, 33, 50]
    snap_c = bc.capture_row("c")
    bc.apply(leave=["a"])  # c moves into slot 0
    assert bc.keys == ["c", "b"] and bc.pos == [50, 33]
    r1, _ = S.restore(snap_c)
    r2, _ = S.restore(bc.capture_row("c"))
    for x, y in zip(r1, r2):
        for p, q in zip(arrays(x), arrays(y)):
            same(p, q, "moved row")
    bc.apply(leave=["b", "c"])
    assert bc.B == 0 and bc.slots_cap == 0 and bc.nbytes == 0


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    sel = sys.argv[1:]
    fails = 0
    for name, fn in tests:
        if sel and not any(s in name for s in sel):
            continue
        t0 = time.perf_counter()
        try:
            fn()
            print(f"ok   {name} ({time.perf_counter() - t0:.1f} s)")
        except Exception:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - fails}/{len(tests)} passed" if not sel else f"{fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
