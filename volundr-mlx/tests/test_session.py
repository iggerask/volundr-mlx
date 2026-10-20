"""session.py on the tiny model (GPU, bf16): snapshot / restore is exact for every cache type, a snapshot is never
changed by the caches made from it or by the cache it was taken from, conv-state views do not pin their projection
buffers, and the prefix cache's lookup, dedupe, LRU eviction and snapshot points. The full-model gates (spec decode,
drafter context, resumed vs fresh prefill) are in session_gate.py.

run: python tests/test_session.py [substring ...]
"""
import gc
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import session as S  # noqa: E402
import volundr  # noqa: E402
from tiny import tiny_model  # noqa: E402

RNG = np.random.default_rng(1234)


def ids_for(n, seed=0):
    return np.random.default_rng(seed).integers(0, 280, size=n).astype(np.int32)


def same(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    if not mx.array_equal(a, b, equal_nan=True).item():
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        raise AssertionError(f"{what}: differs, max abs {d:.3e}")


def feed(model, cache, ids, chunk=None):
    """Logits of every fed position (prefill chunks of `chunk`, or one token at a time)."""
    x = mx.array(np.asarray(ids)[None], dtype=mx.int32)
    out = []
    step = chunk or 1
    for s in range(0, x.shape[1], step):
        lg = model(x[:, s : s + step], cache=cache)
        mx.eval(lg, [c.state for c in cache])
        out.append(lg[0])
    return mx.concatenate(out, axis=0)


def snapshot_np(st):
    return [np.array(a.astype(mx.float32)) if a.dtype != mx.int32 else np.array(a) for a in st.arrays()]


def kinds(cache):
    return sorted({type(c).__name__ for c in cache})


def test_restore_continues_bit_identically():
    """Prefill (window, pooled far blocks and top-k all active: W = 16, C = 4, top-k 3), snapshot, then the original
    cache and two restored ones continue with the same chunks and decode steps: every logit row is identical, and the
    snapshot's arrays are unchanged afterwards."""
    m = tiny_model(0)
    ids = ids_for(97, 1)
    for split in (5, 9, 40, 41, 64):  # inside the first window (compacted rows), a block boundary, mid-block, ...
        cache = m.make_cache()
        feed(m, cache, ids[:split], chunk=13)
        st = S.capture(cache, logits=None)
        before = snapshot_np(st)
        rest = ids[split:]
        a = mx.concatenate([feed(m, cache, rest[:20], chunk=7), feed(m, cache, rest[20:])])  # prefill chunks, decode
        for k in range(2):
            c2, _ = S.restore(st)
            assert kinds(c2) == kinds(cache)
            b = mx.concatenate([feed(m, c2, rest[:20], chunk=7), feed(m, c2, rest[20:])])
            same(a, b, f"split {split}, restore {k}")
            for p, q in zip(S.capture(cache).arrays(), S.capture(c2).arrays()):  # the final cache state too
                same(p, q, f"split {split}: final cache array")
        after = snapshot_np(st)
        for x, y in zip(before, after):
            assert np.array_equal(x, y, equal_nan=True), "snapshot changed"
    print(f"    cache kinds: {kinds(m.make_cache())}; snapshot at 64: {st.nbytes} bytes")


def test_restore_from_restored():
    """A snapshot of a restored cache (a fork of a fork) continues identically too."""
    m = tiny_model(3)
    ids = ids_for(80, 2)
    c = m.make_cache()
    feed(m, c, ids[:30], chunk=11)
    s1 = S.capture(c)
    c1, _ = S.restore(s1)
    feed(m, c1, ids[30:50], chunk=6)
    s2 = S.capture(c1)
    a = feed(m, c1, ids[50:])
    c2, _ = S.restore(s2)
    b = feed(m, c2, ids[50:])
    same(a, b, "fork of a fork")
    c3, _ = S.restore(s1)  # the first snapshot is still intact
    feed(m, c3, ids[30:50], chunk=6)
    same(feed(m, c3, ids[50:]), a, "restore of the first snapshot after both forks ran")


def test_inplace_writes_need_a_new_array_object():
    """Why the snapshot holds copy.copy(a): `a[idx] = x` replaces the contents of the array object a, so the same
    object would see later writes; a copy.copy on the same buffer does not (MLX copies the shared buffer)."""
    import copy

    a = mx.zeros((4, 1024)); mx.eval(a)
    alias, ref = a, copy.copy(a)
    a[1:2] = 3.0
    mx.eval(a)
    assert alias[1, 0].item() == 3.0 and ref[1, 0].item() == 0.0


def test_conv_state_view_is_copied():
    """The KDA conv state after a prefill chunk is a slice of the chunk's projection: the snapshot copies it, so it
    does not keep the projection alive."""
    gc.collect(); mx.clear_cache()
    big = mx.random.normal((1, 8192, 4096)).astype(mx.bfloat16)  # 64 MB
    mx.eval(big)
    c = volundr.EngramState(2)
    c.prev = mx.zeros((1, 2), mx.int32)
    from mlx_lm.models.cache import KVCache, ArraysCache

    kd = ArraysCache(size=2)
    kd[0], kd[1] = big[:, -3:], mx.zeros((1, 4, 8, 8))
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, 5, 8)), mx.zeros((1, 2, 5, 8)))
    st = S.capture([kd, kv, c])
    assert np.array_equal(np.array(st.layers[0][1].astype(mx.float32)), np.array(big[:, -3:].astype(mx.float32)))
    del big, kd
    gc.collect(); mx.clear_cache()
    held = mx.get_active_memory()
    assert held < 32e6, f"{held / 1e6:.1f} MB still active: the view pins its parent"
    assert st.nbytes < 1e6, st.nbytes


def test_decode_step_matches_model():
    """session.decode_step (the plain decode with feature capture) gives Model.__call__'s logits bit for bit, and the
    features verify.forward captures."""
    sys.path.insert(0, str(HERE.parent / "spec"))
    import verify as VF

    m = tiny_model(5)
    ids = ids_for(60, 3)
    c1, c2, c3 = m.make_cache(), m.make_cache(), m.make_cache()
    for c in (c1, c2, c3):
        feed(m, c, ids[:40], chunk=16)
    for t in ids[40:]:
        a = m(mx.array([[int(t)]]), cache=c1)[0, -1]
        b, f = S.decode_step(m, c2, int(t), capture=(1, 3))
        _, cap, _ = VF.forward(m, mx.array([[int(t)]]), c3, capture=(1, 3))
        mx.eval(a, b, f, cap, [c.state for c in c1 + c2 + c3])
        same(a, b, "decode_step logits")
        same(f, cap, "decode_step features")


def _fake_state(pos, nbytes=1000, logits=False):
    st = S.State(pos, [], None, mx.zeros((4,)) if logits else None)
    st.nbytes = nbytes
    return st


def test_lookup_prefix_session_and_logits():
    sc = S.SessionCache(budget_gb=1.0)
    base = list(range(100, 160))
    sc.add("a", "prompt", base[:20], _fake_state(20))
    sc.add("b", "prompt", base[:40], _fake_state(40))
    sc.add("a", "reply", base[:30], _fake_state(30))
    sc.add("b", "reply", base[:30] + [7] * 15, _fake_state(45))  # diverges at 30
    assert sc.find(base, "a").pos == 40  # the longest prefix wins even across sessions
    assert sc.find(base[:35], "a").pos == 30
    assert sc.find(base[:30] + [7] * 20, "a").pos == 45
    assert sc.find([1, 2, 3], "a") is None
    assert sc.find(base[:40], "b").pos == 30  # an exact-length snapshot needs logits (nothing left to prefill)
    sc.add("c", "prompt", base[:40] + [1], _fake_state(41, logits=True))
    assert sc.find(base[:40] + [1], None).pos == 41
    # ties go to the request's own session
    sc.add("x", "prompt", [5] * 10, _fake_state(10))
    s_y = sc.add("y", "prompt", [5] * 10, _fake_state(10))
    assert s_y.session == "x"  # dedupe: same tokens -> the existing snapshot
    assert len([s for s in sc.snaps if s.pos == 10]) == 1


def test_lru_eviction_and_budget():
    sc = S.SessionCache(budget_gb=10_000 / 1e9)  # 10 kB
    for i in range(8):
        sc.add("s", "prompt", [i] * 10, _fake_state(10, nbytes=2000))
    assert sc.nbytes <= sc.budget and len(sc.snaps) == 4, (sc.nbytes, len(sc.snaps))
    assert sorted(s.tokens[0] for s in sc.snaps) == [4, 5, 6, 7]
    sc.find([5] * 12, "s")  # touch 5
    sc.add("s", "prompt", [9] * 10, _fake_state(10, nbytes=2000))
    assert sorted(s.tokens[0] for s in sc.snaps) == [5, 6, 7, 9]
    assert sc.add("s", "prompt", [3] * 10, _fake_state(10, nbytes=20_000)) is None  # larger than the budget
    assert S.SessionCache(budget_gb=0).add("s", "prompt", [1], _fake_state(1)) is None


def test_points():
    sc = S.SessionCache(budget_gb=1, interval=4096, min_gap=256)
    assert sc.points(0, 20000) == [4096, 8192, 12288, 16384]
    assert sc.points(4000, 8300) == []  # 4096 - 4000 and 8300 - 8192 are both < 256
    assert sc.points(4000, 9000) == [8192]
    assert sc.points(8000, 9000) == []  # 8192 - 8000 < 256
    assert sc.points(0, 4200) == []


def main():
    sel = sys.argv[1:]
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and (not sel or any(s in n for s in sel))]
    failed = 0
    for n, f in tests:
        t = time.time()
        try:
            f()
            print(f"PASS {n} ({time.time() - t:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
