"""Session / prefix cache for Volundr MLX: snapshots of the whole decoding state, so a follow-up turn prefills only
its new tokens.

A snapshot (`State`) is everything needed to continue from a position P, i.e. after the tokens ids[:P] were fed:
- per layer: the KDA conv and recurrent state (ArraysCache); the BCSA ring K/V with its far-block scratch rows, the
  pooled far-block K/V, the pooled indexer keys, the open block's indexer keys and the offset (BCSACache); the dense
  layer's K/V and offset (KVCache); the Engram previous ids (EngramState);
- the DFlash2 drafter's context K/V ring (dflash.DraftCache), valid for the last WINDOW positions;
- the logits of position P - 1, which pick token P when a prompt ends exactly at P.
Prompt lookup (spec/lookup.py) needs nothing: as in upstream `agens serve`, each request starts a Lookup from its own
prompt, so its state is a function of the tokens. `lookup_state` / `restore_lookup` cover a mid-reply restore.

Holding references vs copying (measured with MLX 0.32.4, tests/test_session.py):
- Every cache update either rebinds an attribute to a new array (KDA state, Engram ids, KVCache growth) or writes
  into an array in place (`a[idx] = x`: the BCSA ring and pooled blocks, the dense K/V, the drafter ring). MLX's
  Python `__setitem__` replaces the array object's contents, so a snapshot that held the same array object would see
  later writes. A snapshot therefore holds `copy.copy(a)`: a new array object on the same buffer. MLX then does not
  donate the shared buffer to the next slice update and copies it once (copy on write); later writes are in place
  again. Restore hands the new cache copy.copy objects too, so a snapshot is never changed by the caches made from it
  and can be restored any number of times (fork, not mutate).
- A reference keeps the whole underlying buffer alive. The KDA conv state after a prefill chunk is a 3-row slice of
  that chunk's qkv projection (42 MB per layer at 2048 tokens, 2.2 GB over the 54 layers), so a reference would pin
  the projections. Arrays under 4 MiB (conv states, KDA recurrent states, Engram ids, open-block keys, logits rows)
  are copied; the large ones (rings, pooled blocks, dense K/V, drafter ring) are allocated buffers, never views, and
  are referenced.
- Buffers that are at most half in use (the BCSA rings below 4096 positions, pooled blocks, the drafter ring) are
  kept as their used leading rows (`Rows`) and re-expanded with zeros on restore; the rest of each buffer is still
  zeros, so this is exact.
So a snapshot costs its bytes (0.18 GB for a short prompt, ~1.1 GB at 20K context) plus, the first time the live
cache writes each shared buffer, one copy of it.

The prefix cache (`SessionCache`) keeps snapshots at the end of each request's prompt prefill, at the end of each reply
(the cache then holds prompt + reply minus the last token) and every `interval` tokens of a long prefill (at absolute
multiples, so requests that share a prefix share those points). A KDA state cannot be trimmed to an arbitrary prefix,
so a request resumes from the longest snapshot whose tokens are a prefix of its prompt: the request's own session
first, then any session (ties go to the request's session); then it prefills the rest. Snapshots are evicted LRU
under a byte budget. `Engine` runs requests on it: lookup, resume, prefill with snapshot points, greedy speculative
decoding (spec/specdec.spec_decode) or sampled plain decoding, and the end-of-reply snapshot.
"""
import copy
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spec"))
import kernels  # noqa: E402
import volundr as V  # noqa: E402

SMALL = 4 << 20  # arrays below this are copied into the snapshot, larger ones referenced (see the module docstring)


def _own(a):
    """A compact copy (mx.array of a view copies just the view, so it does not pin the parent buffer)."""
    return None if a is None else mx.array(a)


def _ref(a):
    """A new array object on the same buffer: in-place writes to `a` (which replace a's contents) do not reach it."""
    return None if a is None else copy.copy(a)


def _keep(a):
    return None if a is None else (_own(a) if a.nbytes < SMALL else _ref(a))


class Rows:
    """A buffer kept as its leading `arr.shape[axis]` rows: the rest of the full `shape` is zeros (never written yet).
    Used when at most half of a ring or pooled-block buffer is in use (short contexts): a 35-token snapshot is then
    ~0.18 GB instead of ~0.66 GB, as the BCSA rings (4096 + 512 rows per layer), the pooled blocks (capacity in steps of
    1024) and the drafter ring (4096) are allocated full-size."""

    __slots__ = ("arr", "shape", "axis")

    def __init__(self, arr, shape, axis):
        self.arr, self.shape, self.axis = arr, tuple(shape), axis

    def full(self):
        out = mx.zeros(self.shape, self.arr.dtype)
        idx = (slice(None),) * self.axis + (slice(0, self.arr.shape[self.axis]),)
        out[idx] = self.arr
        return out


def _rows(a, n, axis):
    """Keep a (an allocated buffer whose rows from n on along axis are still zeros) compactly if that saves half."""
    if a is None or 2 * n > a.shape[axis]:
        return _keep(a)
    idx = (slice(None),) * axis + (slice(0, n),)
    return Rows(_own(a[idx]), a.shape, axis)


def _restore(x):
    return x.full() if isinstance(x, Rows) else _ref(x)


def _arr(x):
    return x.arr if isinstance(x, Rows) else x


# ----------------------------------------------------------------------------- snapshot / restore
@dataclass
class State:
    pos: int  # tokens in the caches
    layers: list  # per model cache entry: (kind, ...) tuples
    draft: Optional[tuple] = None  # (k list, v list) of the drafter ring
    logits: Optional[mx.array] = None  # [vocab] logits of position pos - 1
    nbytes: int = 0

    def arrays(self):
        out = [_arr(a) for t in self.layers for a in t[1:] if isinstance(_arr(a), mx.array)]
        if self.draft is not None:
            out += [_arr(a) for a in self.draft[0] + self.draft[1]]
        if self.logits is not None:
            out.append(self.logits)
        return out


def cache_pos(cache) -> int:
    """Tokens held by a model cache (its BCSA / dense offsets; every layer agrees)."""
    for c in cache:
        if isinstance(c, (V.BCSACache, KVCache)):
            return c.offset
    raise ValueError("no positional cache layer")


def capture(cache, dcache=None, logits=None) -> State:
    """Snapshot a model cache (+ the drafter's context cache and the last position's logits). Evaluates the arrays
    it keeps. The caches can go on being used; the snapshot does not change."""
    layers = []
    for c in cache:
        if isinstance(c, ArraysCache):
            assert c.left_padding is None and c.lengths is None
            layers.append(("arrays", *[_keep(a) for a in c.cache]))
        elif isinstance(c, V.BCSACache):
            # ring rows past the offset are untouched while offset < W (the far slots fill only once the ring is
            # full); pooled rows past nb are untouched
            ring = c.offset if c.offset < c.W else 1 << 62
            layers.append(("bcsa", c.W, c.C, c.F, c.offset, c.nb, _rows(c.k, ring, 2), _rows(c.v, ring, 2),
                           _rows(c.pk, c.nb, 2), _rows(c.pv, c.nb, 2), _rows(c.bk, c.nb, 1), _keep(c.ik_open)))
        elif isinstance(c, KVCache):
            layers.append(("kv", c.offset, _keep(c.keys), _keep(c.values)))
        elif isinstance(c, V.EngramState):
            layers.append(("engram", c.keep, _keep(c.prev)))
        else:
            raise TypeError(f"no snapshot for cache type {type(c).__name__}")
    pos = cache_pos(cache)
    draft = None
    if dcache is not None:  # ring slots written so far: positions < pos, plus a draft block's 8 (slot = position)
        n = pos + 8 if pos + 8 < dcache.k[0].shape[2] else 1 << 62
        draft = ([_rows(a, n, 2) for a in dcache.k], [_rows(a, n, 2) for a in dcache.v])
    st = State(pos, layers, draft, _own(logits))
    arrs = st.arrays()
    mx.eval(arrs)
    st.nbytes = sum(a.nbytes for a in arrs)
    return st


def restore(st: State, model=None, drafter=None):
    """New caches equal to the snapshot (cache list, drafter cache or None). They share the snapshot's buffers until
    they write (MLX copies a shared buffer on write), so the snapshot stays valid."""
    cache = []
    for t in st.layers:
        kind = t[0]
        if kind == "arrays":
            c = ArraysCache(size=len(t) - 1)
            for i, a in enumerate(t[1:]):
                c[i] = _ref(a)
        elif kind == "bcsa":
            _, W, C, F, offset, nb, k, v, pk, pv, bk, iko = t
            c = V.BCSACache(W, C, F)
            c.offset, c.nb = offset, nb
            c.k, c.v, c.pk, c.pv, c.bk, c.ik_open = map(_restore, (k, v, pk, pv, bk, iko))
        elif kind == "kv":
            c = KVCache()
            c.offset, c.keys, c.values = t[1], _ref(t[2]), _ref(t[3])
        elif kind == "engram":
            c = V.EngramState(t[1])
            c.prev = _ref(t[2])
        else:
            raise TypeError(kind)
        cache.append(c)
    dcache = None
    if st.draft is not None:
        from dflash import DraftCache

        dcache = DraftCache.__new__(DraftCache)  # (the constructor allocates the zero ring)
        dcache.k, dcache.v = [_restore(a) for a in st.draft[0]], [_restore(a) for a in st.draft[1]]
    return cache, dcache


def lookup_state(lk):
    """The prompt-lookup state needed to continue a reply identically (running means, counters; the history is the
    token sequence itself)."""
    return copy.deepcopy({k: getattr(lk, k) for k in ("on", "lk", "dr", "used", "steps", "acc_lookup", "acc_draft")})


def restore_lookup(lk, state, ids):
    for k, v in copy.deepcopy(state).items():
        setattr(lk, k, v)
    lk.reset(list(ids))
    return lk


# ----------------------------------------------------------------------------- prefix cache
@dataclass(eq=False)
class Snapshot:
    id: int
    session: Optional[str]
    kind: str  # "prompt" | "reply" | "interval"
    tokens: np.ndarray  # int32 [pos]: exactly what the caches hold
    state: State
    used: int = 0  # LRU clock
    hits: int = 0
    created: float = field(default_factory=time.time)

    @property
    def pos(self):
        return int(self.tokens.shape[0])

    @property
    def nbytes(self):
        return self.state.nbytes + self.tokens.nbytes


def env_budget_gb(default=12.0) -> float:
    v = os.environ.get("VOLUNDR_SESSION_CACHE_GB", "").strip()
    return default if v == "" else float(v)


class SessionCache:
    """Snapshots under a byte budget, LRU. find() returns the longest snapshot whose tokens are a prefix of the
    prompt (and that leaves at least one token to prefill unless it carries the last position's logits)."""

    def __init__(self, budget_gb: float = None, interval: int = 4096, min_gap: int = 256):
        self.budget = int((env_budget_gb() if budget_gb is None else budget_gb) * 1e9)
        self.interval, self.min_gap = interval, min_gap
        self.snaps: List[Snapshot] = []
        self._ids = itertools.count(1)
        self._clock = itertools.count(1)
        self.evicted = 0
        self.evicted_bytes = 0

    @property
    def enabled(self):
        return self.budget > 0

    @property
    def nbytes(self):
        return sum(s.nbytes for s in self.snaps)

    def find(self, ids, session=None) -> Optional[Snapshot]:
        if not self.enabled or not self.snaps:
            return None
        ids = np.asarray(ids, dtype=np.int32)
        n = ids.shape[0]
        best, key = None, None
        for s in self.snaps:
            p = s.pos
            if p > n or (p == n and s.state.logits is None) or p == 0:
                continue
            if not np.array_equal(ids[:p], s.tokens):
                continue
            k = (p, s.session == session and session is not None, s.used)
            if key is None or k > key:
                best, key = s, k
        if best is not None:
            best.used = next(self._clock)
            best.hits += 1
        return best

    def has(self, tokens) -> bool:
        tokens = np.asarray(tokens, dtype=np.int32)
        return any(s.pos == tokens.shape[0] and np.array_equal(s.tokens, tokens) for s in self.snaps)

    def add(self, session, kind, tokens, state: State) -> Optional[Snapshot]:
        """Keep a snapshot (evicting LRU ones to fit the budget). An existing snapshot of the same tokens is kept
        instead (and touched). None if the snapshot alone exceeds the budget."""
        if not self.enabled:
            return None
        tokens = np.array(tokens, dtype=np.int32)
        assert tokens.shape[0] == state.pos, (tokens.shape, state.pos)
        for s in self.snaps:
            if s.pos == state.pos and np.array_equal(s.tokens, tokens):
                s.used = next(self._clock)
                return s
        snap = Snapshot(next(self._ids), session, kind, tokens, state, used=next(self._clock))
        if snap.nbytes > self.budget:
            return None
        self.evict(self.budget - snap.nbytes)
        self.snaps.append(snap)
        return snap

    def evict(self, limit: int):
        """Drop least recently used snapshots until the total is at most `limit` bytes."""
        self.snaps.sort(key=lambda s: s.used)
        while self.snaps and self.nbytes > limit:
            s = self.snaps.pop(0)
            self.evicted += 1
            self.evicted_bytes += s.nbytes
        if not self.snaps:
            mx.clear_cache()

    def clear(self):
        self.snaps.clear()
        mx.clear_cache()

    def points(self, start: int, n: int) -> List[int]:
        """Snapshot positions inside a prefill of [start, n): multiples of `interval`, not within min_gap of either
        end (the resume point and the end-of-prompt snapshot cover those)."""
        if not self.enabled or self.interval <= 0:
            return []
        first = (start // self.interval + 1) * self.interval
        return [p for p in range(first, n, self.interval) if p - start >= self.min_gap and n - p >= self.min_gap]

    def stats(self) -> dict:
        return {"snapshots": len(self.snaps), "bytes": self.nbytes, "budget_bytes": self.budget,
                "evicted": self.evicted, "evicted_bytes": self.evicted_bytes,
                "sessions": len({s.session for s in self.snaps}),
                "items": [{"id": s.id, "session": s.session, "kind": s.kind, "pos": s.pos, "bytes": s.nbytes,
                           "hits": s.hits} for s in sorted(self.snaps, key=lambda s: s.id)]}


# ----------------------------------------------------------------------------- prefill and decode on a state
class Cancelled(Exception):
    pass


def prefill(model, drafter, cache, dcache, ids, start=0, chunk=2048, points=(), on_point=None, cancelled=None,
            layer_sync=None):
    """Feed ids[start:] into caches that hold ids[:start] (dcache: the drafter context of their last WINDOW
    positions). Chunks of `chunk` tokens, split at each of `points` (positions in (start, len(ids))), where
    on_point(pos, logits) is called with the caches holding ids[:pos]. The drafter context is filled for the WINDOW
    positions before every point and before the end. specdec.spec_prefill's evaluation (layer by layer by default).
    Returns the last position's logits [vocab]. cancelled(): checked between chunks (raises Cancelled)."""
    import verify as VF

    return VF.drain(prefill_iter(model, drafter, cache, dcache, ids, start, chunk, points, on_point, cancelled,
                                 layer_sync))


def prefill_iter(model, drafter, cache, dcache, ids, start=0, chunk=2048, points=(), on_point=None, cancelled=None,
                 layer_sync=None, yield_layers=False):
    """prefill() as a generator for the batched scheduler: yields ("chunk", end) after each chunk and, with
    yield_layers (and layer_sync), ("layer", i) after each layer of a chunk (verify.forward_iter), so decode steps can
    run in between. The same ops in the same order; returns prefill()'s value."""
    import specdec
    import verify as VF
    from dflash import WINDOW

    if layer_sync is None:
        layer_sync = specdec.PREFILL_LAYER_SYNC
    n = len(ids)
    assert 0 <= start < n
    arr = mx.array(np.asarray(ids, dtype=np.int32))[None]
    stops = sorted(p for p in set(points) if start < p < n) + [n]
    s, logits = start, None
    for stop in stops:
        while s < stop:
            if cancelled is not None and cancelled():
                raise Cancelled(s)
            e = min(stop, s + chunk)
            lo = max(s, stop - WINDOW)  # the rows of this chunk the drafter context needs at `stop`
            want = dcache is not None and lo < e
            fwd = VF.forward_iter(model, arr[:, s:e], cache, capture=drafter.capture_layers if want else (),
                                  layer_sync=layer_sync, yields=yield_layers)
            logits, cap, _ = yield from _tag(fwd, "layer")
            if want:
                drafter.materialize(dcache, cap[lo - s :], lo)
            mx.eval(logits, [c.state for c in cache], dcache.state if dcache is not None else [])
            mx.clear_cache()
            s = e
            yield ("chunk", e)
        if stop < n and on_point is not None:
            on_point(stop, logits[-1])
    return logits[-1]


def _tag(gen, kind):
    """Re-yield a generator's values as (kind, value); return its return value."""
    try:
        while True:
            v = next(gen)
            yield (kind, v)
    except StopIteration as e:
        return e.value


def decode_step(model, cache, tok: int, capture=()):
    """One plain decode step (Model.__call__ on one token, the same kernels and values) that also returns the
    drafter features of that position (stream 0 at the entry of the capture layers, as verify.forward captures them)
    -> (logits [vocab], features [1, 5 * hidden] or None)."""
    inner, a = model.model, model.args
    inputs = mx.array([[tok]], dtype=mx.int32)
    x = inner.embed_tokens(inputs)
    rows = None
    if a.engram_layers:
        st = cache[-1]
        prev = st.prev if st.prev is not None else mx.full((1, st.keep), -1, dtype=mx.int32)
        full = mx.concatenate([prev, inputs], axis=1)
        st.prev = full[:, -st.keep :]
        rows = [V.engram_rows(full, k, 1, a.engram_rows) for k in a.engram_orders]
    H = mx.broadcast_to(x[:, :, None, :], (1, 1, a.mhc_streams, a.hidden_size))
    hmean = x
    caps = []
    if V.use_kernels() and V.DF.active(x):  # VolundrModel._fused_layers
        nl = len(inner.layers)
        hn = kernels.rms_norm(hmean, inner.mhc[0].norm.weight, inner.mhc[0].norm.eps, offset=False)[1]
        for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
            if i in capture:
                caps.append(H[0, :, 0])
            x_in, h, post, res = mhc.mix_norm(H, hn, layer.input_layernorm.weight, layer.input_layernorm.eps)
            y = layer(x_in, c, rows, h=h, split=True)
            H, hn = mhc.combine_norm(H, post, res, y, x_in, inner.mhc[i + 1].norm.weight if i + 1 < nl else None)
    else:
        for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
            if i in capture:
                caps.append(H[0, :, 0])
            x_in, post, res = mhc.mix(H, hmean)
            y = layer(x_in, c, rows)
            H, hmean = mhc.combine(H, post, res, y, x_in)
    logits = model.lm_head(inner.norm(H[:, :, 0]))[0, -1]
    return logits, (mx.concatenate(caps, axis=-1) if caps else None)


# ----------------------------------------------------------------------------- requests
@dataclass
class Sampling:
    """The worker protocol's sampling object (agens-server SamplingParams)."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    max_new_tokens: int = 256
    seed: Optional[int] = None
    logit_bias: list = field(default_factory=list)  # [(id, bias)]
    suppress_tokens: list = field(default_factory=list)
    stop_token_ids: list = field(default_factory=list)
    ignore_eos: bool = False
    reasoning_budget: Optional[int] = None
    reasoning_budget_soft_bias: float = 0.0

    @classmethod
    def from_dict(cls, d: dict):
        names = cls.__dataclass_fields__.keys()
        s = cls(**{k: v for k, v in (d or {}).items() if k in names and v is not None})
        s.logit_bias = [(int(t), float(b)) for t, b in s.logit_bias]
        return s

    @property
    def greedy(self):
        return self.top_k == 1 or self.temperature < 1e-6

    def needs_logits(self, vocab_valid: int, produced: int = 0) -> bool:
        """Anything beyond an argmax over the valid vocabulary (upstream sampler.rs needs_logits)."""
        return (not self.greedy or bool(self.logit_bias) or self.repetition_penalty != 1.0
                or self.frequency_penalty != 0.0 or self.presence_penalty != 0.0
                or (produced < self.min_new_tokens and bool(self.stop_token_ids))
                or any(t < vocab_valid for t in self.suppress_tokens))

    def speculative(self, vocab_valid: int) -> bool:
        """Greedy requests decode speculatively (DFlash2 + prompt lookup); everything else plainly."""
        return self.greedy and not self.needs_logits(vocab_valid) and self.reasoning_budget is None


class Engine:
    """Runs requests one at a time on the model with the session cache (None or budget 0: no cache)."""

    def __init__(self, model, drafter, cache: Optional[SessionCache], T=8, chunk=2048, lookup=True, loop="",
                 think_ids=(None, None)):
        import specdec

        self.model, self.drafter, self.cache = model, drafter, cache
        self.T, self.chunk, self.lookup = T, chunk, lookup
        self.loop = specdec.loop_options([f for f in loop.split("@") if f])
        assert self.loop.get("sched", specdec.SCHED) == "early", "the session engine runs the early spec loop"
        self.loop.pop("sched", None)
        self.think_ids = think_ids
        self.vv = model.args.vocab_valid
        self.requests = 0

    def _snap(self, session, kind, tokens, cache, dcache, logits):
        if self.cache is None or not self.cache.enabled:
            return None
        if self.cache.has(tokens):
            return None
        return self.cache.add(session, kind, tokens, capture(cache, dcache, logits))

    def generate(self, ids, sp: Sampling, session=None, emit: Callable = None, cancelled: Callable = None) -> dict:
        """Run one request. emit(list of ids) streams each step's tokens (returning False cancels); cancelled() is
        polled during the prefill. Returns the request's record (timings, cache use, finish)."""
        import specdec

        t0 = time.perf_counter()
        self.requests += 1
        mx.reset_peak_memory()
        model, drafter = self.model, self.drafter
        ids = [int(t) for t in ids]
        n = len(ids)
        ids_np = np.asarray(ids, dtype=np.int32)
        if n == 0:
            raise ValueError("empty prompt")
        snap = self.cache.find(ids_np, session) if self.cache is not None else None
        if snap is not None:
            cache, dcache = restore(snap.state)
            start, last = snap.pos, (_ref(snap.state.logits) if snap.pos == n else None)
        else:
            cache, dcache = model.make_cache(), drafter.make_cache()
            start, last = 0, None
        t_restore = time.perf_counter()
        rec = {"prompt_tokens": n, "cached_tokens": start, "prefilled_tokens": n - start,
               "snapshot": None if snap is None else {"id": snap.id, "kind": snap.kind, "session": snap.session,
                                                      "same_session": snap.session == session}}
        n_points = [0]

        def on_point(pos, logits):
            if self._snap(session, "interval", ids_np[:pos], cache, dcache, logits) is not None:
                n_points[0] += 1

        if start < n:
            pts = self.cache.points(start, n) if self.cache is not None else []
            try:
                last = prefill(model, drafter, cache, dcache, ids, start, self.chunk, pts, on_point, cancelled)
            except Cancelled as e:  # the worker reports it; the interval snapshots taken so far stay
                rec.update(cancelled_at=e.args[0] if e.args else None, interval_snapshots=n_points[0],
                           prefill_s=time.perf_counter() - t_restore)
                e.rec = rec
                raise
            self._snap(session, "prompt", ids_np, cache, dcache, last)
        t_prefill = time.perf_counter()
        rec.update(restore_s=t_restore - t0, prefill_s=t_prefill - t_restore, interval_snapshots=n_points[0])
        first = [None]
        sent = [0]
        stop_ids = set() if sp.ignore_eos else set(sp.stop_token_ids)
        cancel = [False]

        def _emit(toks):
            if not toks:
                return True
            if first[0] is None:
                first[0] = time.perf_counter()
            sent[0] += len(toks)
            ok = emit(toks) if emit is not None else True
            if ok is False:
                cancel[0] = True
            return ok is not False

        max_new = max(0, int(sp.max_new_tokens))
        out, mode, steps, acc = [], "none", 0, None
        if max_new > 0 and sp.speculative(self.vv):
            mode = "spec"
            from lookup import Lookup

            lk = Lookup() if self.lookup else None
            out, r = specdec.spec_decode(model, drafter, cache, dcache, ids, last, max_new, self.T, stop_ids, lk,
                                         emit=_emit, finalize=True, **self.loop)
            steps, acc = r["steps"], (sum(r["acc"]) / len(r["acc"]) if r["acc"] else None)
            end_logits = r["last_logits"]
        elif max_new > 0:
            mode = "sample"
            out, end_logits = self._sample(cache, dcache, ids, last, sp, _emit)
            steps = max(0, len(out) - 1)
        t_end = time.perf_counter()
        if out:  # the caches hold ids + out[:-1]
            assert cache_pos(cache) == n + len(out) - 1, (cache_pos(cache), n, len(out))
            if len(out) > 1:
                self._snap(session, "reply", np.concatenate([ids_np, np.asarray(out[:-1], np.int32)]), cache, dcache,
                           end_logits)
        t_snap = time.perf_counter()
        finish = "cancelled" if cancel[0] else ("stop" if out and out[-1] in stop_ids else "length")
        ttft = (first[0] - t0) if first[0] is not None else None
        dec_s = (t_end - first[0]) if first[0] is not None else None
        rec.update(mode=mode, generated=len(out), finish=finish, steps=steps, tokens_per_step=acc,
                   ttft_s=ttft, decode_s=dec_s,
                   decode_tok_s=((len(out) - 1) / dec_s) if dec_s and len(out) > 1 else None,
                   snapshot_s=t_snap - t_end, total_s=t_snap - t0, peak_gb=mx.get_peak_memory() / 1e9,
                   active_gb=mx.get_active_memory() / 1e9)
        if self.cache is not None:
            rec["cache_bytes"] = self.cache.nbytes
            rec["cache_snapshots"] = len(self.cache.snaps)
        rec["tokens"] = out
        del cache, dcache
        mx.clear_cache()
        return rec

    def _sample(self, cache, dcache, ids, last, sp: Sampling, emit):
        """Plain decoding with the sampler (sampler.py); the drafter features of every fed token are kept and its
        context filled at the end, so the end-of-reply snapshot serves later speculative requests too."""
        from dflash import WINDOW
        from sampler import Sampler

        smp = Sampler(sp, prompt=ids, think_ids=self.think_ids, vocab_valid=self.vv, request_id=self.requests)
        capl = self.drafter.capture_layers if dcache is not None else ()
        stop_ids = set() if sp.ignore_eos else set(sp.stop_token_ids)
        out, feats = [], []
        logits = last
        P = len(ids)
        while True:
            t = smp.choose(logits)
            out.append(t)
            ok = emit([t])
            if (t in stop_ids) or len(out) >= sp.max_new_tokens or not ok:
                break
            logits, f = decode_step(self.model, cache, t, capl)
            mx.eval(logits, [c.state for c in cache])
            if f is not None:
                feats.append(f)
                if len(feats) > WINDOW:
                    feats.pop(0)
            P += 1
        if feats:
            self.drafter.materialize(dcache, mx.concatenate(feats, axis=0), P - len(feats))
            mx.eval(dcache.state)
        return out, logits
