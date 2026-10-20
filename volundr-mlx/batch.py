"""Batched decode with per-row positions: B live sequences of different lengths in one [B, 1] forward per step.

Layout ("slots"). A `BatchCache` holds one row per live sequence, rows 0..B-1, each at its own position pos[b] (the
number of tokens it holds):
- KDA (ArraysCache) and Engram (EngramState): the [B, ...] arrays the decode kernels already take per row. A step
  rebinds them to new arrays (the recurrence writes a new state), so a row joining or leaving costs one gather of the
  KDA states (~0.17 GB per row), once per membership change, never per step.
- BCSA (`BatchBCSACache`) and the dense layer (`BatchKVCache`): preallocated buffers with S >= B slot rows, written in
  place at each row's own slots (a scatter: ring slot pos % W, the open block's indexer key at pos % C, the pooled
  block a row closes at index pos // C, the dense K/V at pos). The forward reads the first B rows as views, with one
  SDPA per layer over the longest row's keys and a length mask (a row's valid keys are always a prefix [0, N_b): the
  ring, then the far blocks gathered behind it). A leaving row is swap-removed: the last row is copied into its slot
  (one row). A joining row is written into slot B. S doubles when full; everything is freed when the batch empties.
- Ring width. Rows short of the window do not need the whole 4096 + 512-row ring: the BCSA ring buffers start at 256
  rows and grow (256, 512, 1024, 2048, then W + F) when the longest row reaches them.

Per-row positions everywhere else: RoPE takes the [B] offsets (mx.fast.rope with an array offset, the same values as
per-row scalar offsets), far-block selection runs the B = 1 scoring and radix-select kernels per row with the row's
own position and block count, the far blocks are gathered per row.

Numerics. Every op works per row with the B = 1 arithmetic, so a batch of one row gives session.decode_step's logits
bit for bit (tests/test_batch.py). At B > 1 the matmuls change kernels with the row count (MLX's bf16 matmul, the
stock / NAX 4-bit kernels) and the masked SDPA can switch between MLX's one- and two-pass kernels, so rows match their
B = 1 runs up to bf16 rounding: greedy output differs only at near-ties (NOTES_BATCHED.md, NOTES_BATCHED_SERVE.md).

Snapshots: `capture_row` writes a row in session.State's format (the same layout a B = 1 cache at that position
gives, compact rows), so the session cache serves batched requests exactly like bs = 1 ones.
"""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

import session as S
import volundr as V

RING_SIZES = (256, 512, 1024, 2048)  # BCSA ring widths short of the full window (+ far scratch rows)
BLOCK_STEP = 256  # pooled-block capacity step of the batch buffers (a multiple of 32: bcsa_scores)
KV_STEP = 256  # dense K/V capacity step
B1_BLOCK_STEP = V.BCSACache.block_step  # pooled-block capacity of a B = 1 cache (snapshot layout)


def _row(a, b):
    """Row b of a [S, ...] buffer as a view (an integer index copies in MLX 0.32, see volundr._first)."""
    return a[b : b + 1].squeeze(0)


def _ceil(n, step):
    return ((n + step - 1) // step) * step


class BatchRows:
    """The rows' positions, shared by every layer cache of a batch. pos[b] = tokens row b holds = the position of the
    token it is fed next."""

    def __init__(self):
        self.pos = []
        self._arr = None
        self._steps = {}

    @property
    def B(self):
        return len(self.pos)

    def arr(self):
        if self._arr is None:
            self._arr = mx.array(self.pos, dtype=mx.int32)
        return self._arr

    def changed(self):
        self._arr = None
        self._steps = {}

    def advance(self):
        """After a step's graph is built: every row holds one more token."""
        self.pos = [p + 1 for p in self.pos]
        self.changed()

    def bcsa(self, W, C, topk, F):
        key = (W, C, topk, F)
        if key not in self._steps:
            self._steps[key] = BCSAStep(self.pos, W, C, topk, F)
        return self._steps[key]

    def dense(self):
        if "kv" not in self._steps:
            n = [p + 1 for p in self.pos]
            N = max(n)
            mask = None if min(n) == N else (mx.arange(N)[None] < mx.array(n)[:, None])[:, None, None, :]
            self._steps["kv"] = (N, mask)
        return self._steps["kv"]


class BCSAStep:
    """The BCSA layout of one decode step (the same for every BCSA layer): per row the ring slot written, whether
    the step closes a block, the valid far blocks nv and the far keys nf it attends to (all of them, or the top-k),
    its key count N_b (ring then far, a prefix of the buffer), the mask when the rows' counts differ, and the ring
    width the step needs."""

    def __init__(self, pos, W, C, topk, F):
        B = len(pos)
        self.pos = pos
        self.slot = [p % W for p in pos]
        self.close = [b for b in range(B) if (pos[b] + 1) % C == 0]
        self.open = [b for b in range(B) if (pos[b] + 1) % C != 0]
        self.nv = [max(0, (p - W + 1) // C) for p in pos]  # blocks lying entirely before the window
        self.nf = [min(n, topk) for n in self.nv]
        self.nb_close = [pos[b] // C for b in self.close]  # index of the block each closing row closes
        nk = [W + nf if nf else min(p + 1, W) for p, nf in zip(pos, self.nf)]
        self.N = max(nk)
        self.mask = None if min(nk) == self.N else (mx.arange(self.N)[None] < mx.array(nk)[:, None])[:, None, None, :]
        need = max(max(self.slot) + 1, self.N)
        self.ring = next((r for r in RING_SIZES if r >= need and r < W), W + F)
        self.blocks = max(p // C for p in pos) + 1  # pooled-block rows the step may write
        self.slot_arr = mx.array(self.slot, dtype=mx.int32)
        self.open_idx = mx.array(self.open, dtype=mx.int32) if len(self.open) < B else None
        self.open_r0 = mx.array([pos[b] % C for b in self.open], dtype=mx.int32)
        if self.close:
            self.close_idx = mx.array(self.close, dtype=mx.int32)
            self.close_slots = mx.array([[(pos[b] - C + 1 + i) % W for i in range(C)] for b in self.close],
                                        dtype=mx.int32)
            self.close_nb = mx.array(self.nb_close, dtype=mx.int32)


# ----------------------------------------------------------------------------- slot caches
class _Slots:
    """Buffers with S slot rows on axis 0 (rows 0..B-1 live). Subclasses list their buffer names in `bufs`."""

    batched = True
    bufs = ()

    def __init__(self, rows: BatchRows):
        self.rows = rows
        for n in self.bufs:
            setattr(self, n, None)

    @property
    def S(self):
        a = getattr(self, self.bufs[0])
        return 0 if a is None else a.shape[0]

    @property
    def state(self):
        return [a for a in (getattr(self, n) for n in self.bufs) if a is not None]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.state)

    def free(self):
        for n in self.bufs:
            setattr(self, n, None)

    def _resize(self, name, shape):
        """Grow buffer `name` to `shape` (each axis at least its current size), keeping its contents."""
        a = getattr(self, name)
        if a is not None and tuple(a.shape) == tuple(shape):
            return
        new = mx.zeros(shape, a.dtype if a is not None else self.dtypes[name])
        if a is not None:
            new[tuple(slice(0, n) for n in a.shape)] = a
        setattr(self, name, new)

    def move_row(self, src, dst):
        """Slot dst := slot src (swap-remove of a leaving row). The row is copied out and evaluated first: a pending
        read of the buffer would stop the write from reusing it (MLX would copy the whole buffer)."""
        rows = {n: mx.array(getattr(self, n)[src : src + 1]) for n in self.bufs if getattr(self, n) is not None}
        mx.eval(list(rows.values()))
        for n, r in rows.items():
            getattr(self, n)[dst : dst + 1] = r


class BatchBCSACache(_Slots):
    """BCSA state of a batch: ring K/V [S, KVH, R, D] (R = a RING_SIZES width while every row is short of it, else
    W + F: the window plus the far-block scratch rows), pooled K/V [S, KVH, cap, D], pooled indexer keys [S, cap, Di],
    the open block's indexer keys [S, C, Di]. Positions are the shared BatchRows."""

    bufs = ("k", "v", "pk", "pv", "bk", "ik_open")

    def __init__(self, rows, W, C, F, KVH, D, Di, dtype=V.BF16):
        super().__init__(rows)
        self.W, self.C, self.F, self.KVH, self.D, self.Di = W, C, F, KVH, D, Di
        self.dtypes = {n: dtype for n in self.bufs}

    @property
    def R(self):
        return 0 if self.k is None else self.k.shape[2]

    @property
    def cap(self):
        return 0 if self.pk is None else self.pk.shape[2]

    def ensure(self, S, R, blocks):
        """At least S slots, ring width R, `blocks` pooled-block rows."""
        S, R = max(S, self.S), max(R, self.R)
        cap = max(self.cap, _ceil(max(blocks, 1), BLOCK_STEP))
        KVH, D, Di, C = self.KVH, self.D, self.Di, self.C
        self._resize("k", (S, KVH, R, D))
        self._resize("v", (S, KVH, R, D))
        self._resize("pk", (S, KVH, cap, D))
        self._resize("pv", (S, KVH, cap, D))
        self._resize("bk", (S, cap, Di))
        self._resize("ik_open", (S, C, Di))

    def put(self, b, c: V.BCSACache):
        """Write a B = 1 cache into slot b (the slot's rows past the cache's contents are never read)."""
        P, R = c.offset, self.R
        assert c.k is not None and (P < R or R == self.W + self.F), (P, R)
        self.k[b : b + 1] = c.k[:, :, :R]
        self.v[b : b + 1] = c.v[:, :, :R]
        nb = P // self.C
        assert nb == c.nb, (nb, c.nb)
        if nb:
            self.pk[b : b + 1, :, :nb] = c.pk[:, :, :nb]
            self.pv[b : b + 1, :, :nb] = c.pv[:, :, :nb]
            self.bk[b : b + 1, :nb] = c.bk[:, :nb]
        self.ik_open[b : b + 1] = c.ik_open

    # A step writes in this order (Attention._bcsa_batched): write_ring, the far blocks behind the ring, close_blocks,
    # then the SDPA. Each buffer is read only at its last version of the step, except the pooled blocks the far
    # gather reads before close_blocks appends to them; that read is part of the logits graph, which the caller
    # evaluates first, so every write can still reuse its buffer (a pending read of an older version would make MLX
    # copy the whole buffer instead).
    def write_ring(self, k, v, ik, st: BCSAStep):
        """One decode step's keys: k, v [B, KVH, 1, D] (post-RoPE) into each row's ring slot; ik [B, 1, Di] (normed
        indexer keys) into the open block of the rows that do not close one."""
        B, KVH = k.shape[0], self.KVH
        self.ensure(B, st.ring, st.blocks)
        rows, heads = mx.arange(B), mx.arange(KVH)
        self.k[rows[:, None], heads[None, :], st.slot_arr[:, None]] = k[:, :, 0, :]
        self.v[rows[:, None], heads[None, :], st.slot_arr[:, None]] = v[:, :, 0, :]
        if st.open:  # the open block's indexer key at pos % C
            ik = ik[:, 0]
            vals = ik if st.open_idx is None else mx.take(ik, st.open_idx, axis=0)
            self.ik_open[rows if st.open_idx is None else st.open_idx, st.open_r0] = vals

    def put_far(self, b, fk, fv):
        """Row b's far keys [1, KVH, nf, D] into the scratch rows behind the ring."""
        n = fk.shape[2]
        self.k[b : b + 1, :, self.W : self.W + n] = fk
        self.v[b : b + 1, :, self.W : self.W + n] = fv

    def close_blocks(self, ik, st: BCSAStep):
        """Rows whose step closes block [pos - C + 1, pos]: its mean K/V from the ring (after the write) and the mean of
        its indexer keys (the open block's C - 1 and this step's), appended at block index pos // C. BCSACache.update's
        arithmetic for one position."""
        if not st.close:
            return
        ci, C = st.close_idx, self.C
        heads = mx.arange(self.KVH)
        ck = self.k[ci[:, None, None], heads[None, :, None], st.close_slots[:, None, :]]  # [nc, KVH, C, D]
        cv = self.v[ci[:, None, None], heads[None, :, None], st.close_slots[:, None, :]]
        iko = mx.concatenate([mx.take(self.ik_open, ci, axis=0)[:, : C - 1], mx.take(ik[:, 0], ci, axis=0)[:, None]],
                             axis=1)  # [nc, C, Di]
        pk, pv = V.BCSACache._pool(ck, C, 2), V.BCSACache._pool(cv, C, 2)  # [nc, KVH, 1, D]
        pi = V.BCSACache._pool(iko, C, 1)  # [nc, 1, Di]
        self.pk[ci[:, None], heads[None, :], st.close_nb[:, None]] = pk[:, :, 0]
        self.pv[ci[:, None], heads[None, :], st.close_nb[:, None]] = pv[:, :, 0]
        self.bk[ci, st.close_nb] = pi[:, 0]

    def capture_row(self, b, P):
        """Row b (holding P tokens) in session.capture's "bcsa" layout of a B = 1 cache, as own (compact) copies."""
        W, C, F = self.W, self.C, self.F
        full = (1, self.KVH, W + F, self.D)
        if P < W:
            k = S.Rows(mx.array(self.k[b : b + 1, :, :P]), full, 2)
            v = S.Rows(mx.array(self.v[b : b + 1, :, :P]), full, 2)
        else:
            assert self.R == W + F
            k, v = mx.array(self.k[b : b + 1]), mx.array(self.v[b : b + 1])
        nb = P // C
        if nb:
            cap = _ceil(nb, B1_BLOCK_STEP)
            pk = S.Rows(mx.array(self.pk[b : b + 1, :, :nb]), (1, self.KVH, cap, self.D), 2)
            pv = S.Rows(mx.array(self.pv[b : b + 1, :, :nb]), (1, self.KVH, cap, self.D), 2)
            bk = S.Rows(mx.array(self.bk[b : b + 1, :nb]), (1, cap, self.Di), 1)
        else:
            pk = pv = bk = None
        return ("bcsa", W, C, F, P, nb, k, v, pk, pv, bk, mx.array(self.ik_open[b : b + 1]))


class BatchKVCache(_Slots):
    """Dense-attention K/V of a batch: [S, KVH, cap, D], row b's position pos[b] written in place."""

    bufs = ("keys", "values")

    def __init__(self, rows, KVH, D, dtype=V.BF16):
        super().__init__(rows)
        self.KVH, self.D = KVH, D
        self.dtypes = {n: dtype for n in self.bufs}

    @property
    def cap(self):
        return 0 if self.keys is None else self.keys.shape[2]

    def ensure(self, S, n):
        S, cap = max(S, self.S), max(self.cap, _ceil(max(n, 1), KV_STEP))
        self._resize("keys", (S, self.KVH, cap, self.D))
        self._resize("values", (S, self.KVH, cap, self.D))

    def put(self, b, c):
        P = c.offset
        self.keys[b : b + 1, :, :P] = c.keys[:, :, :P]
        self.values[b : b + 1, :, :P] = c.values[:, :, :P]

    def attend(self, q, k, v, scale):
        """Write each row's K/V at its position, then attend to positions [0, pos_b] (KVCache + SDPA at L = 1)."""
        B = q.shape[0]
        rows = self.rows
        N, mask = rows.dense()
        self.ensure(B, N)
        r, h = mx.arange(B), mx.arange(self.KVH)
        self.keys[r[:, None], h[None, :], rows.arr()[:, None]] = k[:, :, 0, :]
        self.values[r[:, None], h[None, :], rows.arr()[:, None]] = v[:, :, 0, :]
        return mx.fast.scaled_dot_product_attention(q, self.keys[:B, :, :N], self.values[:B, :, :N], scale=scale,
                                                    mask=mask)

    def capture_row(self, b, P):
        return ("kv", P, mx.array(self.keys[b : b + 1, :, :P]), mx.array(self.values[b : b + 1, :, :P]))


# ----------------------------------------------------------------------------- the batch
class BatchCache:
    """The model cache of a decode batch: one row per live sequence, keyed by an arbitrary object (the scheduler's
    request). `apply(leave, join)` changes the membership; `forward` runs one step for every row."""

    def __init__(self, model):
        a = model.args
        self.args = a
        self.rows = BatchRows()
        self.keys = []
        self.layers = []
        dt = model.model.norm.weight.dtype  # the activations' dtype (bf16; fp32 for the tiny test model)
        for m in a.mixer_types:
            if m == "kda":
                self.layers.append(ArraysCache(size=2))
            elif m == "bcsa":
                self.layers.append(BatchBCSACache(self.rows, a.bcsa_window, a.bcsa_compress, a.bcsa_topk,
                                                  a.num_key_value_heads, a.head_dim, a.bcsa_index_dim, dt))
            else:
                self.layers.append(BatchKVCache(self.rows, a.num_key_value_heads, a.head_dim, dt))
        self.layers.append(V.EngramState(max(a.engram_orders) - 1))

    @property
    def B(self):
        return self.rows.B

    @property
    def pos(self):
        return self.rows.pos

    @property
    def state(self):
        out = []
        for c in self.layers:
            out += [x for x in c.state if x is not None] if not isinstance(c, ArraysCache) else \
                [x for x in c.cache if x is not None]
        return out

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.state)

    def row(self, key) -> int:
        return self.keys.index(key)

    def apply(self, leave=(), join=()):
        """Remove the rows of the keys in `leave` (swap-remove: the last row moves into the hole) and append one row
        per (key, B = 1 cache) in `join`. Returns nothing; self.keys gives the new order."""
        old_B = self.B
        perm = list(range(old_B))  # perm[new row] = old row, for the per-row arrays (KDA, Engram)
        slots = [c for c in self.layers if isinstance(c, _Slots)]
        for b in sorted((self.keys.index(k) for k in leave), reverse=True):
            last = len(perm) - 1
            if b != last:
                for c in slots:
                    c.move_row(last, b)
                perm[b], self.keys[b], self.rows.pos[b] = perm[last], self.keys[last], self.rows.pos[last]
            perm.pop()
            self.keys.pop()
            self.rows.pos.pop()
        new = []
        if join:
            base = len(self.keys)
            S_need = base + len(join)
            S_new = max(self.slots_cap, 1)
            while S_new < S_need:
                S_new *= 2
            maxP = max(S.cache_pos(c1) for _, c1 in join)
            for c in slots:
                if isinstance(c, BatchBCSACache):
                    need = max(maxP + 1, max(self.rows.pos, default=0) + 1)
                    R = next((r for r in RING_SIZES if r >= need and r < c.W), c.W + c.F)
                    c.ensure(S_new, R, maxP // c.C + 1)
                else:
                    c.ensure(S_new, maxP + 1)
            for i, (key, c1) in enumerate(join):
                b = base + i
                for c, cc in zip(self.layers, c1):
                    if isinstance(c, _Slots):
                        c.put(b, cc)
                self.keys.append(key)
                self.rows.pos.append(S.cache_pos(c1))
                new.append(c1)
        # per-row arrays: kept rows (in the new order), then the joined rows
        ident = perm == list(range(len(perm)))
        for j, c in enumerate(self.layers):
            if isinstance(c, ArraysCache):
                for i in range(2):
                    parts = []
                    if perm:
                        old = c[i]
                        parts.append(old if (ident and len(perm) == old_B) else
                                     old[: len(perm)] if ident else mx.take(old, mx.array(perm), axis=0))
                    parts += [c1[j][i] for c1 in new]
                    c[i] = None if not parts else (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0))
            elif isinstance(c, V.EngramState):
                parts = []
                if perm:
                    old = c.prev
                    parts.append(old if (ident and len(perm) == old_B) else
                                 old[: len(perm)] if ident else mx.take(old, mx.array(perm), axis=0))
                parts += [c1[j].prev for c1 in new]
                c.prev = None if not parts else (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0))
        self.rows.changed()
        if not self.keys:
            for c in slots:
                c.free()
        mx.eval(self.state)

    @property
    def slots_cap(self):
        for c in self.layers:
            if isinstance(c, _Slots):
                return c.S
        return 0

    def capture_row(self, key, dcache=None, logits=None) -> S.State:
        """Row `key` as a session.State (own copies, the layout session.capture gives for a B = 1 cache at that
        position). dcache: the request's drafter context (B = 1), logits: the logits of its last position."""
        b = self.keys.index(key)
        P = self.rows.pos[b]
        layers = []
        for c in self.layers:
            if isinstance(c, ArraysCache):
                layers.append(("arrays", mx.array(c[0][b : b + 1]), mx.array(c[1][b : b + 1])))
            elif isinstance(c, (BatchBCSACache, BatchKVCache)):
                layers.append(c.capture_row(b, P))
            else:
                layers.append(("engram", c.keep, mx.array(c.prev[b : b + 1])))
        draft = None
        if dcache is not None:
            n = P + 8 if P + 8 < dcache.k[0].shape[2] else 1 << 62
            draft = ([S._rows(a, n, 2) for a in dcache.k], [S._rows(a, n, 2) for a in dcache.v])
        st = S.State(P, layers, draft, None if logits is None else mx.array(logits))
        arrs = st.arrays()
        mx.eval(arrs)
        st.nbytes = sum(a.nbytes for a in arrs)
        return st

    def row_cache(self, key):
        """Row `key` as a B = 1 model cache (copies), e.g. to continue it with the speculative loop."""
        cache, _ = S.restore(self.capture_row(key))
        return cache


def forward(model, bc: BatchCache, toks, capture=()):
    """One decode step of every row: toks [B] int32 (each row's next token, lazy is fine). Returns (logits [B, vocab],
    features [B, 5 * hidden] of the capture layers or None). session.decode_step with B rows; the rows' positions
    advance by one (the graph is built, not evaluated)."""
    inner, a = model.model, model.args
    B = bc.B
    inputs = toks.reshape(B, 1).astype(mx.int32)
    x = inner.embed_tokens(inputs)
    st = bc.layers[-1]
    full = mx.concatenate([st.prev, inputs], axis=1)
    st.prev = full[:, -st.keep :]
    rows = [V.engram_rows(full, n, 1, a.engram_rows) for n in a.engram_orders]
    H = mx.broadcast_to(x[:, :, None, :], (B, 1, a.mhc_streams, a.hidden_size))
    hmean = x
    caps = []
    cache = bc.layers
    if V.use_kernels() and V.DF.active(x):  # VolundrModel._fused_layers
        import kernels

        nl = len(inner.layers)
        hn = kernels.rms_norm(hmean, inner.mhc[0].norm.weight, inner.mhc[0].norm.eps, offset=False)[1]
        for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
            if i in capture:
                caps.append(H[:, 0, 0])
            x_in, h, post, res = mhc.mix_norm(H, hn, layer.input_layernorm.weight, layer.input_layernorm.eps)
            y = layer(x_in, c, rows, h=h, split=True)
            H, hn = mhc.combine_norm(H, post, res, y, x_in, inner.mhc[i + 1].norm.weight if i + 1 < nl else None)
    else:
        for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
            if i in capture:
                caps.append(H[:, 0, 0])
            x_in, post, res = mhc.mix(H, hmean)
            y = layer(x_in, c, rows)
            H, hmean = mhc.combine(H, post, res, y, x_in)
    logits = model.lm_head(inner.norm(H[:, :, 0]))[:, -1]
    bc.rows.advance()
    return logits, (mx.concatenate(caps, axis=-1) if caps else None)
