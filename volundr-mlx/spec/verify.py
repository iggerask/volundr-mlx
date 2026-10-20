"""Target side of speculative decoding: a Volundr forward over T tokens that captures the drafter's context features
and, in verify mode, keeps what `commit(n)` needs to keep exactly the first n of the T tokens.

Mirrors volundr.py (VolundrModel / DecoderLayer / KDA / Attention, GPU kernel paths) except for the state handling:
- KDA: runs the T tokens; commit restores the pre-call conv / recurrent state and replays the recurrence on the first
  n rows' q, k, v, gate, beta (no projections re-run). n = T keeps the verify result as it is.
- BCSA: attends to the window (ring slots + the T new keys) and each query's top-k far blocks (no block a verify query
  can select is closed inside the chunk) without touching the cache, in bcsa_verify.attend; commit appends the
  first n rows.
- Dense attention: the KVCache takes all T rows; commit trims T - n.
- Engram: commit sets the previous-ids state from the committed ids. mHC has no state.
Not bit-exact against T sequential decode steps (different matmul / attention kernels round differently), so greedy
output matches plain decoding except where the top two logits are within rounding.
"""
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import bcsa_verify  # noqa: E402
import kernels  # noqa: E402
import volundr as V  # noqa: E402

F32, BF16 = V.F32, V.BF16
BCSA_IMPL = os.environ.get("VOLUNDR_BCSA_VERIFY", "8q")  # "8q" | "prefill" (see _bcsa_verify)
assert BCSA_IMPL in ("8q", "prefill"), BCSA_IMPL


def _kda(self, x, cache, rec):
    B, L, _ = x.shape
    qkv = self.in_proj_qkv(x)
    z = self.in_proj_z(x).reshape(B, L, self.HV, self.V)
    b = self.in_proj_b(x)
    a = self.f_up(self.f_down(x)).reshape(B, L, self.HV, self.K)
    conv0 = cache[0] if cache[0] is not None else mx.zeros((B, self.kernel - 1, self.conv_dim), qkv.dtype)
    state0 = cache[1] if cache[1] is not None else mx.zeros((B, self.HV, self.V, self.K), F32)
    assert self.K == 128 and self.V == 128 and qkv.dtype == BF16
    q, k, v = kernels.kda_conv_qk4(qkv, conv0, self.conv1d.weight[:, :, 0], self.key_dim, self.K)
    q, k = q.reshape(B, L, self.H, self.K), k.reshape(B, L, self.H, self.K)
    v = v.reshape(B, L, self.HV, self.V)
    g = compute_g(self.A_log.reshape(self.HV, 1), a, self.dt_bias.reshape(self.HV, self.K))
    beta = mx.sigmoid(b.astype(F32))
    out, state = kernels.gated_delta_vec(q, k, v, g, beta, state0)
    keep = self.kernel - 1
    cache[0] = mx.concatenate([conv0, qkv], axis=1)[:, -keep:]
    cache[1] = state
    rec.update(kind="kda", T=L, conv0=conv0, state0=state0, qkv=qkv, q=q, k=k, v=v, g=g, beta=beta)
    out = kernels.gated_norm(out.astype(x.dtype), self.norm.weight, z, self.eps)
    return self.out_proj(out.reshape(B, L, self.value_dim))


def _bcsa_verify(self, x, q, k, v, cache):
    """BCSA attention of the T verify queries without the cache update (see the module docstring); returns the
    attention output and the rows to append on commit.
    "8q" (default): bcsa_verify.attend reads the window from the ring in place and splits the keys over many
    threadgroups with the G heads x T queries of a KV head packed into one row block; far blocks are the union of the
    queries' top-k selections, masked per query. "prefill": the prefill kernels (window gathered out of the ring and
    concatenated, then one threadgroup per 32-query block and head)."""
    B, H, L, D = q.shape
    P, W, C = cache.offset, self.W, self.C
    ik = self.idx_norm(self.idx_k(x))
    if BCSA_IMPL == "8q" and B == 1 and L <= 16 and P > 0 and V.use_kernels():
        # squeeze, not [0]: in MLX 0.32 integer indexing copies the array (~50 us for a 9 MB ring per call), while
        # squeeze is a view; the kernel copies only the small non-contiguous q / k / v
        sq = lambda a: a.squeeze(0)
        nv_max = max(0, (P + L - 1 - W + 1) // C)
        assert nv_max <= cache.nb
        far = ()
        if nv_max:
            qi = self._index_q(x)
            if self.ID == 128 and qi.dtype == BF16:  # Attention._select's kernel path, without the copies
                s = kernels.bcsa_scores(sq(qi), sq(cache.bk), nv_max, self.ID**-0.5)
                sel = kernels.bcsa_topk(s, P, W, C, self.topk)
            else:
                sel = self._select(x, cache, P, nv_max)
            far = (sq(cache.pk), sq(cache.pv), *kernels.far_union(sel, nv_max, 32))
        o = bcsa_verify.attend(sq(q), sq(k), sq(v), sq(cache.k), sq(cache.v), P, W, self.scale, *far)
        return o[None], ik
    nprev = min(P, W - 1)
    if nprev:
        wk, wv = cache.window(nprev)
        keys, vals = mx.concatenate([wk, k], axis=2), mx.concatenate([wv, v], axis=2)
    else:
        keys, vals = k, v
    nv_max = max(0, (P + L - 1 - W + 1) // C)
    assert nv_max <= cache.nb
    if nv_max == 0:
        if nprev == 0:
            o = mx.fast.scaled_dot_product_attention(q, keys, vals, scale=self.scale, mask="causal")
        else:
            o = kernels.bcsa_window(q[0], keys[0], vals[0], self.scale, nprev, W)[None]
    else:
        sel = self._select(x, cache, P, nv_max)
        o = kernels.bcsa_attend(q[0], keys[0], vals[0], cache.pk[0], cache.pv[0], sel, nv_max, self.scale, nprev,
                                W)[None]
    return o, ik


def _attn(self, x, cache, rec):
    B, L, _ = x.shape
    D = self.head_dim
    qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * D)
    q, gate = qg[..., :D], qg[..., D:].reshape(B, L, self.n_heads * D)
    q = self.q_norm(q).transpose(0, 2, 1, 3)
    k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, D)).transpose(0, 2, 1, 3)
    v = self.v_proj(x).reshape(B, L, self.n_kv, D).transpose(0, 2, 1, 3)
    q, k = self._rope(q, cache.offset), self._rope(k, cache.offset)
    if not self.bcsa:
        kk, vv = cache.update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=self.scale, mask="causal" if L > 1 else None)
        rec.update(kind="kv", T=L)
    else:
        o, ik = _bcsa_verify(self, x, q, k, v, cache)
        rec.update(kind="bcsa", T=L, k=k, v=v, ik=ik)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * D)
    return self.o_proj(V._sigmoid_gate(o, gate))


def _layer(self, x, cache, rows, rec):
    eps = self.input_layernorm.eps
    h = kernels.rms_norm(x, self.input_layernorm.weight, eps, offset=True)[1]
    r = _kda(self.linear_attn, h, cache, rec) if self.mixer == "kda" else _attn(self.self_attn, h, cache, rec)
    h, hn = kernels.rms_norm(x, self.post_attention_layernorm.weight, eps, offset=True, r=r)
    y = h + self.mlp(hn)
    if "engram" in self:
        y = self.engram(y, rows)
    return y


# ----------------------------------------------------------------------------- compiled KDA verify layers
# The verify graph of the 54 KDA layers has no Python-side state (no positions, no cache offsets): mHC mix, norms,
# projections, conv, recurrence, gated norm, MLP, Engram, mHC combine are a pure function of (H, hmean, conv state,
# recurrent state, Engram rows). mx.compile traces it once per layer and shape, and later calls replay the trace in C++
# instead of ~30 Python op calls, which cuts the CPU time of building the verify graph (the spec loop builds it while
# the GPU drafts). The same primitives in the same order; checked bit for bit against the eager layer
# (spec_accept/check_compile.py). Keyed on the switches ab_modes.Modes flips, so a trace never outlives its config.
_compiled = {}


def _kda_layer_compiled(layer, mhc, i):
    key = (i, id(layer), V.DF.ENABLED, kernels.rms_norm, V._F32Weight.get)
    f = _compiled.get(key)
    if f is None:
        def fn(H, hmean, conv0, state0, *rows):
            cache = [conv0, state0]
            x_in, post, res = mhc.mix(H, hmean)
            rec = {}
            y = _layer(layer, x_in, cache, list(rows), rec)
            H2, hmean2 = mhc.combine(H, post, res, y, x_in)
            return (H2, hmean2, cache[0], cache[1], rec["qkv"], rec["q"], rec["k"], rec["v"], rec["g"], rec["beta"])

        f = _compiled[key] = mx.compile(fn)
    return f


def forward(model, ids, cache, capture=(), verify=False, all_logits=False, layer_sync=False, compiled=False):
    """ids [1, L] int32. Returns (logits [L or 1, vocab] (all rows when verify or all_logits), features [L, 5 * HID]
    of the capture layers (stream 0 at layer entry; None without capture), the commit record (verify only)).
    layer_sync (prefill chunks): evaluate layer by layer, at most two layers in flight (each layer is submitted with
    async_eval, then the previous one is waited for). Under the spec harness's large command buffers
    (MLX_MAX_OPS_PER_BUFFER=100, no MB cap) an unsynchronised 2048-token chunk lets ~10 command buffers of large
    activations stay alive at once (55-57 GB peak at 12.5K tokens); this bounds them to two layers. Same kernels and
    values; only the evaluation boundaries change."""
    return drain(forward_iter(model, ids, cache, capture, verify, all_logits, layer_sync, compiled))


def drain(gen):
    """Run a generator to its end and return its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def forward_iter(model, ids, cache, capture=(), verify=False, all_logits=False, layer_sync=False, compiled=False,
                 yields=False):
    """forward() as a generator: with layer_sync and yields, it yields after each layer's sync (the previous layer
    done, this one submitted), so a scheduler can run other work between layers (the batched server interleaves decode
    steps with a long prefill this way). The same ops in the same order; returns forward()'s value."""
    a, inner = model.args, model.model
    B, L = ids.shape
    assert B == 1
    x = inner.embed_tokens(ids)
    st = cache[-1]
    prev = st.prev if st.prev is not None else mx.full((B, st.keep), -1, dtype=mx.int32)
    full = mx.concatenate([prev, ids.astype(mx.int32)], axis=1)
    st.prev = full[:, -st.keep :]
    rows = [V.engram_rows(full, n, L, a.engram_rows) for n in a.engram_orders]
    recs = [{} for _ in inner.layers] if verify else None
    H = mx.broadcast_to(x[:, :, None, :], (B, L, a.mhc_streams, a.hidden_size))
    hmean = x
    caps = []
    pending = None
    for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
        if i in capture:
            caps.append(H[0, :, 0])
        if compiled and verify and layer.mixer == "kda":
            conv0, state0 = c[0], c[1]
            if conv0 is None or state0 is None:  # (verify after a prefill: never)
                conv0 = mx.zeros((B, layer.linear_attn.kernel - 1, layer.linear_attn.conv_dim), BF16)
                state0 = mx.zeros((B, layer.linear_attn.HV, layer.linear_attn.V, layer.linear_attn.K), F32)
            H, hmean, c[0], c[1], qkv, q, k, v, g, beta = _kda_layer_compiled(layer, mhc, i)(
                H, hmean, conv0, state0, *rows)
            recs[i].update(kind="kda", T=L, conv0=conv0, state0=state0, qkv=qkv, q=q, k=k, v=v, g=g, beta=beta)
            continue
        x_in, post, res = mhc.mix(H, hmean)
        y = _layer(layer, x_in, c, rows, recs[i]) if verify else layer(x_in, c, rows)
        H, hmean = mhc.combine(H, post, res, y, x_in)
        if layer_sync:
            done = [H, hmean, *caps, *c.state]
            mx.async_eval(done)
            if pending is not None:
                mx.eval(pending)
            pending = done
            if yields:
                yield i
    h = inner.norm(H[:, :, 0])
    logits = model.lm_head(h if (verify or all_logits) else h[:, -1:])[0]
    cap = mx.concatenate(caps, axis=-1) if caps else None
    rec = {"layers": recs, "full": full, "T": L} if verify else None
    return logits, cap, rec


# ----------------------------------------------------------------------------- batched verify (B candidate chains)
# B chains of T tokens verified in one forward against the same (B = 1) cache state, nothing written to the cache;
# commit_batch keeps the first n tokens of one chain. KDA: each chain runs the recurrence from the shared pre-call state
# (broadcast, so the kernel gets its own copy per chain). BCSA: the B = 1 verify branch once per chain (same cache, own
# queries / new keys). Dense attention: the cached keys plus the chain's own, causal. The B = 1 functions above are not
# used here and are unchanged.
def _kda_b(self, x, cache, rec):
    B, L, _ = x.shape
    qkv = self.in_proj_qkv(x)
    z = self.in_proj_z(x).reshape(B, L, self.HV, self.V)
    b = self.in_proj_b(x)
    a = self.f_up(self.f_down(x)).reshape(B, L, self.HV, self.K)
    conv0 = cache[0] if cache[0] is not None else mx.zeros((1, self.kernel - 1, self.conv_dim), qkv.dtype)
    state0 = cache[1] if cache[1] is not None else mx.zeros((1, self.HV, self.V, self.K), F32)
    assert self.K == 128 and self.V == 128 and qkv.dtype == BF16 and conv0.shape[0] == 1
    convb = mx.broadcast_to(conv0, (B, *conv0.shape[1:]))
    q, k, v = kernels.kda_conv_qk4(qkv, convb, self.conv1d.weight[:, :, 0], self.key_dim, self.K)
    q, k = q.reshape(B, L, self.H, self.K), k.reshape(B, L, self.H, self.K)
    v = v.reshape(B, L, self.HV, self.V)
    g = compute_g(self.A_log.reshape(self.HV, 1), a, self.dt_bias.reshape(self.HV, self.K))
    beta = mx.sigmoid(b.astype(F32))
    out, state = kernels.gated_delta_vec(q, k, v, g, beta, mx.broadcast_to(state0, (B, *state0.shape[1:])))
    rec.update(kind="kda", T=L, conv0=conv0, state0=state0, qkv=qkv, q=q, k=k, v=v, g=g, beta=beta, stateT=state)
    out = kernels.gated_norm(out.astype(x.dtype), self.norm.weight, z, self.eps)
    return self.out_proj(out.reshape(B, L, self.value_dim))


def _attn_b(self, x, cache, rec):
    B, L, _ = x.shape
    D = self.head_dim
    qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * D)
    q, gate = qg[..., :D], qg[..., D:].reshape(B, L, self.n_heads * D)
    q = self.q_norm(q).transpose(0, 2, 1, 3)
    k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, D)).transpose(0, 2, 1, 3)
    v = self.v_proj(x).reshape(B, L, self.n_kv, D).transpose(0, 2, 1, 3)
    q, k = self._rope(q, cache.offset), self._rope(k, cache.offset)
    if not self.bcsa:
        P = cache.offset
        os_ = []
        for i in range(B):
            if P:
                kk = mx.concatenate([cache.keys[..., :P, :], k[i : i + 1]], axis=2)
                vv = mx.concatenate([cache.values[..., :P, :], v[i : i + 1]], axis=2)
            else:
                kk, vv = k[i : i + 1], v[i : i + 1]
            os_.append(mx.fast.scaled_dot_product_attention(q[i : i + 1], kk, vv, scale=self.scale, mask="causal"))
        o = mx.concatenate(os_, axis=0)
        rec.update(kind="kv", T=L, k=k, v=v)
    else:
        os_, iks = [], []
        for i in range(B):
            o_i, ik_i = _bcsa_verify(self, x[i : i + 1], q[i : i + 1], k[i : i + 1], v[i : i + 1], cache)
            os_.append(o_i)
            iks.append(ik_i)
        o = mx.concatenate(os_, axis=0)
        rec.update(kind="bcsa", T=L, k=k, v=v, ik=mx.concatenate(iks, axis=0))
    o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * D)
    return self.o_proj(V._sigmoid_gate(o, gate))


def _layer_b(self, x, cache, rows, rec):
    eps = self.input_layernorm.eps
    h = kernels.rms_norm(x, self.input_layernorm.weight, eps, offset=True)[1]
    r = _kda_b(self.linear_attn, h, cache, rec) if self.mixer == "kda" else _attn_b(self.self_attn, h, cache, rec)
    h, hn = kernels.rms_norm(x, self.post_attention_layernorm.weight, eps, offset=True, r=r)
    y = h + self.mlp(hn)
    if "engram" in self:
        y = self.engram(y, rows)
    return y


def forward_batch(model, ids, cache, capture=()):
    """ids [B, L] int32: B candidate chains (row 0 of each is the same bonus token) after the cached context. Returns
    (logits [B, L, vocab], features [B, L, 5 * HID] of the capture layers (None without capture), the commit record).
    The cache is not changed (commit_batch does that)."""
    a, inner = model.args, model.model
    B, L = ids.shape
    x = inner.embed_tokens(ids)
    st = cache[-1]
    prev = st.prev if st.prev is not None else mx.full((1, st.keep), -1, dtype=mx.int32)
    full = mx.concatenate([mx.broadcast_to(prev, (B, st.keep)), ids.astype(mx.int32)], axis=1)
    rows = [V.engram_rows(full, n, L, a.engram_rows) for n in a.engram_orders]
    recs = [{} for _ in inner.layers]
    H = mx.broadcast_to(x[:, :, None, :], (B, L, a.mhc_streams, a.hidden_size))
    hmean = x
    caps = []
    for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
        if i in capture:
            caps.append(H[:, :, 0])
        x_in, post, res = mhc.mix(H, hmean)
        y = _layer_b(layer, x_in, c, rows, recs[i])
        H, hmean = mhc.combine(H, post, res, y, x_in)
    h = inner.norm(H[:, :, 0])
    logits = model.lm_head(h)
    cap = mx.concatenate(caps, axis=-1) if caps else None
    return logits, cap, {"layers": recs, "full": full, "T": L, "B": B}


def commit_batch(cache, rec, b, n):
    """Keep the first n tokens of chain b of a forward_batch in the caches."""
    T = rec["T"]
    st = cache[-1]
    st.prev = rec["full"][b : b + 1, : st.keep + n][:, -st.keep :]
    for c, r in zip(cache, rec["layers"]):
        kind = r["kind"]
        if kind == "kda":
            keep = r["conv0"].shape[1]
            c[0] = mx.concatenate([r["conv0"], r["qkv"][b : b + 1, :n]], axis=1)[:, -keep:]
            if n == T:
                c[1] = r["stateT"][b : b + 1]
            else:
                _, c[1] = kernels.gated_delta_vec(r["q"][b : b + 1, :n], r["k"][b : b + 1, :n], r["v"][b : b + 1, :n],
                                                  r["g"][b : b + 1, :n], r["beta"][b : b + 1, :n], r["state0"])
        elif kind == "bcsa":
            c.update(r["k"][b : b + 1, :, :n], r["v"][b : b + 1, :, :n], r["ik"][b : b + 1, :n])
        elif kind == "kv":
            c.update_and_fetch(r["k"][b : b + 1, :, :n], r["v"][b : b + 1, :, :n])


def commit(cache, rec, n):
    """Keep the first n of the T verified tokens in the caches."""
    T = rec["T"]
    st = cache[-1]
    st.prev = rec["full"][:, : st.keep + n][:, -st.keep :]
    for c, r in zip(cache, rec["layers"]):
        kind = r["kind"]
        if kind == "kda":
            if n == T:
                continue
            c[0] = mx.concatenate([r["conv0"], r["qkv"][:, :n]], axis=1)[:, -(r["conv0"].shape[1]) :]
            _, c[1] = kernels.gated_delta_vec(r["q"][:, :n], r["k"][:, :n], r["v"][:, :n], r["g"][:, :n],
                                              r["beta"][:, :n], r["state0"])
        elif kind == "bcsa":
            c.update(r["k"][:, :, :n], r["v"][:, :, :n], r["ik"][:, :n])
        elif kind == "kv":
            if n < T:
                c.trim(T - n)
