"""Batched speculative decoding: every row of a decode batch (batch.BatchCache) verifies its own draft of T tokens at
its own position in one forward of B x T rows, and keeps its own number of them (NOTES_BATCHED_SERVE.md).

- Verify (`verify_forward`): spec/verify.py's B = 1 verify per layer, with B rows:
  - KDA: verify._kda as it is (the conv / recurrence kernels are per row); the record keeps what a commit replays;
  - BCSA: bcsa_verify.attend (the 8q kernel) once per row against that row's ring, pooled blocks and top-k selection
    (the B = 1 kernels with the row's position), nothing written;
  - dense: each row's T keys written at its positions, one SDPA with a per-row causal mask;
  - RoPE with the rows' offsets (an array offset: position offset_b + t).
- Commit (`commit`): row b keeps its first n_b tokens:
  - KDA: the conv state from the row's own n_b rows; the recurrence replayed from the pre-verify state with the
    steps past n_b neutral (decay 1, beta 0: the state passes through unchanged), one call for all rows;
  - BCSA: n_b ring rows, the blocks they close, the open block's indexer keys, row by row (BCSACache.update's
    arithmetic), all ring writes first so no write waits on a pending read of the ring;
  - dense: nothing (keys past a row's position are never read and are overwritten later);
  - Engram: the last ids per row.
NAX covers 16 verify rows, so the scheduler uses this for B = 2 with T = 8 (B x T > 16 would take the stock
matmuls, slower than plain batched decode).
"""
import mlx.core as mx

import bcsa_verify
import kernels
import volundr as V
from batch import BatchBCSACache, BatchCache, BatchKVCache, _row

F32, BF16 = V.F32, V.BF16


def _attn(attn, x, cache, rec):
    B, L, _ = x.shape
    D = attn.head_dim
    qg = attn.q_proj(x).reshape(B, L, attn.n_heads, 2 * D)
    q, gate = qg[..., :D], qg[..., D:].reshape(B, L, attn.n_heads * D)
    q = attn.q_norm(q).transpose(0, 2, 1, 3)
    k = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.n_kv, D)).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, L, attn.n_kv, D).transpose(0, 2, 1, 3)
    off = cache.rows.arr()
    q, k = attn._rope(q, off), attn._rope(k, off)
    if not attn.bcsa:
        o = _dense(cache, q, k, v, attn.scale)
        rec.update(kind="kv", T=L)
    else:
        o, ik = _bcsa(attn, x, q, k, v, cache)
        rec.update(kind="bcsa", T=L, k=k, v=v, ik=ik)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, attn.n_heads * D)
    return attn.o_proj(V._sigmoid_gate(o, gate))


def _dense(cache: BatchKVCache, q, k, v, scale):
    B, _, T, _ = q.shape
    pos = cache.rows.pos
    N = max(pos) + T
    cache.ensure(B, N)
    p2 = cache.rows.arr()[:, None] + mx.arange(T)[None]  # [B, T]
    r, h = mx.arange(B), mx.arange(cache.KVH)
    cache.keys[r[:, None, None], h[None, :, None], p2[:, None, :]] = k
    cache.values[r[:, None, None], h[None, :, None], p2[:, None, :]] = v
    if min(pos) == max(pos):  # (KVCache + SDPA "causal", as verify.py's dense layer)
        mask = "causal"
    else:  # query t of row b sees positions <= pos_b + t
        mask = (mx.arange(N)[None, None] <= p2[:, :, None])[:, None]
    return mx.fast.scaled_dot_product_attention(q, cache.keys[:B, :, :N], cache.values[:B, :, :N], scale=scale,
                                                mask=mask)


def _bcsa(attn, x, q, k, v, cache: BatchBCSACache):
    """verify._bcsa_verify's 8q branch per row: the row's ring read in place, its far blocks (the union of its
    queries' top-k), nothing written."""
    B, H, L, D = q.shape
    W, C = attn.W, attn.C
    ik = attn.idx_norm(attn.idx_k(x))
    qi = None
    outs = []
    for b in range(B):
        P = cache.rows.pos[b]
        nv_max = max(0, (P + L - 1 - W + 1) // C)
        far = ()
        if nv_max:
            if qi is None:
                qi = attn._index_q(x)
            if attn.ID == 128 and qi.dtype == BF16:
                s = kernels.bcsa_scores(_row(qi, b), _row(cache.bk, b), nv_max, attn.ID**-0.5)
            else:
                s = V._block_scores(qi[b : b + 1], cache.bk[b : b + 1, :nv_max], attn.ID**-0.5)[0]
            sel = kernels.bcsa_topk(s, P, W, C, attn.topk)
            far = (_row(cache.pk, b), _row(cache.pv, b), *kernels.far_union(sel, nv_max, 32))
        o = bcsa_verify.attend(_row(q, b), _row(k, b), _row(v, b), _row(cache.k, b), _row(cache.v, b), P, W,
                               attn.scale, *far)
        outs.append(o[None])
    return (outs[0] if B == 1 else mx.concatenate(outs, axis=0)), ik


def _layer(layer, x, cache, rows, rec):
    import verify as VF

    eps = layer.input_layernorm.eps
    h = kernels.rms_norm(x, layer.input_layernorm.weight, eps, offset=True)[1]
    r = VF._kda(layer.linear_attn, h, cache, rec) if layer.mixer == "kda" else _attn(layer.self_attn, h, cache, rec)
    h, hn = kernels.rms_norm(x, layer.post_attention_layernorm.weight, eps, offset=True, r=r)
    y = h + layer.mlp(hn)
    if "engram" in layer:
        y = layer.engram(y, rows)
    return y


def verify_forward(model, bc: BatchCache, ids, capture=()):
    """ids [B, T] int32: row b's [bonus, drafts] at its position. Returns (logits [B, T, vocab], features
    [B, T, 5 * hidden] of the capture layers or None, the commit record). The rows' positions do not move (commit
    does that); KDA and dense state hold all T tokens until the commit."""
    a, inner = model.args, model.model
    B, L = ids.shape
    x = inner.embed_tokens(ids)
    st = bc.layers[-1]
    full = mx.concatenate([st.prev, ids.astype(mx.int32)], axis=1)
    rows = [V.engram_rows(full, n, L, a.engram_rows) for n in a.engram_orders]
    recs = [{} for _ in inner.layers]
    H = mx.broadcast_to(x[:, :, None, :], (B, L, a.mhc_streams, a.hidden_size))
    hmean = x
    caps = []
    for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, bc.layers)):
        if i in capture:
            caps.append(H[:, :, 0])
        x_in, post, res = mhc.mix(H, hmean)
        y = _layer(layer, x_in, c, rows, recs[i])
        H, hmean = mhc.combine(H, post, res, y, x_in)
    logits = model.lm_head(inner.norm(H[:, :, 0]))
    return logits, (mx.concatenate(caps, axis=-1) if caps else None), {"layers": recs, "full": full, "T": L}


def _append_rows(c: BatchBCSACache, rec, ns, pos):
    """Row b appends its first ns[b] verified tokens (BCSACache.update for L = ns[b]): ring rows, the blocks they
    close (pooled from the ring after every row's writes), the open block's indexer keys."""
    W, C = c.W, c.C
    k, v, ik = rec["k"], rec["v"], rec["ik"]
    ensure_ring = max(p + n for p, n in zip(pos, ns))
    from batch import RING_SIZES

    R = next((r for r in RING_SIZES if r >= ensure_ring and r < W), W + c.F)
    c.ensure(len(ns), R, max((p + n) // C for p, n in zip(pos, ns)) + 1)
    for b, (P, n) in enumerate(zip(pos, ns)):  # ring rows: slot = position % W
        s = P % W
        n1 = min(n, W - s)
        c.k[b : b + 1, :, s : s + n1] = k[b : b + 1, :, :n1]
        c.v[b : b + 1, :, s : s + n1] = v[b : b + 1, :, :n1]
        if n1 < n:
            c.k[b : b + 1, :, : n - n1] = k[b : b + 1, :, n1:n]
            c.v[b : b + 1, :, : n - n1] = v[b : b + 1, :, n1:n]
    blocks, opens = [], []
    for b, (P, n) in enumerate(zip(pos, ns)):
        r0 = P % C
        nfull = (r0 + n) // C
        m = nfull * C
        ci = mx.concatenate([c.ik_open[b : b + 1, :r0], ik[b : b + 1, :n]], axis=1) if r0 else ik[b : b + 1, :n]
        if nfull:  # positions [P - r0, P - r0 + m) from the ring (after the writes)
            idx = (mx.arange(P - r0, P - r0 + m) % W).astype(mx.int32)
            ck = mx.take(c.k[b : b + 1], idx, axis=2)
            cv = mx.take(c.v[b : b + 1], idx, axis=2)
            blocks.append((b, P // C, V.BCSACache._pool(ck, C, 2), V.BCSACache._pool(cv, C, 2),
                           V.BCSACache._pool(ci[:, :m], C, 1)))
        rem = r0 + n - m
        if rem:
            opens.append((b, rem, ci[:, m:]))
    for b, nb, pk, pv, bk in blocks:
        m = pk.shape[2]
        c.pk[b : b + 1, :, nb : nb + m] = pk
        c.pv[b : b + 1, :, nb : nb + m] = pv
        c.bk[b : b + 1, nb : nb + m] = bk
    for b, rem, x in opens:
        c.ik_open[b : b + 1, :rem] = x


def commit(bc: BatchCache, rec, ns):
    """Row b keeps the first ns[b] (1..T) of its T verified tokens; the rows' positions move by ns."""
    T = rec["T"]
    B = len(ns)
    pos = list(bc.rows.pos)
    st = bc.layers[-1]
    keep = st.keep
    full = rec["full"]
    st.prev = mx.concatenate([full[b : b + 1, ns[b] : ns[b] + keep] for b in range(B)], axis=0)
    all_t = all(n == T for n in ns)
    nsa = mx.array(ns, dtype=mx.int32)
    for c, r in zip(bc.layers, rec["layers"]):
        kind = r.get("kind")
        if kind == "kda":
            if all_t:
                continue  # the verify already left the state after all T tokens
            kc = r["conv0"].shape[1]
            conv = mx.concatenate([r["conv0"], r["qkv"]], axis=1)  # [B, kc + T, C]
            c[0] = mx.concatenate([conv[b : b + 1, ns[b] : ns[b] + kc] for b in range(B)], axis=0)
            live = mx.arange(T)[None] < nsa[:, None]  # [B, T]: steps a row keeps
            g = mx.where(live[:, :, None, None], r["g"], 1.0)  # (decay 1, beta 0: the state passes unchanged)
            beta = mx.where(live[:, :, None], r["beta"], 0.0)
            _, c[1] = kernels.gated_delta_vec(r["q"], r["k"], r["v"], g, beta, r["state0"])
        elif kind == "bcsa":
            _append_rows(c, r, ns, pos)
    bc.rows.pos = [p + n for p, n in zip(pos, ns)]
    bc.rows.changed()
