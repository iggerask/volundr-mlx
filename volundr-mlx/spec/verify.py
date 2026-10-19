"""Target side of speculative decoding: a Volundr forward over T tokens that captures the drafter's context features
and, in verify mode, keeps what `commit(n)` needs to keep exactly the first n of the T tokens.

Mirrors volundr.py (VolundrModel / DecoderLayer / KDA / Attention, GPU kernel paths) except for the state handling:
- KDA: runs the T tokens; commit restores the pre-call conv / recurrent state and replays the recurrence on the first
  n rows' q, k, v, gate, beta (no projections re-run). n = T keeps the verify result as it is.
- BCSA: attends like a prefill chunk (window keys from the ring + the T new keys, top-k far blocks; no block a verify
  query can select is closed inside the chunk) without touching the cache; commit appends the first n rows.
- Dense attention: the KVCache takes all T rows; commit trims T - n.
- Engram: commit sets the previous-ids state from the committed ids. mHC has no state.
Not bit-exact against T sequential decode steps (different matmul / attention kernels round differently), so greedy
output matches plain decoding except where the top two logits are within rounding.
"""
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kernels  # noqa: E402
import volundr as V  # noqa: E402

F32, BF16 = V.F32, V.BF16


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
    """_bcsa_prefill without the cache update (see the module docstring); returns the attention output and the rows
    to append on commit."""
    B, H, L, D = q.shape
    P, W, C = cache.offset, self.W, self.C
    ik = self.idx_norm(self.idx_k(x))
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


def forward(model, ids, cache, capture=(), verify=False, all_logits=False):
    """ids [1, L] int32. Returns (logits [L or 1, vocab] (all rows when verify or all_logits), features [L, 5 * HID]
    of the capture layers (stream 0 at layer entry; None without capture), the commit record (verify only))."""
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
    for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
        if i in capture:
            caps.append(H[0, :, 0])
        x_in, post, res = mhc.mix(H, hmean)
        y = _layer(layer, x_in, c, rows, recs[i]) if verify else layer(x_in, c, rows)
        H, hmean = mhc.combine(H, post, res, y, x_in)
    h = inner.norm(H[:, :, 0])
    logits = model.lm_head(h if (verify or all_logits) else h[:, -1:])[0]
    cap = mx.concatenate(caps, axis=-1) if caps else None
    rec = {"layers": recs, "full": full, "T": L} if verify else None
    return logits, cap, rec


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
