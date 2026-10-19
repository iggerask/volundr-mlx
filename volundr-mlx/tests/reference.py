"""NumPy fp64 reference of the Volundr text model, written from the release's torch code (modeling_volundr.py,
bcsa.py, volundr_extras.py) and agens-core reference.rs, independently of volundr.py. Whole-sequence forward with
dense masks (bcsa.py's O(T^2) reference), naive per-token KDA recurrence (kda_decode.cu's math). Slow; tiny configs.
"""
import numpy as np

MASK61 = (1 << 61) - 1
PRIMES = (1000003, 1000033, 1000037, 1000039, 1000081)


# ---------------------------------------------------------------- reference.rs
def engram_hash_last(ids, n, rows):
    """reference.rs engram::hash_last: u64 wrapping arithmetic, & MASK61, % rows."""
    code = 0
    for j in range(n):
        pos = len(ids) - n + j
        tok = -1 if pos < 0 else int(ids[pos])
        code = ((code * PRIMES[j % len(PRIMES)] + (tok + 1)) % (1 << 64)) & MASK61
    return code % rows


def engram_hash_seq(ids, n, rows):
    return [engram_hash_last(ids[:t], n, rows) for t in range(1, len(ids) + 1)]


def w4_pack_row(q):
    """reference.rs w4::pack_row: values in [-8, 7] -> i32 words, nibble = q + 8, lowest first."""
    q = np.asarray(q)
    assert len(q) % 8 == 0 and q.min() >= -8 and q.max() <= 7
    out = []
    for c in q.reshape(-1, 8):
        w = 0
        for j, v in enumerate(c):
            w |= ((int(v) + 8) & 0xF) << (4 * j)
        out.append(np.uint32(w).view(np.int32))
    return np.array(out, dtype=np.int32)


# ---------------------------------------------------------------- model pieces
def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.logaddexp(x, 0)


def rms(x, w, eps, offset):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * ((1 + w) if offset else w)


def sinkhorn(logits, iters):
    m = np.exp(logits - logits.max(axis=(-1, -2), keepdims=True))
    for _ in range(iters):
        m = m / np.maximum(m.sum(-1, keepdims=True), 1e-9)
        m = m / np.maximum(m.sum(-2, keepdims=True), 1e-9)
    return m


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def rope_neox(x, pos, rd, theta):
    """x [T, heads, D]; Qwen3.5 rotate_half RoPE on the first rd dims."""
    inv = theta ** (-np.arange(0, rd, 2, dtype=np.float64) / rd)
    ang = pos[:, None] * inv[None]
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x1, x2 = x[..., : rd // 2], x[..., rd // 2 : rd]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., rd:]], axis=-1)


class Ref:
    def __init__(self, params: dict, cfg: dict):
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
        self.c = cfg

    def lin(self, x, name):
        y = x @ self.p[name + ".weight"].T
        if name + ".bias" in self.p:
            y = y + self.p[name + ".bias"]
        return y

    # ------------------------------------------------------------ mixers
    def kda(self, pre, h):
        c, p = self.c, self.p
        T = h.shape[0]
        Hk, HV, K, V = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = self.lin(h, pre + "in_proj_qkv")
        z = self.lin(h, pre + "in_proj_z").reshape(T, HV, V)
        b = self.lin(h, pre + "in_proj_b")
        a = self.lin(self.lin(h, pre + "f_down"), pre + "f_up").reshape(T, HV, K)
        w = p[pre + "conv1d.weight"][:, :, 0]  # [C, Kc]: tap j multiplies x[t - (Kc-1) + j]
        Kc = w.shape[1]
        xp = np.concatenate([np.zeros((Kc - 1, qkv.shape[1])), qkv], 0)
        conv = silu(sum(xp[j : j + T] * w[:, j] for j in range(Kc)))
        q, k, v = conv[:, : Hk * K], conv[:, Hk * K : 2 * Hk * K], conv[:, 2 * Hk * K :]
        q, k, v = q.reshape(T, Hk, K), k.reshape(T, Hk, K), v.reshape(T, HV, V)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * K**-0.5
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        g = -np.exp(p[pre + "A_log"])[:, None] * softplus(a + p[pre + "dt_bias"].reshape(HV, K))
        beta = sigmoid(b)
        S = np.zeros((HV, V, K))
        o = np.zeros((T, HV, V))
        for t in range(T):
            for hv in range(HV):
                kh = hv // (HV // Hk)
                S[hv] *= np.exp(g[t, hv])[None, :]
                vnew = (v[t, hv] - S[hv] @ k[t, kh]) * beta[t, hv]
                S[hv] += np.outer(vnew, k[t, kh])
                o[t, hv] = S[hv] @ q[t, kh]
        o = rms(o, p[pre + "norm.weight"], c["rms_norm_eps"], False) * silu(z)
        return self.lin(o.reshape(T, HV * V), pre + "out_proj")

    def attention(self, pre, h, bcsa):
        c, p = self.c, self.p
        T = h.shape[0]
        nH, nKV, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        G, eps = nH // nKV, c["rms_norm_eps"]
        qg = self.lin(h, pre + "q_proj").reshape(T, nH, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(T, nH * D)
        q = rms(q, p[pre + "q_norm.weight"], eps, True)
        k = rms(self.lin(h, pre + "k_proj").reshape(T, nKV, D), p[pre + "k_norm.weight"], eps, True)
        v = self.lin(h, pre + "v_proj").reshape(T, nKV, D)
        pos = np.arange(T, dtype=np.float64)
        rd = int(round(D * c["partial_rotary_factor"]))
        q, k = rope_neox(q, pos, rd, c["rope_theta"]), rope_neox(k, pos, rd, c["rope_theta"])
        kf, vf = np.repeat(k, G, axis=1), np.repeat(v, G, axis=1)  # [T, nH, D]
        scale = D**-0.5
        s_loc = np.einsum("thd,shd->hts", q, kf) * scale
        t_, s_ = np.arange(T)[:, None], np.arange(T)[None, :]
        ok = s_ <= t_
        if bcsa:
            W, C, topk = c["bcsa_window"], c["bcsa_compress"], c["bcsa_topk"]
            ok &= s_ > t_ - W
        s_loc = np.where(ok[None], s_loc, -np.inf)
        scores, vals = s_loc, np.broadcast_to(vf.transpose(1, 0, 2), (nH, T, D))
        if bcsa:
            nb = T // C
            n_ = np.arange(nb)[None, :]
            far_valid = ((n_ + 1) * C - 1) <= (t_ - W)
            if nb > 0 and far_valid.any():
                kb = kf[: nb * C].reshape(nb, C, nH, D).mean(1)
                vb = vf[: nb * C].reshape(nb, C, nH, D).mean(1)
                s_far = np.einsum("thd,nhd->htn", q, kb) * scale
                IH, ID = c["bcsa_index_heads"], c["bcsa_index_dim"]
                qi = rms(self.lin(h, pre + "idx_q").reshape(T, IH, ID), p[pre + "idx_norm.weight"], eps, True)
                ki = rms(self.lin(h, pre + "idx_k"), p[pre + "idx_norm.weight"], eps, True)
                kbi = ki[: nb * C].reshape(nb, C, ID).mean(1)
                idx = np.einsum("thd,nd->thn", qi, kbi).mean(1) * ID**-0.5
                idx = np.where(far_valid, idx, -np.inf)
                ksel = min(topk, nb)
                top = np.argsort(-idx, axis=-1, kind="stable")[:, :ksel]
                sel = np.zeros_like(far_valid)
                np.put_along_axis(sel, top, True, axis=-1)
                sel &= far_valid
                s_far = np.where(sel[None], s_far, -np.inf)
                scores = np.concatenate([s_loc, s_far], -1)
                vals = np.concatenate([vf.transpose(1, 0, 2), vb.transpose(1, 0, 2)], 1)
        out = np.einsum("hts,hsd->thd", softmax(scores), vals).reshape(T, nH * D)
        return self.lin(out * sigmoid(gate), pre + "o_proj")

    def engram(self, pre, y, ids):
        c, p = self.c, self.p
        rows = [engram_hash_seq(ids, n, c["engram_rows"]) for n in c["engram_orders"]]
        e = np.concatenate([p[f"{pre}tables.{i}.weight"][r] for i, r in enumerate(rows)], -1)
        e = rms(e, p[pre + "norm.weight"], c["rms_norm_eps"], False)
        return sigmoid(self.lin(y, pre + "gate")) * self.lin(e, pre + "value")

    # ------------------------------------------------------------ model
    def forward(self, ids):
        c, p = self.c, self.p
        ids = list(ids)
        eps, n = c["rms_norm_eps"], c["mhc_streams"]
        x = p["model.embed_tokens.weight"][ids]
        H = np.repeat(x[:, None], n, axis=1)
        for i, mixer in enumerate(c["mixer_types"]):
            m = f"model.mhc.{i}."
            d = self.lin(rms(H.mean(1), p[m + "norm.weight"], eps, False), m + "dyn")
            pre_w = p[m + "pre_static"] + d[:, :n]
            post_w = p[m + "post_static"] + d[:, n : 2 * n]
            res = sinkhorn(p[m + "res_static"] + d[:, 2 * n :].reshape(-1, n, n), c["mhc_sinkhorn_iters"])
            x_in = np.einsum("tn,tnd->td", pre_w, H)
            L = f"model.layers.{i}."
            h = rms(x_in, p[L + "input_layernorm.weight"], eps, True)
            if mixer == "kda":
                r = self.kda(L + "linear_attn.", h)
            else:
                r = self.attention(L + "self_attn.", h, mixer == "bcsa")
            h2 = x_in + r
            g_, u_ = self.lin(rms(h2, p[L + "post_attention_layernorm.weight"], eps, True), L + "mlp.gate_proj"), \
                self.lin(rms(h2, p[L + "post_attention_layernorm.weight"], eps, True), L + "mlp.up_proj")
            y = h2 + self.lin(silu(g_) * u_, L + "mlp.down_proj")
            if i in c["engram_layers"]:
                y = y + self.engram(L + "engram.", y, ids)
            H = np.einsum("tij,tjd->tid", res, H) + post_w[..., None] * (y - x_in)[:, None]
        return self.lin(rms(H[:, 0], p["model.norm.weight"], eps, True), "lm_head")
