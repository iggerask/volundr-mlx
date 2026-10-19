"""DFlash2 block drafter for Volundr (Blockway/Agens-Volundr-32B-v1-DFlash2) in MLX.

Semantics follow the engine (ref/agens-engine kernels/m1/m1_draft.cu; spec.rs loads it):
- Context: the target's residual stream 0 at the entry of layers target_layer_ids + 1 (5 x 5120 per token) ->
  ctx = RMSNorm(fc(concat), hidden_norm); every layer's context K/V = rope(k_norm(k_proj(ctx))), v_proj(ctx), kept in
  a ring of RING positions (`materialize`).
- Draft: one parallel pass over the block [anchor, mask x 7] at positions P..P+7. Five layers, each
  r += conv1(o_proj(attn(conv0(x)))), r += conv1(down(swiglu(gate_up(conv0(x))))) with grouped 2-tap convs whose
  per-group taps a projection of the normed input adjusts; attention is non-causal inside the block, plus a 2048
  sliding window of context. Rows 1..7 -> the target lm_head -> top-16 per row -> a greedy walk that adds a low-rank
  bigram score A[prev] * hp * B[cand] (`propose`).
Its numerics only move acceptance (the target verifies every token), so the drafter may be quantized.
"""
import glob
import os
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

F32, BF16 = mx.float32, mx.bfloat16
HID, NL, QH, KVH, HD, INTER, GROUPS, GSZ = 5120, 5, 32, 8, 128, 17408, 320, 16
RANK, TOPK, WINDOW, RING, BLOCK = 256, 16, 2048, 4096, 8
EPS, ROPE_BASE = 1e-6, 1e7
REPO = "models--Blockway--Agens-Volundr-32B-v1-DFlash2"


def find_checkpoint():
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{REPO}/snapshots/*/model.safetensors"))
    if not hits:
        raise FileNotFoundError(f"no {REPO} snapshot in the HF cache")
    return Path(hits[0]).parent


def grouped_conv(x, delta, base, side):
    """y[t] = c0[t] * x[t] + c1[t] * x[t-1] (none for row 0), c_tap[t][c] = base[side][tap][c] + delta[t][side][tap][c/16]
    in fp32, rounded once. x [T, HID] (one block, T = 8), delta [T, 4 * GROUPS], base [2, 2, HID]."""
    T = x.shape[0]
    d = mx.repeat(delta.reshape(T, 2, 2, GROUPS)[:, side].astype(F32), GSZ, axis=-1)  # [T, 2, HID]
    c = base[side].astype(F32)[None] + d
    xf = x.astype(F32)
    prev = mx.concatenate([mx.zeros((1, x.shape[1]), F32), xf[:-1]], axis=0)
    return (c[:, 0] * xf + c[:, 1] * prev).astype(x.dtype)


_conv = [mx.compile(partial(grouped_conv, side=s)) for s in (0, 1)]


@mx.compile
def _walk(uval, cid, hp, A, B, pred):
    """Greedy walk over the 7 slots: score[c] = logit[e][c] + sum_r A[pred][r] hp[e][r] B[cand_c][r] (fp32), first
    max; pred starts as the anchor and becomes each chosen token."""
    out = []
    for e in range(BLOCK - 1):
        a = mx.take(A, pred, axis=0).astype(F32) * hp[e]
        s = uval[e] + mx.take(B, cid[e], axis=0).astype(F32) @ a
        pred = mx.take(cid[e], mx.argmax(s))
        out.append(pred)
    return mx.stack(out)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_kernel = mx.zeros((2, 2, HID), BF16)
        self.kernel_projection = nn.Linear(HID, 4 * GROUPS, bias=False)


class SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HID, QH * HD, bias=False)
        self.k_proj = nn.Linear(HID, KVH * HD, bias=False)
        self.v_proj = nn.Linear(HID, KVH * HD, bias=False)
        self.o_proj = nn.Linear(QH * HD, HID, bias=False)
        self.q_norm = nn.RMSNorm(HD, EPS)
        self.k_norm = nn.RMSNorm(HD, EPS)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HID, INTER, bias=False)
        self.up_proj = nn.Linear(HID, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HID, bias=False)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(HID, EPS)
        self.post_attention_layernorm = nn.RMSNorm(HID, EPS)
        self.attention_conv = Conv()
        self.mlp_conv = Conv()
        self.self_attn = SelfAttn()
        self.mlp = MLP()


class Selector(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.hidden_projection = nn.Linear(HID, RANK, bias=False)
        self.predecessor_codebook = mx.zeros((vocab, RANK), BF16)
        self.successor_codebook = mx.zeros((vocab, RANK), BF16)


class DraftCache:
    """Context K/V per layer in a ring: slot = position % RING, [1, KVH, RING, HD] bf16."""

    def __init__(self):
        self.k = [mx.zeros((1, KVH, RING, HD), BF16) for _ in range(NL)]
        self.v = [mx.zeros((1, KVH, RING, HD), BF16) for _ in range(NL)]

    def store(self, l, k, v, p0):
        n = k.shape[2]
        s = p0 % RING
        n1 = min(n, RING - s)
        self.k[l][..., s : s + n1, :] = k[..., :n1, :]
        self.v[l][..., s : s + n1, :] = v[..., :n1, :]
        if n1 < n:
            self.k[l][..., : n - n1, :] = k[..., n1:, :]
            self.v[l][..., : n - n1, :] = v[..., n1:, :]

    def window(self, l, lo, hi):
        """K/V of positions [lo, hi) in order."""
        n, s = hi - lo, lo % RING
        if s + n <= RING:
            return self.k[l][..., s : s + n, :], self.v[l][..., s : s + n, :]
        idx = (mx.arange(lo, hi) % RING).astype(mx.int32)
        return mx.take(self.k[l], idx, axis=2), mx.take(self.v[l], idx, axis=2)

    @property
    def state(self):
        return self.k + self.v


class DFlash2(nn.Module):
    def __init__(self, vocab=248320, mask_id=248070, layer_ids=(8, 21, 35, 53, 67)):
        super().__init__()
        self.mask_id = mask_id
        self.capture_layers = [i + 1 for i in layer_ids]  # sglang records the residual at layer entry
        self.fc = nn.Linear(5 * HID, HID, bias=False)
        self.hidden_norm = nn.RMSNorm(HID, EPS)
        self.norm = nn.RMSNorm(HID, EPS)
        self.layers = [Layer() for _ in range(NL)]
        self.candidate_selector = Selector(vocab)

    @classmethod
    def load(cls, path=None, bits=0, group_size=128, lm_head_bits=0):
        path = Path(path) if path else find_checkpoint()
        import json

        cfg = json.loads((path / "config.json").read_text())
        dc = cfg["dflash_config"]
        assert dc["block_size"] == BLOCK and len(dc["target_layer_ids"]) == 5
        m = cls(cfg["vocab_size"], dc["mask_token_id"], dc["target_layer_ids"])
        m.load_weights(list(mx.load(str(path / "model.safetensors")).items()), strict=True)
        if bits:  # the linears; norms, conv bases and the selector codebooks (gathered, never streamed) stay bf16
            nn.quantize(m, group_size=group_size, bits=bits, class_predicate=lambda p, mod: isinstance(mod, nn.Linear))
        m.lm_head_bits = lm_head_bits
        m.eval()
        mx.eval(m.parameters())
        return m

    def attach(self, target, vocab_valid):
        """Share the target's embedding and lm_head (an optional quantized copy of the head for drafting only)."""
        self.embed = target.model.embed_tokens
        self.head = target.lm_head
        if self.lm_head_bits:
            w = target.lm_head.weight
            q = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=128, bits=self.lm_head_bits)
            q.weight, q.scales, q.biases = mx.quantize(w, group_size=128, bits=self.lm_head_bits)
            mx.eval(q.parameters())
            self.head = q
        self.vocab_valid = vocab_valid

    def make_cache(self):
        return DraftCache()

    def materialize(self, cache, cap, p0):
        """Context K/V of target features cap [n, 5 * HID] at positions p0..p0+n-1 (only the last WINDOW matter)."""
        n = cap.shape[0]
        if n > WINDOW:
            cap, p0, n = cap[n - WINDOW :], p0 + n - WINDOW, WINDOW
        ctx = self.hidden_norm(self.fc(cap))
        for l, L in enumerate(self.layers):
            a = L.self_attn
            k = a.k_norm(a.k_proj(ctx).reshape(n, KVH, HD)).transpose(1, 0, 2)[None]
            k = mx.fast.rope(k, HD, traditional=False, base=ROPE_BASE, scale=1.0, offset=p0)
            v = a.v_proj(ctx).reshape(n, KVH, HD).transpose(1, 0, 2)[None]
            cache.store(l, k, v, p0)

    def hidden(self, cache, anchor, P):
        """Final normed hidden [BLOCK, HID] of the block [anchor, mask x 7] at positions P..P+7."""
        ids = mx.concatenate([anchor.reshape(1).astype(mx.int32), mx.full((BLOCK - 1,), self.mask_id, mx.int32)])
        h = self.embed(ids)
        lo = max(0, P + BLOCK - 1 - WINDOW + 1)  # oldest key any block query sees
        mask = None
        if P - WINDOW + 1 > 0:  # query j sees keys >= P + j - WINDOW + 1 (and the whole block)
            kp, qp = mx.arange(lo, P + BLOCK), mx.arange(P, P + BLOCK)
            mask = kp[None] >= qp[:, None] - WINDOW + 1
        r = None
        for l, L in enumerate(self.layers):
            r = h if l == 0 else r + h
            x = L.input_layernorm(r)
            da = L.attention_conv.kernel_projection(x)
            t = _conv[0](x, da, L.attention_conv.base_kernel)
            a = L.self_attn
            q = a.q_norm(a.q_proj(t).reshape(BLOCK, QH, HD)).transpose(1, 0, 2)[None]
            k = a.k_norm(a.k_proj(t).reshape(BLOCK, KVH, HD)).transpose(1, 0, 2)[None]
            v = a.v_proj(t).reshape(BLOCK, KVH, HD).transpose(1, 0, 2)[None]
            q = mx.fast.rope(q, HD, traditional=False, base=ROPE_BASE, scale=1.0, offset=P)
            k = mx.fast.rope(k, HD, traditional=False, base=ROPE_BASE, scale=1.0, offset=P)
            cache.store(l, k, v, P)
            keys, vals = cache.window(l, lo, P + BLOCK)
            o = mx.fast.scaled_dot_product_attention(q, keys, vals, scale=HD**-0.5, mask=mask)
            o = o[0].transpose(1, 0, 2).reshape(BLOCK, QH * HD)
            h = _conv[1](a.o_proj(o), da, L.attention_conv.base_kernel)
            r = r + h
            x = L.post_attention_layernorm(r)
            dm = L.mlp_conv.kernel_projection(x)
            t = _conv[0](x, dm, L.mlp_conv.base_kernel)
            m = L.mlp
            h = _conv[1](m.down_proj(nn.silu(m.gate_proj(t)) * m.up_proj(t)), dm, L.mlp_conv.base_kernel)
        return self.norm(r + h)

    def propose(self, cache, anchor, P):
        """The 7 drafts (int32 [7], lazy) after anchor (int32 scalar, lazy) at position P."""
        hs = self.hidden(cache, anchor, P)[1:]
        logits = self.head(hs)[:, : self.vocab_valid]
        idx = mx.argpartition(-logits, TOPK - 1, axis=-1)[:, :TOPK]
        val = mx.take_along_axis(logits, idx, axis=-1)
        order = mx.argsort(-val.astype(F32), axis=-1)  # value descending: argmax below takes the first max
        cid = mx.take_along_axis(idx, order, axis=-1).astype(mx.int32)
        uval = mx.take_along_axis(val, order, axis=-1).astype(F32)
        sel = self.candidate_selector
        hp = sel.hidden_projection(hs).astype(F32)  # [7, RANK]
        return _walk(uval, cid, hp, sel.predecessor_codebook, sel.successor_codebook,
                     anchor.reshape(()).astype(mx.int32))
