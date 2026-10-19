"""Agens Volundr 32B (Blockway) for MLX, text model.

72 layers on a 4-stream mHC residual: 54 KDA (Kimi Delta Attention), 17 BCSA (sliding window + top-k mean-pooled far
blocks), 1 dense gated attention (last), Engram n-gram memory after layers 2 and 22. Semantics follow the CUDA engine
(agens-engine kernels/m1/m1_model.cu, kernels/src/{mhc,kda_decode,far_decode}.cu) and the release's torch reference
(modeling_volundr.py, bcsa.py, volundr_extras.py); bf16 rounding points are kept where the engine rounds.

Weights come from convert.py (names `model.*`, linears in MLX affine layout).
"""
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, KVCache, _BaseCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_update, normalize_qk

import kernels

F32, BF16 = mx.float32, mx.bfloat16


def use_kernels() -> bool:
    """Custom Metal kernels on the GPU; the ops reference paths elsewhere (CPU tests)."""
    return mx.metal.is_available() and mx.default_device() == mx.gpu


# Opt-in W4A8 prefill matmuls (w4a8/: int8 activations per (row, 128-group) on the int8 tensor-op path). Not exact:
# it raises the per-matmul error several-fold (w4a8/NOTES.md), so it is off unless VOLUNDR_W4A8=1. Decode steps
# keep the bf16 path either way, and so do the linears named in VOLUNDR_W4A8_SKIP (default k_proj, v_proj).
W4A8_PREFILL = os.environ.get("VOLUNDR_W4A8", "") == "1"


def enable_w4a8(model) -> int:
    """Route the 4-bit linears of `model` through w4a8/w4a8_hook.py for calls with >= 64 rows; returns the count."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / "w4a8"))
    import w4a8_hook

    return w4a8_hook.enable(model)


ENGRAM_PRIMES = (1000003, 1000033, 1000037, 1000039, 1000081)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "volundr"
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 72
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    vocab_size: int = 248320
    vocab_valid: int = 248077  # rows past the tokenizer (suppress_tokens) never win the argmax
    rms_norm_eps: float = 1e-6
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128
    kda_gate_rank: int = 128
    mixer_types: List[str] = field(default_factory=list)
    bcsa_window: int = 4096
    bcsa_compress: int = 4
    bcsa_topk: int = 512
    bcsa_index_heads: int = 4  # not in config.json: bcsa.py defaults (the engine checks them against shapes)
    bcsa_index_dim: int = 128
    engram_layers: List[int] = field(default_factory=list)
    engram_orders: List[int] = field(default_factory=lambda: [2, 3])
    engram_rows: int = 1 << 20
    engram_dim: int = 512
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 10
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    tie_word_embeddings: bool = False
    eos_token_id: Any = None
    bcsa_prefill_tile: int = 128  # queries per tile, far-regime prefill, ops path (memory, not numerics)
    bcsa_select_rows: int = 512  # queries per top-k scoring pass (bounds the [rows, heads, blocks] scores)

    def __post_init__(self):
        if not self.mixer_types:
            raise ValueError("mixer_types is required")
        assert len(self.mixer_types) == self.num_hidden_layers
        assert self.engram_rows & (self.engram_rows - 1) == 0, "Engram hashing keeps only the low bits: rows = 2^k"


# ----------------------------------------------------------------------------- norms / small ops
class OffsetRMSNorm(nn.Module):
    """Qwen3.5 (Gemma-style) RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w) in fp32, rounded once to x's dtype."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight.astype(F32), self.eps).astype(x.dtype)


class RMSNorm(nn.Module):
    """Plain RMSNorm (torch nn.RMSNorm: mHC and Engram norms), fp32 math, one rounding."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(F32), self.eps).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _silu_mul(g, u):
    return (nn.silu(g.astype(F32)) * u.astype(F32)).astype(g.dtype)


@partial(mx.compile, shapeless=True)
def _gated_norm(x, w, z, eps):
    # fla gated RMSNorm (swish gate): ((x * rstd) * w) * z * sigmoid(z), fp32, one rounding
    return (mx.fast.rms_norm(x, w.astype(F32), eps) * nn.silu(z.astype(F32))).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _sigmoid_gate(o, gate):
    return (o.astype(F32) * mx.sigmoid(gate.astype(F32))).astype(o.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(_silu_mul(self.gate_proj(x), self.up_proj(x)))


# ----------------------------------------------------------------------------- mHC
def _make_sinkhorn_kernel():
    if not mx.metal.is_available():
        return None
    # One thread per 4x4 matrix, the engine's order (mhc.cu): M = exp(l - max); ITERS x (rows, then columns), each a
    # division by max(sum, 1e-9); row sums pairwise (m0 + m2) + (m1 + m3), column sums serial.
    source = """
        uint n = thread_position_in_grid.x;
        if (n >= logits_shape[0]) {
            return;
        }
        float m[16];
        float mx = -INFINITY;
        for (int k = 0; k < 16; k++) {
            m[k] = logits[n * 16 + k];
            mx = metal::max(mx, m[k]);
        }
        for (int k = 0; k < 16; k++) {
            m[k] = metal::precise::exp(m[k] - mx);
        }
        for (int it = 0; it < ITERS; it++) {
            for (int r = 0; r < 4; r++) {
                float s = metal::max((m[4 * r] + m[4 * r + 2]) + (m[4 * r + 1] + m[4 * r + 3]), 1e-9f);
                for (int c = 0; c < 4; c++) {
                    m[4 * r + c] = metal::precise::divide(m[4 * r + c], s);
                }
            }
            for (int c = 0; c < 4; c++) {
                float s = metal::max(((m[c] + m[4 + c]) + m[8 + c]) + m[12 + c], 1e-9f);
                for (int r = 0; r < 4; r++) {
                    m[4 * r + c] = metal::precise::divide(m[4 * r + c], s);
                }
            }
        }
        for (int k = 0; k < 16; k++) {
            out[n * 16 + k] = m[k];
        }
    """
    return mx.fast.metal_kernel(name="volundr_sinkhorn4", input_names=["logits"], output_names=["out"], source=source)


_sinkhorn_kernel = _make_sinkhorn_kernel()


def sinkhorn_ops(logits: mx.array, iters: int) -> mx.array:
    """Reference ops path: logits [..., 4, 4] fp32 -> doubly stochastic [..., 4, 4] fp32 (volundr_extras.sinkhorn)."""
    m = mx.exp(logits - logits.max(axis=(-1, -2), keepdims=True))
    for _ in range(iters):
        m = m / mx.maximum(m.sum(-1, keepdims=True), 1e-9)
        m = m / mx.maximum(m.sum(-2, keepdims=True), 1e-9)
    return m


def sinkhorn(logits: mx.array, iters: int) -> mx.array:
    """logits [..., 16] fp32 -> [..., 4, 4] fp32."""
    shape = logits.shape[:-1]
    if _sinkhorn_kernel is None or mx.default_device() != mx.gpu:
        return sinkhorn_ops(logits.reshape(*shape, 4, 4), iters)
    flat = logits.reshape(-1, 16)
    n = flat.shape[0]
    (out,) = _sinkhorn_kernel(
        inputs=[flat],
        template=[("ITERS", iters)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[F32],
    )
    return out.reshape(*shape, 4, 4)


@mx.compile
def _mhc_params(d, pre_s, post_s, res_s):
    # d [B, L, 2n + n^2] bf16 (the dyn linear); statics bf16. pre/post rounded to bf16 as the engine does.
    df = d.astype(F32)
    pre = (pre_s.astype(F32) + df[..., :4]).astype(d.dtype)
    post = (post_s.astype(F32) + df[..., 4:8]).astype(d.dtype)
    logits = res_s.astype(F32).reshape(16) + df[..., 8:24]
    return pre, post, logits


@mx.compile
def _mhc_collapse(pre, H):
    # x_in = bf16(sum_s bf16(pre_s * H_s)), sum in stream order
    p = (pre.astype(F32)[..., None] * H.astype(F32)).astype(H.dtype).astype(F32)
    return (((p[:, :, 0] + p[:, :, 1]) + p[:, :, 2]) + p[:, :, 3]).astype(H.dtype)


@mx.compile
def _mhc_combine(mixed, post, y, x_in):
    # out_s = bf16(mixed_s + bf16(post_s * bf16(y - x_in))); mean = bf16(sum_s out_s * 0.25) (next layer's norm input)
    dt = mixed.dtype
    delta = (y.astype(F32) - x_in.astype(F32)).astype(dt).astype(F32)
    upd = (post.astype(F32)[..., None] * delta[:, :, None]).astype(dt).astype(F32)
    o = (mixed.astype(F32) + upd).astype(dt)
    of = o.astype(F32)
    mean = ((((of[:, :, 0] + of[:, :, 1]) + of[:, :, 2]) + of[:, :, 3]) * 0.25).astype(dt)
    return o, mean


class MHC(nn.Module):
    """Manifold hyper-connections (n = 4): x_in = pre . H; H' = res . H + post (x) (y - x_in)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        n = args.mhc_streams
        assert n == 4, "the fused mix/combine are written for 4 streams"
        self.iters = args.mhc_sinkhorn_iters
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.dyn = nn.Linear(args.hidden_size, 2 * n + n * n, bias=False)
        self.pre_static = mx.zeros((n,))
        self.post_static = mx.ones((n,))
        self.res_static = mx.zeros((n, n))

    def mix(self, H, hmean):
        B, L, n, D = H.shape
        if use_kernels() and H.dtype == BF16:
            _, hn = kernels.rms_norm(hmean, self.norm.weight, self.norm.eps, offset=False)
            pre, post, res = kernels.mhc_params(self.dyn(hn).reshape(B * L, -1), self.pre_static, self.post_static,
                                                self.res_static, self.iters)
            x_in = kernels.mhc_mix(H.reshape(B * L, n, D), pre).reshape(B, L, D)
            return x_in, post.reshape(B, L, n), res.reshape(B, L, n, n)
        d = self.dyn(self.norm(hmean))
        pre, post, logits = _mhc_params(d, self.pre_static, self.post_static, self.res_static)
        res = sinkhorn(logits, self.iters).astype(H.dtype)
        return _mhc_collapse(pre, H), post, res

    def combine(self, H, post, res, y, x_in):
        B, L, n, D = H.shape
        if use_kernels() and H.dtype == BF16:
            Hn, mean = kernels.mhc_combine(H.reshape(B * L, n, D), res.reshape(B * L, n, n), post.reshape(B * L, n),
                                           y.reshape(B * L, D), x_in.reshape(B * L, D))
            return Hn.reshape(B, L, n, D), mean.reshape(B, L, D)
        mixed = res @ H  # [B, L, 4, 4] @ [B, L, 4, D], bf16 out (fp32 accumulation), torch's bmm
        return _mhc_combine(mixed, post, y, x_in)


# ----------------------------------------------------------------------------- KDA
@mx.compile
def _conv_silu(x, w):
    # x [B, K-1+L, C] bf16 (cached context + new), w [C, K, 1] fp32 -> silu(depthwise causal conv) [B, L, C] bf16
    K = w.shape[1]
    L = x.shape[1] - K + 1
    xf = x.astype(F32)
    acc = xf[:, 0:L] * w[:, 0, 0]
    for j in range(1, K):
        acc = acc + xf[:, j : j + L] * w[:, j, 0]
    return nn.silu(acc).astype(x.dtype)


class KDA(nn.Module):
    """Kimi Delta Attention on the Qwen3.5 GatedDeltaNet layout: fused qkv projection + depthwise conv (k = 4) over
    2*H*K + HV*V channels, per-channel gate g = -exp(A_log) * softplus(f_up(f_down(h)) + dt_bias), beta = sigmoid(b),
    l2-normalised q/k (scale K^-0.5), fp32 state [HV, V, K], z-gated RMSNorm, out_proj."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.H, self.HV = args.linear_num_key_heads, args.linear_num_value_heads
        self.K, self.V = args.linear_key_head_dim, args.linear_value_head_dim
        self.key_dim, self.value_dim = self.H * self.K, self.HV * self.V
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = args.linear_conv_kernel_dim
        self.eps = args.rms_norm_eps
        h = args.hidden_size
        self.in_proj_qkv = nn.Linear(h, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(h, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(h, self.HV, bias=False)
        self.f_down = nn.Linear(h, args.kda_gate_rank, bias=False)
        self.f_up = nn.Linear(args.kda_gate_rank, self.HV * self.K, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, bias=False)
        self.A_log = mx.zeros((self.HV,), dtype=F32)
        self.dt_bias = mx.zeros((self.HV * self.K,), dtype=F32)
        self.norm = RMSNorm(self.V, args.rms_norm_eps)  # weight only; applied through _gated_norm
        self.out_proj = nn.Linear(self.value_dim, h, bias=False)

    def __call__(self, x, cache):
        B, L, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, L, self.HV, self.V)
        b = self.in_proj_b(x)
        a = self.f_up(self.f_down(x)).reshape(B, L, self.HV, self.K)

        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros((B, self.kernel - 1, self.conv_dim), dtype=qkv.dtype)
        fused = use_kernels() and qkv.dtype == BF16
        if fused:  # conv + SiLU + q/k l2 norm in one kernel; packed recurrence with the per-channel decay
            conv = kernels.kda_conv_qk4 if self.K == 128 else kernels.kda_conv_qk
            q, k, v = conv(qkv, conv_state, self.conv1d.weight[:, :, 0], self.key_dim, self.K)
            keep = self.kernel - 1
            cache[0] = qkv[:, -keep:] if L >= keep else mx.concatenate([conv_state, qkv], axis=1)[:, -keep:]
            q, k = q.reshape(B, L, self.H, self.K), k.reshape(B, L, self.H, self.K)
            v = v.reshape(B, L, self.HV, self.V)
        else:
            conv_in = mx.concatenate([conv_state, qkv], axis=1)
            cache[0] = conv_in[:, -(self.kernel - 1) :]
            y = _conv_silu(conv_in, self.conv1d.weight)
            q, k, v = mx.split(y, [self.key_dim, 2 * self.key_dim], axis=-1)
            q = q.reshape(B, L, self.H, self.K).astype(F32)
            k = k.reshape(B, L, self.H, self.K).astype(F32)
            v = v.reshape(B, L, self.HV, self.V)
            q, k = normalize_qk(q, k, inv_scale=self.K**-0.5, eps=1e-6)
        A_log, dt_bias = self.A_log.reshape(self.HV, 1), self.dt_bias.reshape(self.HV, self.K)
        if fused and self.K == 128:
            state = cache[1] if cache[1] is not None else mx.zeros((B, self.HV, self.V, self.K), dtype=F32)
            rec = kernels.gated_delta_tg if L >= 64 else kernels.gated_delta_vec  # same arithmetic, bit for bit
            out, state = rec(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b.astype(F32)), state)
        else:
            out, state = gated_delta_update(q, k, v, a, b.astype(F32), A_log, dt_bias, cache[1])
        cache[1] = state
        if fused and self.V == 128:
            out = kernels.gated_norm(out.astype(x.dtype), self.norm.weight, z, self.eps)
        else:
            out = _gated_norm(out.astype(x.dtype), self.norm.weight, z, self.eps)
        return self.out_proj(out.reshape(B, L, self.value_dim))


# ----------------------------------------------------------------------------- BCSA cache
class BCSACache(_BaseCache):
    """BCSA layer state: a ring of the last W keys/values (post-RoPE), the mean-pooled K/V of every closed block of
    C positions, the pooled indexer keys of those blocks, and the indexer keys of the open block. The K/V buffers
    hold `far` extra rows after the ring, where decode gathers its selected far blocks, so one SDPA reads window and
    far keys without concatenating them."""

    block_step = 1024

    def __init__(self, window: int, compress: int, far: int = 0):
        self.W, self.C, self.F = window, compress, far
        assert window % compress == 0
        self.offset = 0
        self.k = self.v = None  # [B, KVH, W + far, D]
        self.pk = self.pv = None  # [B, KVH, cap, D]
        self.bk = None  # [B, cap, Di]
        self.nb = 0
        self.ik_open = None  # [B, C, Di]

    def window(self, n: int, end: Optional[int] = None):
        """K/V rows of positions [end - n, end) in temporal order (end defaults to the offset)."""
        end = self.offset if end is None else end
        s = (end - n) % self.W
        if s + n <= self.W:
            return self.k[..., s : s + n, :], self.v[..., s : s + n, :]
        idx = (mx.arange(end - n, end) % self.W).astype(mx.int32)
        return mx.take(self.k, idx, axis=2), mx.take(self.v, idx, axis=2)

    @staticmethod
    def _pool(x, C, axis):
        # mean of C bf16 rows in fp32, rounded once (the engine's block means)
        shp = list(x.shape)
        n = shp[axis] // C
        shp[axis : axis + 1] = [n, C]
        return (x.astype(F32).reshape(shp).sum(axis=axis + 1) * (1.0 / C)).astype(x.dtype)

    def _append_blocks(self, pk, pv, bk):
        m = pk.shape[2]
        if self.pk is None or self.nb + m > self.pk.shape[2]:
            B, KVH, _, D = pk.shape
            cap = ((self.nb + m + self.block_step - 1) // self.block_step) * self.block_step
            npk = mx.zeros((B, KVH, cap, D), pk.dtype)
            npv = mx.zeros((B, KVH, cap, D), pv.dtype)
            nbk = mx.zeros((B, cap, bk.shape[-1]), bk.dtype)
            if self.pk is not None and self.nb:
                npk[..., : self.nb, :] = self.pk[..., : self.nb, :]
                npv[..., : self.nb, :] = self.pv[..., : self.nb, :]
                nbk[:, : self.nb] = self.bk[:, : self.nb]
            self.pk, self.pv, self.bk = npk, npv, nbk
        self.pk[..., self.nb : self.nb + m, :] = pk
        self.pv[..., self.nb : self.nb + m, :] = pv
        self.bk[:, self.nb : self.nb + m] = bk
        self.nb += m

    def update(self, k, v, ik):
        """Append L positions: k, v [B, KVH, L, D], ik (normed indexer keys) [B, L, Di]."""
        B, KVH, L, D = k.shape
        P, W, C = self.offset, self.W, self.C
        if self.k is None:
            self.k = mx.zeros((B, KVH, W + self.F, D), k.dtype)
            self.v = mx.zeros((B, KVH, W + self.F, D), v.dtype)
            self.ik_open = mx.zeros((B, C, ik.shape[-1]), ik.dtype)
        r0 = P % C  # rows of the open block seen before
        nfull = (r0 + L) // C
        n = nfull * C
        ci = mx.concatenate([self.ik_open[:, :r0], ik], axis=1) if r0 else ik
        # Closed blocks pool from the ring after the write when it still holds all their rows (decode, short chunks):
        # nothing then refers to the pre-write ring, so the slice update reuses its buffer instead of copying it.
        from_ring = r0 + L <= W
        if nfull and not from_ring:
            if r0:
                tk, tv = self.window(r0)
                ck, cv = mx.concatenate([tk, k], axis=2), mx.concatenate([tv, v], axis=2)
            else:
                ck, cv = k, v
            self._append_blocks(self._pool(ck[..., :n, :], C, 2), self._pool(cv[..., :n, :], C, 2),
                                self._pool(ci[:, :n], C, 1))
        # ring: slot = position % W
        if L >= W:
            self.k[..., :W, :] = mx.roll(k[..., L - W :, :], (P + L - W) % W, axis=2)
            self.v[..., :W, :] = mx.roll(v[..., L - W :, :], (P + L - W) % W, axis=2)
        else:
            s = P % W
            n1 = min(L, W - s)
            self.k[..., s : s + n1, :] = k[..., :n1, :]
            self.v[..., s : s + n1, :] = v[..., :n1, :]
            if n1 < L:
                self.k[..., : L - n1, :] = k[..., n1:, :]
                self.v[..., : L - n1, :] = v[..., n1:, :]
        if nfull and from_ring:
            ck, cv = self.window(n, end=P - r0 + n)
            self._append_blocks(self._pool(ck, C, 2), self._pool(cv, C, 2), self._pool(ci[:, :n], C, 1))
        rem = r0 + L - n
        if rem:
            self.ik_open[:, :rem] = ci[:, n:]
        self.offset += L

    @property
    def state(self):
        return [a for a in (self.k, self.v, self.pk, self.pv, self.bk, self.ik_open) if a is not None]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.state)

    def empty(self):
        return self.k is None

    def is_trimmable(self):
        return False


class EngramState(_BaseCache):
    """The last (max order - 1) token ids of the sequence (-1 = none yet)."""

    def __init__(self, keep: int):
        self.keep = keep
        self.prev = None

    @property
    def state(self):
        return [] if self.prev is None else [self.prev]

    def empty(self):
        return self.prev is None

    def is_trimmable(self):
        return False


def engram_rows(full_ids: mx.array, n: int, L: int, rows: int) -> mx.array:
    """Row of the n-gram ending at each of the last L positions of full_ids [B, keep + L] (int32, -1 = pad).
    Torch hashes in wrapping int64 masked to 61 bits, then % rows; with rows = 2^k only the low k bits survive, and
    those depend only on the low bits of every operand, so uint32 wrapping arithmetic gives the same rows."""
    ext = (full_ids + 1).astype(mx.uint32)
    off = full_ids.shape[1] - L
    code = mx.zeros((full_ids.shape[0], L), dtype=mx.uint32)
    for j in range(n):
        s = off - (n - 1) + j
        code = code * ENGRAM_PRIMES[j % len(ENGRAM_PRIMES)] + ext[:, s : s + L]
    return code & (rows - 1)


class Engram(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        cat = args.engram_dim * len(args.engram_orders)
        self.tables = [nn.Embedding(args.engram_rows, args.engram_dim) for _ in args.engram_orders]
        self.norm = RMSNorm(cat, args.rms_norm_eps)
        self.gate = nn.Linear(args.hidden_size, 1, bias=True)
        self.value = nn.Linear(cat, args.hidden_size, bias=False)

    def __call__(self, y, rows):
        e = self.norm(mx.concatenate([t(r) for t, r in zip(self.tables, rows)], axis=-1))
        g = mx.sigmoid(self.gate(y).astype(F32)).astype(y.dtype)
        return y + g * self.value(e)


# ----------------------------------------------------------------------------- attention
def _block_scores(qi, bk, scale):
    """Indexer relevance of blocks, bf16 as torch computes it: einsum (bf16 out) -> mean over heads -> * scale.
    qi [B, L, IH, Di], bk [B, NB, Di] -> [B, L, NB]."""
    e = qi @ bk[:, None].swapaxes(-1, -2)
    s = e.astype(F32).mean(axis=2).astype(qi.dtype)
    return (s.astype(F32) * scale).astype(qi.dtype)


def _topk_blocks(s, k):
    """Indices of the k largest bf16 scores along the last axis, ties to the lower index (torch topk / the engine's
    radix select order). -inf scores lose to every finite one."""
    nb = s.shape[-1]
    if nb > 65536 or s.dtype != BF16:
        return mx.argpartition(-s.astype(F32), k - 1, axis=-1)[..., :k].astype(mx.int32)
    u = s.view(mx.uint16).astype(mx.uint32)
    key = mx.where(u >= 0x8000, 0xFFFF - u, u | 0x8000)  # monotone in the float value
    key = (key << 16) | (0xFFFF - mx.arange(nb, dtype=mx.uint32))
    return mx.argpartition(0xFFFFFFFF - key, k - 1, axis=-1)[..., :k].astype(mx.int32)


def _gather_blocks(p, idx):
    """p [B, KVH, cap, D], idx [B, ...] int32 -> [B, KVH, ..., D]."""
    B, KVH, _, D = p.shape
    out = [mx.take(p[b], idx[b].reshape(-1), axis=1).reshape(KVH, *idx.shape[1:], D) for b in range(B)]
    return out[0][None] if B == 1 else mx.stack(out)


class Attention(nn.Module):
    """Qwen3.5 gated attention (q_proj -> q | gate per head, Gemma RMSNorm on q/k, NeoX RoPE on the first 64 of 256
    dims, out * sigmoid(gate)); dense causal (`attn`) or BCSA (`bcsa`)."""

    def __init__(self, args: ModelArgs, bcsa: bool):
        super().__init__()
        self.n_heads, self.n_kv = args.num_attention_heads, args.num_key_value_heads
        self.head_dim = D = args.head_dim
        self.scale = D**-0.5
        h = args.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * D * 2, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * D, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * D, bias=False)
        self.o_proj = nn.Linear(self.n_heads * D, h, bias=False)
        self.q_norm = OffsetRMSNorm(D, args.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(D, args.rms_norm_eps)
        self.rope_dims = int(round(D * args.partial_rotary_factor))
        self.rope_theta = args.rope_theta
        self.bcsa = bcsa
        if bcsa:
            self.W, self.C, self.topk = args.bcsa_window, args.bcsa_compress, args.bcsa_topk
            self.IH, self.ID = args.bcsa_index_heads, args.bcsa_index_dim
            self.tile = args.bcsa_prefill_tile
            self.select_rows = args.bcsa_select_rows
            self.idx_q = nn.Linear(h, self.IH * self.ID, bias=False)
            self.idx_k = nn.Linear(h, self.ID, bias=False)
            self.idx_norm = OffsetRMSNorm(self.ID, args.rms_norm_eps)

    def _rope(self, x, offset):
        return mx.fast.rope(x, self.rope_dims, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)

    def __call__(self, x, cache):
        B, L, _ = x.shape
        D = self.head_dim
        qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(B, L, self.n_heads * D)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, D)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, D).transpose(0, 2, 1, 3)
        q, k = self._rope(q, cache.offset), self._rope(k, cache.offset)
        if not self.bcsa:
            k, v = cache.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal" if L > 1 else None)
        elif L == 1:
            o = self._bcsa_decode(x, q, k, v, cache)
        else:
            o = self._bcsa_prefill(x, q, k, v, cache)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * D)
        return self.o_proj(_sigmoid_gate(o, gate))

    def _index_q(self, x):
        B, L, _ = x.shape
        return self.idx_norm(self.idx_q(x).reshape(B, L, self.IH, self.ID))

    def _select(self, x, cache: BCSACache, p0: int, nv_max: int):
        """Top-k far blocks of the queries x (positions p0..) among their valid blocks -> [L, topk] int32 (B = 1).
        Scores as torch rounds them (bf16 einsum, head mean, scale); selection in the radix-select kernel."""
        qi = self._index_q(x)
        if self.ID == 128 and qi.dtype == BF16:  # fused NAX scoring, same rounding points, no per-head scores
            s = kernels.bcsa_scores(qi[0], cache.bk[0], nv_max, self.ID**-0.5)
            return kernels.bcsa_topk(s, p0, self.W, self.C, self.topk)
        bk = cache.bk[:, :nv_max]
        sel, rows = [], self.select_rows
        for r0 in range(0, x.shape[1], rows):
            s = _block_scores(qi[:, r0 : r0 + rows], bk, self.ID**-0.5)[0]
            sel.append(kernels.bcsa_topk(s, p0 + r0, self.W, self.C, self.topk))
        return sel[0] if len(sel) == 1 else mx.concatenate(sel, axis=0)

    def _bcsa_decode(self, x, q, k, v, cache: BCSACache):
        # one query at position P: SDPA over the window (P - W, P] and the top-k far blocks, joint softmax
        P, W = cache.offset, self.W
        cache.update(k, v, self.idx_norm(self.idx_k(x)))
        nv = max(0, (P - W + 1) // self.C)  # blocks lying entirely before the window
        if not nv:
            n = min(P + 1, W)
            return mx.fast.scaled_dot_product_attention(q, cache.k[..., :n, :], cache.v[..., :n, :], scale=self.scale)
        if nv > self.topk:
            if use_kernels() and x.shape[0] == 1:
                sel = self._select(x, cache, P, nv)  # [L = 1, topk] doubles as [B = 1, topk]
            else:
                s = _block_scores(self._index_q(x), cache.bk[:, :nv], self.ID**-0.5)
                sel = _topk_blocks(s[:, 0], self.topk)
            fk, fv = _gather_blocks(cache.pk, sel), _gather_blocks(cache.pv, sel)
        else:
            fk, fv = cache.pk[..., :nv, :], cache.pv[..., :nv, :]
        nf = fk.shape[2]
        cache.k[..., W : W + nf, :] = fk  # far slots after the ring (the ring is full here)
        cache.v[..., W : W + nf, :] = fv
        return mx.fast.scaled_dot_product_attention(
            q, cache.k[..., : W + nf, :], cache.v[..., : W + nf, :], scale=self.scale)

    def _bcsa_prefill(self, x, q, k, v, cache: BCSACache):
        B, H, L, D = q.shape
        P, W, C = cache.offset, self.W, self.C
        nprev = min(P, W - 1)
        if nprev:
            wk, wv = cache.window(nprev)
            keys, vals = mx.concatenate([wk, k], axis=2), mx.concatenate([wv, v], axis=2)
        else:
            keys, vals = k, v
        cache.update(k, v, self.idx_norm(self.idx_k(x)))
        kpos0 = P - nprev  # position of keys[..., 0, :]
        nv_max = max(0, (P + L - 1 - W + 1) // C)
        if nv_max == 0:  # no query reaches a far block: sliding-window attention
            if nprev == 0 and L <= W:
                return mx.fast.scaled_dot_product_attention(q, keys, vals, scale=self.scale, mask="causal")
            if use_kernels() and B == 1:
                return kernels.bcsa_window(q[0], keys[0], vals[0], self.scale, nprev, W)[None]
            qp = mx.arange(P, P + L)[:, None]
            kp = mx.arange(kpos0, P + L)[None]
            mask = (kp <= qp) & (kp > qp - W)
            return mx.fast.scaled_dot_product_attention(q, keys, vals, scale=self.scale, mask=mask)
        if not use_kernels() or B != 1:
            qi = self._index_q(x)
            outs = []
            for t0 in range(0, L, self.tile):
                t1 = min(L, t0 + self.tile)
                o = self._bcsa_tile(q[:, :, t0:t1], qi[:, t0:t1], keys, vals, cache, P + t0, P + t1, kpos0)
                mx.eval(o)
                outs.append(o)
            return mx.concatenate(outs, axis=2)
        # top-k far blocks per query; then one NAX flash-attention pass per 32-query block: the union of the block's
        # selected far blocks (each query masked to its own), then its sliding window, in one joint softmax
        sel = self._select(x, cache, P, nv_max)
        return kernels.bcsa_attend(q[0], keys[0], vals[0], cache.pk[0], cache.pv[0], sel, nv_max, self.scale, nprev,
                                   W)[None]

    def _bcsa_tile(self, q, qi, keys, vals, cache, p0, p1, kpos0):
        """Queries at positions [p0, p1): local window keys shared by the tile, per-query top-k far blocks,
        one fp32 softmax over both."""
        B, H, T, D = q.shape
        W, C, KVH = self.W, self.C, self.n_kv
        G = H // KVH
        lo = max(0, p0 - W + 1 - kpos0)
        hi = p1 - kpos0
        kl, vl = keys[..., lo:hi, :].astype(F32), vals[..., lo:hi, :].astype(F32)
        qp = mx.arange(p0, p1)
        kp = mx.arange(kpos0 + lo, kpos0 + hi)
        ok_loc = (kp[None] <= qp[:, None]) & (kp[None] > qp[:, None] - W)  # [T, nk]
        qf = q.astype(F32).reshape(B, KVH, G, T, D)
        s_loc = (qf @ kl[:, :, None].swapaxes(-1, -2)) * self.scale  # [B, KVH, G, T, nk]
        s_loc = mx.where(ok_loc, s_loc, -mx.inf)
        nv_max = max(0, (p1 - 1 - W + 1) // C)
        if nv_max == 0:
            o = mx.softmax(s_loc, axis=-1) @ vl[:, :, None]
            return o.reshape(B, H, T, D).astype(q.dtype)
        nv = mx.maximum((qp - W + 1) // C, 0)  # far blocks per query
        valid = mx.arange(nv_max)[None] < nv[:, None]  # [T, nv_max]
        s = _block_scores(qi, cache.bk[:, :nv_max], self.ID**-0.5)
        s = mx.where(valid, s, -mx.inf)
        kk = min(self.topk, nv_max)
        if kk < nv_max:
            sel = _topk_blocks(s, kk)  # [B, T, kk]
        else:
            sel = mx.broadcast_to(mx.arange(nv_max, dtype=mx.int32), (B, T, nv_max))
        ok_far = mx.take_along_axis(mx.broadcast_to(valid, (B, T, nv_max)), sel, axis=-1)  # [B, T, kk]
        fk = _gather_blocks(cache.pk, sel).astype(F32)  # [B, KVH, T, kk, D]
        fv = _gather_blocks(cache.pv, sel).astype(F32)
        s_far = (qf.transpose(0, 1, 3, 2, 4) @ fk.swapaxes(-1, -2)) * self.scale  # [B, KVH, T, G, kk]
        s_far = mx.where(ok_far[:, None, :, None, :], s_far, -mx.inf).transpose(0, 1, 3, 2, 4)
        m = mx.maximum(s_loc.max(-1, keepdims=True), s_far.max(-1, keepdims=True))
        p_loc, p_far = mx.exp(s_loc - m), mx.exp(s_far - m)
        z = p_loc.sum(-1, keepdims=True) + p_far.sum(-1, keepdims=True)
        o = p_loc @ vl[:, :, None]  # [B, KVH, G, T, D]
        o = o + (p_far.transpose(0, 1, 3, 2, 4) @ fv).transpose(0, 1, 3, 2, 4)
        return (o / z).reshape(B, H, T, D).astype(q.dtype)


# ----------------------------------------------------------------------------- layers / model
class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, idx: int):
        super().__init__()
        self.mixer = args.mixer_types[idx]
        if self.mixer == "kda":
            self.linear_attn = KDA(args)
        elif self.mixer in ("bcsa", "attn"):
            self.self_attn = Attention(args, bcsa=self.mixer == "bcsa")
        else:
            raise ValueError(f"unknown mixer {self.mixer!r}")
        self.input_layernorm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        if idx in args.engram_layers:
            self.engram = Engram(args)

    def __call__(self, x, cache, rows):
        fused = use_kernels() and x.dtype == BF16
        eps = self.input_layernorm.eps
        h = kernels.rms_norm(x, self.input_layernorm.weight, eps, offset=True)[1] if fused else self.input_layernorm(x)
        r = self.linear_attn(h, cache) if self.mixer == "kda" else self.self_attn(h, cache)
        if fused:  # residual add and the post-attention norm in one kernel
            h, hn = kernels.rms_norm(x, self.post_attention_layernorm.weight, eps, offset=True, r=r)
        else:
            h = x + r
            hn = self.post_attention_layernorm(h)
        y = h + self.mlp(hn)
        if "engram" in self:
            y = self.engram(y, rows)
        return y


class VolundrModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.mhc = [MHC(args) for _ in range(args.num_hidden_layers)]
        self.norm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, inputs, cache):
        a = self.args
        B, L = inputs.shape
        x = self.embed_tokens(inputs)
        rows = None
        if a.engram_layers:
            st = cache[-1]
            prev = st.prev if st.prev is not None else mx.full((B, st.keep), -1, dtype=mx.int32)
            full = mx.concatenate([prev, inputs.astype(mx.int32)], axis=1)
            st.prev = full[:, -st.keep :]
            rows = [engram_rows(full, n, L, a.engram_rows) for n in a.engram_orders]
        H = mx.broadcast_to(x[:, :, None, :], (B, L, a.mhc_streams, a.hidden_size))
        hmean = x  # the mean of identical streams
        for layer, mhc, c in zip(self.layers, self.mhc, cache):
            x_in, post, res = mhc.mix(H, hmean)
            y = layer(x_in, c, rows)
            H, hmean = mhc.combine(H, post, res, y, x_in)
        return self.norm(H[:, :, 0])  # stream 0 feeds the head


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = VolundrModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, last_only: bool = False):
        if W4A8_PREFILL and not getattr(self, "_w4a8", False):
            self._w4a8 = enable_w4a8(self) > 0
        if cache is None:
            cache = self.make_cache()
        h = self.model(inputs, cache)
        if last_only:
            h = h[:, -1:]
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        a = self.args
        caches = []
        for m in a.mixer_types:
            if m == "kda":
                caches.append(ArraysCache(size=2))
            elif m == "bcsa":
                caches.append(BCSACache(a.bcsa_window, a.bcsa_compress, a.bcsa_topk))
            else:
                caches.append(KVCache())
        caches.append(EngramState(max(a.engram_orders) - 1))
        return caches

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if not k.endswith(".weight_shape")}
