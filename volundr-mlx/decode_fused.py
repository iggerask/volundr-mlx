"""Fused small-op kernels for Volundr decode and verify steps (mx.fast.metal_kernel), bit-exact with the kernels and
MLX ops they replace (tests/test_decode_fused.py; the parity dump with them on is bit-identical to parity/mlx.bin).

Outside the 4-bit matmuls and the lm_head a decode step spent ~6 ms (NOTES_decode.md): ~1450 small dependent kernels,
the mHC Sinkhorn on one thread per token on every layer's critical path, and 5120-wide single-threadgroup row
reductions at ~11 us each. These kernels cut the count, parallelise the Sinkhorn and widen the row kernels, keeping
every rounding point and every summation order:

- `mix_norm`: mHC pre/post/Sinkhorn (kernels.mhc_params) + the stream mix (kernels.mhc_mix) + the layer's input
  RMSNorm (kernels.rms_norm's reduction tree). One threadgroup per token; one simdgroup runs the 10 Sinkhorn
  iterations with one lane per matrix element (the same sums, in the same order, via shuffles) while the others mix
  and normalise. Without the norm it is MHC.mix (also the verify path).
- `combine_norm`: the layer's residual add (y = h + mlp) + the mHC combine (kernels.mhc_combine) + the next layer's
  mHC RMSNorm of the stream mean, which only the norm read (the mean itself is no longer written).
- Wide variants (<= 64 rows, D >= 1024; also kernels.rms_norm): 1024 threads per row stage the row in threadgroup
  memory, then 256 of them sum the squares in kernels.rms_norm's order, so many more loads are in flight.
- `kda_conv_gates`: kernels.kda_conv_qk4 + the new conv state (was a concatenate) + the per-channel decay
  (mlx_lm compute_g: exp(-exp(A_log) * softplus(a + dt_bias))) + beta (sigmoid(b)), with MLX's own op formulas
  (unary_ops.h / binary_ops.h Exp, LogAddExp, Sigmoid), in one dispatch.
- `rms_norm_mlx`: mx.fast.rms_norm(x.astype(f32), w).astype(bf16) (OffsetRMSNorm / RMSNorm: q/k/indexer norms) in
  one dispatch, reading strided x in place.

Disable with VOLUNDR_FUSED=0 (or decode_fused.ENABLED = False at run time) to get the previous kernel sequence.
"""
import os
import re

import mlx.core as mx

ENABLED = os.environ.get("VOLUNDR_FUSED", "1") != "0"
BF16, F32 = mx.bfloat16, mx.float32


def norm_threads(D: int) -> int:
    """Threads per row of kernels.rms_norm: the fused norms must use the same count (same reduction tree)."""
    return 256 if D >= 1024 else 32 * max(1, min(8, D // 32))


def active(x) -> bool:
    return ENABLED and x.dtype == BF16 and mx.metal.is_available() and mx.default_device() == mx.gpu


# ----------------------------------------------------------------------------- mHC mix (+ input RMSNorm)
# Rounding points as kernels.py: pre/post = bf16(static + d); res = bf16(Sinkhorn(res_static + d[8:24])) (fp32, mhc.cu
# order: row sums (m0 + m2) + (m1 + m3), column sums ((m0 + m4) + m8) + m12, each a division by max(sum, 1e-9));
# x_in = bf16(sum_s bf16(pre_s * H_s)); h = bf16(x_in * rsqrt(mean(x_in^2) + eps) * (1 + w)).
_MIX_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device bfloat16_t* dr = d + size_t(row) * 24;
    const size_t hb = size_t(row) * 4 * D;
    const size_t base = size_t(row) * D;
    threadgroup float red[NT / 32];
    float hv[(D + NT - 1) / NT];
    float ss = 0.0f;
    if (tid >= NT) {
        // Sinkhorn: lane l holds element l & 15 of the 4x4 matrix (lanes 16..31 mirror 0..15)
        const uint l = lane & 15u;
        const uint r4 = l & ~3u, c = l & 3u;
        float m = float(res_s[l]) + float(dr[8 + l]);
        const float mxv = simd_max(m);
        m = metal::precise::exp(m - mxv);
        for (int it = 0; it < ITERS; it++) {
            const float a0 = simd_shuffle(m, ushort(r4)), a1 = simd_shuffle(m, ushort(r4 + 1));
            const float a2 = simd_shuffle(m, ushort(r4 + 2)), a3 = simd_shuffle(m, ushort(r4 + 3));
            float s = metal::max((a0 + a2) + (a1 + a3), 1e-9f);
            m = metal::precise::divide(m, s);
            const float b0 = simd_shuffle(m, ushort(c)), b1 = simd_shuffle(m, ushort(4 + c));
            const float b2 = simd_shuffle(m, ushort(8 + c)), b3 = simd_shuffle(m, ushort(12 + c));
            s = metal::max(((b0 + b1) + b2) + b3, 1e-9f);
            m = metal::precise::divide(m, s);
        }
        if (lane < 16) res[size_t(row) * 16 + lane] = bfloat16_t(m);
        if (lane < 4) post[size_t(row) * 4 + lane] = bfloat16_t(float(post_s[lane]) + float(dr[4 + lane]));
    } else {
        float p[4];
        for (int s = 0; s < 4; s++) p[s] = float(bfloat16_t(float(pre_s[s]) + float(dr[s])));
        for (int k = 0; k < (D + NT - 1) / NT; k++) {
            const int i = tid + k * NT;
            if (D % NT != 0 && i >= D) break;
            float acc = 0.0f;
            for (int s = 0; s < 4; s++) {
                acc += float(bfloat16_t(p[s] * float(H[hb + s * D + i])));
            }
            const bfloat16_t xb = bfloat16_t(acc);
            x[base + i] = xb;
            const float v = float(xb);
            hv[k] = v;
            ss += v * v;
        }
    }
    NORM_PART
"""
# the input RMSNorm, kernels.rms_norm's reduction (simd sums, then the NT / 32 partials in simdgroup 0)
_NORM_PART = """
    if (tid < NT) {
        ss = simd_sum(ss);
        if (lane == 0) red[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < NT / 32 ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NT) {
        const float rs = metal::precise::rsqrt(red[0] / float(D) + EPS_VAL);
        for (int k = 0; k < (D + NT - 1) / NT; k++) {
            const int i = tid + k * NT;
            if (D % NT != 0 && i >= D) break;
            y[base + i] = bfloat16_t(hv[k] * rs * (OFFSET + float(w[i])));
        }
    }
"""
# Wide variants (D >= 1024): NW threads per row instead of NT, so each thread has a few elements and many more loads are
# in flight. The row's norm inputs are staged in threadgroup memory, and the first NT threads then sum their squares
# exactly as above (thread t: elements t, t + NT, ... in order, the same simd and partial sums): bit-identical. In the
# mix kernel the last simdgroup runs the Sinkhorn and the other NW - 32 threads mix.
_MIX_WIDE_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device bfloat16_t* dr = d + size_t(row) * 24;
    const size_t hb = size_t(row) * 4 * D;
    const size_t base = size_t(row) * D;
    constexpr int NM = NW - 32;
    SV_DECL
    if (tid >= uint(NM)) {
        const uint l = lane & 15u;
        const uint r4 = l & ~3u, c = l & 3u;
        float m = float(res_s[l]) + float(dr[8 + l]);
        const float mxv = simd_max(m);
        m = metal::precise::exp(m - mxv);
        for (int it = 0; it < ITERS; it++) {
            const float a0 = simd_shuffle(m, ushort(r4)), a1 = simd_shuffle(m, ushort(r4 + 1));
            const float a2 = simd_shuffle(m, ushort(r4 + 2)), a3 = simd_shuffle(m, ushort(r4 + 3));
            float s = metal::max((a0 + a2) + (a1 + a3), 1e-9f);
            m = metal::precise::divide(m, s);
            const float b0 = simd_shuffle(m, ushort(c)), b1 = simd_shuffle(m, ushort(4 + c));
            const float b2 = simd_shuffle(m, ushort(8 + c)), b3 = simd_shuffle(m, ushort(12 + c));
            s = metal::max(((b0 + b1) + b2) + b3, 1e-9f);
            m = metal::precise::divide(m, s);
        }
        if (lane < 16) res[size_t(row) * 16 + lane] = bfloat16_t(m);
        if (lane < 4) post[size_t(row) * 4 + lane] = bfloat16_t(float(post_s[lane]) + float(dr[4 + lane]));
    } else {
        float p[4];
        for (int s = 0; s < 4; s++) p[s] = float(bfloat16_t(float(pre_s[s]) + float(dr[s])));
        for (int j = 0; j < (D + NM - 1) / NM; j++) {
            const int i = tid + j * NM;
            if (i >= D) break;
            float acc = 0.0f;
            for (int s = 0; s < 4; s++) {
                acc += float(bfloat16_t(p[s] * float(H[hb + s * D + i])));
            }
            const bfloat16_t xb = bfloat16_t(acc);
            x[base + i] = xb;
            SV_STORE(float(xb))
        }
    }
    NORM_PART
"""
_NORM_PART_WIDE = """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NT) {
        float ss = 0.0f;
        for (int k = 0; k < (D + NT - 1) / NT; k++) {
            const int i = tid + k * NT;
            if (D % NT != 0 && i >= D) break;
            const float v = sv[i];
            ss += v * v;
        }
        ss = simd_sum(ss);
        if (lane == 0) red[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < NT / 32 ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs = metal::precise::rsqrt(red[0] / float(D) + EPS_VAL);
    for (int j = 0; j < (D + NW - 1) / NW; j++) {
        const int i = tid + j * NW;
        if (D % NW != 0 && i >= D) break;
        y[base + i] = bfloat16_t(sv[i] * rs * (OFFSET + float(w[i])));
    }
"""
WIDE = 1024  # threads per row of the wide variants (0: the NT-thread kernels)
WIDE_MAX_ROWS = 64  # decode / verify; prefill-size calls keep the NT-thread kernels (one row's 20 KB of threadgroup
# memory per 1024-thread group would cap occupancy there)


def _wide(D: int, N: int) -> int:
    return WIDE if WIDE > norm_threads(D) and 1024 <= D <= 6144 and N <= WIDE_MAX_ROWS else 0


def _wide_src(src: str, norm: bool, eps: float) -> str:
    """Fill the SV_DECL / SV_STORE / NORM_PART slots of a wide source."""
    if norm:
        src = src.replace("SV_DECL", "threadgroup float sv[D];\n    threadgroup float red[NT / 32];").replace(
            "NORM_PART", _NORM_PART_WIDE.replace("EPS_VAL", repr(float(eps)) + "f"))
    else:
        src = src.replace("SV_DECL", "").replace("NORM_PART", "")
    return re.sub(r"SV_STORE\((.*)\)\n", (r"sv[i] = \1;\n" if norm else "\n"), src)


_mix_kernels = {}


def _mix_kernel(norm: bool, eps: float, wide: bool):
    key = (norm, eps if norm else None, wide)
    if key not in _mix_kernels:
        if wide:
            src = _wide_src(_MIX_WIDE_SRC, norm, eps)
        else:
            src = _MIX_SRC.replace("NORM_PART", _NORM_PART.replace("EPS_VAL", repr(float(eps)) + "f") if norm else "")
        _mix_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_df_mix_{int(norm)}{int(wide)}_{len(_mix_kernels)}",
            input_names=["d", "pre_s", "post_s", "res_s", "H"] + (["w"] if norm else []),
            output_names=["x"] + (["y"] if norm else []) + ["post", "res"], source=src)
    return _mix_kernels[key]


def mix_norm(d, pre_s, post_s, res_s, H, iters, w=None, eps=1e-6, offset=True):
    """d [N, 24] bf16 (the dyn linear), statics bf16, H [N, 4, D] bf16, w [D] (input norm weight; None = no norm) ->
    x_in [N, D], h [N, D] (None without w), post [N, 4], res [N, 4, 4] (bf16)."""
    N, n, D = H.shape
    assert n == 4 and d.shape == (N, 24)
    nt = norm_threads(D)
    nw = _wide(D, N)
    tg = nw or nt + 32
    norm = w is not None
    outs = _mix_kernel(norm, eps, nw > 0)(
        inputs=[d, pre_s, post_s, res_s, H] + ([w] if norm else []),
        template=[("D", D), ("NT", nt), ("ITERS", iters)] + ([("OFFSET", 1 if offset else 0)] if norm else [])
        + ([("NW", nw)] if nw else []),
        grid=(tg, N, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)] + ([(N, D)] if norm else []) + [(N, 4), (N, 4, 4)],
        output_dtypes=[BF16] * (4 if norm else 3))
    if norm:
        return outs[0], outs[1], outs[2], outs[3]
    return outs[0], None, outs[1], outs[2]


# ----------------------------------------------------------------------------- mHC combine (+ residual add, next norm)
# Per element, as kernels.mhc_combine: delta = bf16(y - x_in); out_s = bf16(bf16(sum_j res_sj * H_j) + bf16(post_s *
# delta)); mean = bf16(0.25 * sum_s out_s). ADD: y = bf16(h + m) (MLX's bf16 add). NORM: the next layer's mHC RMSNorm
# of mean (plain weight) as kernels.rms_norm(offset=False) computes it; the mean is not written.
_COMBINE_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t hb = size_t(row) * 4 * D;
    const size_t base = size_t(row) * D;
    float rr[16], pp[4];
    for (int j = 0; j < 16; j++) rr[j] = float(res[size_t(row) * 16 + j]);
    for (int s = 0; s < 4; s++) pp[s] = float(post[size_t(row) * 4 + s]);
    threadgroup float red[NT / 32];
    float hv[(D + NT - 1) / NT];
    float ss = 0.0f;
    for (int k = 0; k < (D + NT - 1) / NT; k++) {
        const int i = tid + k * NT;
        if (D % NT != 0 && i >= D) break;
        float h[4];
        for (int s = 0; s < 4; s++) h[s] = float(H[hb + s * D + i]);
        LOAD_Y
        const float delta = float(bfloat16_t(float(yb) - float(x_in[base + i])));
        float sum = 0.0f;
        for (int s = 0; s < 4; s++) {
            float mixed = 0.0f;
            for (int j = 0; j < 4; j++) mixed += rr[s * 4 + j] * h[j];
            const float upd = float(bfloat16_t(pp[s] * delta));
            const bfloat16_t o = bfloat16_t(float(bfloat16_t(mixed)) + upd);
            Hn[hb + s * D + i] = o;
            sum += float(o);
        }
        const float v = float(bfloat16_t(sum * 0.25f));
        hv[k] = v;
        ss += v * v;
    }
    NORM_PART
"""
_COMBINE_WIDE_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t hb = size_t(row) * 4 * D;
    const size_t base = size_t(row) * D;
    float rr[16], pp[4];
    for (int j = 0; j < 16; j++) rr[j] = float(res[size_t(row) * 16 + j]);
    for (int s = 0; s < 4; s++) pp[s] = float(post[size_t(row) * 4 + s]);
    SV_DECL
    for (int j = 0; j < (D + NW - 1) / NW; j++) {
        const int i = tid + j * NW;
        if (D % NW != 0 && i >= D) break;
        float h[4];
        for (int s = 0; s < 4; s++) h[s] = float(H[hb + s * D + i]);
        LOAD_Y
        const float delta = float(bfloat16_t(float(yb) - float(x_in[base + i])));
        float sum = 0.0f;
        for (int s = 0; s < 4; s++) {
            float mixed = 0.0f;
            for (int jj = 0; jj < 4; jj++) mixed += rr[s * 4 + jj] * h[jj];
            const float upd = float(bfloat16_t(pp[s] * delta));
            const bfloat16_t o = bfloat16_t(float(bfloat16_t(mixed)) + upd);
            Hn[hb + s * D + i] = o;
            sum += float(o);
        }
        SV_STORE(float(bfloat16_t(sum * 0.25f)))
    }
    NORM_PART
"""
_LOAD_Y_ADD = "const bfloat16_t yb = bfloat16_t(float(hr[base + i]) + float(mo[base + i]));"
_LOAD_Y = "const bfloat16_t yb = yi[base + i];"
_combine_kernels = {}


def _combine_kernel(add: bool, norm: bool, eps: float, wide: bool):
    key = (add, norm, eps if norm else None, wide)
    if key not in _combine_kernels:
        if wide:
            src = _wide_src(_COMBINE_WIDE_SRC, norm, eps).replace("LOAD_Y", _LOAD_Y_ADD if add else _LOAD_Y)
        else:
            src = _COMBINE_SRC.replace("LOAD_Y", _LOAD_Y_ADD if add else _LOAD_Y).replace(
                "NORM_PART", _NORM_PART.replace("EPS_VAL", repr(float(eps)) + "f") if norm else "")
        _combine_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_df_combine_{int(add)}{int(norm)}{int(wide)}_{len(_combine_kernels)}",
            input_names=["H", "res", "post"] + (["hr", "mo"] if add else ["yi"]) + ["x_in"] + (["w"] if norm else []),
            output_names=["Hn"] + (["y"] if norm else []), source=src)
    return _combine_kernels[key]


def combine_norm(H, res, post, y, x_in, w=None, eps=1e-6):
    """H [N, 4, D], res [N, 4, 4], post [N, 4], y: the layer output [N, D] or a pair (h, m) summed here as bf16(h + m),
    x_in [N, D], w: the next layer's mHC norm weight (None: no norm) -> H' [N, 4, D], hn [N, D] or None."""
    N, n, D = H.shape
    add = isinstance(y, tuple)
    norm = w is not None
    nt = norm_threads(D)
    nw = _wide(D, N)
    tg = nw or nt
    outs = _combine_kernel(add, norm, eps, nw > 0)(
        inputs=[H, res.reshape(N, 16), post] + (list(y) if add else [y]) + [x_in] + ([w] if norm else []),
        template=[("D", D), ("NT", nt)] + ([("OFFSET", 0)] if norm else []) + ([("NW", nw)] if nw else []),
        grid=(tg, N, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(N, n, D)] + ([(N, D)] if norm else []), output_dtypes=[BF16] * (2 if norm else 1))
    return outs[0], (outs[1] if norm else None)


# ----------------------------------------------------------------------------- KDA conv + conv state + gates
# MLX's op formulas (mlx/backend/metal/kernels/unary_ops.h Exp, Sigmoid; binary_ops.h LogAddExp; utils.h log1p, which
# the kernel preamble already includes), so the decay and beta round exactly as mlx_lm's compiled compute_g and
# mx.sigmoid do.
_GATES_HEADER = """
inline float vdf_exp(float x) { return metal::precise::exp(x); }
inline float vdf_logaddexp(float x, float y) {
    if (metal::isnan(x) || metal::isnan(y)) {
        return metal::numeric_limits<float>::quiet_NaN();
    }
    constexpr float inf = metal::numeric_limits<float>::infinity();
    float maxval = metal::max(x, y);
    float minval = metal::min(x, y);
    return (minval == -inf || maxval == inf) ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
}
inline float vdf_sigmoid(float x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
"""
# Threads c4 < C / 4: kernels.kda_conv_qk4 verbatim (4 channels, TT timesteps per thread) + the new conv state (the
# last KW - 1 rows of [state, qkv]) for their channels. Threads past C / 4: 4 decay channels x TT timesteps each, and
# beta for their head when they hold its first channel.
_CONV_GATES_SRC = """
    const uint c4 = thread_position_in_grid.x;
    const uint tb = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const int L = qkv_shape[1];
    const int C = qkv_shape[2];
    const int t0 = tb * TT;
    if (int(c4) >= C / 4) {
        const int e0 = (int(c4) - C / 4) * 4;
        const int h = e0 / DK;
        const float na = -vdf_exp(A_log[h]);
        for (int s = 0; s < TT; s++) {
            const int t = t0 + s;
            if (t >= L) break;
            const size_t gi = (size_t(b) * L + t) * GK + e0;
            const uint2 u = *(const device uint2*)(a + gi);
            const float av[4] = {as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                                 as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u)};
            float4 gv;
            for (int i = 0; i < 4; i++) gv[i] = vdf_exp(na * vdf_logaddexp(av[i] + dt_bias[e0 + i], 0.0f));
            *(device float4*)(g + gi) = gv;
            if (e0 % DK == 0) {
                const size_t bi = (size_t(b) * L + t) * HV + h;
                beta[bi] = vdf_sigmoid(float(bb[bi]));
            }
        }
        return;
    }
    const int c0 = c4 * 4;
    if (tb == 0) {
        for (int r = 0; r < KW - 1; r++) {
            const int tt = L + r - (KW - 1);
            const uint2 u = tt < 0 ? *(const device uint2*)(state + (size_t(b) * (KW - 1) + (KW - 1 + tt)) * C + c0)
                                   : *(const device uint2*)(qkv + (size_t(b) * L + tt) * C + c0);
            *(device uint2*)(cs + (size_t(b) * (KW - 1) + r) * C + c0) = u;
        }
    }
    float wv[4][KW];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < KW; j++) wv[i][j] = w[(c0 + i) * KW + j];
    float xr[TT + KW - 1][4];
    for (int r = 0; r < TT + KW - 1; r++) {
        const int tt = t0 + r - (KW - 1);
        uint2 u = uint2(0, 0);
        if (tt < 0) {
            u = *(const device uint2*)(state + (size_t(b) * (KW - 1) + (KW - 1 + tt)) * C + c0);
        } else if (tt < L) {
            u = *(const device uint2*)(qkv + (size_t(b) * L + tt) * C + c0);
        }
        xr[r][0] = as_type<float>(u.x << 16);
        xr[r][1] = as_type<float>(u.x & 0xFFFF0000u);
        xr[r][2] = as_type<float>(u.y << 16);
        xr[r][3] = as_type<float>(u.y & 0xFFFF0000u);
    }
    const bool isqk = c0 < 2 * KD;
    const bool isq = c0 < KD;
    for (int s = 0; s < TT; s++) {
        const int t = t0 + s;
        float y[4];
        float ss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float acc = 0.0f;
            for (int j = 0; j < KW; j++) acc += xr[s + j][i] * wv[i][j];
            const float sv = float(bfloat16_t(acc / (1.0f + metal::precise::exp(-acc))));
            y[i] = sv;
            ss += sv * sv;
        }
        if (isqk) {
            ss = simd_sum(ss);
            const float inv = 1.0f / metal::precise::sqrt(ss + 1e-6f);
            const float sc = isq ? inv * QSCALE : inv;
            if (t < L) {
                device float* dst = isq ? q : k;
                const int cc = isq ? c0 : c0 - KD;
                *(device float4*)(dst + (size_t(b) * L + t) * KD + cc) = float4(y[0], y[1], y[2], y[3]) * sc;
            }
        } else if (t < L) {
            const int cv = c0 - 2 * KD;
            uint2 o;
            o.x = uint(as_type<ushort>(bfloat16_t(y[0]))) | (uint(as_type<ushort>(bfloat16_t(y[1]))) << 16);
            o.y = uint(as_type<ushort>(bfloat16_t(y[2]))) | (uint(as_type<ushort>(bfloat16_t(y[3]))) << 16);
            *(device uint2*)(v + (size_t(b) * L + t) * (C - 2 * KD) + cv) = o;
        }
    }
"""
_conv_gates_kernels = {}


def kda_conv_gates(qkv, state, w, key_dim, A_log, a, dt_bias, b, tt=4):
    """qkv [B, L, C] bf16, state [B, K-1, C] bf16, w [C, K] fp32, A_log [HV] fp32, a [B, L, HV * 128] bf16 (f_up),
    dt_bias [HV * 128] fp32, b [B, L, HV] bf16 (in_proj_b) -> q, k [B, L, key_dim] fp32 (as kernels.kda_conv_qk4),
    v [B, L, C - 2 key_dim] bf16, new conv state [B, K-1, C] bf16, decay g [B, L, HV * 128] fp32, beta [B, L, HV]
    fp32."""
    B, L, C = qkv.shape
    KW = w.shape[1]
    GK = a.shape[-1]
    HV = b.shape[-1]
    DK = GK // HV
    assert DK == 128 and C % 128 == 0 and key_dim % 128 == 0 and GK % 4 == 0
    assert a.shape == (B, L, GK) and b.shape == (B, L, HV) and a.dtype == BF16 and qkv.dtype == BF16
    if tt not in _conv_gates_kernels:
        _conv_gates_kernels[tt] = mx.fast.metal_kernel(
            name=f"volundr_df_conv_gates_{tt}",
            input_names=["qkv", "state", "w", "A_log", "a", "dt_bias", "bb"],
            output_names=["q", "k", "v", "cs", "g", "beta"],
            source=_CONV_GATES_SRC.replace("QSCALE", repr(float(DK) ** -0.5) + "f"), header=_GATES_HEADER)
    nx = C // 4 + GK // 4
    return _conv_gates_kernels[tt](
        inputs=[qkv, state, w, A_log, a, dt_bias, b],
        template=[("KW", KW), ("KD", key_dim), ("TT", tt), ("GK", GK), ("HV", HV), ("DK", DK)],
        grid=(nx, (L + tt - 1) // tt, B), threadgroup=(min(256, C // 4), 1, 1),
        output_shapes=[(B, L, key_dim), (B, L, key_dim), (B, L, C - 2 * key_dim), (B, KW - 1, C), (B, L, GK), (B, L, HV)],
        output_dtypes=[F32, F32, BF16, BF16, F32, F32])


# ----------------------------------------------------------------------------- RMSNorm as mx.fast.rms_norm (fp32)
# OffsetRMSNorm / RMSNorm compute mx.fast.rms_norm(x.astype(f32), w_f32, eps).astype(x.dtype): a strided copy to fp32,
# MLX's single-row rms kernel, and a cast back. This is that kernel's arithmetic (4 consecutive elements per thread,
# simd sums, the per-simdgroup partials summed by simdgroup 0, out = w * (x * rsqrt(mean + eps))) reading bf16 x in
# place (any strides) and writing bf16, in one dispatch.
_MLX_RMS_SRC = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    float acc = 0;
    float xv[4];
    for (int i = 0; i < 4; i++) {
        const int64_t e = int64_t(row) * D + lid * 4 + i;
        xv[i] = (lid * 4 + i < D) ? float(x[elem_to_loc(e, x_shape, x_strides, x_ndim)]) : 0.0f;
        if (lid * 4 + i < D) acc += xv[i] * xv[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / uint(D) + EPS_VAL);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device bfloat16_t* o = out + size_t(row) * D + lid * 4;
    for (int i = 0; i < 4; i++) {
        if (lid * 4 + i < D) o[i] = bfloat16_t(wf[lid * 4 + i] * (xv[i] * local_inv_mean[0]));
    }
"""
_mlx_rms_kernels = {}


def rms_norm_mlx(x, wf, eps):
    """== mx.fast.rms_norm(x.astype(f32), wf, eps).astype(bf16) for bf16 x [..., D] (D <= 4096, any strides) and fp32
    wf [D]; the output is row contiguous."""
    D = x.shape[-1]
    assert x.dtype == BF16 and wf.dtype == F32 and D <= 4096
    if eps not in _mlx_rms_kernels:
        _mlx_rms_kernels[eps] = mx.fast.metal_kernel(
            name=f"volundr_df_mlx_rms_{len(_mlx_rms_kernels)}", input_names=["x", "wf"], output_names=["out"],
            source=_MLX_RMS_SRC.replace("EPS_VAL", repr(float(eps)) + "f"), ensure_row_contiguous=False)
    nt = 32 * ((D + 4 * 32 - 1) // (4 * 32))
    N = x.size // D
    return _mlx_rms_kernels[eps](
        inputs=[x, wf], template=[("D", D)], grid=(N * nt, 1, 1), threadgroup=(nt, 1, 1),
        output_shapes=[x.shape], output_dtypes=[BF16])[0]
