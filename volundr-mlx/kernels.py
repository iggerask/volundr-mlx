"""Metal kernels for Volundr MLX (mx.fast.metal_kernel). Each has an ops reference in volundr.py or tests/."""
from pathlib import Path

import mlx.core as mx

F32 = mx.float32

# ----------------------------------------------------------------------------- BCSA top-k
# Top-k block selection per query: the KSEL largest bf16-rounded scores among the query's valid blocks, ties to the
# lower block index (torch topk / the engine's radix select). Radix select on unique 32-bit keys
# (ordered bf16 << 16 | 0xFFFF - index), then a deterministic compaction in index order.
_TOPK_SRC = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint t = threadgroup_position_in_grid.y;
    const int nbc = s_shape[1];
    const int pos = params[0] + int(t);
    int nv = (pos - WIN + 1) >= 0 ? (pos - WIN + 1) / CMP : 0;
    nv = min(nv, nbc);
    device int* out = sel + t * KSEL;
    if (nv <= KSEL) {
        for (int j = tid; j < KSEL; j += NT) out[j] = j < nv ? j : -1;
        return;
    }
    auto row = s + size_t(t) * nbc;
    threadgroup atomic_uint hist[256];
    threadgroup uint sh_prefix, sh_need;
    threadgroup uint sh_sums[NT / 32];
    uint prefix = 0, need = KSEL;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tid; j < nv; j += NT) {
            uint key = make_key(float(row[j]), j);
            if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8)))
                atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint acc = 0;
            int b = 255;
            for (; b > 0; b--) {
                uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (acc + h >= need) break;
                acc += h;
            }
            sh_prefix = prefix | (uint(b) << shift);
            sh_need = need - acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh_prefix;
        need = sh_need;
    }
    // prefix is now the key of the KSEL-th largest; keys are unique, so exactly KSEL keys are >= prefix.
    // compaction in index order: thread tid owns indices [tid * R, (tid + 1) * R)
    const int R = (nv + NT - 1) / NT;
    const int j0 = int(tid) * R, j1 = min(j0 + R, nv);
    uint c = 0;
    for (int j = j0; j < j1; j++) c += make_key(float(row[j]), j) >= prefix ? 1u : 0u;
    uint ex = simd_prefix_exclusive_sum(c);
    if (thread_index_in_simdgroup == 31) sh_sums[simdgroup_index_in_threadgroup] = ex + c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint a = 0;
        for (uint i = 0; i < NT / 32; i++) { uint v = sh_sums[i]; sh_sums[i] = a; a += v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint p = sh_sums[simdgroup_index_in_threadgroup] + ex;
    for (int j = j0; j < j1; j++)
        if (make_key(float(row[j]), j) >= prefix) out[p++] = j;
"""
_TOPK_HEADER = """
inline uint make_key(float x, int j) {
    uint u = as_type<uint>(x);
    u += 0x7FFFu + ((u >> 16) & 1u);            // round to bf16, nearest even
    uint h = u >> 16;
    uint o = (h & 0x8000u) ? (~h & 0xFFFFu) : (h | 0x8000u);   // monotone in the value
    return (o << 16) | (0xFFFFu - uint(j));
}
"""
_topk_kernel = mx.fast.metal_kernel(
    name="volundr_bcsa_topk", input_names=["s", "params"], output_names=["sel"], source=_TOPK_SRC,
    header=_TOPK_HEADER)


def bcsa_topk(s: mx.array, p0: int, window: int, compress: int, ksel: int) -> mx.array:
    """s [T, NB] block scores (fp32 or bf16, already scaled) of queries at positions p0..p0+T-1 -> sel [T, ksel] int32:
    the selected block indices (ascending), -1 padded when a query has fewer than ksel valid blocks."""
    T, nb = s.shape
    assert nb <= 65536, "unique keys need block indices < 2^16"
    nt = 1024
    (sel,) = _topk_kernel(
        inputs=[s, mx.array([p0], dtype=mx.int32)],
        template=[("WIN", window), ("CMP", compress), ("KSEL", ksel), ("NT", nt)],
        grid=(nt, T, 1), threadgroup=(nt, 1, 1),
        output_shapes=[(T, ksel)], output_dtypes=[mx.int32],
    )
    return sel


# ----------------------------------------------------------------------------- mHC
# Streams' rounding points as mhc.cu: x_in = bf16(sum_s bf16(pre_s * H_s)); out_s = bf16(mixed_s + bf16(post_s *
# bf16(y - x_in))) with mixed_s = bf16(sum_j res_sj * H_j) (torch's bmm, fp32 accumulation); mean = bf16(sum_s out_s
# * 0.25), the next layer's norm input.
_MHC_MIX_SRC = """
    const uint d = thread_position_in_grid.x;
    const uint n = thread_position_in_grid.y;
    if (d >= D) return;
    const size_t hb = size_t(n) * 4 * D + d;
    float acc = 0.0f;
    for (int s = 0; s < 4; s++) {
        const float p = float(pre[n * 4 + s]);
        acc += float(bfloat16_t(p * float(H[hb + s * D])));
    }
    x[size_t(n) * D + d] = bfloat16_t(acc);
"""
_mhc_mix_kernel = mx.fast.metal_kernel(
    name="volundr_mhc_mix", input_names=["H", "pre"], output_names=["x"], source=_MHC_MIX_SRC)

_MHC_COMBINE_SRC = """
    const uint d = thread_position_in_grid.x;
    const uint n = thread_position_in_grid.y;
    if (d >= D) return;
    const size_t hb = size_t(n) * 4 * D + d;
    float h[4];
    for (int s = 0; s < 4; s++) h[s] = float(H[hb + s * D]);
    const float delta = float(bfloat16_t(float(y[size_t(n) * D + d]) - float(x_in[size_t(n) * D + d])));
    float sum = 0.0f;
    for (int s = 0; s < 4; s++) {
        float mixed = 0.0f;
        for (int j = 0; j < 4; j++) mixed += float(res[n * 16 + s * 4 + j]) * h[j];
        const float upd = float(bfloat16_t(float(post[n * 4 + s]) * delta));
        const bfloat16_t o = bfloat16_t(float(bfloat16_t(mixed)) + upd);
        Hn[hb + s * D] = o;
        sum += float(o);
    }
    mean[size_t(n) * D + d] = bfloat16_t(sum * 0.25f);
"""
_mhc_combine_kernel = mx.fast.metal_kernel(
    name="volundr_mhc_combine", input_names=["H", "res", "post", "y", "x_in"], output_names=["Hn", "mean"],
    source=_MHC_COMBINE_SRC)

# pre / post (bf16) and the Sinkhorn projection of res from the dyn output: one thread per token
_MHC_PARAMS_SRC = """
    const uint n = thread_position_in_grid.x;
    if (n >= d_shape[0]) return;
    const device bfloat16_t* dr = d + n * 24;
    for (int s = 0; s < 4; s++) {
        pre[n * 4 + s] = bfloat16_t(float(pre_s[s]) + float(dr[s]));
        post[n * 4 + s] = bfloat16_t(float(post_s[s]) + float(dr[4 + s]));
    }
    float m[16];
    float mx = -INFINITY;
    for (int k = 0; k < 16; k++) {
        m[k] = float(res_s[k]) + float(dr[8 + k]);
        mx = metal::max(mx, m[k]);
    }
    for (int k = 0; k < 16; k++) m[k] = metal::precise::exp(m[k] - mx);
    for (int it = 0; it < ITERS; it++) {
        for (int r = 0; r < 4; r++) {
            float s = metal::max((m[4 * r] + m[4 * r + 2]) + (m[4 * r + 1] + m[4 * r + 3]), 1e-9f);
            for (int c = 0; c < 4; c++) m[4 * r + c] = metal::precise::divide(m[4 * r + c], s);
        }
        for (int c = 0; c < 4; c++) {
            float s = metal::max(((m[c] + m[4 + c]) + m[8 + c]) + m[12 + c], 1e-9f);
            for (int r = 0; r < 4; r++) m[4 * r + c] = metal::precise::divide(m[4 * r + c], s);
        }
    }
    for (int k = 0; k < 16; k++) res[n * 16 + k] = bfloat16_t(m[k]);
"""
_mhc_params_kernel = mx.fast.metal_kernel(
    name="volundr_mhc_params", input_names=["d", "pre_s", "post_s", "res_s"], output_names=["pre", "post", "res"],
    source=_MHC_PARAMS_SRC)


def mhc_params(d, pre_s, post_s, res_s, iters):
    """d [N, 24] bf16 -> pre [N, 4], post [N, 4], res [N, 4, 4] (bf16)."""
    N = d.shape[0]
    pre, post, res = _mhc_params_kernel(
        inputs=[d, pre_s, post_s, res_s], template=[("ITERS", iters)],
        grid=(N, 1, 1), threadgroup=(min(N, 256), 1, 1),
        output_shapes=[(N, 4), (N, 4), (N, 4, 4)], output_dtypes=[mx.bfloat16] * 3)
    return pre, post, res


def mhc_mix(H, pre):
    """H [N, 4, D], pre [N, 4] -> x_in [N, D] (bf16)."""
    N, _, D = H.shape
    (x,) = _mhc_mix_kernel(inputs=[H, pre], template=[("D", D)], grid=(D, N, 1), threadgroup=(min(256, D), 1, 1),
                           output_shapes=[(N, D)], output_dtypes=[mx.bfloat16])
    return x


def mhc_combine(H, res, post, y, x_in):
    """-> H' [N, 4, D], mean [N, D] (bf16)."""
    N, _, D = H.shape
    Hn, mean = _mhc_combine_kernel(
        inputs=[H, res, post, y, x_in], template=[("D", D)], grid=(D, N, 1), threadgroup=(min(256, D), 1, 1),
        output_shapes=[(N, 4, D), (N, D)], output_dtypes=[mx.bfloat16] * 2)
    return Hn, mean


# ----------------------------------------------------------------------------- RMSNorm (+ residual add)
# One threadgroup per row: h = bf16(x + r) (if ADD), y = bf16(h * rsqrt(mean(h^2) + eps) * (OFFSET + w)), fp32 math.
_NORM_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const size_t base = size_t(row) * D;
    threadgroup float red[NT / 32];
    float hv[(D + NT - 1) / NT];
    float ss = 0.0f;
    // constant trip count (all rows' loads issued up front); the same elements in the same order as a strided loop
    for (int k = 0; k < (D + NT - 1) / NT; k++) {
        const int i = tid + k * NT;
        if (D % NT != 0 && i >= D) break;
        LOAD
        hv[k] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        float t = thread_index_in_simdgroup < NT / 32 ? red[thread_index_in_simdgroup] : 0.0f;
        t = simd_sum(t);
        if (thread_index_in_simdgroup == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs = metal::precise::rsqrt(red[0] / float(D) + EPS_VAL);
    for (int k = 0; k < (D + NT - 1) / NT; k++) {
        const int i = tid + k * NT;
        if (D % NT != 0 && i >= D) break;
        y[base + i] = bfloat16_t(hv[k] * rs * (OFFSET + float(w[i])));
    }
"""
# Wide variant for D >= 1024: NW (> NT) threads load / write the row with NW-strided coalesced accesses and stage the
# values in threadgroup memory; then the first NT threads sum the squares exactly as above (thread t: elements t, t + NT,
# ... in order, then the same simd / partial sums), so the result is bit-identical with more loads in flight.
_NORM_WIDE_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const size_t base = size_t(row) * D;
    threadgroup float red[NT / 32];
    threadgroup float sv[D];
    for (int j = 0; j < (D + NW - 1) / NW; j++) {
        const int i = tid + j * NW;
        if (D % NW != 0 && i >= D) break;
        LOAD
        sv[i] = v;
    }
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
        if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        float t = thread_index_in_simdgroup < NT / 32 ? red[thread_index_in_simdgroup] : 0.0f;
        t = simd_sum(t);
        if (thread_index_in_simdgroup == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs = metal::precise::rsqrt(red[0] / float(D) + EPS_VAL);
    for (int j = 0; j < (D + NW - 1) / NW; j++) {
        const int i = tid + j * NW;
        if (D % NW != 0 && i >= D) break;
        y[base + i] = bfloat16_t(sv[i] * rs * (OFFSET + float(w[i])));
    }
"""
NORM_WIDE = 1024  # threads per row of the wide variant (0: always the NT-thread kernel); used for <= 64 rows
_NORM_LOAD = "float v = float(x[base + i]);"
_NORM_LOAD_ADD = """const bfloat16_t hb = bfloat16_t(float(x[base + i]) + float(r[base + i]));
        h[base + i] = hb;
        float v = float(hb);"""
_norm_kernels = {}


def rms_norm(x, w, eps, offset: bool, r=None):
    """x [..., D] bf16 (+ r) -> (h = bf16(x + r) or None, y). offset: Gemma-style (1 + w)."""
    D = x.shape[-1]
    N = x.size // D
    add = r is not None
    nt = 256 if D >= 1024 else 32 * max(1, min(8, D // 32))
    nw = NORM_WIDE if (NORM_WIDE > nt and 1024 <= D <= 6144 and N <= 64) else 0
    key = (add, offset, eps, nw > 0)
    if key not in _norm_kernels:
        src = (_NORM_WIDE_SRC if nw else _NORM_SRC).replace("EPS_VAL", repr(float(eps)) + "f").replace(
            "LOAD", _NORM_LOAD_ADD if add else _NORM_LOAD)
        _norm_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_rmsnorm_{int(add)}{int(offset)}{int(nw > 0)}_{len(_norm_kernels)}",
            input_names=["x", "w"] + (["r"] if add else []), output_names=(["h"] if add else []) + ["y"],
            source=src)
    tg = nw or nt
    outs = _norm_kernels[key](
        inputs=[x, w] + ([r] if add else []),
        template=[("D", D), ("NT", nt), ("OFFSET", 1 if offset else 0)] + ([("NW", nw)] if nw else []),
        grid=(tg, N, 1), threadgroup=(tg, 1, 1),
        output_shapes=[x.shape] * (2 if add else 1), output_dtypes=[mx.bfloat16] * (2 if add else 1))
    return (outs[0], outs[1]) if add else (None, outs[0])


# ----------------------------------------------------------------------------- KDA conv + q/k l2 norm
# Depthwise causal conv (fp32 taps) + SiLU over the fused qkv channels, rounded to bf16 as the engine stores it; then
# per head of q and k: x / sqrt(sum(x^2) + 1e-6) (q also * K^-0.5), fp32. Dk / 32 channels per thread, so a
# simdgroup covers exactly one head.
_KDA_CONV_SRC = """
    const uint c4 = thread_position_in_grid.x;      // channel group (CPT channels)
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const int L = qkv_shape[1];
    const int C = qkv_shape[2];
    float y[CPT];
    float ss = 0.0f;
    for (int i = 0; i < CPT; i++) {
        const int c = c4 * CPT + i;
        float acc = 0.0f;
        for (int j = 0; j < KW; j++) {
            const int tt = int(t) + j - (KW - 1);       // input row relative to this chunk
            const float xv = tt < 0 ? float(state[(size_t(b) * (KW - 1) + (KW - 1 + tt)) * C + c])
                                    : float(qkv[(size_t(b) * L + tt) * C + c]);
            acc += xv * w[c * KW + j];
        }
        const float sv = float(bfloat16_t(acc / (1.0f + metal::precise::exp(-acc))));
        y[i] = sv;
        ss += sv * sv;
    }
    const int c0 = c4 * CPT;
    if (c0 < 2 * KD) {                                  // q or k head: l2 norm over the simdgroup's channels
        ss = simd_sum(ss);
        const float inv = 1.0f / metal::precise::sqrt(ss + 1e-6f);
        const bool isq = c0 < KD;
        const float sc = isq ? inv * QSCALE : inv;
        device float* dst = isq ? q : k;
        const int cc = isq ? c0 : c0 - KD;
        for (int i = 0; i < CPT; i++) dst[(size_t(b) * L + t) * KD + cc + i] = y[i] * sc;
    } else {
        const int cv = c0 - 2 * KD;
        for (int i = 0; i < CPT; i++) v[(size_t(b) * L + t) * (C - 2 * KD) + cv + i] = bfloat16_t(y[i]);
    }
"""
_kda_conv_kernels = {}


def kda_conv_qk(qkv, state, w, key_dim, head_k_dim):
    """qkv [B, L, C] bf16 (new rows), state [B, K-1, C] bf16 (previous rows), w [C, K] fp32 ->
    q, k [B, L, key_dim] fp32 (l2-normed, q * K^-0.5), v [B, L, C - 2 key_dim] bf16."""
    B, L, C = qkv.shape
    KW = w.shape[1]
    cpt = head_k_dim // 32
    assert head_k_dim % 32 == 0 and C % head_k_dim == 0 and key_dim % head_k_dim == 0
    if head_k_dim not in _kda_conv_kernels:
        _kda_conv_kernels[head_k_dim] = mx.fast.metal_kernel(
            name=f"volundr_kda_conv_{head_k_dim}", input_names=["qkv", "state", "w"], output_names=["q", "k", "v"],
            source=_KDA_CONV_SRC.replace("QSCALE", repr(float(head_k_dim) ** -0.5) + "f"))
    q, k, v = _kda_conv_kernels[head_k_dim](
        inputs=[qkv, state, w], template=[("KW", KW), ("KD", key_dim), ("CPT", cpt)],
        grid=(C // cpt, L, B), threadgroup=(min(256, C // cpt), 1, 1),
        output_shapes=[(B, L, key_dim), (B, L, key_dim), (B, L, C - 2 * key_dim)],
        output_dtypes=[F32, F32, mx.bfloat16])
    return q, k, v


# ----------------------------------------------------------------------------- KDA recurrence (per-channel gate)
# mlx-lm's packed gated-delta kernel (Copyright 2025 Apple Inc., MIT; mlx_lm/models/gated_delta.py) extended to a
# per-channel decay g [B, T, Hv, Dk]: four lanes per value row, 32 contiguous state elements per lane, eight rows per
# simdgroup, so a layer needs Hv * Dv / 8 simdgroups instead of Hv * Dv.
_GDN_VEC_SRC = r"""
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = Dk / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;

    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);

    auto lane = thread_index_in_simdgroup;
    auto row_in_simdgroup = lane / lanes_per_row;
    auto lane_in_row = lane & (lanes_per_row - 1);
    auto row_group = thread_position_in_grid.y;
    auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

    auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk + lane_in_row * values_per_lane;
    auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
    y += (b_idx * T * Hv + hv_idx) * Dv;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) state[i] = static_cast<float>(i_state[i]);
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
        float part[partials_per_lane];
        for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] * static_cast<float>(g_[e]);
                acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
        }
        float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);
        auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);
        for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] + static_cast<float>(k_[e]) * delta;
                acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
        }
        float out = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        if (lane_in_row == 0) y[dv_idx] = static_cast<InT>(out);
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        g_ += Hv * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        beta_ += Hv;
    }
    for (int i = 0; i < values_per_lane; ++i) o_state[i] = static_cast<StT>(state[i]);
"""
_gdn_vec_kernel = mx.fast.metal_kernel(
    name="volundr_gated_delta_packed_vec", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out"], source=_GDN_VEC_SRC)


def gated_delta_vec(q, k, v, g, beta, state, out_dtype=None):
    """q, k [B, T, Hk, 128] fp32 (normalized), v [B, T, Hv, Dv], g [B, T, Hv, 128] fp32 decay (not log), beta
    [B, T, Hv] fp32, state [B, Hv, Dv, 128] fp32 -> y [B, T, Hv, Dv] (out_dtype, default q's), new state.
    out_dtype=bf16 rounds y in the kernel exactly as y.astype(bf16) would, without the extra pass."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    assert Dk == 128 and Dv % 8 == 0
    ot = out_dtype or q.dtype
    return _gdn_vec_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", ot), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv // 8, B * Hv), threadgroup=(32, 2, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[ot, state.dtype])


# Same recurrence, one threadgroup per (batch, value head) holding all Dv rows (Dv / 8 simdgroups); TS timesteps of
# q, k, g, v, beta are staged in threadgroup memory per barrier, so the sequential loop reads on-chip memory instead
# of waiting on device loads every step.
_GDN_TG_SRC = r"""
    constexpr int lanes_per_row = 4;
    constexpr int values_per_lane = Dk / lanes_per_row;        // 32
    constexpr int NTH = (Dv / 8) * 32;                         // threads per threadgroup
    const uint tid = thread_position_in_threadgroup.x;
    const uint n = threadgroup_position_in_grid.y;             // b * Hv + hv
    const uint b_idx = n / Hv, hv_idx = n % Hv, hk_idx = hv_idx / (Hv / Hk);
    const uint lane = thread_index_in_simdgroup;
    const uint row = simdgroup_index_in_threadgroup * 8 + lane / lanes_per_row;   // dv
    const uint lane_in_row = lane & (lanes_per_row - 1);
    const uint e0 = lane_in_row * values_per_lane;

    // q, k, g rows are swizzled (element c at c + c / 32) so the 4 lanes of a value row, which read elements
    // e0 + e with e0 = 0, 32, 64, 96, hit 4 different banks instead of one
    constexpr int DKP = Dk + Dk / 32;
    threadgroup float sq[TS * DKP], sk[TS * DKP], sg[TS * DKP], sv[TS * Dv], sb[TS];

    auto i_state = state_in + (size_t(n) * Dv + row) * Dk + e0;
    auto o_state = state_out + (size_t(n) * Dv + row) * Dk + e0;
    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) state[i] = static_cast<float>(i_state[i]);

    for (int t0 = 0; t0 < T; t0 += TS) {
        const int ts = min(TS, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < ts * Dk; i += NTH) {
            const int s = i / Dk, c = i % Dk;
            const size_t qi = ((size_t(b_idx) * T + t0 + s) * Hk + hk_idx) * Dk + c;
            const size_t gi = ((size_t(b_idx) * T + t0 + s) * Hv + hv_idx) * Dk + c;
            const int si = s * DKP + c + (c >> 5);
            sq[si] = static_cast<float>(q[qi]);
            sk[si] = static_cast<float>(k[qi]);
            sg[si] = static_cast<float>(g[gi]);
        }
        for (int i = tid; i < ts * Dv; i += NTH) {
            const int s = i / Dv, c = i % Dv;
            sv[i] = static_cast<float>(v[((size_t(b_idx) * T + t0 + s) * Hv + hv_idx) * Dv + c]);
        }
        if (tid < uint(ts)) sb[tid] = static_cast<float>(beta[(size_t(b_idx) * T + t0 + tid) * Hv + hv_idx]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 0; s < ts; ++s) {
            const threadgroup float* q_ = sq + s * DKP + e0 + lane_in_row;
            const threadgroup float* k_ = sk + s * DKP + e0 + lane_in_row;
            const threadgroup float* g_ = sg + s * DKP + e0 + lane_in_row;
            float part[8];
            for (int pb = 0; pb < 8; ++pb) {
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const int e = pb * 4 + i;
                    state[e] = state[e] * g_[e];
                    acc += state[e] * k_[e];
                }
                part[pb] = acc;
            }
            float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
            kv_mem += simd_shuffle_xor(kv_mem, 1);
            kv_mem += simd_shuffle_xor(kv_mem, 2);
            const float delta = (sv[s * Dv + row] - kv_mem) * sb[s];
            for (int pb = 0; pb < 8; ++pb) {
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const int e = pb * 4 + i;
                    state[e] = state[e] + k_[e] * delta;
                    acc += state[e] * q_[e];
                }
                part[pb] = acc;
            }
            float out = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
            out += simd_shuffle_xor(out, 1);
            out += simd_shuffle_xor(out, 2);
            if (lane_in_row == 0) y[((size_t(b_idx) * T + t0 + s) * Hv + hv_idx) * Dv + row] = static_cast<InT>(out);
        }
    }
    for (int i = 0; i < values_per_lane; ++i) o_state[i] = static_cast<StT>(state[i]);
"""
_gdn_tg_kernel = mx.fast.metal_kernel(
    name="volundr_gated_delta_tg", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out"], source=_GDN_TG_SRC)


def gated_delta_tg(q, k, v, g, beta, state, ts=8, out_dtype=None):
    """As gated_delta_vec, staged through threadgroup memory (one threadgroup per value head)."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    assert Dk == 128 and Dv % 8 == 0 and Dv <= 256
    nth = (Dv // 8) * 32
    ot = out_dtype or q.dtype
    return _gdn_tg_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", ot), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("TS", ts)],
        grid=(nth, B * Hv, 1), threadgroup=(nth, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[ot, state.dtype])


# ----------------------------------------------------------------------------- BCSA window attention (NAX)
# MLX's head-dim-split NAX attention (steel_attention_nax.h, attention_nax_dsplit; Apple, MIT) with three changes:
# - the mask is BCSA's sliding window, computed from indices; only key blocks that intersect a query block's
#   window are visited;
# - bcsa_attend first runs the query block's far blocks through the same online softmax (_WIN_FAR_UNION), so the
#   result is the joint softmax over window + far blocks;
# - masked scores are -inf rather than the finite minimum, so a row whose block is entirely outside its window
#   adds nothing. The row max starts finite, so exp2(-inf - max) = 0, never NaN.
# The runtime compiler has no include path, so the NAX tile headers are inlined into the kernel header.
def _inline_mlx_header(rel, seen):
    import re
    root = Path(mx.__file__).parent / "include"
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (root / rel).read_text().splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_inline_mlx_header(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out)


_WIN_HEADER = None


def _win_header():
    global _WIN_HEADER
    if _WIN_HEADER is None:
        _WIN_HEADER = _inline_mlx_header("mlx/backend/metal/kernels/steel/attn/nax.h", set()) + r"""
using namespace mlx::steel;
struct VMaxOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return metal::max(x, y); } };
struct VSumOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x + y; } };
struct VMulOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x * y; } };
struct VExpSubOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return fast::exp2(x - y); } };
"""
    return _WIN_HEADER


_WIN_SRC = r"""
    using AccumType = float;
    constexpr short kU = 16;
    constexpr int TQ = BQ / (WM * kU);
    constexpr int TD = BD / kU;
    constexpr short TK = BK / kU;
    constexpr int TDh = TD / WN;
    constexpr int BDh = BD / WN;
    static_assert(TQ == 1 && TD % WN == 0 && TDh % 2 == 0 && TK % 2 == 0, "tile shape");
    constexpr float NEG_INF = -INFINITY;
    constexpr float FMIN = -3.4028234663852886e38f;

    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const int qL = dims[0], kL = dims[1], qoff = dims[2], WIN = dims[3];
    const int NQ_aligned = qL / BQ, qL_rem = qL - NQ_aligned * BQ;
    const int NK_aligned = kL / BK, kL_rem = kL - NK_aligned * BK;
    const bool align_K = kL_rem == 0;
    const bool is_last_q = int(tid.x) == NQ_aligned;
    const int h = tid.y;
    const int kvh = h / G;
    const int q0 = tid.x * BQ;

    const short row_group = simd_group_id / WN;
    const short d_group = simd_group_id % WN;
    const short tm = kU * TQ * row_group;
    const short lim_rows_q = qL_rem - tm;
    const short lim_rows_k = kL_rem;

    const device T* Q = q + (size_t(h) * qL + q0 + tm) * BD + d_group * BDh;
    const device T* K = k + size_t(kvh) * kL * BD + d_group * BDh;
    const device T* V = v + size_t(kvh) * kL * BD + d_group * BDh;
    device T* O = out + (size_t(h) * qL + q0 + tm) * BD + d_group * BDh;
    const float scale2 = scale[0] * 1.44269504089f;

    using otile_t = NAXTile<AccumType, TQ, TDh>;
    using stile_t = NAXTile<AccumType, TQ, TK>;
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
    const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;

    otile_t Otile;
    Otile.clear();
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = FMIN;
        sum_score[i] = 0.0f;
    }

    threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];

    NAXTile<T, 1, 1> Qtiles[TDh];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
        if (is_last_q) {
            Qtiles[id].load_rows(Q + id * kU, BD, lim_rows_q);
        } else {
            Qtiles[id].load(Q + id * kU, BD);
        }
    }

    // FAR_UNION_PHASE

    // key blocks that intersect the window of any query in this block: keys (r + qoff - WIN, r + qoff]
    const int q_last = min(q0 + BQ, qL) - 1;
    const int j_lo = max(0, q0 + qoff - WIN + 1);
    const int j_hi = min(kL - 1, q_last + qoff);
    const int kb_lo = j_lo / BK, kb_hi = j_hi / BK + 1;
    // rows of this simdgroup, for the cheap "block entirely inside the window" test
    const int r_lo = q0 + tm, r_hi = q0 + tm + kU - 1;

    K += size_t(kb_lo) * BK * BD;
    V += size_t(kb_lo) * BK * BD;
    for (int kb = kb_lo; kb < kb_hi; kb++) {
        const bool is_last_k = !align_K && kb == NK_aligned;
        stile_t Stile;
        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TDh; id++) {
                NAXTile<T, 2, 1> Ktile;
                const int K_load_off = ik * kU * BD + id * kU;
                if (is_last_k) {
                    Ktile.load_rows(K + K_load_off, BD, lim_rows_k - ik * kU);
                } else {
                    Ktile.load(K + K_load_off, BD);
                }
                stile_t::NAXFrag_t::mma(
                    Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{},
                    Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
            }
        }
        // sum the WN partial scores in a fixed order
        threadgroup AccumType* slot = s_xchg[row_group][d_group];
        const short base = short(simd_lane_id) * (TK * kEPF);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) slot[base + ik * kEPF + i] = s[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) s[i] = s_xchg[row_group][0][base + ik * kEPF + i];
            STEEL_PRAGMA_UNROLL
            for (short pg = 1; pg < WN; pg++) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < kEPF; i++) s[i] += s_xchg[row_group][pg][base + ik * kEPF + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) Stile.elems()[ii] *= scale2;

        const int c_lo = kb * BK, c_hi = kb * BK + BK - 1;
        const bool inside = c_hi <= r_lo + qoff && c_lo > r_hi + qoff - WIN && c_hi < kL;
        if (!inside) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                        const int r = r_lo + ii * stile_t::kFragRowsJump + sm;
                        const int c = c_lo + ik * kU + jj + sn;
                        const auto loc = ii * stile_t::kFragThrCols + jj;
                        const bool ok = c < kL && c <= r + qoff && c > r + qoff - WIN;
                        fg[loc] = ok ? fg[loc] : NEG_INF;
                    }
                }
            }
        }

        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<VMaxOp>(new_max);
        Stile.template row_bin_op<VExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<VSumOp>(sum_score);
        Otile.template row_bin_op<VMulOp>(factor);
        simdgroup_barrier(mem_flags::mem_none);

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDh; id += 2) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                NAXTile<T, 1, 2> Vtile;
                const int V_load_off = ik * kU * BD + id * kU;
                if (is_last_k) {
                    Vtile.load_rows(V + V_load_off, BD, lim_rows_k - ik * kU);
                } else {
                    Vtile.load(V + V_load_off, BD);
                }
                otile_t::NAXFrag_t::mma(
                    Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
                    Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
            }
        }
        K += BK * BD;
        V += BK * BD;
    }

    threadgroup_barrier(mem_flags::mem_none);
    metal::vec<AccumType, kRowsPT> rcp;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
    Otile.template row_bin_op<VMulOp>(rcp);
    if (is_last_q) {
        if (lim_rows_q > 0) Otile.store_rows(O, BD, lim_rows_q);
    } else {
        Otile.store(O, BD);
    }
"""
_win_kernels = {}


def bcsa_window(q, keys, vals, scale, qoff, window, bq=64, bk=32, wm=4, wn=2):
    """Sliding-window attention (B = 1): q [H, T, D]; keys/vals [KVH, nk, D], query i sitting at key index i + qoff,
    attending to keys (i + qoff - window, i + qoff] -> out [H, T, D]."""
    H, T, D = q.shape
    KVH, nk, _ = keys.shape
    key = (bq, bk, wm, wn)
    if key not in _win_kernels:
        _win_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_bcsa_window_{bq}_{bk}_{wm}_{wn}", input_names=["q", "k", "v", "scale", "dims"],
            output_names=["out"], source=_WIN_SRC.replace("    // FAR_UNION_PHASE\n", ""), header=_win_header())
    nth = wm * wn * 32
    dims = mx.array([T, nk, qoff, window], dtype=mx.int32)
    return _win_kernels[key](
        inputs=[q, keys, vals, mx.array([scale], dtype=F32), dims],
        template=[("T", q.dtype), ("BQ", bq), ("BK", bk), ("BD", D), ("WM", wm), ("WN", wn), ("G", H // KVH)],
        grid=(((T + bq - 1) // bq) * nth, H, 1), threadgroup=(nth, 1, 1),
        output_shapes=[(H, T, D)], output_dtypes=[q.dtype])[0]


# ----------------------------------------------------------------------------- KDA conv + q/k l2 norm, vectorized
# Same arithmetic as _KDA_CONV_SRC (CPT = 4 channels per thread), with 8-byte loads of 4 bf16 channels and TT
# timesteps per thread, so each input row is loaded once per thread instead of KW times per channel.
_KDA_CONV4_SRC = """
    const uint c4 = thread_position_in_grid.x;
    const uint tb = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const int L = qkv_shape[1];
    const int C = qkv_shape[2];
    const int c0 = c4 * 4;
    const int t0 = tb * TT;
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
_kda_conv4_kernels = {}


def kda_conv_qk4(qkv, state, w, key_dim, head_k_dim, tt=4):
    """As kda_conv_qk (head_k_dim == 128: one simdgroup of 4-channel threads per head)."""
    B, L, C = qkv.shape
    KW = w.shape[1]
    assert head_k_dim == 128 and C % 128 == 0 and key_dim % 128 == 0
    if tt not in _kda_conv4_kernels:
        _kda_conv4_kernels[tt] = mx.fast.metal_kernel(
            name=f"volundr_kda_conv4_{tt}", input_names=["qkv", "state", "w"], output_names=["q", "k", "v"],
            source=_KDA_CONV4_SRC.replace("QSCALE", repr(float(head_k_dim) ** -0.5) + "f"))
    return _kda_conv4_kernels[tt](
        inputs=[qkv, state, w], template=[("KW", KW), ("KD", key_dim), ("TT", tt)],
        grid=(C // 4, (L + tt - 1) // tt, B), threadgroup=(min(256, C // 4), 1, 1),
        output_shapes=[(B, L, key_dim), (B, L, key_dim), (B, L, C - 2 * key_dim)],
        output_dtypes=[F32, F32, mx.bfloat16])


# ----------------------------------------------------------------------------- KDA gated output norm
# w * rmsnorm(x) * silu(z) over each value head (Dv = 128), fp32 with one rounding; one simdgroup per row,
# 4 channels per lane.
_GATED_NORM_SRC = """
    const uint gid = thread_position_in_grid.x;
    const uint row = gid / 32;
    const uint lane = gid % 32;
    const size_t base = size_t(row) * 128 + lane * 4;
    const uint2 ux = *(const device uint2*)(x + base);
    const uint2 uz = *(const device uint2*)(z + base);
    float xv[4] = {as_type<float>(ux.x << 16), as_type<float>(ux.x & 0xFFFF0000u),
                   as_type<float>(ux.y << 16), as_type<float>(ux.y & 0xFFFF0000u)};
    float zv[4] = {as_type<float>(uz.x << 16), as_type<float>(uz.x & 0xFFFF0000u),
                   as_type<float>(uz.y << 16), as_type<float>(uz.y & 0xFFFF0000u)};
    float ss = 0.0f;
    for (int i = 0; i < 4; i++) ss += xv[i] * xv[i];
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / 128.0f + EPS_VAL);
    ushort o[4];
    for (int i = 0; i < 4; i++) {
        const float sg = zv[i] / (1.0f + metal::precise::exp(-zv[i]));
        o[i] = as_type<ushort>(bfloat16_t(float(w[lane * 4 + i]) * (xv[i] * inv) * sg));
    }
    *(device uint2*)(y + base) = uint2(uint(o[0]) | (uint(o[1]) << 16), uint(o[2]) | (uint(o[3]) << 16));
"""
_gated_norm_kernels = {}


def gated_norm(x, w, z, eps):
    """x, z [..., 128] bf16, w [128] -> bf16(w * rmsnorm(x) * silu(z))."""
    assert x.shape[-1] == 128 and x.dtype == mx.bfloat16 and z.dtype == mx.bfloat16
    rows = x.size // 128
    if eps not in _gated_norm_kernels:
        _gated_norm_kernels[eps] = mx.fast.metal_kernel(
            name=f"volundr_gated_norm_{len(_gated_norm_kernels)}", input_names=["x", "w", "z"], output_names=["y"],
            source=_GATED_NORM_SRC.replace("EPS_VAL", repr(float(eps)) + "f"))
    return _gated_norm_kernels[eps](
        inputs=[x, w, z], grid=(rows * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.bfloat16])[0]


# ----------------------------------------------------------------------------- BCSA far-block union per query block
# For each block of BQ consecutive queries: the ascending union of their selected far blocks (uidx, ucnt) and each
# query's membership bits over that union (umask), so attention can run the union as dense NAX tiles and mask per
# query. The union bitmap lives in threadgroup memory (NBW words cover the valid blocks).
_UNION_SRC = """
    const uint qb = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int T = sel_shape[0];
    const int K = sel_shape[1];
    const int UW = UMAX / 32;
    threadgroup atomic_uint bm[NBW];
    threadgroup uint offs[NBW];
    threadgroup uint part[NTH];
    for (int w = tid; w < NBW; w += NTH) atomic_store_explicit(&bm[w], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int q0 = qb * BQ;
    const int nq = min(BQ, T - q0);
    for (int e = tid; e < nq * K; e += NTH) {
        const int idx = sel[(size_t(q0) * K) + e];
        if (idx >= 0) atomic_fetch_or_explicit(&bm[idx >> 5], 1u << (idx & 31), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // exclusive scan of popcounts: WPT consecutive words per thread
    constexpr int WPT = (NBW + NTH - 1) / NTH;
    uint local = 0;
    for (int i = 0; i < WPT; i++) {
        const int w = tid * WPT + i;
        if (w < NBW) local += popcount(atomic_load_explicit(&bm[w], memory_order_relaxed));
    }
    part[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint run = 0;
        for (int i = 0; i < NTH; i++) { const uint p = part[i]; part[i] = run; run += p; }
        ucnt[qb] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint run = part[tid];
    for (int i = 0; i < WPT; i++) {
        const int w = tid * WPT + i;
        if (w < NBW) {
            const uint bits = atomic_load_explicit(&bm[w], memory_order_relaxed);
            offs[w] = run;
            uint b = bits;
            uint r = run;
            while (b) {
                const int j = ctz(b);
                uidx[size_t(qb) * UMAX + r] = w * 32 + j;
                r++;
                b &= b - 1;
            }
            run += popcount(bits);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < nq * K; e += NTH) {
        const int idx = sel[(size_t(q0) * K) + e];
        if (idx >= 0) {
            const uint bits = atomic_load_explicit(&bm[idx >> 5], memory_order_relaxed);
            const uint pos = offs[idx >> 5] + popcount(bits & ((1u << (idx & 31)) - 1u));
            const int row = e / K;
            atomic_fetch_or_explicit((device atomic_uint*)&umask[(size_t(qb) * BQ + row) * UW + (pos >> 5)],
                                     1u << (pos & 31), memory_order_relaxed);
        }
    }
"""
_union_kernels = {}


def far_union(sel, nv_max, bq):
    """sel [T, K] int32 (-1 padded) -> uidx [NQB, UMAX] int32 (0-padded), ucnt [NQB] int32, umask [NQB, BQ, UMAX/32]
    uint32. UMAX bounds the union: min(BQ * K, nv_max) rounded up to 32."""
    T, K = sel.shape
    nqb = (T + bq - 1) // bq
    nbw = 64
    while nbw * 32 < nv_max:
        nbw *= 2
    assert nbw <= 4096, "union bitmap exceeds threadgroup memory"
    umax = (min(bq * K, nv_max) + 31) // 32 * 32
    nth = 256
    key = (bq, nbw, umax)
    if key not in _union_kernels:
        _union_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_far_union_{bq}_{nbw}_{umax}", input_names=["sel"], output_names=["uidx", "ucnt", "umask"],
            source=_UNION_SRC)
    return _union_kernels[key](
        inputs=[sel], template=[("BQ", bq), ("NBW", nbw), ("UMAX", umax), ("NTH", nth)],
        grid=(nqb * nth, 1, 1), threadgroup=(nth, 1, 1),
        output_shapes=[(nqb, umax), (nqb,), (nqb, bq, umax // 32)],
        output_dtypes=[mx.int32, mx.int32, mx.uint32], init_value=0)


# Far phase for the window kernel: the union of the query block's selected far blocks (far_union) as dense NAX tiles,
# rows gathered from the pooled K/V, each query masked to its own selection. The softmax state it leaves is where the
# window loop starts.
_WIN_FAR_UNION = r"""
    {
        const int UMAX = dims[4];
        const int UW = UMAX / 32;
        const int CAP = pk_shape[1];
        const int ucount = ucnt[tid.x];
        const device int* U = uidx + size_t(tid.x) * UMAX;
        const device T* PK = pk + size_t(kvh) * CAP * BD + d_group * BDh;
        const device T* PV = pv + size_t(kvh) * CAP * BD + d_group * BDh;
        const device uint* M0 = umask + (size_t(tid.x) * BQ + tm + sm) * UW;   // row sm of this simdgroup
        const device uint* M1 = M0 + 8 * UW;                                   // row sm + 8
        for (int u0 = 0; u0 < ucount; u0 += BK) {
            // gathered rows: frag r of the BK batch holds union entries u0 + r * 16 + {sm, sm + 8}
            const device T* rk[TK][2];
            STEEL_PRAGMA_UNROLL
            for (short r = 0; r < TK; r++) {
                rk[r][0] = PK + size_t(U[u0 + r * kU + sm]) * BD + sn;
                rk[r][1] = PK + size_t(U[u0 + r * kU + sm + 8]) * BD + sn;
            }
            stile_t Stile;
            Stile.clear();
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik += 2) {
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < TDh; id++) {
                    NAXTile<T, 2, 1> Ktile;
                    STEEL_PRAGMA_UNROLL
                    for (short r = 0; r < 2; r++) {
                        thread auto& fk = Ktile.frag_at(r, 0);
                        STEEL_PRAGMA_UNROLL
                        for (short i = 0; i < 2; i++) {
                            STEEL_PRAGMA_UNROLL
                            for (short j = 0; j < 4; j++) fk[i * 4 + j] = rk[ik + r][i][id * kU + j];
                        }
                    }
                    stile_t::NAXFrag_t::mma(
                        Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{},
                        Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
                }
            }
            threadgroup AccumType* slot = s_xchg[row_group][d_group];
            const short base = short(simd_lane_id) * (TK * kEPF);
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                thread auto& s = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < kEPF; i++) slot[base + ik * kEPF + i] = s[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                thread auto& s = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < kEPF; i++) s[i] = s_xchg[row_group][0][base + ik * kEPF + i];
                STEEL_PRAGMA_UNROLL
                for (short pg = 1; pg < WN; pg++) {
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < kEPF; i++) s[i] += s_xchg[row_group][pg][base + ik * kEPF + i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // each query keeps only its own selected blocks (bits of the 32-entry batch, BK == 32)
            const uint mb[2] = {M0[u0 >> 5], M1[u0 >> 5]};
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short ii = 0; ii < 2; ii++) {
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < 4; jj++) {
                        const auto loc = ii * 4 + jj;
                        const bool ok = (mb[ii] >> (ik * kU + sn + jj)) & 1u;
                        fg[loc] = ok ? fg[loc] * scale2 : NEG_INF;
                    }
                }
            }
            metal::vec<AccumType, kRowsPT> new_max;
            metal::vec<AccumType, kRowsPT> factor;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
            Stile.template row_reduce<VMaxOp>(new_max);
            Stile.template row_bin_op<VExpSubOp>(new_max);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kRowsPT; ++i) {
                factor[i] = fast::exp2(max_score[i] - new_max[i]);
                max_score[i] = new_max[i];
                sum_score[i] = sum_score[i] * factor[i];
            }
            Stile.template row_reduce<VSumOp>(sum_score);
            Otile.template row_bin_op<VMulOp>(factor);
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TDh; id += 2) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    NAXTile<T, 1, 2> Vtile;
                    STEEL_PRAGMA_UNROLL
                    for (short c = 0; c < 2; c++) {
                        thread auto& fv = Vtile.frag_at(0, c);
                        STEEL_PRAGMA_UNROLL
                        for (short i = 0; i < 2; i++) {
                            const device T* rv = PV + (rk[ik][i] - PK) + (id + c) * kU;
                            STEEL_PRAGMA_UNROLL
                            for (short j = 0; j < 4; j++) fv[i * 4 + j] = rv[j];
                        }
                    }
                    otile_t::NAXFrag_t::mma(
                        Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
                        Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
                }
            }
        }
    }
"""
_attend_kernels = {}


def bcsa_attend(q, keys, vals, pk, pv, sel, nv_max, scale, qoff, window, bq=32, bk=32, wm=2, wn=2):
    """BCSA prefill attention in two kernels (B = 1): the union of each query block's selected far blocks, then one
    NAX flash-attention pass over that union (masked per query) followed by the sliding window.
    q [H, T, D]; keys/vals [KVH, nk, D] with query i at key index i + qoff; pk/pv [KVH, cap, D] pooled blocks;
    sel [T, topk] int32 ascending, -1 padded -> out [H, T, D]."""
    H, T, D = q.shape
    KVH, nk, _ = keys.shape
    assert bk == 32 and bq == wm * 16
    uidx, ucnt, umask = far_union(sel, nv_max, bq)
    key = (bq, bk, wm, wn)
    if key not in _attend_kernels:
        src = _WIN_SRC.replace("    // FAR_UNION_PHASE\n", _WIN_FAR_UNION)
        _attend_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_bcsa_attend_{bq}_{bk}_{wm}_{wn}",
            input_names=["q", "k", "v", "scale", "dims", "pk", "pv", "uidx", "ucnt", "umask"],
            output_names=["out"], source=src, header=_win_header())
    nth = wm * wn * 32
    dims = mx.array([T, nk, qoff, window, uidx.shape[1]], dtype=mx.int32)
    return _attend_kernels[key](
        inputs=[q, keys, vals, mx.array([scale], dtype=F32), dims, pk, pv, uidx, ucnt, umask],
        template=[("T", q.dtype), ("BQ", bq), ("BK", bk), ("BD", D), ("WM", wm), ("WN", wn), ("G", H // KVH)],
        grid=(((T + bq - 1) // bq) * nth, H, 1), threadgroup=(nth, 1, 1),
        output_shapes=[(H, T, D)], output_dtypes=[q.dtype])[0]


# ----------------------------------------------------------------------------- BCSA indexer block scores (NAX)
# s[t, j] = bf16(bf16(mean_h bf16(q[t, h] . bk[j])) * scale), torch's rounding points (bf16 einsum -> fp32 mean ->
# bf16 -> * scale -> bf16), without materializing the per-head scores. A simdgroup computes 16 queries x 32 blocks
# for all IH heads; the threadgroup's 4 simdgroups take 64 consecutive queries against the same 32 blocks.
_SCORES_SRC = r"""
    constexpr short kU = 16;
    constexpr int ID = 128;
    constexpr int TDD = ID / kU;
    const uint3 tid = threadgroup_position_in_grid;
    const uint sgi = simdgroup_index_in_threadgroup;
    const int T = dims[0], NB = dims[1];
    const int j0 = tid.x * 32;
    const int t0 = tid.y * 64 + sgi * kU;
    if (t0 >= T) return;
    const short rows_q = min(int(kU), T - t0);
    using stile_t = NAXTile<float, 1, 2>;
    stile_t S[IH];
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < IH; h++) S[h].clear();
    const device T_IN* Qb = qi + size_t(t0) * IH * ID;
    const device T_IN* Kb = bk + size_t(j0) * ID;
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < TDD; d++) {
        NAXTile<T_IN, 2, 1> Ktile;
        Ktile.load(Kb + d * kU, ID);
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < IH; h++) {
            NAXTile<T_IN, 1, 1> Qtile;
            if (rows_q < kU) {
                Qtile.load_rows(Qb + h * ID + d * kU, IH * ID, rows_q);
            } else {
                Qtile.load(Qb + h * ID + d * kU, IH * ID);
            }
            stile_t::NAXFrag_t::mma(S[h].frag_at(0, 0), S[h].frag_at(0, 1), Qtile.frag_at(0, 0), metal::false_type{},
                                    Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
    }
    const short2 sc = stile_t::NAXFrag_t::get_coord();
    const float scale = scale_in[0];
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < 2; f++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const int t = t0 + sc.y + i * 8;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 4; jj++) {
                const int j = j0 + f * kU + sc.x + jj;
                float acc = 0.0f;
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < IH; h++) acc += float(bfloat16_t(S[h].frag_at(0, f)[i * 4 + jj]));
                const float m = float(bfloat16_t(acc / float(IH)));
                if (t < T && j < NB) s[size_t(t) * NB + j] = bfloat16_t(m * scale);
            }
        }
    }
"""
_scores_kernels = {}


def bcsa_scores(qi, bk, nb, scale):
    """qi [T, IH, 128] bf16 (normed indexer queries), bk [cap, 128] bf16 (pooled indexer keys; cap a multiple of 32,
    rows past nb unread in the result) -> s [T, nb] bf16."""
    T, IH, ID = qi.shape
    assert ID == 128 and bk.shape[0] % 32 == 0 and nb <= bk.shape[0]
    if IH not in _scores_kernels:
        _scores_kernels[IH] = mx.fast.metal_kernel(
            name=f"volundr_bcsa_scores_{IH}", input_names=["qi", "bk", "scale_in", "dims"], output_names=["s"],
            source=_SCORES_SRC, header=_win_header())
    return _scores_kernels[IH](
        inputs=[qi, bk, mx.array([scale], dtype=F32), mx.array([T, nb], dtype=mx.int32)],
        template=[("T_IN", qi.dtype), ("IH", IH)],
        grid=(((nb + 31) // 32) * 128, (T + 63) // 64, 1), threadgroup=(128, 1, 1),
        output_shapes=[(T, nb)], output_dtypes=[qi.dtype])[0]
