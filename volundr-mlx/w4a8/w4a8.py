"""W4A8 prefill matmul for MLX affine 4-bit g128 weights on the M5 neural accelerators (spike).

y[m, n] = sum_k x[m, k] * w[n, k] with w = s[n, g] * q[n, k] + b[n, g] (MLX layout, q in 0..15), computed as

  y = sum_g  s[n, g] * (sx[m, g] * D_g[m, n] + d8[m, g])  +  b[n, g] * xs[m, g]

  xq = rint(x / sx) in int8 per (row, 128-group), sx = absmax / 127   (act_quant)
  D_g = sum_{k in g} xq[m, k] * q[n, k]                                (int8 x int8 -> int32 on the accelerator)
  xs = sum_{k in g} x[m, k] (exact fp32), d8 = 8 * (xs - sx * sum_{k in g} xq[m, k])

The d8 term turns the uncentred codes into centred ones (q - 8) without touching the weights, so the activation
rounding error enters as s * sum_k e_k (q_k - 8) rather than s * sum_k e_k q_k. Codes stay unsigned in the
register tile (0..15 fits int8), so the unpack is one shift and one mask per four weights.

Layouts:
- MLX packs nibbles of k = 0..7 in one uint32; (w >> 4 * sel) & 0x0F0F0F0F yields the even (sel 0) or odd (sel 1)
  nibbles as four int8 codes. act_quant stores each 8-chunk of activations permuted as k = (0, 2, 4, 6, 1, 3, 5, 7),
  so the four bytes a lane holds line up with the four codes it unpacks.
- act_quant writes xq lane-major: per 32-row block and 32-k step a 1 KB block in which lane L's eight words are
  contiguous and already in the left cooperative tensor's storage order (two 16-byte loads per lane and tile). The
  per-(row, group) params (sx, d8, xs) are stored the same way: a lane's four rows are one contiguous 64 bytes.
- The cooperative-tensor layout (probe_layout.py: MLX's 16x16 NAX fragment layout, frags ordered differently for
  the right operand) is implementation-defined; check.py tests the kernel against an emulation.

Default GEMM (what w4a8_hook.py runs, "32x32x2x2RV" in bench_w4a8.py): one 32 x 32 accelerator tile per simdgroup,
2 x 2 simdgroups, A from the lane-major layout (two 16-byte loads), B gathered per lane straight from the MLX
packing (eight 4-byte loads per k step), both prefetched one k step ahead, int32 partials scaled into fp32 once per
group. The other switches (BTG threadgroup staging, BROW row-chunk B loads with act_quant perm 1, SBT transposed
scales, PLAIN register tiles, larger SM / SN) are kept for the record in NOTES.md; none beat the default.
Finer activation groups (act_quant(ag=32 / 64), w4a8_matmul(ag=1..5)) are correct but slower than stock: reading
the int32 tile after every 32- or 64-k op is expensive on this accelerator (NOTES.md, "Addendum").
"""
import mlx.core as mx

GS = 128
MPP_HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""

# --------------------------------------------------------------------------------------------- activation quant
# One simdgroup per (row, group). Lane L owns the physical positions 4L..4L+3 of the group, which hold the logical
# elements 8 * (L / 2) + 2 * j + (L % 2), j = 0..3; the lane pair (L, L ^ 1) loads that 8-chunk once each and picks
# its half. Physical position 4L sits at k step ks = L / 8, fc = (L / 4) % 2, fn = 4 * (L % 4) of the GEMM's left tile.
# Rows >= M (padding up to the GEMM tile) are zero.
_AQ_SRC = """
  const uint lane = thread_index_in_simdgroup;
  const int g = int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup);
  const int row = int(threadgroup_position_in_grid.y);
  if (g >= G) return;
  float v[4];
  if (row < MROWS) {
    const uint4 c = ((const device uint4*)((const device bfloat*)x + int64_t(row) * K + g * 128))[lane >> 1];
    const uint sh = (lane & 1) ? 16 : 0;
    v[0] = as_type<float>(((c.x >> sh) & 0xFFFFu) << 16);
    v[1] = as_type<float>(((c.y >> sh) & 0xFFFFu) << 16);
    v[2] = as_type<float>(((c.z >> sh) & 0xFFFFu) << 16);
    v[3] = as_type<float>(((c.w >> sh) & 0xFFFFu) << 16);
  } else {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 4; j++) v[j] = 0.0f;
  }
  float amax = max(max(fabs(v[0]), fabs(v[1])), max(fabs(v[2]), fabs(v[3])));
  float s = (v[0] + v[1]) + (v[2] + v[3]);
  amax = simd_max(amax);
  s = simd_sum(s);
  const float sx = amax / 127.0f;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int qs = 0;
  uint word = 0;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; j++) {
    const int q = clamp(int(rint(v[j] * inv)), -127, 127);
    qs += q;
    word |= (uint(q) & 0xFFu) << (8 * j);
  }
  qs = simd_sum(qs);
  // lane-major destination: block (row / 32, k step), lane Ld(fm, fn), word j(r, fc)
  const int mb = row >> 5, ml = row & 31, fm = ml & 7, r = ml >> 3;
#if PERM
  // tile k = 32 ks + 16 fc + fn + c  <->  physical k = 32 (fn / 4) + 8 ks + 2 c + fc
  const int ks = g * 4 + (int(lane >> 1) & 3), fc = int(lane) & 1, fn = 4 * int(lane >> 3);
#else
  const int ks = g * 4 + int(lane >> 3), fc = int(lane >> 2) & 1, fn = 4 * (int(lane) & 3);
#endif
  const int ld = ((fm >> 2) << 4) | (((fn >> 3) & 1) << 3) | (((fm >> 1) & 1) << 2) | ((fm & 1) << 1) | ((fn >> 2) & 1);
  const int j = 4 * (r >> 1) + 2 * fc + (r & 1);
  ((device uint*)xq)[((int64_t(mb) * (K / 32) + ks) * 32 + ld) * 8 + j] = word;
  if (lane == 0) {
    ((device float4*)prm)[((int64_t(mb) * G + g) * 8 + fm) * 4 + r] = float4(sx, 8.0f * (s - sx * float(qs)), s, 0.0f);
  }
"""
_aq_kernels = {p: mx.fast.metal_kernel(name=f"w4a8_act_quant_v3_p{p}", input_names=["x"], output_names=["xq", "prm"],
                                       source=f"#define PERM {p}\n" + _AQ_SRC) for p in (0, 1)}


# 32-wide activation groups (AG32): with PERM 0 the lanes 8j..8j+7 hold logical k 32j..32j+31, which is exactly
# the GEMM's k step j of the group, so each k step gets its own scale sx_j (absmax over 8 lanes). Params per
# (row, group): (sx_0, sx_1, sx_2, sx_3) and (d8, xs, 0, 0), d8 = 8 * (xs - sum_j sx_j * sum_{k in j} xq), stored
# lane-major as two float4.
_AQ32_SRC = """
  const uint lane = thread_index_in_simdgroup;
  const int g = int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup);
  const int row = int(threadgroup_position_in_grid.y);
  if (g >= G) return;
  float v[4];
  if (row < MROWS) {
    const uint4 c = ((const device uint4*)((const device bfloat*)x + int64_t(row) * K + g * 128))[lane >> 1];
    const uint sh = (lane & 1) ? 16 : 0;
    v[0] = as_type<float>(((c.x >> sh) & 0xFFFFu) << 16);
    v[1] = as_type<float>(((c.y >> sh) & 0xFFFFu) << 16);
    v[2] = as_type<float>(((c.z >> sh) & 0xFFFFu) << 16);
    v[3] = as_type<float>(((c.w >> sh) & 0xFFFFu) << 16);
  } else {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 4; j++) v[j] = 0.0f;
  }
  float amax = max(max(fabs(v[0]), fabs(v[1])), max(fabs(v[2]), fabs(v[3])));
  float s = (v[0] + v[1]) + (v[2] + v[3]);
  amax = max(amax, simd_shuffle_xor(amax, 1));
  amax = max(amax, simd_shuffle_xor(amax, 2));
  amax = max(amax, simd_shuffle_xor(amax, 4));
#if AGW == 64
  amax = max(amax, simd_shuffle_xor(amax, 8));
#endif
  s = simd_sum(s);
  const float sx = amax / 127.0f;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int qs = 0;
  uint word = 0;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; j++) {
    const int q = clamp(int(rint(v[j] * inv)), -127, 127);
    qs += q;
    word |= (uint(q) & 0xFFu) << (8 * j);
  }
  qs += simd_shuffle_xor(qs, 1);
  qs += simd_shuffle_xor(qs, 2);
  qs += simd_shuffle_xor(qs, 4);
#if AGW == 64
  qs += simd_shuffle_xor(qs, 8);
#endif
  constexpr uint SUBL = AGW / 4 - 1;  // lanes per subgroup - 1
  const float t = simd_sum((lane & SUBL) == 0 ? sx * float(qs) : 0.0f);
  const int mb = row >> 5, ml = row & 31, fm = ml & 7, r = ml >> 3;
  const int ks = g * 4 + int(lane >> 3), fc = int(lane >> 2) & 1, fn = 4 * (int(lane) & 3);
  const int ld = ((fm >> 2) << 4) | (((fn >> 3) & 1) << 3) | (((fm >> 1) & 1) << 2) | ((fm & 1) << 1) | ((fn >> 2) & 1);
  const int j = 4 * (r >> 1) + 2 * fc + (r & 1);
  ((device uint*)xq)[((int64_t(mb) * (K / 32) + ks) * 32 + ld) * 8 + j] = word;
  device float* pp = (device float*)prm + (((int64_t(mb) * G + g) * 8 + fm) * 4 + r) * 8;
  if ((lane & SUBL) == 0) pp[lane / (SUBL + 1)] = sx;
  if (lane == 0) {
    pp[4] = 8.0f * (s - t);
    pp[5] = s;
    pp[6] = 0.0f;
    pp[7] = 0.0f;
  }
"""
_aq_sub_kernels = {w: mx.fast.metal_kernel(name=f"w4a8_act_quant_ag{w}", input_names=["x"],
                                            output_names=["xq", "prm"], source=f"#define AGW {w}\n" + _AQ32_SRC)
                   for w in (32, 64)}


def act_quant(x: mx.array, mpad: int = 128, perm: int = 0, ag: int = 128):
    """x [M, K] bf16 -> (xq [Mp, K] int8, prm) in the lane-major layouts above, Mp = M up to a multiple of mpad
    (>= 32). ag = 128: prm [Mp, G, 4] fp32 (sx, d8, xs, 0); ag = 32 / 64 (perm 0 only): prm [Mp, G, 8]
    (sx_0..sx_3 (64: sx_0, sx_1, -, -), d8, xs, 0, 0)."""
    M, K = x.shape
    G = K // GS
    Mp = -(-M // mpad) * mpad
    nsg = 4
    if ag in (32, 64):
        assert perm == 0, "sub-group scales need the 8-chunk layout (perm 0)"
        kern, pw = _aq_sub_kernels[ag], 8
    else:
        assert ag == 128
        kern, pw = _aq_kernels[perm], 4
    xq, prm = kern(inputs=[x], template=[("K", K), ("G", G), ("MROWS", M), ("NSG", nsg)],
                   grid=(-(-G // nsg) * nsg * 32, Mp, 1), threadgroup=(nsg * 32, 1, 1),
                   output_shapes=[(Mp, K), (Mp, G, pw)], output_dtypes=[mx.int8, mx.float32])
    return xq, prm


# --------------------------------------------------------------------------------------------- GEMM
# Per simdgroup an SM x SN output tile = (SM/32) x (SN/32) accelerator tiles of 32 x 32 (K step 32, both operands in
# registers). WM x WN simdgroups per threadgroup. Lane coordinates (fm, fn) as in MLX's BaseNAXFrag.
#   left  word j (0..7):  m = fm + 8 * r, k = fn + 16 * fc,  j = 4 * (r / 2) + 2 * fc + (r % 2)
#   right word j (0..7):  n = fm + 8 * r, k = fn + 16 * fc,  j = 4 * fc + r
#   dest element i:       m = fm + 8 * r, n = fn + 16 * fc + c, r = 2 * (i / 16) + (i / 4) % 2, fc = (i / 8) % 2
# BTG 1 stages each group's packed weights (BN rows x 64 bytes, row stride RS words) and its scales / biases in
# threadgroup memory, double-buffered, loaded with coalesced 16-byte reads one group ahead; BTG 0 gathers the words
# from device memory per lane.
_GEMM_HEAD = """
  constexpr int G = K / 128;
  constexpr int KW = K / 8;  // uint32 words per weight row
  constexpr int KS = K / 32;  // k steps
  constexpr int BN = SN * WN;
  constexpr int RS = 20;      // words per staged row (16 + 4 pad: conflict-free lane reads, 16-byte aligned)
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
#if MFAST
  const int tgm = int(threadgroup_position_in_grid.x), tgn = int(threadgroup_position_in_grid.y);
#else
  const int tgn = int(threadgroup_position_in_grid.x), tgm = int(threadgroup_position_in_grid.y);
#endif
  const int m0 = tgm * (SM * WM) + int(sg / WN) * SM;
  const int nl0 = int(sg % WN) * SN;  // first column of this simdgroup within the threadgroup tile
  const int nb = tgn * BN;
  const int n0 = nb + nl0;
  const int qid = int(lane >> 2);
  const int fm = (qid & 4) | (int(lane >> 1) & 3);
  const int fn = ((qid & 2) | int(lane & 1)) * 4;
  const uint sh = uint(fn & 4);  // 0: even nibbles, 4: odd nibbles
  const int kw = fn >> 3;
  const device uint4* xq4 = (const device uint4*)xq;
  const device float4* prm4 = (const device float4*)prm;

  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      32, 32, 32, false, true, false,
      (AG && AG != 5 && !AGACC) ? tensor_ops::matmul2d_descriptor::mode::multiply
                     : tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using a_t = decltype(op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>());
  using b_t = decltype(op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>());
  using d_t = decltype(op.template get_destination_cooperative_tensor<a_t, b_t, int32_t>());
#if BTG
  constexpr int NT = WM * WN * 32;
  constexpr int NV = BN * 4;  // 16-byte vectors per staged group
  static_assert(NV % NT == 0, "staging split");
  threadgroup uint Bs[2][BN * RS];
  threadgroup float Ss[2][BN], Sb[2][BN];
  const int tid = int(thread_index_in_threadgroup);
  uint4 bv[NV / NT];
  auto fetch = [&](int g) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NV / NT; i++) {
      const int u = tid + NT * i;
      bv[i] = ((const device uint4*)(w + int64_t(nb + (u >> 2)) * KW + g * 16))[u & 3];
    }
  };
  float sv = 0.0f, bvv = 0.0f;
  auto fetch_s = [&](int g) {
    if (tid < BN) { sv = float(scales[(nb + tid) * G + g]); bvv = float(biases[(nb + tid) * G + g]); }
  };
  auto stage = [&](int buf) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NV / NT; i++) {
      const int u = tid + NT * i;
      *(threadgroup uint4*)&Bs[buf][(u >> 2) * RS + 4 * (u & 3)] = bv[i];
    }
    if (tid < BN) { Ss[buf][tid] = sv; Sb[buf][tid] = bvv; }
  };
  fetch(0);
  fetch_s(0);
  stage(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
"""
# per accelerator tile code, {tm} / {tn} substituted by _gemm_src
_DECL_A = """
  a_t a{tm} = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  const device uint4* xr{tm} = xq4 + (int64_t((m0 + 32 * {tm}) >> 5) * KS * 32 + lane) * 2;
  const device float4* pr{tm}p = prm4 + (int64_t((m0 + 32 * {tm}) >> 5) * G * 8 + fm) * 4;
#if PREFETCH
  uint4 an{tm}[PFD][2];
#endif
"""
# prefetch: fetch step `st` (global k step index) into the next-registers
_PF_A = """
      if (!NO_ALOAD || st == 0) {{
        an{tm}[st % PFD][0] = xr{tm}[st * 64];
        an{tm}[st % PFD][1] = xr{tm}[st * 64 + 1];
      }}
"""
_PF_B = """
      if (!NO_BLOAD || st == 0) {{
#if BVEC
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) bn{tn}[st % PFD][r] = ((const device uint4*)wr{tn}[r])[st];
#else
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++)
        _Pragma("clang loop unroll(full)")
        for (int fc = 0; fc < 2; fc++) bn{tn}[st % PFD][4 * fc + r] = wr{tn}[r][st * 4 + 2 * fc];
#endif
      }}
"""
# move the prefetched registers into the cooperative tensors (B unpacked)
_MV_A = """
      {{
        thread uint4* aw = (thread uint4*)&a{tm}[0];
        aw[0] = an{tm}[ks % PFD][0];
        aw[1] = an{tm}[ks % PFD][1];
      }}
"""
_MV_B = """
      {{
        thread uint* bw = (thread uint*)&b{tn}[0];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; r++)
          _Pragma("clang loop unroll(full)")
          for (int fc = 0; fc < 2; fc++) {{
#if BVEC
            const uint4 bq = bn{tn}[ks % PFD][r];
            const uint word = fc ? (kw ? bq.w : bq.z) : (kw ? bq.y : bq.x);
#else
            const uint word = bn{tn}[ks % PFD][4 * fc + r];
#endif
            bw[4 * fc + r] = NO_UNPACK ? word : (word >> sh) & 0x0F0F0F0Fu;
          }}
      }}
"""
_DECL_B = """
  b_t b{tn} = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
#if !BTG
  const device uint* wr{tn}[4];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < 4; r++) wr{tn}[r] = w + int64_t((n0 + 32 * {tn} + fm + 8 * r) & (B_HOT ? 255 : 0x7fffffff)) * KW + (BVEC ? 0 : kw);
#endif
#if BROW
  const device uint4* wq{tn}[4];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < 4; r++) wq{tn}[r] = (const device uint4*)(w + int64_t(n0 + 32 * {tn} + fm + 8 * r) * KW) + (fn >> 2);
  uint4 qc{tn}[4], qn{tn}[4];
#endif
#if PREFETCH
#if BVEC
  uint4 bn{tn}[PFD][4];
#else
  uint bn{tn}[PFD][8];
#endif
#endif
"""
_DECL_D = """
  d_t d{tm}{tn} = op.template get_destination_cooperative_tensor<a_t, b_t, int32_t>();
  float acc{tm}{tn}[32];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 32; i++) acc{tm}{tn}[i] = 0.0f;
"""
_ZERO_D = """
    if (!NO_EPI || g == 0)
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 32; i++) d{tm}{tn}[i] = 0;
"""
_LOAD_A = """
      if (!NO_ALOAD) {{
        thread uint4* aw = (thread uint4*)&a{tm}[0];
        const device uint4* p = xr{tm} + (g * 4 + ks) * 64;
        aw[0] = p[0];
        aw[1] = p[1];
      }}
"""
_LOAD_B = """
      if (!NO_BLOAD) {{
        thread uint* bw = (thread uint*)&b{tn}[0];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; r++)
          _Pragma("clang loop unroll(full)")
          for (int fc = 0; fc < 2; fc++) {{
#if BTG
            const uint word = Bs[g & 1][(nl0 + 32 * {tn} + fm + 8 * r) * RS + ks * 4 + 2 * fc + kw];
#else
            const uint word = wr{tn}[r][g * 16 + ks * 4 + 2 * fc];
#endif
            bw[4 * fc + r] = NO_UNPACK ? word : (word >> sh) & 0x0F0F0F0Fu;
          }}
      }}
"""
# BROW: next group's chunks (prefetched a group ahead), and the per-k-step unpack from the current ones
_PF_BG = """
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) qn{tn}[r] = wq{tn}[r][gn * 4];
"""
_CUR_BG = """
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) qc{tn}[r] = qn{tn}[r];
"""
_MV_BG = """
      {{
        thread uint* bw = (thread uint*)&b{tn}[0];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; r++) {{
          const uint word = ks == 0 ? qc{tn}[r].x : ks == 1 ? qc{tn}[r].y : ks == 2 ? qc{tn}[r].z : qc{tn}[r].w;
          bw[r] = NO_UNPACK ? word : word & 0x0F0F0F0Fu;
          bw[4 + r] = NO_UNPACK ? word : (word >> 4) & 0x0F0F0F0Fu;
        }}
      }}
"""
_RUN = """
      op.run(a{tm}, b{tn}, d{tm}{tn});
#if READ_EVERY  // timing ablation: touch the result tile after every op (READ_EVERY 2: every other op)
      if (ks % READ_EVERY == READ_EVERY - 1)
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) acc{tm}{tn}[i] = fma(float(d{tm}{tn}[i]), 1e-30f, acc{tm}{tn}[i]);
#endif
"""
_PRM = """
    float4 pr{tm}[4];
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) pr{tm}[r] = pr{tm}p[g * 32 + r];
"""
_COLS = """
    float cs{tn}[8], cb{tn}[8];
#if BTG
    _Pragma("clang loop unroll(full)")
    for (int fc = 0; fc < 2; fc++) {{
      const float4 s4 = *(const threadgroup float4*)&Ss[g & 1][nl0 + 32 * {tn} + fn + 16 * fc];
      const float4 b4 = *(const threadgroup float4*)&Sb[g & 1][nl0 + 32 * {tn} + fn + 16 * fc];
      cs{tn}[4 * fc] = s4.x; cs{tn}[4 * fc + 1] = s4.y; cs{tn}[4 * fc + 2] = s4.z; cs{tn}[4 * fc + 3] = s4.w;
      cb{tn}[4 * fc] = b4.x; cb{tn}[4 * fc + 1] = b4.y; cb{tn}[4 * fc + 2] = b4.z; cb{tn}[4 * fc + 3] = b4.w;
    }}
#elif SBT
    _Pragma("clang loop unroll(full)")
    for (int fc = 0; fc < 2; fc++) {{
      const uint4 v = *(const device uint4*)(sbt + int64_t(g) * N + n0 + 32 * {tn} + fn + 16 * fc);
      cs{tn}[4 * fc] = as_type<float>(v.x << 16); cb{tn}[4 * fc] = as_type<float>(v.x & 0xFFFF0000u);
      cs{tn}[4 * fc + 1] = as_type<float>(v.y << 16); cb{tn}[4 * fc + 1] = as_type<float>(v.y & 0xFFFF0000u);
      cs{tn}[4 * fc + 2] = as_type<float>(v.z << 16); cb{tn}[4 * fc + 2] = as_type<float>(v.z & 0xFFFF0000u);
      cs{tn}[4 * fc + 3] = as_type<float>(v.w << 16); cb{tn}[4 * fc + 3] = as_type<float>(v.w & 0xFFFF0000u);
    }}
#else
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 8; j++) {{
      const int n = n0 + 32 * {tn} + fn + 16 * (j / 4) + j % 4;
      cs{tn}[j] = float(scales[n * G + g]);
      cb{tn}[j] = float(biases[n * G + g]);
    }}
#endif
"""
# acc += s * (sx * D + d8) + b * xs
_EPI = """
    if (!NO_EPI || g == G - 1)
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 32; i++) {{
      const int r = 2 * (i / 16) + (i / 4) % 2;
      const int cc = 4 * ((i / 8) % 2) + i % 4;
      const float t = fma(float(d{tm}{tn}[i]), pr{tm}[r].x, pr{tm}[r].y);
      acc{tm}{tn}[i] = fma(t, cs{tn}[cc], fma(cb{tn}[cc], pr{tm}[r].z, acc{tm}{tn}[i]));
    }}
"""
_STORE = """
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < 4; r++) {{
    const int m = m0 + 32 * {tm} + fm + 8 * r;
    if (m < MROWS) {{
      _Pragma("clang loop unroll(full)")
      for (int fc = 0; fc < 2; fc++) {{
        const int i0 = 16 * (r / 2) + 8 * fc + 4 * (r % 2);
        const vec<bfloat, 4> o = {{bfloat(acc{tm}{tn}[i0]), bfloat(acc{tm}{tn}[i0 + 1]), bfloat(acc{tm}{tn}[i0 + 2]),
                                  bfloat(acc{tm}{tn}[i0 + 3])}};
        *(device vec<bfloat, 4>*)(y + int64_t(m) * N + n0 + 32 * {tn} + fn + 16 * fc) = o;
      }}
    }}
  }}
"""
# AG32 path (32-wide activation groups): per-k-step scales, prm as two float4 per (row, group)
_DECL_A_AG = """
  a_t a{tm} = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  const device uint4* xr{tm} = xq4 + (int64_t((m0 + 32 * {tm}) >> 5) * KS * 32 + lane) * 2;
  const device float4* pr{tm}p = prm4 + (int64_t((m0 + 32 * {tm}) >> 5) * G * 8 + fm) * 8;
  uint4 an{tm}[PFD][2];
"""
_DECL_D_AG = """
  d_t d{tm}{tn} = op.template get_destination_cooperative_tensor<a_t, b_t, int32_t>();
  float acc{tm}{tn}[32];
#if AG == 1
  float p{tm}{tn}[32];
#endif
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 32; i++) acc{tm}{tn}[i] = 0.0f;
"""
_PRM_AG = """
    float4 px{tm}[4], pb{tm}[4];
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) {{ px{tm}[r] = pr{tm}p[g * 64 + 2 * r]; pb{tm}[r] = pr{tm}p[g * 64 + 2 * r + 1]; }}
"""
_KEPI = """
      if (!NO_KEPI)
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) {{
        const int r = 2 * (i / 16) + (i / 4) % 2;
        const float sxv = ks == 0 ? px{tm}[r].x : ks == 1 ? px{tm}[r].y : ks == 2 ? px{tm}[r].z : px{tm}[r].w;
#if AG == 1
        p{tm}{tn}[i] = ks == 0 ? float(d{tm}{tn}[i]) * sxv : fma(float(d{tm}{tn}[i]), sxv, p{tm}{tn}[i]);
#else
        const int cc = 4 * ((i / 8) % 2) + i % 4;
        acc{tm}{tn}[i] = fma(float(d{tm}{tn}[i]) * sxv, cs{tn}[cc], acc{tm}{tn}[i]);
#endif
      }}
"""
_EPI_AG = """
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 32; i++) {{
      const int r = 2 * (i / 16) + (i / 4) % 2;
      const int cc = 4 * ((i / 8) % 2) + i % 4;
#if AG == 1
      acc{tm}{tn}[i] = fma(p{tm}{tn}[i] + pb{tm}[r].x, cs{tn}[cc], fma(cb{tn}[cc], pb{tm}[r].y, acc{tm}{tn}[i]));
#else  // AG 2, 3
      acc{tm}{tn}[i] = fma(cs{tn}[cc], pb{tm}[r].x, fma(cb{tn}[cc], pb{tm}[r].y, acc{tm}{tn}[i]));
#endif
    }}
"""
_DECL_D_AG3 = """
  d_t d{tm}{tn}_0 = op.template get_destination_cooperative_tensor<a_t, b_t, int32_t>();
  d_t d{tm}{tn}_1 = op.template get_destination_cooperative_tensor<a_t, b_t, int32_t>();
  float acc{tm}{tn}[32];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 32; i++) acc{tm}{tn}[i] = 0.0f;
"""
_RUN3 = """
#if AGACC
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) d{tm}{tn}_{buf}[i] = 0;
#endif
      op.run(a{tm}, b{tn}, d{tm}{tn}_{buf});
"""
_RUN_AG = """
#if AGACC
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) d{tm}{tn}[i] = 0;
#endif
      op.run(a{tm}, b{tn}, d{tm}{tn});
"""
_KEPI3 = """
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) {{
        const int r = 2 * (i / 16) + (i / 4) % 2;
        const int cc = 4 * ((i / 8) % 2) + i % 4;
        acc{tm}{tn}[i] = fma(float(d{tm}{tn}_{buf}[i]) * px{tm}[r].{comp}, cs{tn}[cc], acc{tm}{tn}[i]);
      }}
"""
# AG 4: as AG 2 with the fewest live registers: the step's four row scales are loaded per k step, the column
# scales once per group, the bias-term params only after the group's last step
_DECL_A_AG4 = """
  a_t a{tm} = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  const device uint4* xr{tm} = xq4 + (int64_t((m0 + 32 * {tm}) >> 5) * KS * 32 + lane) * 2;
  const device float* pr{tm}f = (const device float*)prm + (int64_t((m0 + 32 * {tm}) >> 5) * G * 8 + fm) * 32;
  uint4 an{tm}[PFD][2];
"""
_SX4 = """
      float sx{tm}[4];
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) sx{tm}[r] = pr{tm}f[g * 256 + r * 8 + ks];
"""
_KEPI4 = """
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) {{
        const int r = 2 * (i / 16) + (i / 4) % 2;
        const int cc = 4 * ((i / 8) % 2) + i % 4;
        acc{tm}{tn}[i] = fma(float(d{tm}{tn}[i]) * sx{tm}[r], cs{tn}[cc], acc{tm}{tn}[i]);
      }}
"""
_CS4 = """
    float cs{tn}[8];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 8; j++) cs{tn}[j] = float(scales[(n0 + 32 * {tn} + fn + 16 * (j / 4) + j % 4) * G + g]);
"""
_EPI4 = """
    {{
      float2 pb{tm}[4];
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) pb{tm}[r] = *(const device float2*)(pr{tm}f + g * 256 + r * 8 + 4);
      float cb{tn}[8];
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; j++) cb{tn}[j] = float(biases[(n0 + 32 * {tn} + fn + 16 * (j / 4) + j % 4) * G + g]);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) {{
        const int r = 2 * (i / 16) + (i / 4) % 2;
        const int cc = 4 * ((i / 8) % 2) + i % 4;
        acc{tm}{tn}[i] = fma(cs{tn}[cc], pb{tm}[r].x, fma(cb{tn}[cc], pb{tm}[r].y, acc{tm}{tn}[i]));
      }}
    }}
"""
_TG_NEXT = """
#if BTG
    if (g + 1 < G) {
      fetch(g + 1);
      fetch_s(g + 1);
    }
#endif
"""
_TG_STAGE = """
#if BTG
    if (g + 1 < G) stage((g + 1) & 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
"""


def _gemm_src(TM, TN, prefetch=False, brow=False, ag=0, order=1):
    T = [(tm, tn) for tm in range(TM) for tn in range(TN)]
    fa = lambda t: "".join(t.format(tm=tm, tn=0) for tm in range(TM))  # noqa: E731
    fb = lambda t: "".join(t.format(tm=0, tn=tn) for tn in range(TN))  # noqa: E731
    fd = lambda t: "".join(t.format(tm=tm, tn=tn) for tm, tn in T)  # noqa: E731
    if ag == 5:  # 64-wide groups, lean registers: zero at even k steps, scale after odd ones
        zero = fd('      _Pragma("clang loop unroll(full)")\n      for (int i = 0; i < 32; i++) d{tm}{tn}[i] = 0;\n')
        sx = _SX4.replace("r * 8 + ks]", "r * 8 + ks / 2]")
        return (_GEMM_HEAD + fa(_DECL_A_AG4) + fb(_DECL_B) + fd(_DECL_D)
                + "  {\n    const int st = 0;\n" + fa(_PF_A) + fb(_PF_B) + "  }\n"
                + "  for (int g = 0; g < G; g++) {\n" + fb(_CS4)
                + '    _Pragma("clang loop unroll(full)")\n    for (int ks = 0; ks < 4; ks++) {\n'
                + fa(_MV_A) + fb(_MV_B)
                + "      const int st = g * 4 + ks + PFD;\n      if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B) + "      }\n"
                + "      if (ks % 2 == 0) {\n" + zero + "      }\n"
                + fd(_RUN) + "      if (ks % 2 == 1) {\n" + fa(sx) + fd(_KEPI4) + "      }\n    }\n"
                + fd(_EPI4) + "  }\n" + fd(_STORE))
    if ag == 4:  # 32-wide groups, lean registers; the 128-wide kernel's step order (unpack, prefetch, op, scale)
        return (_GEMM_HEAD + fa(_DECL_A_AG4) + fb(_DECL_B) + fd(_DECL_D)
                + "  {\n    const int st = 0;\n" + fa(_PF_A) + fb(_PF_B) + "  }\n"
                + "  for (int g = 0; g < G; g++) {\n" + fb(_CS4)
                + '    _Pragma("clang loop unroll(full)")\n    for (int ks = 0; ks < 4; ks++) {\n'
                + fa(_MV_A) + fb(_MV_B)
                + "      const int st = g * 4 + ks + PFD;\n      if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B) + "      }\n"
                + fd(_RUN_AG) + fa(_SX4) + fd(_KEPI4) + "    }\n"
                + fd(_EPI4) + "  }\n" + fd(_STORE))
    if ag == 3:  # 32-wide groups, two result tiles: the scaling of step k - 1 overlaps the op of step k
        fdb = lambda t, **kw: "".join(t.format(tm=tm, tn=tn, **kw) for tm, tn in T)  # noqa: E731
        body = ""
        for ks in range(4):
            body += f"    {{\n      const int ks = {ks};\n" + fdb(_RUN3, buf=ks % 2)
            body += ("      {\n        const int sn = g * 4 + ks + 1;\n        if (sn < KS) {\n" + fa(_MV_A) + fb(_MV_B)
                     + "          const int st = sn + 1;\n          if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B)
                     + "          }\n        }\n      }\n")
            if ks:
                body += fdb(_KEPI3, buf=(ks - 1) % 2, comp="xyzw"[ks - 1])
            body += "    }\n"
        body += fdb(_KEPI3, buf=1, comp="w")
        return (_GEMM_HEAD + fa(_DECL_A_AG) + fb(_DECL_B) + fd(_DECL_D_AG3)
                + "  {\n    const int st = 0;\n" + fa(_PF_A) + fb(_PF_B) + "  }\n"
                + "  {\n    const int ks = 0;\n" + fa(_MV_A) + fb(_MV_B) + "  }\n"
                + "  {\n    const int st = 1;\n    if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B) + "    }\n  }\n"
                + "  for (int g = 0; g < G; g++) {\n" + fa(_PRM_AG) + fb(_COLS) + body
                + fd(_EPI_AG) + "  }\n" + fd(_STORE))
    if ag:  # 32-wide activation groups: a fresh tile per k step; the next step's operands are unpacked while the op
        # runs, then the step's result is scaled (PFD 1: a / b hold step s + 1, the prefetch registers step s + 2)
        cols = fb(_COLS)
        return (_GEMM_HEAD + fa(_DECL_A_AG) + fb(_DECL_B) + fd(_DECL_D_AG)
                + "  {\n    const int st = 0;\n" + fa(_PF_A) + fb(_PF_B) + "  }\n"
                + "  {\n    const int ks = 0;\n" + fa(_MV_A) + fb(_MV_B) + "  }\n"
                + "  {\n    const int st = 1;\n    if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B) + "    }\n  }\n"
                + "  for (int g = 0; g < G; g++) {\n" + fa(_PRM_AG) + (cols if ag == 2 else "")
                + '    _Pragma("clang loop unroll(full)")\n    for (int ks = 0; ks < 4; ks++) {\n'
                + ("" if not order else fd(_RUN_AG))
                + "      {\n        const int sn = g * 4 + ks + 1;\n        if (sn < KS) {\n"
                + ("" if order else fd(_RUN_AG))
                + fa(_MV_A) + fb(_MV_B)
                + "          const int st = sn + 1;\n          if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B)
                + "          }\n        }\n" + ("" if order else "        else {\n" + fd(_RUN_AG) + "        }\n")
                + "      }\n"
                + fd(_KEPI) + "    }\n"
                + (cols if ag == 1 else "") + fd(_EPI_AG) + "  }\n" + fd(_STORE))
    if brow:  # A prefetched a k step ahead, B a group ahead
        return (_GEMM_HEAD + fa(_DECL_A) + fb(_DECL_B) + fd(_DECL_D)
                + '  _Pragma("clang loop unroll(full)")\n  for (int st = 0; st < PFD; st++) {\n' + fa(_PF_A) + "  }\n"
                + "  {\n    const int gn = 0;\n" + fb(_PF_BG) + "  }\n"
                + "  for (int g = 0; g < G; g++) {\n" + fd(_ZERO_D) + fb(_CUR_BG)
                + "    if (g + 1 < G) {\n      const int gn = g + 1;\n" + fb(_PF_BG) + "    }\n"
                + '    _Pragma("clang loop unroll(full)")\n    for (int ks = 0; ks < 4; ks++) {\n'
                + fa(_MV_A) + fb(_MV_BG)
                + "      const int st = g * 4 + ks + PFD;\n      if (st < KS) {\n" + fa(_PF_A) + "      }\n"
                + fd(_RUN) + "    }\n"
                + fa(_PRM) + fb(_COLS) + fd(_EPI) + "  }\n" + fd(_STORE))
    if prefetch:
        return (_GEMM_HEAD + fa(_DECL_A) + fb(_DECL_B) + fd(_DECL_D)
                + '  _Pragma("clang loop unroll(full)")\n  for (int st = 0; st < PFD; st++) {\n'
                + fa(_PF_A) + fb(_PF_B) + "  }\n"
                + "  for (int g = 0; g < G; g++) {\n" + fd(_ZERO_D)
                + '    _Pragma("clang loop unroll(full)")\n    for (int ks = 0; ks < 4; ks++) {\n'
                + fa(_MV_A) + fb(_MV_B)
                + "      const int st = g * 4 + ks + PFD;\n      if (st < KS) {\n" + fa(_PF_A) + fb(_PF_B) + "      }\n"
                + fd(_RUN) + "    }\n"
                + fa(_PRM) + fb(_COLS) + fd(_EPI) + "  }\n" + fd(_STORE))
    return (_GEMM_HEAD + fa(_DECL_A) + fb(_DECL_B) + fd(_DECL_D)
            + "  for (int g = 0; g < G; g++) {\n" + _TG_NEXT + fd(_ZERO_D)
            + '    _Pragma("clang loop unroll(full)")\n    for (int ks = 0; ks < 4; ks++) {\n'
            + fa(_LOAD_A) + fb(_LOAD_B) + fd(_RUN) + "    }\n"
            + fa(_PRM) + fb(_COLS) + fd(_EPI) + _TG_STAGE + "  }\n" + fd(_STORE))


# PLAIN: MLX-style register tiles. A / B words and the int32 partials live in plain arrays; each accelerator op builds
# its cooperative tensors, copies in, runs and copies out (the copies are register renames once allocated). This
# keeps several output tiles per simdgroup without holding cooperative tensors across ops.
_PLAIN_SRC = """
  constexpr int G = K / 128;
  constexpr int KW = K / 8;
  constexpr int KS = K / 32;
  constexpr int TM = SM / 32, TN = SN / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
#if MFAST
  const int tgm = int(threadgroup_position_in_grid.x), tgn = int(threadgroup_position_in_grid.y);
#else
  const int tgn = int(threadgroup_position_in_grid.x), tgm = int(threadgroup_position_in_grid.y);
#endif
  const int m0 = tgm * (SM * WM) + int(sg / WN) * SM;
  const int n0 = tgn * (SN * WN) + int(sg % WN) * SN;
  const int qid = int(lane >> 2);
  const int fm = (qid & 4) | (int(lane >> 1) & 3);
  const int fn = ((qid & 2) | int(lane & 1)) * 4;
  const uint sh = uint(fn & 4);
  const int kw = fn >> 3;
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      32, 32, 32, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  const device uint4* xr[TM];
  const device float4* prp[TM];
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++) {
    const int mb = (m0 + 32 * tm) >> 5;
    xr[tm] = (const device uint4*)xq + (int64_t(mb) * KS * 32 + lane) * 2;
    prp[tm] = (const device float4*)prm + (int64_t(mb) * G * 8 + fm) * 4;
  }
#if BROW
  const device uint4* wq[TN][4];
  uint4 qc[TN][4], qn[TN][4];
#else
  const device uint* wr[TN][4];
  uint bn[TN][8];
#endif
  _Pragma("clang loop unroll(full)")
  for (int tn = 0; tn < TN; tn++)
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) {
      const int64_t row = int64_t(n0 + 32 * tn + fm + 8 * r) * KW;
#if BROW
      wq[tn][r] = (const device uint4*)(w + row) + (fn >> 2);
#else
      wr[tn][r] = w + row + kw;
#endif
    }
  uint4 an[TM][2];
  uint aw[TM][8], bw[TN][8];
  int dd[TM][TN][32];
  float acc[TM][TN][32];
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++)
    _Pragma("clang loop unroll(full)")
    for (int tn = 0; tn < TN; tn++)
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 32; i++) acc[tm][tn][i] = 0.0f;

  auto fetch_a = [&](int st) {
    _Pragma("clang loop unroll(full)")
    for (int tm = 0; tm < TM; tm++) { an[tm][0] = xr[tm][st * 64]; an[tm][1] = xr[tm][st * 64 + 1]; }
  };
#if BROW
  auto fetch_b = [&](int g) {
    _Pragma("clang loop unroll(full)")
    for (int tn = 0; tn < TN; tn++)
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) qn[tn][r] = wq[tn][r][g * 4];
  };
  fetch_b(0);
#else
  auto fetch_b = [&](int st) {
    _Pragma("clang loop unroll(full)")
    for (int tn = 0; tn < TN; tn++)
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++)
        _Pragma("clang loop unroll(full)")
        for (int fc = 0; fc < 2; fc++) bn[tn][4 * fc + r] = wr[tn][r][st * 4 + 2 * fc];
  };
  fetch_b(0);
#endif
  fetch_a(0);

  for (int g = 0; g < G; g++) {
    _Pragma("clang loop unroll(full)")
    for (int tm = 0; tm < TM; tm++)
      _Pragma("clang loop unroll(full)")
      for (int tn = 0; tn < TN; tn++)
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 32; i++) dd[tm][tn][i] = 0;
#if BROW
    _Pragma("clang loop unroll(full)")
    for (int tn = 0; tn < TN; tn++)
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < 4; r++) qc[tn][r] = qn[tn][r];
    if (g + 1 < G) fetch_b(g + 1);
#endif
    _Pragma("clang loop unroll(full)")
    for (int ks = 0; ks < 4; ks++) {
      _Pragma("clang loop unroll(full)")
      for (int tm = 0; tm < TM; tm++) {
        aw[tm][0] = an[tm][0].x; aw[tm][1] = an[tm][0].y; aw[tm][2] = an[tm][0].z; aw[tm][3] = an[tm][0].w;
        aw[tm][4] = an[tm][1].x; aw[tm][5] = an[tm][1].y; aw[tm][6] = an[tm][1].z; aw[tm][7] = an[tm][1].w;
      }
      _Pragma("clang loop unroll(full)")
      for (int tn = 0; tn < TN; tn++)
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; r++) {
#if BROW
          const uint word = ks == 0 ? qc[tn][r].x : ks == 1 ? qc[tn][r].y : ks == 2 ? qc[tn][r].z : qc[tn][r].w;
          bw[tn][r] = word & 0x0F0F0F0Fu;
          bw[tn][4 + r] = (word >> 4) & 0x0F0F0F0Fu;
#else
          bw[tn][r] = (bn[tn][r] >> sh) & 0x0F0F0F0Fu;
          bw[tn][4 + r] = (bn[tn][4 + r] >> sh) & 0x0F0F0F0Fu;
#endif
        }
      const int st = g * 4 + ks + 1;
      if (st < KS) {
        fetch_a(st);
#if !BROW
        fetch_b(st);
#endif
      }
      _Pragma("clang loop unroll(full)")
      for (int tm = 0; tm < TM; tm++)
        _Pragma("clang loop unroll(full)")
        for (int tn = 0; tn < TN; tn++) {
          auto ca = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
          auto cb = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
          auto cd = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), int32_t>();
          thread uint* cap = (thread uint*)&ca[0];
          thread uint* cbp = (thread uint*)&cb[0];
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < 8; j++) { cap[j] = aw[tm][j]; cbp[j] = bw[tn][j]; }
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < 32; i++) cd[i] = dd[tm][tn][i];
          op.run(ca, cb, cd);
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < 32; i++) dd[tm][tn][i] = cd[i];
        }
    }
    // epilogue: acc += s * (sx * D + d8) + b * xs
    _Pragma("clang loop unroll(full)")
    for (int tn = 0; tn < TN; tn++) {
      float cs[8], cb[8];
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; j++) {
        const int n = n0 + 32 * tn + fn + 16 * (j / 4) + j % 4;
        cs[j] = float(scales[n * G + g]);
        cb[j] = float(biases[n * G + g]);
      }
      _Pragma("clang loop unroll(full)")
      for (int tm = 0; tm < TM; tm++) {
        float4 pr[4];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; r++) pr[r] = prp[tm][g * 32 + r];
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 32; i++) {
          const int r = 2 * (i / 16) + (i / 4) % 2;
          const int cc = 4 * ((i / 8) % 2) + i % 4;
          const float t = fma(float(dd[tm][tn][i]), pr[r].x, pr[r].y);
          acc[tm][tn][i] = fma(t, cs[cc], fma(cb[cc], pr[r].z, acc[tm][tn][i]));
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int tm = 0; tm < TM; tm++)
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) {
      const int m = m0 + 32 * tm + fm + 8 * r;
      if (m < MROWS) {
        _Pragma("clang loop unroll(full)")
        for (int tn = 0; tn < TN; tn++)
          _Pragma("clang loop unroll(full)")
          for (int fc = 0; fc < 2; fc++) {
            const int i0 = 16 * (r / 2) + 8 * fc + 4 * (r % 2);
            const vec<bfloat, 4> o = {bfloat(acc[tm][tn][i0]), bfloat(acc[tm][tn][i0 + 1]), bfloat(acc[tm][tn][i0 + 2]),
                                      bfloat(acc[tm][tn][i0 + 3])};
            *(device vec<bfloat, 4>*)(y + int64_t(m) * N + n0 + 32 * tn + fn + 16 * fc) = o;
          }
      }
    }
"""
_plain_kernels = {}


def _plain_kernel(sm, sn, wm, wn, mfast, brow):
    key = (sm, sn, wm, wn, mfast, brow)
    if key not in _plain_kernels:
        defs = dict(SM=sm, SN=sn, WM=wm, WN=wn, MFAST=int(mfast), BROW=int(brow))
        _plain_kernels[key] = mx.fast.metal_kernel(
            name=f"w4a8_plain_{sm}_{sn}_{wm}_{wn}_{int(mfast)}{int(brow)}",
            input_names=["xq", "prm", "w", "scales", "biases"], output_names=["y"],
            source="".join(f"#define {k} {v}\n" for k, v in defs.items()) + _PLAIN_SRC, header=MPP_HEADER)
    return _plain_kernels[key]


def w4a8_matmul_plain(xq, prm, M, w, scales, biases, sm=32, sn=32, wm=2, wn=2, mfast=True, brow=False):
    """Same contract as w4a8_matmul (xq from act_quant(perm=int(brow)))."""
    Mp, K = xq.shape
    N = w.shape[0]
    bm, bn = sm * wm, sn * wn
    assert Mp % bm == 0 and N % bn == 0 and K % GS == 0, (Mp, N, K, bm, bn)
    gm, gn = Mp // bm, N // bn
    grid = (gm, gn) if mfast else (gn, gm)
    return _plain_kernel(sm, sn, wm, wn, mfast, brow)(
        inputs=[xq, prm, w, scales, biases], template=[("K", K), ("N", N), ("MROWS", M)],
        grid=(grid[0] * wm * wn * 32, grid[1], 1), threadgroup=(wm * wn * 32, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]


_gemm_kernels = {}
ABLATE = ("NO_ALOAD", "NO_BLOAD", "NO_UNPACK", "NO_EPI", "B_HOT", "NO_KEPI")


def _gemm_kernel(sm, sn, wm, wn, mfast, btg, pf, bvec, sbt, brow, ag=0, ablate=()):
    pfd = 1 if ag else (int(pf) if pf else 1)
    key = (sm, sn, wm, wn, mfast, btg, pf, bvec, sbt, brow, ag, tuple(ablate))
    if key not in _gemm_kernels:
        defs = {f: int(f in ablate) for f in ABLATE}
        defs["READ_EVERY"] = next((int(f[10:]) for f in ablate if f.startswith("READ_EVERY")), 0)
        defs["AGACC"] = int("AGACC" in ablate)
        order = int("AGORDER0" not in ablate)
        defs.update(SM=sm, SN=sn, WM=wm, WN=wn, MFAST=int(mfast), BTG=int(btg), PREFETCH=int(bool(pf)), PFD=pfd,
                    BVEC=int(bvec), SBT=int(sbt), BROW=int(brow), AG=int(ag))
        src = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _gemm_src(sm // 32, sn // 32, pf and not btg,
                                                                                   brow, ag, order)
        _gemm_kernels[key] = mx.fast.metal_kernel(
            name=f"w4a8_gemm_{sm}_{sn}_{wm}_{wn}_{int(mfast)}{int(btg)}{pfd if pf else 0}{int(bvec)}{int(sbt)}{int(brow)}"
            f"_ag{ag}"
            + "".join("_" + f.lower() for f in ablate),
            input_names=["xq", "prm", "w", "scales", "biases", "sbt"], output_names=["y"], source=src,
            header=MPP_HEADER)
    return _gemm_kernels[key]


_SBT_SRC = """
  const int n = int(thread_position_in_grid.x), g = int(thread_position_in_grid.y);
  if (n >= N) return;
  const uint s = as_type<ushort>(scales[n * G + g]), b = as_type<ushort>(biases[n * G + g]);
  sbt[g * N + n] = s | (b << 16);
"""
_sbt_kernel = mx.fast.metal_kernel(name="w4a8_sbt", input_names=["scales", "biases"], output_names=["sbt"],
                                   source=_SBT_SRC)


def sb_transpose(scales, biases):
    """bf16 scales / biases [N, G] -> uint32 [G, N] with (s | b << 16), the epilogue's per-column reads."""
    assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    N, G = scales.shape
    return _sbt_kernel(inputs=[scales, biases], template=[("N", N), ("G", G)], grid=(-(-N // 256) * 256, G, 1),
                       threadgroup=(256, 1, 1), output_shapes=[(G, N)], output_dtypes=[mx.uint32])[0]


def w4a8_matmul(xq, prm, M, w, scales, biases, sm=32, sn=32, wm=2, wn=2, mfast=True, btg=False, pf=1, bvec=False,
                sbt=None, brow=False, ag=0, ablate=()):
    """ag: 0 = 128-wide activation groups (prm from act_quant(ag=128)); 1 / 2 / 3 = 32-wide groups
    (act_quant(ag=32)), scaled through an fp32 per-group partial (1), straight into the accumulator (2), or as 2
    with two result tiles so each step's scaling overlaps the next step's op (3), as 2 with the fewest live registers
    (4); 5 = 64-wide groups (act_quant(ag=64)) on the lean path."""
    """(act_quant output, M real rows) x MLX 4-bit g128 weights [N, K/8] -> y [M, N] bf16. N % (sn * wn) == 0 and
    xq rows padded to a multiple of sm * wm."""
    Mp, K = xq.shape
    N = w.shape[0]
    bm, bn = sm * wm, sn * wn
    assert Mp % bm == 0 and N % bn == 0 and K % GS == 0, (Mp, N, K, bm, bn)
    gm, gn = Mp // bm, N // bn
    grid = (gm, gn) if mfast else (gn, gm)
    bvec = bvec and pf and not btg  # the vector B loads exist only in the prefetching main loop
    brow = brow and not btg  # BROW needs xq from act_quant(perm=1), the others perm=0
    if ag:
        assert not (btg or brow) and prm.shape[-1] == 8, "sub-group scales: default B path, act_quant(ag=32 / 64)"
    pf = pf or (1 if brow else 0)
    use_sbt = sbt is not None and sbt is not False
    if sbt is True:
        sbt = sb_transpose(scales, biases)
    return _gemm_kernel(sm, sn, wm, wn, mfast, btg, pf, bvec, use_sbt, brow, ag, ablate)(
        inputs=[xq, prm, w, scales, biases, sbt if use_sbt else scales], template=[("K", K), ("N", N), ("MROWS", M)],
        grid=(grid[0] * wm * wn * 32, grid[1], 1), threadgroup=(wm * wn * 32, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]


def qmm(x, w, scales, biases, plain=False, **kw):
    """Convenience: act_quant + GEMM."""
    bm = kw.get("sm", 32) * kw.get("wm", 2)
    xq, prm = act_quant(x, mpad=bm, perm=int(kw.get("brow", False) and not kw.get("btg", False)),
                        ag=64 if kw.get("ag") == 5 else 32 if kw.get("ag") else 128)
    if plain:
        return w4a8_matmul_plain(xq, prm, x.shape[0], w, scales, biases, **kw)
    return w4a8_matmul(xq, prm, x.shape[0], w, scales, biases, **kw)
