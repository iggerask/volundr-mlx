"""Small-M 2-bit (and 3-bit) affine quantized matmul on the neural accelerators: a feasibility probe.

MPP has no 2- or 3-bit operand format, so each lane unpacks its share of the weight tile into an
int8 cooperative (register) tensor and the tensor op reads it from there. The right-input
cooperative layout (probe_layout.py) gives lane L, for a 32-row x 128-k tile, the elements
  i = kb*16 + nj*4 + t  ->  k = 4*qsel + 16*kb + t,  n = nbase + 8*nj
with qsel = (L & 1) + 2*((L >> 3) & 1) and nbase = ((L >> 1) & 3) + 4*(L >> 4).

repack() reorders each (row, group)'s 32 bytes of 2-bit codes so that lane qsel reads one uint2,
and value (kb%4, t) of word kb/4 sits at bit 8t + 2*(kb%4). Then (word >> 2*(kb%4)) & 0x03030303
is exactly the four int8 codes for elements kb*16 + nj*4 + 0..3, one register store.

For 3-bit, repack() splits the codes into the same 2-bit plane plus a 1-bit plane laid out the same way
(bit 8t + kb), so a 3-bit code is (lo2 | hi1 << 2) for four values at once. Codes are then
centred (q - 2^(bits-1)), so callers pass centred_bias() instead of the MLX biases.

This layout is implementation-defined by MPP; check.py-style tests catch a change.
"""
import mlx.core as mx

HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""

SOURCE = """
  using ext = metal::dextents<int32_t, 2>;
  using strd = metal::array<int32_t, 2>;
  constexpr int G = K / GS;
  constexpr int NJ = NT / 8;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const int n0 = int(threadgroup_position_in_grid.x) * NT;
  const int qsel = int(lane & 1) + 2 * int((lane >> 3) & 1);
  const int nbase = int((lane >> 1) & 3) + 4 * int(lane >> 4);

  metal::tensor<device XT, ext, metal::tensor_inline> tX((device XT*)x, ext(K, MT), strd{1, K});
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      MT, NT, GS, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using xs_t = decltype(tX.template slice<GS, MT>(0, 0));

  auto r = op.template get_right_input_cooperative_tensor<XT, int8_t, DT>();
  using r_t = decltype(r);
  auto acc = op.template get_destination_cooperative_tensor<xs_t, r_t, float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;
  const device bfloat* sp = (const device bfloat*)scales + int64_t(n0) * G;
  const device bfloat* bp = (const device bfloat*)biases + int64_t(n0) * G;

  // words per (row, group): 8 for the 2-bit plane, then (3-bit only) 4 for the 1-bit plane.
  constexpr int WPG = BITS == 2 ? 8 : 12;
  const device uint2* w2 = (const device uint2*)w;
  const device uint* w1 = (const device uint*)w;
  auto row_word = [&](int nj, int g) { return (int64_t(n0 + nbase + 8 * nj) * G + g) * WPG; };

  uint2 lo[NJ];
  uint hi[NJ];
  auto fetch = [&](int g) {
    _Pragma("clang loop unroll(full)")
    for (int nj = 0; nj < NJ; nj++) {
      const int64_t base = row_word(nj, g);
      lo[nj] = w2[base / 2 + qsel];
      if constexpr (BITS == 3) hi[nj] = w1[base + 8 + qsel];
    }
  };
  int g = sg;
  if (g < G) fetch(g);
#if PTRWRITE
  thread uint* rw = (thread uint*)&r[0];
#endif
  for (; g < G; g += KS) {
    // Unpack the fetched words into the register tile, then fetch the next group.
    _Pragma("clang loop unroll(full)")
    for (int nj = 0; nj < NJ; nj++) {
      _Pragma("clang loop unroll(full)")
      for (int kb = 0; kb < 8; kb++) {
        const uint word = kb < 4 ? lo[nj].x : lo[nj].y;
        uint v = (word >> (2 * (kb & 3))) & 0x03030303u;
        if constexpr (BITS == 3) v |= ((hi[nj] >> kb) & 0x01010101u) << 2;
        // Centre the codes (q - 2^(BITS-1)) bytewise without borrows; to the bias side this is
        // folded by centred_bias(). Keeps int8-activation rounding errors from adding up.
        v = ((v | 0x80808080u) - (0x01010101u << (BITS - 1))) ^ 0x80808080u;
#if PTRWRITE
        rw[kb * NJ + nj] = v;
#else
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < 4; t++) r[(kb * NJ + nj) * 4 + t] = int8_t((v >> (8 * t)) & 0xFFu);
#endif
      }
    }
    if (g + KS < G) fetch(g + KS);
    auto xs = tX.template slice<GS, MT>(g * GS, 0);
    auto d = op.template get_destination_cooperative_tensor<xs_t, r_t, DT>();
    op.run(xs, r, d);
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < d.get_capacity(); i++) {
      auto idx = d.get_multidimensional_index(i);
      const int n = idx[0];
      const int m = idx[1];
#if XINT8
      acc[i] += float(sp[n * G + g]) * xscale[m * G + g] * float(d[i]) + float(bp[n * G + g]) * xsum[m * G + g];
#else
      acc[i] += float(sp[n * G + g]) * d[i] + float(bp[n * G + g]) * xsum[m * G + g];
#endif
    }
  }

  threadgroup float red[KS * MT * NT];
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);
      red[sg * (MT * NT) + idx[1] * NT + idx[0]] = acc[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = thread_index_in_threadgroup; e < MROWS * NT; e += KS * 32) {
    float s = 0.0f;
    for (int k = 0; k < KS; k++) s += red[k * (MT * NT) + e];
    y[(e / NT) * N + n0 + (e % NT)] = bfloat(s);
  }
"""

MT, GS = 16, 128
_DT = {"bf16": ("bfloat", "float"), "i8": ("int8_t", "int32_t")}
_kernels = {}


def _kernel(dtype):
    if dtype not in _kernels:
        xt, dt = _DT[dtype]
        # PTRWRITE 1 stores each unpacked uint32 straight into the cooperative tensor's storage.
        # Both paths are correct. In a cache-warm 2-bit gate_up test, the pointer path was 28% faster
        # with bf16 x and 21% slower with int8 x. The tables in NOTES.md use 0.
        src = f"#define XINT8 {int(dtype == 'i8')}\n#define PTRWRITE 0\n" + SOURCE.replace("XT", xt).replace("DT", dt)
        _kernels[dtype] = mx.fast.metal_kernel(
            name=f"qmm_nax_lowbit_{dtype}", input_names=["x", "w", "scales", "biases", "xsum", "xscale"],
            output_names=["y"], source=src, header=HEADER)
    return _kernels[dtype]


def _codes(w, bits, K):
    """MLX-packed codes (N, K*bits/32) uint32 -> (N, K) uint32 values."""
    N = w.shape[0]
    if bits == 2:
        return ((w[..., None] >> (2 * mx.arange(16, dtype=mx.uint32))) & 3).reshape(N, K)
    # 3-bit: MLX packs 8 values into 3 bytes (24 bits), little-endian.
    b = mx.view(w, mx.uint8).reshape(N, K // 8, 3).astype(mx.uint32)
    v24 = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    return ((v24[..., None] >> (3 * mx.arange(8, dtype=mx.uint32))) & 7).reshape(N, K)


def _plane(v, nbits):
    """(N, K) codes of nbits (1 or 2) -> lane words: (N, G, 4 qsel, 2 halves) for 2-bit planes or
    (N, G, 4 qsel) for 1-bit planes, with value (kb, t) at bit 8t + nbits*kb_in_word."""
    N, K = v.shape
    G = K // GS
    # k = 16*kb + 4*qsel + t
    v = v.reshape(N, G, 8, 4, 4)                      # (kb, qsel, t)
    if nbits == 2:
        v = v.reshape(N, G, 2, 4, 4, 4).transpose(0, 1, 4, 2, 3, 5)   # (half, j, qsel, t) -> (qsel, half, j, t)
        sh = (8 * mx.arange(4, dtype=mx.uint32)[None, :] + 2 * mx.arange(4, dtype=mx.uint32)[:, None])
        return (v << sh).sum(axis=(-2, -1)).astype(mx.uint32)           # (N, G, 4, 2)
    v = v.transpose(0, 1, 3, 2, 4)                    # (qsel, kb, t)
    sh = (8 * mx.arange(4, dtype=mx.uint32)[None, :] + mx.arange(8, dtype=mx.uint32)[:, None])
    return (v << sh).sum(axis=(-2, -1)).astype(mx.uint32)               # (N, G, 4)


def repack(w, bits, K):
    """MLX 2/3-bit packing -> this kernel's lane-major packing (same number of bytes)."""
    v = _codes(w, bits, K)
    N = v.shape[0]
    G = K // GS
    lo = _plane(v & 3, 2).reshape(N, G, 8)
    if bits == 2:
        return lo.reshape(N, G * 8)
    hi = _plane(v >> 2, 1).reshape(N, G, 4)
    return mx.concatenate([lo, hi], axis=-1).reshape(N, G * 12)


def centred_bias(scales, biases, bits):
    """Bias for centred codes: s*q + b = s*(q - c) + (b + c*s), c = 2^(bits-1)."""
    c = 2 ** (bits - 1)
    return (biases.astype(mx.float32) + c * scales.astype(mx.float32)).astype(biases.dtype)


def prep_x(x, dtype="bf16"):
    M, K = x.shape
    if M < MT:
        x = mx.pad(x, [(0, MT - M), (0, 0)])
    xf = x.astype(mx.float32).reshape(MT, K // GS, GS)
    xsum = xf.sum(-1)
    if dtype == "i8":
        xscale = mx.maximum(mx.abs(xf).max(-1), 1e-12) / 127.0
        return M, mx.round(xf / xscale[..., None]).astype(mx.int8).reshape(MT, K), xsum, xscale
    return M, x, xsum, xsum


def qmm_lowbit_prepped(px, wr, scales, biases, bits, dtype="bf16", nt=32, ks=8):
    M, xin, xsum, xscale = px
    K = xin.shape[1]
    N = wr.shape[0]
    return _kernel(dtype)(
        inputs=[xin, wr, scales, biases, xsum, xscale],
        template=[("MT", MT), ("NT", nt), ("KS", ks), ("GS", GS), ("K", K), ("N", N), ("MROWS", M),
                  ("BITS", bits)],
        grid=(N // nt * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
