"""Small-M (2..16 tokens) affine 4-bit quantized matmul on the M5 neural accelerators (NAX).

y[M, N] = x[M, K] @ dequant(w)[N, K].T for MLX's affine packing (w = s * q + b per 128-group).

The MPP matmul2d tensor op reads MLX's packed 4-bit weights straight from device memory as a
`metal::uint4b_format` right operand, so nothing is dequantized. Per 128-group the tensor op
computes D = x_g @ q_g.T, and the epilogue adds s[n, g] * D + b[n, g] * sum(x_g) into the
accumulator.

Tiles: the tokens, padded to 16, are the left operand; 64 weight rows are the right. The
accelerator pads 8-token tiles to 16 internally (measured: no faster), so M = 1..16 all cost the
same. Each threadgroup owns 64 output columns; its 8 simdgroups split the K groups and reduce
through threadgroup memory.

dtype="i8" quantizes x per (row, group) to int8 and runs int8 x int4 tensor ops, which are about
1.5x faster on the accelerator, at about 3x the numerical error of bf16. It needs weights
repacked to signed int4 with to_int4() (an offline step in a real converter).
See variants.py for the experiments behind these choices.
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
  const uint sg = simdgroup_index_in_threadgroup;
  const int n0 = int(threadgroup_position_in_grid.x) * NT;

  metal::tensor<device XT, ext, metal::tensor_inline> tX((device XT*)x, ext(K, MT), strd{1, K});
  metal::tensor<device WF, ext, metal::tensor_inline> tW((device uchar*)w, ext(K, N), strd{1, K});
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      MT, NT, GS, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using xs_t = decltype(tX.template slice<GS, MT>(0, 0));
  using ws_t = decltype(tW.template slice<GS, NT>(0, 0));

  auto acc = op.template get_destination_cooperative_tensor<xs_t, ws_t, float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;
  const device bfloat* sp = (const device bfloat*)scales + int64_t(n0) * G;
  const device bfloat* bp = (const device bfloat*)biases + int64_t(n0) * G;

  // Groups interleave across the simdgroups, so neighbours read adjacent halves of each line.
  for (int g = sg; g < G; g += KS) {
    auto xs = tX.template slice<GS, MT>(g * GS, 0);
    auto ws = tW.template slice<GS, NT>(g * GS, n0);
    auto d = op.template get_destination_cooperative_tensor<xs_t, ws_t, DT>();
    op.run(xs, ws, d);
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < d.get_capacity(); i++) {
      auto idx = d.get_multidimensional_index(i);
      const int n = idx[0];
      const int m = idx[1];
#if XINT8
      // x_g ~= xscale[m, g] * xq, and the weights hold q - 8 (to_int4 folds 8 s into b).
      acc[i] += float(sp[n * G + g]) * xscale[m * G + g] * float(d[i])
              + float(bp[n * G + g]) * xsum[m * G + g];
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

MT, NT, KS, GS = 16, 64, 8, 128
_DT = {"bf16": ("bfloat", "metal::uint4b_format", "float"),
       "i8": ("int8_t", "metal::int4b_format", "int32_t")}
_kernels = {}


def _kernel(dtype):
    if dtype not in _kernels:
        xt, wf, dt = _DT[dtype]
        src = (f"#define XINT8 {int(dtype == 'i8')}\n"
               + SOURCE.replace("XT", xt).replace("WF", wf).replace("DT", dt))
        _kernels[dtype] = mx.fast.metal_kernel(
            name=f"qmm_nax_{dtype}", input_names=["x", "w", "scales", "biases", "xsum", "xscale"],
            output_names=["y"], source=src, header=HEADER)
    return _kernels[dtype]


def prep_x(x, dtype="bf16"):
    """Pad x to 16 rows and build the per-group sums (and int8 codes/scales for dtype='i8').

    Several matmuls that share an input (q/k/v, gate/up) can share one prep_x result; a real
    model would fuse it into the norm that produces x."""
    M, K = x.shape
    assert M <= MT and K % GS == 0
    if M < MT:
        x = mx.pad(x, [(0, MT - M), (0, 0)])
    xf = x.astype(mx.float32).reshape(MT, K // GS, GS)
    xsum = xf.sum(-1)
    if dtype == "i8":
        xscale = mx.maximum(mx.abs(xf).max(-1), 1e-12) / 127.0
        xq = mx.round(xf / xscale[..., None]).astype(mx.int8).reshape(MT, K)
        return M, xq, xsum, xscale
    return M, x, xsum, xsum


def to_int4(w, s, b):
    """MLX unsigned 4-bit packing -> signed int4 (q - 8), with 8 s folded into the bias."""
    w4 = w ^ mx.array(0x88888888, dtype=mx.uint32)
    b4 = (b.astype(mx.float32) + 8 * s.astype(mx.float32)).astype(b.dtype)
    return w4, s, b4


def qmm_nax_prepped(px, w, scales, biases, dtype="bf16"):
    """y = x @ dequant(w).T for a prep_x() result. Returns (M, N) bf16."""
    M, xin, xsum, xscale = px
    K = xin.shape[1]
    N = w.shape[0]
    assert N % NT == 0 and scales.dtype == mx.bfloat16
    y = _kernel(dtype)(
        inputs=[xin, w, scales, biases, xsum, xscale],
        template=[("MT", MT), ("NT", NT), ("KS", KS), ("GS", GS), ("K", K), ("N", N), ("MROWS", M)],
        grid=(N // NT * KS * 32, 1, 1), threadgroup=(KS * 32, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
    return y


def qmm_nax(x, w, scales, biases, group_size=128, bits=4, dtype="bf16"):
    """x: (M, K) bf16 with M <= 16. Returns (M, N) bf16. dtype='i8' expects to_int4() weights."""
    assert bits == 4 and group_size == GS
    return qmm_nax_prepped(prep_x(x, dtype), w, scales, biases, dtype)
