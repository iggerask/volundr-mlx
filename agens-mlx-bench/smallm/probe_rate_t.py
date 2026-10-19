"""Tensor-op rate with weights as the LEFT operand (int8 / bf16) and tokens as the right (N = 8, 16).

The 4-bit formats can only be the right operand, so this only matters for weights unpacked to
int8 or bf16 (the 2- and 3-bit case). Weights cache-resident; no epilogue."""
import mlx.core as mx
from common import compare, spin_up

HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""
SRC = """
  using ext = metal::dextents<int32_t, 2>;
  using strd = metal::array<int32_t, 2>;
  constexpr int G = K / GS;
  const uint sg = simdgroup_index_in_threadgroup;
  const int nw = (int(threadgroup_position_in_grid.x) * WR) % 256;
  // left: weights W[n, k] (WR rows), right: x^T (k x TOK), dest: (WR x TOK)
  metal::tensor<device WT, ext, metal::tensor_inline> tW((device WT*)w, ext(K, 256), strd{1, K});
  metal::tensor<device XT, ext, metal::tensor_inline> tX((device XT*)x, ext(K, TOK), strd{1, K});
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      WR, TOK, GS, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using ws_t = decltype(tW.template slice<GS, WR>(0, 0));
  using xs_t = decltype(tX.template slice<GS, TOK>(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<ws_t, xs_t, AT>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = AT(0);
  for (int g = sg; g < G; g += KS) {
    auto ws = tW.template slice<GS, WR>(g * GS, nw);
    auto xs = tX.template slice<GS, TOK>(g * GS, 0);
    op.run(ws, xs, acc);
  }
  AT s = AT(0);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) s += acc[i];
  y[threadgroup_position_in_grid.x * KS * 32 + thread_index_in_threadgroup] = float(s);
"""
K, N = 17408, 5120
w = mx.random.randint(-8, 8, (256, K)).astype(mx.int8)
wb = w.astype(mx.bfloat16)
mx.eval(w, wb)
fns = {}
for wt, xt, at, xdt, wa in [("int8_t", "int8_t", "int32_t", mx.int8, w), ("int8_t", "bfloat", "float", mx.bfloat16, w),
                            ("bfloat", "bfloat", "float", mx.bfloat16, wb)]:
    kern = mx.fast.metal_kernel(name=f"ratet_{wt}_{xt}", input_names=["x", "w"], output_names=["y"],
                                source=SRC.replace("WT", wt).replace("XT", xt).replace("AT", at), header=HEADER)
    for wr, tok in [(32, 8), (32, 16), (64, 8), (64, 16)]:
        x = mx.random.randint(-3, 3, (tok, K)).astype(xdt)
        mx.eval(x)
        fns[f"W {wt} (left, {wr} rows) x X {xt} (right, {tok} tok)"] = (
            lambda kern=kern, x=x, wa=wa, wr=wr, tok=tok: kern(
                inputs=[x, wa], template=[("WR", wr), ("TOK", tok), ("KS", 8), ("GS", 128), ("K", K)],
                grid=(N // wr * 8 * 32, 1, 1), threadgroup=(8 * 32, 1, 1),
                output_shapes=[(N // wr * 8 * 32,)], output_dtypes=[mx.float32])[0])
spin_up()
t = compare(fns, K=8, reps=5)
for name, v in t.items():
    tok = int(name.split(", ")[-1].split()[0])
    print(f"  {name:55s} {v*1e3:7.3f} ms  {2*16*N*K/v/1e12:5.1f} TFLOP/s-at-16-tok-equiv  "
          f"{2*tok*N*K/v/1e12:5.1f} TFLOP/s useful")
