"""Tensor-op throughput by operand types, weights cache-resident (no DRAM), no epilogue.

Every threadgroup multiplies the same 256 x K weight rows, so this measures the neural
accelerator's rate for small-M tiles, not memory bandwidth.
"""
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
  const int nw = (int(threadgroup_position_in_grid.x) * NT) % 256;
  metal::tensor<device XT, ext, metal::tensor_inline> tX((device XT*)x, ext(K, MT), strd{1, K});
  metal::tensor<device WT, ext, metal::tensor_inline> tW((device WP*)w, ext(K, 256), strd{1, K});
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      MT, NT, GS, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using xs_t = decltype(tX.template slice<GS, MT>(0, 0));
  using ws_t = decltype(tW.template slice<GS, NT>(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<xs_t, ws_t, AT>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = AT(0);
  for (int g = sg; g < G; g += KS) {
    auto xs = tX.template slice<GS, MT>(g * GS, 0);
    auto ws = tW.template slice<GS, NT>(g * GS, nw);
    op.run(xs, ws, acc);
  }
  AT s = AT(0);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) s += acc[i];
  y[threadgroup_position_in_grid.x * KS * 32 + thread_index_in_threadgroup] = float(s);
"""
K, N = 17408, 5120
combos = [("bfloat", "metal::uint4b_format", "float", mx.bfloat16),
          ("half", "metal::uint4b_format", "float", mx.float16),
          ("int8_t", "metal::int4b_format", "int32_t", mx.int8),
          ("uint8_t", "metal::uint4b_format", "int32_t", mx.uint8),
          ("bfloat", "int8_t", "float", mx.bfloat16),
          ("int8_t", "int8_t", "int32_t", mx.int8)]
w = mx.random.randint(0, 255, (256, K)).astype(mx.uint8)  # big enough for 8-bit weights too
mx.eval(w)
fns = {}
for xt, wt, at, xdt in combos:
    kern = mx.fast.metal_kernel(name=f"rate_{xt}_{wt.split(':')[-1]}", input_names=["x", "w"],
                                output_names=["y"],
                                source=SRC.replace("XT", xt).replace("WT", wt).replace("AT", at).replace("WP", "uchar" if "4b" in wt else wt),
                                header=HEADER)
    for mt, nt, ks in [(16, 64, 8), (16, 32, 8), (32, 64, 4)]:
        x = mx.random.randint(-3, 3, (mt, K)).astype(xdt)
        mx.eval(x)
        fns[f"{xt} x {wt.split(':')[-1]} mt={mt} nt={nt}"] = (
            lambda kern=kern, x=x, mt=mt, nt=nt, ks=ks: kern(
                inputs=[x, w], template=[("MT", mt), ("NT", nt), ("KS", ks), ("GS", 128), ("K", K)],
                grid=(N // nt * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
                output_shapes=[(N // nt * ks * 32,)], output_dtypes=[mx.float32])[0])
spin_up()
t = compare(fns, K=8, reps=5)
for name, v in t.items():
    mt = int(name.split("mt=")[1].split()[0])
    print(f"  {name:45s} {v*1e3:7.3f} ms  {2*mt*N*K/v/1e12:5.1f} TFLOP/s  "
          f"(= {N*K*0.53125/v/1e9:5.0f} GB/s of 4-bit g128 weights)")
