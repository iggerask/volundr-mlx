"""Does MPP matmul2d accept MLX-packed 4-bit weights as a uint4b_format right operand?"""
import mlx.core as mx

header = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""
def make(MT, NT, KT):
    src = f"""
  using ext = metal::dextents<int32_t, 2>;
  using strd = metal::array<int32_t, 2>;
  metal::tensor<device bfloat, ext, metal::tensor_inline> tX((device bfloat*)x, ext({KT}, {MT}), strd{{1, {KT}}});
  metal::tensor<device metal::uint4b_format, ext, metal::tensor_inline> tW((device uchar*)w, ext({KT}, {NT}), strd{{1, {KT}}});
  metal::tensor<device float, ext, metal::tensor_inline> tY((device float*)y, ext({NT}, {MT}), strd{{1, {NT}}});
  constexpr auto desc = tensor_ops::matmul2d_descriptor({MT}, {NT}, {KT}, false, true, false,
      tensor_ops::matmul2d_descriptor::mode::multiply);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  op.run(tX, tW, tY);
"""
    return mx.fast.metal_kernel(name=f"probe_u4_{MT}_{NT}_{KT}", input_names=["x", "w"],
                                output_names=["y"], source=src, header=header)

for MT, NT, KT in [(8, 32, 128), (16, 32, 128), (8, 16, 64), (16, 64, 128)]:
    k = make(MT, NT, KT)
    x = mx.random.normal((MT, KT)).astype(mx.bfloat16)
    wint = mx.random.randint(0, 16, (NT, KT)).astype(mx.uint32)
    shifts = (mx.arange(8, dtype=mx.uint32) * 4)
    packed = (wint.reshape(NT, KT // 8, 8) << shifts).sum(axis=-1).astype(mx.uint32)  # MLX layout
    y = k(inputs=[x, packed], grid=(32, 1, 1), threadgroup=(32, 1, 1),
          output_shapes=[(MT, NT)], output_dtypes=[mx.float32])[0]
    ref = x.astype(mx.float32) @ wint.astype(mx.float32).T
    # also verify against mlx's own dequantize with scale 1 bias 0
    s = mx.ones((NT, KT // 64 if KT >= 64 else 1), dtype=mx.bfloat16)
    mx.eval(y, ref)
    print(MT, NT, KT, "max abs err", mx.abs(y - ref).max().item(), "ref max", mx.abs(ref).max().item())
