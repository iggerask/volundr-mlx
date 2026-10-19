"""Can mx.fast.metal_kernel call MetalPerformancePrimitives tensor_ops (matmul2d)?"""
import mlx.core as mx

header = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""
# C (32 x 16, row-major: N=32 cols? ) -- tensors are (inner, outer) extents in MPP: extents(cols, rows)
src = """
  using ext = metal::dextents<int32_t, 2>;
  using strd = metal::array<int32_t, 2>;
  using tdev = metal::tensor<device bfloat16_t, ext, metal::tensor_inline>;
  using tdevf = metal::tensor<device float, ext, metal::tensor_inline>;
  // A: M x K (16 x 16) row-major -> extents (K, M); B: K x N (16 x 32) row-major -> extents (N, K)
  tdev tA((device bfloat16_t*)a, ext(16, 16), strd{1, 16});
  tdev tB((device bfloat16_t*)b, ext(32, 16), strd{1, 32});
  tdevf tC((device float*)c, ext(32, 16), strd{1, 32});
  constexpr auto desc = tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false,
      tensor_ops::matmul2d_descriptor::mode::multiply);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  op.run(tA, tB, tC);
"""
k = mx.fast.metal_kernel(name="probe_mpp", input_names=["a", "b"], output_names=["c"],
                         source=src, header=header)
a = mx.random.normal((16, 16)).astype(mx.bfloat16)
b = mx.random.normal((16, 32)).astype(mx.bfloat16)
c = k(inputs=[a, b], grid=(32, 1, 1), threadgroup=(32, 1, 1),
      output_shapes=[(16, 32)], output_dtypes=[mx.float32])[0]
ref = a.astype(mx.float32) @ b.astype(mx.float32)
mx.eval(c, ref)
print("max abs err", mx.abs(c - ref).max().item(), "ref max", mx.abs(ref).max().item())
