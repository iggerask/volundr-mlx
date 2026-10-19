"""Which (k, n) elements does each lane own in a right-input cooperative tensor (int8 weights)?"""
import sys
import mlx.core as mx
HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""
MT, NT, KT, XT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
src = f"""
  constexpr auto desc = tensor_ops::matmul2d_descriptor({MT}, {NT}, {KT}, false, true, false,
      tensor_ops::matmul2d_descriptor::mode::multiply);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto r = op.template get_right_input_cooperative_tensor<{XT}, int8_t, {'int32_t' if XT == 'int8_t' else 'float'}>();
  const uint lane = thread_index_in_simdgroup;
  out[lane * 1024] = r.get_capacity();
  for (uint16_t i = 0; i < r.get_capacity() && i < 500; i++) {{
    auto idx = r.get_multidimensional_index(i);
    out[lane * 1024 + 1 + 2 * i] = idx[0];
    out[lane * 1024 + 2 + 2 * i] = idx[1];
  }}
"""
k = mx.fast.metal_kernel(name=f"layout_{MT}_{NT}_{KT}_{XT}", input_names=["dummy"], output_names=["out"],
                         source=src, header=HEADER)
o = k(inputs=[mx.zeros((1,))], grid=(32, 1, 1), threadgroup=(32, 1, 1),
      output_shapes=[(32 * 1024,)], output_dtypes=[mx.int32], init_value=-1)[0]
o = o.reshape(32, 1024).tolist()
cap = o[0][0]
print("capacity per lane:", cap)
for lane in (0, 1, 2, 3, 4, 8, 31):
    pairs = [(o[lane][1 + 2 * i], o[lane][2 + 2 * i]) for i in range(min(cap, 40))]
    print(f"lane {lane:2d}: (idx0, idx1) =", pairs)
