"""Peak tensor-op rate by operand type and simdgroup descriptor, operands in registers (no memory traffic).

usage: python probe_rate.py
"""
import mlx.core as mx
from w4common import MPP_HEADER, compare, spin_up

SRC = """
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      TM, TN, TK, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
  auto b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
  auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
  const uint lane = thread_index_in_simdgroup;
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < a.get_capacity(); i++) a[i] = AT(x[(lane + i) & 255]);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < b.get_capacity(); i++) b[i] = BT(x[(lane * 3 + i) & 255]);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < c.get_capacity(); i++) c[i] = CT(0);
  for (int it = 0; it < ITERS; it++) op.run(a, b, c);
  CT s = CT(0);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < c.get_capacity(); i++) s += c[i];
  y[threadgroup_position_in_grid.x * NSG * 32 + thread_index_in_threadgroup] = float(s);
"""
TYPES = [("int8_t", "int8_t", "int32_t"), ("bfloat", "bfloat", "float"), ("half", "half", "float")]
DESCS = [(16, 32, 16), (32, 32, 32), (64, 32, 32), (32, 64, 32), (64, 64, 32), (32, 32, 64), (64, 64, 64),
         (16, 64, 128), (32, 32, 128)]
ITERS, NSG, NTG = 256, 4, 640
x = mx.random.randint(-3, 3, (256,)).astype(mx.int8)
mx.eval(x)
fns, flops = {}, {}
for at, bt, ct in TYPES:
    for tm, tn, tk in DESCS:
        name = f"{at:6s} {tm}x{tn}x{tk}"
        src = SRC.replace("TM", str(tm)).replace("TN", str(tn)).replace("TK", str(tk)).replace("ITERS", str(ITERS))
        src = src.replace("NSG", str(NSG)).replace("AT", at).replace("BT", bt).replace("CT", ct)
        try:
            kern = mx.fast.metal_kernel(name=f"rate_{at}_{tm}_{tn}_{tk}", input_names=["x"], output_names=["y"],
                                        source=src, header=MPP_HEADER)
            f = (lambda kern=kern: kern(inputs=[x], grid=(NTG * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                                        output_shapes=[(NTG * NSG * 32,)], output_dtypes=[mx.float32])[0])
            mx.eval(f())
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: failed: {str(e).splitlines()[0][:150]}")
            continue
        fns[name] = f
        flops[name] = 2 * tm * tn * tk * ITERS * NSG * NTG
spin_up()
t = compare(fns, K=4, reps=5)
for name, v in t.items():
    print(f"  {name:22s} {v * 1e3:7.3f} ms  {flops[name] / v / 1e12:5.1f} TFLOP/s")
