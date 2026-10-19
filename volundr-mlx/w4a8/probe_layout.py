"""Which (idx0, idx1) elements does each lane own in the left / right / destination cooperative tensors?

usage: python probe_layout.py TM TN TK [AT BT CT]   (defaults: int8_t int8_t int32_t)
Prints per-lane coordinate lists and a summary of contiguous runs along idx0.
"""
import sys
import mlx.core as mx
from w4common import MPP_HEADER

TM, TN, TK = (int(a) for a in sys.argv[1:4])
AT, BT, CT = sys.argv[4:7] if len(sys.argv) > 6 else ("int8_t", "int8_t", "int32_t")
CAP = 1024
src = f"""
  constexpr auto desc = tensor_ops::matmul2d_descriptor({TM}, {TN}, {TK}, false, true, false,
      tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto a = op.template get_left_input_cooperative_tensor<{AT}, {BT}, {CT}>();
  auto b = op.template get_right_input_cooperative_tensor<{AT}, {BT}, {CT}>();
  auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), {CT}>();
  const uint lane = thread_index_in_simdgroup;
  device int* o = out + lane * 3 * {CAP} * 2;
  o[0] = a.get_capacity(); o[{2 * CAP}] = b.get_capacity(); o[{4 * CAP}] = c.get_capacity();
  for (uint16_t i = 0; i < a.get_capacity(); i++) {{ auto x = a.get_multidimensional_index(i); o[2 + 2 * i] = x[0]; o[3 + 2 * i] = x[1]; }}
  for (uint16_t i = 0; i < b.get_capacity(); i++) {{ auto x = b.get_multidimensional_index(i); o[{2 * CAP} + 2 + 2 * i] = x[0]; o[{2 * CAP} + 3 + 2 * i] = x[1]; }}
  for (uint16_t i = 0; i < c.get_capacity(); i++) {{ auto x = c.get_multidimensional_index(i); o[{4 * CAP} + 2 + 2 * i] = x[0]; o[{4 * CAP} + 3 + 2 * i] = x[1]; }}
"""
k = mx.fast.metal_kernel(name=f"layout_{TM}_{TN}_{TK}_{AT}", input_names=["dummy"], output_names=["out"],
                         source=src, header=MPP_HEADER)
o = k(inputs=[mx.zeros((1,))], grid=(32, 1, 1), threadgroup=(32, 1, 1),
      output_shapes=[(32 * 3 * CAP * 2,)], output_dtypes=[mx.int32], init_value=-1)[0]
o = o.reshape(32, 3, CAP * 2).tolist()


def coords(lane, which):
    row = o[lane][which]
    cap = row[0]
    return [(row[2 + 2 * i], row[3 + 2 * i]) for i in range((cap * 2 - 2) // 2 + 1) if 2 + 2 * i < len(row)][:cap]


for which, name in enumerate(("left (A, M x K)", "right (B^T, N x K)", "dest (C, M x N)")):
    cap = o[0][which][0]
    print(f"== {name}: capacity {cap}")
    for lane in (0, 1, 2, 3, 4, 8, 16, 31):
        c = coords(lane, which)
        runs, r = [], 1
        for p, q in zip(c, c[1:]):
            if q[1] == p[1] and q[0] == p[0] + 1:
                r += 1
            else:
                runs.append(r)
                r = 1
        runs.append(r)
        print(f"  lane {lane:2d}: runs along idx0 {sorted(set(runs))}, first {c[:12]}")
    # distinct idx0 / idx1 values per lane
    c = coords(0, which)
    print(f"  lane 0 distinct idx0 {sorted(set(x for x, _ in c))[:40]}")
    print(f"  lane 0 distinct idx1 {sorted(set(y for _, y in c))[:40]}")
