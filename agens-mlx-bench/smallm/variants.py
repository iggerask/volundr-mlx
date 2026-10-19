"""Experimental variants of the NAX small-M kernel, to find what limits bandwidth.

python variants.py <shape> <spec> [<spec> ...]
  spec: comma-separated key=value from mt, nt, ks, gps, cg, epi (see parse()).
  Every spec is timed interleaved with stock mx.quantized_matmul at M=1, best of --reps.

Knobs:
  MT   token rows per tile (8 or 16); NT weight rows per simdgroup tile
  KS   simdgroups per threadgroup; they split the K groups and reduce in threadgroup memory
  GPS  consecutive 128-groups a simdgroup handles per step (GPS=2 reads full 128-byte lines)
  CG   1: each simdgroup takes one contiguous run of K; 0: steps interleave across simdgroups
  STAGE 1: copy weight tiles to threadgroup memory with wide loads, one step ahead, and run the
       tensor op on the threadgroup copy (0: the tensor op reads device memory itself)
  EPI  full   = production math: per-group multiply, acc += s*D + b*sum(x_g)
       none   = one multiply_accumulate per group, no scales (tensor-op ceiling)
       stream = the same tile walk with plain loads and no tensor op (access-pattern ceiling)
Tried and dropped (slower): touching the next tile with plain loads as a prefetch, and keeping
x in a cooperative left-input tensor. A cooperative uint4 right operand is not supported by MPP.
"""
import argparse
import mlx.core as mx
from common import bench, compare, qweight, qbytes, spin_up, SHAPES, BF  # noqa: F401

HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""

SRC = """
  using ext = metal::dextents<int32_t, 2>;
  using strd = metal::array<int32_t, 2>;
  constexpr int G = K / GS;
  constexpr int NSTEP = G / GPS;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const int n0 = int(threadgroup_position_in_grid.x) * NT;
#if HOT
  // Cache-resident test: every threadgroup reads the same 256 weight rows.
  const int nw = n0 % 256;
#else
  const int nw = n0;
#endif

  metal::tensor<device XT, ext, metal::tensor_inline> tX((device XT*)x, ext(K, MT), strd{1, K});
  metal::tensor<device WF, ext, metal::tensor_inline> tW((device uchar*)w, ext(K, N), strd{1, K});
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      MT, NT, GS, false, true, false, tensor_ops::matmul2d_descriptor::mode::MODE);
  tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using xs_t = decltype(tX.template slice<GS, MT>(0, 0));
  using ws_t = decltype(tW.template slice<GS, NT>(0, 0));
  auto acc = op.template get_destination_cooperative_tensor<xs_t, ws_t, AT>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = AT(0);
  const device bfloat* sp = (const device bfloat*)scales + int64_t(nw) * G;
  const device bfloat* bp = (const device bfloat*)biases + int64_t(nw) * G;
  const device uint4* wrow = (const device uint4*)((const device uchar*)w + int64_t(nw) * (K / 2));
  uint sink = 0;

#if CG
  constexpr int per = (NSTEP + KS - 1) / KS;
  const int s_begin = sg * per;
  const int s_end = min(NSTEP, s_begin + per);
  constexpr int s_inc = 1;
#else
  const int s_begin = sg;
  constexpr int s_end = NSTEP;
  constexpr int s_inc = KS;
#endif
#if STAGE
  // Each simdgroup copies its next weight tile (NT rows x GPS groups) with wide coalesced loads
  // into registers while the tensor op works on the current tile in threadgroup memory.
  constexpr int SEG = 4 * GPS;               // uint4 segments per row per step
  constexpr int PER_LANE = NT * SEG / 32;
  threadgroup uint4 wbuf[KS * NT * SEG];
  threadgroup uint4* mybuf = wbuf + sg * (NT * SEG);
  metal::tensor<threadgroup WF, ext, metal::tensor_inline> tWs(
      (threadgroup uchar*)mybuf, ext(GS * GPS, NT), strd{1, GS * GPS});
  uint4 r[PER_LANE];
  if (s_begin < s_end) {
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < PER_LANE; j++) {
      const int e = int(lane) + 32 * j;
      r[j] = wrow[(e / SEG) * (K / 32) + s_begin * SEG + (e % SEG)];
    }
  }
#endif
  for (int st = s_begin; st < s_end; st += s_inc) {
#if STAGE
    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < PER_LANE; j++) mybuf[int(lane) + 32 * j] = r[j];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (st + s_inc < s_end) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < PER_LANE; j++) {
        const int e = int(lane) + 32 * j;
        r[j] = wrow[(e / SEG) * (K / 32) + (st + s_inc) * SEG + (e % SEG)];
      }
    }
#endif
    _Pragma("clang loop unroll(full)")
    for (int gg = 0; gg < GPS; gg++) {
      const int g = st * GPS + gg;
#if EPI_STREAM
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NT / 8; j++) {
        uint4 v = wrow[(int(lane / 4) + 8 * j) * (K / 32) + g * 4 + (lane % 4)];
        sink ^= v.x ^ v.y ^ v.z ^ v.w;
      }
#elif EPI_NONE
      auto xs = tX.template slice<GS, MT>(g * GS, 0);
      auto ws = tW.template slice<GS, NT>(g * GS, nw);
      op.run(xs, ws, acc);
#else
      auto xs = tX.template slice<GS, MT>(g * GS, 0);
#if STAGE
      auto ws = tWs.template slice<GS, NT>(gg * GS, 0);
#else
      auto ws = tW.template slice<GS, NT>(g * GS, nw);
#endif
      auto d = op.template get_destination_cooperative_tensor<xs_t, decltype(ws), DT>();
      op.run(xs, ws, d);
      _Pragma("clang loop unroll(full)")
      for (uint16_t i = 0; i < d.get_capacity(); i++) {
        auto idx = d.get_multidimensional_index(i);
        const int n = idx[0];
        const int m = idx[1];
#if XINT8
        // int8 activations: x_g ~= xscale[m, g] * xq; weights stored as signed int4 (q - 8).
        acc[i] += float(sp[n * G + g]) * xscale[m * G + g] * float(d[i]) + float(bp[n * G + g]) * xsum[m * G + g];
#else
        acc[i] += float(sp[n * G + g]) * d[i] + float(bp[n * G + g]) * xsum[m * G + g];
#endif
      }
#endif
    }
  }
  // Keeps the plain loads of the stream variant alive; flag is always 0.
  if (flag[0] != 0) y[0] = bfloat(float(sink));

#if STAGE
  // Reuse the staging buffers for the cross-simdgroup reduction.
  static_assert(KS * NT * SEG * 16 >= KS * MT * NT * 4, "staging buffer too small for reduction");
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* red = (threadgroup float*)wbuf;
#else
  threadgroup float red[KS * MT * NT];
#endif
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < acc.get_capacity(); i++) {
    if (acc.is_valid_element(i)) {
      auto idx = acc.get_multidimensional_index(i);
      red[sg * (MT * NT) + idx[1] * NT + idx[0]] = acc[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = thread_index_in_threadgroup; e < MT * NT; e += KS * 32) {
    float s = 0.0f;
    for (int k = 0; k < KS; k++) s += float(red[k * (MT * NT) + e]);
    y[(e / NT) * N + n0 + (e % NT)] = bfloat(s);
  }
"""

_cache = {}
_FLAG = mx.zeros((1,), dtype=mx.int32)


DTYPES = {  # x type, weight format, tensor-op destination type, final accumulator type
    "bf16": ("bfloat", "metal::uint4b_format", "float", "float"),
    "f16": ("half", "metal::uint4b_format", "float", "float"),
    "i8": ("int8_t", "metal::int4b_format", "int32_t", "float"),
}


def build(epi="full", cg=False, stage=False, hot=False, dt="bf16"):
    key = (epi, cg, stage, hot, dt)
    if key not in _cache:
        mode = "multiply" if epi == "full" else "multiply_accumulate"
        xt, wf, dtt, at = DTYPES[dt]
        if epi != "full":
            at = dtt
        src = (f"#define EPI_STREAM {int(epi == 'stream')}\n#define EPI_NONE {int(epi == 'none')}\n"
               f"#define CG {int(cg)}\n#define STAGE {int(stage)}\n#define HOT {int(hot)}\n"
               f"#define XINT8 {int(dt == 'i8')}\n"
               + SRC.replace("MODE", mode).replace("XT", xt).replace("WF", wf).replace("DT", dtt)
               .replace("AT", at))
        _cache[key] = mx.fast.metal_kernel(
            name=f"qmm_var_{epi}_{int(cg)}_{int(stage)}_{int(hot)}_{dt}",
            input_names=["x", "w", "scales", "biases", "xsum", "xscale", "flag"],
            output_names=["y"], source=src, header=HEADER)
    return _cache[key]


def prep_x(x, mt, dt):
    """Pad x to mt rows and convert it for the chosen dtype. Returns (x_in, xsum, xscale)."""
    M, K = x.shape
    if M < mt:
        x = mx.pad(x, [(0, mt - M), (0, 0)])
    xf = x.astype(mx.float32).reshape(mt, K // 128, 128)
    xsum = xf.sum(-1)
    if dt == "i8":
        xscale = mx.maximum(mx.abs(xf).max(-1), 1e-12) / 127.0
        xq = mx.round(xf / xscale[..., None]).astype(mx.int8).reshape(mt, K)
        return xq, xsum, xscale
    return (x.astype(mx.float16) if dt == "f16" else x), xsum, xsum


def to_int4(w, s, b):
    """MLX unsigned 4-bit packing -> signed int4 (q - 8) packing, with the bias folded to match."""
    return w ^ mx.array(0x88888888, dtype=mx.uint32), s, (b.astype(mx.float32) + 8 * s.astype(mx.float32)).astype(b.dtype)


def run(x, w, s, b, xsum, mt=16, nt=32, ks=4, gps=1, cg=0, epi="full", stage=0, hot=0, dt="bf16", xscale=None):
    K = x.shape[1]
    N = w.shape[0]
    return build(epi, bool(cg), bool(stage), bool(hot), dt)(
        inputs=[x, w, s, b, xsum, xsum if xscale is None else xscale, _FLAG],
        template=[("MT", mt), ("NT", nt), ("KS", ks), ("GPS", gps), ("GS", 128), ("K", K), ("N", N)],
        grid=(N // nt * ks * 32, 1, 1), threadgroup=(ks * 32, 1, 1),
        output_shapes=[(mt, N)], output_dtypes=[mx.bfloat16])[0]


def parse(spec):
    d = dict(mt=16, nt=32, ks=4, gps=1, cg=0, epi="full", stage=0, hot=0, dt="bf16")
    for kv in spec.split(","):
        k, v = kv.split("=")
        d[k] = v if k in ("epi", "dt") else int(v)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shape")
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    n, k = SHAPES[a.shape]
    copies = max(4, int(1.5e9 // qbytes(n, k)))
    ws = [qweight(n, k) for _ in range(copies)]
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) % copies
        return ws[i[0]]
    x16 = mx.random.normal((16, k)).astype(BF)
    x1 = x16[:1]
    mx.eval(x16, x1)
    fns = {"stock M=1": lambda: mx.quantized_matmul(x1, *nxt(), transpose=True, group_size=128, bits=4)}
    for spec in a.specs:
        d = parse(spec)
        xin, xsum, xscale = prep_x(x16[:d["mt"]], d["mt"], d["dt"])
        mx.eval(xin, xsum, xscale)
        # The int8 path wants signed-int4 packing; XOR is a stand-in for an offline repack, so time
        # it on pre-converted copies (same bytes, same layout).
        if d["dt"] == "i8" and "ws_i4" not in globals():
            ws_i4 = [to_int4(*t) for t in ws]
            mx.eval(ws_i4)
        src = ws_i4 if d["dt"] == "i8" else ws

        def f(xin=xin, xsum=xsum, xscale=xscale, d=d, src=src):
            nxt()
            return run(xin, *src[i[0]], xsum, xscale=xscale, **d)
        fns[spec] = f
    spin_up()
    t = compare(fns, K=copies, reps=a.reps)
    t1 = t["stock M=1"]
    print(f"{a.shape} ({n}x{k}), best of {a.reps} interleaved:")
    for name, v in t.items():
        print(f"  {name:40s} {v*1e3:7.3f} ms {qbytes(n, k)/v/1e9:5.0f} GB/s  {v/t1:5.2f}x stock M=1")
