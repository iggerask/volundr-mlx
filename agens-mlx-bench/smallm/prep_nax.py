"""Fused input prep for qmm_nax: the op that produces a linear's input also writes it padded to
16 rows, plus its per-group sums, in one metal_kernel launch.

qmm_nax_prepped() needs x as (16, K) bf16 (rows >= M zero) and xsum[m, g] = sum of the bf16 values
of x[m] over group g. Done as pad + astype + sum, that is 3 MLX ops and up to 0.25x of a matmul.
Here each producer writes both directly:

  prep(x)                   plain copy (2-D, any strides)
  rms_prep(x, w)            mx.fast.rms_norm(x, w, 1e-6)
  swiglu_prep(gu)           silu(gu[:, :n]) * gu[:, n:]
  sgate_prep(o, gate)       o * sigmoid(gate), o and gate as (T, H, D) views of any strides
                            (SDPA output transposed, and a slice of the q projection)
  kda_out_prep(o, w, z)     rms_norm(o, w) per 128-wide head, times silu(z)

Layout: one threadgroup per padded row (16 of them); a simdgroup takes one group at a time, each
lane GS/32 consecutive elements, so a group sum is one simd_sum. The returned tuple is the same
(M, xp, xsum, xscale) that qmm_nax.prep_x(x, "bf16") returns, so several matmuls that read the same
input share one call. gs=64 builds the sums for 8-bit g64 weights (qmm_nax_u8).
"""
import mlx.core as mx

MT, NSG, EPS = 16, 8, "1e-6f"

SOURCE = """
  constexpr int G = K / GS;
  constexpr int EPT = GS / 32;
  const int m = int(threadgroup_position_in_grid.x);
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const bool live = m < M;
  float r = 1.0f;
#if ROWNORM
  {
    threadgroup float part[NSG];
    float ss = 0.0f;
    if (live)
      for (int k = int(thread_index_in_threadgroup); k < K; k += NSG * 32) {
        const float v = float(x[m * K + k]);
        ss += v * v;
      }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (int i = 0; i < NSG; i++) ss += part[i];
    r = metal::rsqrt(ss / float(K) + EPS);
  }
#endif
  for (int g = sg; g < G; g += NSG) {
    const int k0 = g * GS + int(lane) * EPT;
    float v[EPT];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < EPT; j++) v[j] = live ? LOAD(k0 + j) : 0.0f;
#if GROUPNORM
    // KDA output: rms_norm over the 128-wide head, which is exactly one group
    float ss = 0.0f;
    for (int j = 0; j < EPT; j++) ss += v[j] * v[j];
    const float rg = metal::rsqrt(simd_sum(ss) / float(GS) + EPS);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < EPT; j++) v[j] = live ? FINISH(k0 + j, v[j], rg) : 0.0f;
#endif
    float s = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < EPT; j++) {
      const bfloat b = bfloat(v[j]);
      xp[m * K + k0 + j] = b;
      s += float(b);
    }
    s = simd_sum(s);
    if (lane == 0) xsum[m * G + g] = s;
  }
"""

HELPERS = """
inline float silu_f(float z) { return z / (1.0f + metal::exp(-z)); }
inline float sigmoid_f(float z) { return 1.0f / (1.0f + metal::exp(-z)); }
"""

# variant -> (inputs, ensure_row_contiguous, defines)
_VARIANTS = {
    "plain": (["x"], False, {
        "LOAD(k)": "float(x[m * x_strides[0] + (k) * x_strides[1]])"}),
    "rms": (["x", "w"], True, {
        "ROWNORM": "1",
        "LOAD(k)": "float(bfloat(float(bfloat(float(x[m * K + (k)]) * r)) * float(w[k])))"}),
    "swiglu": (["gu"], True, {
        "LOAD(k)": "float(bfloat(silu_f(float(gu[m * 2 * K + (k)])))) * float(gu[m * 2 * K + K + (k)])"}),
    # o, gate: (T, H, D) views; k = h * D + d
    "sgate": (["o", "gate"], False, {
        "LOAD(k)": "(float(o[m * o_strides[0] + ((k) / D) * o_strides[1] + ((k) % D) * o_strides[2]])"
                   " * sigmoid_f(float(gate[m * gate_strides[0] + ((k) / D) * gate_strides[1]"
                   " + ((k) % D) * gate_strides[2]])))"}),
    "kdaout": (["o", "w", "z"], True, {
        "GROUPNORM": "1",
        "LOAD(k)": "float(o[m * K + (k)])",
        "FINISH(k, v, rg)": "(float(bfloat(float(bfloat((v) * (rg))) * float(w[(k) % GS])))"
                            " * float(bfloat(silu_f(float(z[m * K + (k)])))))"}),
}
_kernels = {}


def _kernel(variant):
    if variant not in _kernels:
        inputs, contig, defs = _VARIANTS[variant]
        d = {"ROWNORM": "0", "GROUPNORM": "0", "EPS": EPS, **defs}
        src = "".join(f"#define {k} {v}\n" for k, v in d.items()) + SOURCE
        _kernels[variant] = mx.fast.metal_kernel(
            name=f"prep_nax_{variant}", input_names=inputs, output_names=["xp", "xsum"],
            source=src, header=HELPERS, ensure_row_contiguous=contig)
    return _kernels[variant]


def _run(variant, inputs, M, K, gs=128, extra=()):
    assert M <= MT and K % gs == 0 and gs % 32 == 0
    xp, xsum = _kernel(variant)(
        inputs=inputs,
        template=[("K", K), ("GS", gs), ("M", M), ("NSG", NSG), *extra],
        grid=(MT * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(MT, K), (MT, K // gs)], output_dtypes=[mx.bfloat16, mx.float32])
    return M, xp, xsum, xsum


def prep(x, gs=128):
    M, K = x.shape
    return _run("plain", [x], M, K, gs)


def rms_prep(x, w, gs=128):
    M, K = x.shape
    return _run("rms", [x, w], M, K, gs)


def swiglu_prep(gu, gs=128):
    M, K2 = gu.shape
    return _run("swiglu", [gu], M, K2 // 2, gs)


def sgate_prep(o, gate, gs=128):
    """o, gate: (T, H, D) arrays or views."""
    M, H, D = o.shape
    return _run("sgate", [o, gate], M, H * D, gs, extra=[("D", D)])


def kda_out_prep(o, w, z):
    """o: (T, H, 128) contiguous, w: (128,), z: (T, H * 128)."""
    M, H, D = o.shape
    assert D == 128
    return _run("kdaout", [o, w, z], M, H * D, 128)


def unprep(px):
    """The (M, K) bf16 activations of a prep result, for consumers that are not NAX matmuls."""
    return px[1][: px[0]]
