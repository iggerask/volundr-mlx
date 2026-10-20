"""decode_fused.py kernels are bit-exact against the kernel / MLX-op sequence they replace, and the fused model equals
the unfused one bit for bit (GPU, bf16). Same runner as test_volundr.py.

run: python tests/test_decode_fused.py [substring ...]
"""
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import decode_fused as DF  # noqa: E402
import kernels  # noqa: E402
import volundr  # noqa: E402
from tiny import TINY, tiny_model  # noqa: E402

RNG = np.random.default_rng(4321)
BF, F32 = mx.bfloat16, mx.float32


def bf(*sh, sc=1.0):
    return (mx.array(RNG.standard_normal(sh).astype(np.float32)) * sc).astype(BF)


def same(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    if not mx.array_equal(a, b, equal_nan=True).item():
        diff = (a != b).astype(F32)
        raise AssertionError(f"{what}: {int(diff.sum().item())} of {a.size} differ, max abs "
                             f"{mx.abs(a.astype(F32) - b.astype(F32)).max().item():.3e}")


def _statics(scale=1.0):
    return bf(4) * 0.3, bf(4) * 0.3 + 1, (bf(4, 4) + 4 * mx.eye(4).astype(BF)) * scale


def test_mix_norm():
    """mhc_params + mhc_mix + the input RMSNorm (kernels.py) == DF.mix_norm; also the no-norm variant (MHC.mix)."""
    for N, D, dsc, ssc in ((1, 5120, 0.5, 1.0), (8, 5120, 2.0, 1.0), (4096, 256, 0.5, 1.0), (3000, 128, 3.0, 2.5),
                           (5, 640, 0.05, 0.1), (2048, 5120, 1.0, 1.0), (7, 3000, 1.0, 1.0), (3, 1024, 1.0, 1.0)):
        d = bf(N, 24, sc=dsc)
        pre_s, post_s, res_s = _statics(ssc)
        H = bf(N, 4, D)
        w = bf(D, sc=0.1)
        pre, post, res = kernels.mhc_params(d, pre_s, post_s, res_s, 10)
        x_in = kernels.mhc_mix(H, pre)
        h = kernels.rms_norm(x_in, w, 1e-6, offset=True)[1]
        gx, gh, gp, gr = DF.mix_norm(d, pre_s, post_s, res_s, H, 10, w=w, eps=1e-6)
        for a, b, n in ((gx, x_in, "x_in"), (gh, h, "h"), (gp, post, "post"), (gr, res, "res")):
            same(a, b, f"{n} N={N} D={D}")
        gx, gh, gp, gr = DF.mix_norm(d, pre_s, post_s, res_s, H, 10)
        assert gh is None
        for a, b, n in ((gx, x_in, "x_in"), (gp, post, "post"), (gr, res, "res")):
            same(a, b, f"{n} (no norm) N={N} D={D}")


def test_combine_norm():
    """y = h + m (MLX bf16 add) -> kernels.mhc_combine -> kernels.rms_norm(mean, offset=False) == DF.combine_norm."""
    for N, D in ((1, 5120), (8, 5120), (777, 640), (2048, 128), (64, 5120), (7, 3000), (2, 1024)):
        d = bf(N, 24, sc=0.5)
        pre_s, post_s, res_s = _statics()
        _, post, res = kernels.mhc_params(d, pre_s, post_s, res_s, 10)
        H, hr, m, x_in, w = bf(N, 4, D), bf(N, D), bf(N, D, sc=0.3), bf(N, D), bf(D, sc=0.2) + 1
        y = hr + m
        Hn, mean = kernels.mhc_combine(H, res, post, y, x_in)
        hn = kernels.rms_norm(mean, w, 1e-6, offset=False)[1]
        gH, ghn = DF.combine_norm(H, res, post, (hr, m), x_in, w)
        same(gH, Hn, f"H' add N={N} D={D}")
        same(ghn, hn, f"hn add N={N} D={D}")
        gH, ghn = DF.combine_norm(H, res, post, y, x_in, w)
        same(gH, Hn, f"H' N={N} D={D}")
        same(ghn, hn, f"hn N={N} D={D}")
        gH, ghn = DF.combine_norm(H, res, post, (hr, m), x_in)
        assert ghn is None
        same(gH, Hn, f"H' no norm N={N} D={D}")


def test_kda_conv_gates():
    """kernels.kda_conv_qk4 + concatenate([state, qkv])[:, -3:] + mlx_lm compute_g + sigmoid(b) == one kernel."""
    from mlx_lm.models.gated_delta import compute_g

    Hk, Dk, HV = 16, 128, 48
    key_dim = Hk * Dk
    C = 2 * key_dim + HV * Dk
    A_log = mx.array(RNG.standard_normal(HV).astype(np.float32) * 1.5)
    dt_bias = mx.array(RNG.standard_normal(HV * Dk).astype(np.float32) * 2.0)
    wc = mx.array(RNG.standard_normal((C, 4)).astype(np.float32) * 0.5)
    for B, L, asc in ((1, 1, 1.0), (1, 2, 4.0), (1, 3, 0.3), (1, 8, 8.0), (2, 19, 2.0), (1, 64, 1.0), (1, 131, 30.0)):
        qkv, state = bf(B, L, C), bf(B, 3, C)
        a, b = bf(B, L, HV * Dk, sc=asc), bf(B, L, HV, sc=asc)
        q, k, v = kernels.kda_conv_qk4(qkv, state, wc, key_dim, Dk)
        cs = mx.concatenate([state, qkv], axis=1)[:, -3:]
        g = compute_g(A_log.reshape(HV, 1), a.reshape(B, L, HV, Dk), dt_bias.reshape(HV, Dk))
        beta = mx.sigmoid(b.astype(F32))
        out = DF.kda_conv_gates(qkv, state, wc, key_dim, A_log, a, dt_bias, b)
        for got, want, n in zip(out, (q, k, v, cs, g.reshape(B, L, HV * Dk), beta), ("q", "k", "v", "state", "g", "beta")):
            same(got, want, f"{n} B={B} L={L}")


def test_gated_delta_bf16_out():
    """out_dtype=bf16 in the recurrence kernels == their fp32 output cast to bf16 (both variants)."""
    B, T, Hk, Hv, Dk = 1, 9, 2, 6, 128
    q = mx.array(RNG.standard_normal((B, T, Hk, Dk)).astype(np.float32)) * 0.1
    k = mx.array(RNG.standard_normal((B, T, Hk, Dk)).astype(np.float32)) * 0.1
    v = bf(B, T, Hv, Dk)
    g = mx.sigmoid(mx.array(RNG.standard_normal((B, T, Hv, Dk)).astype(np.float32)) + 2)
    beta = mx.sigmoid(mx.array(RNG.standard_normal((B, T, Hv)).astype(np.float32)))
    st = mx.array(RNG.standard_normal((B, Hv, Dk, Dk)).astype(np.float32) * 0.1)
    for f in (kernels.gated_delta_vec, kernels.gated_delta_tg):
        y, s = f(q, k, v, g, beta, st)
        yb, sb = f(q, k, v, g, beta, st, out_dtype=BF)
        same(yb, y.astype(BF), f.__name__)
        same(sb, s, f.__name__ + " state")


def test_rms_norm_mlx():
    """mx.fast.rms_norm(x.astype(f32), w32).astype(bf16) (OffsetRMSNorm / RMSNorm) == DF.rms_norm_mlx, contiguous and
    strided inputs (the q half of q_proj's output, a transposed view)."""
    for shape, D in (((1, 1, 24, 512), 256), ((1, 8, 24, 512), 256), ((1, 1, 4, 128), 128), ((3, 77, 256), 256),
                     ((1, 8, 4, 128), 128), ((5, 1000), 1000), ((2, 3, 4096), 4096), ((7, 96), 96)):
        x = bf(*shape, sc=3.0)
        xs = x[..., :D]
        w32 = 1.0 + bf(D, sc=0.2).astype(F32)
        want = mx.fast.rms_norm(xs, w32, 1e-6).astype(BF)
        same(DF.rms_norm_mlx(xs, w32, 1e-6), want, f"rms {shape}[..., :{D}]")
    xt = bf(4, 128, 64).swapaxes(-1, -2)
    w32 = bf(128).astype(F32)
    same(DF.rms_norm_mlx(xt, w32, 1e-5), mx.fast.rms_norm(xt, w32, 1e-5).astype(BF), "rms transposed")


# kernels.rms_norm as it was before its row loops got a constant trip count (the reference for that change)
_OLD_NORM_SRC = """
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const size_t base = size_t(row) * D;
    threadgroup float red[NT / 32];
    float hv[(D + NT - 1) / NT];
    float ss = 0.0f;
    int k = 0;
    for (int i = tid; i < D; i += NT, k++) {
        LOAD
        hv[k] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        float t = thread_index_in_simdgroup < NT / 32 ? red[thread_index_in_simdgroup] : 0.0f;
        t = simd_sum(t);
        if (thread_index_in_simdgroup == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs = metal::precise::rsqrt(red[0] / float(D) + EPS_VAL);
    k = 0;
    for (int i = tid; i < D; i += NT, k++) y[base + i] = bfloat16_t(hv[k] * rs * (OFFSET + float(w[i])));
"""


def test_rms_norm_unrolled_equals_strided():
    """kernels.rms_norm (constant-trip row loops) == the strided-loop kernel it replaced, with and without the add."""
    for add in (False, True):
        src = _OLD_NORM_SRC.replace("EPS_VAL", "1e-06f").replace(
            "LOAD", kernels._NORM_LOAD_ADD if add else kernels._NORM_LOAD)
        old = mx.fast.metal_kernel(name=f"t_old_norm_{int(add)}", input_names=["x", "w"] + (["r"] if add else []),
                                   output_names=(["h"] if add else []) + ["y"], source=src)
        for N, D, offset in ((1, 5120, True), (8, 5120, False), (300, 640, True), (17, 128, True), (5, 1000, False),
                             (3, 96, True), (7, 3000, True), (2, 1024, False), (4, 6144, True)):
            x, r, w = bf(N, D), bf(N, D), bf(D, sc=0.1)
            nt = 256 if D >= 1024 else 32 * max(1, min(8, D // 32))
            want = old(inputs=[x, w] + ([r] if add else []), template=[("D", D), ("NT", nt), ("OFFSET", int(offset))],
                       grid=(nt, N, 1), threadgroup=(nt, 1, 1), output_shapes=[x.shape] * (2 if add else 1),
                       output_dtypes=[BF] * (2 if add else 1))
            got = kernels.rms_norm(x, w, 1e-6, offset=offset, r=r if add else None)
            same(got[1], want[-1], f"norm add={add} N={N} D={D}")
            if add:
                same(got[0], want[0], f"h add N={N} D={D}")


def test_bcsa_views_equal_integer_index():
    """volundr._first / _gather_blocks (views) == the integer indexing they replace (x[0], p[b], which copy)."""
    for B in (1, 2):
        p = bf(B, 4, 300, 256)
        idx = mx.array(RNG.integers(0, 300, size=(B, 1, 512)).astype(np.int32))
        want = [mx.take(p[b], idx[b].reshape(-1), axis=1).reshape(4, 1, 512, 256) for b in range(B)]
        want = want[0][None] if B == 1 else mx.stack(want)
        same(volundr._gather_blocks(p, idx), want, f"gather B={B}")
        same(volundr._first(p), p[0], f"first B={B}")


def _paths(model, ids, chunk):
    x = mx.array(ids[None], dtype=mx.int32)
    outs = [model(x)]
    c = model.make_cache()
    outs.append(mx.concatenate([model(x[:, s : s + chunk], cache=c) for s in range(0, len(ids), chunk)], 1))
    c = model.make_cache()
    outs.append(mx.concatenate([model(x[:, t : t + 1], cache=c) for t in range(len(ids))], 1))
    mx.eval(outs)
    return outs


def test_tiny_model_fused_equals_unfused():
    """Tiny model (every mixer kind, Engram, BCSA far regime), bf16 on the GPU: logits with the fused decode kernels ==
    without, bit for bit: one-shot prefill, chunked prefill and token-by-token decode."""
    m = tiny_model(1)
    ids = np.random.default_rng(7).integers(0, TINY["vocab_valid"], size=60)
    try:
        DF.ENABLED = False
        ref = _paths(m, ids, 17)
        DF.ENABLED = True
        got = _paths(m, ids, 17)
    finally:
        DF.ENABLED = True
    for n, a, b in zip(("full", "chunked", "decode"), got, ref):
        same(a, b, f"tiny {n}")


def main():
    sel = sys.argv[1:]
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and (not sel or any(s in n for s in sel))]
    failed = 0
    for n, f in tests:
        t = time.time()
        try:
            f()
            print(f"PASS {n} ({time.time() - t:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
