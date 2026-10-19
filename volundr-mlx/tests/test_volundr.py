"""Op-level and tiny-model tests against the NumPy reference (tests/reference.py: release torch code + reference.rs).

run: python tests/test_volundr.py [substring ...]
"""
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import convert  # noqa: E402
import kernels  # noqa: E402
import volundr  # noqa: E402
from reference import Ref, engram_hash_last, engram_hash_seq, sinkhorn as ref_sinkhorn, w4_pack_row  # noqa: E402
from tiny import TINY, tiny_model  # noqa: E402

RNG = np.random.default_rng(1234)


def np_params(model):
    return {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


def ids_for(T, seed=0):
    return np.random.default_rng(seed).integers(0, TINY["vocab_valid"], size=T)


def run_paths(model, ids, chunk=13):
    x = mx.array(ids[None], dtype=mx.int32)
    full = model(x)
    c = model.make_cache()
    chunked = mx.concatenate([model(x[:, s : s + chunk], cache=c) for s in range(0, len(ids), chunk)], 1)
    c = model.make_cache()
    dec = mx.concatenate([model(x[:, t : t + 1], cache=c) for t in range(len(ids))], 1)
    return [np.array(a[0].astype(mx.float32)) for a in (full, chunked, dec)]


# ------------------------------------------------------------------------------------------ ops
def test_engram_hash():
    rows = 1 << 20
    assert engram_hash_last([5], 2, rows) == 6  # reference.rs hand values
    assert engram_hash_last([5, 7], 2, rows) == (6 * 1000033 + 8) % rows
    ids = [int(i) for i in RNG.integers(0, 248320, size=257)] + [248045, 248046, 248069, 0, 1]
    for r in (rows, 1024):
        for n in (2, 3):
            ref = engram_hash_seq(ids, n, r)
            got = volundr.engram_rows(mx.array([[-1, -1] + ids], dtype=mx.int32), n, len(ids), r)
            assert np.array_equal(np.array(got[0]), ref), (r, n)
            # incremental: the previous two ids carried in front of a later chunk
            got2 = volundr.engram_rows(mx.array([ids[98:100] + ids[100:]], dtype=mx.int32), n, len(ids) - 100, r)
            assert np.array_equal(np.array(got2[0]), ref[100:]), (r, n, "chunked")


def test_sinkhorn_kernel():
    logits = (RNG.standard_normal((4000, 16)) * 3).astype(np.float32)
    logits[:10] += np.eye(4).reshape(16) * 16  # identity-like init regime
    ref = ref_sinkhorn(logits.astype(np.float64).reshape(-1, 4, 4), 10)
    got = np.array(volundr.sinkhorn(mx.array(logits), 10))
    ops = np.array(volundr.sinkhorn_ops(mx.array(logits).reshape(-1, 4, 4), 10))
    assert np.abs(got - ref).max() < 2e-6, np.abs(got - ref).max()
    assert np.abs(ops - ref).max() < 2e-6, np.abs(ops - ref).max()


def test_gated_delta_vec_kernel():
    """mlx-lm's gated_delta kernel with a per-channel (4-D) gate == the KDA recurrence of kda_decode.cu."""
    from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

    B, T, Hk, HV, K, V = 1, 29, 2, 6, 128, 128
    q = RNG.standard_normal((B, T, Hk, K)).astype(np.float32)
    k = RNG.standard_normal((B, T, Hk, K)).astype(np.float32)
    v = RNG.standard_normal((B, T, HV, V)).astype(np.float32)
    a = RNG.standard_normal((B, T, HV, K)).astype(np.float32)
    b = RNG.standard_normal((B, T, HV)).astype(np.float32)
    A_log = RNG.standard_normal(HV).astype(np.float32) * 0.5
    dt = RNG.standard_normal(HV * K).astype(np.float32) * 0.5
    vb = mx.array(v).astype(mx.bfloat16)
    ab = mx.array(a).astype(mx.bfloat16)
    qn, kn = normalize_qk(mx.array(q), mx.array(k), inv_scale=K**-0.5, eps=1e-6)
    args = lambda sl: (qn[:, sl], kn[:, sl], vb[:, sl], ab[:, sl], mx.array(b)[:, sl], mx.array(A_log).reshape(HV, 1),
                       mx.array(dt).reshape(HV, K))
    y, S = gated_delta_update(*args(slice(None)))
    # step by step through the same kernel (decode) == one launch over T (prefill), bit for bit
    st, ys = None, []
    for t in range(T):
        yt, st = gated_delta_update(*args(slice(t, t + 1)), st)
        ys.append(yt)
    assert np.array_equal(np.array(mx.concatenate(ys, 1)), np.array(y)) and np.array_equal(np.array(st), np.array(S))
    # against the fp64 recurrence
    qd = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * K**-0.5
    kd = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    vd = np.array(vb.astype(mx.float32), np.float64)
    ad = np.array(ab.astype(mx.float32), np.float64)
    g = np.exp(-np.exp(A_log.astype(np.float64))[:, None] * np.logaddexp(ad + dt.reshape(HV, K), 0))
    beta = 1 / (1 + np.exp(-b.astype(np.float64)))
    Sr = np.zeros((HV, V, K))
    yr = np.zeros((T, HV, V))
    for t in range(T):
        for h in range(HV):
            kh = h // (HV // Hk)
            Sr[h] *= g[0, t, h][None]
            Sr[h] += np.outer((vd[0, t, h] - Sr[h] @ kd[0, t, kh]) * beta[0, t, h], kd[0, t, kh])
            yr[t, h] = Sr[h] @ qd[0, t, kh]
    assert rel(np.array(y[0]), yr) < 1e-5, rel(np.array(y[0]), yr)
    assert rel(np.array(S[0]), Sr) < 1e-5


def _container(q_rows, scale_rows):
    """q [n, k] ints, scales [n, k/g] -> (weight_packed I32 [n, k/8], weight_scale BF16) via reference.rs packing."""
    packed = np.stack([w4_pack_row(r) for r in q_rows])
    return mx.array(packed), mx.array(scale_rows).astype(mx.bfloat16)


def test_convert_linear():
    n, k = 24, 512
    x = mx.array(RNG.standard_normal((5, k)).astype(np.float32)).astype(mx.bfloat16)
    for bits in (4, 3, 2):
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        q = RNG.integers(lo, hi + 1, size=(n, k))
        for group, stored in ((128, 128), (128, 64), (64, 64)):
            s = np.abs(RNG.standard_normal((n, k // group))).astype(np.float32) * 0.01 + 1e-3
            s_st = np.repeat(s, group // stored, axis=1)  # g128 stored as g64 with repeated scales
            packed, scale = _container(q, s_st)
            w, sc, bi, b2, g2 = convert.convert_linear("t", packed, scale, (n, k), bits, group)
            assert (b2, g2) == (bits, group)
            deq = mx.dequantize(w, sc, bi, group_size=group, bits=bits).astype(mx.float32)
            sf = np.array(scale.astype(mx.float32))
            exact = np.array(mx.array(q * np.repeat(sf, stored, axis=1)).astype(mx.bfloat16).astype(mx.float32))
            assert np.array_equal(np.array(deq), exact), (bits, group, stored)
            y = mx.quantized_matmul(x, w, sc, bi, transpose=True, group_size=group, bits=bits)
            yr = x.astype(mx.float32) @ mx.array(exact).T
            assert rel(np.array(y.astype(mx.float32)), np.array(yr)) < 2e-2
    # scales that do not repeat over the requested group keep the stored group
    q = RNG.integers(-4, 4, size=(n, k))
    packed, scale = _container(q, np.abs(RNG.standard_normal((n, k // 64))).astype(np.float32) + 0.1)
    assert convert.convert_linear("t", packed, scale, (n, k), 3, 128)[4] == 64
    # values outside the declared width are rejected
    packed, scale = _container(RNG.integers(-8, 8, size=(n, k)), np.ones((n, k // 64), np.float32))
    try:
        convert.convert_linear("t", packed, scale, (n, k), 2, 64)
        raise AssertionError("out-of-range W2 accepted")
    except ValueError:
        pass


def test_topk_tie_order():
    s = mx.array([[1.0, 3.0, 3.0, 2.0, 3.0, -np.inf, 3.0, -1.0]]).astype(mx.bfloat16)
    assert sorted(np.array(volundr._topk_blocks(s, 3))[0].tolist()) == [1, 2, 4]
    s = mx.array([[-1.0, -2.0, -0.5, -np.inf]]).astype(mx.bfloat16)
    assert np.array(volundr._topk_blocks(s, 1))[0].tolist() == [2]
    s = mx.array(RNG.standard_normal((3, 700)).astype(np.float32)).astype(mx.bfloat16)
    got = np.sort(np.array(volundr._topk_blocks(s, 50)), -1)
    sf = np.array(s.astype(mx.float32))
    ref = np.sort(np.argsort(-sf, axis=-1, kind="stable")[:, :50], -1)
    assert np.array_equal(got, ref)


def test_kernels_match_ops():
    """Each fused Metal kernel against the MLX ops path it replaces (same rounding points)."""
    bf = lambda *sh, sc=1.0: (mx.array(RNG.standard_normal(sh).astype(np.float32)) * sc).astype(mx.bfloat16)
    N, D = 37, 640
    # mHC params (pre/post + Sinkhorn)
    d, pre_s, post_s, res_s = bf(N, 24, sc=0.5), bf(4), bf(4) + 1, bf(4, 4) + 4 * mx.eye(4).astype(mx.bfloat16)
    pre, post, res = kernels.mhc_params(d, pre_s, post_s, res_s, 10)
    pre_r, post_r, logits = volundr._mhc_params(d[None], pre_s, post_s, res_s)
    res_r = volundr.sinkhorn(logits, 10).astype(mx.bfloat16)[0]
    assert mx.array_equal(pre, pre_r[0]) and mx.array_equal(post, post_r[0])
    assert mx.abs(res.astype(mx.float32) - res_r.astype(mx.float32)).max().item() <= 2 ** -8
    # mix / combine
    H, y, = bf(N, 4, D), bf(N, D)
    x_in = kernels.mhc_mix(H, pre)
    assert mx.array_equal(x_in, volundr._mhc_collapse(pre[None], H[None])[0])
    Hn, mean = kernels.mhc_combine(H, res, post, y, x_in)
    Hr, mr = volundr._mhc_combine((res @ H)[None], post[None], y[None], x_in[None])
    assert mx.abs(Hn.astype(mx.float32) - Hr[0].astype(mx.float32)).max().item() < 2e-2  # bmm vs FMA order
    assert (Hn != Hr[0]).astype(mx.float32).mean().item() < 0.02
    # RMSNorm (+ residual add)
    x, r, w = bf(N, D), bf(N, D), bf(D, sc=0.1)
    _, yk = kernels.rms_norm(x, w, 1e-6, offset=True)
    yr = mx.fast.rms_norm(x, 1.0 + w.astype(mx.float32), 1e-6).astype(mx.bfloat16)
    assert (yk != yr).astype(mx.float32).mean().item() < 0.01 and mx.abs((yk - yr).astype(mx.float32)).max() < 0.05
    hk, yk = kernels.rms_norm(x, w, 1e-6, offset=False, r=r)
    assert mx.array_equal(hk, x + r)
    yr = mx.fast.rms_norm(x + r, w.astype(mx.float32), 1e-6).astype(mx.bfloat16)
    assert (yk != yr).astype(mx.float32).mean().item() < 0.01
    # KDA conv + SiLU + q/k l2 norm
    B, L, Hk, Dk, Hv = 1, 19, 2, 128, 6
    C = 2 * Hk * Dk + Hv * Dk
    qkv, state, wc = bf(B, L, C), bf(B, 3, C), mx.array(RNG.standard_normal((C, 4)).astype(np.float32) * 0.5)
    q, k, v = kernels.kda_conv_qk(qkv, state, wc, Hk * Dk, Dk)
    yo = volundr._conv_silu(mx.concatenate([state, qkv], 1), wc[:, :, None])
    qo, ko, vo = mx.split(yo, [Hk * Dk, 2 * Hk * Dk], axis=-1)
    qo, ko = volundr.normalize_qk(qo.reshape(B, L, Hk, Dk).astype(mx.float32), ko.reshape(B, L, Hk, Dk).astype(mx.float32),
                                  inv_scale=Dk**-0.5, eps=1e-6)
    assert mx.array_equal(v, vo)
    assert rel(np.array(q), np.array(qo.reshape(B, L, -1))) < 1e-5 and rel(np.array(k), np.array(ko.reshape(B, L, -1))) < 1e-5
    # the vectorized variant (Dk = 128) is the same arithmetic, bit for bit, including a partial timestep block
    assert all(mx.array_equal(a, b) for a, b in zip(kernels.kda_conv_qk4(qkv, state, wc, Hk * Dk, Dk), (q, k, v)))
    # KDA gated output norm vs the compiled ops path
    xg, zg, wg = bf(3, L, Hv, Dk), bf(3, L, Hv, Dk), bf(Dk, sc=0.1) + 1
    gk, gr = kernels.gated_norm(xg, wg, zg, 1e-6), volundr._gated_norm(xg, wg, zg, 1e-6)
    assert (gk != gr).astype(mx.float32).mean().item() < 0.001 and rel(np.array(gk.astype(mx.float32)), np.array(gr.astype(mx.float32))) < 1e-2
    # packed per-channel-gate recurrence vs mlx-lm's vector-gate kernel
    from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel_unpacked
    T = 23
    qn = q.reshape(B, L, Hk, Dk)[:, :T]; kn = k.reshape(B, L, Hk, Dk)[:, :T]; vv = v.reshape(B, L, Hv, Dk)[:, :T]
    g = compute_g(mx.array(RNG.standard_normal((Hv, 1)).astype(np.float32) * 0.3), bf(B, L, Hv, Dk)[:, :T],
                  mx.array(RNG.standard_normal((Hv, Dk)).astype(np.float32) * 0.3))
    beta = mx.sigmoid(mx.array(RNG.standard_normal((B, L, Hv)).astype(np.float32)))[:, :T]
    st = mx.array(RNG.standard_normal((B, Hv, Dk, Dk)).astype(np.float32) * 0.1)
    yk, sk = kernels.gated_delta_vec(qn, kn, vv, g, beta, st)
    yr, sr = gated_delta_kernel_unpacked(qn, kn, vv, g, beta, st)
    assert rel(np.array(yk), np.array(yr)) < 1e-5 and rel(np.array(sk), np.array(sr)) < 1e-5
    yt, stt = kernels.gated_delta_tg(qn, kn, vv, g, beta, st)  # staged variant: identical arithmetic
    assert mx.array_equal(yt, yk) and mx.array_equal(stt, sk)
    # BCSA indexer block scores: the fused kernel equals the ops path (bf16 einsum, fp32 head mean, scale) exactly
    qi_, bk_ = bf(70, 4, 128), bf(1, 96, 128)
    assert mx.array_equal(kernels.bcsa_scores(qi_, bk_[0], 83, 128**-0.5),
                          volundr._block_scores(qi_[None], bk_[:, :83], 128**-0.5)[0])
    # BCSA prefill attention (far-block union + sliding window, one joint softmax) vs fp64. Covers partial query and
    # key tiles, a window that starts mid-block, queries without far blocks, more valid blocks than the bitmap's
    # first word, and window-only mode.
    for H_, KVH, D_, T_, qoff, W_, nv, topk, p_empty in ((24, 4, 256, 100, 300, 256, 300, 64, 0.2),
                                                        (4, 2, 64, 77, 50, 40, 37, 8, 0.0),
                                                        (4, 2, 64, 45, 100, 64, 3000, 16, 0.1),
                                                        (4, 2, 64, 130, 170, 128, 0, 0, 1.0)):
        cap = max(64, nv)
        q_, k_, v_ = bf(H_, T_, D_, sc=2.0), bf(KVH, T_ + qoff, D_), bf(KVH, T_ + qoff, D_)
        pk_, pv_ = bf(KVH, cap, D_), bf(KVH, cap, D_)
        sel = np.full((T_, max(topk, 1)), -1, dtype=np.int32)
        pool = RNG.permutation(max(nv, 1))[: 3 * topk]
        for t in range(T_):
            if RNG.uniform() >= p_empty:
                sel[t, :topk] = np.sort(RNG.choice(pool, topk, replace=False))
        if nv:
            got = kernels.bcsa_attend(q_, k_, v_, pk_, pv_, mx.array(sel), nv, D_**-0.5, qoff, W_)
        else:
            got = kernels.bcsa_window(q_, k_, v_, D_**-0.5, qoff, W_)
        qd, kd, vd, pkd, pvd = (np.array(a.astype(mx.float32), dtype=np.float64) for a in (q_, k_, v_, pk_, pv_))
        G_ = H_ // KVH
        r_, c_ = np.arange(T_)[:, None] + qoff, np.arange(T_ + qoff)[None]
        okw = (c_ <= r_) & (c_ > r_ - W_)
        want = np.zeros((H_, T_, D_))
        for h in range(H_):
            g_ = h // G_
            sw = np.where(okw, qd[h] @ kd[g_].T * D_**-0.5, -np.inf)
            for t in range(T_):
                idx = sel[t][sel[t] >= 0]
                s_all = np.concatenate([sw[t], pkd[g_][idx] @ qd[h, t] * D_**-0.5])
                p_ = np.exp(s_all - s_all.max())
                want[h, t] = p_ @ np.concatenate([vd[g_], pvd[g_][idx]]) / p_.sum()
        err = np.abs(np.array(got.astype(mx.float32)) - want)
        assert (err <= 2**-7 * np.abs(want) + 2e-3).all(), (H_, D_, T_, nv, err.max())


# ------------------------------------------------------------------------------------------ model
def test_model_matches_reference_cpu_fp32():
    """Whole tiny model (KDA, Engram, BCSA far regime with top-k, dense, mHC) in fp32 on the CPU vs the fp64
    reference: one-shot prefill, chunked prefill and token-by-token decode."""
    mx.set_default_device(mx.cpu)
    try:
        m = tiny_model(0)
        m.update(tree_map(lambda v: v.astype(mx.float32), m.parameters()))
        ids = ids_for(44)
        ref = Ref(np_params(m), TINY).forward(ids)
        for name, got in zip(("full", "chunked", "decode"), run_paths(m, ids)):
            e = rel(got, ref)
            assert e < 1e-4, (name, e)
        # the far branch is live: at the last position 7 blocks are valid, top-3 selected
        assert (44 - 1 - TINY["bcsa_window"] + 1) // TINY["bcsa_compress"] > TINY["bcsa_topk"]
    finally:
        mx.set_default_device(mx.gpu)


def test_model_gpu_fp32_kernels():
    """fp32 weights on the GPU: the Metal kernel paths (BCSA top-k + far attention + LSE merge, Sinkhorn, gated
    delta) against the fp64 reference. GPU fp32 GEMMs carry ~7e-4 relative error, so the bar is 5e-3."""
    m = tiny_model(0)
    m.update(tree_map(lambda v: v.astype(mx.float32), m.parameters()))
    ids = ids_for(44)
    ref = Ref(np_params(m), TINY).forward(ids)
    for name, got in zip(("full", "chunked", "decode"), run_paths(m, ids)):
        e = rel(got, ref)
        print(f"    gpu fp32 {name}: max rel err {e:.2e}")
        assert e < 5e-3, (name, e)


def test_model_gpu_bf16():
    """bf16 on the GPU (Metal kernels) vs the fp64 reference on the same bf16 weights. A random tiny model amplifies
    rounding, so the bar is the bf16 floor: the same model in bf16 on the CPU (ops paths, no Metal kernels)."""
    m = tiny_model(1)
    ids = ids_for(60, seed=3)
    ref = Ref(np_params(m), TINY).forward(ids)
    gpu = run_paths(m, ids, chunk=17)
    mx.set_default_device(mx.cpu)
    try:
        cpu = run_paths(m, ids, chunk=17)
    finally:
        mx.set_default_device(mx.gpu)
    for name, got, floor in zip(("full", "chunked", "decode"), gpu, cpu):
        e, ef = rel(got, ref), rel(floor, ref)
        agree = (got[:, : TINY["vocab_valid"]].argmax(-1) == ref[:, : TINY["vocab_valid"]].argmax(-1)).mean()
        print(f"    gpu bf16 {name}: max rel err {e:.2e} (cpu bf16 floor {ef:.2e}), argmax agreement {agree:.2%}")
        assert e < 1.25 * ef + 1e-2 and agree >= 0.9, (name, e, ef, agree)


def _quantize_sym(w, bits, group):
    """Symmetric per-group round-to-nearest (reference.rs quantize_row, any width) -> (q ints, scales fp32)."""
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(g).max(-1)
    s = np.array(mx.array(np.where(amax == 0, 1.0, amax / qmax)).astype(mx.bfloat16).astype(mx.float32))
    q = np.clip(np.rint(g / s[..., None]), -qmax - 1, qmax).astype(np.int64).reshape(n, k)
    return q, s


def test_convert_end_to_end():
    """A fake release folder (compressed-tensors W4 g64 default, W3/W2 g128 stored as g64 from an audit file, bf16
    linears, vision + torch conv layout) -> convert.py -> mlx_lm load_model -> the same logits as the model built
    from the dequantized weights."""
    from mlx_lm.utils import load_model

    m = tiny_model(2)
    params = dict(tree_flatten(m.parameters()))
    keep_bf16 = ("in_proj_b", "idx_q", "idx_k", "dyn", "engram.value", "engram.gate", "lm_head", "f_up")
    audit, expected, hf = [], {}, {}
    for k_, v in params.items():
        name = k_.replace("model.", "model.language_model.", 1) if k_.startswith("model.") else k_
        base = name[: -len(".weight")]
        is_lin = k_.endswith(".weight") and v.ndim == 2 and "embed" not in k_ and "tables" not in k_ and "mhc" not in k_
        if is_lin and not any(t in k_ for t in keep_bf16):
            bits = 3 if "down_proj" in k_ else 2 if "o_proj" in k_ else 4
            group = 64 if bits == 4 else 128
            w = np.array(v.astype(mx.float32))
            q, s = _quantize_sym(w, bits, group)
            s64 = np.repeat(s, group // 64, axis=1)
            hf[base + ".weight_packed"], hf[base + ".weight_scale"] = _container(q, s64)
            hf[base + ".weight_shape"] = mx.array(list(w.shape), dtype=mx.int64)
            if bits != 4:
                audit.append({"name": base, "shape": list(w.shape), "gsq_type": f"w{bits}g{group}"})
            expected[k_] = mx.array(q * np.repeat(s, group, axis=1)).astype(mx.bfloat16)
        elif k_.endswith("conv1d.weight"):
            hf[name] = v.moveaxis(2, 1).astype(mx.bfloat16)  # torch [C, 1, K]
            expected[k_] = hf[name].moveaxis(2, 1).astype(mx.float32)
        elif k_.endswith(("A_log", "dt_bias")):
            hf[name] = v.astype(mx.bfloat16)
            expected[k_] = hf[name].astype(mx.float32)
        else:
            hf[name] = v
            expected[k_] = v
    hf["model.visual.blocks.0.attn.qkv.weight"] = mx.zeros((8, 8), dtype=mx.bfloat16)
    m.update(tree_unflatten(list(expected.items())))

    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "hf", Path(td) / "mlx"
        src.mkdir()
        names = sorted(hf)
        half = len(names) // 2
        wmap = {}
        for i, part in enumerate((names[:half], names[half:])):
            fn = f"model-{i:05d}.safetensors"
            mx.save_safetensors(str(src / fn), {n: hf[n] for n in part})
            wmap.update({n: fn for n in part})
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
        tc = {k: v for k, v in TINY.items() if k not in ("bcsa_prefill_tile", "vocab_valid", "bcsa_index_heads",
                                                          "bcsa_index_dim", "rope_theta", "partial_rotary_factor")}
        tc["rope_parameters"] = {"rope_theta": TINY["rope_theta"], "partial_rotary_factor": TINY["partial_rotary_factor"]}
        qc = {"quant_method": "compressed-tensors", "format": "pack-quantized",
              "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 64}}}}
        (src / "config.json").write_text(json.dumps({"text_config": tc, "quantization_config": qc}))
        (src / "generation_config.json").write_text(json.dumps(
            {"eos_token_id": [7, 8], "suppress_tokens": list(range(TINY["vocab_valid"], TINY["vocab_size"]))}))
        (src / "quantization_audit.json").write_text(json.dumps({"quantized_matrices": audit}))
        r = subprocess.run([sys.executable, str(HERE.parent / "convert.py"), "--hf-path", str(src), "--mlx-path",
                            str(dst)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        cfg = json.loads((dst / "config.json").read_text())
        assert cfg["vocab_valid"] == TINY["vocab_valid"] and cfg["bcsa_index_heads"] == TINY["bcsa_index_heads"]
        assert cfg["quantization"]["model.layers.0.mlp.down_proj"] == {"group_size": 128, "bits": 3}
        loaded, _ = load_model(dst, get_model_classes=lambda config: (volundr.Model, volundr.ModelArgs))
        lw = dict(tree_flatten(loaded.parameters()))
        assert not any("visual" in k for k in lw)
        # every quantized linear dequantizes to q * scale exactly
        for k_ in expected:
            if k_.endswith(".weight") and k_[: -len(".weight")] + ".scales" in lw:
                b = k_[: -len(".weight")]
                mod_q = cfg["quantization"].get(b, {"group_size": 64, "bits": 4})
                deq = mx.dequantize(lw[k_], lw[b + ".scales"], lw[b + ".biases"], **mod_q)
                assert mx.array_equal(deq, expected[k_]), k_
        # fp32 on the CPU, where the quantized matmuls must equal the dense ones on the dequantized weights up to
        # accumulation order. In bf16 this would compare rounding noise, which a random tiny model amplifies. The
        # dense twin takes every tensor from the converted model, dequantized in fp32 (q * scale is exact there).
        ids = mx.array(ids_for(30, seed=5)[None], dtype=mx.int32)
        mx.set_default_device(mx.cpu)
        try:
            loaded.set_dtype(mx.float32)
            lw = dict(tree_flatten(loaded.parameters()))
            dense = {}
            for k_ in expected:
                b = k_[: -len(".weight")]
                if k_.endswith(".weight") and b + ".scales" in lw:
                    mod_q = cfg["quantization"].get(b, {"group_size": 64, "bits": 4})
                    dense[k_] = mx.dequantize(lw[k_], lw[b + ".scales"], lw[b + ".biases"], **mod_q)
                else:
                    dense[k_] = lw[k_]
            m.update(tree_unflatten(list(dense.items())))
            a = np.array(loaded(ids)[0])
            b = np.array(m(ids)[0])
        finally:
            mx.set_default_device(mx.gpu)
        e = rel(a, b)
        print(f"    converted (quantized matmuls) vs dequantized dense, fp32 CPU: max rel {e:.2e}")
        assert e < 1e-4


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
