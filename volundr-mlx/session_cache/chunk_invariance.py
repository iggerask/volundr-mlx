"""Which parts of a prefill depend on where the chunk boundaries fall (the source of the resumed-prefill KL in
NOTES_SESSION.md). Each component of the real model runs on the same random bf16 input in two chunkings of 6000
positions: [0, 2048, 4096, 6000) and [0, 2048, 4096, 4483, 6000) (a split at an unaligned resume point); the outputs
for positions 4483..6000 are compared bit for bit. Far blocks are active there (positions > 4096).

usage (from volundr-mlx, under ./with_gpu_lock.sh): python session_cache/chunk_invariance.py
"""
import json
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402
import volundr as V  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

N, SPLIT = 6000, 4483
A = [0, 2048, 4096, N]
B = [0, 2048, 4096, SPLIT, N]


def run(fn, make_cache, x, bounds):
    c = make_cache()
    outs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        o = fn(x[:, s:e], c)
        mx.eval(o, [] if c is None else c.state)
        outs.append(o)
    return mx.concatenate(outs, axis=1)


def compare(name, fn, make_cache, x):
    a = run(fn, make_cache, x, A)[:, SPLIT:]
    b = run(fn, make_cache, x, B)[:, SPLIT:]
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    r = {"component": name, "identical": bool(mx.array_equal(a, b).item()),
         "rows_differing": int((d.max(axis=-1) > 0).sum().item()), "rows": int(a.shape[1]),
         "max_abs": float(d.max().item()), "mean_abs": float(d.mean().item()),
         "ref_mean_abs": float(mx.abs(a.astype(mx.float32)).mean().item())}
    print(json.dumps(r), flush=True)
    return r


def main():
    model, _ = G.load(HERE / "models/volundr-int4-mlx")
    a = model.args
    mx.random.seed(0)
    x = mx.random.normal((1, N, a.hidden_size)).astype(mx.bfloat16)
    layers = model.model.layers
    kda = next(i for i, m in enumerate(a.mixer_types) if m == "kda")
    bcsa = next(i for i, m in enumerate(a.mixer_types) if m == "bcsa")
    dense = next(i for i, m in enumerate(a.mixer_types) if m == "attn")
    res = []
    L = layers[kda].linear_attn
    res.append(compare("4-bit matmul (KDA in_proj_qkv)", lambda t, c: L.in_proj_qkv(t), lambda: None, x))
    res.append(compare("MLP (4-bit)", lambda t, c: layers[kda].mlp(t), lambda: None, x))
    res.append(compare("KDA mixer (projections, conv, recurrence, gated norm, out_proj)",
                       lambda t, c: L(t, c), lambda: ArraysCache(size=2), x))
    # the KDA mixer's two kernels separately: conv + SiLU + q/k norm + gates (decode_fused.kda_conv_gates), and the
    # recurrence (kernels.gated_delta_tg, which stages 8 timesteps in threadgroup memory) on identical inputs
    import decode_fused as DF
    import kernels

    B_, H, HV, K = 1, L.H, L.HV, L.K
    qkv_all, a_all, b_all = L.in_proj_qkv(x), L.f_up(L.f_down(x)), L.in_proj_b(x)
    mx.eval(qkv_all, a_all, b_all)

    def conv_gates(t, c, s, e):
        cs = c[0] if c[0] is not None else mx.zeros((1, L.kernel - 1, L.conv_dim), mx.bfloat16)
        q, k, v, c[0], g, beta = DF.kda_conv_gates(qkv_all[:, s:e], cs, L.conv1d.weight[:, :, 0], L.key_dim, L.A_log,
                                                   a_all[:, s:e], L.dt_bias, b_all[:, s:e])
        f = lambda t: t.reshape(1, e - s, -1).astype(mx.float32)  # noqa: E731 (exact for bf16 q, k, v)
        return mx.concatenate([f(q), f(k), f(v), f(g), f(beta)], axis=-1)

    def run_pos(fn, make_cache, bounds):
        c = make_cache()
        outs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            o = fn(None, c, s, e)
            mx.eval(o, c[0] if c[0] is not None else [])
            outs.append(o)
        return mx.concatenate(outs, axis=1)

    def compare_pos(name, fn, make_cache):
        ra = run_pos(fn, make_cache, A)[:, SPLIT:]
        rb = run_pos(fn, make_cache, B)[:, SPLIT:]
        d = mx.abs(ra.astype(mx.float32) - rb.astype(mx.float32))
        r = {"component": name, "identical": bool(mx.array_equal(ra, rb).item()),
             "rows_differing": int((d.max(axis=-1) > 0).sum().item()), "rows": int(ra.shape[1]),
             "max_abs": float(d.max().item()), "mean_abs": float(d.mean().item()),
             "ref_mean_abs": float(mx.abs(ra.astype(mx.float32)).mean().item())}
        print(json.dumps(r), flush=True)
        return r

    res.append(compare_pos("KDA conv + gates kernel (q, k, v, g, beta)", conv_gates, lambda: [None]))
    whole = conv_gates(None, [None], 0, N)  # inputs of the recurrence from one unsplit pass
    mx.eval(whole)
    kd = L.key_dim
    vd = L.value_dim
    q_all = whole[..., :kd].astype(mx.bfloat16).reshape(1, N, H, K)
    k_all = whole[..., kd : 2 * kd].astype(mx.bfloat16).reshape(1, N, H, K)
    v_all = whole[..., 2 * kd : 2 * kd + vd].astype(mx.bfloat16).reshape(1, N, HV, L.V)
    rest = 2 * kd + vd
    g_all = whole[..., rest : rest + HV * K].reshape(1, N, HV, K)
    be_all = whole[..., rest + HV * K :].reshape(1, N, HV)

    def recur(t, c, s, e):
        st = c[0] if c[0] is not None else mx.zeros((1, HV, L.V, K), mx.float32)
        rec = kernels.gated_delta_tg if e - s >= 64 else kernels.gated_delta_vec
        o, c[0] = rec(q_all[:, s:e], k_all[:, s:e], v_all[:, s:e], g_all[:, s:e], be_all[:, s:e], st,
                      out_dtype=mx.bfloat16)
        return o.reshape(1, e - s, -1)

    res.append(compare_pos("KDA recurrence (gated_delta_tg)", recur, lambda: [None]))
    att = layers[bcsa].self_attn
    res.append(compare("BCSA attention (window + top-k far blocks)", lambda t, c: att(t, c),
                       lambda: V.BCSACache(a.bcsa_window, a.bcsa_compress, a.bcsa_topk), x))
    den = layers[dense].self_attn
    res.append(compare("dense attention (SDPA)", lambda t, c: den(t, c), KVCache, x))
    out = HERE / "session_cache/chunk_invariance.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
