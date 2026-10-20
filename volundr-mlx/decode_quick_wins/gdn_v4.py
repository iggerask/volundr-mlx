"""gated_delta_vec variants with explicit 16-byte loads/stores (same per-lane arithmetic): exactness + speed at decode
shapes (54 layers' states, chained)."""
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx
import numpy as np

import kernels

SRC4 = r"""
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = Dk / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;

    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);

    auto lane = thread_index_in_simdgroup;
    auto row_in_simdgroup = lane / lanes_per_row;
    auto lane_in_row = lane & (lanes_per_row - 1);
    auto row_group = thread_position_in_grid.y;
    auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

    auto q_ = (const device float4*)(q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane);
    auto k_ = (const device float4*)(k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane);
    auto g_ = (const device float4*)(g + (b_idx * T * Hv + hv_idx) * Dk + lane_in_row * values_per_lane);
    auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
    y += (b_idx * T * Hv + hv_idx) * Dv;

    auto i_state = (const device float4*)(state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane);
    auto o_state = (device float4*)(state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane);

    float state[values_per_lane];
    for (int j = 0; j < partials_per_lane; ++j) {
        const float4 s4 = i_state[j];
        state[4 * j] = s4.x; state[4 * j + 1] = s4.y; state[4 * j + 2] = s4.z; state[4 * j + 3] = s4.w;
    }
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
        float part[partials_per_lane];
        float kk[values_per_lane];
        for (int pb = 0; pb < partials_per_lane; ++pb) {
            const float4 g4 = g_[pb];
            const float4 k4 = k_[pb];
            kk[pb * 4] = k4.x; kk[pb * 4 + 1] = k4.y; kk[pb * 4 + 2] = k4.z; kk[pb * 4 + 3] = k4.w;
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] * g4[i];
                acc += state[e] * kk[e];
            }
            part[pb] = acc;
        }
        float kv_mem = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);
        auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);
        for (int pb = 0; pb < partials_per_lane; ++pb) {
            const float4 q4 = q_[pb];
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] + kk[e] * delta;
                acc += state[e] * q4[i];
            }
            part[pb] = acc;
        }
        float out = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        if (lane_in_row == 0) y[dv_idx] = static_cast<InT>(out);
        q_ += Hk * Dk / 4;
        k_ += Hk * Dk / 4;
        g_ += Hv * Dk / 4;
        v_ += Hv * Dv;
        y += Hv * Dv;
        beta_ += Hv;
    }
    for (int j = 0; j < partials_per_lane; ++j)
        o_state[j] = float4(state[4 * j], state[4 * j + 1], state[4 * j + 2], state[4 * j + 3]);
"""
K4 = mx.fast.metal_kernel(name="gdn_vec4_test", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                          output_names=["y", "state_out"], source=SRC4)


def gdn4(q, k, v, g, beta, state, out_dtype=None, tgy=2):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    ot = out_dtype or q.dtype
    return K4(inputs=[q, k, v, g, beta, state, T],
              template=[("InT", ot), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
              grid=(32, Dv // 8, B * Hv), threadgroup=(32, tgy, 1),
              output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[ot, state.dtype])


def gdn_tg(q, k, v, g, beta, state, out_dtype=None, tgy=2):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    ot = out_dtype or q.dtype
    return kernels._gdn_vec_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", ot), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv // 8, B * Hv), threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[ot, state.dtype])


rng = np.random.default_rng(0)
f = lambda *s, sc=1.0: mx.array(rng.standard_normal(s).astype(np.float32) * sc)
B, Hk, Hv, Dk = 1, 16, 48, 128
for T in (1, 3, 8):
    q, k = f(B, T, Hk, Dk, sc=0.1), f(B, T, Hk, Dk, sc=0.1)
    v = f(B, T, Hv, Dk).astype(mx.bfloat16)
    g = mx.sigmoid(f(B, T, Hv, Dk) + 2)
    beta = mx.sigmoid(f(B, T, Hv))
    st = f(B, Hv, Dk, Dk, sc=0.1)
    y0, s0 = kernels.gated_delta_vec(q, k, v, g, beta, st)
    for tgy in (2, 4, 8):
        y1, s1 = gdn4(q, k, v, g, beta, st, tgy=tgy)
        print(f"T={T} tgy={tgy}: vec4 == vec: {mx.array_equal(y0, y1).item()} {mx.array_equal(s0, s1).item()}")

NL = 54
T = 1
states = [f(B, Hv, Dk, Dk, sc=0.1) for _ in range(NL)]
q, k = f(B, T, Hk, Dk, sc=0.1), f(B, T, Hk, Dk, sc=0.1)
v = f(B, T, Hv, Dk).astype(mx.bfloat16)
g = mx.sigmoid(f(B, T, Hv, Dk) + 2)
beta = mx.sigmoid(f(B, T, Hv))
mx.eval(states, q, k, v, g, beta)
variants = {"vec tg2": lambda *a: gdn_tg(*a, tgy=2), "vec tg4": lambda *a: gdn_tg(*a, tgy=4),
            "vec tg8": lambda *a: gdn_tg(*a, tgy=8), "vec4 tg2": lambda *a: gdn4(*a, tgy=2),
            "vec4 tg4": lambda *a: gdn4(*a, tgy=4), "vec4 tg8": lambda *a: gdn4(*a, tgy=8)}
best = {}
for rep in range(15):
    for name, fn in variants.items():
        vv = v
        outs = []
        for s in states:
            o, s2 = fn(q, k, vv, g, beta, s, mx.bfloat16)
            vv = o
            outs.append(s2)
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(vv, outs)
        mx.synchronize()
        best[name] = min(best.get(name, 1e9), (time.perf_counter() - t) / NL)
for name in variants:
    print(f"{name:10s} {best[name] * 1e6:6.1f} us per layer ({6.29e6 / best[name] / 1e9:4.0f} GB/s)")
