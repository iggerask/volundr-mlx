"""In-situ cost of each small-op kind in a real decode step: matmuls removed (nomm), then one kind at a time replaced by
a no-kernel stand-in (precomputed arrays / inputs passed through). Interleaved, best of reps."""
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx
import mlx.nn as nn

import generate as G
import kernels as K
import volundr as V
import decode_fused as DF

model, tok = G.load(WT + "/models/volundr-int4-mlx")
G.set_wired_limit()
vv = model.args.vocab_valid
ids = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],
                              add_generation_prompt=True, enable_thinking=True)
NOMM = "real" not in sys.argv


def run(n=32, warm=6):
    cache = model.make_cache()
    logits = G.prefill(model, cache, ids)
    ts = []
    for t in G.greedy_tokens(model, cache, logits, n + warm, set(), vv):
        ts.append(time.perf_counter())
    ts = ts[warm:]
    return 1000 * (ts[-1] - ts[0]) / (len(ts) - 1)


real_q, real_l = nn.QuantizedLinear.__call__, nn.Linear.__call__
_z = {}


def const(shape, dtype):
    key = (tuple(shape), dtype)
    if key not in _z:
        _z[key] = mx.zeros(shape, dtype=dtype) + 0.01
        mx.eval(_z[key])
    return _z[key]


def nomm_q(self, x):
    return const((*x.shape[:-1], self.weight.shape[0]), x.dtype)


def nomm_l(self, x):
    if self.weight.shape[0] > 100000:
        return const((*x.shape[:-1], self.weight.shape[0]), x.dtype)
    return real_l(self, x)


orig = {n: getattr(K, n) for n in ("mhc_params", "mhc_mix", "mhc_combine", "rms_norm", "kda_conv_qk4", "gated_delta_vec",
                                   "gated_norm")}
orig_cg = V.compute_g
orig_sig = V.mx.sigmoid


def ab_params(d, pre_s, post_s, res_s, iters):
    N = d.shape[0]
    return const((N, 4), mx.bfloat16), const((N, 4), mx.bfloat16), const((N, 4, 4), mx.bfloat16)


def ab_mix(H, pre):
    return const((H.shape[0], H.shape[2]), mx.bfloat16)


def ab_combine(H, res, post, y, x_in):
    return H, const(y.shape, mx.bfloat16)


def ab_norm(x, w, eps, offset, r=None):
    return (x if r is not None else None), x


def ab_conv(qkv, state, w, key_dim, head_k_dim, tt=4):
    B, L, C = qkv.shape
    return const((B, L, key_dim), mx.float32), const((B, L, key_dim), mx.float32), const((B, L, C - 2 * key_dim), mx.bfloat16)


def ab_gdn(q, k, v, g, beta, state):
    return const(v.shape, q.dtype), state


def ab_gnorm(x, w, z, eps):
    return z


ABL = {
    "params": lambda: setattr(K, "mhc_params", ab_params),
    "mix": lambda: setattr(K, "mhc_mix", ab_mix),
    "combine": lambda: setattr(K, "mhc_combine", ab_combine),
    "rmsnorm": lambda: setattr(K, "rms_norm", ab_norm),
    "conv": lambda: setattr(K, "kda_conv_qk4", ab_conv),
    "gdn": lambda: setattr(K, "gated_delta_vec", ab_gdn),
    "gnorm": lambda: setattr(K, "gated_norm", ab_gnorm),
    "gates": lambda: setattr(V, "compute_g", lambda A, a, dt: const(a.shape, mx.float32)),
    "fused": lambda: setattr(DF, "ENABLED", True),
}


def reset():
    DF.ENABLED = False
    for n, f in orig.items():
        setattr(K, n, f)
    V.compute_g = orig_cg
    if NOMM:
        nn.QuantizedLinear.__call__, nn.Linear.__call__ = nomm_q, nomm_l
    else:
        nn.QuantizedLinear.__call__, nn.Linear.__call__ = real_q, real_l


names = ["base"] + [a for a in ABL if a in sys.argv or "all" in sys.argv]
best = {}
for rep in range(int(sys.argv[-1]) if sys.argv[-1].isdigit() else 3):
    for nm in names:
        reset()
        if nm != "base":
            ABL[nm]()
        dt = run()
        best[nm] = min(best.get(nm, 1e9), dt)
        print(f"rep {rep} {nm:10s} {dt:7.2f} ms/step", flush=True)
reset()
print("best:")
for nm in names:
    print(f"  {nm:10s} {best[nm]:7.2f} ms/step  delta {best['base'] - best[nm]:+6.2f}")
