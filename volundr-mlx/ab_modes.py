"""Run-time configs for the interleaved A/Bs (bench_decode.py, spec/ab_suite.py), switched on one loaded model.

config = base[+mod...]
  base  main     the decode path as on main (0732112): decode_fused off, kernels.rms_norm with its strided loops, the
                 norm weights cast on every call, BCSA integer indexing (x[0], p[b] copy the array)
        unfused  decode_fused off; everything else as on this branch
        fused    this branch (the default)
  mods  q8g64 / q8g128  8-bit affine lm_head with that group size (VOLUNDR_LMHEAD8's head)
        copyidx         BCSA integer indexing as on main
        vprefill / v8q  BCSA verify path of spec/verify.py (VOLUNDR_BCSA_VERIFY): the prefill kernels as on main, or
                        bcsa_verify.attend. Default: prefill for main, 8q for unfused / fused. Only matters when
                        spec/verify.py is imported (speculative runs).
  aliases  baseline = main (main's decode path and main's prefill-kernel verify)
           exact    = fused (the defaults: fused decode, views, 8q verify, bf16 head); exact+q8g64 adds the 8-bit head
All of them give the same decode logits as main bit for bit, except the q8 heads; v8q changes verify logits by
rounding (NOTES_BCSA_VERIFY.md).
"""
import sys

import mlx.core as mx
import mlx.nn as nn

import decode_fused as DF
import kernels
import volundr as V

F32 = mx.float32
_new = {"rms_norm": kernels.rms_norm, "first": V._first, "gather": V._gather_blocks, "wget": V._F32Weight.get}

# kernels.rms_norm as on main: the row loops strided (for i = tid; i < D; i += NT), the NT-thread kernel only
_MAIN_NORM_SRC = """
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
_main_norm_kernels = {}


def _main_rms_norm(x, w, eps, offset: bool, r=None):
    D = x.shape[-1]
    N = x.size // D
    add = r is not None
    key = (add, offset, eps)
    if key not in _main_norm_kernels:
        src = _MAIN_NORM_SRC.replace("EPS_VAL", repr(float(eps)) + "f").replace(
            "LOAD", kernels._NORM_LOAD_ADD if add else kernels._NORM_LOAD)
        _main_norm_kernels[key] = mx.fast.metal_kernel(
            name=f"ab_main_rmsnorm_{int(add)}{int(offset)}_{len(_main_norm_kernels)}",
            input_names=["x", "w"] + (["r"] if add else []), output_names=(["h"] if add else []) + ["y"], source=src)
    nt = 256 if D >= 1024 else 32 * max(1, min(8, D // 32))
    outs = _main_norm_kernels[key](
        inputs=[x, w] + ([r] if add else []), template=[("D", D), ("NT", nt), ("OFFSET", 1 if offset else 0)],
        grid=(nt, N, 1), threadgroup=(nt, 1, 1),
        output_shapes=[x.shape] * (2 if add else 1), output_dtypes=[mx.bfloat16] * (2 if add else 1))
    return (outs[0], outs[1]) if add else (None, outs[0])


def _copy_gather(p, idx):  # volundr._gather_blocks as on main
    B, KVH, _, D = p.shape
    out = [mx.take(p[b], idx[b].reshape(-1), axis=1).reshape(KVH, *idx.shape[1:], D) for b in range(B)]
    return out[0][None] if B == 1 else mx.stack(out)


def _uncached_get(self, w, offset):  # the norm weight as on main: cast (and offset) on every call
    return (1.0 + w.astype(F32)) if offset else w.astype(F32)


ALIASES = {"baseline": "main", "exact": "fused"}


def describe() -> dict:
    """What the last Modes.set switched on, for run records."""
    vf = sys.modules.get("verify")
    return {"fused": DF.ENABLED, "views": V._first is _new["first"], "bcsa_verify": vf.BCSA_IMPL if vf else None}


class Modes:
    def __init__(self, model, configs):
        self.model = model
        assert "scales" not in model.lm_head, "Modes needs the bf16 head (unset VOLUNDR_LMHEAD8): configs choose it"
        self.heads = {"bf16": model.lm_head}
        for c in configs:
            for part in c.split("+")[1:]:
                if part.startswith("q8g") and part not in self.heads:
                    self.heads[part] = nn.QuantizedLinear.from_linear(self.heads["bf16"], int(part[3:]), 8)
                    mx.eval(self.heads[part].parameters())

    def set(self, c):
        base, *mods = c.split("+")
        base = ALIASES.get(base, base)
        assert base in ("main", "unfused", "fused") and all(
            m in ("copyidx", "vprefill", "v8q") or m.startswith("q8g") for m in mods), c
        assert not ("vprefill" in mods and "v8q" in mods), c
        main = base == "main"
        vf = sys.modules.get("verify")  # spec/verify.py: imported by speculative runs only; read at call time
        if vf is not None:
            vf.BCSA_IMPL = "prefill" if "vprefill" in mods or (main and "v8q" not in mods) else "8q"
        DF.ENABLED = base == "fused"
        kernels.rms_norm = _main_rms_norm if main else _new["rms_norm"]
        copy = main or "copyidx" in mods
        V._first = (lambda x: x[0]) if copy else _new["first"]
        V._gather_blocks = _copy_gather if copy else _new["gather"]
        V._F32Weight.get = _uncached_get if main else _new["wget"]
        hs = [m for m in mods if m.startswith("q8g")]
        self.model.lm_head = self.heads[hs[0] if hs else "bf16"]

    def reset(self):
        self.set("fused")
