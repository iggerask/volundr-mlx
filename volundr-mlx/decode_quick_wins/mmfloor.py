"""Matmul floor of a decode step: every linear of the real model at M = 1 in the real step's dependency stages (inputs
taken from the previous stage's outputs, so stages serialise like the real step), no other ops. Interleaved with the
real decode step (fused and unfused)."""
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx

import decode_fused as DF
import generate as G

model, tok = G.load(WT + "/models/volundr-int4-mlx", lm_head_q=False)
G.set_wired_limit()
vv = model.args.vocab_valid
ids = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],
                              add_generation_prompt=True, enable_thinking=True)
inner = model.model


def mm_step(x, head=True, small=True):
    h = x
    for L, mhc in zip(inner.layers, inner.mhc):
        d = mhc.dyn(h) if small else None
        if L.mixer == "kda":
            a = L.linear_attn
            qkv, z, fd = a.in_proj_qkv(h), a.in_proj_z(h), a.f_down(h)
            b = a.in_proj_b(h) if small else None
            fu = a.f_up(fd)
            o = a.out_proj(z + fu[..., : z.shape[-1]] * 0 if False else z)
            extra = [qkv, fu] + ([b, d] if small else [])
        else:
            a = L.self_attn
            q, k, v = a.q_proj(h), a.k_proj(h), a.v_proj(h)
            extra = [k, v] + ([d] if small else [])
            if a.bcsa and small:
                extra += [a.idx_q(h), a.idx_k(h)]
            o = a.o_proj(q[..., : a.n_heads * a.head_dim])
        gt, up = L.mlp.gate_proj(o), L.mlp.up_proj(o)
        h = L.mlp.down_proj(gt) + 0 * up[..., : 5120].sum() if False else L.mlp.down_proj(gt)
        yield extra + [up]
    if head:
        yield [model.lm_head(h)]
    yield [h]


def run_mm(n=48, **kw):
    x = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    mx.eval(x)
    ts = []
    prev = None
    for i in range(n):  # pipelined like generate.greedy_tokens: the next step's graph is queued before waiting
        outs = [a for grp in mm_step(x, **kw) for a in grp]
        mx.async_eval(outs)
        if prev is not None:
            mx.eval(prev)
        ts.append(time.perf_counter())
        prev, x = outs, outs[-1]
    ts = ts[8:]
    return 1000 * (ts[-1] - ts[0]) / (len(ts) - 1)


def run_real(n=48, warm=8):
    cache = model.make_cache()
    logits = G.prefill(model, cache, ids)
    ts = []
    for t in G.greedy_tokens(model, cache, logits, n + warm, set(), vv):
        ts.append(time.perf_counter())
    ts = ts[warm:]
    return 1000 * (ts[-1] - ts[0]) / (len(ts) - 1)


best = {}
for rep in range(4):
    for name, f in (("matmuls only (+head)", lambda: run_mm()), ("4-bit matmuls only (no head, no bf16 smalls)", lambda: run_mm(head=False, small=False)),
                    ("real unfused", lambda: (setattr(DF, "ENABLED", False), run_real())[1]),
                    ("real fused", lambda: (setattr(DF, "ENABLED", True), run_real())[1])):
        dt = f()
        if rep:
            best[name] = min(best.get(name, 1e9), dt)
        print(f"rep {rep} {name:45s} {dt:7.2f} ms/step", flush=True)
for k, v in best.items():
    print(f"best {k:45s} {v:7.2f}")
