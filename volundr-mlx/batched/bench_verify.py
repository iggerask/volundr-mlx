"""GPU cost of a batched speculative step's parts for B independent sequences (NOTES_BATCHED.md, the spec estimate).

Not a batched spec loop. It times, from an idle GPU, median of --reps, variants interleaved per rep:
- the verify forward of B independent sequences x T rows in one forward (B * T matmul rows), + argmax and the drafter
  features: verify.py's B = 1 pieces with a batched cache (KDA: verify._kda on the [B] states; BCSA: verify's 8q branch
  once per row against that row's own cache; dense: the row's cached keys plus its own T, causal). The cache is not
  changed (no commit), so every rep sees the same context;
- verify.forward (the production B = 1 verify, compiled KDA layers) as the reference;
- the drafter: B DFlash2 proposals (each with its own drafter cache) submitted together; B = 1 is the loop's drafter.
Matmul modes for the 4-bit linears: "nax" (nax_hook: 2..16 rows on the NAX kernel, more rows on stock MLX), "split"
(more than 16 rows cut into 16-row NAX calls) and "stock" (nax_hook off). The target uses the spec config's heads:
the caller sets VOLUNDR_LMHEAD8=64; the drafter is 4-bit with a 4-bit head copy, NAX on.

usage (from volundr-mlx, under ./with_gpu_lock.sh):
  VOLUNDR_LMHEAD8=64 python batched/bench_verify.py --tokens 200 --batches 1,2,4 --out spec/runs/batched-bench_verify.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spec"))
import bench_batched as BB  # noqa: E402
import kernels  # noqa: E402
import nax_hook  # noqa: E402
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402
import volundr as V  # noqa: E402

prep_nax, qmm_nax = nax_hook.prep_nax, nax_hook.qmm_nax
_memo = {"x": None, "p": None}


class NaxSplit(nax_hook.NaxLinear):
    """NaxLinear that also takes more than 16 rows, as 16-row NAX calls (one prep per input array and piece)."""

    def __call__(self, x):
        K = x.shape[-1]
        rows = x.size // K
        if rows <= qmm_nax.MT or x.dtype != mx.bfloat16:
            return super().__call__(x)
        N = self.weight.shape[0]
        if _memo["x"] is not x:
            xf = x.reshape(rows, K)
            _memo["x"], _memo["p"] = x, [prep_nax.prep(xf[i : i + qmm_nax.MT]) for i in range(0, rows, qmm_nax.MT)]
        outs = [qmm_nax.qmm_nax_prepped(p, self.weight, self.scales, self.biases) for p in _memo["p"]]
        return mx.concatenate(outs, axis=0)[:rows].reshape(*x.shape[:-1], N)


_traces = {}


def set_mode(model, mode):
    # verify.forward's compiled KDA layers trace the linears' Python calls, so each matmul mode keeps its own traces
    VF._compiled = _traces.setdefault(mode, {})
    for _, m in model.named_modules():
        if type(m) is NaxSplit:
            m.__class__ = nn.QuantizedLinear
    nax_hook.disable(model)
    _memo["x"] = _memo["p"] = None
    if mode in ("nax", "split"):
        nax_hook.enable(model)
    if mode == "split":
        for _, m in model.named_modules():
            if type(m) is nax_hook.NaxLinear:
                m.__class__ = NaxSplit


# ----------------------------------------------------------------------------- batched verify, independent rows
def row_views(cache, B):
    """Per-row BCSACache views of a batched cache (for verify's B = 1 BCSA branch)."""
    out = []
    for c in cache:
        if isinstance(c, V.BCSACache):
            rows = []
            for b in range(B):
                v = V.BCSACache(c.W, c.C, c.F)
                v.offset, v.nb = c.offset, c.nb
                for a in ("k", "v", "pk", "pv", "bk", "ik_open"):
                    setattr(v, a, getattr(c, a)[b : b + 1])
                rows.append(v)
            out.append(rows)
        else:
            out.append(None)
    return out


def _attn_rows(self, x, cache, views):
    B, L, _ = x.shape
    D = self.head_dim
    qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * D)
    q, gate = qg[..., :D], qg[..., D:].reshape(B, L, self.n_heads * D)
    q = self.q_norm(q).transpose(0, 2, 1, 3)
    k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, D)).transpose(0, 2, 1, 3)
    v = self.v_proj(x).reshape(B, L, self.n_kv, D).transpose(0, 2, 1, 3)
    q, k = self._rope(q, cache.offset), self._rope(k, cache.offset)  # every row at the same offset
    if not self.bcsa:
        P = cache.offset
        kk = mx.concatenate([cache.keys[..., :P, :], k], axis=2)
        vv = mx.concatenate([cache.values[..., :P, :], v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=self.scale, mask="causal")
    else:
        o = mx.concatenate([VF._bcsa_verify(self, x[b : b + 1], q[b : b + 1], k[b : b + 1], v[b : b + 1], views[b])[0]
                            for b in range(B)], axis=0)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * D)
    return self.o_proj(V._sigmoid_gate(o, gate))


def _layer_rows(self, x, cache, rows, views):
    eps = self.input_layernorm.eps
    h = kernels.rms_norm(x, self.input_layernorm.weight, eps, offset=True)[1]
    if self.mixer == "kda":
        r = VF._kda(self.linear_attn, h, [cache[0], cache[1]], {})  # a list copy: the cache is not changed
    else:
        r = _attn_rows(self.self_attn, h, cache, views)
    h, hn = kernels.rms_norm(x, self.post_attention_layernorm.weight, eps, offset=True, r=r)
    y = h + self.mlp(hn)
    if "engram" in self:
        y = self.engram(y, rows)
    return y


def verify_rows(model, ids, cache, views, capture):
    """ids [B, T]: B independent sequences' verify rows -> (logits [B, T, vocab], features [B, T, 5 * HID])."""
    a, inner = model.args, model.model
    B, L = ids.shape
    x = inner.embed_tokens(ids)
    full = mx.concatenate([cache[-1].prev, ids.astype(mx.int32)], axis=1)
    rows = [V.engram_rows(full, n, L, a.engram_rows) for n in a.engram_orders]
    H = mx.broadcast_to(x[:, :, None, :], (B, L, a.mhc_streams, a.hidden_size))
    hmean = x
    caps = []
    for i, (layer, mhc, c) in enumerate(zip(inner.layers, inner.mhc, cache)):
        if i in capture:
            caps.append(H[:, :, 0])
        x_in, post, res = mhc.mix(H, hmean)
        y = _layer_rows(layer, x_in, c, rows, views[i])
        H, hmean = mhc.combine(H, post, res, y, x_in)
    return model.lm_head(inner.norm(H[:, :, 0])), mx.concatenate(caps, axis=-1)


def timed(fn):
    mx.synchronize()
    t0 = time.perf_counter()
    outs = fn()
    t1 = time.perf_counter()
    mx.eval(outs)
    t2 = time.perf_counter()
    return 1e3 * (t2 - t0), 1e3 * (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--modes", default="nax,split,stock")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    modes = args.modes.split(",")
    nmax, T = max(batches), args.T

    model, drafter, tok = S.load(4, 4, True)
    vv = model.args.vocab_valid
    prompts = BB.make_prompts(tok, args.tokens, nmax)
    caches, dcaches, nxt = [], [], []
    for ids in prompts:
        c, dc = model.make_cache(), drafter.make_cache()
        lg = S.spec_prefill(model, drafter, c, dc, ids)
        nxt.append(int(mx.argmax(lg[:vv]).item()))
        caches.append(c)
        dcaches.append(dc)
    snap = BB.stack_caches(model, caches)
    del caches
    mx.clear_cache()
    P = args.tokens
    # verify rows: the bonus token, then T - 1 tokens of the prompt (content does not change the cost)
    blocks = [[nxt[b]] + prompts[b][-(T - 1):] for b in range(nmax)]

    variants = []
    for B in batches:
        cache = BB.slice_cache(model, snap, 0, B, nmax)
        views = row_views(cache, B)
        ids = mx.array(blocks[:B], dtype=mx.int32)
        for mode in modes:
            if mode == "split" and B * T <= qmm_nax.MT:
                continue  # same as nax
            variants.append((f"verify_rows B={B} {mode}", B, mode,
                             lambda c=cache, w=views, i=ids: (lambda r: (mx.argmax(r[0][..., :vv], axis=-1), r[1]))(
                                 verify_rows(model, i, c, w, drafter.capture_layers))))
        if B == 1:
            for mode in ("nax", "stock"):
                if mode not in modes:
                    continue

                def ref(c=cache, i=ids):
                    c2 = BB.slice_cache(model, snap, 0, 1, nmax)  # forward writes the cache: a fresh view each call
                    lg, cap, rec = VF.forward(model, i, c2, capture=drafter.capture_layers, verify=True, compiled=True)
                    return mx.argmax(lg[:, :vv], axis=-1), cap

                variants.append(("verify.forward B=1 " + mode, 1, mode, ref))
        anchors = [mx.array(nxt[b], dtype=mx.int32) for b in range(B)]
        variants.append((f"drafter B={B}", B, "nax",
                         lambda B=B, an=anchors: [drafter.propose(dcaches[b], an[b], P) for b in range(B)]))

    res = {v[0]: {"gpu": [], "build": []} for v in variants}
    for rep in range(args.reps + 1):
        for name, B, mode, fn in variants:
            set_mode(model, mode)
            ms, build = timed(fn)
            if rep:
                res[name]["gpu"].append(ms)
                res[name]["build"].append(build)
    set_mode(model, "nax")
    with open(args.out, "a") as f:
        for name, B, mode, _ in variants:
            g, b = sorted(res[name]["gpu"]), sorted(res[name]["build"])
            rec = {"kind": "verify", "name": name, "B": B, "mode": mode, "T": T, "rows": B * T, "context": P,
                   "reps": args.reps, "ms_median": round(g[len(g) // 2], 2), "ms_min": round(g[0], 2),
                   "build_ms_median": round(b[len(b) // 2], 2), "lmhead8": V.lm_head_q_group()}
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
