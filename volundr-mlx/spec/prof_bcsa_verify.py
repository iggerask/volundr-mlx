"""Where the long-context verify step's extra time goes: the BCSA verify path at a real long context.

Prefills a tune prompt (optionally two concatenated, for ~20K), then
1. times the 8-token verify forward + commit, and a plain decode step (frozen config: NAX verify, 4-bit drafter);
2. times the verify forward with the BCSA attention replaced by cheaper stand-ins (no far blocks; no attention at
   all), so the difference is what the BCSA verify attention costs inside the real graph;
3. captures one BCSA layer's verify inputs and times its parts in isolation, 17 copies per eval (one per BCSA layer):
   the prefill-kernel path (window fetch + concat, far-block selection: indexer q, scores, top-k; union; attend
   kernel), the 8-query path (bcsa_verify.attend at a few split counts, window only, whole verify branch), the cost
   of integer-indexing a cache buffer, and the decode path's parts for one query on the same layer.
--impl picks verify.BCSA_IMPL ("8q" or "prefill") for parts 1 and 2.

usage (from spec/, under ../with_gpu_lock.sh):
  python prof_bcsa_verify.py --prompts long/03-long3 [--prompts long/03-long3,long/02-long2-think] [--impl 8q]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402

V, kernels = VF.V, VF.kernels
NL = 17


def timed(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return round(1e3 * ts[len(ts) // 2], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="long/03-long3")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--impl", default="", help="verify.BCSA_IMPL for the step timings (default: module default)")
    ap.add_argument("--max-tokens", type=int, default=0, help="keep only the first N prompt tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.impl:
        VF.BCSA_IMPL = args.impl
    model, drafter, tok = S.load(4, 4, True)
    suite = {e["name"]: e for e in S.suite(HERE / "suite.json", "tune")}
    ids = sum((suite[n]["ids"] for n in args.prompts.split(",")), [])
    if args.max_tokens:
        ids = ids[: args.max_tokens]
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = S.spec_prefill(model, drafter, cache, dcache, ids)
    st = {"P": len(ids)}
    rec = {"prompts": args.prompts, "P": len(ids), "impl": getattr(VF, "BCSA_IMPL", "prefill")}
    gen = [int(t) for t in tok.encode(" the quick brown fox jumps over the lazy dog and runs far away" * 4)]

    def verify_step(T=8, commit=True):
        def f():
            inp = mx.array(gen[:T], dtype=mx.int32)[None]
            lg, cap, r = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
            top = mx.argmax(lg, axis=-1)
            mx.eval(top, cap)
            if commit:
                VF.commit(cache, r, T)
                drafter.materialize(dcache, cap, st["P"])
                mx.eval([c.state for c in cache], dcache.state)
                st["P"] += T
        return f

    def decode():
        y = mx.array([[gen[0]]], dtype=mx.int32)
        lg = model(y, cache=cache, last_only=True)
        mx.eval(mx.argmax(lg), [c.state for c in cache])
        st["P"] += 1

    rec["verify8_ms"] = timed(verify_step(), args.reps)
    rec["verify8_nocommit_ms"] = timed(verify_step(commit=False), args.reps)
    rec["decode_ms"] = timed(decode, args.reps)
    # the commit and the drafter's context update on their own (each evaluated separately)
    tc, tm = [], []
    for _ in range(args.reps + 1):
        inp = mx.array(gen[:8], dtype=mx.int32)[None]
        lg, cap, r = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
        mx.eval(lg, cap, r["layers"][3].get("k"), [c.state for c in cache])
        t0 = time.perf_counter()
        VF.commit(cache, r, 8)
        mx.eval([c.state for c in cache])
        t1 = time.perf_counter()
        drafter.materialize(dcache, cap, st["P"])
        mx.eval(dcache.state)
        t2 = time.perf_counter()
        st["P"] += 8
        tc.append(t1 - t0)
        tm.append(t2 - t1)
    rec["commit_ms"] = round(1e3 * sorted(tc[1:])[len(tc) // 2 - 1], 3)
    rec["materialize_ms"] = round(1e3 * sorted(tm[1:])[len(tm) // 2 - 1], 3)

    # 2. stand-ins for the BCSA verify attention inside the full forward (no commit: the context stays put)
    orig = VF._bcsa_verify

    def no_far(self, x, q, k, v, cache):
        P, W = cache.offset, self.W
        nprev = min(P, W - 1)
        wk, wv = cache.window(nprev)
        keys, vals = mx.concatenate([wk, k], axis=2), mx.concatenate([wv, v], axis=2)
        return kernels.bcsa_window(q[0], keys[0], vals[0], self.scale, nprev, W)[None], self.idx_norm(self.idx_k(x))

    def no_attn(self, x, q, k, v, cache):
        return q, self.idx_norm(self.idx_k(x))

    for name, fn in (("no_far", no_far), ("no_attn", no_attn)):
        VF._bcsa_verify = fn
        rec[f"verify8_{name}_ms"] = timed(verify_step(commit=False), args.reps)
    VF._bcsa_verify = orig

    # 3. one layer's parts in isolation
    grab = {}

    def capture(self, x, q, k, v, cache):
        if not grab:
            grab.update(self=self, x=x, q=q, k=k, v=v, cache=cache)
        return orig(self, x, q, k, v, cache)

    VF._bcsa_verify = capture
    inp = mx.array(gen[:8], dtype=mx.int32)[None]
    lg, cap, r = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
    mx.eval(lg, grab["x"], grab["q"], grab["k"], grab["v"])
    VF._bcsa_verify = orig
    a, x, q, k, v, c = (grab[n] for n in ("self", "x", "q", "k", "v", "cache"))
    P, W, C = c.offset, a.W, a.C
    nprev = min(P, W - 1)
    nv_max = max(0, (P + 8 - 1 - W + 1) // C)
    rec["nv_max"] = nv_max
    if nv_max == 0 or P < W:  # the isolated parts below are for the far regime
        rec["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return

    def rep17(fn):
        def f():
            mx.eval([fn() for _ in range(NL)])
        return f

    def win():
        wk, wv = c.window(nprev)
        return mx.concatenate([wk, k], axis=2), mx.concatenate([wv, v], axis=2)

    keys, vals = win()
    mx.eval(keys, vals)
    sel = a._select(x, c, P, nv_max)
    qi = a._index_q(x)
    s = kernels.bcsa_scores(qi[0], c.bk[0], nv_max, a.ID**-0.5)
    mx.eval(sel, qi, s)
    uidx, ucnt, umask = kernels.far_union(sel, nv_max, 32)
    mx.eval(ucnt)
    rec["union"] = int(ucnt[0].item())
    rec["ring_wraps"] = bool((P - nprev) % W + nprev > W)
    parts = {
        "window_concat": lambda: win(),
        "idx_q": lambda: a._index_q(x),
        "scores": lambda: kernels.bcsa_scores(qi[0], c.bk[0], nv_max, a.ID**-0.5),
        "topk": lambda: kernels.bcsa_topk(s, P, W, C, a.topk),
        "select": lambda: a._select(x, c, P, nv_max),
        "far_union": lambda: kernels.far_union(sel, nv_max, 32),
        "attend": lambda: kernels.bcsa_attend(q[0], keys[0], vals[0], c.pk[0], c.pv[0], sel, nv_max, a.scale, nprev, W),
        "window_attend": lambda: kernels.bcsa_window(q[0], keys[0], vals[0], a.scale, nprev, W),
        "bcsa_verify_total": lambda: impl("prefill"),
        "v8_total": lambda: impl("8q"),
    }

    def impl(name):
        keep, VF.BCSA_IMPL = VF.BCSA_IMPL, name
        try:
            return orig(a, x, q, k, v, c)[0]
        finally:
            VF.BCSA_IMPL = keep

    sq = lambda t: t.squeeze(0)
    for ntg in (8, 16, 32):
        parts[f"v8_attend_ntg{ntg}"] = lambda ntg=ntg: VF.bcsa_verify.attend(
            sq(q), sq(k), sq(v), sq(c.k), sq(c.v), P, W, a.scale, sq(c.pk), sq(c.pv), uidx, ucnt, umask, ntg=ntg)
    parts["v8_window_only"] = lambda: VF.bcsa_verify.attend(sq(q), sq(k), sq(v), sq(c.k), sq(c.v), P, W, a.scale)
    # integer indexing copies in MLX 0.32 (x[0] of the 9 MB ring, per call), squeeze is a view: a 1-thread kernel
    # reading one element of each
    probe = mx.fast.metal_kernel(name="probe_in", input_names=["a"], output_names=["out"],
                                 source="if (thread_position_in_grid.x == 0) out[0] = float(a[0]);")
    run1 = lambda t: probe(inputs=[t], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)],
                           output_dtypes=[mx.float32])[0]
    parts["probe_ring_index0"] = lambda: run1(c.k[0])
    parts["probe_ring_squeeze"] = lambda: run1(sq(c.k))
    parts["probe_pooled_index0"] = lambda: run1(c.pk[0])
    o_new, o_old = impl("8q"), impl("prefill")
    rec["v8_vs_prefill_maxdiff"] = float(mx.abs(o_new.astype(mx.float32) - o_old.astype(mx.float32)).max().item())
    rec["v8_vs_prefill_frac"] = float((o_new != o_old).astype(mx.float32).mean().item())
    # decode reference for one query on this layer (no cache write: SDPA over the ring + gathered far blocks)
    x1, q1 = x[:, :1], q[:, :, :1]
    sel1 = a._select(x1, c, P, (P - W + 1) // C)
    mx.eval(sel1)
    fk, fv = V._gather_blocks(c.pk, sel1), V._gather_blocks(c.pv, sel1)
    kk1, vv1 = mx.concatenate([c.k[..., :W, :], fk], axis=2), mx.concatenate([c.v[..., :W, :], fv], axis=2)
    mx.eval(kk1, vv1)
    parts["dec_select"] = lambda: a._select(x1, c, P, (P - W + 1) // C)
    parts["dec_sdpa"] = lambda: mx.fast.scaled_dot_product_attention(q1, kk1, vv1, scale=a.scale)
    parts["dec_sdpa_8q"] = lambda: mx.fast.scaled_dot_product_attention(q, kk1, vv1, scale=a.scale)
    for name, fn in parts.items():
        rec[f"{name}_x17_ms"] = timed(rep17(fn), args.reps)
    rec["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    line = json.dumps(rec)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
