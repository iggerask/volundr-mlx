"""Full-model gates for batched serving (NOTES_BATCHED_SERVE.md), in the worker's configuration (specdec.load with the
4-bit drafter and head copy, NAX, the spec command buffers).

  exact   a batch of one row (batch.forward) vs session.decode_step: logits bit for bit at 4 context lengths (inside
          the window, past it with far blocks <= top-k, with top-k selection), 48 steps each; capture_row restored and
          continued vs the B = 1 cache continued
  rows    8 prompts of 120-9,000 tokens joining a batch at different steps and decoding greedily together, against
          each prompt's own B = 1 greedy run: tokens, and the B = 1 / batched top-1 - top-2 margins at a divergence
  timing  ms per batched decode step at B = 1, 2, 4, 8, 16 (200-token prompts, the engine's synchronous step with its
          NAX policy)
  engine  scheduler.BatchEngine in-process: concurrent greedy + sampled requests, a session follow-up, a cancellation,
          against the same requests one at a time (max_batch 1): tokens, divergence margins, records
  serve   the server-mode workload (19 prompts x 256 tokens) on the engine in process, a pool of c clients
  bspec   batched speculation (batch_spec): verify + per-row commits against the B = 1 verify path on the same states
          and drafts; the engine with two concurrent greedy requests, batched speculation on / off, vs each alone

usage (from volundr-mlx, under ./with_gpu_lock.sh): python batched_serve/gate.py exact|rows|timing|engine --out F
"""
import os

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")  # the worker's (spec) command buffers
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spec"))
sys.path.insert(0, str(ROOT / "batched"))
sys.path.insert(0, str(HERE))
import batch as BT  # noqa: E402
import nax_hook  # noqa: E402
import session as S  # noqa: E402
import specdec  # noqa: E402
from bench_batched import make_prompts  # noqa: E402

OUT = None


def log(rec):
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tokens", "margins")}), flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")


def load():
    model, drafter, tok = specdec.load(4, 4, True)
    return model, drafter, tok


def prompts_of(tok, lengths, seed=0):
    out = []
    for i, n in enumerate(lengths):
        out.append(make_prompts(tok, n, i + 1 + seed)[i + seed])
    return out


def prefilled(model, drafter, ids):
    c, d = model.make_cache(), drafter.make_cache()
    last = S.prefill(model, drafter, c, d, ids)
    mx.eval(last)
    return c, d, last


def same(a, b):
    return bool(mx.array_equal(a, b).item())


def margin(lg, vv):
    t = mx.topk(lg[:vv].astype(mx.float32), 2)
    return float((t.max() - t.min()).item())


# ----------------------------------------------------------------------------- exact
def cmd_exact(args):
    model, drafter, tok = load()
    vv = model.args.vocab_valid
    capl = drafter.capture_layers
    for n, ids in zip(args.lengths, prompts_of(tok, args.lengths)):
        c1, d1, last = prefilled(model, drafter, ids)
        c2, _ = S.restore(S.capture(c1))
        bc = BT.BatchCache(model)
        bc.apply(join=[("r", c2)])
        t = int(mx.argmax(last[:vv]).item())
        diff, fdiff = [], []
        for i in range(args.steps):
            a, fa = S.decode_step(model, c1, t, capture=capl)
            b, fb = BT.forward(model, bc, mx.array([t]), capture=capl)
            mx.eval(a, fa, b, fb, [c.state for c in c1], bc.state)
            if not same(a, b[0]):
                diff.append(i)
            if not same(fa, fb):
                fdiff.append(i)
            t = int(mx.argmax(a[:vv]).item())
        # the row's snapshot, restored, continues like the B = 1 cache
        r2, _ = S.restore(bc.capture_row("r"))
        cont = []
        for j in range(8):
            a, _ = S.decode_step(model, c1, t)
            b, _ = S.decode_step(model, r2, t)
            mx.eval(a, b)
            cont.append(same(a, b))
            t = int(mx.argmax(a[:vv]).item())
        log({"gate": "exact", "prompt_tokens": n, "steps": args.steps, "logits_differ": diff,
             "features_differ": fdiff, "capture_row_continues": all(cont),
             "ring_rows": next(c.R for c in bc.layers if isinstance(c, BT.BatchBCSACache))})
        del c1, c2, bc, r2
        mx.clear_cache()


# ----------------------------------------------------------------------------- rows
def greedy_ref(model, drafter, ids, steps, vv):
    c, _, last = prefilled(model, drafter, ids)
    toks, margins, lg = [], [], last
    for _ in range(steps):
        t = int(mx.argmax(lg[:vv]).item())
        toks.append(t)
        margins.append(margin(lg, vv))
        lg, _ = S.decode_step(model, c, t)
        mx.eval(lg, [x.state for x in c])
    return toks, margins


def cmd_rows(args):
    model, drafter, tok = load()
    vv = model.args.vocab_valid
    ps = prompts_of(tok, args.lengths)
    joins = [int(j) for j in args.joins.split(",")]
    refs = [greedy_ref(model, drafter, ids, args.steps, vv) for ids in ps]
    mx.clear_cache()
    bc = BT.BatchCache(model)
    toks = {i: [] for i in range(len(ps))}
    margs = {i: [] for i in range(len(ps))}
    nxt, step_ms, t = {}, [], 0
    while any(len(toks[i]) < args.steps for i in toks):
        join = []
        for i, ids in enumerate(ps):
            if joins[i] == t:
                c, _, last = prefilled(model, drafter, ids)
                nxt[i] = int(mx.argmax(last[:vv]).item())
                toks[i].append(nxt[i])
                margs[i].append(margin(last, vv))
                join.append((i, c))
        leave = [i for i in bc.keys if len(toks[i]) >= args.steps]
        bc.apply(leave=leave, join=join)
        t += 1
        if not bc.B:
            continue
        t0 = time.perf_counter()
        B = bc.B
        with nax_hook.min_rows(4 if B < 4 else 2):
            lg, _ = BT.forward(model, bc, mx.array([nxt[i] for i in bc.keys]))
        g = mx.argmax(lg[:, :vv], axis=-1)
        mx.eval(lg, g, bc.state)
        step_ms.append((B, 1e3 * (time.perf_counter() - t0), list(bc.pos)))
        for r, i in enumerate(bc.keys):
            nxt[i] = g[r].item()
            toks[i].append(nxt[i])
            margs[i].append(margin(lg[r], vv))
    res = []
    for i, (n, (rt, rm)) in enumerate(zip(args.lengths, refs)):
        div = next((k for k, (a, b) in enumerate(zip(rt, toks[i])) if a != b), None)
        r = {"prompt_tokens": n, "join": joins[i], "identical": div is None}
        if div is not None:
            r.update(diverge_at=div, ref_margin=rm[div], batched_margin=margs[i][div])
        res.append(r)
    log({"gate": "rows", "steps": args.steps, "rows": res, "step_ms": step_ms})


# ----------------------------------------------------------------------------- timing
def cmd_timing(args):
    model, drafter, tok = load()
    vv = model.args.vocab_valid
    ps = make_prompts(tok, args.tokens, max(args.batches))
    caches = []
    for ids in ps:
        c, _, last = prefilled(model, drafter, ids)
        caches.append((c, int(mx.argmax(last[:vv]).item())))
    def build(bc, toks, B):
        with nax_hook.min_rows(4 if B < 4 else 2):
            lg, _ = BT.forward(model, bc, toks)
        return lg, mx.argmax(lg[:, :vv], axis=-1).astype(mx.int32), bc.state

    for B in args.batches:
        for pipe in (False, True):  # the engine's step: synchronous, or the next step built while one runs
            bc = BT.BatchCache(model)
            bc.apply(join=[(i, S.restore(S.capture(caches[i][0]))[0]) for i in range(B)])
            toks = mx.array([caches[i][1] for i in range(B)], dtype=mx.int32)
            times, builds, toks_seen = [], [], []
            mx.reset_peak_memory()
            for s in range(args.warm + args.steps):
                t0 = time.perf_counter()
                pre = bc.take_prebuilt("next") if pipe else None
                if pre is None:
                    pre = build(bc, toks, B)
                tb = time.perf_counter()
                lg, g, st = pre
                if pipe:
                    mx.async_eval(lg, g, st)
                    bc.prebuild("next", lambda g=g, bc=bc: build(bc, g, B))
                mx.eval(lg, g, st)
                del pre, st
                toks_seen.append(g.tolist())
                toks = g
                times.append(time.perf_counter() - t0)
                builds.append(tb - t0)
            bc.rollback()
            ms = 1e3 * sum(times[args.warm:]) / args.steps
            log({"gate": "timing", "B": B, "pipelined": pipe, "prompt_tokens": args.tokens, "ms_step": round(ms, 2),
                 "agg_tok_s": round(B * 1e3 / ms, 1), "build_ms": round(1e3 * sum(builds[args.warm:]) / args.steps, 2),
                 "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "batch_gb": round(bc.nbytes / 1e9, 3),
                 "tokens": toks_seen})
            del bc
            mx.clear_cache()


# ----------------------------------------------------------------------------- engine
class _Conn:
    def __init__(self, cancel_after=None, job=None):
        self.frames, self.cancel_after, self.job = [], cancel_after, job
        self.gone = threading.Event()

    def send(self, obj):
        self.frames.append(obj)
        if self.cancel_after is not None and sum(len(f.get("ids", [])) for f in self.frames) >= self.cancel_after:
            self.job.cancel.set()
        return True

    def close(self):
        pass


class _Job:
    def __init__(self, msg, cancel_after=None):
        self.msg = msg
        self.t_recv = time.perf_counter()
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.context_length = 262144
        self.conn = _Conn(cancel_after, self)


def run_engine(engine, reqs):
    """Submit all reqs at once; tick until done. Returns {name: (tokens, rec)}."""
    out = {}

    def done(job, req, rec):
        out[job.name] = (rec.get("tokens"), rec)

    engine.on_done = done
    for name, msg, cancel_after in reqs:
        j = _Job(msg, cancel_after)
        j.name = name
        engine.submit(j)
    while engine.busy:
        engine.tick()
    return out


def tie(model, drafter, ids, prefix, a, b, vv):
    """Logits of tokens a and b after ids + prefix (B = 1 prefill + decode steps), and the top-2 margin there."""
    c, _, lg = prefilled(model, drafter, ids)
    for t in prefix:
        lg, _ = S.decode_step(model, c, t)
    mx.eval(lg)
    lf = lg.astype(mx.float32)
    return float(lf[a].item()), float(lf[b].item()), margin(lg, vv)


def cmd_engine(args):
    import scheduler

    model, drafter, tok = load()
    vv = model.args.vocab_valid
    lens = [150, 400, 900, 1600, 220, 600]
    ps = prompts_of(tok, lens, seed=3)
    g = {"top_k": 1, "max_new_tokens": args.new}
    smp = lambda seed: {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "max_new_tokens": args.new, "seed": seed}
    reqs = [(f"g{i}", {"prompt_ids": ps[i], "session_id": f"s{i}", "sampling": g}, None) for i in range(4)]
    reqs += [("x4", {"prompt_ids": ps[4], "sampling": smp(11)}, None),
             ("x5", {"prompt_ids": ps[5], "session_id": "s5", "sampling": smp(12)}, None),
             ("c0", {"prompt_ids": ps[2][:500], "sampling": g}, args.new // 3)]  # cancelled after a third
    results, fps = {}, {}
    for mode, mb in (("concurrent", args.max_batch), ("sequential", 1)):
        cache = S.SessionCache(budget_gb=8)
        eng = scheduler.BatchEngine(model, drafter, cache, max_batch=mb)
        t0 = time.perf_counter()
        r1 = run_engine(eng, reqs)
        t1 = time.perf_counter()
        # session follow-up: turn 2 = turn 1 prompt + its reply + more text
        follow = []
        for i in (0, 1):
            toks = r1[f"g{i}"][0]
            more = tok.encode("\n\nNow say the same in one sentence.", add_special_tokens=False)
            follow.append((f"f{i}", {"prompt_ids": ps[i] + toks + more, "session_id": f"s{i}", "sampling": g}, None))
        r2 = run_engine(eng, follow)
        t2 = time.perf_counter()
        fprompts = {name: msg["prompt_ids"] for name, msg, _ in follow}
        fps[mode] = fprompts
        results[mode] = {**r1, **r2}
        for name, (toks, rec) in sorted({**r1, **r2}.items()):
            log({"gate": "engine", "mode": mode, "name": name, "generated": rec.get("generated"),
                 "finish": rec.get("finish"), "decode_mode": rec.get("mode"), "mean_batch": rec.get("mean_batch"),
                 "switches": rec.get("switches"), "cached_tokens": rec.get("cached_tokens"),
                 "snapshot": rec.get("snapshot"), "ttft_s": rec.get("ttft_s"), "decode_tok_s": rec.get("decode_tok_s"),
                 "tokens": toks})
        log({"gate": "engine", "mode": mode, "wall_s_turn1": round(t1 - t0, 2), "wall_s_turn2": round(t2 - t1, 2),
             "engine_stats": eng.stats, "cache_snapshots": len(cache.snaps)})
        del eng, cache
        mx.clear_cache()
    # compare concurrent vs sequential; margins at the first divergence
    allp = {f"g{i}": ps[i] for i in range(4)} | {"x4": ps[4], "x5": ps[5]} | fprompts
    for name in sorted(results["sequential"]):
        a, ra = results["sequential"][name]
        b, rb = results["concurrent"][name]
        div = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
        rec = {"gate": "engine-compare", "name": name, "identical": div is None and len(a) == len(b),
               "len": [len(a), len(b)]}
        if name[0] == "f" and fps["concurrent"][name] != fps["sequential"][name]:
            rec["prompt_differs"] = True  # turn 1 diverged, so turn 2's prompts differ: not comparable
        elif div is not None and name in allp and name[0] in "gf":
            la, lb, m = tie(model, drafter, allp[name], a[:div], a[div], b[div], vv)
            rec.update(diverge_at=div, logit_seq=la, logit_conc=lb, top2_margin=m)
        elif div is not None:
            rec.update(diverge_at=div)
        log(rec)


# ----------------------------------------------------------------------------- bspec
def cmd_bspec(args):
    """Batched verify + per-row commit (batch_spec) against the B = 1 verify path (verify.forward + commit) on the
    same states and drafts, random accepted counts: per-row logits and the final states; then the engine with two
    concurrent greedy requests, batched speculation on and off."""
    import random

    import batch_spec
    import scheduler
    import verify as VF

    model, drafter, tok = load()
    vv = model.args.vocab_valid
    capl = drafter.capture_layers
    ps = prompts_of(tok, args.lengths[:2], seed=5)
    rng = random.Random(0)
    refs = []
    for ids in ps:
        c, d, last = prefilled(model, drafter, ids)
        refs.append((c, int(mx.argmax(last[:vv]).item())))
    bc = BT.BatchCache(model)
    bc.apply(join=[(i, S.restore(S.capture(c))[0]) for i, (c, _) in enumerate(refs)])
    b1 = [S.restore(S.capture(c))[0] for c, _ in refs]
    bonus = [t for _, t in refs]
    worst, agree, rows_n, kl = 0.0, 0, 0, []
    for step in range(args.steps):
        inps, ns = [], []
        for i in range(2):  # drafts: the B = 1 greedy continuation with some tokens replaced, so n varies
            inps.append([bonus[i]] + [rng.randrange(1000, 200000) for _ in range(args.T - 1)])
        inp = mx.array(inps, dtype=mx.int32)
        lb, capb, recb = batch_spec.verify_forward(model, bc, inp, capture=capl)
        mx.eval(lb, capb, bc.state)
        for i in range(2):
            lr, capr, recr = VF.forward(model, inp[i : i + 1], b1[i], capture=capl, verify=True)
            mx.eval(lr, capr, [c.state for c in b1[i]])
            a, b = lr.astype(mx.float32), lb[i].astype(mx.float32)
            worst = max(worst, mx.abs(a - b).max().item())
            agree += int(mx.sum(mx.argmax(a[:, :vv], -1) == mx.argmax(b[:, :vv], -1)).item())
            rows_n += a.shape[0]
            pa, pb = mx.softmax(a, -1), mx.softmax(b, -1)
            kl.append(mx.sum(pa * (mx.log(pa + 1e-30) - mx.log(pb + 1e-30)), -1).max().item())
            n = rng.randint(1, args.T)
            ns.append(n)
            VF.commit(b1[i], recr, n)
            bonus[i] = int(mx.argmax(lr[n - 1, :vv]).item())
        batch_spec.commit(bc, recb, ns)
        mx.eval(bc.state, [c.state for c in b1[0] + b1[1]])
    same_state = []
    for i in range(2):
        r2, _ = S.restore(bc.capture_row(i))
        d = 0.0
        for x, y in zip(b1[i], r2):
            xs = [a for a in (x.cache if hasattr(x, "cache") else x.state) if a is not None] if not hasattr(x, "trim") \
                else [x.keys[..., : x.offset, :], x.values[..., : x.offset, :]]
            ys = [a for a in (y.cache if hasattr(y, "cache") else y.state) if a is not None] if not hasattr(y, "trim") \
                else [y.keys[..., : y.offset, :], y.values[..., : y.offset, :]]
            for p_, q_ in zip(xs, ys):
                if p_.shape == q_.shape:
                    d = max(d, mx.abs(p_.astype(mx.float32) - q_.astype(mx.float32)).max().item())
        same_state.append(d)
        assert S.cache_pos(b1[i]) == bc.pos[i]
    log({"gate": "bspec-verify", "prompt_tokens": args.lengths[:2], "steps": args.steps, "rows": rows_n,
         "max_abs_logit_diff": worst, "argmax_agree": agree, "max_kl": max(kl), "state_max_abs_diff": same_state})
    del bc, b1, refs
    mx.clear_cache()
    # engine: two concurrent greedy requests, batched speculation on / off, and each alone (the spec loop)
    lens = [180, 700]
    ps = prompts_of(tok, lens, seed=7)
    g = {"top_k": 1, "max_new_tokens": args.new}
    out = {}
    for mode in ("alone", "bspec", "plain"):
        eng = scheduler.BatchEngine(model, drafter, None, bspec=(mode == "bspec"))
        reqs = [(f"r{i}", {"prompt_ids": ps[i], "sampling": g}, None) for i in range(2)]
        t0 = time.perf_counter()
        if mode == "alone":
            res = {}
            for rq in reqs:
                res.update(run_engine(eng, [rq]))
        else:
            res = run_engine(eng, reqs)
        wall = time.perf_counter() - t0
        out[mode] = res
        st = eng.stats
        log({"gate": "bspec-engine", "mode": mode, "wall_s": round(wall, 2),
             "engine": st, "recs": {k: {x: v[1].get(x) for x in ("generated", "decode_tok_s", "bspec_steps",
                                                                  "bspec_tokens", "mode", "ttft_s")}
                                     for k, v in res.items()}})
        del eng
        mx.clear_cache()
    for name in ("r0", "r1"):
        a = out["alone"][name][0]
        for mode in ("bspec", "plain"):
            b = out[mode][name][0]
            div = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
            rec = {"gate": "bspec-compare", "name": name, "mode": mode, "identical_to_alone": div is None}
            if div is not None:
                la, lb, m = tie(model, drafter, ps[int(name[1])], a[:div], a[div], b[div], vv)
                rec.update(diverge_at=div, logit_alone=la, logit_other=lb, top2_margin=m)
            log(rec)


# ----------------------------------------------------------------------------- serve (in-process server-mode run)
def cmd_serve(args):
    """The server-mode workload (serve_bench.py's 19 prompts x --new tokens, greedy, ignore_eos) on the engine in
    process: a pool of c clients, each submitting the next prompt when its last one ends; no HTTP. Per level: the
    aggregate, the engine's time split, step counts."""
    import scheduler
    from serve_bench import PROMPTS

    model, drafter, tok = load()
    ids = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                   enable_thinking=False) for t in PROMPTS]
    g = {"top_k": 1, "max_new_tokens": args.new, "ignore_eos": True}
    for c in args.batches:
        for bspec in ([True, False] if c == 2 and args.bspec_ab else [args.bspec]):
            eng = scheduler.BatchEngine(model, drafter, S.SessionCache(budget_gb=args.cache_gb), bspec=bspec)
            todo = list(range(len(ids)))
            recs = {}

            def submit(eng=eng):
                i = todo.pop(0)
                j = _Job({"prompt_ids": ids[i], "sampling": g})
                j.name = i
                eng.submit(j)

            def done(job, req, rec, eng=eng):
                recs[job.name] = rec
                if todo:
                    submit()

            eng.on_done = done
            for _ in range(min(c, len(todo))):
                submit()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            while eng.busy:
                eng.tick()
            wall = time.perf_counter() - t0
            toks = sum(r.get("generated", 0) for r in recs.values())
            st = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in eng.stats.items()}
            log({"gate": "serve", "concurrency": c, "bspec": bspec, "requests": len(recs), "tokens": toks,
                 "wall_s": round(wall, 2), "agg_tok_s": round(toks / wall, 2),
                 "ttft_p50_s": round(sorted(r["ttft_s"] for r in recs.values())[len(recs) // 2], 3),
                 "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "engine": st,
                 "outputs": {i: r.get("tokens") for i, r in recs.items()}})
            del eng
            mx.clear_cache()


# ----------------------------------------------------------------------------- prof (where the serving step goes)
def cmd_prof(args):
    """B rows decoding: equal prompts (the timing gate) vs the serve prompts (rows at different positions), without
    and with the drafter feature capture; then the engine on the same serve prompts (all admitted at once), its
    decode steps by batch size once every prefill is done."""
    import scheduler
    from serve_bench import PROMPTS

    model, drafter, tok = load()
    vv = model.args.vocab_valid
    B = max(args.batches)
    serve_ids = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                         enable_thinking=False) for t in PROMPTS][:B]
    lens = [len(x) for x in serve_ids]
    eq = make_prompts(tok, sorted(lens)[len(lens) // 2], B)

    def build(bc, toks, capl, n):
        with nax_hook.min_rows(4 if n < 4 else 2):
            lg, f = BT.forward(model, bc, toks, capture=capl)
        return lg, mx.argmax(lg[:, :vv], axis=-1).astype(mx.int32), [] if f is None else f, bc.state

    for name, ps in (("equal", eq), ("serve", serve_ids)) if args.steps else ():  # --steps 0: the engine only
        caches = []
        for ids in ps:
            c, _, last = prefilled(model, drafter, ids)
            caches.append((c, int(mx.argmax(last[:vv]).item())))
        for n in args.batches:
            for capl in ((), drafter.capture_layers):
                bc = BT.BatchCache(model)
                bc.apply(join=[(i, S.restore(S.capture(caches[i][0]))[0]) for i in range(n)])
                toks = mx.array([caches[i][1] for i in range(n)], dtype=mx.int32)
                times = []
                for s in range(args.warm + args.steps):
                    t0 = time.perf_counter()
                    pre = bc.take_prebuilt("next") or build(bc, toks, capl, n)
                    lg, g, f, st = pre
                    mx.async_eval(lg, g, f, st)
                    bc.prebuild("next", lambda g=g, bc=bc, capl=capl, n=n: build(bc, g, capl, n))
                    mx.eval(lg, g, f, st)
                    del pre, st
                    g.tolist()
                    toks = g
                    times.append(time.perf_counter() - t0)
                bc.rollback()
                ms = 1e3 * sum(times[args.warm:]) / args.steps
                log({"gate": "prof", "prompts": name, "B": n, "capture": bool(capl),
                     "prompt_tokens": [len(x) for x in ps[:n]], "ms_step": round(ms, 2),
                     "agg_tok_s": round(n * 1e3 / ms, 1)})
                del bc
                mx.clear_cache()
        del caches
    g = {"top_k": 1, "max_new_tokens": args.new, "ignore_eos": True}
    for cache_gb in (12.0, 0.0)[: args.engines]:
        eng = scheduler.BatchEngine(model, drafter, S.SessionCache(budget_gb=cache_gb) if cache_gb else None)
        eng.step_log = []
        recs = {}
        eng.on_done = lambda job, req, rec, recs=recs: recs.__setitem__(job.name, rec)
        for i, ids in enumerate(serve_ids):
            j = _Job({"prompt_ids": ids, "sampling": g})
            j.name = i
            eng.submit(j)
        t0 = time.perf_counter()
        while eng.busy:
            eng.tick()
        wall = time.perf_counter() - t0
        sl = eng.step_log
        full = [dt for n, dt, pw in sl if n == B and pw == 0.0]
        with_pre = [(dt, pw) for n, dt, pw in sl if pw > 0.0]
        st = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in eng.stats.items()}
        log({"gate": "prof-engine", "session_cache": bool(cache_gb), "B": B, "wall_s": round(wall, 2),
             "steps": len(sl), "full_steps": len(full),
             "full_ms_median": round(1e3 * sorted(full)[len(full) // 2], 2) if full else None,
             "full_ms_mean": round(1e3 * sum(full) / len(full), 2) if full else None,
             "ticks_with_prefill": len(with_pre),
             "prefill_s": round(sum(pw for _, pw in with_pre), 2),
             "step_ms_with_prefill_mean": round(1e3 * sum(dt for dt, _ in with_pre) / len(with_pre), 2)
             if with_pre else None,
             "ttft_s": sorted(round(r["ttft_s"], 2) for r in recs.values()), "engine": st,
             "step_log": [(n, round(dt * 1e3, 1), round(pw * 1e3, 1)) for n, dt, pw in sl]})
        del eng
        mx.clear_cache()


# ----------------------------------------------------------------------------- bspecn (batched spec at B rows)
def cmd_bspecn(args):
    """k concurrent greedy requests (the serve prompts, --new tokens each, ignore_eos) on the engine: batched
    speculation at k rows (T = 16 // k) against plain batched decode; decode tok/s over the decode steps."""
    import scheduler
    from serve_bench import PROMPTS

    model, drafter, tok = load()
    ids = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                   enable_thinking=False) for t in PROMPTS]
    g = {"top_k": 1, "max_new_tokens": args.new, "ignore_eos": True}
    for k in args.batches:
        res = {}
        for mode in ("bspec", "plain"):
            eng = scheduler.BatchEngine(model, drafter, None, bspec=(mode == "bspec"), bspec_max=max(args.batches))
            reqs = [(f"r{i}", {"prompt_ids": ids[i], "sampling": g}, None) for i in range(k)]
            t0 = time.perf_counter()
            res[mode] = run_engine(eng, reqs)
            wall = time.perf_counter() - t0
            st = eng.stats
            toks = sum(v[1]["generated"] for v in res[mode].values())
            dec_s = st["t_bspec"] + st["t_batch"] + st["t_spec"]
            log({"gate": "bspecn", "k": k, "mode": mode, "T": max(2, min(8, 16 // k)), "wall_s": round(wall, 2),
                 "tokens": toks, "decode_s": round(dec_s, 2), "decode_tok_s": round(toks / dec_s, 1),
                 "steps": {x: st[x] for x in ("steps_bspec", "steps_batch", "steps_spec")},
                 "accepted_per_row_step": round(sum(v[1].get("bspec_tokens", 0) for v in res[mode].values())
                                                / max(1, k * st["steps_bspec"]), 2),
                 "ms_per_bspec_step": round(1e3 * st["t_bspec"] / max(1, st["steps_bspec"]), 1),
                 "ms_per_batch_step": round(1e3 * st["t_batch"] / max(1, st["steps_batch"]), 1),
                 "peak_gb": round(mx.get_peak_memory() / 1e9, 2)})
            del eng
            mx.clear_cache()
        same = sum(res["bspec"][n][0] == res["plain"][n][0] for n in res["plain"])
        div = [next((j for j, (x, y) in enumerate(zip(res["bspec"][n][0], res["plain"][n][0])) if x != y), None)
               for n in res["plain"]]
        log({"gate": "bspecn-compare", "k": k, "identical": same, "of": k, "diverge_at": div})


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["exact", "rows", "timing", "engine", "bspec", "serve", "prof", "bspecn"])
    ap.add_argument("--cache-gb", type=float, default=12.0)
    ap.add_argument("--bspec", type=int, default=1)
    ap.add_argument("--bspec-ab", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lengths", default="150,3000,4200,9000")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--joins", default="0,0,4,8,12,16,20,24")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--engines", type=int, default=2)
    args = ap.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]
    args.batches = [int(x) for x in args.batches.split(",")]
    OUT = args.out
    {"exact": cmd_exact, "rows": cmd_rows, "timing": cmd_timing, "engine": cmd_engine, "bspec": cmd_bspec,
     "serve": cmd_serve, "prof": cmd_prof,
     "bspecn": cmd_bspecn}[
        args.cmd](args)


if __name__ == "__main__":
    main()
