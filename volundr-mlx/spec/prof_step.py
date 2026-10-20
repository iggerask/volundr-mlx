"""Where a speculative step's time goes, on the real loop (spec_generate's ops, frozen config, lookup on).

Per prompt the loop runs twice from the same start (fresh caches, fresh Lookup, so both runs take the same steps):
- pipelined: specdec.spec_generate's schedule, with host timestamps around each CPU part:
    book     .item() reads of the step's result, the emitted list, lookup.record (host)
    commit   VF.commit graph build (CPU)
    mat      drafter.materialize graph build (CPU)
    lookup   lookup.propose (host)
    draft    drafter.propose graph build (CPU; 0 when the lookup fills all 7 drafts)
    async    mx.async_eval of the step's first GPU work (encoding + commit of the command buffers)
    vbuild   VF.forward verify graph build (CPU), argmax / accept graph
    wait     mx.eval of the step's results (encoding the verify, then blocked on the GPU)
  so wall = sum of these;
- serialized: the same ops, but every GPU part is evaluated on its own from an idle GPU and timed with a sync:
    g_commit, g_mat, g_draft, g_verify (verify forward + argmax + accept count; includes encoding, which overlaps the
    GPU from the first command buffer on), plus the same CPU parts.
GPU busy per step ~ g_commit + g_mat + g_draft + g_verify; the pipelined wall minus that is GPU idle time (host work the
pipeline does not hide, sync latency). The tokens of both runs must match spec_generate's.

usage (from volundr-mlx/spec, under ../with_gpu_lock.sh; VOLUNDR_LMHEAD8=64 for the baseline config):
  python prof_step.py --names coding/00-12623519-think,long/01-long1 --new 160 --out runs/spec-accept_prof_x.jsonl
"""
import os
import sys

if "MLX_MAX_OPS_PER_BUFFER" not in os.environ:  # as specdec.py bench
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "100"
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import specdec as SD  # noqa: E402
import verify as VF  # noqa: E402
from lookup import Lookup  # noqa: E402

PARTS = ("book", "commit", "mat", "lookup", "draft", "async", "vbuild", "wait")
GPARTS = ("g_commit", "g_mat", "g_draft", "g_verify")


def run(model, drafter, ids, max_new, eos, serial, T=8, compiled=False, long=0):
    """spec_generate's loop (same ops, same order) with per-step timings; serial=True evaluates each GPU part alone."""
    vv = model.args.vocab_valid
    lookup = Lookup()
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = SD.spec_prefill(model, drafter, cache, dcache, ids)
    bonus = mx.argmax(last[:vv]).astype(mx.int32)
    out = [bonus.item()]
    P = len(ids)
    lookup.reset(list(ids) + out)
    steps = []
    pend = None  # (rec, cap, n, emitted) of the previous step, committed at the start of this one
    t_start = time.perf_counter()
    te = t_start
    while True:
        st = {k: 0.0 for k in PARTS + GPARTS}
        t0 = te
        if pend is not None:
            rec, cap, n = pend
            t1 = time.perf_counter()
            VF.commit(cache, rec, n)
            t2 = time.perf_counter()
            if serial:
                mx.eval([c.state for c in cache])
                t2b = time.perf_counter()
                st["g_commit"] = t2b - t2
                t2 = t2b
            drafter.materialize(dcache, cap[:n], P - n)
            t3 = time.perf_counter()
            if serial:
                mx.eval(dcache.state)
                t3b = time.perf_counter()
                st["g_mat"] = t3b - t3
                t3 = t3b
            st["commit"], st["mat"] = t2 - t1 - st["g_commit"], t3 - t2 - st["g_mat"]
            t0 = t3
        if len(out) >= max_new or out[-1] in eos:
            break
        hint = lookup.propose(long - 1) if long else lookup.propose()
        t4 = time.perf_counter()
        st["lookup"] = t4 - t0
        if len(hint) >= T - 1:
            hint = hint[: SD._long_copy(hint, lookup, T, long, 0)]
            drafts = mx.array(hint, dtype=mx.int32)
        else:
            drafts = drafter.propose(dcache, bonus, P)
            if hint:
                drafts = mx.concatenate([mx.array(hint, dtype=mx.int32), drafts[len(hint):]])
        inp = mx.concatenate([bonus.reshape(1), drafts if len(hint) > T - 1 else drafts[: T - 1]])[None]
        t5 = time.perf_counter()
        st["draft"] = t5 - t4
        if serial:
            mx.eval(inp)
            t5b = time.perf_counter()
            st["g_draft"] = t5b - t5
            t5 = t5b
        else:
            mx.async_eval(inp)
        t6 = time.perf_counter()
        st["async"] = t6 - t5 if not serial else 0.0
        logits, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True,
                                      compiled=compiled)
        tok = mx.argmax(logits[:, :vv], axis=-1).astype(mx.int32)
        nacc = mx.cumprod((inp[0, 1:] == tok[:-1]).astype(mx.int32)).sum()
        t7 = time.perf_counter()
        st["vbuild"] = t7 - t6
        mx.eval(tok, nacc, inp, [c.state for c in cache], dcache.state)
        t8 = time.perf_counter()
        if serial:
            st["g_verify"] = t8 - t7
        else:
            st["wait"] = t8 - t7
        n = nacc.item() + 1
        emitted = inp[0, 1:n].tolist() + [tok[n - 1].item()]
        stop = next((i for i, t in enumerate(emitted) if t in eos), None)
        te = time.perf_counter()
        st["book"] = te - t8
        st.update(n=n, src=len(hint), P=P)
        if stop is not None:
            out += emitted[: stop + 1]
            steps.append(st)
            break
        out += emitted
        lookup.record(emitted)
        st["book"] = time.perf_counter() - t8
        te = time.perf_counter()
        steps.append(st)
        bonus = tok[n - 1]
        P += n
        pend = (rec, cap, n)
    # the commit of the last step is not part of spec_generate's timed loop when it ends on max_new, but its commit
    # / materialize ran at the end of that step there: here they ran at the start of the next iteration and were
    # booked on the last step, which is then dropped from the per-step stats (it has no verify)
    wall = time.perf_counter() - t_start
    return out[:max_new], steps, wall, (dcache, bonus, P)


def drafter_parts(drafter, dcache, bonus, P, reps=7):
    """GPU time of the drafter's pieces from an idle GPU (median ms): the 5-layer block pass, + the head over 7 rows,
    + top-16 (argpartition, sort), the whole propose (+ the walk)."""
    import dflash as DFL

    def hid():
        return drafter.hidden(dcache, bonus, P)[1:]

    def head():
        return drafter.head(hid())[:, : drafter.vocab_valid]

    def topk():
        lg = head()
        idx = mx.argpartition(-lg, DFL.TOPK - 1, axis=-1)[:, : DFL.TOPK]
        return mx.take_along_axis(lg, idx, axis=-1)

    res = {}
    for k, f in (("hidden", hid), ("hidden+head", head), ("hidden+head+top16", topk),
                 ("propose", lambda: drafter.propose(dcache, bonus, P))):
        ts = []
        for _ in range(reps):
            mx.synchronize()
            t0 = time.perf_counter()
            x = f()
            t1 = time.perf_counter()
            mx.eval(x)
            t2 = time.perf_counter()
            ts.append((t2 - t1, t1 - t0))
        ts.sort()
        res[k] = {"gpu_ms": round(1e3 * ts[reps // 2][0], 2), "build_ms": round(1e3 * ts[reps // 2][1], 2)}
    return res


def summarize(steps, keys):
    full = [s for s in steps if "n" in s]
    res = {}
    for k in keys:
        v = [1e3 * s[k] for s in full]
        res[k] = round(statistics.mean(v), 2)
    res["steps"] = len(full)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="coding/00-12623519-think,long/01-long1")
    ap.add_argument("--new", type=int, default=160)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compiled", action="store_true", help="compiled KDA verify layers (loop flag cv)")
    ap.add_argument("--long", type=int, default=0, help="long lookup copies (loop flag long16: 16)")
    args = ap.parse_args()
    model, drafter, tok = SD.load(drafter_bits=4, lm_head_bits=4, nax=True)
    eos = set(tok.eos_token_ids)
    suite = {e["name"]: e for e in json.loads((HERE / "suite.json").read_text())}
    w = tok.encode("Warm up the kernels, please.")
    SD.spec_generate(model, drafter, w, 24, 8)
    SD.warm_up(model, drafter, w, 8, compiled=args.compiled, long=args.long)
    run(model, drafter, w, 24, eos, serial=True, compiled=args.compiled, long=args.long)
    fout = open(args.out, "a") if args.out else None
    for name in args.names.split(","):
        ids = suite[name]["ids"]
        # the reference: spec_generate with the same options (the base loop; the early loop with long copies)
        ref, rr = SD.spec_generate(model, drafter, ids, args.new, 8, eos, lookup=Lookup(), compiled=args.compiled,
                                   long=args.long, sched="early" if args.long else None)
        # the early loop's own CPU segments (book + draft build | async_eval + commit build | verify build) and wait
        ge, re_ = SD.spec_generate(model, drafter, ids, args.new, 8, eos, lookup=Lookup(), compiled=args.compiled,
                                   long=args.long, sched="early")
        assert ge == ref
        res_early = {"ms_step": round(1e3 * re_["decode_s"] / re_["steps"], 2),
                     "seg_ms": [round(1e3 * x / re_["steps"], 2) for x in re_["seg_s"]],
                     "wait_ms": round(1e3 * re_["wait_s"] / re_["steps"], 2)}
        res = {"name": name, "prompt_tokens": len(ids),
               "target_head_q8": model.lm_head.group_size if "scales" in model.lm_head else 0,
               "spec_generate_ms_step": round(1e3 * rr["decode_s"] / rr["steps"], 2),
               "compiled": args.compiled, "long": args.long, "early_loop": res_early}
        for serial in (False, True):
            got, steps, wall, fin = run(model, drafter, ids, args.new, eos, serial, compiled=args.compiled,
                                        long=args.long)
            assert got == ref, f"{name}: the instrumented loop (serial={serial}) diverged from spec_generate"
            # per-step stats over the steps that have a verify, booking each step's commit / materialize (done at the
            # start of the next iteration) on the step that produced them
            sh = []
            for i, s in enumerate(steps):
                if "n" not in s:
                    continue
                d = dict(s)
                nxt = steps[i + 1] if i + 1 < len(steps) else None
                for k in ("commit", "mat", "g_commit", "g_mat"):
                    d[k] = nxt[k] if nxt is not None else 0.0
                sh.append(d)
            sh = sh[:-1]  # the last step's commit happens after the timed loop
            tag = "serial" if serial else "pipe"
            keys = PARTS + (GPARTS if serial else ())
            s_all = summarize(sh, keys)
            s_all["wall_ms_step"] = round(sum(1e3 * sum(x[k] for k in keys) for x in sh) / len(sh), 2)
            s_draft = summarize([x for x in sh if x["src"] < 7], keys)
            s_lk = summarize([x for x in sh if x["src"] >= 7], keys) if any(x["src"] >= 7 for x in sh) else None
            res[tag] = {"all": s_all, "drafter_steps": s_draft, "lookup7_steps": s_lk}
            res[f"{tag}_tokens_per_step"] = round(sum(x["n"] for x in sh) / len(sh), 3)
            if serial:
                res["drafter_parts"] = drafter_parts(drafter, *fin)
            del fin
        print(json.dumps(res), flush=True)
        if fout:
            fout.write(json.dumps(res) + "\n")
            fout.flush()
        mx.clear_cache()


if __name__ == "__main__":
    main()
