"""Greedy speculative decoding for Volundr MLX with the DFlash2 block drafter, and the plain greedy baseline.

One step: the drafter proposes 7 tokens after the bonus token in one parallel pass; the target verifies
[bonus, d1..d_{T-1}] in one forward; the longest prefix of drafts that matches the target's argmax is kept, plus the
target's next token (the new bonus). Then the caches keep those n tokens (verify.commit) and the drafter's context
K/V gets their target features.

usage (from volundr-mlx/spec):
  python specdec.py check  [--suite suite.json] [--split tune] [--per-cat 1] [--new 128]   greedy == spec?
  python specdec.py bench  [--suite suite.json] [--split tune] [--new 256] [--T 8] --out runs/x.jsonl
  python specdec.py base   [--suite suite.json] [--split tune] [--new 256] --out runs/base.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Speculative runs: larger Metal command buffers. MLX lets the CPU run at most ~10 command buffers ahead of the GPU, and
# the drafter's ~400 kernels span more than that at the default size, so async_eval of the draft blocked until it was
# nearly done (26 ms) and the verify graph was built with the GPU idle. With 100 ops per buffer it returns in ~1 ms.
# Plain greedy decoding is ~4% slower with it (its own async pipeline prefers small buffers), so `base` keeps MLX's
# default. Read by MLX at device init, hence before the import.
if len(sys.argv) > 1 and sys.argv[1] in ("bench", "check") and "MLX_MAX_OPS_PER_BUFFER" not in os.environ:
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "100"
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import generate as G  # noqa: E402
import verify as VF  # noqa: E402
from dflash import BLOCK, WINDOW, DFlash2  # noqa: E402

MODEL = HERE.parent / "models/volundr-int4-mlx"


def load(drafter_bits=0, lm_head_bits=0, nax=False):
    model, tok = G.load(MODEL)
    G.set_wired_limit()
    drafter = DFlash2.load(bits=drafter_bits, lm_head_bits=lm_head_bits)
    drafter.attach(model, model.args.vocab_valid)
    if nax:
        import nax_hook

        nax_hook.enable(model)
        nax_hook.enable(drafter)
    return model, drafter, tok


def spec_prefill(model, drafter, cache, dcache, ids, chunk=2048):
    """Chunked prefill that also fills the drafter's context with the last WINDOW positions' features."""
    n = len(ids)
    arr = mx.array(ids, dtype=mx.int32)[None]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        want = e > n - WINDOW
        logits, cap, _ = VF.forward(model, arr[:, s:e], cache, capture=drafter.capture_layers if want else ())
        if want:
            lo = max(s, n - WINDOW)
            drafter.materialize(dcache, cap[lo - s :], lo)
        mx.eval(logits, [c.state for c in cache], dcache.state)
        mx.clear_cache()
    return logits[-1]


def spec_generate(model, drafter, ids, max_new, T=BLOCK, eos=(), chunk=2048, lookup=None):
    """Greedy tokens (up to max_new, stopping after an eos) and a record of the run (per-step accepted counts and
    draft source). lookup: a lookup.Lookup; when its copy proposal fills all T - 1 draft rows the drafter pass is
    skipped, a shorter one overwrites the first drafter slots (the engine's apply_hint)."""
    assert 2 <= T <= BLOCK
    vv = model.args.vocab_valid
    cache, dcache = model.make_cache(), drafter.make_cache()
    t0 = time.perf_counter()
    last = spec_prefill(model, drafter, cache, dcache, ids, chunk)
    bonus = mx.argmax(last[:vv]).astype(mx.int32)
    out = [bonus.item()]
    t1 = time.perf_counter()
    P, acc, src = len(ids), [], []
    t_wait = 0.0  # time blocked on the GPU result; the rest of decode_s is CPU (graph building, host bookkeeping)
    seg, te = [0.0, 0.0, 0.0], time.perf_counter()  # CPU: bookkeeping + commit + draft build, async_eval, verify build
    if lookup is not None:
        lookup.reset(list(ids) + out)
    while len(out) < max_new and out[-1] not in eos:
        hint = lookup.propose() if lookup is not None else []
        if len(hint) >= T - 1:
            drafts = mx.array(hint[: T - 1], dtype=mx.int32)
        else:
            drafts = drafter.propose(dcache, bonus, P)
            if hint:
                drafts = mx.concatenate([mx.array(hint, dtype=mx.int32), drafts[len(hint) :]])
        inp = mx.concatenate([bonus.reshape(1), drafts[: T - 1]])[None]
        ta = time.perf_counter()
        mx.async_eval(inp)  # the GPU drafts (and commits the last step) while the CPU builds the verify graph
        tb = time.perf_counter()
        logits, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
        tok = mx.argmax(logits[:, :vv], axis=-1).astype(mx.int32)
        nacc = mx.cumprod((inp[0, 1:] == tok[:-1]).astype(mx.int32)).sum()
        tw = time.perf_counter()
        seg[0] += ta - te
        seg[1] += tb - ta
        seg[2] += tw - tb
        mx.eval(tok, nacc, inp, [c.state for c in cache], dcache.state)
        te = time.perf_counter()
        t_wait += te - tw
        n = nacc.item() + 1
        emitted = inp[0, 1:n].tolist() + [tok[n - 1].item()]
        acc.append(n)
        src.append(len(hint))
        stop = next((i for i, t in enumerate(emitted) if t in eos), None)
        if stop is not None:  # an eos inside the accepted run ends the reply there
            out += emitted[: stop + 1]
            break
        out += emitted
        if lookup is not None:
            lookup.record(emitted)
        VF.commit(cache, rec, n)
        drafter.materialize(dcache, cap[:n], P)
        bonus = tok[n - 1]
        P += n
    t2 = time.perf_counter()
    out = out[:max_new]
    return out, {"prefill_s": t1 - t0, "decode_s": t2 - t1, "wait_s": t_wait, "seg_s": seg, "steps": len(acc),
                 "acc": acc, "src": src}


def greedy_generate(model, ids, max_new, eos=(), chunk=2048, margins=False):
    """Plain greedy decoding (generate.py's pipelined loop); margins=True records top1 - top2 logits per token
    instead (unpipelined, for the exactness check)."""
    vv = model.args.vocab_valid
    cache = model.make_cache()
    t0 = time.perf_counter()
    last = G.prefill(model, cache, ids, chunk)
    mx.eval(last)
    t1 = time.perf_counter()
    if not margins:
        out = list(G.greedy_tokens(model, cache, last, max_new, set(eos), vv))
        return out, {"prefill_s": t1 - t0, "decode_s": time.perf_counter() - t1}
    out, mg, logits = [], [], last
    for _ in range(max_new):
        lv = logits[:vv].astype(mx.float32)
        top2 = mx.topk(lv, 2)
        y = mx.argmax(lv).astype(mx.int32)
        mx.eval(y, top2)
        out.append(y.item())
        mg.append(abs(top2[1].item() - top2[0].item()))
        if out[-1] in eos:
            break
        logits = model(y.reshape(1, 1), cache=cache, last_only=True)[0, -1]
        mx.eval(logits, [c.state for c in cache])
    return out, {"prefill_s": t1 - t0, "margins": mg}


def suite(path, split, per_cat=0, cats=None):
    s = [e for e in json.loads(Path(path).read_text()) if split in (e["split"], "all")]
    if cats:
        s = [e for e in s if e["category"] in cats]
    if per_cat:
        seen = {}
        s = [e for e in s if seen.setdefault(e["category"], []).append(e) or len(seen[e["category"]]) <= per_cat]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "bench", "base"])
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--per-cat", type=int, default=0)
    ap.add_argument("--cats", default="")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--T", type=int, default=BLOCK)
    ap.add_argument("--drafter-bits", type=int, default=0)
    ap.add_argument("--head-bits", type=int, default=0, help="quantized copy of the lm_head for drafting")
    ap.add_argument("--nax", action="store_true", help="small-M verify matmuls on the NAX kernel (nax_hook.py)")
    ap.add_argument("--lookup", action="store_true", help="copy-from-context drafts (lookup.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, drafter, tok = load(args.drafter_bits, args.head_bits, args.nax)
    eos = set(tok.eos_token_ids)
    prompts = suite(args.suite, args.split, args.per_cat, [c for c in args.cats.split(",") if c])
    # warm up the kernels (Metal JIT) on a short prompt
    w = tok.encode("Warm up the kernels, please.")
    spec_generate(model, drafter, w, 24, args.T)
    greedy_generate(model, w, 8)
    look = None
    if args.lookup:
        from lookup import Lookup

        look = Lookup()
    fout = open(args.out, "a") if args.out else None
    for e in prompts:
        rec = {"name": e["name"], "category": e["category"], "split": e["split"], "prompt_tokens": len(e["ids"]),
               "T": args.T, "drafter_bits": args.drafter_bits, "head_bits": args.head_bits, "nax": args.nax,
               "lookup": args.lookup}
        if args.cmd == "check":
            ref, r0 = greedy_generate(model, e["ids"], args.new, eos, margins=True)
            got, r1 = spec_generate(model, drafter, e["ids"], args.new, args.T, eos, lookup=look)
            div = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
            if div is None and len(ref) != len(got):
                div = min(len(ref), len(got))
            rec.update(match=div is None, first_diverge=div, n_ref=len(ref), n_spec=len(got),
                       margin_at_div=None if div is None or div >= len(r0["margins"]) else r0["margins"][div],
                       min_margin=min(r0["margins"]), mean_acc=sum(r1["acc"]) / max(1, len(r1["acc"])))
        elif args.cmd == "bench":
            got, r1 = spec_generate(model, drafter, e["ids"], args.new, args.T, eos, lookup=look)
            rec.update(generated=len(got), prefill_s=r1["prefill_s"], decode_s=r1["decode_s"], wait_s=r1["wait_s"],
                       seg_s=r1["seg_s"], steps=r1["steps"],
                       acc=r1["acc"], src=r1["src"], mean_acc=sum(r1["acc"]) / max(1, len(r1["acc"])), tok_s=(len(got) - 1) / r1["decode_s"] if r1["decode_s"] > 0 else None,
                       tokens=got)
        else:
            got, r0 = greedy_generate(model, e["ids"], args.new, eos)
            rec.update(generated=len(got), prefill_s=r0["prefill_s"], decode_s=r0["decode_s"],
                       tok_s=(len(got) - 1) / r0["decode_s"] if r0["decode_s"] > 0 else None, tokens=got)
        mx.clear_cache()
        short = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("acc", "src", "tokens")}
        print(json.dumps(short), flush=True)
        if fout:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
