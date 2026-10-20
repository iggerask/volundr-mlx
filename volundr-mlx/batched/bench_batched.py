"""Batched plain greedy decode: B independent sequences decoded together, one [B, 1] forward per step.

Measurement harness only (NOTES_BATCHED.md), not a server. Every prompt has the same token count, so every row of
the batch sits at the same position and the caches' scalar offsets (BCSA ring, dense KVCache) stay valid:
- prompts are repo files (own text, not the spec suite) behind a chat template, the file text cut so that the
  templated ids are exactly --tokens long;
- each prompt is prefilled alone at B = 1 (generate.prefill), and the per-prompt caches are stacked along the batch
  axis into one snapshot (the per-prompt arrays are freed layer by layer as they are stacked);
- the B = 1 reference greedy run of each prompt and every batched run start from row slices of that snapshot (views;
  the first decode step copies them), so the snapshot itself is never changed. With --distinct D < B, row i decodes
  prompt i % D (the snapshot then holds D prompts; rows past D are tiled copies), to bound the harness's memory;
- the step is volundr.Model's own decode path with B rows; EOS is ignored (fixed-length runs).

Each run records the tokens and the top-1 / top-2 logit margin of every row and step; per-step wall times after
--warm untimed steps; mx.get_peak_memory() from a reset at the start of the run.

usage (from volundr-mlx, under ./with_gpu_lock.sh):
  python batched/bench_batched.py --tokens 200 --batches 1,2,4,8,16 --steps 128 --out spec/runs/batched-bench_short.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spec"))
import generate as G  # noqa: E402
import nax_hook  # noqa: E402
import volundr as V  # noqa: E402

MODEL = ROOT / "models/volundr-int4-mlx"
# own text: tracked repo files (code and notes); the spec suite is not used
SOURCES = ["volundr-mlx/volundr.py", "volundr-mlx/kernels.py", "volundr-mlx/decode_fused.py", "volundr-mlx/README.md",
           "volundr-mlx/NOTES_COMBINED.md", "volundr-mlx/NOTES_SPEC_ACCEPT.md", "volundr-mlx/NOTES_SESSION.md",
           "volundr-mlx/spec/verify.py", "volundr-mlx/spec/specdec.py", "volundr-mlx/spec/dflash.py",
           "volundr-mlx/session.py", "volundr-mlx/worker.py", "volundr-mlx/convert.py", "volundr-mlx/PARITY.md",
           "volundr-mlx/NOTES_decode.md", "volundr-mlx/bcsa_verify.py", "volundr-mlx/spec/NOTES.md",
           "volundr-mlx/NOTES_BCSA_VERIFY.md", "agens-mlx-bench/smallm/NOTES.md", "README.md"]
ASKS = ["Explain what this file does and point out anything that looks wrong.",
        "Summarise this file for a new contributor, then list three questions you would ask its author.",
        "Write a short review of this code: structure, naming, and the riskiest part.",
        "What are the main ideas in this text? Answer in a numbered list."]


def make_prompts(tok, n_tokens, count):
    """count distinct prompts of exactly n_tokens ids: chat template around (ask + a repo file, cut to fit)."""
    marker = "⁣MARK⁣"
    tmpl = tok.apply_chat_template([{"role": "user", "content": marker}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    pre, post = tmpl.split(marker)
    pre_ids, post_ids = tok.encode(pre, add_special_tokens=False), tok.encode(post, add_special_tokens=False)
    m = n_tokens - len(pre_ids) - len(post_ids)
    out = []
    for i in range(count):
        files = SOURCES[i % len(SOURCES):] + SOURCES[: i % len(SOURCES)]
        text = ASKS[i % len(ASKS)] + "\n\n"
        for f in files:  # long prompts run on into the next files
            text += f"File {f}:\n" + (REPO / f).read_text()[(i // len(SOURCES)) * 2000:] + "\n\n"
            if len(text) > 6 * m:
                break
        body = tok.encode(text, add_special_tokens=False)[:m]
        assert len(body) == m, (i, len(body), m)
        out.append(pre_ids + body + post_ids)
    return out


# ----------------------------------------------------------------------------- caches
def stack_caches(model, caches):
    """Per-prompt B = 1 caches -> one cache with the prompts along the batch axis. Frees the inputs as it goes."""
    out = model.make_cache()
    for j, c in enumerate(out):
        src = [cs[j] for cs in caches]
        if isinstance(c, ArraysCache):
            for i in range(2):
                c[i] = mx.concatenate([s[i] for s in src], axis=0)
            arrs = c.cache
        elif isinstance(c, V.BCSACache):
            assert len({(s.offset, s.nb) for s in src}) == 1, "rows at different positions"
            c.offset, c.nb = src[0].offset, src[0].nb
            for a in ("k", "v", "pk", "pv", "bk", "ik_open"):
                setattr(c, a, None if getattr(src[0], a) is None else mx.concatenate([getattr(s, a) for s in src]))
            arrs = c.state
        elif isinstance(c, KVCache):
            assert len({s.offset for s in src}) == 1
            c.offset = src[0].offset
            c.keys = mx.concatenate([s.keys for s in src])
            c.values = mx.concatenate([s.values for s in src])
            arrs = [c.keys, c.values]
        else:  # EngramState
            c.prev = mx.concatenate([s.prev for s in src])
            arrs = [c.prev]
        mx.eval(arrs)
        for cs in caches:
            cs[j] = None
        mx.clear_cache()
    return out


def slice_cache(model, snap, lo, hi, n):
    """A cache holding rows [lo, hi) of a snapshot of n rows, row i taken from snapshot row i % n (views when hi <= n,
    then decode's first writes copy them; otherwise tiled copies)."""
    def rows(x):
        if hi <= n:
            return x[lo:hi]
        return mx.concatenate([x] * (-(-hi // n)))[lo:hi]

    out = model.make_cache()
    for c, s in zip(out, snap):
        if isinstance(c, ArraysCache):
            c[0], c[1] = rows(s[0]), rows(s[1])
        elif isinstance(c, V.BCSACache):
            c.offset, c.nb = s.offset, s.nb
            for a in ("k", "v", "pk", "pv", "bk", "ik_open"):
                x = getattr(s, a)
                setattr(c, a, None if x is None else rows(x))
        elif isinstance(c, KVCache):
            c.offset, c.keys, c.values = s.offset, rows(s.keys), rows(s.values)
        else:
            c.prev = rows(s.prev)
    return out


def cache_bytes(cache):
    n = 0
    for c in cache:
        if isinstance(c, ArraysCache):
            n += sum(a.nbytes for a in c.cache if a is not None)
        elif isinstance(c, KVCache):
            n += c.keys.nbytes + c.values.nbytes
        elif isinstance(c, V.BCSACache):
            n += c.nbytes
        else:
            n += c.prev.nbytes
    return n


# ----------------------------------------------------------------------------- decode
def pick(logits, vv):
    """logits [B, V] -> greedy token [B] int32, top-1 minus top-2 logit [B] fp32."""
    lg = logits[:, :vv]
    t = mx.argmax(lg, axis=-1).astype(mx.int32)
    top = mx.take_along_axis(lg, t[:, None], axis=-1)[:, 0].astype(mx.float32)
    second = mx.max(mx.where(mx.arange(vv)[None] == t[:, None], -mx.inf, lg), axis=-1).astype(mx.float32)
    return t, top - second


def decode(model, cache, first, steps, vv):
    """Greedy decode of B rows from first = (tokens [B], margins [B]): steps forwards, the next step's graph built
    while the current one runs (generate.greedy_tokens' pipelining). Returns tokens [B][steps + 1], margins, the wall
    time at which each step's result was read."""
    B = first[0].shape[0]

    def step(y):
        return pick(model(y.reshape(B, 1), cache=cache, last_only=True)[:, -1], vv)

    cur = first
    mx.async_eval(*cur)
    toks, margins, times = [], [], []
    for i in range(steps + 1):
        nxt = None
        if i < steps:
            nxt = step(cur[0])
            mx.async_eval(*nxt, [c.state for c in cache])
        toks.append(cur[0].tolist())
        margins.append(cur[1].tolist())
        times.append(time.perf_counter())
        cur = nxt
    return [list(r) for r in zip(*toks)], [list(r) for r in zip(*margins)], times


def compare(ref_toks, ref_marg, toks, marg):
    """First index where toks leaves ref_toks (None = identical), with both runs' margins there."""
    for i, (a, b) in enumerate(zip(ref_toks, toks)):
        if a != b:
            return {"diverge_at": i, "ref_margin": ref_marg[i], "margin": marg[i]}
    return {"diverge_at": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200, help="prompt length (every prompt)")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--nax", default="0,1", help="NAX small-M matmuls off / on (interleaved per B)")
    ap.add_argument("--steps", type=int, default=128, help="timed decode steps")
    ap.add_argument("--warm", type=int, default=4, help="untimed steps first (the snapshot copy, Metal JIT)")
    ap.add_argument("--refs", type=int, default=1, help="run the B = 1 reference of every prompt")
    ap.add_argument("--reps", type=int, default=1, help="repeats of the whole B x NAX grid")
    ap.add_argument("--selects", default="kernel", help="volundr.BATCH_SELECT values to run (B > 1 far selection)")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--distinct", type=int, default=0,
                    help="distinct prompts (default max B); batch row i decodes prompt i %% distinct")
    ap.add_argument("--max-gb", type=float, default=46.0, help="skip a run whose estimated peak exceeds this")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    naxes = [int(n) for n in args.nax.split(",")]
    nmax = max(batches)
    outf = open(args.out, "a")

    def log(rec):
        print(json.dumps({k: v for k, v in rec.items() if k not in ("tokens", "margins", "ref")}), flush=True)
        outf.write(json.dumps(rec) + "\n")
        outf.flush()

    model, tok = G.load(MODEL)
    G.set_wired_limit()
    vv = model.args.vocab_valid
    weights_gb = mx.get_active_memory() / 1e9
    nd = args.distinct or nmax
    prompts = make_prompts(tok, args.tokens, nd)

    # warm the Metal kernels on a short prompt (B = 1 and B = 2 decode), untimed
    c = model.make_cache()
    lg = G.prefill(model, c, prompts[0][:64], args.chunk)
    decode(model, c, pick(lg[None], vv), 4, vv)
    del c
    mx.clear_cache()

    t0 = time.perf_counter()
    caches, firsts = [], []
    for ids in prompts:
        c = model.make_cache()
        lg = G.prefill(model, c, ids, args.chunk)
        t, m = pick(lg[None], vv)
        mx.eval(t, m)
        caches.append(c)
        firsts.append((t, m))
    t_prefill = time.perf_counter() - t0
    per_stream = cache_bytes(caches[0])
    snap = stack_caches(model, caches)
    del caches
    mx.clear_cache()
    first_t = mx.concatenate([f[0] for f in firsts])
    first_m = mx.concatenate([f[1] for f in firsts])
    mx.eval(first_t, first_m)
    snap_gb = cache_bytes(snap) / 1e9
    log({"kind": "setup", "prompt_tokens": args.tokens, "prompts": nd, "prefill_s": round(t_prefill, 1),
         "weights_gb": round(weights_gb, 2), "cache_gb_per_stream": round(per_stream / 1e9, 3),
         "snapshot_gb": round(snap_gb, 2), "active_gb": round(mx.get_active_memory() / 1e9, 2),
         "lmhead8": V.lm_head_q_group(), "batch_select": V.BATCH_SELECT})

    def run(B, nax, lo=0):
        if nax:
            nax_hook.enable(model)
        else:
            nax_hook.disable(model)
        mx.clear_cache()
        mx.reset_peak_memory()
        cache = slice_cache(model, snap, lo, lo + B, nd)
        idx = mx.array([i % nd for i in range(lo, lo + B)])
        toks, marg, times = decode(model, cache, (first_t[idx], first_m[idx]), args.warm + args.steps, vv)
        dt = (times[-1] - times[args.warm]) / args.steps
        peak = mx.get_peak_memory() / 1e9
        del cache
        nax_hook.disable(model)
        mx.clear_cache()
        return toks, marg, dt, peak

    refs = {}
    if args.refs:
        for i in range(nd):
            toks, marg, dt, peak = run(1, 0, i)
            refs[i] = (toks[0], marg[0])
            log({"kind": "ref", "stream": i, "ms_step": round(1e3 * dt, 2), "peak_gb": round(peak, 2),
                 "min_margin": min(marg[0]), "tokens": toks[0], "margins": marg[0]})

    grid = [(rep, B, nax, sel) for rep in range(args.reps) for B in batches for nax in naxes
            for sel in args.selects.split(",")]
    for rep, B, nax, sel in grid:
        V.BATCH_SELECT = sel
        if B == 1 and (nax and 0 in naxes or sel != args.selects.split(",")[0]):
            continue  # NAX only takes 2..16 rows and B = 1 always selects with the kernels
        est = weights_gb + snap_gb + B * per_stream / 1e9 + 1.0
        if est > args.max_gb:
            log({"kind": "skip", "B": B, "nax": nax, "est_gb": round(est, 1)})
            continue
        toks, marg, dt, peak = run(B, nax)
        rec = {"kind": "batch", "rep": rep, "B": B, "nax": nax, "select": sel, "prompt_tokens": args.tokens,
               "distinct": min(B, nd),
               "steps": args.steps, "ms_step": round(1e3 * dt, 2), "agg_tok_s": round(B / dt, 2),
               "stream_tok_s": round(1 / dt, 2), "peak_gb": round(peak, 2),
               "peak_less_snapshot_gb": round(peak - snap_gb, 2), "tokens": toks, "margins": marg}
        if refs:
            cmp = [compare(*refs[i % nd], toks[i], marg[i]) for i in range(B)]
            rec["identical"] = sum(c["diverge_at"] is None for c in cmp)
            rec["diverged"] = [dict(stream=i, **c) for i, c in enumerate(cmp) if c["diverge_at"] is not None]
        log(rec)
    outf.close()


if __name__ == "__main__":
    main()
