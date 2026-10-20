"""Upside of more accepted tokens per step: per step of the real spec loop (frozen config, lookup on), what alternative
draft chains would have accepted, measured against the run's own greedy output.

The loop is spec_generate's (same ops, same trajectory: asserted against spec_generate's tokens). In addition, every
step also runs the drafter when the lookup filled all 7 drafts (production skips it; its block K/V written into the
drafter's ring at P..P+7 is overwritten before anything reads it, see dflash.hidden), and records:
- the DFlash chain (7), its top-16 candidates per slot with the walk's scores, and for each slot j the chain with the
  walk's second choice at j and the greedy walk re-run after it ("alt j"), with the score margin top1 - top2 at j;
- the lookup's best copy regardless of whether lookup.propose used it, extended to 15 tokens, and its match length;
- the chain actually verified (src = number of lookup tokens in it) and the accepted count n.
Acceptance of a chain = its longest prefix equal to the tokens the run emitted after the step's bonus token (the
target's greedy continuation; capped where the output ends, flagged). The actual chain's acceptance is n - 1 by construction.

usage (from volundr-mlx/spec, under ../with_gpu_lock.sh, VOLUNDR_LMHEAD8=64):
  python accept_log.py --split tune --out runs/spec-accept_acclog.jsonl [--cats ...] [--resume]
"""
import os
import sys

if "MLX_MAX_OPS_PER_BUFFER" not in os.environ:  # as specdec.py bench
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "100"
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash as DFL  # noqa: E402
import specdec as SD  # noqa: E402
import verify as VF  # noqa: E402
import lookup as LK  # noqa: E402
from lookup import Lookup  # noqa: E402

F32 = mx.float32
LONG = 15


def drafter_details(drafter, dcache, anchor, P):
    """drafter.propose's ops (identical drafts), plus the candidates and the per-slot alternative walks."""
    hs = drafter.hidden(dcache, anchor, P)[1:]
    logits = drafter.head(hs)[:, : drafter.vocab_valid]
    idx = mx.argpartition(-logits, DFL.TOPK - 1, axis=-1)[:, : DFL.TOPK]
    val = mx.take_along_axis(logits, idx, axis=-1)
    order = mx.argsort(-val.astype(F32), axis=-1)
    cid = mx.take_along_axis(idx, order, axis=-1).astype(mx.int32)
    uval = mx.take_along_axis(val, order, axis=-1).astype(F32)
    sel = drafter.candidate_selector
    hp = sel.hidden_projection(hs).astype(F32)
    A, B = sel.predecessor_codebook, sel.successor_codebook
    a0 = anchor.reshape(()).astype(mx.int32)
    drafts = DFL._walk(uval, cid, hp, A, B, a0)
    # the walk again, uncompiled, keeping every slot's scores; then for each slot j the walk with the second choice at j
    def scores(e, pred):
        a = mx.take(A, pred, axis=0).astype(F32) * hp[e]
        return uval[e] + mx.take(B, cid[e], axis=0).astype(F32) @ a

    alts, margins = [], []
    for j in range(DFL.BLOCK - 1):
        pred, chain = a0, []
        for e in range(DFL.BLOCK - 1):
            s = scores(e, pred)
            if e == j:
                o = mx.argsort(-s)
                margins.append(s[o[0]] - s[o[1]])
                pred = mx.take(cid[e], o[1])
            elif e < j:
                pred = drafts[e]
            else:
                pred = mx.take(cid[e], mx.argmax(s))
            chain.append(pred)
        alts.append(mx.stack(chain))
    return drafts, cid, mx.stack(alts), mx.stack(margins)


def lookup_long(lookup):
    """lookup.find()'s best copy (same candidates and match rule), extended to LONG tokens, and its match length;
    ([], 0) if none. Pure apart from indexing the history up to n - 1, as find() does."""
    h, n = lookup.hist, len(lookup.hist)
    if n < 3:
        return [], 0
    lookup._index_upto(n - 1)
    cands = lookup.idx.get((h[n - 2], h[n - 1]))
    if not cands:
        return [], 0
    best = None
    for p in reversed(cands[-LK.CANDIDATES:]):
        if p + 1 >= n:
            continue
        m = 2
        while m < 64 and m <= p and h[p - m] == h[n - 1 - m]:
            m += 1
        if best is None or m > best[0]:
            best = (m, p)
    if best is None:
        return [], 0
    m, p = best
    return h[p + 1 : min(p + 1 + LONG, n)], m


def run(model, drafter, ids, max_new, eos, lookup, T=8):
    vv = model.args.vocab_valid
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = SD.spec_prefill(model, drafter, cache, dcache, ids)
    bonus = mx.argmax(last[:vv]).astype(mx.int32)
    out = [bonus.item()]
    P = len(ids)
    lookup.reset(list(ids) + out)
    steps = []
    while len(out) < max_new and out[-1] not in eos:
        lk, lk_m = lookup_long(lookup)
        hint = lookup.propose()
        dd, cid, alts, margins = drafter_details(drafter, dcache, bonus, P)
        if len(hint) >= T - 1:
            drafts = mx.array(hint[: T - 1], dtype=mx.int32)
        else:
            drafts = dd
            if hint:
                drafts = mx.concatenate([mx.array(hint, dtype=mx.int32), drafts[len(hint):]])
        inp = mx.concatenate([bonus.reshape(1), drafts[: T - 1]])[None]
        mx.async_eval(inp, dd, cid, alts, margins)
        logits, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
        tok = mx.argmax(logits[:, :vv], axis=-1).astype(mx.int32)
        nacc = mx.cumprod((inp[0, 1:] == tok[:-1]).astype(mx.int32)).sum()
        mx.eval(tok, nacc, inp, [c.state for c in cache], dcache.state)
        n = nacc.item() + 1
        emitted = inp[0, 1:n].tolist() + [tok[n - 1].item()]
        steps.append({"i": len(out) - 1, "P": P, "n": n, "src": len(hint), "used": inp[0, 1:].tolist(),
                      "dflash": dd.tolist(), "cid": cid.tolist(), "alts": alts.tolist(),
                      "margins": [round(x, 4) for x in margins.tolist()], "lookup": lk, "lookup_m": lk_m})
        stop = next((i for i, t in enumerate(emitted) if t in eos), None)
        if stop is not None:
            out += emitted[: stop + 1]
            break
        out += emitted
        lookup.record(emitted)
        VF.commit(cache, rec, n)
        drafter.materialize(dcache, cap[:n], P)
        bonus = tok[n - 1]
        P += n
    out = out[:max_new]
    return out, steps


def acc_len(chain, tgt):
    k = 0
    while k < len(chain) and k < len(tgt) and chain[k] == tgt[k]:
        k += 1
    return k


def annotate(out, steps):
    for s in steps:
        tgt = out[s["i"] + 1 : s["i"] + 1 + LONG]
        s["tgt_len"] = len(tgt)
        s["acc_used"] = acc_len(s["used"], tgt)
        s["acc_dflash"] = acc_len(s["dflash"], tgt)
        s["acc_lookup7"] = acc_len(s["lookup"][:7], tgt) if s["lookup"] else None
        s["acc_lookup15"] = acc_len(s["lookup"], tgt) if s["lookup"] else None
        s["acc_alts"] = [acc_len(a, tgt) for a in s["alts"]]
        # the target token at the first slot the DFlash chain got wrong: its rank among that slot's 16 candidates
        # (-1: not among them; None: no miss inside the chain / the output ended)
        j = s["acc_dflash"]
        s["dflash_miss_rank"] = None
        if j < 7 and j < len(tgt):
            s["dflash_miss_rank"] = s["cid"][j].index(tgt[j]) if tgt[j] in s["cid"][j] else -1
        assert s["acc_used"] == min(s["n"] - 1, len(tgt)), (s["acc_used"], s["n"], len(tgt))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--cats", default="")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", action="store_true", help="also run spec_generate per prompt and assert equal tokens")
    args = ap.parse_args()
    model, drafter, tok = SD.load(drafter_bits=4, lm_head_bits=4, nax=True)
    eos = set(tok.eos_token_ids)
    w = tok.encode("Warm up the kernels, please.")
    SD.spec_generate(model, drafter, w, 24, 8)
    look = Lookup()
    look_ref = Lookup()
    done = {}
    if args.resume and Path(args.out).exists():
        done = {r["name"]: r for r in map(json.loads, filter(str.strip, Path(args.out).read_text().splitlines()))}
    fout = open(args.out, "a")
    for e in SD.suite(args.suite, args.split, 0, [c for c in args.cats.split(",") if c]):
        if e["name"] in done:
            r = done[e["name"]]
            SD.lookup_replay(look, e["ids"], {"name": r["name"], "tokens": r["tokens"],
                                              "acc": [s["n"] for s in r["steps"]],
                                              "src": [s["src"] for s in r["steps"]]}, eos)
            continue
        t0 = time.perf_counter()
        out, steps = run(model, drafter, e["ids"], args.new, eos, look)
        rec = {"name": e["name"], "category": e["category"], "prompt_tokens": len(e["ids"]), "tokens": out,
               "target_head_q8": model.lm_head.group_size if "scales" in model.lm_head else 0,
               "steps": annotate(out, steps), "s": round(time.perf_counter() - t0, 1)}
        if args.check:
            ref, _ = SD.spec_generate(model, drafter, e["ids"], args.new, 8, eos, lookup=look_ref)
            rec["matches_spec_generate"] = ref == out
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        st = rec["steps"]
        print(json.dumps({"name": e["name"], "steps": len(st), "tok_step": round(sum(s["n"] for s in st) / len(st), 3),
                          "dflash": round(sum(s["acc_dflash"] for s in st) / len(st) + 1, 3),
                          "check": rec.get("matches_spec_generate"), "s": rec["s"]}), flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()
