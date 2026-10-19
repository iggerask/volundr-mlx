"""Time the parts of a speculative step on a real context: the drafter pass, the target verify forward at several
widths T (all T committed, so the context advances), and a plain decode step.

usage: python time_parts.py [--ctx short|long] [--drafter-bits 0|4|8] [--head-bits 0|4|8] [--nax] [--reps 6]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402


def timed(f, reps):
    f()  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return 1e3 * ts[len(ts) // 2]  # median ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="short")
    ap.add_argument("--drafter-bits", type=int, default=0)
    ap.add_argument("--head-bits", type=int, default=0)
    ap.add_argument("--nax", action="store_true")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--Ts", default="2,4,6,8,12,16")
    args = ap.parse_args()
    model, drafter, tok = S.load(args.drafter_bits, args.head_bits, args.nax)
    suite = S.suite(HERE / "suite.json", "tune")
    e = next(x for x in suite if (x["category"] == "long") == (args.ctx == "long"))
    ids = e["ids"]
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = S.spec_prefill(model, drafter, cache, dcache, ids)
    st = {"P": len(ids), "bonus": mx.argmax(last[: model.args.vocab_valid]).astype(mx.int32)}
    rec = {"ctx": args.ctx, "prompt_tokens": len(ids), "drafter_bits": args.drafter_bits,
           "head_bits": args.head_bits, "nax": args.nax}

    def draft():
        mx.eval(drafter.propose(dcache, st["bonus"], st["P"]))

    rec["draft_ms"] = timed(draft, args.reps)
    gen = [int(t) for t in tok.encode(" the quick brown fox jumps over the lazy dog and runs far away" * 4)]

    def verify(T):
        def f():
            inp = mx.array(gen[:T], dtype=mx.int32)[None]
            lg, cap, r = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True)
            top = mx.argmax(lg, axis=-1)
            mx.eval(top, cap, [c.state for c in cache])
            VF.commit(cache, r, T)
            drafter.materialize(dcache, cap, st["P"])
            mx.eval([c.state for c in cache], dcache.state)
            st["P"] += T
        return f

    for T in [int(t) for t in args.Ts.split(",")]:
        rec[f"verify{T}_ms"] = timed(verify(T), args.reps)

    def decode():
        y = mx.array([[gen[0]]], dtype=mx.int32)
        lg = model(y, cache=cache, last_only=True)
        mx.eval(mx.argmax(lg), [c.state for c in cache])
        st["P"] += 1

    rec["decode_ms"] = timed(decode, args.reps)
    rec["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)


if __name__ == "__main__":
    main()
