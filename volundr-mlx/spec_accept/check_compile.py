"""Bit-identity of the compiled KDA verify layers (spec/verify.py forward(..., compiled=True)) against the eager verify,
and the CPU time of building each.

Two caches prefilled identically; per step the same block (a recorded greedy continuation) is verified on both, eager
on one and compiled on the other, and a seeded random n in 1..T committed on both. Every step compares the logits,
the captured drafter features and, after the commit, every cache array, bit for bit (mx.array_equal).

usage (from volundr-mlx, under ./with_gpu_lock.sh): python spec_accept/check_compile.py [--names a,b] [--steps 24]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spec"))
sys.path.insert(0, str(ROOT))
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402

NAMES = "coding/00-12623519-think,agent/00-marshmallow-code__apispe-think,long/03-long3"


def flat(x):
    if isinstance(x, mx.array):
        return [x]
    if isinstance(x, (list, tuple)):
        return [a for y in x for a in flat(y)]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=NAMES)
    ap.add_argument("--ref", default=str(ROOT / "spec/runs/combined-test_tune_spec_exact+q8g64.jsonl"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--Ts", default="8,16")
    args = ap.parse_args()
    model, drafter, _ = S.load(4, 4, True)
    suite = {e["name"]: e for e in json.loads((ROOT / "spec/suite.json").read_text())}
    ref = {r["name"]: r["tokens"] for r in map(json.loads, open(args.ref))}
    ok = True
    for T in [int(t) for t in args.Ts.split(",")]:
        for name in args.names.split(","):
            ids, toks = suite[name]["ids"], ref[name]
            rng = random.Random(11)
            ca, cb = model.make_cache(), model.make_cache()
            da, db = drafter.make_cache(), drafter.make_cache()
            S.spec_prefill(model, drafter, ca, da, ids)
            S.spec_prefill(model, drafter, cb, db, ids)
            i, same, tb = 0, True, {"eager": [], "compiled": []}
            for step in range(args.steps):
                if i + T > len(toks):
                    break
                inp = mx.array([toks[i : i + T]], dtype=mx.int32)
                outs = {}
                for tag, c in (("eager", ca), ("compiled", cb)):
                    t0 = time.perf_counter()
                    lg, cap, rec = VF.forward(model, inp, c, capture=drafter.capture_layers, verify=True,
                                              compiled=tag == "compiled")
                    t1 = time.perf_counter()
                    mx.eval(lg, cap, [x.state for x in c])
                    if step:  # the first compiled call traces
                        tb[tag].append(1e3 * (t1 - t0))
                    outs[tag] = (lg, cap, rec)
                (la, pa, ra), (lb, pb, rb) = outs["eager"], outs["compiled"]
                same &= mx.array_equal(la, lb).item() and mx.array_equal(pa, pb).item()
                n = rng.randint(1, T)
                VF.commit(ca, ra, n)
                VF.commit(cb, rb, n)
                sa, sb = flat([x.state for x in ca]), flat([x.state for x in cb])
                mx.eval(sa, sb)
                same &= len(sa) == len(sb) and all(x.shape == y.shape and mx.array_equal(x, y).item()
                                                   for x, y in zip(sa, sb))
                i += n
            for k in tb:
                tb[k].sort()
            print(json.dumps({"T": T, "name": name, "prompt_tokens": len(ids), "steps": step + 1, "bit_identical": same,
                              "build_ms_eager": round(tb["eager"][len(tb["eager"]) // 2], 2),
                              "build_ms_compiled": round(tb["compiled"][len(tb["compiled"]) // 2], 2)}), flush=True)
            ok &= same
            del ca, cb, da, db
            mx.clear_cache()
    print("ALL BIT-IDENTICAL" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
