"""Merge check for combined-test: decode-quick-wins' kernels are bit-exact inside both BCSA verify paths.

Per prompt and per config (../ab_modes.py), from a fresh cache: prefill, then verify forwards over fixed blocks of T
tokens (a recorded greedy continuation) with a seeded random commit n in 1..T, so the KDA replay, BCSA partial append,
dense-KV trim and Engram rollbacks all run. Every verify row's logits are kept. Then:
- main+v8q == fused (= exact) and main == fused+vprefill must be bit-identical (the fusion, views, wide norms and
  cached norm weights change no verify value under either path);
- main vs fused (8q against prefill verify) is reported as KL, which should sit at the parity noise floor.

usage (from volundr-mlx/spec, under ../with_gpu_lock.sh):
  python check_combined.py [--names a,b] [--steps 24] [--ref runs/decode-quick-wins_tune_spec_main.jsonl]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import specdec as S  # noqa: E402
import verify as VF  # noqa: E402
from ab_modes import Modes, describe  # noqa: E402

CONFIGS = ["main", "main+v8q", "fused+vprefill", "fused"]
NAMES = "coding/00-12623519-think,agent/00-marshmallow-code__apispe-think,long/03-long3"


def run(model, ids, toks, steps, T, seed):
    vv = model.args.vocab_valid
    rng = random.Random(seed)
    cache = model.make_cache()
    S.G.prefill(model, cache, ids)
    out, i = [], 0
    for _ in range(steps):
        if i + T > len(toks):
            break
        inp = mx.array(toks[i : i + T], dtype=mx.int32)[None]
        lg, _, rec = VF.forward(model, inp, cache, verify=True)
        lg = lg[:, :vv]
        mx.eval(lg, [c.state for c in cache])
        out.append(lg)
        n = rng.randint(1, T)
        VF.commit(cache, rec, n)
        i += n
    mx.eval([c.state for c in cache])
    del cache
    mx.clear_cache()
    return out


def kl(p, q):
    lp = p.astype(mx.float32) - mx.logsumexp(p.astype(mx.float32), axis=-1, keepdims=True)
    lq = q.astype(mx.float32) - mx.logsumexp(q.astype(mx.float32), axis=-1, keepdims=True)
    return (mx.exp(lp) * (lp - lq)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--names", default=NAMES)
    ap.add_argument("--ref", default=str(HERE / "runs/decode-quick-wins_tune_spec_main.jsonl"),
                    help="recorded outputs to teacher-force (any run with 'tokens' per prompt)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--nax", action="store_true")
    args = ap.parse_args()
    model, _, _ = S.load(nax=args.nax)
    modes = Modes(model, CONFIGS)
    suite = {e["name"]: e for e in json.loads(Path(args.suite).read_text())}
    ref = {r["name"]: r["tokens"] for r in map(json.loads, open(args.ref))}
    ok = True
    for name in args.names.split(","):
        e = suite[name]
        logits = {}
        for c in CONFIGS:
            modes.set(c)
            logits[c] = run(model, e["ids"], ref[name], args.steps, args.T, seed=11)
            print(json.dumps({"name": name, "config": c, **describe(), "verify_calls": len(logits[c])}), flush=True)
        res = {"name": name, "prompt_tokens": len(e["ids"]), "rows": sum(x.shape[0] for x in logits["main"])}
        for a, b in (("main+v8q", "fused"), ("main", "fused+vprefill")):
            same = all(mx.array_equal(x, y).item() for x, y in zip(logits[a], logits[b]))
            res[f"{a} == {b}"] = same
            ok &= same
        ks = mx.concatenate([kl(x, y) for x, y in zip(logits["main"], logits["fused"])])
        top = sum((mx.argmax(x, -1) == mx.argmax(y, -1)).sum().item() for x, y in zip(logits["main"], logits["fused"]))
        res.update(kl_prefill_vs_8q_mean=ks.mean().item(), kl_prefill_vs_8q_max=ks.max().item(), argmax_agree=top)
        print(json.dumps(res), flush=True)
    modes.reset()
    print("ALL BIT-IDENTICAL" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
