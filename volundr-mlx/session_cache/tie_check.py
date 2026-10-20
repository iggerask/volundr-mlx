"""Where the e2e conversation's greedy replies with and without the cache differ: are those near-ties?

Input: the worker log of an e2e run with --log-tokens (session_cache/e2e.sh), which holds every request's exact
prompt ids and output tokens: the cached run (session agent-1) and the replay without the cache (agent-1-replay).
In-process, session.Engine (the worker's code path) replays the cached run's requests in order with a session cache,
so every turn resumes from the same snapshot as served; its tokens are checked against the logged ones. At each turn
whose two replies differ, the target's logits at the first differing token are read teacher-forced along the common
prefix (plain decode steps), from the resumed state and from a fresh prefill: the gap between the two candidate
tokens' logits says whether the flip is a tie (one or two bf16 steps at the logit's magnitude).

usage (from volundr-mlx, under ./with_gpu_lock.sh):
  python session_cache/tie_check.py session_cache/e2e/worker.jsonl [--session agent-1] [--out ...]
"""
import os
import sys

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spec"))
import session as SS  # noqa: E402
import specdec  # noqa: E402


def logits_at(model, cache, last, toks, d, vv):
    """Logits predicting toks[d] after feeding toks[:d] with plain decode steps."""
    lg = last
    for t in toks[:d]:
        lg, _ = SS.decode_step(model, cache, int(t))
        mx.eval(lg, [c.state for c in cache])
    return lg[:vv].astype(mx.float32)


def bf16_step(x):
    """Spacing of bf16 values at |x|."""
    e = np.floor(np.log2(max(abs(x), 1e-30)))
    return float(2.0 ** (e - 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--session", default="agent-1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = [json.loads(x) for x in Path(a.log).read_text().splitlines() if x.strip()]
    cached = [r for r in recs if r.get("session_id") == a.session and "prompt_ids" in r]
    plain = {r["prompt_tokens"]: r for r in recs if r.get("session_id") == a.session + "-replay" and "prompt_ids" in r}
    model, drafter, tok = specdec.load(4, 4, nax=True)
    specdec.warm_up(model, drafter, tok.encode("Warm up the kernels, please."))
    vv = model.args.vocab_valid
    eng = SS.Engine(model, drafter, SS.SessionCache(budget_gb=12))
    out = []
    for i, rc in enumerate(cached):
        ids = rc["prompt_ids"]
        rn = plain.get(rc["prompt_tokens"])
        assert rn is not None and rn["prompt_ids"] == ids, f"turn {i}: no replay record with the same prompt"
        sp = SS.Sampling(top_k=1, max_new_tokens=len(rc["tokens"]), stop_token_ids=[248044, 248046])
        r = eng.generate(ids, sp, session=a.session)
        tc, tn = rc["tokens"], rn["tokens"]
        o = {"turn": i, "prompt_tokens": len(ids), "cached_tokens": r["cached_tokens"],
             "served_cached_tokens": rc["cached_tokens"], "reproduces_served": r["tokens"] == tc,
             "same_tokens": tc == tn, "n_cached": len(tc), "n_nocache": len(tn)}
        d = next((j for j, (x, y) in enumerate(zip(tc, tn)) if x != y), None)
        if d is not None:
            o["first_diff_token"] = d
            o["tokens_at_diff"] = [tok.decode([tc[d]]), tok.decode([tn[d]])]
            o["context"] = tok.decode(tc[max(0, d - 12) : d])
            for name in ("resumed", "fresh"):
                if name == "fresh":
                    cache, dcache = model.make_cache(), drafter.make_cache()
                    last = SS.prefill(model, drafter, cache, dcache, ids, 0)
                else:  # the prompt snapshot this turn just left (pos == len(ids), with its logits)
                    snap = eng.cache.find(np.asarray(ids, np.int32), a.session)
                    assert snap.pos == len(ids)
                    cache, dcache = SS.restore(snap.state)
                    last = SS._ref(snap.state.logits)
                lg = logits_at(model, cache, last, tc, d, vv)
                lc, ln = lg[tc[d]].item(), lg[tn[d]].item()
                am = int(mx.argmax(lg).item())
                o[name] = {"logit_cached_tok": lc, "logit_nocache_tok": ln, "gap": abs(lc - ln),
                           "bf16_step": bf16_step(max(abs(lc), abs(ln))),
                           "argmax": "cached" if am == tc[d] else ("nocache" if am == tn[d] else "other")}
                del cache, dcache
                mx.clear_cache()
        out.append(o)
        print(json.dumps(o), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
