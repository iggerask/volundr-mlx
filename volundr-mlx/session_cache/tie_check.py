"""Where the e2e conversation's replies with and without the cache differ: are those near-ties?

Replays the recorded e2e requests in-process (session.Engine, the worker's code path) twice: with a session cache
(each turn resumes as in the served run) and without. Both reproduce the served runs' replies (checked against the
recorded text). At each turn whose replies differ, the first differing token is located, and the logits of the
target model there are read teacher-forced along the common prefix, from both states (resumed and fresh): the gap
between the two candidate tokens' logits tells whether the flip is a tie (a bf16 step or two).

usage (from volundr-mlx, under ./with_gpu_lock.sh):
  python session_cache/tie_check.py session_cache/e2e/cached.json session_cache/e2e/nocache.json [--out ...]
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


def text_of(tok, ids):
    s = tok.decode(ids, skip_special_tokens=False)
    return s


def teacher_logits(model, drafter, cache, dcache, last, toks, d, vv):
    """Logits predicting toks[d] after feeding toks[:d] (plain decode steps from the given state)."""
    lg = last
    for t in toks[:d]:
        lg, _ = SS.decode_step(model, cache, int(t))
        mx.eval(lg, [c.state for c in cache])
    return lg[:vv].astype(mx.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cached")
    ap.add_argument("nocache")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, drafter, tok = specdec.load(4, 4, nax=True)
    specdec.warm_up(model, drafter, tok.encode("Warm up the kernels, please."))
    vv = model.args.vocab_valid
    cj = json.loads(Path(a.cached).read_text())
    nj = json.loads(Path(a.nocache).read_text())
    eos = list(tok.eos_token_ids)
    cached_eng = SS.Engine(model, drafter, SS.SessionCache(budget_gb=12))
    plain_eng = SS.Engine(model, drafter, None)
    out = []
    for i, (ct, nt) in enumerate(zip(cj["turns"], nj["turns"])):
        body = ct["body"]
        kw = {"tools": body.get("tools"), "add_generation_prompt": True}
        if body.get("reasoning_effort"):
            kw["reasoning_effort"] = body["reasoning_effort"]
        ids = tok.apply_chat_template(body["messages"], **kw)
        n_srv = ct["stats"]["prompt_tokens"]
        sp = SS.Sampling(top_k=1, max_new_tokens=body["max_tokens"], stop_token_ids=eos)
        rc = cached_eng.generate(ids, sp, session=cj["session"])
        rn = plain_eng.generate(ids, sp)
        tc, tn = rc["tokens"], rn["tokens"]
        rec_c = ct["reply"]["reasoning"] + ct["reply"]["content"]
        o = {"turn": i, "prompt_tokens": len(ids), "server_prompt_tokens": n_srv, "cached_tokens": rc["cached_tokens"],
             "same_tokens": tc == tn, "n_cached": len(tc), "n_nocache": len(tn),
             "served_same_reply": nt["stats"].get("same_reply")}
        # do the in-process runs reproduce the served ones? (text, reasoning + content, special tokens stripped)
        strip = lambda s: s.replace("<|/think|>", "").replace("<|agens_end|>", "").replace("\n", "").replace(" ", "")  # noqa: E731
        o["reproduces_served_cached"] = strip(text_of(tok, tc)).startswith(strip(rec_c)[:200])
        d = next((j for j, (x, y) in enumerate(zip(tc, tn)) if x != y), None)
        if d is not None:
            o["first_diff_token"] = d
            o["tokens_at_diff"] = [tok.decode([tc[d]]), tok.decode([tn[d]])]
            gaps = {}
            for name, (state, last) in {"fresh": (None, None), "resumed": ("resumed", None)}.items():
                if name == "fresh":
                    cache, dcache = model.make_cache(), drafter.make_cache()
                    lst = SS.prefill(model, drafter, cache, dcache, ids, 0)
                else:
                    snap = cached_eng.cache.find(np.asarray(ids, np.int32), cj["session"])
                    # the prompt snapshot this turn left (pos == len(ids)) carries the logits
                    cache, dcache = SS.restore(snap.state)
                    lst = SS._ref(snap.state.logits) if snap.pos == len(ids) else SS.prefill(
                        model, drafter, cache, dcache, ids, snap.pos)
                lg = teacher_logits(model, drafter, cache, dcache, lst, tc, d, vv)
                top2 = mx.topk(lg, 2)
                gaps[name] = {"logit_cached_tok": lg[tc[d]].item(), "logit_nocache_tok": lg[tn[d]].item(),
                              "gap": abs(lg[tc[d]].item() - lg[tn[d]].item()),
                              "top2_margin": abs(top2[1].item() - top2[0].item()),
                              "argmax_is": "cached" if int(mx.argmax(lg).item()) == tc[d] else
                              ("nocache" if int(mx.argmax(lg).item()) == tn[d] else "other")}
                del cache, dcache
                mx.clear_cache()
            o["at_diff"] = gaps
        out.append(o)
        print(json.dumps(o), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
