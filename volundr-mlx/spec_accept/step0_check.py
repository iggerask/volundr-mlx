"""Step 0 gate (prefill memory fix): the layer-synced spec prefill against the old unsynchronised one, interleaved in
one process under the spec harness's command-buffer settings (MLX_MAX_OPS_PER_BUFFER=100, no MB cap).

Per prompt and rep, variants interleaved (new, old, new, old ...):
- spec prefill alone: wall time, peak memory from a reset counter, and bit-identity of the last logits and of every
  cache / drafter-cache array against the first run;
- then a full spec_generate (prefill + --new tokens, lookup on; one Lookup per variant, as ab_suite): tokens and
  per-step acceptance must match between variants, and the decode ms/step is compared.

usage (from volundr-mlx, under ./with_gpu_lock.sh, VOLUNDR_LMHEAD8=64):
  python spec_accept/step0_check.py --names long/03-long3,coding/00-12623519-think --reps 2 --out <jsonl>
"""
import os
import sys

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spec"))
sys.path.insert(0, str(ROOT))
import specdec as SD  # noqa: E402
from lookup import Lookup  # noqa: E402

GB = 1e9


def flat(x):
    if isinstance(x, mx.array):
        return [x]
    if isinstance(x, (list, tuple)):
        return [a for y in x for a in flat(y)]
    return []


def states(cache, dcache):
    return flat([c.state for c in cache]) + flat(dcache.state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="long/03-long3,coding/00-12623519-think")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, drafter, tok = SD.load(drafter_bits=4, lm_head_bits=4, nax=True)
    eos = set(tok.eos_token_ids)
    suite = {e["name"]: e for e in json.loads((ROOT / "spec/suite.json").read_text())}
    w = tok.encode("Warm up the kernels, please.")
    SD.spec_generate(model, drafter, w, 24, 8)
    looks = {v: Lookup() for v in ("new", "old")}
    fout = open(args.out, "a")
    for name in args.names.split(","):
        ids = suite[name]["ids"]
        ref = {}
        for rep in range(args.reps):
            for var in ("new", "old"):
                sync = var == "new"
                mx.synchronize()
                mx.clear_cache()
                mx.reset_peak_memory()
                cache, dcache = model.make_cache(), drafter.make_cache()
                t0 = time.perf_counter()
                last = SD.spec_prefill(model, drafter, cache, dcache, ids, layer_sync=sync)
                mx.eval(last)
                mx.synchronize()
                t_pre = time.perf_counter() - t0
                peak = mx.get_peak_memory() / GB
                arrs = [last] + states(cache, dcache)
                if not ref:
                    ref["arrs"] = arrs
                    same = None
                else:
                    same = len(arrs) == len(ref["arrs"]) and all(
                        a.shape == b.shape and mx.array_equal(a, b).item() for a, b in zip(arrs, ref["arrs"]))
                del cache, dcache, arrs, last
                mx.clear_cache()
                mx.reset_peak_memory()
                SD.PREFILL_LAYER_SYNC = sync
                got, r = SD.spec_generate(model, drafter, ids, args.new, 8, eos, lookup=looks[var])
                gen_peak = mx.get_peak_memory() / GB
                rec = {"name": name, "prompt_tokens": len(ids), "rep": rep, "variant": var, "prefill_s": round(t_pre, 3),
                       "prefill_tok_s": round(len(ids) / t_pre, 1), "prefill_peak_gb": round(peak, 2),
                       "prefill_state_same_as_first": same, "gen_prefill_s": round(r["prefill_s"], 3),
                       "gen_peak_gb": round(gen_peak, 2), "steps": r["steps"],
                       "ms_step": round(1e3 * r["decode_s"] / r["steps"], 2), "mean_acc": sum(r["acc"]) / len(r["acc"]),
                       "tokens": got, "acc": r["acc"]}
                key = (rep, var)
                ref.setdefault("gen", {})[key] = (got, r["acc"])
                print(json.dumps({k: v for k, v in rec.items() if k not in ("tokens", "acc")}), flush=True)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
        g = ref["gen"]  # per rep, new against old (each variant's Lookup has seen the same history)
        print(json.dumps({"name": name,
                          "tokens_new_eq_old": [g[(r, "new")][0] == g[(r, "old")][0] for r in range(args.reps)],
                          "acc_new_eq_old": [g[(r, "new")][1] == g[(r, "old")][1] for r in range(args.reps)]}),
              flush=True)
        del ref
        mx.clear_cache()
    SD.PREFILL_LAYER_SYNC = True


if __name__ == "__main__":
    main()
