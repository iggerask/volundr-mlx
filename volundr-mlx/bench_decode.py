"""Interleaved in-process A/B of plain greedy decode: configs (ab_modes.py: main / unfused / fused, +q8g64 / +q8g128
heads, +copyidx) switched at run time on one loaded model, so every config sees the same machine state. Run under
with_gpu_lock.sh.

usage: python bench_decode.py --model models/volundr-int4-mlx [--configs main,fused,fused+q8g64] [--reps 3]
       [--tokens 128] [--prompt-file f.txt] [--no-think] [--raw] [--shared-cache]
Prints ms/token per run and the best / median per config, and whether each config's tokens equal the first config's.
--shared-cache: one prefill, then the configs take turns decoding segments of the same sequence (for long prompts,
where a prefill per run costs more than the decode; no token comparison).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as G  # noqa: E402
from ab_modes import Modes  # noqa: E402

SHORT = "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="main,fused")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warm", type=int, default=8, help="untimed tokens at the start of every run")
    ap.add_argument("--prompt", default=SHORT)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--shared-cache", action="store_true")
    args = ap.parse_args()

    model, tok = G.load(args.model, lm_head_q=False)
    G.set_wired_limit()
    text = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    ids = tok.encode(text) if args.raw else tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=not args.no_think)
    vv = model.args.vocab_valid
    configs = args.configs.split(",")
    modes = Modes(model, configs)
    setup = modes.set

    print(f"prompt {len(ids)} tokens; {args.tokens} timed tokens after {args.warm}; configs {configs}", flush=True)
    times = {c: [] for c in configs}
    toks = {}
    if args.shared_cache:
        cache = model.make_cache()
        t0 = time.perf_counter()
        logits = G.prefill(model, cache, ids)
        mx.eval(logits)
        print(f"prefill {time.perf_counter() - t0:.1f} s", flush=True)
        for rep in range(args.reps + 1):
            for c in configs:
                setup(c)
                ts, out = [], []
                for t in G.greedy_tokens(model, cache, logits, args.warm + args.tokens + 1, set(), vv):
                    ts.append(time.perf_counter())
                    out.append(t)
                dt = 1000 * (ts[-1] - ts[args.warm]) / (len(ts) - 1 - args.warm)
                # the loop evaluated one step past the last yielded token; continue from a fresh step on that token
                logits = model(mx.array([[out[-1]]], dtype=mx.int32), cache=cache, last_only=True)[0, -1]
                mx.eval(logits, [st.state for st in cache])
                if rep:
                    times[c].append(dt)
                    print(f"rep {rep} {c:24s} {dt:7.2f} ms/token ({1000 / dt:5.2f} tok/s) at {cache[-2].offset} "
                          "positions", flush=True)
        setup(configs[0])
        print("summary (ms/token):")
        b0 = min(times[configs[0]])
        for c in configs:
            best, med = min(times[c]), statistics.median(times[c])
            print(f"  {c:24s} best {best:7.2f}  median {med:7.2f}  ({1000 / best:5.2f} tok/s)  "
                  f"{100 * (best / b0 - 1):+6.1f}% vs {configs[0]}")
        return
    for rep in range(args.reps + 1):  # rep 0: Metal JIT warm-up of every config, untimed
        for c in configs:
            setup(c)
            cache = model.make_cache()
            t0 = time.perf_counter()
            logits = G.prefill(model, cache, ids)
            mx.eval(logits)
            tp = time.perf_counter() - t0
            ts, out = [], []
            for t in G.greedy_tokens(model, cache, logits, args.warm + args.tokens + 1, set(), vv):
                ts.append(time.perf_counter())
                out.append(t)
            dt = 1000 * (ts[-1] - ts[args.warm]) / (len(ts) - 1 - args.warm)
            del cache
            mx.clear_cache()
            if rep == 0:
                toks[c] = out
                continue
            times[c].append(dt)
            same = out == toks[configs[0]]
            print(f"rep {rep} {c:16s} {dt:7.2f} ms/token ({1000 / dt:5.2f} tok/s), prefill {tp:6.2f} s, "
                  f"tokens == {configs[0]}: {same}", flush=True)
    setup(configs[0])
    print("summary (ms/token):")
    b0 = min(times[configs[0]])
    for c in configs:
        best, med = min(times[c]), statistics.median(times[c])
        div = next((i for i, (a, b) in enumerate(zip(toks[c], toks[configs[0]])) if a != b), None)
        print(f"  {c:16s} best {best:7.2f}  median {med:7.2f}  ({1000 / best:5.2f} tok/s)  {100 * (best / b0 - 1):+6.1f}% "
              f"vs {configs[0]}; greedy tokens vs {configs[0]}: "
              + ("identical" if div is None else f"first differ at {div} of {len(toks[c])}"))


if __name__ == "__main__":
    main()
