"""End-to-end prefill speed (no per-op syncs): chunked prefill of a long prompt, time per chunk, then greedy decode.

usage: python bench_prefill.py --model models/volundr-int4-mlx [--tokens 20480] [--chunk 2048] [--decode 64]
"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import greedy_tokens, load, set_wired_limit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=20480)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=64)
    args = ap.parse_args()
    model, tok = load(args.model)
    set_wired_limit()
    src = (Path(__file__).parent / "ref/agens-engine/kernels/m1/m1_model.cu").read_text()
    ids = tok.encode(src * (1 + args.tokens * 4 // len(src)))[: args.tokens]
    # warm-up: compile every chunk shape once on a throwaway cache
    wc = model.make_cache()
    for s in range(0, min(len(ids), 2 * args.chunk), args.chunk):
        mx.eval(model(mx.array([ids[s : s + args.chunk]]), cache=wc, last_only=True))
    del wc
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = model.make_cache()
    t_all = time.perf_counter()
    for s in range(0, len(ids), args.chunk):
        t = time.perf_counter()
        logits = model(mx.array([ids[s : s + args.chunk]]), cache=cache, last_only=True)
        mx.eval(logits, [c.state for c in cache])
        dt = time.perf_counter() - t
        n = min(args.chunk, len(ids) - s)
        print(f"  chunk at {s:6d}: {dt:6.2f} s  {n / dt:6.0f} tok/s", flush=True)
    t_all = time.perf_counter() - t_all
    print(f"prefill {len(ids)} tokens: {t_all:.1f} s, {len(ids) / t_all:.0f} tok/s, peak {mx.get_peak_memory() / 1e9:.2f} GB")
    # the first decode steps compile the single-token kernel variants: time from the 8th token on
    times = []
    for _ in greedy_tokens(model, cache, logits[0, -1], args.decode + 8, set(), model.args.vocab_valid):
        times.append(time.perf_counter())
    times = times[8:]
    n = len(times) - 1
    print(f"decode at {len(ids)}: {n / (times[-1] - times[0]):.2f} tok/s ({1000 * (times[-1] - times[0]) / n:.1f} ms), "
          f"peak {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
