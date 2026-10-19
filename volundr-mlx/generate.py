"""Greedy generation with Volundr MLX (no speculative decoding), with prefill / decode timing and peak memory.

usage: python generate.py --model models/volundr-int4-mlx --prompt "Hello" [--max-tokens 256] [--no-think]
       [--raw] [--prompt-file f.txt] [--chunk 2048]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volundr  # noqa: E402


def load(path):
    from mlx_lm.utils import load_model, load_tokenizer

    path = Path(path)
    model, config = load_model(path, get_model_classes=lambda config: (volundr.Model, volundr.ModelArgs))
    eos = config.get("eos_token_id")
    tok = load_tokenizer(path, eos_token_ids=eos if isinstance(eos, list) else [eos])
    return model, tok


def set_wired_limit():
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])


def prefill(model, cache, ids, chunk=2048):
    """Feed ids [n] through the cache in chunks; returns the last position's logits [V]."""
    ids = mx.array(ids, dtype=mx.int32)[None]
    n = ids.shape[1]
    for s in range(0, n, chunk):
        logits = model(ids[:, s : s + chunk], cache=cache, last_only=True)
        mx.eval(logits, [c.state for c in cache])
        mx.clear_cache()
    return logits[0, -1]


def greedy_tokens(model, cache, first_logits, max_tokens, eos, vocab_valid):
    """Yield greedy tokens; the next step's graph is built while the current one runs."""

    def pick(logits):
        return mx.argmax(logits[..., :vocab_valid], axis=-1).astype(mx.int32)

    def step(y):
        return pick(model(y.reshape(1, 1), cache=cache, last_only=True)[0, -1])

    # cache updates the next token does not read (pooled far blocks at short context, the Engram ids) are evaluated
    # with it, so no lazy chain keeps old cache buffers alive
    y = pick(first_logits)
    mx.async_eval(y)
    nxt = None
    for i in range(max_tokens):
        if i + 1 < max_tokens:
            nxt = step(y)
            mx.async_eval(nxt, [c.state for c in cache])
        t = y.item()
        yield t
        if t in eos:
            break
        y = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Write a Python function that checks whether a string is a palindrome.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-think", action="store_true", help="chat template with enable_thinking=False")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--json", default=None, help="write a metrics record here")
    ap.add_argument("--warmup", type=int, default=0, help="untimed greedy tokens on a short prompt first (Metal JIT)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load(args.model)
    set_wired_limit()
    t_load = time.perf_counter() - t0
    weights_gb = mx.get_active_memory() / 1e9

    text = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    if args.raw:
        ids = tok.encode(text)
    else:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=not args.no_think)
    eos = set(tok.eos_token_ids)
    vv = model.args.vocab_valid
    if args.warmup:
        c = model.make_cache()
        for _ in greedy_tokens(model, c, prefill(model, c, tok.encode("Warm up the kernels."), args.chunk),
                               args.warmup, set(), vv):
            pass
        del c
        mx.clear_cache()

    mx.reset_peak_memory()
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = prefill(model, cache, ids, args.chunk)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0

    det = tok.detokenizer
    det.reset()
    out, times = [], []
    for t in greedy_tokens(model, cache, logits, args.max_tokens, eos, vv):
        times.append(time.perf_counter())
        out.append(t)
        det.add_token(t)
        print(det.last_segment, end="", flush=True)
    det.finalize()
    print(det.last_segment, flush=True)

    n_dec = len(out) - 1
    dec_s = times[-1] - times[0] if n_dec > 0 else float("nan")
    rec = {
        "prompt_tokens": len(ids),
        "generated": len(out),
        "load_s": round(t_load, 1),
        "weights_gb": round(weights_gb, 2),
        "prefill_s": round(t_prefill, 3),
        "prefill_tok_s": round(len(ids) / t_prefill, 1),
        "decode_tok_s": round(n_dec / dec_s, 2) if n_dec > 0 else None,
        "decode_ms_per_tok": round(1000 * dec_s / n_dec, 2) if n_dec > 0 else None,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    print("\n" + json.dumps(rec), file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps({**rec, "tokens": out}, indent=1))


if __name__ == "__main__":
    main()
