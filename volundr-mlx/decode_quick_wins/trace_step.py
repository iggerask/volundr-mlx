"""A few pipelined decode steps for a Metal System Trace (warm-up first, then a marker sleep, then 12 steps)."""
import os
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx

import decode_fused as DF
import generate as G

DF.ENABLED = os.environ.get("VOLUNDR_FUSED", "1") != "0"
model, tok = G.load(WT + "/models/volundr-int4-mlx")
G.set_wired_limit()
vv = model.args.vocab_valid
ids = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],
                              add_generation_prompt=True, enable_thinking=True)
cache = model.make_cache()
logits = G.prefill(model, cache, ids)
for _ in G.greedy_tokens(model, cache, logits, 24, set(), vv):
    pass
mx.synchronize()
time.sleep(0.5)  # gap in the GPU timeline before the measured steps
cache = model.make_cache()
logits = G.prefill(model, cache, ids)
mx.synchronize()
time.sleep(0.5)
ts = []
for _ in G.greedy_tokens(model, cache, logits, 13, set(), vv):
    ts.append(time.perf_counter())
print(f"steps: {[round(1000 * (b - a), 2) for a, b in zip(ts, ts[1:])]}", flush=True)
