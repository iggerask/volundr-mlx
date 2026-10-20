"""Profile a real Volundr decode step: graph node counts per primitive, step time, and step time without matmuls."""
import collections
import re
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx
import mlx.nn as nn

import generate as G
import volundr as V

model, tok = G.load(WT + "/models/volundr-int4-mlx")
G.set_wired_limit()
vv = model.args.vocab_valid
ids = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],
                              add_generation_prompt=True, enable_thinking=True)


def run(n=48, warm=8):
    cache = model.make_cache()
    logits = G.prefill(model, cache, ids)
    ts = []
    for t in G.greedy_tokens(model, cache, logits, n + warm, set(), vv):
        ts.append(time.perf_counter())
    ts = ts[warm:]
    return 1000 * (ts[-1] - ts[0]) / (len(ts) - 1), cache


def dot_counts():
    cache = model.make_cache()
    logits = G.prefill(model, cache, ids)
    y = mx.argmax(logits[:vv]).astype(mx.int32)
    mx.eval(y)
    out = model(y.reshape(1, 1), cache=cache, last_only=True)
    path = "/private/tmp/claude-501/-Users-ingemarrask-personal-dev-llms/b08d8102-657c-4e37-b76d-c8780a9ddb09/scratchpad/step.dot"
    st = [a for c in cache for a in (c.state if isinstance(c.state, (list, tuple)) else [c.state]) if isinstance(a, mx.array)]
    mx.export_to_dot(path, out, *st)
    txt = open(path).read()
    labels = re.findall(r'label\s*=\s*"([^"]+)"', txt)
    c = collections.Counter(l.split()[0] for l in labels)
    print("graph nodes:", sum(c.values()))
    for k, v in c.most_common(60):
        print(f"  {k:40s} {v}")


if "dot" in sys.argv:
    dot_counts()

if "time" in sys.argv:
    real_q = nn.QuantizedLinear.__call__
    real_l = nn.Linear.__call__

    def nomm_q(self, x):
        return mx.zeros((*x.shape[:-1], self.weight.shape[0]), dtype=x.dtype) + x[..., :1]

    def nomm_head(self, x):
        if self.weight.shape[0] > 100000:
            return mx.zeros((*x.shape[:-1], self.weight.shape[0]), dtype=x.dtype) + x[..., :1]
        return real_l(self, x)

    for rep in range(3):
        nn.QuantizedLinear.__call__, nn.Linear.__call__ = real_q, real_l
        a, _ = run()
        nn.QuantizedLinear.__call__, nn.Linear.__call__ = nomm_q, nomm_head
        b, _ = run()
        nn.QuantizedLinear.__call__, nn.Linear.__call__ = real_q, nomm_head
        c, _ = run()
        print(f"rep {rep}: real {a:.2f} ms/step, no 4-bit matmuls and no head {b:.2f}, no head {c:.2f}", flush=True)
