"""lm_head cost: bf16 vs 8-bit affine g64 / g128 (stock mx.quantized_matmul), M = 1 and 8, weights the real head."""
import sys
import time

WT = "/Users/ingemarrask/personal-dev/llms/.claude/worktrees/agent-ae877536cb39838a7/volundr-mlx"
sys.path.insert(0, WT)
import mlx.core as mx
import mlx.nn as nn

w = mx.load(WT + "/models/volundr-int4-mlx/model-00004.safetensors")
key = [k for k in w if "lm_head" in k]
if not key:
    import json
    idx = json.load(open(WT + "/models/volundr-int4-mlx/model.safetensors.index.json"))["weight_map"]
    f = idx["lm_head.weight"]
    w = mx.load(WT + "/models/volundr-int4-mlx/" + f)
W = w["lm_head.weight"]
mx.eval(W)
print(W.shape, W.dtype, W.nbytes / 1e9, "GB")
lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
lin.weight = W
heads = {"bf16": lin}
for g in (64, 128):
    q = nn.QuantizedLinear.from_linear(lin, g, 8)
    mx.eval(q.parameters())
    heads[f"q8g{g}"] = q
    print(f"q8g{g}: {sum(v.nbytes for v in (q.weight, q.scales, q.biases)) / 1e9:.3f} GB")
mx.random.seed(0)
for M in (1, 8):
    x = (mx.random.normal((M, W.shape[1])) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    best = {}
    for rep in range(7):
        for n, h in heads.items():
            mx.eval(h(x))
            mx.synchronize()
            t = time.perf_counter()
            ys = []
            for i in range(10):
                ys.append(mx.argmax(h(x + i * 1e-3), axis=-1))
                mx.eval(ys[-1])
            mx.synchronize()
            dt = (time.perf_counter() - t) / 10
            best[n] = min(best.get(n, 1e9), dt)
    print(f"M={M}: " + ", ".join(f"{n} {best[n] * 1e3:.2f} ms" for n in heads), flush=True)
