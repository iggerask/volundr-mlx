"""Opt-in W4A8 prefill matmuls for volundr.py (volundr.enable_w4a8; VOLUNDR_W4A8=1 turns it on at the first call).

enable(model) switches every 4-bit g128 affine QuantizedLinear to W4A8Linear, except those whose name ends in one
of VOLUNDR_W4A8_SKIP (comma-separated, default SKIP; "" switches all). A call with at least MIN_ROWS rows of bf16
input on the GPU goes through act_quant + w4a8_matmul; decode steps and everything else take the stock path
unchanged. Linears that read the same input array (q/k/v/index, qkv/z/b/f_down, gate/up) share one act_quant.
"""
import os

import mlx.core as mx
import mlx.nn as nn

import w4a8

MIN_ROWS = 64
CFG = dict(sm=32, sn=32, wm=2, wn=2)  # w4a8_matmul's default path ("32x32x2x2RV" in bench_w4a8.py)
PERM = 0  # act_quant layout for that path
# Attention k / v projections (N = 1024) stay on the stock path: they are among the worst-hit linears on real
# activations and among the cheapest (NOTES.md, "k_proj / v_proj").
SKIP = "k_proj,v_proj"
_memo = {"x": None, "q": None}


def _bn():
    return CFG["sn"] * CFG["wn"]


class W4A8Linear(nn.QuantizedLinear):
    def __call__(self, x):
        K = x.shape[-1]
        rows = x.size // K
        N = self.weight.shape[0]
        if (rows < MIN_ROWS or x.dtype != mx.bfloat16 or N % _bn() or "bias" in self
                or mx.default_device() != mx.gpu):
            return super().__call__(x)
        if _memo["x"] is not x:
            _memo["x"], _memo["q"] = x, w4a8.act_quant(x.reshape(rows, K), mpad=CFG["sm"] * CFG["wm"], perm=PERM)
        xq, prm = _memo["q"]
        y = w4a8.w4a8_matmul(xq, prm, rows, self.weight, self.scales, self.biases, **CFG)
        return y.reshape(*x.shape[:-1], N)


def enable(model, skip=None) -> int:
    if skip is None:
        skip = os.environ.get("VOLUNDR_W4A8_SKIP", SKIP)
    skip = tuple(v for v in skip.split(",") if v)
    n = 0
    for name, m in model.named_modules():
        if name.split(".")[-1] in skip:
            continue
        if type(m) is nn.QuantizedLinear and m.bits == 4 and m.group_size == 128 and m.mode == "affine":
            m.__class__ = W4A8Linear
            n += 1
    return n


def disable(model) -> int:
    n = 0
    for _, m in model.named_modules():
        if type(m) is W4A8Linear:
            m.__class__ = nn.QuantizedLinear
            n += 1
    _memo["x"] = _memo["q"] = None
    return n
