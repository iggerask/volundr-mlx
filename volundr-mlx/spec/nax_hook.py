"""Small-M quantized matmuls on the M5 neural accelerators for speculative verify / drafting.

enable(model) switches every 4-bit g128 affine QuantizedLinear (N % 64 == 0, no bias) to NaxLinear: a call with
2..16 rows of bf16 input on the GPU goes through agens-mlx-bench/smallm's qmm_nax (prep_nax.prep pads x to 16 rows
and builds its group sums in one kernel; linears that read the same input array share one prep). Decode (1 row),
prefill (>16 rows) and everything else take the stock path unchanged. smallm/VERIFY_STEP.md: an 8-token verify step
costs 1.25-1.27x a decode step this way against 2.16x on stock MLX; per-linear error equals stock's.
"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2] / "agens-mlx-bench/smallm"))
import prep_nax  # noqa: E402
import qmm_nax  # noqa: E402

_memo = {"x": None, "p": None}


class NaxLinear(nn.QuantizedLinear):
    def __call__(self, x):
        K = x.shape[-1]
        rows = x.size // K
        N = self.weight.shape[0]
        if not 2 <= rows <= qmm_nax.MT or x.dtype != mx.bfloat16 or mx.default_device() != mx.gpu:
            return super().__call__(x)
        if _memo["x"] is not x:
            _memo["x"], _memo["p"] = x, prep_nax.prep(x.reshape(rows, K))
        return qmm_nax.qmm_nax_prepped(_memo["p"], self.weight, self.scales, self.biases).reshape(*x.shape[:-1], N)


def enable(model) -> int:
    n = 0
    for _, m in model.named_modules():
        if (type(m) is nn.QuantizedLinear and m.bits == 4 and m.group_size == 128 and m.mode == "affine"
                and "bias" not in m and m.weight.shape[0] % qmm_nax.NT == 0 and m.scales.dtype == mx.bfloat16):
            m.__class__ = NaxLinear
            n += 1
    return n


def disable(model) -> int:
    n = 0
    for _, m in model.named_modules():
        if type(m) is NaxLinear:
            m.__class__ = nn.QuantizedLinear
            n += 1
    _memo["x"] = _memo["p"] = None
    return n
