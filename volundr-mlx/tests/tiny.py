"""Tiny random Volundr for tests: every mixer kind, Engram, mHC, BCSA far regime at short lengths."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volundr  # noqa: E402

TINY = dict(
    hidden_size=128, intermediate_size=256, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, vocab_size=300, vocab_valid=290, rms_norm_eps=1e-6, linear_conv_kernel_dim=4,
    linear_key_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_value_head_dim=32,
    kda_gate_rank=16, mixer_types=["kda", "kda", "bcsa", "kda", "bcsa", "attn"], bcsa_window=16, bcsa_compress=4,
    bcsa_topk=3, bcsa_index_heads=2, bcsa_index_dim=32, engram_layers=[1], engram_orders=[2, 3], engram_rows=1024,
    engram_dim=16, mhc_streams=4, mhc_sinkhorn_iters=10, rope_theta=1e7, partial_rotary_factor=0.25,
    bcsa_prefill_tile=8, bcsa_select_rows=16,
)


def tiny_args(**kw):
    return volundr.ModelArgs.from_dict({**TINY, **kw})


def tiny_model(seed=0, **kw):
    mx.random.seed(seed)
    m = volundr.Model(tiny_args(**kw))
    params = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith(("A_log", "dt_bias", "conv1d.weight")):
            v = (mx.random.normal(v.shape) * (0.5 if "conv1d" in k else 0.3)).astype(mx.float32)
        elif k.endswith(("pre_static", "post_static")):
            v = (1.0 + 0.2 * mx.random.normal(v.shape)).astype(mx.bfloat16)
        elif k.endswith("res_static"):
            v = (4.0 * mx.eye(v.shape[0]) + mx.random.normal(v.shape)).astype(mx.bfloat16)
        elif k.endswith("norm.weight"):
            v = (0.1 * mx.random.normal(v.shape)).astype(mx.bfloat16) + (1 if ("mhc" in k or "engram" in k or "linear_attn.norm" in k) else 0)
            v = v.astype(mx.bfloat16)
        else:
            v = v.astype(mx.bfloat16)
        params.append((k, v))
    m.update(tree_unflatten(params))
    mx.eval(m.parameters())
    return m
