"""Memory probe for the group-3 OOMs: active memory after load, and the peak of each phase (spec prefill, spec decode)
per config on one agent prompt, in the frozen spec setup (MLX_MAX_OPS_PER_BUFFER=100 as ab_suite / specdec bench).
usage (from volundr-mlx/spec, under ../with_gpu_lock.sh): python ../combined_test/mem_probe.py [name] [configs]"""
import os
import sys

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "spec"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import specdec as SD  # noqa: E402
from ab_modes import Modes  # noqa: E402

GB = 1e9
name = sys.argv[1] if len(sys.argv) > 1 else "agent/01-pytest-dev__iniconfig"
cfgs = (sys.argv[2] if len(sys.argv) > 2 else "baseline,main+v8q,fused+vprefill,exact").split(",")
model, drafter, tok = SD.load(drafter_bits=4, lm_head_bits=4, nax=True)
mx.synchronize()
print(f"after load: active {mx.get_active_memory() / GB:.2f} GB, peak {mx.get_peak_memory() / GB:.2f} GB", flush=True)
modes = Modes(model, cfgs)
print(f"after Modes: active {mx.get_active_memory() / GB:.2f} GB", flush=True)
e = next(x for x in json.loads((HERE / "suite.json").read_text()) if x["name"] == name)
ids = e["ids"]
for c in cfgs + cfgs:  # twice: the second pass is after every config's kernels are compiled
    modes.set(c)
    mx.clear_cache()
    mx.reset_peak_memory()
    a0 = mx.get_active_memory()
    cache, dcache = model.make_cache(), drafter.make_cache()
    SD.spec_prefill(model, drafter, cache, dcache, ids)
    mx.synchronize()
    pp, ap = mx.get_peak_memory(), mx.get_active_memory()
    del cache, dcache
    mx.clear_cache()
    mx.reset_peak_memory()
    got, r = SD.spec_generate(model, drafter, ids, 64, 8, set(tok.eos_token_ids))
    mx.synchronize()
    pg = mx.get_peak_memory()
    mx.clear_cache()
    print(json.dumps({"config": c, "prompt_tokens": len(ids), "active_before_gb": round(a0 / GB, 2),
                      "prefill_peak_gb": round(pp / GB, 2), "after_prefill_active_gb": round(ap / GB, 2),
                      "spec_generate_peak_gb": round(pg / GB, 2), "steps": r["steps"]}), flush=True)
modes.reset()
