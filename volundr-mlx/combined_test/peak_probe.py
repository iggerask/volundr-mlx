"""Peak memory of one prompt in a fresh process, for any checkout of volundr-mlx (main, the two branches, combined):
which part of the 57-58 GB the group-3 records show is the merged code and which the harness.

usage: python peak_probe.py --tree <volundr-mlx dir> [--mode spec|greedy] [--bufs spec|default]
       [--name long/03-long3] [--config exact+q8g64] [--modes baseline,exact,exact+q8g64]
  --bufs spec: MLX_MAX_OPS_PER_BUFFER=100, MLX_MAX_MB_PER_BUFFER=1000000 (specdec bench / ab_suite spec); default: MLX's
  --config / --modes: combined tree only (ab_modes.py; --modes builds every listed config's head, as ab_suite does)
  VOLUNDR_LMHEAD8 from the environment, where the tree supports it.
Prints one JSON line: active memory after load, peak of the spec (or greedy) prefill alone, peak of the full
generation (a second prefill + 256 tokens), each from a reset peak counter."""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--mode", default="spec", choices=["spec", "greedy"])
ap.add_argument("--bufs", default="spec", choices=["spec", "default"])
ap.add_argument("--name", default="long/03-long3")
ap.add_argument("--config", default="")
ap.add_argument("--modes", default="")
ap.add_argument("--new", type=int, default=256)
ap.add_argument("--label", default="")
args = ap.parse_args()
if args.bufs == "spec":
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "100"
    os.environ["MLX_MAX_MB_PER_BUFFER"] = "1000000"
else:
    os.environ.pop("MLX_MAX_OPS_PER_BUFFER", None)
    os.environ.pop("MLX_MAX_MB_PER_BUFFER", None)

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

TREE = Path(args.tree).resolve()
sys.path.insert(0, str(TREE / "spec"))
sys.path.insert(0, str(TREE))
import specdec as SD  # noqa: E402

GB = 1e9
rec = {"label": args.label, "tree": str(TREE), "mode": args.mode, "bufs": args.bufs, "name": args.name,
       "config": args.config or None, "modes": args.modes or None, "lmhead8": os.environ.get("VOLUNDR_LMHEAD8", "")}
e = next(x for x in json.loads(Path("/Users/ingemarrask/personal-dev/llms/volundr-mlx/spec/suite.json").read_text())
         if x["name"] == args.name)
ids = e["ids"]
rec["prompt_tokens"] = len(ids)
if args.mode == "spec":
    model, drafter, tok = SD.load(drafter_bits=4, lm_head_bits=4, nax=True)
else:
    model, tok = SD.G.load(SD.MODEL)
    SD.G.set_wired_limit()
    drafter = None
if args.config or args.modes:
    import ab_modes

    modes = ab_modes.Modes(model, (args.modes or args.config).split(","))
    modes.set(args.config or args.modes.split(",")[0])
    rec["switched"] = ab_modes.describe()
mx.synchronize()
rec["target_head_q8"] = model.lm_head.group_size if "scales" in model.lm_head else 0
rec["active_after_load_gb"] = round(mx.get_active_memory() / GB, 2)
rec["peak_load_gb"] = round(mx.get_peak_memory() / GB, 2)
eos = set(tok.eos_token_ids)
try:
    w = tok.encode("Warm up the kernels, please.")  # as specdec / ab_suite
    if drafter is not None:
        SD.spec_generate(model, drafter, w, 24, 8)
    SD.greedy_generate(model, w, 8)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    cache = model.make_cache()
    if drafter is not None:
        dcache = drafter.make_cache()
        SD.spec_prefill(model, drafter, cache, dcache, ids)
    else:
        mx.eval(SD.G.prefill(model, cache, ids))
    mx.synchronize()
    rec["prefill_s"] = round(time.perf_counter() - t0, 2)
    rec["prefill_peak_gb"] = round(mx.get_peak_memory() / GB, 2)
    rec["active_after_prefill_gb"] = round(mx.get_active_memory() / GB, 2)
    del cache
    if drafter is not None:
        del dcache
    mx.clear_cache()
    mx.reset_peak_memory()
    if drafter is not None:
        from lookup import Lookup

        got, r = SD.spec_generate(model, drafter, ids, args.new, 8, eos, lookup=Lookup())
        rec["tok_s"] = round((len(got) - 1) / r["decode_s"], 2)
    else:
        got, r = SD.greedy_generate(model, ids, args.new, eos)
        rec["tok_s"] = round((len(got) - 1) / r["decode_s"], 2)
    mx.synchronize()
    rec["generate_peak_gb"] = round(mx.get_peak_memory() / GB, 2)
except RuntimeError as err:
    rec["error"] = str(err)[:120]
    rec["peak_at_error_gb"] = round(mx.get_peak_memory() / GB, 2)
print("PEAK " + json.dumps(rec), flush=True)
