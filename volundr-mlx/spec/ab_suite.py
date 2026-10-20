"""Interleaved in-process A/B on the prompt suite: each prompt runs once per config, configs back to back, so they see
the same machine state (other apps move per-step time by up to ~25% for minutes at a time). Configs switch at run
time on one loaded model (../ab_modes.py: main / unfused / fused or the aliases baseline / exact, +q8g64 / +q8g128
heads, +copyidx, +vprefill / +v8q BCSA verify). Writes one specdec-format jsonl per config, so analyze.py reads them;
each record carries what the config switched on (fused, views, bcsa_verify).

usage (from volundr-mlx/spec, under ../with_gpu_lock.sh):
  python ab_suite.py greedy --split tune --configs main,fused,fused+q8g64 --out runs/decode-quick-wins_tune_greedy
  python ab_suite.py spec   --split tune --configs main,fused,fused+q8g64 --out runs/decode-quick-wins_tune_spec
spec runs the frozen config of NOTES.md (4-bit drafter + 4-bit head copy, NAX verify, prompt lookup, T = 8). A spec
config may add spec-loop flags after "@" (specdec.loop_options), e.g. exact+q8g64@early@cv@long16; each config keeps its
own Lookup, and --resume replays it with that config's flags.
"""
import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "spec" and "MLX_MAX_OPS_PER_BUFFER" not in os.environ:
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "100"  # as specdec.py bench (see there)
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import specdec as SD  # noqa: E402
from ab_modes import Modes, describe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["greedy", "spec"])
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--cats", default="")
    ap.add_argument("--per-cat", type=int, default=0)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--configs", default="main,fused")
    ap.add_argument("--out", required=True, help="prefix; writes <out>_<config>.jsonl (appends)")
    ap.add_argument("--resume", action="store_true", help="skip (prompt, config) pairs already in the out files; spec: "
                    "replay their records into each config's Lookup, so its history is that of an uninterrupted run")
    args = ap.parse_args()

    spec = args.mode == "spec"
    if spec:
        model, drafter, tok = SD.load(drafter_bits=4, lm_head_bits=4, nax=True)
    else:
        model, tok = SD.G.load(SD.MODEL, lm_head_q=False)
        SD.G.set_wired_limit()
        drafter = None
    assert "scales" not in model.lm_head, "run without VOLUNDR_LMHEAD8: configs choose the head"
    configs = args.configs.split(",")
    # config = <ab_modes config>[@<spec loop flag>...] (specdec.loop_options, e.g. exact+q8g64@early)
    modes = Modes(model, [c.split("@")[0] for c in configs])
    setup = lambda c: modes.set(c.split("@")[0])  # noqa: E731
    loop = {c: SD.loop_options(c.split("@")[1:]) for c in configs}

    eos = set(tok.eos_token_ids)
    looks = {}
    if spec:  # one Lookup per config: its running means carry across prompts, as in a specdec.py run of that config
        from lookup import Lookup

        looks = {c: Lookup() for c in configs}
    w = tok.encode("Warm up the kernels, please.")
    for c in configs:  # Metal JIT for every config (as specdec.py's warm-up: no lookup)
        setup(c)
        if spec:
            SD.warm_up(model, drafter, w, 8, **loop[c])
        SD.greedy_generate(model, w, 8)
    prompts = SD.suite(args.suite, args.split, args.per_cat, [c for c in args.cats.split(",") if c])
    done = {}
    if args.resume:
        for c in configs:
            p = Path(f"{args.out}_{c}.jsonl")
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        done[(r["name"], c)] = r
        if spec:  # the Lookup history of this invocation's prompts that already ran, as if the run had not stopped
            for e in prompts:
                for c in configs:
                    if (e["name"], c) in done:
                        SD.lookup_replay(looks[c], e["ids"], done[(e["name"], c)], eos, **loop[c])
    outs = {c: open(f"{args.out}_{c}.jsonl", "a") for c in configs}
    print(f"{args.mode}: {len(prompts)} prompts x {configs}", flush=True)
    for e in prompts:
        for c in configs:
            if (e["name"], c) in done:
                continue
            setup(c)
            rec = {"name": e["name"], "category": e["category"], "split": e["split"], "prompt_tokens": len(e["ids"]),
                   "config": c, "mode": args.mode, **describe()}
            if spec:
                rec.update(T=8, drafter_bits=4, head_bits=4, nax=True, lookup=True, loop=loop[c])
                got, r = SD.spec_generate(model, drafter, e["ids"], args.new, 8, eos, lookup=looks[c], **loop[c])
                rec.update(generated=len(got), prefill_s=r["prefill_s"], decode_s=r["decode_s"], wait_s=r["wait_s"],
                           seg_s=r["seg_s"], steps=r["steps"], acc=r["acc"], src=r["src"],
                           mean_acc=sum(r["acc"]) / max(1, len(r["acc"])),
                           **{k: r[k] for k in ("b2", "chain") if k in r})
            else:
                got, r = SD.greedy_generate(model, e["ids"], args.new, eos)
                rec.update(generated=len(got), prefill_s=r["prefill_s"], decode_s=r["decode_s"])
            rec.update(tok_s=(len(got) - 1) / rec["decode_s"] if rec["decode_s"] > 0 else None, tokens=got,
                       peak_gb=round(mx.get_peak_memory() / 1e9, 2))
            mx.clear_cache()
            mx.reset_peak_memory()
            outs[c].write(json.dumps(rec) + "\n")
            outs[c].flush()
            short = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                     if k in ("name", "config", "generated", "decode_s", "tok_s", "steps", "mean_acc", "peak_gb")}
            print(json.dumps(short), flush=True)
    modes.reset()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f} s", flush=True)
