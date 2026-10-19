"""Teacher-forced logit dump in the agens-engine format (tools/compare_logits.py, AGENS_M1_LOGITS_DUMP).

prompts.json is the `agens m1` input: [{"name", "ids": [prompt ids], "force": [continuation ids]}, ...].
For each prompt, exactly as `agens m1` does with a "force" list: prefill(ids) -> one row, then decode(t) for every
t in force[:-1] -> one row each. A row is the bf16 logits of the first VOCAB_VALID (248077) ids, little-endian,
appended to --out. --tokens-out gets the greedy pick per row in the `agens m1` JSON layout
({"name", "n_prompt", "tokens", "force"}), which tools/compare_tokens.py reads.

usage: python dump_logits.py --model models/volundr-int4-mlx --prompts prompts.json --out mlx.bin
       [--tokens-out mlx.json] [--chunk 4096]
then:  python ref/agens-engine/tools/compare_logits.py engine.bin mlx.bin mlx
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import load, prefill, set_wired_limit  # noqa: E402


def row_bytes(logits: mx.array, vv: int) -> bytes:
    return np.array(logits[:vv].astype(mx.bfloat16).view(mx.uint16)).astype("<u2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-out", default=None)
    ap.add_argument("--chunk", type=int, default=4096, help="prefill chunk (the engine's default is 4096)")
    args = ap.parse_args()

    model, _ = load(args.model)
    set_wired_limit()
    vv = model.args.vocab_valid
    prompts = json.loads(Path(args.prompts).read_text())
    Path(args.out).write_bytes(b"")
    results = []
    with open(args.out, "ab") as f:
        for p in prompts:
            cache = model.make_cache()
            logits = prefill(model, cache, p["ids"], args.chunk)
            rows = [logits]
            f.write(row_bytes(logits, vv))
            for t in p["force"][:-1]:
                logits = model(mx.array([[t]], dtype=mx.int32), cache=cache, last_only=True)[0, -1]
                mx.eval(logits, [c.state for c in cache])
                f.write(row_bytes(logits, vv))
                rows.append(logits)
            toks = [int(mx.argmax(r[:vv]).item()) for r in rows]
            same = sum(a == b for a, b in zip(toks, p["force"]))
            print(f"{p.get('name', '?')}: {len(p['ids'])} prompt tokens, teacher-forced top-1 agreement "
                  f"{same}/{len(p['force'])}", file=sys.stderr)
            results.append({"name": p.get("name", "?"), "n_prompt": len(p["ids"]), "tokens": toks, "force": p["force"]})
    if args.tokens_out:
        Path(args.tokens_out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
