"""Build a teacher-forced prompt set for the CUDA-vs-MLX parity run (`agens m1` input format).

Each entry: {"name", "ids": chat-templated prompt ids, "force": MLX greedy continuation}. The engine then teacher-forces
the same continuation, so both sides produce logits at identical positions. The set covers short chat (thinking on
and off), Chinese, and two long code prompts: ~6K tokens (BCSA far branch live, every far block selected) and ~14K
tokens (more valid far blocks than the top-512 budget, so the indexer's selection matters).

usage: python make_prompts.py --model models/volundr-int4-mlx --out parity/prompts.json [--force 48]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate import greedy_tokens, load, prefill, set_wired_limit  # noqa: E402

M1 = HERE / "ref/agens-engine/kernels/m1/m1_model.cu"


def long_code(tok, n_tokens):
    """The head of m1_model.cu cut to about n_tokens, then a question about it."""
    src = M1.read_text()
    ids = tok.encode(src)
    src = tok.decode(ids[:n_tokens])
    return (f"Here is part of a CUDA source file:\n\n```cuda\n{src}\n```\n\n"
            "Explain in a few sentences what kda_mixer does and which kernels it calls.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", type=int, default=48, help="forced positions per prompt")
    args = ap.parse_args()
    model, tok = load(args.model)
    set_wired_limit()
    prompts = [
        ("chat-nothink", "What is the capital of Sweden, and what is it known for? Answer in two sentences.", False),
        ("code-think", "Write a Python function that returns the n-th Fibonacci number iteratively.", True),
        ("zh-nothink", "用三句话介绍一下长城。", False),
        ("long-code-6k", long_code(tok, 6000), False),
        ("long-code-14k", long_code(tok, 14000), False),
    ]
    out = []
    for name, text, think in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=think)
        cache = model.make_cache()
        logits = prefill(model, cache, ids, 4096)
        force = list(greedy_tokens(model, cache, logits, args.force, set(), model.args.vocab_valid))
        out.append({"name": name, "ids": ids, "force": force})
        print(f"{name}: {len(ids)} prompt tokens -> {tok.decode(force)!r}", file=sys.stderr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
