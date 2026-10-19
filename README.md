# Volundr MLX

An MLX port of [Agens Volundr 32B](https://huggingface.co/Blockway/Agens-Volundr-32B-v1-INT4) (Blockway) for Apple
silicon. The model is a hybrid: 54 Kimi Delta Attention layers, 17 BCSA sparse-attention layers and one dense
attention layer, with Engram n-gram memory and 4-stream mHC residuals. The upstream engine is CUDA-only
([agens-engine](https://github.com/BlockWayz/agens-engine)). Everything here was built and measured on an M5 Pro
(64 GB, 20-core GPU) with MLX 0.32.4.

Unofficial; not affiliated with Blockway.

| Folder | What it is |
|---|---|
| [`volundr-mlx/`](volundr-mlx/README.md) | Converter, model, prefill kernels, opt-in W4A8 prefill ([`w4a8/`](volundr-mlx/w4a8/NOTES.md)), DFlash2 speculative decoding ([`spec/`](volundr-mlx/spec/NOTES.md)), parity tooling ([`PARITY.md`](volundr-mlx/PARITY.md)) |
| [`agens-mlx-bench/`](agens-mlx-bench/) | Synthetic Volundr-shaped benchmarks, and the small-batch 4-bit matmul on the M5 GPU neural accelerators used for speculative verify ([`smallm/`](agens-mlx-bench/smallm/NOTES.md)) |

## Measured on the M5 Pro (INT4 weights)

| | |
|---|---|
| Decode, greedy | ~13.5 tok/s |
| Decode, DFlash2 speculative (greedy, 84 held-out prompts) | 24.0 tok/s (1.77×), 2.63 tokens per verify step |
| Prefill, exact | 374 tok/s on a 19K-token prompt |
| Prefill, opt-in W4A8 (`VOLUNDR_W4A8=1`) | 490–509 tok/s at 20K, +25–32% |
| Peak memory | ~25 GB short context, ~29 GB at 19K |

Per-category speculative results, methods and caveats are in the linked NOTES files.

## Setup

The scripts expect a Python 3.12 virtualenv at `.venv` in this folder. To use another interpreter, set `PY=...`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd volundr-mlx
../.venv/bin/hf download Blockway/Agens-Volundr-32B-v1-INT4 --exclude "assets/*"
../.venv/bin/python convert.py --hf-path Blockway/Agens-Volundr-32B-v1-INT4 --mlx-path models/volundr-int4-mlx
../.venv/bin/python generate.py --model models/volundr-int4-mlx --prompt "Hello" --max-tokens 256
../.venv/bin/python tests/test_volundr.py
```

Benchmarks, the parity tools and the prompt-suite builder read upstream reference files from `volundr-mlx/ref/`,
which isn't tracked. To recreate it, run `volundr-mlx/fetch_ref.sh`.

Model weights aren't included. They're covered by their Hugging Face repos' licenses.

## Status

Experimental:
- Speculative decoding is greedy-only.
- The CUDA-side parity run (PARITY.md section 3) hasn't been done yet.
- Serving through agens-server isn't built.
