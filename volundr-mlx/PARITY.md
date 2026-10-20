# Parity: Volundr MLX vs the CUDA engine

The reference is Agens Engine (github.com/BlockWayz/agens-engine, commit b0e9a95) running the same
INT4 checkpoint (`Blockway/Agens-Volundr-32B-v1-INT4`). Both sides use identical weights. The MLX
converter reproduces every dequantized weight bit for bit (`q * scale`). Any difference is therefore
kernel numerics, not quantization.

There are no shipped prompt sets. We build one on the Mac, run it through both engines teacher-forced,
and compare the logits position by position.

## 1. Format

The format is the engine's own `AGENS_M1_LOGITS_DUMP`, so the comparison scripts in
`ref/agens-engine/tools/` work unchanged.

- **`prompts.json`** is the `agens m1` input: `[{"name", "ids": [...], "force": [...]}, ...]`.
  - `ids` is the chat-templated prompt.
  - `force` is the continuation both engines are fed.
- **Logit dump (`.bin`):** one row per forced position. Each row has `VOCAB_VALID = 248077` bf16
  logits, little-endian and row-major. Rows from all prompts are appended in order.
  - For each prompt: the row after `prefill(ids)`, then one row after `decode(t)` for every `t` in
    `force[:-1]`. That gives `len(force)` rows per prompt.
  - This matches `crates/agens-cli/src/m1.rs` (`run`, the `force` branch).
- **Token JSON** follows `agens m1`'s output layout:
  `[{"name", "n_prompt", "tokens": greedy pick per row, "force"}]`. `tools/compare_tokens.py` reads it.

## 2. On the Mac (MLX side)

```bash
cd volundr-mlx
../.venv/bin/python make_prompts.py --model models/volundr-int4-mlx --out parity/prompts.json
../.venv/bin/python dump_logits.py --model models/volundr-int4-mlx --prompts parity/prompts.json \
    --out parity/mlx.bin --tokens-out parity/mlx.json
```

`make_prompts.py` writes 5 prompts, with `force` set to MLX's own greedy continuation of 48 tokens:

| Prompt | Exercises |
|---|---|
| short chat, thinking off | short context |
| code, thinking on | thinking path |
| Chinese, thinking off | non-English text |
| ~6K-token code prompt | BCSA far branch live, every far block selected |
| ~14K-token code prompt | more far blocks than the 512 budget, so the indexer's top-k decides |

`dump_logits.py` prefills in 4096-token chunks, like the engine's default `CHUNK`. Its stderr
reports teacher-forced top-1 agreement against `force`. Against the greedy continuation of the same
code this is `len(force)/len(force)`. The shipped `force` comes from phase 1, so see the end of
section 4.

Copy `parity/prompts.json` to the CUDA box.

## 3. On a CUDA box (reference side)

You need a GPU with at least 32 GB of memory. The engine was measured on 48 GB Ada (sm_89) and on
H100/H200 (sm_90).

**Weights:**

```bash
hf download Blockway/Agens-Volundr-32B-v1-INT4 --local-dir /data/volundr-int4
cd /data/volundr-int4 && sha256sum -c SHA256SUMS
```

**Engine, option A: the release image.** Its entrypoint starts `agens serve`, so override the
entrypoint to reach the `m1` subcommand:

```bash
docker run --rm --gpus device=0 --ipc=host --entrypoint agens \
  -v /data/volundr-int4:/model:ro -v $PWD/parity:/work \
  -e AGENS_M1_LOGITS_DUMP=/work/engine.bin \
  ghcr.io/blockwayz/agens-engine:1.0.0 \
  m1 /model /work/prompts.json /work/engine.json --gpu-gib 40 --wctx 36868
```

First check that the image's `agens` binary was built with the `m1` feature
(`docker run --rm --entrypoint agens ghcr.io/blockwayz/agens-engine:1.0.0 m1`). If it wasn't, use
option B.

**Engine, option B: build from source at b0e9a95** (CUDA 12.9, CMake 3.27+, Rust stable, libtorch
from PyTorch 2.11):

```bash
git clone https://github.com/BlockWayz/agens-engine && cd agens-engine && git checkout b0e9a95
cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;90" \
  -DAGENS_BUILD_ORACLE=ON -DAGENS_TORCH_PREFIX=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
cmake --build kernels/build -j
cargo build --release --features m1 -p agens-cli
rm -f parity/engine.bin
AGENS_M1_LOGITS_DUMP=parity/engine.bin target/release/agens m1 /data/volundr-int4 \
  parity/prompts.json parity/engine.json --gpu-gib 40 --wctx 36868
```

**Rules for the reference run:**

- **Delete `engine.bin` before every run.** The dump appends.
- **Don't pass `--budget-gib`, and don't use the image's `--profile` flags.** Budget mode switches
  on product numerics: INT8 lm_head, KV tiering, prefill chunk 1024, fused far attend. The reference
  is the plain `m1` path, with a bf16 lm_head from the checkpoint and the default prefill chunk of
  4096.
- **Keep `AGENS_M1_BI` and `AGENS_KV_INT8_DENSE` unset.**
- **`--wctx` must cover the longest prompt plus its `force`.** 36868 covers the 14K prompt.
- **Check stderr.** It prints `<name>: teacher-forced top-1 agreement a/b` per prompt. That is
  already the token-level answer: how often the engine's argmax equals MLX's greedy token.

## 4. Compare

```bash
python3 ref/agens-engine/tools/compare_logits.py parity/engine.bin parity/mlx.bin mlx
python3 ref/agens-engine/tools/compare_tokens.py parity/engine.json parity/mlx.json
```

`compare_logits.py` prints top-1 agreement and KL(p_engine ‖ p_mlx) at temperature 1 over the valid
vocabulary (mean, p50, p99, max).

**Proposed gate** (numerics only, same weights):

- mean KL < 5e-3 nats
- p99 KL < 5e-2
- top-1 agreement ≥ 98%

For scale: the INT4 build against the bf16 model scores mean KL 0.0419 and 94.2% top-1 (release
README). A pure-numerics difference should be an order of magnitude below that.

Calibration on the Mac: MLX against itself, with prefill chunks of 2048 instead of 4096, over the
same 240 positions (`parity/mlx_chunk2048.bin`):

| Measure | Value |
|---|---|
| top-1 | 240/240 |
| mean KL | 2.5e-4 |
| p99 KL | 3.3e-3 |
| max KL | 4.7e-3 |

The short prompts fit in one chunk and are identical, so the KL comes from the two long prompts. A
change of prefill chunking alone sits about 20× under the gate. It moves the BCSA attention rounding;
the KDA recurrence is chunk-invariant (NOTES_SESSION.md, "Where the resumed-prefill KL comes from").

The prefill kernels changed the rounding again: NAX attention for BCSA, fused indexer scores, and the
fused KDA conv, norm and recurrence. The phase-1 dump is kept as `parity/mlx_phase1.bin`. Against it,
over the same 240 positions:

| Measure | Value |
|---|---|
| top-1 | 238/240 |
| mean KL | 8.0e-4 |
| p99 KL | 6.7e-3 |
| max KL | 7.7e-3 |

That is about 3× the chunking calibration and still 6× under the gate's mean. The two flips are one
position each in `chat-nothink` and `long-code-6k`, where the current dump scores 47/48 against
phase 1's greedy continuation.

`force` is phase 1's greedy continuation, kept fixed so dumps stay comparable. Phase 1
teacher-forced on it scored 48/48 on all 5 prompts; the current kernels score 47/48 on two of them,
as above. The 14K prompt is
the one to read separately: far-block selection can flip at bf16 ties of indexer scores, which
shows up as isolated high-KL positions, not a raised mean.

## 5. Op-level tests (no CUDA needed)

`python tests/test_volundr.py` runs a tiny random config: hidden 128, layers `kda kda bcsa kda bcsa
attn`, Engram on layer 1, BCSA window 16 / compress 4 / top-3, so the far branch and top-k are live
within 44 tokens. It checks against `tests/reference.py`, a NumPy fp64 port of the release's torch
code (`modeling_volundr.py`, `bcsa.py`, `volundr_extras.py`) and `agens-core/src/reference.rs`.
That port is written independently of `volundr.py`.

| Test | Checks |
|---|---|
| `engram_hash` | uint32 hashing equals `reference.rs` `hash_last` / `hash_seq` (u64 wrap, 61-bit mask, `% rows`), including the hand values in its tests and the carried-over previous ids of a later chunk |
| `sinkhorn_kernel` | the Metal 4x4 Sinkhorn (engine summation order) and the ops fallback equal the fp64 reference within 2e-6 |
| `gated_delta_vec_kernel` | mlx-lm's gated-delta kernel with a per-channel gate equals the KDA recurrence of `kda_decode.cu` (fp64) within 1e-5; T single steps equal one T-step launch bit for bit |
| `convert_linear` | W4/W3/W2 from the int4 container (packed with `reference.rs` `pack_row`), g128 stored as g64: `mx.dequantize` equals `q * scale` exactly; non-repeating scales keep g64; out-of-range values are rejected |
| `topk_tie_order` | far-block top-k on bf16 scores breaks ties toward the lower block index, like torch topk and the engine's radix select |
| `kernels_match_ops` | each Metal kernel against the ops path it replaces: mHC, norms, the KDA conv (both variants bit for bit), the recurrence (staged variant bit for bit), the gated norm, the fused indexer scores (bit for bit), and BCSA prefill attention (far union + window) against fp64 |
| `model_matches_reference_cpu_fp32` | whole model in fp32 on the CPU equals the fp64 reference to 1e-4 (measured ~2e-6), for one-shot prefill, chunked prefill and token-by-token decode |
| `model_gpu_bf16` | GPU bf16 run stays within 1.25x of the bf16 floor (the same model in bf16 on the CPU, without Metal kernels), argmax agreement ≥ 90% |
| `convert_end_to_end` | a fake release folder goes through `convert.py` and `mlx_lm.utils.load_model`. Checks: every quantized linear dequantizes exactly; vision tensors dropped; per-module bits in `config.json`; logits match the dequantized dense model |

The CPU test runs on the CPU because MLX's fp32 GEMM on the M5 GPU is not full fp32 for M ≥ 8: it
measured about 7e-4 relative error, versus 1e-7 at M = 1.

## 6. Known numerics differences from the engine

All are within bf16 noise unless noted.

- **KDA prefill:** MLX runs the exact per-token recurrence (one launch over the chunk). The engine
  uses FLA-style chunked kernels. The engine notes that its chunk boundaries change prefill numerics
  too.
- **Attention:** the dense layer is MLX SDPA vs FlashInfer. BCSA prefill runs one NAX flash-attention
  kernel per 32-query block: the union of the block's selected far blocks, each query masked to its own
  512, then the sliding window, in one online softmax. That means bf16 Q·K products accumulated in fp32,
  `exp2` softmax in fp32 and P·V on the accelerator. It is MLX's own SDPA algorithm, and without far
  blocks it is bit-identical to SDPA. The engine uses bf16 einsums and bf16 softmax weights for the far
  branch.
- **mHC:** the rounding points match `mhc.cu`, but `exp` and division are Metal's, not the PTX
  transcriptions.
- **Far-block selection:** scores are rounded exactly as torch does (bf16 einsum, mean, scale), with
  the same tie order. A score that differs by one bf16 ulp can still swap the 512th block.
