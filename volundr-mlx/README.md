# Volundr MLX

An MLX port of **Agens Volundr 32B** (Blockway) for Apple silicon: text model, INT4 weights, plain greedy
decoding. It reproduces the CUDA engine ([agens-engine](https://github.com/BlockWayz/agens-engine) at b0e9a95)
and the release's torch reference: `modeling_volundr.py`, `bcsa.py` and `volundr_extras.py`, shipped in the
HF repo.

| File | What it is |
|---|---|
| `convert.py` | compressed-tensors `pack-quantized` checkpoint → MLX safetensors (details below) |
| `volundr.py` | the model, mlx-lm style (`Model`, `ModelArgs`, `make_cache`) |
| `kernels.py` | Metal kernels for the GPU paths (below); `volundr.py` keeps an ops path for each, used on the CPU |
| `generate.py` | greedy generation with timing and peak memory |
| `bench_prefill.py`, `profile_prefill.py` | chunked prefill tok/s per chunk; per-component time of a chunk at any offset |
| `dump_logits.py`, `make_prompts.py` | teacher-forced logit dumps in the engine's format; see [PARITY.md](PARITY.md) |
| `tests/` | op-level and tiny-model tests against a NumPy fp64 port of the torch reference and `reference.rs` |
| `bench.sh`, `with_gpu_lock.sh` | runs under the shared GPU lock, with the canary before and after |
| `ref/` | `agens-engine` at b0e9a95 and the HF repos' code and config files (read-only reference) |

## Use

```bash
hf download Blockway/Agens-Volundr-32B-v1-INT4 --exclude "assets/*"
../.venv/bin/python convert.py --hf-path Blockway/Agens-Volundr-32B-v1-INT4 --mlx-path models/volundr-int4-mlx
../.venv/bin/python generate.py --model models/volundr-int4-mlx --prompt "Hello" --max-tokens 256 [--no-think]
../.venv/bin/python tests/test_volundr.py
```

The conversion takes about 40 s and writes 23 GB.

## Conversion

The W4 linears are repacked without being re-quantized:

- The release stores `weight_packed` I32 [n, k/8] with nibble = q + 8, lowest nibble first. That is MLX's
  affine 4-bit layout, so the words are reinterpreted as uint32.
- `scales = weight_scale` and `biases = -8 * scale` (exact in bf16), so every dequantized weight equals
  `q * scale` bit for bit.

The 16 GB build adds three cases:

- W3/W2 values sitting in the int4 container are repacked to real MLX 3/2-bit, with biases `-4*scale` and
  `-2*scale`. Per-linear types come from `quantization_audit.json` or an RCO assignment file.
- g128 stored as g64 with repeated scales is collapsed back to g128.
- Values outside the declared width are rejected.

Also:

- Dropped: the vision tower and MTP.
- Kept in bf16: norms, mHC statics, `mhc.*.dyn`, Engram, `in_proj_b`, the BCSA indexer, `lm_head` and
  embeddings.
- Stored in fp32: conv1d, `A_log`, `dt_bias`.
- `config.json` gets per-module `quantization` entries, `vocab_valid` (248077) and the indexer shape.

## Model notes

These are the details that matter for numerics, taken from the engine and the torch reference.

- **Residual:** mHC with 4 streams.
  - The layer input is `pre · H`.
  - The update is `H' = res · H + post ⊗ (y − x_in)`, with `res` from 10 Sinkhorn iterations. That runs as a
    Metal kernel in `mhc.cu`'s summation order.
  - The next layer's `dyn` reads the norm of the stream mean.
  - The final norm reads **stream 0 only**.
  - bf16 rounding points follow `mhc.cu`.
- **Norms:**
  - Gemma-style `(1 + w)`: `input/post_attention_layernorm`, `q/k_norm`, `idx_norm` and the final norm.
  - Plain `w`: the mHC and Engram norms.
  - The KDA output norm is `w · rmsnorm(x) · silu(z)`.
  - All run in fp32 with one rounding.
- **KDA:**
  - Fused qkv conv (k = 4, 10240 channels, fp32 taps).
  - l2-normalised q and k, scale 1/√128.
  - Per-channel decay `exp(−exp(A_log) · softplus(f_up(f_down(h)) + dt_bias))`, β = sigmoid(b).
  - The exact per-token recurrence with a per-channel gate (a Metal kernel on the GPU).
  - Value head h uses key head h // 3.
- **BCSA:**
  - Ring of the last 4096 K/V.
  - Mean-pooled K/V and indexer keys appended as each 4-token block closes.
  - Decode: SDPA over the window plus the top-512 valid far blocks.
  - Top-k: block scores are rounded like torch, and ties go to the lower index.
  - Prefill: see the kernels below.
- **Engram:** layers 2 and 22, orders 2 and 3, 2^20 × 512 tables. Hashing uses uint32 on device; only the low
  20 bits of `reference.rs`'s 61-bit hash survive `% 2^20`.
- **Attention:** NeoX RoPE on the first 64 of 256 dims, θ = 1e7. Multimodal RoPE (mrope) reduces to this for
  text. Output is `out · sigmoid(gate)`. Scale is 1/16 everywhere.

## Kernels (`kernels.py`)

All are `mx.fast.metal_kernel`s, used on the GPU in bf16. Each is tested against the ops path it replaces
(`tests/test_volundr.py`, `kernels_match_ops`).

- **BCSA prefill attention** (`far_union`, `bcsa_attend`, `bcsa_window`).
  - Built on MLX's own NAX flash-attention kernel (the head-dim-split `steel_attention_nax`), copied with the
    tile headers inlined.
  - Per 32-query block it first takes the union of the queries' selected far blocks. Those rows are gathered
    from the pooled K/V and run as dense accelerator tiles, each query masked to its own 512.
  - The sliding window follows in the same online softmax.
  - With no far blocks it is bit-identical to MLX's SDPA.
  - Far selections of neighbouring queries overlap little: 32 queries select about 2,000 of 3,500 blocks at 16K
    context. The union stays near 6 × 512 as context grows, so the cost is nearly flat in context length.
- **Far-block selection**:
  - `bcsa_scores`: the indexer scores in one NAX kernel, with torch's rounding points (per-head bf16, fp32
    head mean, bf16, × scale, bf16). Bit-identical to the ops path, and 3–6× faster because the per-head
    scores are never written.
  - `bcsa_topk`: a radix select with the engine's tie order.
- **KDA**:
  - `kda_conv_qk4`: the conv, SiLU and q/k l2 norm, with vectorized loads.
  - `gated_delta_tg` (prefill) and `gated_delta_vec` (decode): the recurrence. The prefill variant stages 8
    timesteps in threadgroup memory. Both are bit-identical to each other.
  - `gated_norm`: the gated output norm.
- **mHC**: the Sinkhorn parameters, the stream mix and the combine, with `mhc.cu`'s rounding points.
- **RMSNorm**: plus the residual add.

## Measured on this M5 Pro (64 GB)

MLX 0.32.4. Warm runs; greedy; bf16 `lm_head` as shipped. Canary 26–27 TFLOP/s before each run (idle).

| Run | Prefill | Decode | Peak memory |
|---|---|---|---|
| 85-token prompt, thinking, 256 new tokens | — | **13.0 tok/s** (77 ms/token) | **24.9 GB** |
| 19,045-token prompt, 192 new tokens | **374 tok/s** (50.9 s) | **12.1 tok/s** (83 ms/token) | **28.5 GB** |
| `bench_prefill.py`, 20,480 tokens in 2048-token chunks | **366 tok/s** (56.0 s; 349–382 per chunk) | | 28.5 GB |

Phase 1, without the prefill kernels, measured 218 tok/s prefill and 76 / 88 ms decode, short and at 19K.

Decode measured kernels-on against ops-only, interleaved in one process (best of 3):

| Context | Kernels | Ops |
|---|---|---|
| 512 | 71.7 ms | 77.6 ms |
| 20K | 79.0 ms | 81.1 ms |

The weights take 24.15 GB resident.

**Where a prefill chunk goes** (`profile_prefill.py`, 2048 tokens at 16K context, synced per piece, canary
25 TFLOP/s): 5.70 s in all, against 10.99 s before the kernels.

- KDA layers: 3.6 s, 66 ms each. About 60 ms of that is the 4-bit matmuls.
- BCSA layers: 1.67 s, 98 ms each.
- Dense layer: 0.11 s.
- mHC: 0.18 s.

**The ceiling.** The 4-bit matmuls are about 4.2 s of every 2048-token chunk. They run at 27–29 TFLOP/s, the
accelerator's rate for 16-bit activations; fp16 activations are no faster. So prefill with exact
(bf16-activation) numerics tops out near 480–500 tok/s. What's left above the matmuls:

- the KDA recurrence: about 5 ms × 54 layers, bound by its 2048 sequential steps;
- mHC;
- small elementwise ops.

Measured accelerator rates at prefill tile sizes, operands in cache:

| Operands | TFLOP/s |
|---|---|
| bf16 × bf16 | ~25–27 |
| bf16 × int4 | ~25–27 |
| int8 × int4 | ~31 |
| int8 × int8 | ~50 |

A W4A8 prefill path is the remaining large lever: int8 activations, with int4 weights unpacked to int8 in
registers. It changes numerics, so it would be an opt-in mode.

Projected 262K-token cache, from the cache layout: dense KV 1.07 GB, BCSA pooled blocks 4.9 GB, window
0.29 GB, KDA state 0.17 GB. That gives about 35 GB peak with a 2048-token prefill chunk. Not measured; a full
262K prefill takes at least 12 min at the 20K speed (the dense layer and the far-block scoring grow with context).

## Not done yet (later phases)

- Speculative decoding with DFlash2. This needs its own loop: mlx-lm refuses non-trimmable hybrid caches.
- Serving through `agens-server`'s `Backend` trait.
- Prefill:
  - a W4A8 matmul path (above);
  - a chunked (tensor-core) KDA recurrence with per-channel decay, which is the engine's form too.
- Decode: the NAX small-M matmul from the spike, and fusing the remaining small ops.
- The CUDA-side parity run ([PARITY.md](PARITY.md)).
