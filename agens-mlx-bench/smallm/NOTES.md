# Small-M quantized matmul on the M5 neural accelerators: spike results (2026-10-18)

M5 Pro (20-core GPU, 307 GB/s), MLX 0.32.4, macOS 26.5, Metal toolchain 32023.883.
Synthetic random weights (the bytes and shapes are what matter for speed).

## Go/no-go

**INT4 g128: go, at the top of the 1.1–1.25× band, not at 1.1×.** In the full Volundr step
(`VERIFY_STEP.md`), an 8-token verify costs **1.25× a decode step** (83.3 ms against 144.5 ms on
stock MLX). Its matmuls alone cost 1.19× decode. At 16 tokens the step costs 1.33×.

The kernel-only tables below say about 1.07×, because they compare against an *isolated* stock
decode kernel at 242–275 GB/s. In the real 72-layer chain, stock decode streams at about 293 GB/s
(back-to-back kernels overlap). This kernel stays at its device-operand ceiling of about 247 GB/s.
Closing that gap is the main lever left (see "Limits"). Revised INT4 projection: about 27/36/48
tok/s (chat/code/agent), down from 31/41/54.

Kernel-only, on the clean run (cost for M=1..16 against an isolated 1-token stock decode):

| Shape | M=1..16 cost |
|---|---|
| gate_up | 1.01–1.06× |
| down | 1.10–1.17× |
| kda_qkv | 1.01–1.04× |
| out | 1.02–1.10× |

Byte-weighted, that is about 1.07× (1.06–1.10× across runs). Stock `mx.quantized_matmul` costs
2.5–2.9× at M=8 and 1.5–4.8× at M=16. The input prep is fused in `prep_nax.py` and costs about
2 ms per step.

**2- and 3-bit: not yet.** Prototype in `qmm_nax_lowbit.py`, measured at M=16:

- With int8 activations: 1.07–1.36×.
- With bf16 activations: 1.24–1.62×.

(Clean run, best configuration per shape.)

At 3 bits and below, the accelerator's own compute for 16-token tiles comes close to the weight
streaming time (see "Limits"). Reaching 1.1× needs int8 activations and a cheaper unpack. These
are kernel-only ratios against an isolated decode. Going by the INT4 result, the full step will
cost more. The 2.82-bit build's projection (42–73 tok/s, which assumed a 1.1× verify) should use
at least 1.3–1.4× until that work is done.

## INT4 table: time(M) / time(stock M=1)

Kernel-only timings from `bench.py`, re-run on a quiet GPU after the parent session finished
benchmarking. The bf16 8192³ matmul canary read 25.6–26.6 TFLOP/s, and no idle waits were
triggered (`results_4bit_clean.md`).

The two earlier runs (`results_4bit.md`, `results_4bit_rerun.md`) used the idle wait and
best-of-7 rounds while other GPU jobs ran on and off. Their ratios agree within ±0.05.
The stock M=1 baseline varies 242–275 GB/s between runs even on a quiet GPU.

Row key:

- **stock**: `mx.quantized_matmul`.
- **nax**: this kernel, with x already prepped.
- **nax+prep**: prep done per call as 3 separate MLX ops.
- **nax-i8**: int8 activations.

gate_up 34816×5120 (stock M=1: 0.379 ms (250 GB/s); nax M=16: 0.398 ms (238 GB/s))

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 1.00 | 0.96 | 1.04 | 1.25 | 1.50 | 2.31 | 2.76 | 2.93 | 3.36 | 3.57 | 4.43 | 4.56 | 1.73 | 1.68 | 1.69 | 1.55 |
| nax | 1.02 | 1.01 | 1.02 | 1.03 | 1.01 | 1.01 | 1.02 | 1.04 | 1.04 | 1.04 | 1.05 | 1.05 | 1.06 | 1.05 | 1.03 | 1.05 |
| nax+prep | 1.02 | 1.03 | 1.02 | 1.04 | 1.02 | 1.05 | 1.05 | 1.05 | 1.04 | 1.05 | 1.06 | 1.06 | 1.08 | 1.08 | 1.05 | 1.04 |
| nax-i8 | 0.98 | 1.00 | 0.98 | 0.99 | 1.00 | 1.03 | 1.00 | 1.00 | 1.03 | 1.02 | 1.01 | 1.01 | 1.03 | 0.99 | 1.01 | 1.01 |

down 5120×17408 (stock M=1: 0.188 ms (252 GB/s); nax M=16: 0.216 ms (220 GB/s))

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 1.00 | 1.02 | 1.12 | 1.32 | 1.54 | 2.06 | 2.41 | 2.56 | 2.83 | 3.06 | 3.57 | 3.73 | 4.46 | 4.54 | 4.72 | 4.43 |
| nax | 1.14 | 1.17 | 1.15 | 1.17 | 1.14 | 1.14 | 1.16 | 1.17 | 1.11 | 1.16 | 1.12 | 1.16 | 1.11 | 1.14 | 1.10 | 1.15 |
| nax+prep | 1.35 | 1.35 | 1.35 | 1.34 | 1.36 | 1.33 | 1.39 | 1.37 | 1.33 | 1.35 | 1.34 | 1.35 | 1.35 | 1.36 | 1.35 | 1.28 |
| nax-i8 | 1.03 | 1.06 | 1.05 | 1.04 | 1.06 | 1.09 | 1.09 | 1.04 | 1.06 | 1.04 | 1.04 | 1.07 | 1.04 | 1.03 | 1.05 | 1.06 |

kda_qkv 10240×5120 (stock M=1: 0.114 ms (245 GB/s); nax M=16: 0.119 ms (234 GB/s))

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 1.00 | 0.96 | 1.00 | 1.11 | 1.27 | 1.95 | 2.39 | 2.58 | 2.99 | 3.22 | 3.97 | 4.31 | 1.67 | 1.66 | 1.68 | 1.54 |
| nax | 1.01 | 1.01 | 1.02 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 | 1.02 | 1.03 | 1.02 | 1.02 | 1.03 | 1.03 | 1.04 | 1.04 |
| nax+prep | 1.10 | 1.11 | 1.12 | 1.12 | 1.11 | 1.13 | 1.12 | 1.12 | 1.12 | 1.11 | 1.12 | 1.12 | 1.13 | 1.12 | 1.13 | 1.09 |
| nax-i8 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.01 | 0.99 | 1.01 | 1.01 | 1.00 | 1.00 | 1.02 |

out 5120×6144 (stock M=1: 0.069 ms (244 GB/s); nax M=16: 0.073 ms (230 GB/s))

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 1.00 | 0.98 | 1.05 | 1.16 | 1.32 | 2.00 | 2.36 | 2.52 | 3.03 | 3.23 | 3.75 | 4.00 | 4.82 | 4.84 | 4.75 | 4.79 |
| nax | 1.06 | 1.02 | 1.09 | 1.05 | 1.02 | 1.04 | 1.07 | 1.07 | 1.06 | 1.05 | 1.07 | 1.02 | 1.10 | 1.07 | 1.09 | 1.06 |
| nax+prep | 1.24 | 1.22 | 1.24 | 1.25 | 1.21 | 1.27 | 1.22 | 1.23 | 1.25 | 1.25 | 1.25 | 1.24 | 1.22 | 1.27 | 1.24 | 1.15 |
| nax-i8 | 1.00 | 1.01 | 1.01 | 1.04 | 1.02 | 0.99 | 1.00 | 1.03 | 1.02 | 1.03 | 1.04 | 1.01 | 1.00 | 1.01 | 1.01 | 1.02 |

## Correctness (`check.py`, `check_lowbit.py`)

Errors are measured against an fp32 reference `x @ mx.dequantize(w).T`, for M = 1..16 on all
four shapes:

| Path | Max relative error | RMS relative error | Notes |
|---|---|---|---|
| Stock | 3.7e-3 | 2.2e-3 | |
| nax bf16 | 3.7e-3 | 2.2e-3 | Equal to stock's 1-token path. Stock's large-M path reaches 1.7e-3 RMS at M=16. |
| nax-i8 | 8.4e-3 | 7.0e-3 | Per-(row, group) int8 activations; about 3× the bf16 error. |
| 2/3-bit kernel, bf16 | Same as stock | | |
| 2/3-bit kernel, int8 | 6–8e-3 | | |

## How the INT4 kernel works (`qmm_nax.py`)

- `mx.fast.metal_kernel` can `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
  and call `mpp::tensor_ops::matmul2d`. This is undocumented, but MLX compiles with Metal 4. No
  C++ extension is needed.
- MPP accepts MLX's packed 4-bit weights unchanged, as a `metal::uint4b_format` device tensor (the
  right operand, transposed). The nibble order matches, so no dequantization happens. For each
  128-group the tensor op computes `D = x_g @ q_g^T`. The epilogue then adds
  `s[n,g]*D + b[n,g]*sum(x_g)` to the accumulator.
- Tiles: 16 tokens (padded) × 64 weight rows per simdgroup. 8 simdgroups per threadgroup split the
  K groups, interleaved, and reduce through 32 KB of threadgroup memory.
- int8 variant: x is quantized per (row, group). The weights are repacked once to signed int4 with
  `w ^ 0x88888888`, and `8*s` is folded into the bias (`to_int4`). It then uses int8 × int4 tensor
  ops, which are about 1.5× faster on the accelerator.

## Limits found (`probe_rate.py`, `probe_rate_t.py`, `variants.py`)

- **The accelerator's minimum tile is 16 tokens.** 8-token tiles cost the same as 16, in both
  operand orientations. So any M from 1 to 16 costs the same, and drafting up to 15 tokens (or a
  16-node tree) is free on the matmul side.
- **Rates for 16-token tiles with cache-resident weights:**

  | Activations × weights | TFLOP/s |
  |---|---|
  | bf16 × u4 | 21 |
  | f16 × u4 | 25 |
  | bf16 × int8 | 23 |
  | int8 × int8 | 29 |
  | int8 × int4 | 33 |

  The full 32B model (2.75e10 linear weights) at 16 tokens needs 8.8e11 FLOP. That is 41 ms
  (bf16) or 27 ms (int8 × int4) of pure accelerator time, against 55 ms of INT4 weight streaming.
  It overlaps well enough at 4 bits. At 2–3 bits the streaming time shrinks to 30–40 ms and
  compute becomes the bottleneck.
- **Device-memory 4-bit operands top out around 235–250 GB/s.** That is below plain loads with the
  same tiling (up to 283 GB/s), so MPP's own load path is part of the remaining gap. These did not
  help, or made things worse:
  - touching the next tile with plain loads as a prefetch;
  - staging weights in threadgroup memory;
  - keeping x in a cooperative tensor;
  - fp16 activations, once streaming;
  - 128-byte steps;
  - a contiguous K split.
- MPP has no cooperative (register) form for 4-bit operands. It does have one for int8, which is
  what the 2/3-bit kernel uses.

## 2/3-bit prototype (`qmm_nax_lowbit.py`, `bench_lowbit.py`)

- Each lane unpacks its share of the tile into an int8 cooperative tensor. The weights are
  bit-repacked offline (same byte count), so one shift and one mask produce four int8 codes. The
  codes are centred (q − 2^(bits−1)) and the offset is folded into the bias.
- The layout comes from `get_multidimensional_index` (`probe_layout.py`). It is
  implementation-defined, so a real port needs a startup self-test.

M=16 cost / stock M=1 decode at the same bit width, for the best configuration (32 rows; KS=1 for
K=5120, KS=8 for down):

| bits | x | gate_up | down | kda_qkv | out | stock M=16 |
|---|---|---|---|---|---|---|
| 2 | int8 | 1.24 | 1.07 | 1.25 | 1.36 | 2.2–8.4 |
| 2 | bf16 | 1.41 | 1.24 | 1.55 | 1.62 | |
| 3 | int8 | 1.20 | 1.11 | 1.32 | 1.34 | 2.0–6.2 |
| 3 | bf16 | 1.51 | 1.27 | 1.57 | 1.57 | |

Clean run: `clean_2bit.log`, `clean_3bit.log`. An earlier run under intermittent load agreed within
about ±0.1.

- 2-bit with int8 activations sits near the accelerator floor (about 1.05×).
- 3-bit is below its floor, held back by unpack ALU work and register pressure (64-row tiles spill).
- Writing unpacked words straight into the cooperative tensor's storage (`PTRWRITE 1`) is also
  correct. In a cache-warm test it was 28% faster with bf16 x and 21% slower with int8 x. It is not
  in the table above.

## Prep

The kernel needs x padded to 16 rows plus per-group sums (and, for int8, per-group codes and
scales). Done as 3 separate MLX ops per call, this adds 0.01–0.25× on its own, worst on the small
shapes. In the port it belongs in the RMSNorm, or in one small kernel. Its result is shared by
matmuls with the same input (q/k/v/gates, gate/up).

## Measurement notes

- Weights rotate through ≥1.5 GB of copies and K calls are queued per eval (as in `../micro.py`).
  Candidates are timed interleaved, best of N.
- `common.wait_gpu_idle()` holds a lock file in the shared notes folder and pauses while another
  venv Python job runs. Another session's benchmarks were skewing timings by up to 25%.

## Next steps

1. Done in `VERIFY_STEP.md`: prep is fused, the kernel is in `volundr_synth.py`'s verify step, and
   the 8-bit lm_head goes through `qmm_nax_u8.py`. Result: 1.25× at T=8, 1.33× at T=16.
2. Raise the kernel's streaming rate from about 247 GB/s towards the about 293 GB/s stock decode
   gets in-chain. This is about 11 ms of the 16 ms excess at T=8.
3. 2/3-bit: cheaper unpack. Options are unpacking 3-bit to int4 nibbles in threadgroup memory for
   the faster int8 × int4 rate, or splitting tiles to cut registers.
4. Maybe move to an MLX C++ extension later, to cut `metal_kernel` CPU dispatch cost. This is not
   needed for GPU throughput.
