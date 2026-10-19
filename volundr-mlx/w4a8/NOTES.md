# W4A8 prefill matmul spike (2026-10-18)

M5 Pro (20-core GPU), MLX 0.32.4, Volundr 32B INT4 g128 (`../models/volundr-int4-mlx`).

## Verdict

**Go as an opt-in prefill mode (`VOLUNDR_W4A8=1`), no-go as the default.** Final config: every 4-bit linear
uses W4A8 except the attention k_proj / v_proj, which stay on the stock path (`VOLUNDR_W4A8_SKIP`, default
`k_proj,v_proj`).

- **Speed.** The GEMMs are 1.38–1.46× faster than `mx.quantized_matmul` (35–41 against 25–29 TFLOP/s).
  End-to-end prefill is +25% to +32% (two interleaved A/Bs on 20K tokens).
- **Numerics.** Parity KL mean 9.2e-4 overall, 1.5e-3 / 2.5e-3 on the 14K / 6K prompts, p99 1.1e-2. That is
  inside the proposed gate, but still 2–3× the drift the exact kernels caused. The per-matmul error is 6×
  stock's.

## How it works (`w4a8.py`)

```
y = sum_g  s[n,g] * (sx[m,g] * D_g[m,n] + d8[m,g])  +  b[n,g] * xs[m,g]
D_g = sum_{k in g} xq[m,k] * q[n,k]         int8 x int8 -> int32 on the accelerator, 32x32x32 tensor ops
xq  = rint(x / sx), sx = absmax / 127      per (row, 128-group), act_quant
xs  = exact fp32 group sum; d8 = 8 * (xs - sx * sum xq)
```

- **Weights.** They stay in MLX's packing. Codes are unsigned (0..15 fits int8). `(w >> 4*sel) & 0x0F0F0F0F`
  gives four int8 codes per register word.
- **Activations.** `act_quant` permutes each 8-chunk of activations to (0,2,4,6,1,3,5,7) so it lines up with
  those codes.
- **Centring.** The `d8` term centres the codes (q − 8) without touching the weights.
- **Layouts.** `act_quant` writes `xq` and the per-group params lane-major: every lane loads its 32×32 A tile
  with two 16-byte loads.
- **Default kernel (`32x32x2x2RV`).** One 32×32 tile per simdgroup, 2×2 simdgroups. A comes from the
  lane-major layout. B is gathered per lane (eight 4-byte loads per k step). Both are prefetched one k step
  ahead. The int32 partials are scaled into fp32 once per group.
- **Cooperative layout.** The int8 32×32×32 layout is MLX's 16×16 NAX fragment layout, with frags ordered
  n-first for the right operand (`probe_layout.py`). It is implementation-defined. `check.py` compares the
  kernel with a CPU emulation of the same arithmetic: it matches to bf16 output rounding (1.66e-3 RMS) in
  every variant.
- **Hook (`w4a8_hook.py`).** It routes calls with at least 64 rows of bf16 input on the GPU. Decode and the
  short prompts keep the stock path. Linears reading the same input array share one `act_quant`.

## Measurements

GEMM, M = 2048, best of 7 interleaved, weights rotated through ≥ 256 MB. Canary 25.3 → 28.3 TFLOP/s
(`final_shapes.log`).

| shape N×K | stock | W4A8 | × |
|---|---|---|---|
| gate+up 34816×5120 | 27.5 | 38.5 | 1.40 |
| gate/up 17408×5120 | 27.0 | 39.4 | 1.46 |
| down 5120×17408 | 25.4 | 35.9 | 1.41 |
| q_proj 12288×5120 | 28.2 | 39.7 | 1.41 |
| kda qkv 10240×5120 | 28.7 | 40.0 | 1.39 |
| z 6144×5120 | 28.7 | 40.1 | 1.40 |
| out 5120×6144 | 28.4 | 40.3 | 1.42 |

`act_quant` takes 0.16–0.20 ms (K = 5120/6144) and 0.45 ms (K = 17408). It runs about 4× per layer
(shared inputs are quantized once).

End to end: `bench_prefill.py`, 20480 tokens, chunk 2048, interleaved A/B ×2. Decode does not use the W4A8
path; its spread is noise.

| run | config | canary | bf16 prefill | W4A8 prefill | gain | decode at 20K (bf16 / W4A8) | peak |
|---|---|---|---|---|---|---|---|
| `e2e1.log` | all linears | 26.8 → 24.3 | 399 / 390 tok/s | 490 / 509 tok/s | +25% | 79–81 / 80–87 ms | 28.5 GB |
| `e2e_final.log` | k/v on stock | 23.4 → 23.1 | 328 / 337 tok/s | 440 / 444 tok/s | +32% | 81–84 / 75–78 ms | 28.5 GB |

The final run had other apps loading the machine (two renderer processes at ~100% CPU), so its absolute rates
are low on both sides. Moving k/v to stock costs about 7 ms per chunk on paper (18 attention layers × 2 small
GEMMs at the stock rate).

Parity: against `parity/mlx.bin` (the bf16 path), 240 teacher-forced positions, dump chunk 4096
(`dump_w4a8.log`, `dump_w4a8_kv.log`).

| prompt | all linears (`mlx_w4a8.bin`) | k/v on stock (`mlx_w4a8_kv.bin`, final) |
|---|---|---|
| all 240: top-1 / mean / p99 / max | 238/240, 1.11e-3, 1.40e-2, 4.82e-2 | 238/240, 9.19e-4, 1.11e-2, 5.81e-2 |
| code-think (80 tok): mean / max | 6.7e-4 / 7.2e-3 | 6.3e-4 / 6.0e-3 |
| long-code-6k: mean / max / prefill row | 3.24e-3 / 4.8e-2 / 4.3e-3 | 2.46e-3 / 5.8e-2 / 1.7e-3 |
| long-code-14k: mean / max / prefill row | 1.66e-3 / 1.3e-2 / 6.8e-3 | 1.51e-3 / 1.2e-2 / 2.3e-3 |
| chat / zh (< 64 tok, path unused) | 0 | 0 |

For scale:

- Proposed gate: mean < 5e-3, p99 < 5e-2.
- The exact prefill kernels against phase 1: mean 8.0e-4, p99 6.7e-3.
- INT4 against bf16 weights: mean 4.2e-2.

Per-linear error on real activations (`error_real2.log`):

- **Setup.** One 2048-token chunk, 45 linears in layers 0/4/35/40/70/71. Reference: an fp32 CPU product with
  exact fp32 dequantization.
- **Stock vs W4A8.** Stock averages 1.9e-3 relative RMS. W4A8 averages 1.15e-2, 2.7–12× stock's per linear.
  The worst are qkv/z/v_proj and up_proj in the middle layers, at 1.8–2.5e-2, where outlier channels widen
  the int8 step.
- **Finer activation groups (CPU emulation).** 64-wide groups give 8.6e-3 and 32-wide 6.5e-3. Both were built
  and are slower than stock (see the addendum).

## k_proj / v_proj on the stock path (final config)

k_proj and v_proj (N = 1024) are among the worst-hit linears (v_proj 2.3e-2 in layers 35/40) and among the
cheapest. Keeping them on stock gives:

- **Overall KL:** mean −17% (1.11e-3 → 9.2e-4), p99 −21% (1.40e-2 → 1.11e-2).
- **Long prompts:** −24% (6K) and −9% (14K).
- **Prefill rows:** the logits right after the prompt improve the most, 2.5–3× (6K 4.3e-3 → 1.7e-3, 14K
  6.8e-3 → 2.3e-3).
- **Max:** it moves 4.8e-2 → 5.8e-2, at one near-tie position of the 6K prompt.

`VOLUNDR_W4A8_SKIP=""` puts them back on W4A8, and a comma list (leaf module names) skips others.

## What did not work (bench logs kept)

- **The v1 kernel.** It gathered both operands per lane (`sweep1.log`) and reached 8–14 TFLOP/s (0.3–0.5× stock).
- **Ablations (`ablate*.log`).**
  - The accelerator reaches 61.5 TFLOP/s int8 (31.5 bf16) with operands in registers (`probe_rate.py`); the
    full kernel with its loads ablated reaches 65.
  - Removing B loads or A loads one at a time was not additive. Making A lane-major plus prefetching one k
    step got the kernel to 40.
  - A cache-hot B (`B_HOT`) did not help: the remaining limit is in-core (L1 bandwidth or occupancy), not DRAM.
- **Variants that were slower:**
  - B staged in threadgroup memory (`t`).
  - Row-chunk B loads (`BROW`, with act_quant perm 1: 4 × 16-byte loads per group). These tie the default
    when combined with transposed scales.
  - Vector B loads (`V` off).
  - Prefetch depth 2 or 4.
  - MLX-style plain register tiles (`PLAIN`, `L` in bench_w4a8.py).
  - Any per-simdgroup tile above 32×32 (64×32 and 32×64 at 0.4–0.8×). fp32 accumulators plus int32 group
    partials cost 2 registers per output.
  - Finer activation groups (addendum).
- **MPP limits.** With both operands cooperative, M, N and K must each be 16 or 32. Cooperative tensors
  cannot be put in arrays, so the generator unrolls named tiles.

## Next levers

1. **Fuse `act_quant` into the RMSNorm, silu_mul and gated-norm kernels.** That saves about 0.1 s per 2048
   chunk, about 2–3%.
2. **The GEMM sits at about 40 of 61 TFLOP/s, bound in-core.** Bigger register tiles would need fewer than
   2 registers per output. Check occupancy and spills in Xcode's shader profiler before trying more
   variants.
3. **More accuracy.** Skip more linears by name, for example `in_proj_qkv` and `in_proj_z` in the KDA
   layers: the next-worst-hit linears, 14% and 8% of a KDA layer's 4-bit weights. Measure the KL against the
   speed cost first.

## Addendum: 32- and 64-wide activation groups (2026-10-18)

**Verdict: no-go. Keep the 128-wide groups.** Finer groups lose the whole speed gain.

- **Cost.** 64-wide groups run at 0.89–1.01× stock. The best 32-wide variant runs at 0.41–0.44×.
- **Comparison.** The 128-wide default runs at 1.38–1.45×.
- **Skipped.** The brief's condition for the parity and end-to-end runs (≥ 1.2× stock) is not met, so neither
  ran.

**What was built (`w4a8.py`; `check.py` matches the kernels to the CPU emulation at bf16 rounding):**

- **`act_quant(ag=32 / 64)`.** With the 8-chunk layout (perm 0), lanes 8j..8j+7 hold logical k 32j..32j+31,
  which is exactly the GEMM's k step j. Each 32-k or 64-k sub-group gets its own scale (absmax over 8 or 16
  lanes).
- **Params per (row, group).** `(sx_0..sx_3)` and `(d8, xs)`, where d8 = 8·(xs − Σ_j sx_j·Σxq_j).
- **GEMM `ag` modes:**

  | ag | what |
  |---|---|
  | 1 | fp32 per-group partial, a fresh int32 tile per k step |
  | 2 | scale straight into the accumulator |
  | 3 | as 2, with two result tiles so each step's scaling overlaps the next op |
  | 4 | as 2 with the fewest live registers (per-step scale loads; bias params after the loop) |
  | 5 | 64-wide groups on the lean path (the tile accumulates over two k steps) |

**GEMM, M = 2048, TFLOP/s and × stock** (`ag_shapes.log`, canary 26.8 → 24.4):

| N×K | stock | 128-wide (default) | 64-wide (ag 5) | 32-wide lean (ag 4) | 32-wide partial (ag 1) |
|---|---|---|---|---|---|
| 34816×5120 | 28.4 | 40.9 (1.44) | 26.9 (0.94) | 12.3 (0.43) | 9.2 (0.32) |
| 17408×5120 | 27.6 | 40.1 (1.45) | 24.7 (0.89) | 12.1 (0.44) | 9.0 (0.33) |
| 5120×17408 | 25.6 | 36.7 (1.43) | 25.9 (1.01) | 11.1 (0.43) | 7.8 (0.30) |
| 12288×5120 | 29.0 | 39.9 (1.38) | 26.8 (0.92) | 12.0 (0.41) | 9.0 (0.31) |
| 10240×5120 | 28.6 | 40.8 (1.43) | 27.0 (0.94) | 12.2 (0.42) | 9.0 (0.31) |
| 6144×5120 | 28.5 | 40.4 (1.42) | 25.8 (0.91) | 12.4 (0.44) | 8.9 (0.31) |
| 5120×6144 | 28.3 | 40.0 (1.42) | 25.6 (0.91) | 12.0 (0.42) | 9.0 (0.32) |

`act_quant(ag=32)` costs the same as the 128-wide version: 0.17–0.21 ms at K = 5120/6144, 0.54 ms at K = 17408.

**Why** (`ablate.py`, gate shape 17408×5120):

- **Reading the int32 result tile mid-chain is expensive.** On the 128-wide kernel, adding only a read after
  every op (one FMA per element) goes from 9.2 to 16.0 ms. A read after every other op gives 11.9 ms. Those are
  the floors for 32- and 64-wide groups before any scaling work.
- **Overlap does not help.** Two result tiles (ag 3, 58 ms) are no faster than one (ag 2, 58 ms). The
  non-accumulating op mode against accumulate-plus-zeroing makes no difference (39–60 ms everywhere). So the
  cost is not the latency of one op: reading a tile seems to wait for the simdgroup's outstanding ops.
- **The rest is register pressure.** Keeping the four sub-group scales, the bias params and a 32-float partial
  live through the k loop (ag 1/2) costs 41–58 ms. The lean variant (ag 4) gets that down to 30 ms. This is the
  same effect that makes per-simdgroup tiles above 32×32 collapse.

**Finer scales inside the GEMM** would need a design that does not read the tile per sub-group, for example
one int32 tile per sub-group read once per group. That costs 32–96 more registers per lane, which this
accelerator's 32×32 tiles do not have to spare. Keeping k_proj / v_proj on stock (above) was the cheaper
accuracy fix and is in the final config.

## Files

| file | what |
|---|---|
| `w4a8.py` | act_quant, GEMM generator (default + switches), sb_transpose, PLAIN variant, qmm |
| `w4a8_hook.py` | W4A8Linear (rows ≥ 64, bf16, GPU, N % 64 == 0; shared act_quant per input), enable/disable; `SKIP` / `VOLUNDR_W4A8_SKIP` (default `k_proj,v_proj`) |
| `../volundr.py` | `VOLUNDR_W4A8=1` → `enable_w4a8(model)` on the first call (the only change outside this folder) |
| `check.py` | kernels against a CPU fp32 emulation (128-, 64- and 32-wide activation groups) and the fp32 product |
| `bench_w4a8.py`, `ablate.py`, `probe_rate.py`, `probe_layout.py` | GEMM timing (config flags in `parse`), ablations (`--ag` for the sub-group variants), accelerator peak, cooperative layouts |
| `error_real.py`, `kl_by_prompt.py`, `run_e2e.sh` | real-activation error, per-prompt KL, end-to-end A/B |
| `w4common.py` | lock-aware timing (reuses agens-mlx-bench/smallm/common.py; skips the in-process wait when an ancestor holds gpu.lock) |
