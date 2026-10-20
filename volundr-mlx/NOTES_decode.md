# Decode quick wins: 8-bit lm_head (opt-in), decode small-op fusion, BCSA decode views (2026-10-19)

Status: done (2026-10-19).

## Summary

**Defaults.** The decode fusion and the BCSA views are on by default and bit-exact. The 8-bit lm_head is opt-in
(`VOLUNDR_LMHEAD8`); whether it becomes the default is the user's call.

**Speed.** Interleaved tune A/B, 56 prompts × 4 configs, 256 new tokens:

| config | greedy ms/token | greedy tok/s | spec ms/step | spec tok/s | tokens/step |
|---|---|---|---|---|---|
| main | 74.34 | 13.45 | 110.50 | 23.02 | 2.56 |
| fused (new default) | 72.25 | 13.84 (×1.029) | 107.93 | 23.56 (×1.024) | 2.56 |
| fused + q8 g64 | 67.93 | 14.72 (×1.094) | 104.85 | 24.26 (×1.054) | 2.56 |
| fused + q8 g128 | 67.66 | 14.78 (×1.099) | 105.40 | 24.41 (×1.061) | 2.59 |

- **Part 2 + BCSA views** (main → fused):
  - Greedy −2.09 ms/token, spec −2.57 ms/step.
  - Plain decode −1.5 ms/token at short context, −5.2 at 19K, and −11.6 at 23.5K (that run was noisy).
  - Tokens and per-step acceptance are identical on 56/56 prompts.
- **Part 1, 8-bit head** (fused → q8 g64):
  - Greedy −4.32 ms/token, spec −3.08 ms/step.
  - Plain decode −5.4 ms/token at short context, −5.8 at 19K.
- **8-bit head quality:**
  - KL against bf16, mean / max: g64 1.60e-4 / 2.50e-3, g128 2.12e-4 / 2.62e-3. Top-1 matches 239/240 for both.
  - Outputs identical to the bf16 head: greedy 47/56 (g64) and 44/56 (g128); spec 44/56 and 38/56.
  - Tokens per step are unchanged (2.56 / 2.56 / 2.59). Per-category moves sit on the prompts whose text diverged.

**Held-out test split: a partial sample, not an A/B.**
- **Sample.** `runs/decode-quick-wins_test_frozen.jsonl` covers 45 of 84 prompts (coding through reasoning), with the
  frozen config. It was stopped at the user's request; the greedy half did not run.
- **Result.** 27.44 tok/s, 2.83 tokens/step, 102.6 ms/step.
- **Exactness.** Tokens and per-step acceptance are identical to main's held-out frozen runs (`runs/test_frozen.jsonl`,
  `runs/test_frozen_r2.jsonl`) on 45/45.
- **References from other sessions** on the same prompts. These ran on different code, so they are not an A/B:

  | reference | spec tok/s | this run against it | greedy tok/s |
  |---|---|---|---|
  | main, run 1 | 26.46 | ×1.037 | 13.80 |
  | main, run 2 | 27.82 | ×0.986 | 13.58 |
  | this morning's bcsa-verify-8q branch (has the 8q verify; this branch does not) | 28.44 | ×0.965 | 13.94 |

  Against that branch's greedy run this run is ×1.97. That branch's spec run had 2.82 tokens/step, against 2.83 here.
- **Reading.** The spread runs from ×0.965 to ×1.037, and main's two runs differ from each other by 5%. That is
  session-to-session noise of about 4–6%. The tune split's interleaved A/B is the result.

Branch `decode-quick-wins` (worktree `.claude/worktrees/agent-ae877536cb39838a7`, from main 0732112). M5 Pro 64 GB,
MLX 0.32.4. Logs and scripts: `decode_quick_wins/`; suite runs: `spec/runs/decode-quick-wins_*.jsonl`.

## Part 1: 8-bit lm_head (opt-in)

`VOLUNDR_LMHEAD8=1` (or `64`) quantizes the bf16 lm_head to 8-bit affine with group 64 at load time; `=128` uses
group 128. The default is off: the user decides.
- **Code.** `volundr.lm_head_q_group()` and `Model.quantize_lm_head()`. `generate.load()` and `apply_lm_head_q()`
  apply it. `spec/specdec.py` quantizes after `drafter.attach()`, so the drafter's own 4-bit head copy is still made
  from the bf16 weights. With `--head-bits 0` the drafter keeps the bf16 head it shares. Records carry
  `target_head_q8`.
- **Size.** The head streams 2.54 GB per step in bf16, 1.35 GB at g64 and 1.31 GB at g128.
- **Head alone** (`decode_quick_wins/head.log`):

  | rows | bf16 | q8 g64 | q8 g128 |
  |---|---|---|---|
  | 1 | 9.29 ms | 5.11 ms | 4.96 ms |
  | 8 | 10.08 ms | 7.17 ms | 6.96 ms |

  Stock `quantized_matmul` at 8 rows streams only ~190 GB/s; `smallm/qmm_nax_u8.py` (NAX agent's area) is the lever
  there.
- **Parity** against bf16, `parity/prompts.json`, 240 teacher-forced positions (`decode_quick_wins/cmp_head.log`):

  | head | top-1 | KL mean | p50 | p99 | max |
  |---|---|---|---|---|---|
  | g64 | 239/240 | 1.60e-4 | 5.7e-7 | 1.95e-3 | 2.50e-3 |
  | g128 | 239/240 | 2.12e-4 | 1.2e-6 | 2.06e-3 | 2.62e-3 |

  For scale: the exact prefill kernels drift 8.0e-4 against phase 1, W4A8 9.2e-4, INT4 against bf16 weights 4.2e-2.
  The single top-1 flip is one long-code-6k position. The dumps are `parity/dqw_lmhead8_g{64,128}.bin`; the .bin
  files are gitignored and live in this worktree.

**Results.** These are measured against `fused`, the new default with the bf16 head. The suite rows come from the
tune split, 56 prompts, interleaved.

| head | greedy ms/token | greedy tok/s | spec ms/step | spec tok/s | tokens/step |
|---|---|---|---|---|---|
| bf16 | 72.25 | 13.84 | 107.93 | 23.56 | 2.56 |
| q8 g64 | 67.93 (−4.32) | 14.72 (×1.064) | 104.85 (−3.08) | 24.26 (×1.030) | 2.56 |
| q8 g128 | 67.66 (−4.59) | 14.78 (×1.068) | 105.40 (−2.53) | 24.41 (×1.036) | 2.59 |

- **Plain decode** (`bench_decode`, best of 3, against `fused`):

  | context | ms/token | change |
  |---|---|---|
  | short | 70.43 → 64.99 | −5.44 |
  | 19K | 76.84 → 71.02 | −5.82 |
  | 23.5K (noisy) | 75.31 → 70.65 | −4.66 |

- **Spec gains less than greedy.** Verify runs the head on 8 rows, where it only goes from 10.08 to 7.17 ms; at 1 row
  it goes from 9.29 to 5.11 ms.
- **g64 against g128.** g128 shows a higher spec tok/s but saves less per step. Its higher tokens/step comes from the
  diverged text, so the two speeds are within noise.
- **Token agreement** with the bf16 head:

  | head | greedy, identical prompts | greedy, tokens before first divergence | spec, identical prompts |
  |---|---|---|---|
  | g64 | 47/56 | 90.3% (first divergences at tokens 10…247) | 44/56 |
  | g128 | 44/56 | 87.9% (5…232) | 38/56 |

- **Acceptance.** Tokens per verify step: 2.56 (bf16), 2.56 (g64), 2.59 (g128). The per-category moves all sit on
  prompts whose text diverged, so they are not head quality as such. Examples:
  - agent/00 under g128 diverges at token 22, then runs 3.15 tokens/step against 3.86;
  - stem/00 diverges at 88, then runs 3.20 against 2.60.
- **Trade-off for the default** (the user decides):
  - g64 has the lower KL (1.60e-4 against 2.12e-4) and more identical outputs.
  - g128 is 0.15 ms faster per head call and otherwise within noise.
  - Either buys about 6% greedy and 3% speculative throughput. Its KL drift is below the exact prefill kernels'
    8.0e-4 and far below INT4 against bf16 weights (4.2e-2).

## Part 2: decode small-op fusion (`decode_fused.py`, on by default; `VOLUNDR_FUSED=0` turns it off)

**Where the time was.** The brief's "~12 ms of small ops" comes from the synthetic model. On the real model, with
interleaved runs:
- 4-bit matmuls alone: ~55 ms.
- With the bf16 head and the small bf16 linears: ~65.5 ms.
- The full step: ~71 ms (`decode_quick_wins/mmfloor.log`).

So ~6 ms sat outside the matmuls. Chain microbenchmarks at decode shapes (`chain*.py`, `run5/7/9.log`) located it:
- **Single-threadgroup 5120-wide row kernels.** RMSNorm, then mHC combine + norm and mix + norm: ~11 µs each, three
  per layer. They are latency-bound: 256 threads, 20 dependent loads each.
- **The mHC Sinkhorn on one thread per token**, on every layer's critical path: ~11 µs.
- **The KDA recurrence.** ~30 µs per layer, streaming 6.3 MB of fp32 state at ~205 GB/s. 16-byte loads and other
  threadgroup shapes changed nothing.
- **~2 µs per extra dependent kernel**, for ~1450 small kernels.

**What changed.** Every change keeps every rounding point and summation order.

| piece | before | after |
|---|---|---|
| mHC per layer | combine, rms_norm (mean), dyn, params (serial Sinkhorn), mix, rms_norm (input) | `combine_norm` (residual add + combine + next layer's mHC norm), dyn, `mix_norm` (16-lane Sinkhorn + mix + input norm) |
| row kernels (≤ 64 rows, D ≥ 1024) | 256 threads × 20 strided elements | 1024 threads stage the row in threadgroup memory, then 256 of them sum the squares in the original order |
| `kernels.rms_norm` (post-attention, verify) | strided loops | constant trip count + the same wide variant |
| KDA small ops | conv, concatenate (state), compute_g, astype, sigmoid, recurrence (fp32), astype, gated_norm | `kda_conv_gates` (conv + state + decay + beta, MLX's Exp/LogAddExp/Sigmoid formulas), recurrence writing bf16, gated_norm |
| q/k/indexer norms (OffsetRMSNorm) | astype w, add, strided copy, astype x, rms, astype | cached fp32 weight + `rms_norm_mlx` (one dispatch, MLX's single-row rms kernel, strided reads) |

Prefill-size calls (more than 64 rows, or more than 512 rows for the q/k norms) keep the old kernels; only the
row loops changed there. Verify gets the wide `rms_norm`, `MHC.mix` (no-norm `mix_norm`) and the norm-weight cache,
but not the KDA or layer fusions, because `spec/verify.py` has its own copies of those.

**Chain microbenchmarks** (µs per layer, `run9.log`):

| | before | after |
|---|---|---|
| two 5120-wide norms | 21.8 | 8.7 |
| mHC sequence | 41.8 (main) | 20.7 |
| KDA small ops + recurrence | 41.7 | 41.0 |

**Tried and dropped:**
- The KDA gated norm fused into the recurrence (one threadgroup per head): 33 µs against 31.8 µs for the two
  kernels, because 48 threadgroups on 20 cores balance badly.
- 16-byte loads in the recurrence: no change.

**Results.** `main` is main's decode path, reproduced in-process. `fused` is this branch's default, which also
includes the BCSA views.

| | main | fused | change |
|---|---|---|---|
| greedy, ms/token | 74.34 | 72.25 | −2.09 (−2.8%) |
| greedy, tok/s | 13.45 | 13.84 | ×1.029 pooled, ×1.030 geo-mean |
| spec, ms/step | 110.50 | 107.93 | −2.57 (−2.3%) |
| spec, tok/s | 23.02 | 23.56 | ×1.024 pooled, ×1.023 geo-mean |
| spec, tokens/step | 2.56 | 2.56 | – |

- **Exactness on the suite.** Tune split, 56 prompts, interleaved: tokens and per-step acceptance are identical on
  56/56 prompts, and the greedy tokens equal `tune2_base` on 56/56.
- **Plain decode** (`bench_decode`, best of 3):

  | context | ms/token | change |
  |---|---|---|
  | short | 71.92 → 70.43 | −1.5 |
  | 19K | 82.04 → 76.84 | −5.2 |
  | 23.5K | 86.95 → 75.31 | −11.6, but the medians, 88.9 → 88.4, show that run was mostly noise |

- **The split between parts is not measured.** At short context `unfused` reaches 70.37 ms. It is this branch with
  `decode_fused` off, keeping the wide `rms_norm`, the cached norm weights and the views. So in that bench the
  `decode_fused` kernels add nothing measurable on top (70.43, median 71.01 against 70.45). The ~25 µs per layer the
  chain microbenchmarks save sits inside the bench's ±2 ms spread. The suite did not run `unfused`, so how the 2.1
  ms divides between `decode_fused` and the rest is not separated.
- **Noise episode.** One spec prompt, coding/03-97f49a2c, ran during a noise episode: main 104.8, fused 122.9 and q8
  155–163 ms/step. It is left in the pooled numbers, which is conservative against the new configs.

**Exactness.**
- `tests/test_decode_fused.py`, 8/8: each kernel against the kernel or op sequence it replaces, with `mx.array_equal`.
  - Covers wide and narrow paths, D = 96…6144, 1…4096 rows, and strided inputs.
  - `kernels.rms_norm` is checked against main's strided-loop source.
  - The tiny model (every mixer kind, Engram, BCSA far regime) is checked fused == unfused over the full, chunked and
    decode paths.
- `tests/test_volundr.py`: 10/10.
- Real model: `dump_logits.py` with everything on gives a dump bit-identical to `parity/mlx.bin`
  (`parity/dqw_fused3.bin`, `run10.log`). That is prefill plus 240 decode steps, including the 6K and 14K
  far-block prompts.
- **Tune split** (56 prompts, interleaved): `main` and `fused` give identical greedy tokens, spec tokens and per-step
  acceptance on 56/56. Their greedy tokens also equal yesterday's `runs/tune2_base.jsonl` on 56/56.
  - Against yesterday's spec run, `runs/tune2_d4h4_lk.jsonl`, tokens match on 54/56 and acceptance on 40/56 (all 28
    of group 1). The rest is Lookup history, not numerics:
    - Lookup's running means carry across prompts.
    - Each `ab_suite.py` process (one per category group) starts a fresh Lookup, whereas yesterday's single run had
      already seen the earlier groups.
    - A different draft changes the verify block, which can flip a near-tie (agent/00 at token 45).
- **Held-out split:** tokens and per-step acceptance are identical to main's `runs/test_frozen{,_r2}.jsonl` on all 45
  prompts that ran.

## BCSA decode: views instead of integer indexing

The BCSA agent's finding: in MLX 0.32.4, `x[0]` and `p[b]` go through `take` and copy the whole array, while
`x[0:1].squeeze(0)` is a view. `volundr._first` / `_gather_blocks` now use views:
- decode `_gather_blocks`: pooled K and V, ~10 MB each per BCSA layer at 23K;
- `_select`: the indexer keys;
- `_bcsa_prefill`'s kernel arguments.

`test_bcsa_views_equal_integer_index` checks the values.

**Results** (`bench_decode`, `fused+copyidx` against `fused`, ms/token):

| context | best | median |
|---|---|---|
| short (no far blocks) | 70.51 against 70.43 | – |
| 19K | 78.38 → 76.84 (−1.5) | 81.18 → 78.25 (−2.9) |
| 23.5K | 80.64 → 75.31 | 86.11 against 88.38 |

At 23.5K the best and median disagree, so the run is inconclusive.

On the tune split, the two longest categories gain the most from `main` to `fused` in greedy:
- long (8.1–12.5K-token prompts): ×1.055;
- agent (4.5–7.3K): ×1.052;
- overall: ×1.029.

That fits the views mattering with context length, but the suite does not separate them from the fusion.

## Method

- **`bench_decode.py`.** Plain decode, with the configs from `ab_modes.py` switched in-process on one loaded model
  and interleaved per run, best and median of 3–4.
  - `main` reproduces main's decode path: `decode_fused` off, main's strided `rms_norm`, uncached norm weights and
    copying BCSA indexing. Its tokens equal main's.
  - Long prompts use `--shared-cache`: one prefill, then configs take turns on 48-token segments of one sequence.
- **`spec/ab_suite.py`.** Each tune prompt runs under every config, back to back.
  - Greedy uses `generate.py`'s pipelined loop with MLX's default buffers.
  - Spec runs the frozen config (4-bit drafter + 4-bit head copy, NAX verify, lookup, T = 8,
    `MLX_MAX_OPS_PER_BUFFER=100`), with one Lookup per config. That Lookup's history covers the prompts of its own
    category group only, because each group is one process.
  - Metrics follow spec/NOTES.md (pooled tok/s, geo-mean of per-prompt speed-ups, tokens per step).
  - `decode_quick_wins/analyze_ab.py` summarises.
- **Held-out test split.** The plan was one run of the final default config (fused, bf16 head) under the
  spec/NOTES.md protocol: `specdec.py bench` (frozen), then `specdec.py base`, back to back (`test_chain.sh`).
  - At the user's request it was stopped after 45/84 spec prompts, and the greedy half never ran.
  - The sample is compared against other sessions' runs (see Summary), which is not an A/B.
- **Lock and noise.** Every run took the GPU lock, with a canary before and after. Throughout the day fseventsd
  held one core at 100% and 26% of memory, so single runs move by ±2 ms at short context and more at 23–33K.
  - The 33K run in `run10.log` is not usable: steps drifted 88 → 190 ms within the run.
  - `run8.log` was killed mid-way, because the code changed under it.
  - `run4.log` (ablation in real mode) is invalid: MLX pruned layers.
