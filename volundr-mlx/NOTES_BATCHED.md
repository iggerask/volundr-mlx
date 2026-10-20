Status: done (2026-10-20 06:05).

# Batched generation: B independent sequences decoding together (branch batched-bench, 2026-10-20)

A measurement, not a server. Everything else in this port decodes one stream at a time (bs = 1). Here B independent
sequences decode in one forward per step, and we measure aggregate and per-stream throughput.

## Resume

Nothing is pending. Branch `batched-bench` from main 808239f, in the worktree
`.claude/worktrees/agent-a987cc9a005e6d9ce`. The data links are as in the brief: .venv, models, ref, spec/suite.json,
spec/runs and parity/*.bin.

Every GPU job ran as `batched/step.sh <log> <cmd>` under `./with_gpu_lock.sh`, from `volundr-mlx`. step.sh runs the
canary, the command, then the canary again, and records load and swap around it.
- `batched/run_decode.sh`: plain batched decode at 200 and 8,192 tokens (`batched/decode.log`).
- `batched/run_verify.sh`: the batched spec parts (`batched/verify.log`). Run it with `CTXS=200`, then `CTXS=8192`.
- `batched/run_select.sh`: the 8K follow-up, which compares two far-block selection paths and repeats B = 16
  (`batched/select.log`).

Summaries:
- `batched/report.py <run file>`: the decode tables.
- `batched/spec_estimate.py spec/runs/batched-bench_verify.jsonl`: the spec estimate.
- `batched/accept_summary.py <spec run>`: the B = 1 acceptance aggregates.

Run files are `spec/runs/batched-bench_*.jsonl` (shared, gitignored), with tokens and margins per stream.

## Summary

Plain greedy decode, bf16 head as shipped. The aggregate tok/s is B / (ms per step). The NAX column is the
better of NAX on and off at each B (on from B = 4).

| B | 200-token prompts: ms/step | aggregate tok/s | per-stream tok/s | 8K prompts: ms/step | aggregate tok/s | per-stream tok/s | server peak GB (short / 8K) |
|---|---|---|---|---|---|---|---|
| 1 | 70.3 | 14.2 | 14.2 | 77.2 | 12.9 | 12.9 | 24.8 / 25.0 |
| 2 | 78.7 (NAX off) | 25.4 | 12.7 | 90.1 (off) | 22.2 | 11.1 | 25.4 / 25.9 |
| 4 | 87.6 (NAX on) | 45.7 | 11.4 | 103.3 (on) | 38.7 | 9.7 | 26.6 / 27.5 |
| 8 | 95.7 (on) | **83.6** | 10.5 | 128.1 (on) | **62.5** | 7.8 | 29.0 / 30.7 |
| 16 | 120.1 (on) | **133.2** | 8.3 | 178.5 (on) | **89.6** | 5.6 | 33.8 / 36.9 |

- **Batching scales well because decode is weight-bound.** B = 16 costs 1.7× a B = 1 step at short context and 2.3×
  at 8K, for 16× the tokens.
  - The aggregate is 9.4× at short context and 6.9× at 8K.
  - Each extra stream costs about 2-3 ms per step at short context and about 6 ms at 8K.
- **The NAX small-M matmuls are what make this work from B = 4 up.** The stock quantized matmul is compute-bound at
  8-16 rows:
  - B = 8: 155 → 96 ms;
  - B = 16: 230 → 120 ms;
  - at 8K, B = 16: 299 → 179 ms.
  - At B = 2 NAX loses (it pads to 16 rows; stock is faster at 2 rows), and at B = 4 it wins by 4%.
- **Correctness: every stream matches its own B = 1 greedy run, or diverges at a bf16 near-tie.**
  - 159 of 216 batched streams are identical over 69-133 tokens.
  - Each of the 57 divergences starts at a step where one run's top-1 / top-2 margin is 0 and the other's is
    ≤ 0.25, or both are 0.125.
  - None is exact by construction (see Correctness).
- **Batched speculative decoding (ESTIMATE, not run): it doesn't pay past B = 2.**
  - B = 2: 46-52 tok/s aggregate at short context. That's 1.8-2.0× plain B = 2 and close to plain B = 4.
  - B = 4: 50-60 tok/s, below plain B = 8 (84).
  - B × 8 = 32 verify rows exceed the NAX kernel's 16. As 16-row pieces the verify costs 170 ms against 89 at B = 1.
- **Against upstream.** Upstream measures 87.7 / 129.3 / 155.1 tok/s aggregate at 1 / 2 / 4 requests. Ours is
  14.2 / 25.4 / 45.7 plain, or 27.7 (spec, measured) / ~46-52 / ~50-60 (spec, estimated).
  - We reach 133 tok/s only at B = 16.
  - Upstream's GPU has 2.8-3.1× the bandwidth. Its single-stream figure is below the weight-streaming floor for plain
    decode (see Upstream).

## Method

**Equal-length prompts.**
- The caches keep a scalar position: `BCSACache.offset` and `nb` (ring slot = position % 4096, closed blocks) and
  `KVCache.offset`. So every row of a batch must sit at the same position.
- Every prompt is exactly `--tokens` ids, 200 or 8,192.
- Each prompt is a chat template around an instruction plus repo files (code and notes from this repo, rotated per
  prompt). The file text is cut so that the templated ids have exactly that length.
- No spec-suite content is used.

**Caches.**
- Each prompt is prefilled alone at B = 1 (`generate.prefill`, 2048-token chunks). This is the stock path, so the
  prefill is the same as a B = 1 run's.
- The per-prompt caches are then stacked along the batch axis into one snapshot (`stack_caches`):
  - KDA conv / recurrent states;
  - the BCSA ring, pooled K/V, pooled indexer keys and open-block keys;
  - the dense K/V;
  - the Engram ids.
  The source arrays are freed layer by layer.
- Every run, the B = 1 references included, starts from row slices of the snapshot. These are views, which the first
  decode step copies, so all runs start from the same state.
- `--distinct 8`: row i decodes prompt i % 8. This keeps the harness snapshot at 8 prompts (4.6 GB short,
  5.4 GB at 8K) instead of 16.
  - The B = 16 rows 8-15 repeat rows 0-7 as independent rows; no work is shared.
  - Repeated rows produced identical outputs to each other in every run.

**The step.** The step is `volundr.Model`'s own decode path with `[B, 1]` inputs. The fused decode kernels
(decode_fused.py: mHC mix/combine + norms, KDA conv + gates, the row norms) and the KDA recurrence kernel are written
per row (grid over rows / batch), so they take B rows unchanged.

There are two B = 1-only spots, and neither is in the B = 1 path:
- **BCSA far-block selection** (only at 8K, where 1,024 > 512 far blocks are valid).
  - The kernel path (`_select`: `bcsa_scores` + `bcsa_topk`) reads row 0 only. At B > 1 the code took the ops path
    (`_block_scores` + argpartition): the same set of blocks, though possibly in a different order.
  - Added on this branch (`volundr.BATCH_SELECT`, default `kernel`): at B > 1, the B = 1 kernels run once per row.
    `VOLUNDR_BATCH_SELECT=ops` restores the ops path.
  - The two cost the same within noise (8K, NAX on, 2 reps each):

    | B | per-row kernel ms/step | ops ms/step |
    |---|---|---|
    | 8 | 128.1, 125.6 | 127.5, 131.1 |
    | 16 | 177.8, 173.6 | 178.4, 182.2 |

  - Per-row kernels are not closer to B = 1 either: per rep, 18/24 streams identical with the kernels and 22/24 with
    ops. Every difference is at a 0.125 margin (`batched/select.log`).
- **`_gather_blocks`** loops over the batch: one `take` per row and layer. It works, and it's included in the times.

Nothing else needed a fallback.

**Timing.**
- Greedy and pipelined like `generate.greedy_tokens`: the next step's graph is built while the current one runs.
- Each step also computes the top-1 / top-2 margin of every row. That is three reductions over [B, 248K]. It's in
  every run, the B = 1 references included.
- 4 untimed warm-up steps (the snapshot copy, Metal JIT), then 128 timed steps (64 in the select run).
- EOS is ignored (fixed-length runs, as a throughput benchmark).
- Peak memory is `mx.get_peak_memory()` from a reset at the start of each run. MLX's default command buffers are used,
  as in generate.py.

**NAX.**
- `spec/nax_hook.enable` switches every 4-bit g128 linear to the small-M NAX kernel for 2..16 rows. Decode with B
  rows is B rows. B = 1 is the stock path either way.
- The bf16 linears always run on MLX's matmul: lm_head, mHC dyn, in_proj_b, the BCSA indexer and Engram.
- Within each B, NAX off and NAX on run back to back on one loaded model.

## Plain batched decode

200-token prompts (`spec/runs/batched-bench_decode_200.jsonl`, 05:42-05:46):

| B | NAX | ms/step | aggregate tok/s | per-stream tok/s | peak GB | peak less snapshot | identical to B = 1 |
|---|---|---|---|---|---|---|---|
| 1 | - | 70.27 | 14.2 | 14.23 | 29.34 | 24.77 | 1/1 |
| 2 | off | 78.71 | 25.4 | 12.70 | 29.95 | 25.39 | 2/2 |
| 2 | on | 83.44 | 24.0 | 11.98 | 29.96 | 25.40 | 2/2 |
| 4 | off | 91.28 | 43.8 | 10.96 | 31.18 | 26.62 | 4/4 |
| 4 | on | 87.56 | 45.7 | 11.42 | 31.19 | 26.63 | 4/4 |
| 8 | off | 155.02 | 51.6 | 6.45 | 33.59 | 29.02 | 6/8 |
| 8 | on | 95.65 | 83.6 | 10.45 | 33.59 | 29.03 | 8/8 |
| 16 | off | 229.86 | 69.6 | 4.35 | 38.42 | 33.85 | 10/16 |
| 16 | on | 120.14 | 133.2 | 8.32 | 38.41 | 33.84 | 12/16 |

The 8 B = 1 references ran at 69.9-71.5 ms/step. The B = 1 row matches main's plain decode (README: 70.09 ms).

8,192-token prompts (`spec/runs/batched-bench_decode_8192.jsonl`, 05:46-05:53; prefill of the 8 prompts took 173.7 s,
377 tok/s):

| B | NAX | ms/step | aggregate tok/s | per-stream tok/s | peak GB | peak less snapshot | identical to B = 1 |
|---|---|---|---|---|---|---|---|
| 1 | - | 77.21 | 12.9 | 12.95 | 30.44 | 25.01 | 1/1 |
| 2 | off | 90.06 | 22.2 | 11.10 | 31.30 | 25.87 | 1/2 |
| 2 | on | 92.88 | 21.5 | 10.77 | 31.31 | 25.88 | 2/2 |
| 4 | off | 107.87 | 37.1 | 9.27 | 32.95 | 27.52 | 1/4 |
| 4 | on | 103.29 | 38.7 | 9.68 | 32.95 | 27.52 | 3/4 |
| 8 | off | 188.00 | 42.5 | 5.32 | 36.14 | 30.71 | 5/8 |
| 8 | on | 128.06 | 62.5 | 7.81 | 36.17 | 30.74 | 5/8 |
| 16 | off | 298.63 | 53.6 | 3.35 | 42.34 | 36.91 | 4/16 |
| 16 | on | 178.53 | 89.6 | 5.60 | 42.34 | 36.91 | 10/16 |

- References: 76.0-79.6 ms/step.
- The select run repeated NAX on, at a lower system memory load (4 distinct prompts, 2.7 GB snapshot):
  - B = 8: 125.6-128.1 ms;
  - B = 16: 173.6-177.8 ms (`spec/runs/batched-bench_decode_8192_select.jsonl`).
- So the B = 16 figure above is not a swap artefact, although swap grew from 15 to 22.5 GB during that run.

**Memory.**
- "Peak less snapshot" is what a server would hold: the raw peak minus the harness's snapshot of the prompts' prefill
  states.
- A stream's cache is 0.571 GB at 200 tokens and 0.679 GB at 8K:
  - KDA states: 0.17 GB;
  - the BCSA ring: 0.32 GB. It is preallocated at 4096 + 512 rows per layer, whatever the length;
  - pooled K/V and the dense K/V: the rest, growing with context.
- With the 24.15 GB of weights, B = 16 peaks at 33.8 GB (short) and 36.9 GB (8K).
- No OOM. The 8K B = 16 run, raw peak 42.3 GB, pushed swap from 15.4 to 22.5 GB (other apps' memory; fseventsd held
  27.6 GB resident).

**Where a stream's cost goes** (reading the slopes; inferred from the tables and the cache layout, not profiled).

| context | ms per added stream, B 4 → 8 | B 8 → 16 |
|---|---|---|
| short | 2.0 | 3.1 |
| 8K | 6.2 | 6.3 |

- Some per-row traffic is fixed by the cache layout.
  - Each stream reads and writes its 54 KDA recurrent states every step: 2 × 3.1 MB × 54 = 340 MB, about 1.1 ms at
    307 GB/s.
  - At 8K it also reads its BCSA window plus 512 far blocks: 4608 keys × 4 heads × 256 × 2 (K, V), bf16, × 17
    layers = 321 MB, another ~1 ms. The dense layer's 8K keys are ~0.1 ms.
- The rest is per-row work that is not at bandwidth: SDPA, the per-row selection and gathers at 8K, the 16-row NAX
  matmuls (about 1.25× a 1-row matmul, smallm/VERIFY_STEP.md), the bf16 lm_head at B rows, and the mHC / norm kernels
  over B rows.
- The B = 1 step itself (~70 ms) is mostly the per-token weights: about 17.3 GB of the 24.15 GB, since the embedding
  and Engram tables are only gathered, streamed at ~245 GB/s.

## Correctness

**Check.**
- Every stream of every batched run is compared token by token with that prompt's own B = 1 greedy run from the same
  snapshot: 133 tokens, or 69 in the select run.
- At the first differing token, we record both runs' top-1 / top-2 margin.

| run | batched streams | identical | divergences: (B = 1 margin, batched margin) | earliest |
|---|---|---|---|---|
| short | 60 | 48 | (0, 0.125) × 7, (0.125, 0) × 1, (0.25, 0) × 2, (0.125, 0.125) × 2 | token 5 |
| 8K | 60 | 31 | (0, 0.125) × 14, (0.125, 0) × 13, (0.125, 0.125) × 2 | token 21 |
| 8K select run | 96 | 80 | (0.125, 0) × 16 | token 21 |

**Results.**
- Every divergence starts at a near-tie. One run has an exact tie (margin 0) and the other a margin of at most 0.25, or
  both have 0.125. Above 16, bf16 logits come in steps of at least 0.125.
- Ties are common in these texts. The B = 1 references alone have 8 exact-tie steps in 1,064 at short context and 15
  in 1,064 at 8K.
- The batched rows are deterministic and independent of their position in the batch. Rows that repeat a prompt (B = 16,
  and B = 8 in the select run) produced identical tokens to each other, divergences included.

**Why it isn't bit-exact.** Not proven kernel by kernel. The per-row kernels (fused decode, KDA recurrence, mHC) do
the same work per row at any B. What changes with B is the matmul kernel choice:
- As far as we know (not checked in the 0.32 source), MLX's bf16 matmul runs a matrix-vector kernel for 1 row and a
  matrix-matrix kernel above that. This covers lm_head, mHC dyn, in_proj_b, the BCSA indexer and Engram.
- The stock quantized matmul seems to switch kernels between 4 and 8 rows. Its step time jumps from 91 to 155 ms, and
  with NAX off B = 8 and 16 diverge more.
- NAX replaces the 4-bit matmuls from 2 rows on.

Each kernel sums in its own order, so the logits can move by a bf16 step. Exact batched output would need row-invariant
kernels: a per-row matrix-vector for the bf16 linears and the same 4-bit kernel at every B. Neither was attempted.

## Batched speculative decoding: ESTIMATE (not run)

**Why not built.**
- A batched spec loop needs per-stream positions: each stream accepts a different number of tokens per step. The BCSA
  and dense caches have a scalar offset, so that is the server work below.
- It didn't fit the time box. Instead, the step's GPU parts were measured for B independent sequences, and combined
  with the measured B = 1 spec loop.

**Measured** (`batched/bench_verify.py`, `spec/runs/batched-bench_verify.jsonl`).
- From an idle GPU, median of 10, variants interleaved.
- Spec config: 8-bit target head g64, 4-bit drafter with a 4-bit head copy.
- The verify forward runs B independent sequences × 8 rows in one forward, with `[B]` KDA states, verify's 8q BCSA
  branch once per row against that row's own cache, and the dense layer per row, causal. It does not commit, and it
  includes argmax and the drafter features.
- The drafter is B DFlash2 proposals submitted together. Each has its own drafter cache.

| context | B | rows | verify, NAX ≤ 16 rows (stock above) | verify, 16-row NAX pieces | verify, stock | drafter |
|---|---|---|---|---|---|---|
| 200 | 1 | 8 | 89.3 | - | 143.9 | 10.0 |
| 200 | 2 | 16 | 97.5 | - | 206.4 | 18.6 |
| 200 | 4 | 32 | 214.4 | **170.2** | 214.5 | 37.2 |
| 8K | 1 | 8 | 95.6 | - | 148.9 | 11.5 |
| 8K | 2 | 16 | 107.5 | - | 214.6 | 22.0 |
| 8K | 4 | 32 | 231.0 | **187.7** | 232.1 | 43.6 |

- The production B = 1 verify (`verify.forward`, compiled KDA layers) measured 86.9 ms at short context and 91.9 at
  8K. The harness's B = 1 verify, at 89.3 / 95.6, is the baseline for the differences.
- The B = 2 verify is +8.2 ms at short context, +11.9 at 8K. NOTES_SPEC_ACCEPT's two-chain `forward_batch`, which
  shares one context cache, measured +9.1.
- The drafter scales linearly when B proposals are submitted together. That's an upper bound: a batched drafter
  forward would read its weights once.

**Formula.** `step(B) = step(1) + [verify(B) − verify(1)] + [drafter(B) − drafter(1)] + (B − 1) × 3.2 ms`.
- step(1) and tokens per step come from the measured B = 1 loop (spec-accept tune run `exact+q8g64@early@cv`,
  pooled; batched/accept_summary.py):
  - short (48 prompts, < 1K tokens): 93.19 ms and 2.617 tokens/step;
  - ≥ 4K (8 prompts, 4.5-12.5K): 97.45 ms and 2.516 tokens/step.
- 3.2 ms is one stream's commit plus drafter-context update (NOTES_SPEC_ACCEPT's serialized profile).
- The low end assumes a batched drafter and commit that cost what B = 1's do.
- Acceptance is per stream and independent of batching, so the aggregate is B × tokens/step / step(B). Streams run
  in lockstep. The drafter runs every step; at B = 1 the lookup copies skip it on ~10% of steps.

| context | B | est. step ms | est. per-stream tok/s | est. aggregate tok/s | plain batched greedy, same B (measured) |
|---|---|---|---|---|---|
| short | 1 | 93 (measured) | 28.1 | 28.1 | 14.2 |
| short | 2 | 101-113 | 23.1-25.8 | **46-52** | 25.4 |
| short | 4 | 174-211 | 12.4-15.0 | **50-60** | 45.7 (B = 8: 83.6) |
| ≥ 4K | 1 | 97 (measured) | 25.8 | 25.8 | 12.9 (8K) |
| ≥ 4K | 2 | 109-123 | 20.5-23.0 | **41-46** | 22.2 |
| ≥ 4K | 4 | 190-231 | 10.9-13.3 | **44-53** | 38.7 (B = 8: 62.5) |

**Reading.**
- Spec pays at B = 2: about 2× plain batched greedy at the same B, with each stream at 23-26 tok/s.
- At B = 4 it breaks even with plain B = 4 and loses to plain B = 8.
- The step is the verify of B × 8 rows, and above 16 rows the NAX kernel needs two tiles (+81 ms at short context).
- The per-stream speed-up from speculation also shrinks as the step grows.
- The levers are:
  - a NAX tile for 32 rows;
  - shorter drafts at larger B (B × T ≤ 16, for example B = 4 × T = 4);
  - a batched drafter forward instead of B serial ones.
- For throughput at many streams, plain batching wins: 84 tok/s at B = 8 and 133 at B = 16.

## Against upstream

**Upstream reference.** The CUDA Agens Engine, 24 GB profile, server mode, 19 short prompts × 256 tokens, on a 48 GB
Ada GPU (~864-960 GB/s).

| concurrent streams | upstream aggregate tok/s | ours, plain batched greedy (decode only) | ours, spec |
|---|---|---|---|
| 1 | 87.7 | 14.2 | 27.7 (tune, measured) |
| 2 | 129.3 | 25.4 | ~46-52 (estimate) |
| 4 | 155.1 | 45.7 | ~50-60 (estimate) |
| 8 | - | 83.6 | - |
| 16 | - | 133.2 | - |

- **Scaling.** Upstream gains 1.47× from 1 to 2 requests and 1.77× from 1 to 4. Ours gains 1.8× and 3.2× (plain),
  because our single stream is far from compute-bound.
- **Single stream.**
  - 87.7 tok/s is 11.4 ms per token. Streaming our ~17 GB of per-token weights at 960 GB/s takes ~18 ms.
  - So upstream's single-stream figure can't be plain one-token-per-forward decode of the same INT4 weights. It
    is presumably speculative: the release ships an MTP head, which this port drops. A lighter profile is also
    possible.
  - This was not checked against the engine.
- **Per bandwidth.**
  - Our plain B = 1 streams ~17.3 GB in 70 ms, ~245 GB/s effective (80% of 307). This is an estimate: 24.15 GB of
    weights less the gathered embedding (2.54) and Engram tables (4.29).
  - Upstream's c = 4 figure is 3.4× our plain B = 4 and ~2.6-3.1× our estimated spec B = 4, which is roughly the
    bandwidth ratio.
- **Not like for like.**
  - Upstream's number is end to end through a server, prefill included. Ours is decode steps only.
  - For 200-token prompts our prefill is ~0.6 s per prompt against 18 s of decode per 256 tokens at B = 1.
  - We need B = 16 (133 tok/s) to pass upstream's c = 2 figure.

## What a real batched server would need

**Unequal lengths: per-row positions.** This is the main change. Everything here relied on equal offsets.
- **BCSA.**
  - Per-row `offset` / `nb`: an int32 array [B].
  - Ring writes at per-row slots: a scatter, or a small kernel, instead of one slice update.
  - The window length and the valid far-block count `nv` differ per row: SDPA needs a per-row key mask, or a decode
    attention kernel that takes per-row lengths. bcsa_verify's 8q kernel already reads the ring in place by position.
  - Blocks close at different steps per row, so the pooled-block append and the far selection (already per row with
    the B = 1 kernels) take the row's own count.
  - RoPE needs per-row offsets. `mx.fast.rope` takes an array offset in recent MLX; this was not checked for 0.32.
- **Dense layer.** Per-row lengths: a left-padded batch KV, as in mlx_lm's batch caches, or a paged / ragged layout
  with a per-row mask.
- **KDA and Engram** need nothing. Their state is already per row and position-free: conv state, recurrent state, the
  last two ids.
- **Speculative decoding** additionally needs per-row commits of different n:
  - the KDA replay of n rows per stream: gated_delta_vec with a per-row length;
  - BCSA appends of n rows per stream;
  - dense trims per row.

**Memory per stream.**
- The fixed 0.32 GB BCSA ring (4096 + 512 rows × 17 layers) dominates at short context. A server should size the ring
  to min(length, 4096) and grow it.
- The KDA state is 0.17 GB per stream regardless of length.
- Measured per stream: 0.57 GB at 200 tokens and 0.68 GB at 8K. Beyond that, from the layout, it grows by about
  23 MB per 1K tokens: pooled blocks plus dense K/V.
- With 24 GB of weights, ~30 GB of headroom holds roughly 50 short streams or ~35 at 16K, before activations and the
  session cache budget (NOTES_SESSION.md).

**Scheduling.**
- **Continuous batching.** Insert a row when a request's prefill finishes, remove it at EOS or its token limit, and
  compact the batch (slice / concatenate rows, as `stack_caches` does).
- **Prefill in the batch.** Prefill runs at ~380 tok/s and stalls every decoding stream while it runs: an 8K prompt
  is ~21 s. A server needs chunked prefill interleaved with decode steps, with a chunk size that trades
  time-to-first-token against decode latency, or a separate priority for prefill.
- **Kernel choice by batch size.**
  - NAX on from B ≥ 4, off at B = 2.
  - Plain batched decode rather than spec above B ≈ 2-4, unless the verify fits 16 rows (B × T ≤ 16).
  - The NAX kernel covers at most 16 decode rows.
- **Cap B by memory and latency.** Per-stream speed falls from 14.2 (B = 1) to 8.3 tok/s (B = 16) at short context,
  and to 5.6 tok/s at 8K.

## Canaries (TFLOP/s; idle 26-28; none below 25)

| run | before / after |
|---|---|
| smoke test (B ≤ 4, 16 steps, not used) | 28.5 / 27.3 |
| plain decode, 200 tokens | 28.3 / 26.1 |
| plain decode, 8K | 27.3 / 26.9 |
| spec parts, 200 tokens | 27.4 / 27.2 |
| spec parts, 8K | 27.0 / 26.0 |
| 8K select A/B + B = 16 repeat | 26.7 / 25.3 |

- Load average was 4-10 throughout (other apps).
- Swap was nearly full from the start (11.9 of 13.3 GB). macOS grew it to 22.5-23.5 GB during the 8K B = 16 runs.
- The 8K B = 16 timing repeated within 3% in a lower-memory process, so memory pressure did not distort it.

## Files

- `batched/bench_batched.py`: batched plain decode. It builds the prompts, prefills, stacks the snapshot, runs the
  references and the B × NAX (× selection) grid, and compares tokens.
- `batched/bench_verify.py`: the batched spec parts:
  - the verify forward of B independent sequences × T rows;
  - the 16-row NAX split;
  - the drafter at B.
- `batched/spec_estimate.py`, `batched/accept_summary.py`, `batched/report.py`: summaries.
- `batched/run_decode.sh`, `run_verify.sh`, `run_select.sh`, `step.sh`: the runs. `*.log` holds their output and the
  canaries.
- `volundr.py`: `BATCH_SELECT`, the per-row far-block selection for B > 1 decode. The B = 1 path is unchanged, and
  `tests/test_volundr.py` passes 10/10.
