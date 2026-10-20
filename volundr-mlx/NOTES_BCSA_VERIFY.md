# BCSA verify for speculative decoding: an 8-query path (branch bcsa-verify-8q, 2026-10-19)

## Status: done

- **What:** for speculative verify, `bcsa_verify.py` replaces the prefill kernels in the 17 BCSA layers. It is the
  default in `spec/verify.py`; `VOLUNDR_BCSA_VERIFY=prefill` restores the old path. It applies to every context
  length. Same-day A/B on short prompts showed no measurable slowdown (+0.6 ms/step pooled, inside noise), so I
  added no gate by context.
- **Long context:** a long-prompt spec step now takes ~106-112 ms instead of ~124-130 ms on the same day. On the
  held-out long prompts (14-21.5K) the median step is 105.6 ms.
- **Tune A/B, same day:** all 56 prompts, plus long+agent repeated three times.
  - Long: 14.94 -> 19.19 tok/s (+28%).
  - Agent: 32.41 -> 39.49 tok/s (+22%).
  - All prompts: 23.32 -> 23.90 tok/s (+2.5%).
  - Short categories: unchanged within noise.
- **Held-out test (once, frozen config) against greedy on the same day:** 26.09 vs 13.17 tok/s. That is 1.98x pooled
  and 2.12x geo-mean per prompt, at 2.63 tokens/step. Long is 1.79x and agent 2.95x.
- **Correctness:** output matches greedy except at rounding ties, as before. Teacher-forced KL is at the old path's
  noise floor. Free-running, 9/14 prompts match greedy exactly (old path: 6/14), and every divergence is at an exact
  bf16 tie.

## Profile: where the long-context step time went (old path)

`spec/prof_bcsa_verify.py`, frozen config (NAX verify, 4-bit drafter), canary 24.1 / 22.8 TFLOP/s
(`spec/bcsa_verify_8q/bcsa-verify-8q_prof_base.jsonl`). The "no BCSA attn" column replaces the 17 layers' attention
with a pass-through. The difference between the two verify columns is what that attention costs inside the real
graph.

| context | verify 8 (+commit) | verify 8 | verify, no BCSA attn | BCSA attn | decode step |
|---|---|---|---|---|---|
| 197 | 92.5 | 91.5 | 89.4 | 2.1 | 75.5 |
| 7,260 (agent) | 112.3 | 110.7 | 91.6 | 19.1 | 79.8 |
| 12,547 (long) | 123.2 | 114.1 | 93.2 | 20.9 | 82.1 |
| 23,632 (2 long prompts) | 116.9 | 114.7 | 91.4 | 23.4 | 82.0 |

The rest of the verify forward stays at 89-93 ms at every length. Commit plus the drafter context update costs
1.3 + 0.9 ms. The table below times one layer's parts in isolation at 23.6K, 17 calls per eval, in ms for 17 layers:

| part | old path | new path | decode, 1 query |
|---|---|---|---|
| window: gathered out of the ring + concatenated with the new keys | 6.6 | 0 (read in place) | 0 |
| far-block selection (indexer q 0.59, scores 0.63, top-k 0.28) | 1.1 | 1.1 | 0.86 |
| far-block union of the 8 queries | 0.41 | 0.41 | — |
| attention (old window part alone: 8.1) | 11.0 | 2.1-2.4 (+ merge) | 1.36 (SDPA) |
| **whole BCSA verify branch** | **19.4** | **4.3** | **2.2** |

Why the old attention was slow:
- The prefill kernel runs one threadgroup per (32-query block, head). That is 24 threadgroups, each streaming the
  4K-key window plus the union (1565 blocks at 23.6K) one after another.
- 24 of each block's 32 query rows are padding.
- Its `[0]`-indexed inputs are copies (see the MLX section below).
- Plain SDPA with 8 queries over the decode layout is no fix: it costs 7.2 ms, because it reads K/V once per row.

## The 8-query path (`bcsa_verify.py`, `spec/verify.py`)

- **Row packing.** The G = 6 query heads of a KV head times the T = 8 queries are contiguous in q [H, T, D]
  (row g * T + t). That makes 48 rows, exactly 3 NAX row fragments, with no padding at T = 8, and each key is loaded
  once per KV head.
- **Keys are read where they already are:**
  - the ring slots, in any order. Slot s holds the position of age d = (s - P) mod W; query t sees it iff d > t and
    d >= W - P.
  - the T new keys, causal.
  - the far-block union (`kernels.far_union`), gathered from the pooled K/V and masked by each query's membership
    bits.
- **Split-K.** The 32-key batches (ring, then union, then new keys) are spread evenly over NTG = 8 threadgroups per
  KV head, giving 32 threadgroups of 192 threads. Each writes fp32 partials (unnormalized O, max, sum), and a second
  kernel merges them. NTG 4-8 are equally fast at 1-24K (`spec/bench_bcsa_attend.py`).
- **Numerics** match the prefill kernel: NAX tiles, fp32 scores and accumulation, exp2 online softmax, and one joint
  softmax over window and far keys. Against the prefill kernel, outputs differ by at most one bf16 ulp, in 2-9% of
  elements.
- **Kernel time per layer** on synthetic data shaped like the model: 57 us at 1K, 86 at 3K, 99 at 4.5K, 112 at 7.3K,
  127 at 23.6K. The old path at 23.6K, including the gather and concatenate, takes 676 us.

## MLX 0.32.4: integer indexing copies

- `x[0]` goes through `take`. Measured cost: 50-60 us per call on the 9.4 MB ring, and 75 us on the 10.5 MB pooled
  K/V at 23.6K.
- `x.squeeze(0)` and `x[0, ...]` are free views (`spec/prof_bcsa_verify.py` probe parts).
- The verify branch squeezes. Before this fix the new kernel cost 8.5 ms instead of 2.4 ms.
- The same pattern is in volundr.py's decode path, outside this branch's scope. `_gather_blocks` takes `p[b]` of the
  pooled K and V in every BCSA layer of every decode step: about 2.5 ms per decode step at 23.6K by these numbers.
  `_select` and `_bcsa_prefill` also index `[0]`. This is for "Decode quick wins".

## Correctness

- **Kernel tests.** `tests/test_bcsa_verify.py` checks against fp64 and against the old kernels on 9 shapes: ring
  wrap, ring not full, ring filling inside the verify, T = 1..16, padding rows, empty selections. `tests/test_volundr.py`
  passes 10/10.
- **Teacher-forced** (`check_verify.py --nax`, new vs old with the same flags, `bcsa_verify_8q/*check_verify*`).
  "Agree" counts rows where the verify argmax equals the next greedy token.

  | prompts | rows | path | KL mean | KL p99 | KL max | argmax agree |
  |---|---|---|---|---|---|---|
  | one per category | 1520 | new | 3.39e-4 | 3.85e-3 | 1.18e-2 | 1512 |
  | one per category | 1520 | old | 3.46e-4 | 4.53e-3 | 1.04e-2 | 1515 |
  | long + agent tune prompts (4.5-12.5K) | 848 | new | 3.26e-4 | 3.00e-3 | 5.90e-3 | 844 |
  | long + agent tune prompts (4.5-12.5K) | 848 | old | 3.97e-4 | 3.93e-3 | 1.27e-2 | 842 |

  New and old are at the same noise floor, with no rollback breakage.
- **Free-running** (`specdec.py check`, frozen config, one tune prompt per category, 128 tokens):
  - 9/14 identical to greedy (old config: 6/14).
  - All 5 divergences start where greedy's top two logits are an exact bf16 tie (margin 0.0).
  - Agent and long now match fully.
  - In the benchmarks every run's greedy tokens equal yesterday's (56/56 tune, 84/84 test). The spec outputs differ
    from greedy only at ties, which is the existing property.

## Results

Protocol: spec/NOTES.md. Frozen config: 4-bit drafter, 4-bit head copy, NAX verify, lookup, 256 new tokens. Every
run held the GPU lock, with the canary before and after; canaries read 22.8-28.5 TFLOP/s. The machine's load average
was 5-18 from other apps.

### Tune split, same-day A/B: old path vs new path

Interleaved: stage a ran new then old on all 56 prompts; stage c ran old/new on long+agent twice more. Greedy ran
between the stages.

| category | greedy tok/s | old tok/s | new tok/s | new/old | old ms/step | new ms/step | tokens/step old / new |
|---|---|---|---|---|---|---|---|
| **long (8-12.5K), 3 runs** | 9.8 | 14.94 | **19.19** | **1.28** | 126.4 (median) | **108.8** | 1.87-1.91 / 2.09 |
| **agent (4.5-7.3K), 3 runs** | 10.3 | 32.41 | **39.49** | **1.22** | 125.6 (median) | **104.9** | 4.25-4.34 / 4.24-4.34 |
| coding | 11.8 | 28.8 | 27.8 | 0.97 | 105.4 | 105.6 | 3.06 / 2.96 |
| humanities | 11.4 | 21.8 | 21.3 | 0.98 | 105.2 | 107.2 | 2.30 / 2.29 |
| json | 11.2 | 36.1 | 35.5 | 0.98 | 109.8 | 112.1 | 3.98 / 4.02 |
| math | 11.2 | 37.9 | 33.3 | 0.88 | 104.7 | 115.2 | 3.99 / 3.87 |
| multilingual | 11.1 | 21.4 | 20.6 | 0.96 | 104.3 | 109.5 | 2.26 / 2.28 |
| qa | 10.4 | 20.0 | 19.7 | 0.98 | 103.9 | 106.0 | 2.09 / 2.09 |
| rag | 11.2 | 22.2 | 21.7 | 0.98 | 102.5 | 103.9 | 2.29 / 2.27 |
| reasoning | 12.0 | 28.3 | 29.3 | 1.04 | 104.1 | 103.6 | 2.98 / 3.05 |
| roleplay | 11.6 | 21.1 | 21.8 | 1.03 | 113.6 | 104.9 | 2.41 / 2.31 |
| stem | 12.8 | 21.4 | 22.3 | 1.04 | 105.1 | 106.0 | 2.26 / 2.37 |
| summarization | 12.0 | 22.9 | 23.9 | 1.05 | 111.2 | 108.2 | 2.57 / 2.62 |
| writing | 9.4 | 22.7 | 23.1 | 1.02 | 107.1 | 105.8 | 2.44 / 2.45 |
| **all 56 (stage a)** | 11.1 | 23.32 | **23.90** | **1.025** | 109.2 | 107.4 | 2.56 / 2.59 |

How to read the long and agent rows:
- They pool the stage-a run and the two stage-c runs, 12 prompt-runs per config.
- Per run, long + agent above 4K went 18.09 / 18.02 / 18.53 tok/s on the old path and 22.59 / 23.19 / 23.72 on the
  new one.
- Of the long gain, about 16% is step time. The rest is acceptance (1.88 -> 2.09 tokens/step): the two paths break
  bf16 ties differently and so follow different, equally valid greedy paths. Acceptance is identical across repeats
  of the same config.

How to read the short categories:
- Over the 48 prompts under 4K, the pooled step is 106.3 ms old and 106.9 ms new.
- Per prompt, new minus old has a median of +1.0 ms, with a spread of -20 to +26 ms.
- `prof_bcsa_verify.py` at 197, 1.5K and 3.5K tokens, 2 runs each way, shows no consistent difference; the new path
  is ahead at 3.5K (`bcsa_verify_8q/bcsa-verify-8q_prof_short.jsonl`).
- Math's 0.88 is one category of 4 prompts with run-to-run noise of this size (roleplay goes the other way, 1.03).
  I added no gate by context.

The greedy column is today's greedy run. It ran at 11.1 tok/s against 13.81 yesterday, with identical tokens. CPU
load from other apps slows greedy's pipelined loop more than spec, so "x greedy" ratios on tune are inflated today
(new: 2.15x against today's greedy, 1.73x against yesterday's). The new/old column does not depend on greedy.

### Held-out test split: once, frozen config (new path), against greedy on the same day

`runs/bcsa-verify-8q_test_8q.jsonl` against `runs/bcsa-verify-8q_test_base.jsonl`.

**26.09 against 13.17 tok/s, 1.98x pooled, 2.12x geo-mean of per-prompt speed-ups, 2.63 tokens/step.**

| category | greedy tok/s | spec tok/s | x | tokens/step |
|---|---|---|---|---|
| math | 14.0 | 42.0 | 3.00 | 3.96 |
| json | 14.5 | 43.3 | 2.99 | 4.19 |
| **agent (3-7.9K)** | 12.2 | **35.9** | **2.95** | 4.09 |
| rag | 13.3 | 29.2 | 2.19 | 2.95 |
| coding | 14.0 | 30.2 | 2.16 | 3.08 |
| summarization | 13.0 | 26.2 | 2.01 | 2.64 |
| roleplay | 11.1 | 22.2 | 2.00 | 2.22 |
| multilingual | 14.0 | 27.4 | 1.96 | 2.60 |
| stem | 12.4 | 23.9 | 1.93 | 2.39 |
| reasoning | 13.3 | 24.3 | 1.83 | 2.41 |
| writing | 13.3 | 24.0 | 1.80 | 2.48 |
| **long (14-21.5K)** | 11.7 | **20.9** | **1.79** | 2.23 |
| qa | 13.4 | 21.8 | 1.63 | 2.22 |
| humanities | 14.4 | 21.4 | 1.48 | 2.19 |
| **all** | **13.2** | **26.1** | **1.98** | **2.63** |

- **Step time on the held-out long prompts:** median 105.6 ms; per prompt 100.9-113.6 ms at 14-21.5K. On the agent
  prompts the median is 104.6 ms.
- **Reference from a different day (not a same-day A/B):** yesterday's old-path held-out runs (`runs/test_frozen.jsonl`,
  `test_frozen_r2.jsonl` against `test_base*.jsonl`) gave 23.4-24.5 tok/s pooled, long 16.3 tok/s at 1.29x, agent
  30.8-31.9 tok/s, and a long median step of 142.5 ms.
- **Skipped:** a same-day held-out run of the old path was started and then skipped at the user's request. Its
  partial file, `runs/bcsa-verify-8q_test_prefill_skipped_partial.jsonl`, is not a result.

## Not done / next levers

- **Further kernel work, about 1 ms per step left:**
  - The attention streams the window at ~160 GB/s against ~250 achievable, worth ~0.6 ms per step.
  - far_union could be skipped when every valid block is selected (nv <= 512) or fused into the top-k, worth
    ~0.4 ms.
- **Not touched:** volundr.py's prefill and decode paths. Their `[0]` copies are listed above.
- **Not done:** bit-exact greedy, which needs batch-invariant kernels.

## Files

- `bcsa_verify.py`: the kernels (`attend`).
- `spec/verify.py`: the verify branch (`_bcsa_verify`, `BCSA_IMPL` / `VOLUNDR_BCSA_VERIFY`).
- `tests/test_bcsa_verify.py`: comparison against fp64 and against the old kernels.
- `spec/prof_bcsa_verify.py`: the profile above (`--impl`, `--max-tokens`).
- `spec/bench_bcsa_attend.py`: kernel micro-benchmark.
- `spec/bcsa_verify_8q/`:
  - run helpers: `gpurun.sh`, `tune_chain.sh`, `test_chain.sh`, `stepms.py`, `klsum.py`;
  - logs and summaries (`tune_summary.txt`, `test_summary.txt`);
  - copies of this branch's `runs/bcsa-verify-8q_*` files.
