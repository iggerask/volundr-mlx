Status: done (2026-10-20). Continuous batching in the MLX worker behind agens-server, with batched
speculation for 2-4 greedy requests. Server mode, 19 prompts x 256 tokens: 24.0 / 38.0 / 56.1 / 64.2 / 86.2 tok/s
aggregate at 1 / 2 / 4 / 8 / 16 concurrent (upstream: 87.7 / 129.3 / 155.1 at 1 / 2 / 4).

# Batched serving: continuous batching in the MLX worker (branch batched-serve)

## Resume

Branch `batched-serve` from `batched-bench` (6d692bb), worktree `.claude/worktrees/agent-a3bd58d97e91bdf21`. Data
links as in the brief (.venv, models, ref, spec/suite.json, spec/runs, parity/*.bin). Measured code: c79ffdd (the
server-mode run `batched_serve/run2/`). After it, the code changes were:
- the worker warm-up gained a plain batched pass at 2-4 rows (checked with `batched_serve/warmup_check.py`:
  6.5 s with `--short-warmup`; 34 plain batched, 4 batched speculative and 6 speculative steps);
- `serve_bench.py` counts each streamed delta as its request's tokens per delta in the full-batch window;
- `report.py` prints the new columns;
- comments.

- Code: `batch.py` (slot caches, per-row positions), `batch_spec.py` (batched speculation), `scheduler.py`
  (BatchEngine), `worker.py` (runs it), `volundr.Attention._batched`, `nax_hook.min_rows`, `session.prefill_iter`,
  `verify.forward_iter`, `specdec.SpecStepper`; `server/volundr-server` (doc line and a concurrency test).
- Gates: `batched_serve/gates.sh <modes>` under `./with_gpu_lock.sh` (modes tests, exact, rows, timing, timing8k,
  engine, bspec, bspecn, serve, prof, profeng); logs and jsonl in `batched_serve/`. Regression chain:
  `batched_serve/regress.sh`.
- Server runs: `./with_gpu_lock.sh batched_serve/step.sh serve_bench.log batched_serve/serve_bench.sh <outdir>`
  (env PHASES = server,mixed,compare,ties; LEVELS), then `batched_serve/report.py <outdir>` for the tables.
- Next, if resumed: the items under "Limits and next steps". The first two are a batched drafter (batched
  speculation past 4 rows) and per-row (paged) BCSA / KV buffers.

## Results

Server mode through agens-server (OpenAI API, streaming): 19 short prompts of our own (46-59 tokens with the chat
template, nothing from spec/suite.json), 256 new tokens each (`ignore_eos`), greedy, thinking off, a pool of c
clients, session cache cleared before each level. M5 Pro, 64 GB, MLX 0.32.4. Run `batched_serve/run2/`
(`serve_bench.log`; canary before / after each level).

| concurrency | aggregate tok/s | upstream | ours / upstream | per-request tok/s (mean) | TTFT p50 / p95 / max s | mean batch | peak GB | canary before / after |
|---|---|---|---|---|---|---|---|---|
| 1 | **24.0** | 87.7 | 0.27 | 25.0 | 0.24 / 0.25 / 0.26 | spec | 34.5 | 26.3 / 27.6 |
| 2 | **38.0** | 129.3 | 0.29 | 20.1 | 0.25 / 0.55 / 0.58 | 2.0 | 34.5 | 25.8 / 27.3 |
| 4 | **56.1** | 155.1 | 0.36 | 15.3 | 0.25 / 0.94 / 1.29 | 3.9 | 35.4 | 27.0 / 27.5 |
| 8 | **64.2** | - | - | 10.0 | 0.53 / 2.58 / 2.97 | 7.6 | 36.0 | 26.6 / 27.7 |
| 16 | **86.2** | - | - | 8.1 | 2.51 / 6.42 / 6.88 | 14.8 | 39.0 | 27.7 / 26.9 |

- Against our own c = 1: x1.58 at c = 2 and x2.34 at c = 4; upstream's are x1.47 and x1.77. The absolute gap is
  mostly the hardware: every decode step reads all the INT4 weights, and at 307 GB/s a B = 1 step takes 70-74 ms
  (timing gate).
- Aggregate = all completion tokens / (first request sent .. last token). With 19 requests at c = 16, three run
  as a tail at B = 3 after the first 16 finish together (equal lengths, `ignore_eos`). While all 16 streamed the
  rate was 129.8 tok/s, and 80.5 tok/s with all 8 at c = 8. Decode alone at B = 16 is 142 tok/s (timing gate).
- Modes: c = 1 is the speculative loop (`spec`). At c = 2 and 4 the requests run batched speculation, then a
  speculative tail ("mixed" = more than one mode in a request). From c = 8 they run plain batched decode, plus
  batched speculation once 4 or fewer remain.
- The run before batched speculation and before prefill slices could span requests (`run1/`, dc10542):
  26.0 / 26.3 / 40.9 / 58.8 / 71.9 tok/s, with TTFT p95 10.7 s at c = 16. c = 1 is the same code path in both
  runs, so its 26.0 against 24.0 shows the run-to-run spread on this machine.
- Completions against c = 1 (`run2/compare.jsonl`): 7/19, 3/19, 5/19 and 7/19 identical at c = 2, 4, 8 and 16.
  The others first differ at tokens 1-233. Every first difference is at a near-tie (`run2/ties.jsonl`, the
  B = 1 decode logits after c = 1's prefix). Over the 54 first differences, the two tokens' logits are within
  0.375 of each other: 23 are exact ties and 48 are within 0.125 (one bf16 step for logits of 16-32). c = 1 is
  the spec path, whose NAX verify already rounds differently from B = 1 decode. Batched rows add the
  row-count-dependent matmul kernels ("Numerics" below).

Mixed run (`run2/mixed.jsonl`): an agent session (session cache on). Turn 1 alone was 15,586 tokens, prefilled
in 40.1 s. Turn 2 restores turn 1's reply snapshot (15,635 tokens) and prefills ~4.6K new tokens (one 13K-character
file) while 4 short requests decode (384 tokens each, batched speculation at B = 4). Two schedules were run, each
with a different file of the same size:

| schedule | agent TTFT | short streams during the agent prefill | longest gap in the short streams | after the agent joins (B = 5) |
|---|---|---|---|---|
| layer slices (200 ms of prefill, then a decode step; default) | 19.5 s | a step every 0.37-0.50 s, ~2 tokens per step (~4.7 tok/s) | 4.17 s, when the agent's prefill ends and it joins | 7.0 tok/s each |
| whole prefill (slice 10^7 ms) | 12.3 s | nothing | 13.97 s | 7.0 tok/s each |

Slicing turns a 14 s stall into steady progress at about a third of the speed (one step per 200 ms slice plus
the step itself), at a cost of +7 s on the agent's TTFT. A 4.2 s stall is left at the end of the agent's prefill.
In the whole-prefill run the same span (prefill end, join, first steps) was ~1.8 s (13.97 s gap against a 12.1 s
TTFT). It was not profiled. The likely parts:
- the prompt snapshot;
- the join, where the row at 20K positions widens every slot's BCSA buffers (ring 4,096 + 512, pooled blocks
  for 20K) and the dense KV to 20K;
- the first steps, which attend over the longest row for every row.

Peak memory there was 52.8 GB: the weights, the session cache at its 12 GB budget, and slot buffers sized
S x the longest row.

## Design

### Slots and per-row positions (`batch.py`)

A `BatchCache` is the model cache of the decode batch: one row per decoding request, rows 0..B-1, each at its own
position `pos[b]` (the tokens it holds, shared by every layer through `BatchRows`).

| state | layout | a step | a row joins / leaves |
|---|---|---|---|
| 54 KDA layers | `ArraysCache` [B, ...] (conv [B, 3, 10240], recurrent [B, 48, 128, 128] fp32) | the per-row decode kernels rebind them | one gather of the kept rows + the new rows (0.17 GB per row), once per membership change |
| Engram | `EngramState.prev` [B, 2] | rebound | same |
| 17 BCSA layers | `BatchBCSACache`: ring K/V [S, 4, R, 256], pooled K/V [S, 4, cap, 256], pooled indexer keys [S, cap, 128], open-block keys [S, 4, 128] | scatters in place at each row's slots | swap-remove: the last row is copied into the hole (one row); a new row is written into slot B |
| dense layer | `BatchKVCache` [S, 4, cap, 256] | scatter at pos[b] | same |

- **S slots** double as rows join (1, 2, 4, ... 16) and are freed when the batch empties.
- **Ring width R** follows the longest row: 256, 512, 1024, 2048, then the full 4096 + 512 (window plus the far
  scratch) once a row passes 2048 positions. At 200 tokens a row costs 0.21 GB, against 0.57 GB with full rings
  (NOTES_BATCHED). Pooled blocks grow in steps of 256 blocks, dense K/V in steps of 256 positions.
- **What a step does per BCSA layer** (`Attention._bcsa_batched`):
  - RoPE with the [B] offsets. `mx.fast.rope` takes an array offset and gives the same values as per-row scalars
    (checked).
  - The ring write at slot pos % W, and the open block's indexer key at pos % C.
  - Rows past the window gather their far blocks into the scratch rows behind their ring: every valid block up to
    top-k, else the row's own top-k. These use the B = 1 scoring and radix-select kernels with the row's position
    and block count.
  - Blocks the step closes (rows at pos % C = 3) are pooled from the ring and appended at index pos // C.
  - Then one SDPA over the longest row's keys, with the other rows masked to their own prefix. A row's valid keys
    are always a prefix of its buffer (ring, then far blocks), so the mask is a length mask.
- **Write order.** Each buffer is read only at its last version within a step. The exception is the pooled
  blocks: the far gather reads them before the close appends to them, and that read is in the logits graph,
  which is evaluated first. Otherwise a pending read of an older version makes MLX copy the whole buffer instead
  of writing it in place.
- **Snapshots.** `capture_row` writes a row in `session.State`'s layout: the compact rows a B = 1 cache holds at
  that position. So the session cache serves batched requests as it serves bs = 1 ones, and `row_cache` (restore
  of that) hands a row back to the speculative loop.
- **Numerics.** Every op works per row with the B = 1 arithmetic.
  - A batch of one row is `session.decode_step` bit for bit (logits and drafter features) on the full model at
    150, 3,000, 4,200 and 9,000 tokens (gate `exact`).
  - In the tiny model, rows that join and leave at different lengths equal their B = 1 runs bit for bit (its
    bf16 matmuls are row-invariant at those sizes).
  - At B > 1 on the full model, the matmul kernels change with the row count (MLX's bf16 matmul; stock or NAX
    4-bit). The masked SDPA can also switch between MLX's one-pass and two-pass kernels (two-pass from 1,024
    keys). So rows match their B = 1 runs up to bf16 rounding, and greedy text differs only at near-ties.
- **Pipelining.** When every row is greedy, the next step's graph is built while the current one runs. Its input
  is this step's argmax, still on the GPU, and the graph is used if the batch is unchanged. Otherwise the caches go
  back to the state saved before it was built (`prebuild` / `take_prebuilt` / `rollback`; any membership change
  or snapshot rolls back first). It gives the same tokens as the synchronous step and is 3-9% faster at B = 1-8
  (gate `timing`).

### Batched speculation (`batch_spec.py`)

While 2..4 greedy requests decode (`--bspec-max`, 4), each step is a batched speculative step:
- **Draft.** Each row drafts from prompt lookup, else from the DFlash2 drafter on its own context.
- **Verify.** One verify forward over the B x T rows, with T = 16 // B (8, 5, 4), so at most 16 rows, which NAX takes.
  - KDA runs the verify recurrence per row.
  - The dense layer uses a per-row causal mask at each row's positions.
  - BCSA uses `bcsa_verify.attend` per row, with the row's far-block union.
- **Commit.** Each row keeps its own accepted run, n_b tokens, in one commit:
  - KDA replays the T steps with the decay forced to 1 and beta to 0 past n_b, so padded steps leave the state
    alone (`gated_delta_vec` takes the decay, not its log);
  - conv states and Engram are taken per row at n_b;
  - BCSA appends each row's n_b keys to its ring and blocks: ring writes for every row first, then the pools.
  Drafter contexts get the kept rows' features.

Against the B = 1 verify + commit on the same states and drafts (gate `bspec`, 2 rows at 300 / 5,200 tokens, 24
steps, random accepted counts), logits differ by at most 0.52 (bf16) with KL <= 0.0063, and argmax agrees on
375/384 rows.

| requests (gate `bspecn`, 128 tokens) | T | batched spec tok/s | plain batched tok/s | tokens per row-step | ms per step |
|---|---|---|---|---|---|
| 2 | 8 | **40.8** | 25.3 | 2.92 | 118 |
| 3 | 5 | **52.7** | 33.9 | 2.35 | 128 |
| 4 | 4 | **53.7** | 43.6 | 2.03 | 137 |
| 6 | 2 | 48.3 | **57.0** | 1.43 | 166 |
| 8 | 2 | 57.9 | **68.5** | 1.49 | 197 |

Past 4 rows the drafts get too short (T = 2) and the per-row work (B drafter calls, per-row BCSA verify) outweighs
them, so plain batched decode takes over.

### The scheduler (`scheduler.py`, run by `worker.py`)

After upstream's server mode (`agens-cli/src/serve.rs`). One engine thread owns the GPU; each pass:
1. **Admit**: new requests, while fewer than `--max-batch` (16) are active and MLX holds less than
   `--max-active-gb` (46). A request restores its longest snapshot into its own B = 1 caches (or starts fresh).
2. **Prefill work**, on the request's own B = 1 caches with the bs = 1 prefill (`session.prefill_iter`: the same
   chunks, interval and prompt snapshots, drafter context):
   - nobody decodes: one whole chunk (2048 tokens) of the oldest prefilling request;
   - others decode: prefill layers until `--slice-ms` (200 ms) have passed, then the decode step
     (`verify.forward_iter` yields after each layer's sync). A short prompt that ends early hands the rest of the
     slice to the next prefilling request. The chunking, and so the numerics, do not change with load; only where
     the work pauses.
   - At the end: the prompt snapshot, the first token (argmax or the sampler), and the request decodes.
3. **One decode step** for every decoding request:
   - **one decoding request, greedy**: one step of the speculative loop (`specdec.SpecStepper`, the bs = 1 loop:
     DFlash2 4-bit + head copy, NAX verify, prompt lookup, early@cv);
   - **2-4, all greedy**: one batched speculative step (above);
   - **otherwise**: all of them are rows of the batch, in one batched plain decode step. NAX small-M is used from
     4 rows (`nax_hook.min_rows`; stock MLX is faster at 2-3 rows). Greedy rows take the GPU argmax; sampled rows
     use `sampler.py` on their logits row (per-row seeds, penalties, budget).
   - **Switching.** A speculative stream that gets company is finalized (the pending commit applied: the caches hold
     ids + out[:-1]) and joins the batch. If the last row left is greedy, it leaves the batch (its row copied out)
     and continues speculatively. Its drafter context is filled from the target features the batch captured for
     it, materialized every 64 tokens.
4. **The end of a reply**: the reply snapshot (the row's `capture_row`, or the speculative state), the row leaves
   (swap-remove, several rows in one gather), the done record.

Cancellation is checked on every pass:
- a queued request is dropped;
- a prefilling request stops between prefill layers;
- a decoding request stops after the current decode step.

Interval snapshots and the reply snapshot are kept as before. Requests and control messages (`clear`, `budget`,
`stats`, `reset_peak`, `config`) arrive on connection threads; all MLX work runs on the engine thread. The protocol
is unchanged (version 1). `stats` gains `active`, `batch_rows`, `engine` (step counts and a time split) and
`peak_gb`. `config` sets slice_ms, max_batch, spec, pipeline, bspec and bspec_max at run time (for measurements).

### The Rust side

agens-server calls `Backend::generate` once per HTTP request, and `MlxBackend` opens one worker connection per
request on its own task. So concurrent requests were never serialised on the Rust side; the worker was. A new cargo
test (`concurrent_requests_stream_together`, 4/4 pass) checks that two requests reach a fake worker at once and
stream interleaved.

## Gates

| gate | what | result |
|---|---|---|
| `tests/test_batch.py` (tiny model) | B = 1 batch = decode_step; join / leave at different lengths; buffer growth; prebuilt step + rollback; batched-spec ring / pool append; slots | 6/6 |
| regression chain (`regress.sh`, final code) | test_batch, test_session, test_volundr, test_decode_fused, test_bcsa_verify; `spec/check_combined.py` against the spec baseline (the bs = 1 spec and prefill paths) | 6/6, 8/8, 10/10, 8/8, ok; ALL BIT-IDENTICAL |
| `exact` | B = 1 batch path against `session.decode_step`, 48 steps at 150-9,000 tokens; capture_row continues bit-exact | bit-identical logits and features |
| `rows` | 8 rows joining at steps 0-24 at 120-9,000 tokens, 96 steps, against their B = 1 runs | 6/8 identical; 2 differ at near-ties (B = 1 top-2 margin 0.125) |
| `timing` | ms per batched step at 200 tokens, sync / pipelined | B = 1 73.8 / 69.8, 2 77.6 / 74.9, 4 90.6 / 82.1, 8 99.2 / 94.5, 16 112.5 / 111.7 (143 tok/s) |
| `engine` | 6 requests + a cancelled one, sessions, turn 2 from reply snapshots; concurrent against sequential | cancellation and reply snapshots work (turn 2 restores them); greedy differences at near-ties (top-2 margins 0-0.125); f1's turn-2 prompts differed because its turn 1 had (the gate now marks that); the sampled x4 differs from token 70 |
| `bspec`, `bspecn` | batched speculation (above) | above |
| `prof` | B = 8 / 16 steps with equal vs mixed positions, with / without drafter feature capture; the engine at B = 16 | no difference (94 / 112 ms); the engine's full-batch steps 112 ms |

## Where the time goes at high concurrency

- **Prefill is serial and close to compute-bound.** A 50-token prompt costs ~0.2 s at B = 1 (TTFT 0.24 s at
  c = 1). 50 tokens x 64 GFLOP is 3.2 TFLOP, ~0.12 s at the canary's 27 TFLOP/s, so batching prefills would gain
  at most ~40%. At c = 16, the 16th request waits for 15 prefills and the decode steps between them: TTFT p95
  6.4 s (10.7 s in run1, where each tick prefilled at most one request).
- **Membership changes.** A join or leave regathers the KDA state of every row (0.17 GB each), and the step after
  it is synchronous. Such a step costs 100-180 ms more than a plain one (`gate_prof.jsonl`, step logs). One try
  kept MLX's buffer cache across short prefills (`clear_min = 1024`), to save fresh allocations, and its join
  steps took 0.7-1.7 s. A rerun with the default also had 0.5-1.0 s join steps, though, while swap grew to 27 GB.
  Memory pressure from the machine is the more likely cause, and the setting is not settled. The engine clears the cache after
  every prefill chunk, as at bs = 1 (`clear_min = 0`).
- **The tail.** Requests of equal length finish together, so a level of 19 ends with 3 at c = 16 (B = 3, batched
  speculation) and 3 at c = 8.
- Steady state is fine: with all rows decoding, the engine's step is the gate's (112 ms at B = 16). Masks for
  rows at different positions and the drafter feature capture cost nothing measurable there.

## Memory

- A batch row at 200 tokens: 0.21 GB (KDA 0.17 GB + small rings). Peaks in the server-mode run 34.5-39.0 GB
  (weights, prompt snapshots in the session cache, batch). No OOM; nothing backed off.
- The mixed run peaked at 52.8 GB. Slot buffers are sized S x the longest row. With one row at 20K positions and
  S = 8 slots, the buffer shapes give ~6 GB of ring + pooled blocks + dense KV, against ~1 GB of rows' contents.
  The admission guard (`--max-active-gb` 46) stops new requests at that level but does not shrink what is there.
- Swap was nearly full from other apps throughout, and grew to 27 GB during the profile gates.

## Limits and next steps

- **Batched speculation past 4 rows.** It needs one batched drafter forward for all rows instead of B drafter
  calls, a T per row from its acceptance, and NAX for more than 16 verify rows. At B = 4 the step is 137 ms,
  against 112 ms for 16 plain rows.
- **Per-row (paged) BCSA / KV buffers.** Today every slot is as wide as the longest row, and the SDPA computes
  every row over the longest row's keys (masked). That is the 4.2 s stall and the ~6 GB in the mixed run. Pages
  per row would fix both, and allow parking an idle session's row (or spilling it to SSD) instead of snapshotting
  it.
- **Batch invariance.** Greedy text at B > 1 differs from B = 1 at near-ties. Two kernel choices depend on the
  batch: the matmul kernel by row count (stock bf16 / NAX 4-bit), and SDPA one-pass / two-pass by key count.
  Kernels with a fixed reduction order per row would make batched output equal B = 1 output; the cost is unmeasured.
- **Joins.** Pending rows could be joined in groups while other requests are still prefilling. The KDA state
  could also live in slot buffers, so a join writes one row instead of regathering all of them.
- **TTFT under load.** It is bound by prefill FLOPs (above). The W4A8 prefill (+25-32%) would apply here.
- Sampled requests decode plainly, alone or in the batch: there is no speculation for sampling.
