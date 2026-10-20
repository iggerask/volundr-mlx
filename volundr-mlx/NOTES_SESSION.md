# Session / prefix cache and serving through agens-server (branch session-cache, 2026-10-20)

Status: done (2026-10-20).

## Resume

Nothing is pending. Branch `session-cache` from main (66c00f2), committed in a clone of the llms repo at
`/Users/ingemarrask/personal-dev/llms/volundr-mlx/ref/agens-engine/.claude/worktrees/agent-ab9a6f81b8d270d5c/volundr-mlx-session-cache`.
The harness created this agent's worktree in the agens-engine repo, and its guard refuses git commands on the shared
llms checkout, so the coordinator fetches `session-cache` from the clone. Data links as in the brief: .venv, models,
ref, spec/suite.json, spec/runs and parity/*.bin.

Every GPU step ran as `session_cache/step.sh <log> <cmd>` (canary, command, canary, load, swap) under
`./with_gpu_lock.sh`, from `volundr-mlx`, with `PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python`.
- `session_gate.py`: gates 1-4 (`session_cache/gate1-3.log`, `gate*.json`). `session_cache/chunk_invariance.py`
  (`gate4.log`, `gate5.log`).
- `session_cache/regress.sh`: the existing tests and check_combined (`regress.log`).
- `session_cache/e2e.sh`: serve.sh plus `session_e2e.py run / replay / checks / report`. Run 2 is in
  `session_cache/e2e/`, run 1 in `session_cache/e2e_run1/`. `session_cache/tie_check.py` reads `e2e/worker.jsonl`
  (`tie_check.log`).

Spot checks of the scratch work aren't committed: the MLX copy-on-write probe and the spec-loop equivalence check
against main's specdec.py. Their results are quoted below.

## Summary

- **A follow-up turn prefills only its new tokens.** The table below is an agent session through the OpenAI API
  (agens-server + the MLX worker): system prompt, three tool definitions, and repo files as tool results, growing to
  21.7K tokens.
  - At 21.7K tokens, time to first token is 9.1 s against 52.7 s without the cache (5.8x); that turn added 3,483
    new tokens.
  - At 18.0K tokens, with 1,196 new tokens, it's 3.1 s against 43.6 s (13.9x).
  - Every turn resumed from the previous turn's end-of-reply snapshot. TTFT is now the new tokens at ~390 tok/s.
- **Exact where it should be.**
  - Restore-then-continue is bit-identical to continuing the original cache: prefill chunks, plain decode steps,
    verify and commit with random rollback, drafter proposals, every final cache array, two forks, and the snapshot
    is unchanged afterwards.
  - A resumed prefill is bit-identical to a fresh prefill with a chunk boundary at the resume point.
- **Resumed vs fresh is chunking noise.** Teacher-forced KL is at the noise floor of changing the prefill chunk size,
  measured on the same rows: 4.2e-4 against 4.6e-4, and 5.8e-4 against 5.5e-4. Greedy output differs only at
  near-ties. Speculative decoding from a restored state equals plain greedy from it. The sections below explain where
  the KL comes from: BCSA prefill attention rounding depends on the chunk start.
- **Upstream server unchanged.** `server/` uses agens-server as a git dependency (rev b0e9a95) and adds
  `MlxBackend` and a `main`. `cargo build --release` works with cargo 1.87.
- **One command:** `./serve.sh`.
- **Default budget 12 GB.** The measured peak with a full cache was 40.6 GB, against 29.1 GB without it.

## Design

### What a snapshot is (`session.State`)

The state after the tokens `ids[:P]` were fed:

| part | contents |
|---|---|
| 54 KDA layers (`ArraysCache`) | conv state [1, 3, 10240] bf16, recurrent state [1, 48, 128, 128] fp32 (0.17 GB in all) |
| 17 BCSA layers (`BCSACache`) | ring K/V (4096 window + 512 far-scratch rows), pooled far-block K/V, pooled indexer keys, the open block's indexer keys, offset, block count |
| dense layer (`KVCache`) | K/V buffer + offset |
| `EngramState` | the last two token ids |
| drafter (`dflash.DraftCache`) | context K/V ring (4096 slots x 5 layers), valid for the last 2048 positions |
| logits | the last position's row, which picks token P when a prompt ends exactly at P |

Prompt lookup needs no state. As in upstream `agens serve`, each request builds a fresh `Lookup` from its prompt.
`lookup_state` / `restore_lookup` exist for a restore in the middle of a reply.

### References or copies (checked per cache type)

"MLX arrays are immutable and cache updates rebind them" holds only for part of the state:

| array | how the cache updates it | the snapshot keeps |
|---|---|---|
| KDA recurrent state, Engram ids | rebinds a new array | a copy (small) |
| KDA conv state | rebinds, but after a prefill chunk it's a 3-row slice view of the chunk's qkv projection (42 MB per layer at 2048 tokens) | a copy: a reference would pin 2.2 GB of projections per snapshot |
| BCSA ring and pooled blocks, dense K/V, drafter ring | in place, `a[idx] = x` | `copy.copy(a)`: a new array object on the same buffer |
| BCSA ring below 4096 positions, pooled blocks and drafter ring when at most half used | in place | only the used leading rows (`session.Rows`), padded back with zeros on restore |

- MLX's Python `__setitem__` replaces the contents of the array object it's called on. A snapshot holding the same
  object would see later writes; a `copy.copy` doesn't. MLX then won't donate the shared buffer to the next slice
  update, so it copies it once (copy on write), and later writes are in place again.
  - Measured on a 1 GiB array: 32 ms for the first update after a snapshot, then 4.5 ms per update in place
    again.
- `restore` hands out `copy.copy` objects too, so a request never changes the snapshot it started from. Two requests
  sharing a prefix fork from it independently.
- The compaction is exact: those rows haven't been written yet. The far-scratch rows fill only once the ring is full,
  and pooled rows only up to the block count.
- Arrays under 4 MiB are copied. The larger ones are allocated buffers, never views.

`tests/test_session.py` checks all of this on the tiny model. A mutation check (aliasing instead of `copy.copy`, or
referencing the conv-state view) makes its tests fail.

Measured snapshot sizes:

| tokens | 26 | 664 | 1,205 | 4,096 | 6,151 | 11,531 | 17,218 | 23,294 |
|---|---|---|---|---|---|---|---|---|
| GB | 0.18 | 0.26 | 0.31 | 0.67 | 0.76 | 0.85 | 1.03 | 1.13 |

Above 4K tokens a snapshot grows by about 22-27 KB per token: pooled BCSA blocks plus dense K/V.
- A capture takes 3-6 ms when it references the buffers, as every snapshot past 4K tokens in the agent session
  did. A compacted short snapshot copies its rows: 15-60 ms in the checks.
- A restore takes 0.1-1.2 ms.
- The live cache pays one copy of each shared buffer at its first write (~30 ms per GB).

### Snapshot points

- **End of the prompt prefill** ("prompt", with its logits).
- **End of the reply** ("reply"). The caches then hold prompt + reply minus its last token.
  - The spec loop leaves the last step's commit pending, and may have accepted past an eos or max_new_tokens.
    `specdec.spec_decode(finalize=True)` applies that commit and the drafter-context update for exactly the tokens
    kept.
  - Sampled decoding keeps the drafter features of every token it feeds and fills the drafter ring at the end, so
    the snapshot also serves later greedy (speculative) requests.
- **Every 4096 tokens inside a prefill** ("interval"), at absolute multiples, so requests that share a prefix share
  the points. Points within 256 tokens of either end are skipped, since the resume point and the prompt snapshot
  cover them.
- **Drafter ring.** The prefill fills it for the 2048 positions before every point, as spec_prefill does for the
  prompt's end, so a request can decode speculatively from any snapshot.

**The chat template** (`chat_template.jinja`, checked):
- `preserve_thinking` is unset by default, which counts as true: every assistant turn is re-rendered as
  `<|think|>\n{reasoning|trim}\n<|/think|>\n\n{content|trim}`.
- `preserve_thinking=false` drops the reasoning only of assistant turns before the last real user query. Turns in
  the current tool loop keep it.
- Tool calls are re-rendered from the parsed arguments in a fixed format; content and reasoning are trimmed.

In the e2e session the client sent replies back as the server returned them (content, reasoning_content,
tool_calls). The re-rendered history then matched the generated tokens through the reply's second-to-last token every
time, so every turn resumed from the end-of-reply snapshot. That includes turn 2, which hit max_tokens in the middle
of its thinking.

A client that drops reasoning_content or reformats whitespace diverges inside the previous reply. Then the end-of-
prompt snapshot is the fallback, and only the previous reply is prefilled again.

### Lookup, fork, eviction

- **Lookup.** `SessionCache.find` returns the longest snapshot whose tokens are a prefix of the prompt.
  - The request's own session wins ties. Snapshots from other sessions are considered too, which makes this a
    prefix cache.
  - A snapshot exactly as long as the prompt counts only if it carries logits.
  - It's a linear scan with numpy comparisons of the token ids.
- **Fork.** A resumed request works on restored copies (see above). Two requests sharing a prefix start from the same
  snapshot, and neither changes it.
- **Memory.** Each snapshot's bytes are counted exactly (arrays plus token ids). LRU eviction runs under
  `VOLUNDR_SESSION_CACHE_GB` (`worker.py --cache-gb`, or a `budget` protocol message at run time; 0 disables the
  cache).
- **Default 12 GB.**
  - Weights take 24.2 GB, and the speculative working set at 20K is ~4-5 GB.
  - With a full cache the peak was 40.6 GB (21.7K-token turn, 11.9 GB of snapshots). That leaves ~23 GB of the
    64 GB for macOS and the other apps; swap held 11-14 GB throughout.
  - 12 GB holds ~10 snapshots at 20K: the last 3-4 turns of a long agent session (prompt, reply and interval
    snapshots), or the latest turn of ~5 conversations.
  - Spilling to SSD isn't done.

### Requests (`session.Engine`, run by worker.py)

A request runs in this order:
1. Find the snapshot and restore it, or start from fresh caches.
2. Prefill the rest of the prompt with the snapshot points; the drafter context is filled as above.
3. Take the end-of-prompt snapshot.
4. Decode:
   - **Greedy**, meaning `top_k == 1` or temperature 0 with no biases, penalties, min_new_tokens with stop ids,
     suppressed valid ids or reasoning budget. This runs the default spec loop through `specdec.spec_decode`: DFlash2
     with 4-bit linears and head copy, NAX verify, prompt lookup, `early@cv`.
   - **Everything else** decodes plainly. `session.decode_step` gives `Model.__call__`'s logits bit for bit, plus
     the drafter features. `sampler.py` is a port of upstream `crates/agens-cli/src/sampler.rs`, in its order: logit bias, suppressed ids,
     repetition / frequency / presence penalties, min_new_tokens, temperature, top-k, top-p, min-p, then a seeded
     xorshift64* draw. It also has the reasoning budget.
5. Take the end-of-reply snapshot.

The worker honours max_new_tokens, stop_token_ids and ignore_eos.

`spec_decode` is `_spec_loop_early` split off its prefill, with two optional hooks: `emit`, for streaming and
cancellation, and `finalize`. With the hooks off it runs the same ops. Against main's specdec.py (3 prompts, 449 /
3,421 / 9,306 tokens, 128 new, lookup on), tokens, per-step acceptance and draft sources were identical, at
98.3 / 98.6, 102.0 / 100.3 and 103.1 / 102.3 ms per step.

## Protocol (worker.py <-> MlxBackend)

**Framing.** A frame is a 4-byte big-endian length, then that many bytes of UTF-8 JSON (an object with a `"type"`).
There's one request per Unix-socket connection. The default socket is `$TMPDIR/volundr-mlx-worker.sock`, or set
`VOLUNDR_WORKER_SOCKET`.

**Client to worker:**
- `{"type": "generate", "prompt_ids": [...], "session_id": str | null, "sampling": {...}}`. The sampling object
  carries agens-server's SamplingParams fields: temperature, top_p, top_k (null = all), min_p, the three penalties,
  min_new_tokens, max_new_tokens, seed, `logit_bias: [[id, bias]]`, suppress_tokens, stop_token_ids, ignore_eos,
  reasoning_budget and reasoning_budget_soft_bias.
- `{"type": "info"}` returns protocol, context_length, vocab_valid, cache budget and the spec configuration.
- `{"type": "stats"}` returns the cache's snapshots and bytes, the queue length and the recent request records.
- `{"type": "clear"}` and `{"type": "budget", "gb": x}` run between requests.
- `{"type": "cancel"}` is optional on a generate connection; closing the connection does the same.

**Worker to client:**
- `{"type": "tokens", "ids": [...]}`: one frame per decode step, with several ids when speculative.
- `{"type": "done", "finish": "stop" | "length" | "cancelled", "stats": {...}}`. The stats are the request record:
  prompt, cached and prefilled tokens, snapshot used, TTFT, decode tok/s, tokens per step, peak GB and cache bytes.
- `{"type": "error", "message": ...}`.

**Ordering and cancellation.**
- Requests run one at a time in arrival order (bs = 1). Since branch batched-serve the worker batches concurrent
  requests instead, and prefill stops between layers rather than chunks ([NOTES_BATCHED_SERVE.md](NOTES_BATCHED_SERVE.md)).
- Closing the connection cancels a request: it's dropped if still queued, stopped between prefill chunks (at most
  ~5.5 s at 2048 tokens), or stopped after the current decode step. The interval snapshots taken so far, and the
  reply snapshot after a decode, are kept.

**MlxBackend** (server/volundr-server) implements `Backend` on top of this:
- `generate` opens a connection, forwards each `tokens` frame as one step, and drops the connection when the server
  drops the `TokenStream`.
- `main` mounts upstream's router unchanged, plus `GET /session_cache` (the worker's stats) and a middleware
  (`cancel_on_disconnect`) that fills a gap the e2e checks found:
  - agens-server notices a client that went away only when it writes to it. During a 20K-token prefill nothing is
    written, so in run 1 a client disconnect during the prefill, or while queued, wasn't seen until the first token.
  - The middleware gives each HTTP request a disconnect signal, which `generate` picks up from a task-local. It lives
    in the request future and in the response body.
  - It also sends an SSE comment (`: keep-alive`) on streams that have had no data for 2 s, so hyper finds the
    closed socket. Clients ignore SSE comments.

## Run it

```bash
./serve.sh                       # worker (model, cache 12 GB) + volundr-server on 127.0.0.1:30000
./serve.sh --port 30001 --cache-gb 8 -- --short-warmup     # options; worker.py flags after --
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Session-Id: s1' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
curl -s localhost:30000/session_cache
```

- `serve.sh` builds `server/` on first use (`cargo build --release`) and starts both processes. The server waits for
  the worker: 13.7 s to load, then a 13.5 s warm-up (Metal JIT, including the far-block kernels).
- The session comes from the `X-Session-Id` header or the `session_id` body field.
- The model files, including tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json,
  are `volundr-mlx/models/volundr-int4-mlx`; its chat template is byte-identical to the HF snapshot's.
- Opt-in modes pass through the environment: `VOLUNDR_LMHEAD8`, `VOLUNDR_W4A8`.

## Gates

| gate | result |
|---|---|
| `tests/test_session.py` (tiny model, GPU bf16): restore-then-continue for every cache type (prefill chunks, decode, splits inside the first window, at block boundaries, mid-block), fork of a fork, the aliasing and view-pinning cases, decode_step == Model.__call__ (logits and features), lookup / dedupe / LRU / points | 8/8 |
| `session_gate.py` exact (full model, real kernels): snapshot at 6,151 tokens (far blocks active) and at 1,501 (compacted rows); original vs two restores each run a 1,500-token prefill chunk, 20 decode steps with drafter-context updates, 6 verify + commit steps (random n) with drafter proposals | logits (69 rows), drafts and every final cache array **bit-identical**; snapshot unchanged |
| `session_gate.py` resume: fresh vs resumed prefill (below) | KL at the chunking floor; greedy diverges only at one-bf16-step margins; spec from the restored state == greedy from it |
| `session_gate.py` split: fresh prefill with a chunk boundary at the resume point | **bit-identical to the resumed prefill** (rows, model and drafter state) |
| `session_gate.py` engine (worker code path, two turns) | turn 2 resumed from the reply snapshot (2,032 of 2,060 tokens), TTFT 0.24 s vs 4.98 s; tokens identical to a cache-less engine |
| spec loop refactor vs main's specdec.py (3 prompts) | tokens, acceptance and sources identical |
| `tests/test_volundr.py`, `test_decode_fused.py`, `test_bcsa_verify.py` | 10/10, 8/8, ok |
| `spec/check_combined.py` (`--ref spec/runs/combined-test_tune_spec_baseline.jsonl`; its default ref file isn't in spec/runs) | ALL BIT-IDENTICAL; KL prefill vs 8q verify 3.6e-4 / 2.4e-4 / 7.2e-4 |
| `cargo test` (server) | 3/3: framing, the generate frame, a fake worker (steps streamed; dropping the stream or the HTTP request's disconnect signal closes the worker connection within 1 s) |

### Resumed vs fresh prefill (`session_cache/gate2.json`)

Each prompt (repo files plus a question) has a turn-1 prompt A and a longer B = A + more.
- **Fresh:** B prefilled from 0 in chunks of 2048.
- **Resumed:** A prefilled, snapshot at len(A) (not chunk-aligned), restore, then the rest of B.
- **Calibration:** B prefilled in chunks of 4096 instead of 2048, the PARITY.md calibration, on the same rows.
- **Rows:** the last prompt position, then 47 decode steps teacher-forced on fresh's greedy tokens.

| prompt | A / B tokens | prefilled on resume | KL resumed: mean / p99 / max | KL chunk 4096: mean / p99 / max | top-1 resumed |
|---|---|---|---|---|---|
| code, 10.6K | 8,483 / 10,604 | 2,121 (5.4 s vs 25.4 s fresh) | 4.2e-4 / 3.8e-3 / 4.9e-3 | 4.6e-4 / 3.6e-3 / 4.6e-3 | 47/48 |
| notes, 23.3K | 17,470 / 23,294 | 5,824 (14.5 s vs 56.7 s) | 5.8e-4 / 3.7e-3 / 3.7e-3 | 5.5e-4 / 5.0e-3 / 6.4e-3 | 47/48 |

Free-running, 128 greedy tokens:
- Resumed vs fresh first differ at tokens 40 and 21. Fresh's top-two margin there is 0.125, one bf16 step.
- Speculative decoding from the restored state is identical to plain greedy from it over all 128 tokens, on both
  prompts. From the fresh state they first differ at token 40 / 89, margin 0.125, as spec/NOTES.md describes.

### Where the resumed-prefill KL comes from

**The 2.5e-4 floor is an average over five prompts.** PARITY.md's 2.5e-4 is the mean over 240 rows of five prompts.
The three short ones fit in one chunk and are identical, so the two long prompts alone average ~6e-4. The resumed
means (4.2e-4, 5.8e-4) sit at that level and match the chunk-4096 calibration measured here on the same rows
(4.6e-4, 5.5e-4). The p99 and max values match too (3.7-3.8e-3 against PARITY's 3.3e-3 / 4.7e-3).

**The snapshot adds nothing** (gate split, `session_cache/gate3.log`). A fresh prefill of the code prompt with a
chunk boundary at the resume point (8,483) gives bit-identical rows and a bit-identical state (model and drafter) to
the resumed prefill. So the KL is entirely the change of chunk boundaries, and other boundaries give the same amount:
- a 32-aligned split at 8,480: KL 4.7e-4;
- a split at 16 mod 32 (8,496): KL 5.6e-4;
- chunks of 4096: KL 4.6e-4.

**Which component depends on the chunking** (`session_cache/chunk_invariance.py`, `gate5.log`). Each component of
the real model ran on the same random input in two chunkings of 6,000 positions:
[0, 2048, 4096, 6000) and [0, 2048, 4096, 4483, 6000). Outputs for positions 4483-6000 (far blocks active):

| component | rows differing (of 1,517) | max abs | mean abs (output mean abs) |
|---|---|---|---|
| 4-bit matmul (in_proj_qkv), MLP | 0 | 0 | 0 |
| dense attention (SDPA over the KV cache) | 0 | 0 | 0 |
| KDA conv + gates kernel, given the same projections | 0 | 0 | 0 |
| KDA recurrence `gated_delta_tg` (8-step staging), same inputs | 0 | 0 | 0 |
| KDA mixer as a whole | 18 | 0.0039 | 1.9e-8 (0.38) |
| **BCSA attention** (window + top-k far blocks) | **1,517** | 0.031 | 7.9e-6 (0.14) |

- **BCSA attention: the source.** Its prefill path builds the window keys from the cache offset
  (`keys = window(nprev) ++ chunk`, starting at `offset - nprev`). The fused NAX kernel tiles queries in 32-query
  blocks from the chunk start, gathers each block's union of far blocks, and runs them through one online softmax.
  When the chunk start moves, the key tiles, the unions and the accumulation order move, and the rounding changes on
  every row. The 32-aligned split still changes the window's key offset and the chunk lengths, which is why it's no
  closer to fresh.
- **The KDA recurrence isn't the source.** The staged recurrence and the conv/gates kernel are chunk-invariant, as
  the README says (gated_delta_tg = gated_delta_vec bit for bit, fp32 state across chunks). The KDA mixer's 18 rows,
  at most one bf16 step each, come from another part of it: the gate projections or the gated norm. That wasn't
  isolated further; it's 1e-3 of the BCSA difference.
- **What makes it noise.** These are the same rounding differences any change of prefill chunk size produces. Each
  layer passes them on, and greedy output flips only where the top two logits tie.
- **What bit-exact resume would need.** Chunk-invariant BCSA prefill attention: absolute-position tiling, and a
  reduction order independent of the chunk.

## End-to-end (`session_cache/e2e/`, run 2, 04:53-05:02)

**Setup.**
- Through `./serve.sh` on the OpenAI API, one session (`X-Session-Id`), greedy (`temperature 0`).
- `reasoning_effort: low`, `max_tokens 384`, streaming.
- System prompt plus three tool definitions (read_file, grep, list_dir). Each turn appends the assistant message as
  returned (content, reasoning_content, tool_calls), then the next repo file as the tool result (up to 13K
  characters; a file the model asked for when there was one).
- No cache: the same requests replayed with the cache budget at 0, in the same worker process.

TTFT is measured client-side (request sent to first streamed token, through the server). Decode tok/s is the
worker's (generated - 1) / decode time.

| turn | prompt tokens | cached / prefilled (snapshot) | TTFT cached | TTFT no cache | speed-up | decode tok/s cached / no cache | tokens/step cached | cache GB (snapshots) | peak GB | same reply |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 664 | 0 / 664 | 1.67 s | 1.66 s | 1.0x | 37.3 / 37.2 | 3.67 | 0.51 (2) | 27.5 | yes |
| 1 | 4,493 | 773 / 3,720 (reply) | 8.97 s | 10.65 s | 1.2x | 18.8 / 18.8 | 1.94 | 2.68 (5) | 29.3 | tie at token 104 |
| 2 | 8,894 | 4,698 / 4,196 (reply) | 10.82 s | 21.14 s | 2.0x | 34.1 / 35.0 | 3.62 | 5.13 (8) | 31.8 | yes |
| 3 | 12,931 | 9,277 / 3,654 (reply) | 10.01 s | 31.09 s | 3.1x | 21.3 / 29.9 | 2.25 | 7.86 (11) | 34.5 | tie at 47 |
| 4 | 16,748 | 13,180 / 3,568 (reply) | 9.26 s | 40.47 s | 4.4x | 18.2 / 19.5 | 2.00 | 10.86 (14) | 37.5 | tie at 0 |
| 5 | 18,040 | 16,844 / 1,196 (reply) | 3.13 s | 43.60 s | **13.9x** | 18.7 / 20.0 | 2.03 | 11.74 (13) | 39.6 | tie at 94 |
| 6 | 21,717 | 18,234 / 3,483 (reply) | 9.09 s | 52.74 s | **5.8x** | 25.1 / 25.6 | 2.70 | 11.93 (12) | 40.6 | tie at 60 |

**TTFT.**
- With the cache, TTFT is the new tokens' prefill: 366-415 tok/s.
- Without it, the whole prompt each turn (412 tok/s at 21.7K; the README's 374 is a 19K prefill measured
  separately).
- These tool results are large (3.5-4.2K tokens); a turn with 1.2K new tokens answers in 3.1 s at 18K context.

**Decode speed** is unchanged where the reply is the same (turns 0, 2): 37.3 / 37.2 and 34.1 / 35.0. Elsewhere the
two replies are different texts, so they accept differently.

**Snapshot cost.**
- Each turn left 2-3 snapshots (prompt, reply, and an interval point when the new tokens crossed a multiple of 4096).
- From turn 5 the 12 GB budget evicted the oldest.
- Captures took 3-6 ms and restores 0.1-1.2 ms.

**Memory.**
- MLX peak 40.6 GB with 11.9 GB of snapshots, against 27.3-29.1 GB without the cache.
- Swap held 11-14 GB throughout, as other apps were loading the machine.

**Determinism.** Run 1 (`e2e_run1/`, 04:35-04:47) gave the same requests, replies and cache numbers for turns 0-5,
with timings within a few percent. Its turn 6 differs: that turn's tool result is the first 13K characters of
worker.py, and the token log was added inside them between the runs.

### Correctness spot check (`session_cache/tie_check.py`, `tie_check.log`)

Same conversation, same requests, with and without the cache. Five of seven replies differ (turns 1 and 3-6).

The tie check replays the worker's logged prompt ids in-process (session.Engine, the worker's code path). It
reproduces both served runs token for token, and the same cached / prefilled split for every turn. At each first
differing token it reads the target's logits for the two candidates, teacher-forced along the common prefix from
both states:

| turn | first diff (cached / no cache) | resumed state: logits, gap | fresh state: logits, gap |
|---|---|---|---|
| 1 | token 104: " Need" / " We" | 22.125 / 22.0, 0.125 | 22.125 / 22.0, 0.125 |
| 3 | token 47: "," / " step" | 24.375 / 24.375, 0 | 24.25 / 24.5, 0.25 |
| 4 | token 0: "Interesting" / "The" | 21.0 / 20.875, 0.125 | 20.875 / 20.875, 0 |
| 5 | token 94: " Could" / " We" | 22.5 / 22.5, 0 | 22.5 / 22.5, 0 |
| 6 | token 60: " Maybe" / " We" | 23.375 / 23.25, 0.125 | 23.375 / 23.25, 0.125 |

- Every divergence is between two candidates whose logits are equal or one to two bf16 steps apart (0.125 at
  |logit| 16-32), i.e. a tie.
- Some flips (turns 1 and 6) don't come from the cache at all. Both states rank the cached run's token first in a
  plain decode step, so the no-cache run's speculative verify, with its 8-row kernels, broke the tie the other way.
- After a flip, the rest of the reply is a different text.

### Concurrency (`session_cache/e2e/checks.json`, all pass)

- **Two sessions interleaved** (A1, B1, A2, B2): each second turn resumed from its own session's reply snapshot
  (A 1,967 of 1,988 tokens, B 770 of 792), with TTFT 0.24 / 0.23 s against 4.6 / 1.8 s for the first turns. Then A3
  and B3 sent together: B3 waited 0.68 s in the queue, and both resumed from their own snapshots.
- **Eviction** under a 2 GB budget, with three ~4K-token sessions C, D, E:
  - The cache stayed at or under 1.57 GB; each new session's prompt evicted the older sessions' snapshots.
  - E's second turn still resumed from its reply snapshot (4,494 of 4,514 tokens).
  - C's snapshots, the least recently used, were gone, so C's second turn prefilled from 0 and still answered.
- **Cancellation:**
  - Mid-stream: the client closed after 30 deltas of a 1,500-token reply. The worker stopped at 59 tokens and kept
    the reply snapshot; the next request waited 0.14 s.
  - Mid-prefill: a fresh 17.2K-token prompt closed at 6 s. The worker cancelled at position 4,096, kept that interval
    snapshot, and was free after 9.6 s; the next request waited 3.6 s.
  - While queued: a request closed while an 11.5K-token prefill ran. The worker dropped it at dequeue without running
    it ("cancelled: queued").
  - Run 1, before the server's disconnect handling, failed the last two: the prefill and the queued request both ran
    to their first token.

## Known limits

- **Resume points.** A request resumes only at a snapshot point; a KDA state can't be trimmed. A prompt that diverges
  between two points prefills from the earlier one, at most 4096 tokens back, or from the previous request's prompt
  or reply end.
- **Not bit-exact against an uncached run.** A resumed prefill is a prefill with a chunk boundary at the resume
  point, so greedy output equals the uncached output up to ties (above). Bit-exactness would need chunk-invariant
  BCSA prefill attention.
- **Memory.**
  - Snapshots live in GPU memory, with no SSD spill.
  - In a linear session the older prompt and interval snapshots are superseded but leave only by LRU. Dropping a
    session's superseded snapshots would hold more sessions in the same budget.
  - A short snapshot still costs 0.18 GB, which is the KDA state.
- **Scheduling.** bs = 1, no continuous batching; requests queue.
  - Sampled requests decode plainly, not speculatively, at plain-decode speed (12-14 tok/s in README), with the
    sampler on the host. Host sampling over the whole vocabulary (top_k unset) costs more.
- **Cancellation.**
  - Granularity is a prefill chunk (up to ~5.5 s) or a decode step.
  - A non-streaming request is cancelled only when hyper drops its request future; that path wasn't tested.
  - Streams that wait more than 2 s carry `: keep-alive` SSE comments, which upstream doesn't send.
- **Lookup.** Prompt lookup starts per request and indexes the whole prompt on its first step, in Python.
- **Context length.** The worker reports the model's context length of 262,144. Memory at that length wasn't
  measured; README projects ~35 GB peak, before any snapshots.
- **Ports.** The sampler follows upstream sampler.rs step by step. Whether its draws equal upstream's for the same
  seed wasn't checked.

## Files

**New**
- `session.py`: snapshot / restore, `SessionCache`, the resumable prefill, `decode_step` and `Engine`.
- `sampler.py`: port of sampler.rs and the reasoning budget.
- `worker.py`: socket protocol, queue and cancellation.
- `session_gate.py`, `session_e2e.py`, `tests/test_session.py`.
- `session_cache/`: step.sh, regress.sh, e2e.sh, chain2/3.sh, chunk_invariance.py, tie_check.py, and the logs and
  JSON of every run.
- Repo root: `serve.sh`, `server/` (Cargo workspace: `volundr-server` with `MlxBackend`, `cancel_on_disconnect` and
  `main`; Cargo.lock resolved for Rust 1.87).

**Changed**
- `spec/specdec.py`: `spec_decode` (the early loop from a given state, with `emit` and `finalize`);
  `_spec_loop_early` calls it.
- `.gitignore`: `server/`, `serve.sh`, `server/target`.
- `NOTICE`, both READMEs, and one sentence in `PARITY.md` (the chunking calibration moves BCSA attention rounding;
  the KDA recurrence is chunk-invariant).

volundr.py, kernels.py, verify.py, dflash.py and lookup.py are unchanged.
