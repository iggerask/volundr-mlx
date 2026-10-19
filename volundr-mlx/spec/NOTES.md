# DFlash2 speculative decoding for Volundr MLX (2026-10-18/19)

Drafter: Blockway/Agens-Volundr-32B-v1-DFlash2 (HF cache snapshot 0e66f1fe…, SHA256SUMS verified). Target: the INT4
MLX port (../models/volundr-int4-mlx). M5 Pro 64 GB, MLX 0.32.4. Greedy only. Nothing committed.

## Result (held-out test split, 84 prompts, frozen config)

**Speculative decoding: 1.77× pooled decode tok/s over plain greedy (24.0 against 13.5 tok/s, the means of two runs
each; the four run pairings give 1.71–1.82×), 1.85–1.98× as the geometric mean of per-prompt speed-ups; 2.63 tokens
committed per verify step.** Tune split, same config: 1.80× / 1.94×. The table below is the first held-out run
(`test_frozen.jsonl` against `test_base.jsonl`). In that run 83 of 84 test prompts are faster than greedy (the
slowest: a multilingual prompt at 0.91×, 1.36 tokens/step), 6 are below 1.2×, and the best are JSON extraction at
3.8–4.2×. Thinking on 1.79×, off 1.63×.

Frozen config: DFlash2 drafter with 4-bit (g128) linears and a 4-bit copy of the target lm_head for drafting, an
8-token verify on the NAX small-M kernel, the engine's prompt lookup on, `MLX_MAX_OPS_PER_BUFFER=100` for the
speculative process.

| category | greedy tok/s | spec tok/s | × | tokens/step |
|---|---|---|---|---|
| math | 13.8 | 38.7 | 2.80 | 4.14 |
| json | 13.8 | 37.1 | 2.69 | 4.00 |
| agent (2–8K, tool calls) | 13.2 | 30.8 | 2.34 | 4.18 |
| coding | 13.7 | 30.2 | 2.21 | 3.23 |
| rag | 13.7 | 26.5 | 1.94 | 2.88 |
| summarization | 13.7 | 25.1 | 1.83 | 2.76 |
| multilingual | 13.9 | 23.2 | 1.68 | 2.53 |
| stem | 13.9 | 22.5 | 1.62 | 2.44 |
| reasoning | 13.8 | 22.1 | 1.60 | 2.38 |
| roleplay | 13.8 | 21.1 | 1.53 | 2.27 |
| humanities | 13.9 | 21.0 | 1.51 | 2.20 |
| qa | 13.8 | 20.9 | 1.51 | 2.22 |
| writing | 13.8 | 19.1 | 1.38 | 2.17 |
| long (8–21K) | 12.7 | 16.3 | 1.29 | 2.36 |
| **all** | **13.7** | **23.4** | **1.71** | **2.63** |

**Repeat.** In the first held-out spec run every step took ~107 ms against ~97 ms for the same config on
the tune split. The slowdown held from the first prompt to the last, and the canary dropped to 22.4 TFLOP/s;
the greedy run before it had normal step times. So the held-out measurement was repeated once with nothing
changed (spec, then greedy, back to back): spec 24.48 tok/s (median step 98.8 ms), greedy 13.42 tok/s. Tokens,
acceptance and divergences are identical between the runs; only time moves.

| spec run / greedy run | pooled | geo-mean |
|---|---|---|
| 1 / 1 | 1.71× | 1.85× |
| 1 / 2 | 1.75× | 1.89× |
| 2 / 1 | 1.79× | 1.94× |
| 2 / 2 | 1.82× | 1.98× |

Per category the splits differ by up to ±25% (reasoning 2.11× tune / 1.60× test, rag 1.68× / 1.94×) on 4–6 prompts
each; the pooled number is the one to quote.

## Method (how the suite keeps the tuning honest)

- **Suite** (`make_suite.py` → `suite.json`, seed 20261018): 140 prompts, 14 categories × (4 tune + 6 test), never
  overlapping. SPEED-Bench "qualitative" (coding, humanities, math, qa, rag, reasoning, stem, writing, multilingual,
  summarization, roleplay); its "fetch from source" stub rows are dropped and short categories topped up from local
  MMLU / ARC-Challenge (humanities, stem, reasoning) and BoolQ Wikipedia passages (summarization, json extraction).
  Plus agent (SWE-smith trajectories cut after a tool result, 2–8K tokens, rendered with tool definitions) and long
  (8–21K-token documents / engine source with questions). Thinking alternates on/off within each category.
- **Splits:** every decision (fixes, drafter precision, lookup) was made on the 56 tune prompts. The 84 test prompts
  ran once, with the frozen config, after the decisions.
- **Metric:** decode tok/s = (generated − 1) / decode wall time, per category summed over its prompts, 256 new tokens
  (or to EOS). Baseline: plain greedy decoding (generate.py's pipelined loop) on the same prompts. Pooled numbers are
  token-weighted; the geometric mean of per-prompt speed-ups weighs every prompt equally.
- **GPU lock** for every run, canary before/after. The machine had other apps holding CPU cores at 100%
  (fseventsd, a WebKit content process, BTLEServer, bun); they move per-step time by up to ~25% for minutes at a
  time, so config differences under ~5% in tok/s are within noise. Acceptance (tokens per step) does not depend on
  timing and was used to judge drafter-quality changes.

## Implementation

| file | what |
|---|---|
| `dflash.py` | Drafter port, engine semantics (`m1_draft.cu`): context = RMSNorm(fc(target stream 0 at entry of layers 9/22/36/54/68)) → per-layer K/V ring; one non-causal pass over [anchor, mask×7]; grouped 2-tap convs with projected taps; top-16 of the target head per slot + greedy low-rank bigram walk. Optional 4/8-bit linears and a quantized head copy (drafting only). |
| `verify.py` | T-token target forward with feature capture, and `commit(n)`: KDA replays the recurrence on the first n rows' saved q/k/v/g/β from the pre-call state (no projections re-run; n = T keeps the result); BCSA attends like a prefill chunk without touching the cache and appends n rows on commit; dense KV trims; Engram ids reset. |
| `specdec.py` | Loop, greedy baseline, CLI (`check`, `bench`, `base`). One host sync per step. |
| `nax_hook.py` | 2–16-row 4-bit matmuls on the NAX kernel (`agens-mlx-bench/smallm/qmm_nax.py`). |
| `lookup.py` | Port of the engine's prompt lookup (`lookup.rs`): copy proposals chosen per step by running means; a full 7-token copy skips the drafter pass. |
| `check_verify.py`, `time_parts.py`, `prof_build.py`, `probe_async.py`, `analyze.py` | checks, timing, analysis. |

## Correctness

- **Teacher-forced** (`check_verify.py`, `runs/check_verify1.log`): verify forwards over greedy tokens with a random
  commit length each step (exercises every rollback kind), one prompt per category incl. 7K agent and 8K long
  (far blocks active): 1520 rows, KL against the plain decode step's logits mean 3.3e-4, p99 4.3e-3, max 7.2e-3,
  argmax agreement 98.6%. For scale: the exact prefill kernels drift 8e-4 against phase 1. A broken rollback shows
  orders of magnitude more from the next step on.
- **Free-running** (`specdec.py check`, frozen config, one tune prompt per category, 128 tokens,
  `runs/check_frozen.jsonl`): 6/14 identical to plain greedy. The other 8 first differ where greedy's own top two
  logits are an exact bf16 tie (margin 0.0, 7 cases) or one bf16 step apart (0.125, 1 case). Greedy output is the
  target's, up to tie-breaks between equal logits: the verify kernels (8-row NAX matmuls, prefill-style BCSA
  attention) round differently from the 1-row decode kernels, and bf16 logits tie often. The drafter README states
  the same property for the engine. Bit-exact equality would need batch-invariant kernels (the engine's approach).

## Tuning log (tune split, 56 prompts; decisions only from here)

| round | config | tok/s | × greedy | tokens/step | decision |
|---|---|---|---|---|---|
| 1 | bf16 drafter, NAX verify | 17.6 | 1.32 | 2.52 | found two bugs (below); superseded |
| 2 | bf16 drafter | 21.7 | 1.57 | 2.51 | |
| 2 | 4-bit drafter + 4-bit head copy | 24.9 | 1.80 | 2.50 | same acceptance, 12 ms less per step: keep |
| 2 | … + prompt lookup | 24.8 | 1.80 | 2.56 | +31% acceptance multilingual, +17% agent, ±4% elsewhere: keep (engine default) |
| 2 | 8-bit drafter + head | — | — | — | stopped: dominated (bf16 and 4-bit accept the same) |

Round-2 speed-ups are against the fresh round-2 greedy run (13.81 tok/s; round 1 had 13.28, identical tokens on
all 56 prompts: background-load drift). Geometric mean of per-prompt speed-ups for the frozen config: 1.94×.

Found on the tune split:
1. **EOS inside an accepted run** did not end the loop (only the last emitted token was checked), so prompts that
   finish early (agent, qa) spent extra steps past EOS. Agent read 0.97× before the fix, 2.4× after.
2. **`mx.async_eval` of the draft blocked for the whole draft** (26 ms of 26). MLX lets the CPU run at most ~10
   command buffers ahead of the GPU and the drafter's ~400 kernels span more at the default buffer size, so the
   verify graph was built (6–7 ms) with the GPU idle and the planned overlap never happened.
   `MLX_MAX_OPS_PER_BUFFER=100` (+ `MLX_MAX_MB_PER_BUFFER` unbounded) makes the call return in ~1 ms; specdec.py
   sets it for `bench` / `check`. Plain greedy decoding is ~4% slower with it (70.8 → 73.6 ms/token on 3 prompts),
   so the baseline keeps MLX's default: each mode runs its own best setting.

Not pursued, with the reason:
- **Verify width T**: with NAX, verify costs the same from T = 2 to 16 (91–105 ms), so T = 8 (the drafter's whole
  block) is best; smaller T only helps the stock path.
- **Falling back to plain decode on low acceptance**: break-even is ~1.33 tokens/step (97 ms step against ~73 ms
  decode); the lowest tune prompt ran at 1.40 (1.08×), so a fallback gains almost nothing here and adds a threshold.
- **8-bit target lm_head** (≈ −5 ms per verify and per decode step): changes the target's numerics, i.e. the model,
  for spec and baseline alike; a separate decision.

## Where a step's time goes (short context, frozen config)

| part | ms |
|---|---|
| draft (5 layers 4-bit on NAX, 4-bit head copy for 7 rows, top-16 + walk) | ~13 |
| verify T = 8 (NAX matmuls, bf16 lm_head over 8 rows, capture, commit replay) | ~84 |
| CPU between GPU work (host bookkeeping, commit + draft graph build; the verify build overlaps the draft) | ~1.5 |
| **step** | **~97** (decode step 71–74) |

At 8–21K context the step is ~116 ms (decode ~80): BCSA far-block selection for 8 queries costs more than for 1.

## Next levers (not done)

1. **Verify cost** (~84 of ~97 ms): the NAX matmuls stream at ~247 GB/s against decode's ~293
   (`agens-mlx-bench/smallm/VERIFY_STEP.md`); closing that is worth ~10 ms per step (+10%).
2. **8-bit lm_head for the target** (a model decision, applies to greedy too): ~5 ms per verify and per decode step.
3. **Long context** (1.29×): the step is ~116–140 ms at 8–21K against ~80 for decode. The BCSA verify uses the
   prefill kernels for 8 queries (32-query tiles, per-query top-k over all far blocks); an 8-query variant is the
   lever. Acceptance there is 1.9–2.4 tokens/step (the drafter's own window is 2048).
4. **Bit-exact greedy**: batch-invariant verify kernels (each row rounded as its 1-row decode step), as the engine
   does. Today greedy output differs from plain decoding only at bf16 logit ties.
5. **Drafter pass** (~13 ms): ~300 small kernels besides ~1.6 GB of 4-bit weights; more fusion is worth a few ms.
6. Sampling (temperature > 0) is not implemented; the engine's rule (accept draft x with probability q(x), resample
   from q without x) ports directly onto `spec_generate`.

## Runs

`runs/`: `check_verify1.log` (teacher-forced), `time_parts1.log` (part timings), `tune_*.jsonl` (round 1, superseded),
`tune2_*.jsonl` (round 2), `check_frozen.jsonl` (free-running vs greedy), `test_base.jsonl` / `test_frozen.jsonl`
(held-out), `test_*_r2.jsonl` (held-out repeat). `python analyze.py <base.jsonl> <run.jsonl> …` prints the tables.

Reproduce (from spec/, under ../with_gpu_lock.sh):
```
python specdec.py base  --split test --new 256 --out runs/test_base.jsonl
python specdec.py bench --split test --new 256 --nax --drafter-bits 4 --head-bits 4 --lookup --out runs/test_frozen.jsonl
python analyze.py runs/test_base.jsonl runs/test_frozen.jsonl
```
