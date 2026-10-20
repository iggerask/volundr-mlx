# Speculative tok/s: prefill memory fix, step profile, host time and acceptance (branch spec-accept, 2026-10-20)

Status: done (2026-10-20 02:50).

## Resume

Nothing is pending. Branch `spec-accept` from `combined-test` (c5fe389), worktree
`.claude/worktrees/agent-af7e4848a9651c921`. Every GPU step ran as `spec_accept/step.sh <log> <cmd>` (canary, command,
canary, load, swap) inside a chain under `./with_gpu_lock.sh`, from `volundr-mlx`, with
`PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python`:
- `spec_accept/step0.sh`, `step0b.sh`: memory-fix gates (`step0.log`).
- `spec_accept/prof.sh`: step profile and verify widths (`prof.log`); `acclog.sh`: acceptance log (`acclog.log`).
- `spec_accept/next1.sh`, `next2.sh`: quick schedule A/Bs, batched-verify check and cost, compiled-verify bit-identity,
  verify-build profile, long-copy checks (`sched_quick.log`, `b2.log`, `compile.log`, `long16.log`).
- `spec_accept/tune_ab.sh`: the tune A/B (`tune_chain.log`); `final.sh early@cv@long16`: held-out sample, the
  after-profiles and the check_verify gate (`heldout.log`, `prof.log`, `gates.log`).

Summaries: `spec_accept/tune_summary.txt`, `heldout_summary.txt`, `acc_analyze_tune.txt`; re-derive with
`combined_test/analyze.py`, `spec_accept/ab_report.py`, `acc_analyze.py`, `long_buckets.py`, `canaries.py`. Run files:
`spec/runs/spec-accept_*` (shared, gitignored).

How to use the result: since the merge into main, the default spec loop is `early@cv`, which gives the same tokens
as the previous loop. `--loop base`, an ab_suite config `...@base`, or `VOLUNDR_SPEC_SCHED=base VOLUNDR_SPEC_COMPILED=0`
give the previous loop. `@long16` stays opt-in (`--loop long16`, ab_suite `exact+q8g64@long16`), because it changes
outputs at ties. The A/B tables below were measured before the default changed: their "baseline" is the previous loop.

## Summary

| | baseline: exact + 8-bit head (g64) | + lever 3 (`early@cv`) | + lever 3 + lever 2 (`early@cv@long16`) |
|---|---|---|---|
| tune, all 56, tok/s | 26.96 | 27.62 (×1.025) | 27.47 (×1.019) |
| tune, short (<4K, 48) | 27.24 | 27.92 (×1.025) | 27.71 (×1.017) |
| tune, long+agent (8) | 24.99 | 25.55 (×1.023) | 25.78 (×1.032) |
| tune, median ms/step | 95.6 | 93.5 | 93.9 |
| tune, tokens/step | 2.60 | 2.60 | 2.61 |
| tune, tokens identical to baseline | - | 56/56 (acceptance 56/56) | 36/56 (see Correctness) |
| held-out coding+long+agent (18), tok/s | 27.98 | not run | 28.88 (×1.032; agent ×1.111) |

- **Step 0, memory.** The spec prefill now evaluates layer by layer. Peak on long/03 (12.5K tokens): 56.0 → 27.7 GB,
  same session; prefill 2-5% faster; outputs, cache state and spec step time unchanged. The whole tune A/B peaked at
  30.2 GB per process (4 configs in one process), against 57-58.5 GB in combined-test's.
- **Step 1, profile.** At short context a 94 ms step is ~84 ms verify forward, ~9 ms drafter (on drafter steps), ~3 ms
  commit + drafter-context update, and ~1.5-2.5 ms GPU idle at the start of the step. The verify graph build (6.7-8.6 ms
  of CPU) runs while the GPU drafts and was on the critical path: the CPU work between two verify submissions (~10 ms)
  was as long as the GPU work it overlaps. The "15-20 ms outside the verify" estimate is ~12 ms measured.
- **Lever 3 (`early@cv`, adopted): ×1.025, bit-identical.** mx.compile of the 54 KDA verify layers cuts the verify
  build from 7.7 to 4.4 ms per step; the reordered loop starts the GPU earlier. Every category gains 1.6-3.4%.
  Scheduling alone (no compile) and a second GPU stream gave nothing.
- **Lever 2: small.** The measured per-step upside is an upper bound, and most candidates do not pay for their cost.
  - Batched verify of two chains (B = 2) costs +9 ms (short) / +13 ms (9.5K) per step, +10-14%. The best second chain
    (DFlash top-2 at slot 0) would add at most +6.7% tokens per step, the lookup copy +4.9%: not built into the loop.
  - Long lookup copies (verify up to 16 rows when the copy fills the draft): on the same tokens, −2.1% steps for
    +0.5 ms per step (+1.6% tok/s on the 36 prompts whose output did not change); agent ×1.091 over lever 3 alone;
    pooled over all 56 prompts ×0.995 against lever 3, which is inside the trajectory noise of the 20 prompts whose
    text diverged at a tie (±10-20% per prompt either way). Kept as the final config's lever 2 and as an option.
- **Correctness.** Lever 3: tokens and per-step acceptance identical to the baseline on 56/56 tune prompts; the
  compiled layers bit-identical to the eager ones (logits, features, every cache array, T = 8 and 16). Long copies:
  teacher-forced 16-row verify with random rollback at the T = 8 noise floor; free-running against greedy, every
  divergence at an exact bf16 tie (6) or one bf16 step (1). The B = 1 verify path is unchanged: check_verify
  reproduces combined-test's log line for line.

## Step 0: the prefill memory fix

**Problem.** The spec processes set `MLX_MAX_OPS_PER_BUFFER=100` and no MB cap (`specdec.py`'s top comment: the
drafter's async overlap needs it). MLX reads both once at device init, process-wide, so they cannot be switched for
the prefill. Under them, a 2048-token prefill chunk keeps ~10 command buffers of large activations alive.

**Fix.** `spec_prefill` (specdec.py) passes `layer_sync=True` to `verify.forward`: each layer's outputs (mHC streams,
captured features, that layer's cache arrays) are submitted with `mx.async_eval`, then the previous layer is waited
for, so at most two layers are in flight. The kernels and values are unchanged; only evaluation boundaries move.
`VOLUNDR_PREFILL_SYNC=0` restores the old behaviour. The decode loop keeps the spec buffers.

**Gates** (`spec_accept/step0.log`; `runs/spec-accept_step0_check.jsonl`; 8-bit head, spec buffers):

| check | layer-synced (new) | unsynchronised (old) |
|---|---|---|
| `combined_test/peak_probe.py`, long/03, fresh process: prefill peak / generate peak | **27.67 / 27.67 GB** | 56.0 / 56.0 GB |
| same: prefill time (12,547 tokens) | 30.56 s | 31.23 s |
| interleaved in one process, long/03: prefill time, peak | 30.66 s, 27.67 GB | 32.32 s, 50.12 GB (generate 56.93) |
| agent/01 (5.4K), 2 reps | 12.90-12.93 s, 27.4-28.1 GB | 13.11-13.17 s, 56.6-56.8 GB |
| rag/00, json/00, coding/00 (0.2-0.7K), 2 reps each | 0.63-1.82 s, 25.7-26.8 GB | 0.65-1.85 s, 28.5-36.0 GB |
| cache + drafter-cache arrays and last logits after prefill | bit-identical to the first run (9/9) | bit-identical |
| spec tokens and per-step acceptance, new vs old, per rep | identical 9/9 | |
| spec ms/step, pooled over those 9 runs | 98.37 | 98.18 |

Prefill is within ±2% or faster (5% faster at 12.5K); the step time difference is +0.2%, inside run-to-run noise.
The tune A/B below (one process per group, four configs, the bf16 and the 8-bit head both loaded) peaked at 30.21 GB
per process; the baseline config
reproduces combined-test's exact+q8g64 tune run token for token and step for step (56/56, 56/56).

## Step 1: where a step goes

`spec/prof_step.py` runs spec_generate's loop twice from the same start: pipelined (production, host timestamps around
every CPU part) and serialized (each GPU part evaluated alone from an idle GPU). Both runs are asserted to emit
spec_generate's tokens. Baseline config (exact + 8-bit head), 160 new tokens, `runs/spec-accept_prof_base.jsonl`.
Times are ms per step.

| part | short (coding/00, 197 tok) | agent/01 (5.4K) | long/01 (9.5K) |
|---|---|---|---|
| **pipelined: host CPU** | | | |
| bookkeeping (`.item()`, emitted, lookup.record) | 0.25 | 0.30 | 0.28 |
| commit graph build | 0.67 | 0.76 | 0.84 |
| drafter-context (materialize) build | 0.09 | 0.15 | 0.11 |
| lookup.propose | 0.01 | 0.07 | 0.02 |
| drafter graph build | 0.39 (0.48 on drafter steps) | 0.59 | 0.63 |
| `mx.async_eval` of the draft (encode + commit) | 1.14 | 2.18 | 1.82 |
| verify graph build | 6.67 | 8.50 | 8.59 |
| wait (encode verify, then blocked on the GPU) | 84.98 | 88.34 | 89.31 |
| **wall** | **94.21** (drafter steps 94.72, lookup-copy steps 91.97) | **100.89** | **101.60** |
| **serialized: GPU (+ submit latency)** | | | |
| commit (KDA replay of n rows, BCSA append, KV trim) | 2.37 | 2.11 | 2.36 |
| materialize (drafter context K/V of n rows) | 0.76 | 0.81 | 0.74 |
| drafter, on drafter steps | 9.03 | 10.76 | 10.42 |
| verify forward + argmax + accept count | 84.22 | 87.65 | 87.19 |
| verify graph build with the GPU idle | 5.85 | 6.33 | 6.03 |
| **drafter parts** (GPU, from idle): 5-layer block / + 4-bit head (7 rows) / + top-16 / + walk | 5.27 / 7.94 / 8.99 / 9.12 | 6.69 / 9.56 / 10.24 / 10.35 | 6.74 / 9.37 / 10.24 / 10.42 |

Reading:
- **Verify is ~89% of the step.** In the loop it is ~82-84 ms at short context; the synthetic in-chain verify's 87 ms
  is in line.
- **Outside the verify forward: ~12 ms, not 15-20.** Drafter 9-10.8 ms (on the ~90% of steps that run it),
  materialize 0.8, commit 2.1-2.4 (arguably part of the verify), and 1.5-2.5 ms with the GPU idle at the start of the
  step (bookkeeping + commit / materialize / draft builds before anything is submitted).
- **The CPU was on the critical path.** Between two verify submissions the host does ~10.5 ms of work (1.5 before the
  first submit, 1.1-2.2 submitting, 6.7-8.6 building the verify), and the GPU work it overlaps (materialize + draft)
  is ~10 ms. The verify build is 15-40% slower while the GPU runs (6.0 idle against 6.7-8.6 pipelined;
  `spec/prof_vbuild.py`: 558 NAX linear calls per build, ~3.3 ms of it under cProfile). On
  lookup-copy steps there is no draft to hide it: those steps saved only 2.7 of the drafter's 9 ms.
- Verify width (`time_parts.py`, from idle, verify + commit + materialize of T rows): 90.1 ms at T = 8, 97.1 at 16
  (short); 94.6 / 103.0 at 8K. Plain decode in the spec process: 71.9 / 77.3 ms.

**After (adopted loop).** The tune A/B's own host segments, all 56 prompts, ms per step (`spec_accept/tune_summary.txt`;
the early loop's first two segments hold different work, see lever 3):

| config | CPU before the first submit | submit (early: + commit build) | verify build | wait | step |
|---|---|---|---|---|---|
| baseline | 1.70 | 1.41 | 7.68 | 85.22 | 96.0 |
| `early@cv` | 1.31 | 2.59 | 4.44 | 85.35 | 93.7 |
| `early@cv@long16` | 1.43 | 2.67 | 4.47 | 85.74 | 94.3 |

The profiler again with the compiled verify (`--compiled`, and `--compiled --long 16`; 02:40-02:47, a later session
than the "before" profile at 00:29, so up to ~1 ms of drift; `runs/spec-accept_prof_cv*.jsonl`):

| | coding/00 before → cv | agent/01 before → cv | long/01 before → cv |
|---|---|---|---|
| verify graph build, pipelined | 6.67 → 3.67 | 8.50 → 4.31 | 8.59 → 4.28 |
| step, base order (instrumented) | 94.21 → 92.45 | 100.89 → 98.00 | 101.60 → 99.15 |
| step, early loop (spec_generate) | - → 92.41 | - → 96.95 | - → 98.56 |
| lookup-copy steps (8 rows), base order | 91.97 → 88.57 | | |
| GPU: commit / materialize / draft / verify (serialized) | 2.36 / 0.74 / 9.0 / 83.1 | 2.02 / 0.76 / 10.7 / 86.2 | 2.33 / 0.74 / 10.4 / 86.6 |

With long copies on coding/00: a 16-row copy step takes 93.95 ms against 88.57 for an 8-row one (+5.4 ms; GPU verify
87.25 against 83.0), and the prompt's tokens per step go 2.925 → 2.981.

## Lever 3: hiding host time (adopted: `early@cv`, ×1.025, bit-identical)

**What it is** (spec/verify.py, spec/specdec.py):
- **`cv`: compiled KDA verify layers.** The 54 KDA layers' verify graph (mHC mix, norms, NAX projections, conv,
  recurrence, gated norm, MLP, Engram, mHC combine) has no Python-side state, so each layer goes through `mx.compile`
  once per shape; later calls replay the trace instead of ~30 Python op calls. The cache key includes the switches
  ab_modes flips, so a trace never outlives its config. BCSA and dense layers stay eager (their graphs depend on the
  cache offset).
  - `spec_accept/check_compile.py`: bit-identical to the eager verify (logits, captured features and every cache array
    after a random-n commit, 24 steps each, T = 8 and 16, at 197 / 7,260 / 12,547 tokens).
  - Build time with the GPU idle: 5.8-6.2 → 3.2-3.7 ms; in the loop 7.68 → 4.44 ms.
- **`early`: the loop reordered** (`_spec_loop_early`). Right after the sync the drafter-context update is built and
  submitted (the GPU starts on it), then the bookkeeping, lookup and draft build, the draft submit, and only then the
  commit build (the verify needs it, the drafter does not). On lookup-copy steps the commit is submitted on its own so
  the GPU runs it during the verify build. GPU order and values unchanged.

**Results** (interleaved, same process; tokens and acceptance identical to the baseline in every run):

| A/B | prompts | baseline ms/step | candidate ms/step | tok/s × |
|---|---|---|---|---|
| `early` alone (quick, one prompt per category) | 14 | 95.2 | 95.4 | 0.998 |
| `s2`: early + the commit on a second GPU stream (quick) | 14 | 95.2 | 94.8 | 1.005 |
| `cv` alone (quick) | 14 | 95.4 | 92.9 | 1.027 |
| `early@cv` (quick) | 14 | 95.4 | 92.7 | 1.030 |
| **`early@cv`, full tune A/B** | **56** | **96.0** | **93.7** | **1.025** |

- Scheduling alone does not help: the CPU path (~10.5 ms) already matched the GPU path, so starting the GPU earlier
  only moved the wait. With the build 3.2 ms shorter, the GPU path is the longer one again, and the early start adds
  a little on top.
- What is left: the step is now ~GPU-bound on drafter steps (the CPU work fits under the ~10 ms draft). The
  remaining host-side gap is the drafter-free lookup steps (~10% of steps). Not done: compiling the BCSA layers'
  non-attention parts (~1 ms more of build, which only those steps would see) and the drafter itself (its GPU time
  is latency-bound small kernels; any fusion has to keep its drafts bit-exact).
- **The drafter cannot run ahead.** It conditions on the target's features of the accepted tokens (layer 68 is
  among them) and on the bonus token, so the next draft depends on the whole verify; only CPU work can overlap.

Per category (tune, tok/s ×baseline): every category +1.6% (math) to +3.4% (rag).

## Lever 2: more accepted tokens per step

**Measured upside first.** `spec/accept_log.py` runs the real loop (baseline config, tune split, asserted against
spec_generate's tokens on 56/56) and logs per step: the DFlash chain (the drafter runs on lookup steps too; its block
K/V are overwritten before they are read), its top-16 candidates and walk scores, for each slot j the chain with the
walk's second choice at j re-walked after it, and the lookup's best copy whether used or not, extended to 15 tokens.
Acceptance of a chain = its prefix that matches the run's own greedy output. `spec_accept/acc_analyze_tune.txt`:

| candidate (tokens per step on the run's 5,053 steps; run = 2.584) | all | short | long+agent |
|---|---|---|---|
| DFlash chain only (no lookup) | −1.7% | −1.5% | −3.7% |
| lookup copy whenever one exists, else DFlash | −4.3% | −4.2% | −5.1% |
| B = 2: run's chain + the other of {DFlash, lookup} | +4.9% | +4.5% | +8.3% |
| B = 2: + DFlash alt at slot 0 | +6.7% | +6.6% | +7.4% |
| B = 2: + DFlash alt at the min-margin slot | +6.4% | +6.5% | +5.8% |
| B = 2: + lookup / DFlash where available, else alt slot 0 | +10.7% | +10.2% | +14.4% |
| B = 2: + DFlash alt at the first rejected slot (oracle, not realisable) | +18.7% | +18.7% | +18.4% |
| 16-row verify of lookup copies (15 drafts) on full-copy steps | +7.1% | +7.3% | +5.8% |

- At the first slot DFlash gets wrong, the target's token is its 2nd candidate 20% of the time and outside the top 16
  34% of the time.
- **These are upper bounds.** Each step's gain is counted on the run's own trajectory, so a chain that accepts more
  also gets credit for tokens the next step would have produced anyway (a 15-token copy accepted in full replaces
  two 8-row copy steps, not one step of 2.6 tokens). For long copies the A/B below keeps about a third of the 7.1% on
  the prompts whose output did not change (−2.1% steps).

**Batched verify of two chains (built, measured, not adopted).** `verify.forward_batch` / `commit_batch`: B chains
against the B = 1 cache. KDA runs the recurrence from the broadcast state (the kernel gets a copy per chain), BCSA the
8q branch once per chain, the dense layer its cached keys plus the chain's own; commit keeps n rows of one chain. The
B = 1 functions are untouched.
- `spec/check_verify_b2.py` (true chain in a random batch row, decoy in the other, random n): KL against plain decode
  5.3e-4 mean / 7.1e-3 max over 368 rows of 4 prompts (B = 1, 14 prompts: 3.4e-4 / 1.2e-2), argmax agree 364/368,
  B = 2 vs B = 1 KL 4.7e-4.
- Cost (`spec/time_b2.py`, verify + argmax): 89.0 → 98.1 ms (+10%) at 197 tokens, 92.2 → 105.4 ms (+14%) at 9.5K.
  More than one chain of 16 rows (+8%): two BCSA passes, the broadcast KDA states, a second dense-attention pass.
- Verdict: the best realisable second chain is worth at most +6.7% tokens per step, before the trajectory discount,
  against +10-14% per step. Not wired into the loop.

**Long lookup copies (`long16`, the final config's lever 2).** When the lookup's copy fills the draft and the copied
source runs on to 15 tokens, verify all 15 (16 rows: the NAX and 8q BCSA limit) instead of 7. The lookup's decision is
unchanged (it sees 7 tokens); the copy is only longer. `long16c` also caps the accepted count recorded into the
Lookup's running means at 7. Code: `lookup.Lookup.propose(long)`, `specdec._long_copy`, the early loop, and
`lookup_replay` (resumed runs replay the same decisions).
- Correctness: `check_verify.py --nax --T 16` (teacher-forced, random rollback 1..16, 14 prompts, 1,488 rows): KL
  3.37e-4 mean, 3.9e-3 p99, 1.3e-2 max, the T = 8 floor (3.39e-4 / 3.85e-3 / 1.18e-2). `specdec.py check --loop
  long16` (128 tokens, one prompt per category, against greedy): 7/14 identical, every divergence at a greedy margin of
  0.0 (6) or 0.125 = one bf16 step (1); the baseline loop on the same prompts: 10/14, all at 0.0.
- Tune A/B (below): ×1.019 pooled against the baseline, ×0.995 against lever 3 alone. Decomposed:
  - 36 prompts with output identical to lever 3: steps 3,012 → 2,950 (−2.1%), ms/step 93.41 → 93.92, tok/s ×1.016;
  - 20 prompts whose text diverged: ×0.965, from per-prompt swings of −19% to +9% (a different, equally valid
    text accepts differently); this part is deterministic for the prompt set, so a repeat would not average it out;
  - agent ×1.091 over lever 3 (4.24 → 4.71 tokens per step, all four outputs identical);
  - by match length (`spec_accept/long_buckets.py`): copies matched over ≥ 16 tokens accept 12.6 of 16 on average
    (76% accept more than an 8-row verify can); matches under 6 accept 2.4-3.4 (4-8% beyond 8), so those 251 of 407
    16-row steps mostly pay +6 ms for nothing, and their rare long accepts roughly pay for it. A gate was not tried.
  - `long16c` (capped means): ×1.020 pooled, the same within noise.
- Verdict: real but small (+1.5-2% where outputs did not move, +9% on agent); kept in the final config and as an
  option. The B = 2 and alt-chain candidates are not worth their cost.

## Tune A/B (`spec_accept/tune_ab.sh`; all 56 prompts, interleaved per prompt, 4 configs, groups as in combined_test)

`runs/spec-accept_tune_<config>.jsonl`. 02:21 end, one attempt, no OOM.

| group | prompts | config | tok/s | × baseline | median ms/step | pooled ms/step | tokens/step |
|---|---|---|---|---|---|---|---|
| all | 56 | baseline (exact+q8g64) | 26.96 | 1.000 | 95.6 | 96.0 | 2.60 |
| all | 56 | early@cv | 27.62 | **1.025** | 93.5 | 93.7 | 2.60 |
| all | 56 | early@cv@long16 | 27.47 | **1.019** | 93.9 | 94.3 | 2.61 |
| all | 56 | early@cv@long16c | 27.51 | 1.020 | 93.8 | 94.3 | 2.61 |
| short (<4K) | 48 | baseline | 27.24 | 1.000 | 95.4 | 95.5 | 2.62 |
| short | 48 | early@cv | 27.92 | 1.025 | 93.3 | 93.2 | 2.62 |
| short | 48 | early@cv@long16 | 27.71 | 1.017 | 93.7 | 93.8 | 2.62 |
| short | 48 | early@cv@long16c | 27.78 | 1.020 | 93.7 | 93.7 | 2.62 |
| long+agent | 8 | baseline | 24.99 | 1.000 | 99.1 | 99.6 | 2.52 |
| long+agent | 8 | early@cv | 25.55 | 1.023 | 96.6 | 97.5 | 2.52 |
| long+agent | 8 | early@cv@long16 | 25.78 | 1.032 | 98.2 | 98.2 | 2.55 |
| long+agent | 8 | early@cv@long16c | 25.65 | 1.027 | 98.5 | 98.7 | 2.55 |
| long (8-12.5K) | 4 | baseline / early@cv / +long16 | 20.77 / 21.22 / 21.02 | 1.000 / 1.022 / 1.012 | 100.1 / 98.1 / 98.1 | | 2.09 / 2.09 / 2.07 |
| agent (4.5-7.3K) | 4 | baseline / early@cv / +long16 | 42.14 / 43.23 / 47.18 | 1.000 / 1.026 / 1.119 | 98.3 / 95.6 / 98.4 | | 4.24 / 4.24 / 4.71 |

Token identity against the baseline: early@cv 56/56 (acceptance 56/56); long16 36/56 (31/48 short, 5/8
long+agent; acceptance 9/56); long16c 40/56. The baseline against combined-test's exact+q8g64 run: 56/56 and 56/56.

Per category, tok/s (baseline / early@cv / +long16; × for both; tokens per step baseline → long16):

| category | baseline | early@cv | +long16 | × L3 | × L3+L2 | tokens/step |
|---|---|---|---|---|---|---|
| agent | 42.14 | 43.23 | 47.18 | 1.026 | 1.119 | 4.24 → 4.71 |
| coding | 31.96 | 32.70 | 33.55 | 1.023 | 1.049 | 3.04 → 3.16 |
| humanities | 24.05 | 24.67 | 25.09 | 1.026 | 1.043 | 2.28 → 2.33 |
| json | 42.17 | 43.13 | 44.99 | 1.023 | 1.067 | 4.02 → 4.24 |
| long | 20.77 | 21.22 | 21.02 | 1.022 | 1.012 | 2.09 → 2.07 |
| math | 41.97 | 42.65 | 43.02 | 1.016 | 1.025 | 4.02 → 4.06 |
| multilingual | 23.73 | 24.21 | 24.06 | 1.020 | 1.014 | 2.28 → 2.30 |
| qa | 21.97 | 22.61 | 21.60 | 1.029 | 0.983 | 2.12 → 2.04 |
| rag | 24.20 | 25.02 | 24.33 | 1.034 | 1.005 | 2.34 → 2.34 |
| reasoning | 31.74 | 32.65 | 32.10 | 1.029 | 1.011 | 3.05 → 3.01 |
| roleplay | 24.66 | 25.24 | 24.45 | 1.023 | 0.991 | 2.37 → 2.31 |
| stem | 24.75 | 25.33 | 24.42 | 1.024 | 0.987 | 2.38 → 2.30 |
| summarization | 26.44 | 27.21 | 27.12 | 1.029 | 1.026 | 2.56 → 2.58 |
| writing | 25.31 | 25.87 | 26.41 | 1.022 | 1.044 | 2.44 → 2.49 |

## Held-out sample (test split, coding + long + agent, 18 prompts; once, back to back)

`specdec.py bench --split test --cats coding,long,agent --new 256 --nax --drafter-bits 4 --head-bits 4 --lookup`,
`VOLUNDR_LMHEAD8=64`: the baseline (default loop) 02:24-02:32, then the final config (`--loop early@cv@long16`)
02:32-02:40, in one lock session; no OOM. Runs `runs/spec-accept_test_{baseline,final}.jsonl`, summary
`spec_accept/heldout_summary.txt`. No greedy run, the other 66 test prompts not run.

| group | prompts | baseline tok/s | final tok/s | × | median ms/step | pooled ms/step | tokens/step |
|---|---|---|---|---|---|---|---|
| all | 18 | 27.98 | 28.88 | **1.032** | 98.0 → 98.4 | 98.1 → 97.3 | 2.77 → 2.85 |
| short (<4K: 6 coding + 2 agent) | 8 | 33.13 | 34.30 | 1.035 | 95.3 → 94.1 | 95.2 → 94.1 | 3.18 → 3.27 |
| long+agent | 12 | 25.72 | 26.62 | 1.035 | 99.6 → 98.6 | 99.9 → 99.2 | 2.60 → 2.69 |
| long (14-21.5K) | 6 | 22.69 | 23.18 | 1.021 | 99.9 → 99.2 | 100.2 → 99.4 | 2.28 → 2.33 |
| agent (3-7.9K) | 6 | 40.73 | 45.25 | **1.111** | 98.0 → 98.4 | 98.4 → 97.8 | 4.15 → 4.62 |

- The baseline reproduces combined-test's held-out exact+q8g64 run token for token and step for step (18/18, 18/18;
  27.79 tok/s then, 27.98 now).
- Final against baseline: tokens identical on 11/18 (all 6 agent outputs; divergences at tokens 98-212).
  - Agent: same text, 6-20% more tokens per step from the long copies.
  - Coding: every prompt's step 0.6-1.7 ms shorter, lever 3's share.
  - Long: −1.8 to +0.5 ms per step; 4 of 6 outputs diverged.
- Back to back, not interleaved, so the step times carry session drift (canaries 27.8 / 28.1, then 28.6 / 28.2).

## Correctness gates

| gate | result |
|---|---|
| step 0: cache state after prefill, synced vs unsynced | bit-identical, 9/9 runs; spec tokens and acceptance identical |
| `spec_accept/check_compile.py` (compiled vs eager verify, T = 8 and 16, 3 prompts × 24 steps) | **bit-identical** |
| tune A/B: early@cv vs baseline | tokens 56/56, per-step acceptance 56/56 |
| baseline vs combined-test's exact+q8g64 tune run | tokens 56/56, acceptance 56/56 |
| `spec/check_verify.py --nax --T 16` (8-bit head) | KL 3.37e-4 / p99 3.9e-3 / max 1.3e-2, the T = 8 floor |
| `specdec.py check --loop long16` vs greedy | 7/14 identical; divergences at margin 0.0 (6) and 0.125 (1) |
| `spec/check_verify_b2.py` (B = 2, not adopted) | KL 5.3e-4 mean / 7.1e-3 max; B = 2 vs B = 1 4.7e-4 |
| `spec/check_verify.py --nax` (B = 1 path, bf16 head, as combined-test) | **identical line for line** to `combined_test/check_verify.log` (1,520 rows, KL 3.385e-4 / p99 3.85e-3 / max 1.18e-2) |

Not run: tests/test_volundr.py, test_decode_fused.py, test_bcsa_verify.py and check_combined.py. This branch changes
none of volundr.py, kernels.py, decode_fused.py or bcsa_verify.py, and verify.py's B = 1 functions only gained the
optional `layer_sync` / `compiled` paths, which default off and are covered by the gates above.

## Canaries (TFLOP/s, before / after each step; idle 26-28; all 27.0-29.0, none below 22)

From `spec_accept/canaries.py`:

| time | step | canaries |
|---|---|---|
| 00:14, 00:15 | peak_probe, layer-synced / unsynchronised | 28.3 / 28.1, 27.6 / 27.8 |
| 00:17-00:24 | step0_check (first attempt stopped on a comparison bug; then long/03, then 4 short prompts) | 27.6 / 28.1, 28.0 / 27.3, 27.8 / 28.0 |
| 00:29-00:32 | prof_step (before); time_parts short / long | 28.3 / 27.7; 27.4 / 27.4, 28.4 / 27.5 |
| 00:32 | accept_log, tune (56 prompts) | 27.7 / 27.0 |
| 00:56 | quick A/B early / s2 | 28.1 / 28.8 |
| 01:04, 01:07 | check_verify_b2; time_b2 | 28.6 / 28.2; 28.4 / 27.2 |
| 01:08, 01:12 | check_compile; prof_vbuild | 28.6 / 28.1; 29.0 / 28.1 |
| 01:12-01:20 | check_verify T = 16; specdec check base / long16 | 28.6 / 28.4; 28.4 / 27.7, 28.4 / 28.0 |
| 01:25 | quick A/B cv / early@cv | 28.2 / 28.1 |
| 01:35, 01:53, 02:06 | tune A/B groups 1 / 2 / 3 | 28.3 / 28.5, 28.8 / 28.3, 28.3 / 28.0 |
| 02:24, 02:32 | held-out baseline / final | 27.8 / 28.1, 28.6 / 28.2 |
| 02:40, 02:43 | prof_step cv / cv + long16 | 28.6 / 28.0, 28.4 / 27.9 |
| 02:47 | check_verify (B = 1 gate) | 28.6 / 28.2 |

Load average 3-7 throughout. Swap used 11-14 GB for most of the session; the first step-0 check, which ran the old
unsynchronised prefill (50-57 GB peaks), pushed it to 23.0 of 24.5 GB once.

## Open for the user

- **Defaults (decided by the user at the merge).** `early@cv` is now the default loop: it's bit-identical, so
  `specdec.py bench` and ab_suite's `exact+q8g64` give the same tokens as before, about 2.5% faster. `@long16`
  changes outputs at ties (like the 8-bit head) and stays opt-in; the 8-bit head stays opt-in too.
- The prefill fix is on by default (`VOLUNDR_PREFILL_SYNC=0` turns it off).

## Files

- `spec/specdec.py`: layer-synced `spec_prefill`; `loop_options` (flags base / early / s2 / cv / nocv / long<L>[c][g<b>]; default early@cv);
  `_spec_loop_early`; `_long_copy`; `warm_up`; `--loop`; `lookup_replay` with the loop's flags.
- `spec/verify.py`: `forward(layer_sync=, compiled=)`, the compiled KDA layers; `forward_batch` / `commit_batch` (B
  chains, not used by the loop).
- `spec/lookup.py`: `find(max_draft)`, `propose(long)`, `record(cap)`; defaults unchanged.
- `spec/ab_suite.py`: `config@flags`, records seg_s and the loop options; resume replays with the flags.
- New: `spec/prof_step.py`, `spec/prof_vbuild.py`, `spec/accept_log.py`, `spec/check_verify_b2.py`, `spec/time_b2.py`,
  `spec_accept/` (chains, logs, checks, analysis).
