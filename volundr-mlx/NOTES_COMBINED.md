# Combined stack: bcsa-verify-8q + decode-quick-wins (branch combined-test, 2026-10-19/20)

Status: done (2026-10-20 00:05).

## Resume

Nothing is pending. To reproduce, run each script under `./with_gpu_lock.sh` from `volundr-mlx`, with
`PY=/Users/ingemarrask/personal-dev/llms/.venv/bin/python`. The scripts are in `combined_test/`:
- `gates.sh`: tests, parity dump, check_verify, check_combined.
- `retry.sh 8 tune`: the tune spec A/B, via `tune_chain.sh` with `--resume` until complete.
- `decode_chain.sh`: plain decode.
- `retry.sh 10 test`: the held-out sample.
- `peak_chain*.sh`: the peak-memory probes.
- `agent_rerun.sh`: the agent/02-03 repeat.

To summarise the runs: `$PY combined_test/analyze.py spec/runs/combined-test_tune_spec baseline,exact,exact+q8g64`
(for the held-out runs, the same command with `combined-test_test` and `baseline,exact+q8g64`). The run files are
`spec/runs/combined-test_*.jsonl`, in the shared, gitignored runs dir; the summaries are in `combined_test/*.txt`
and `*.log`.

Open for the user:
- the default of the 8-bit head (see the last section);
- the spec harness's prefill memory (see "Peak memory"). Neither is a merge issue.

## Summary

| | baseline (= main) | exact (new defaults) | exact + 8-bit head (g64) |
|---|---|---|---|
| tune spec, all 56, tok/s | 24.47 | 26.01 (×1.063) | 26.99 (×1.103) |
| tune spec, short (<4K, 48) | 25.44 | 26.33 (×1.035) | 27.40 (×1.077) |
| tune spec, long+agent (8) | 19.03 | 23.83 (×1.252) | 24.30 (×1.277) |
| tune spec, median ms/step | 100.9 | 98.1 | 94.7 |
| plain decode, short, ms/token (best of 3) | 72.09 | 70.09 (−2.8%) | 65.84 (−8.7%) |
| plain decode, 19K, ms/token (best of 3) | 78.45 | 73.39 (−6.4%) | 69.04 (−12.0%) |
| held-out coding+long+agent (18), tok/s | 24.67 | not run | 27.79 (×1.126) |

- **The two branches compose exactly.** Every gate passes.
  - The verify path is bit-identical with the fusion on or off, under both the 8q and the prefill BCSA verify.
  - The default parity dump is bit-identical to `parity/mlx.bin`.
  - `baseline` reproduces main's tune run token for token and step for step: 56/56 tokens and 56/56 per-step
    acceptance.
- **Speed.** The long-context gain is bcsa-verify-8q's: about 120 → 102 ms per long/agent verify step. The fusion and
  the 8-bit head add a few ms per step at every length.
- **Memory.** The 57-58 GB per-process peaks in the tune records are mostly the spec harness's command-buffer
  settings, and main has them too (55.7 GB single-config on long/03). With MLX's default buffers, main peaks at
  30.5 GB on the same prompt and the combined exact+q8g64 at 27.1 GB.
  - The merged code adds +1.5 GB under the spec settings, from decode-quick-wins (bcsa-verify-8q adds 0).
  - The three-config A/B process adds a further 1.35-4.7 GB. See below.

## Merge: conflicts and resolution

`git switch -c combined-test` from main 0732112, then the two merges:
- **e3ae575:** merge of `refs/heads/bcsa-verify-8q` (7a42e82).
- **51625d1:** merge of `refs/heads/decode-quick-wins` (9d4026a).

**No textual conflicts.** The briefing expected conflicts in volundr.py's BCSA attention class, but the two branches
touch disjoint files:
- **bcsa-verify-8q** changes `spec/verify.py`, whose `_bcsa_verify` is the verify branch, plus new files:
  `bcsa_verify.py`, `tests/test_bcsa_verify.py`, `spec/prof_bcsa_verify.py`, `spec/bench_bcsa_attend.py` and
  `spec/bcsa_verify_8q/`. It does not touch volundr.py.
- **decode-quick-wins** changes `volundr.py` (decode fusion call sites, `_first` / `_gather_blocks` views,
  lm_head q8), `kernels.py` (`rms_norm` loops and wide variant, `gated_delta_*` `out_dtype`), `generate.py` and
  `spec/specdec.py`, plus new files.
- Neither branch touches a README.

**Semantic composition**, checked by reading and then proved by the gates:
- **Verify, 8q path.** It squeezes its own views and calls `kernels.bcsa_scores / bcsa_topk / far_union` and
  `bcsa_verify.attend`. decode-quick-wins changed none of these.
- **Verify, prefill path.** It keeps `q[0]` and main's `_select`.
- **Verify picks up decode-quick-wins' kernels** where they are shared, all bit-exact: `MHC.mix` (`decode_fused.mix_norm`
  without the norm), the wide `kernels.rms_norm`, and the OffsetRMSNorm one-dispatch `rms_norm_mlx` with cached fp32
  weights.
- **The layer call stays valid.** `DecoderLayer.__call__` gained `h=None, split=False`, so verify's
  `layer(x_in, c, rows)` call is unchanged.
- **The KDA / layer fusions do not reach verify.** `spec/verify.py` has its own `_kda` / `_layer`.
- **Result:** the verify path uses the 8q kernels, and decode uses the fused kernels and the views.

**Code added on this branch** (0e096de, 720ebdc):
- **`ab_modes.py`:**
  - mods `+vprefill` / `+v8q`, which set `spec/verify.py`'s `BCSA_IMPL` at run time. The default is prefill for `main`
    and 8q otherwise;
  - aliases `baseline` = main (decode_fused off, main's strided `rms_norm`, uncached norm weights, copying
    `x[0]` / `p[b]`, prefill verify) and `exact` = fused;
  - `describe()`.
- **`spec/ab_suite.py`:**
  - records `fused / views / bcsa_verify / peak_gb`;
  - **`--resume` now replays the done prompts' records into each config's Lookup** (`specdec.lookup_replay`), so a
    resumed group keeps the Lookup history of an uninterrupted run.
  - `combined_test/check_replay.py` re-derives every recorded proposal on ~31K steps: main's reference run, the
    decode-quick-wins runs and these. The group-3 results came from 6 processes because of OOMs (below), and they
    match main's uninterrupted reference run step for step.
- **`spec/specdec.py`:**
  - `--config` (an ab_modes config; the held-out baseline needs it, because env vars cannot turn the views off);
  - `--resume` for bench;
  - records the actual head and switches.
- **`spec/check_combined.py`:** see Gates.

Not changed: the code paths under test (volundr.py, verify.py, kernels, decode_fused.py, bcsa_verify.py).

**Prose notes.**
- NOTES_BCSA_VERIFY.md lists volundr.py's decode `x[0]` / `p[b]` copies as "for Decode quick wins". In this branch
  decode-quick-wins' views have fixed them.
- NOTES_decode.md says prefill-size calls keep the old kernels. That holds for the row norms only. `MHC.mix`'s fused
  path, the KDA fused path and `_fused_layers` are not size-gated, so decode_fused's kernels also run at prefill
  sizes. This is exact (the parity dump covers prefill), and it is relevant to the memory note below.

## Gates (all passed before any benchmark; 22:10-22:21)

| gate | result |
|---|---|
| `tests/test_volundr.py` | 10/10 |
| `tests/test_decode_fused.py` | 8/8 (including the views and the tiny model fused == unfused) |
| `tests/test_bcsa_verify.py` (bcsa-verify-8q's kernel test) | ok: 9 shapes against fp64 and against the prefill kernels (max 1 bf16 ulp, 4-9% of elements) |
| `dump_logits.py`, default config (fused + views, bf16 head), `parity/prompts.json` | **bit-identical to `parity/mlx.bin`** (`cmp`); 5 prompts up to 14K, prefill + 240 decode rows |
| `spec/check_verify.py --nax` (rollback, 14 prompts, 1520 rows, seed 7) | KL mean **3.385e-4**, p99 3.85e-3, max 1.18e-2; argmax agree 1512/1520. **Identical, line for line and to every digit, to `bcsa-verify-8q_check_verify_8q.log`**, so the fusion changes nothing in verify or decode |
| `spec/check_combined.py --nax` (new; 3 prompts of 197 / 7,260 / 12,547 tokens, 24 verify+commit steps each, random n) | `main+v8q == fused` and `main == fused+vprefill` **bit-identical** on all 576 rows. Prefill against 8q verify: KL mean 1.6e-4-6.2e-4, argmax agree 570/576 |

Logs: `combined_test/gates.log`, `check_verify.log`, `check_combined.log`; tokens `parity/combined-test_default.json`.

## Spec A/B, tune split

**Protocol:**
- `spec/ab_suite.py spec` with the frozen config: NAX verify, 4-bit drafter, 4-bit head copy, lookup, T = 8,
  256 new tokens, `MLX_MAX_OPS_PER_BUFFER=100`.
- All 56 tune prompts; per prompt the three configs run back to back.
- One process per category group, as in decode-quick-wins' tune_chain.sh (coding..rag / reasoning..writing /
  agent,long), so each config's Lookup sees the same history as that run.
- **Group 3 needed 9 processes in all.** It hit 8 Metal OOMs under system memory pressure (see "Peak memory").
  - The first start OOM'd at agent/01. Its agent/00 records were set aside in
    `runs/combined-test_tune_spec_*_g3_oom_partial.jsonl`.
  - The second start reran group 3 from the beginning and completed agent/00 and agent/01 before it OOM'd.
  - Then came 7 attempts of `retry.sh`, each resuming with the exact Lookup replay; attempt 7 finished long/03.
  - The kept records come from 6 of these processes.

| group | prompts | config | tok/s | × baseline | median ms/step | pooled ms/step | tokens/step |
|---|---|---|---|---|---|---|---|
| all | 56 | baseline | 24.47 | 1.000 | 100.9 | 103.9 | 2.56 |
| all | 56 | exact | 26.01 | **1.063** | 98.1 | 99.1 | 2.59 |
| all | 56 | exact+q8g64 | 26.99 | **1.103** | 94.7 | 95.9 | 2.60 |
| short (<4K) | 48 | baseline | 25.44 | 1.000 | 100.2 | 101.5 | 2.60 |
| short (<4K) | 48 | exact | 26.33 | **1.035** | 96.7 | 98.4 | 2.61 |
| short (<4K) | 48 | exact+q8g64 | 27.40 | **1.077** | 94.1 | 95.0 | 2.62 |
| long+agent | 8 | baseline | 19.03 | 1.000 | 120.6 | 120.3 | 2.30 |
| long+agent | 8 | exact | 23.83 | **1.252** | 101.9 | 104.5 | 2.52 |
| long+agent | 8 | exact+q8g64 | 24.30 | **1.277** | 99.4 | 102.5 | 2.52 |
| long (8-12.5K) | 4 | baseline | 15.57 | 1.000 | 120.1 | 120.0 | 1.87 |
| long | 4 | exact | 20.21 | 1.298 | 102.5 | 102.8 | 2.09 |
| long | 4 | exact+q8g64 | 20.93 | 1.344 | 99.4 | 99.3 | 2.09 |
| agent (4.5-7.3K) | 4 | baseline | 34.36 | 1.000 | 121.4 | 121.4 | 4.25 |
| agent | 4 | exact | 37.15 | 1.081 | 101.9 | 111.4 | 4.24 |
| agent | 4 | exact+q8g64 | 35.90 | 1.045 | 106.1 | 115.3 | 4.24 |

**The agent row's pooled numbers carry one noise episode.**
- agent/02 ran during the first resume attempt (swap at ~24 of 25.6 GB). Its exact and exact+q8g64 runs took 121.4
  and 128.7 ms/step, against ~100 for every other long/agent prompt.
- A repeat of agent/02-03 with the same, replayed Lookup history (`combined_test/agent_rerun_summary.txt`) gave
  identical tokens and acceptance for all three configs, and 117.8 / 101.2 / 98.2 ms/step on agent/02.
- With the repeat substituted:
  - agent is 34.91 / 40.85 (×1.170) / 41.37 (×1.185);
  - long+agent is 19.09 / 24.30 (×1.273) / 25.05 (×1.312).
- One more spread: agent/03 under exact+q8g64 is 112 ms/step in both runs, against 102 under exact. It has only 16
  steps, so it is not diagnosed.

**Token identity and acceptance against baseline:**
- **baseline vs main's run** (`runs/decode-quick-wins_tune_spec_main.jsonl`, same grouping): tokens 56/56,
  per-step acceptance 56/56.
- **exact vs baseline:** tokens identical on 22/56 (18/48 short, 4/8 long+agent), acceptance identical on 6/56.
  Tokens per step: 2.59 against 2.56; long 2.09 against 1.87.
  - The 8q verify rounds differently from the prefill kernels (at most 1 bf16 ulp), so spec breaks exact bf16 ties
    differently and follows another, equally valid greedy path (NOTES_BCSA_VERIFY.md: every divergence from greedy
    starts at a zero-margin tie).
  - For comparison: exact vs bcsa-verify-8q's own 8q tune run (a different Lookup history) gives tokens 50/56.
- **exact+q8g64 vs baseline:** tokens 25/56 (21/48 short, 4/8 long+agent), acceptance 7/56, tokens per step 2.60.
  The 8-bit head adds its own divergences (decode-quick-wins measured 44/56 identical against the bf16 head).

The long gain matches bcsa-verify-8q's own A/B (×1.28) and its step time (126 → 109 ms there, 120 → 102.5 here).
The short gain (×1.035) is decode-quick-wins' fusion (×1.024 there, measured against an earlier main); the 8q verify
is neutral below 4K. The 8-bit head saves a further ~3.3 ms per step (median 98.1 → 94.7, pooled 99.1 → 95.9),
against 3.1 in decode-quick-wins.

## Plain decode (`bench_decode.py`, configs interleaved per rep on one loaded model)

| context | config | best ms/token | median | tok/s (best) | vs baseline |
|---|---|---|---|---|---|
| short (85-token prompt, 128 timed tokens) | baseline | 72.09 | 72.21 | 13.87 | - |
| | exact | 70.09 | 70.27 | 14.27 | −2.8% |
| | exact+q8g64 | 65.84 | 65.91 | 15.19 | −8.7% |
| 19K (`bench/long_prompt.txt`, 19,045 tokens, shared cache, 48 timed tokens per segment) | baseline | 78.45 | 79.17 | 12.75 | - |
| | exact | 73.39 | 73.56 | 13.63 | −6.4% |
| | exact+q8g64 | 69.04 | 69.26 | 14.48 | −12.0% |

- Best and median agree within 0.8 ms everywhere.
- Short greedy tokens are identical across all three configs. The 19K run uses a shared cache, so its tokens are not
  compared.
- Plain decode has no verify, so `exact` here is decode-quick-wins' fusion plus views:
  - short: −2.0 ms (−1.5 there);
  - 19K: −5.1 ms (−5.2 there).
- The 8-bit head saves a further 4.3 ms at both lengths (5.4 / 5.8 there).

## Held-out test split: a short sample

`specdec.py bench --split test --cats coding,long,agent`, frozen config, 18 prompts:
- baseline (`--config baseline`), 23:37-23:45, then exact+q8g64 (`VOLUNDR_LMHEAD8=64`, the env-selected defaults
  otherwise), 23:45-23:53. Back to back in one lock session; no OOM.
- No greedy run; the full 84 prompts were not run. Files: `runs/combined-test_test_{baseline,exact+q8g64}.jsonl`;
  summary in `combined_test/test_summary.txt`.

| group | prompts | baseline tok/s | exact+q8g64 tok/s | × | median ms/step | tokens/step |
|---|---|---|---|---|---|---|
| all | 18 | 24.67 | 27.79 | **1.126** | 117.7 → 97.8 | 2.88 → 2.77 |
| short (<4K: 6 coding + 2 agent) | 8 | 32.14 | 33.13 | 1.031 | 101.9 → 95.0 | 3.32 → 3.18 |
| long+agent | 12 | 21.54 | 25.46 | **1.182** | 122.2 → 99.9 | 2.68 → 2.60 |
| long (14-21.5K) | 6 | 18.96 | 22.40 | 1.182 | 123.5 → 101.2 | 2.36 → 2.28 |
| agent (3-7.9K) | 6 | 34.65 | 40.94 | 1.182 | 117.7 → 97.8 | 4.24 → 4.15 |

- **Step time is the clean signal:**
  - every long prompt went from 122-126 to 100-104 ms;
  - every agent prompt from 108-128 to 95-100 ms;
  - every coding prompt from ~101 to ~95 ms.
- **Tokens per step fell slightly, from trajectory divergence:** tokens are identical on 9/18, and two prompts shifted
  most (coding/04 4.41 → 3.00, long/07 2.50 → 1.83). On the tune split it rose instead (2.56 → 2.60), so this is
  sample noise, not a quality loss.
- **baseline against main's earlier held-out runs** (`runs/test_frozen{,_r2}.jsonl`): tokens 16/18, acceptance
  14/18. Those were one 84-prompt process, so their Lookup history differs; this is not an exactness failure (the tune
  check above is).
- **exact+q8g64 against bcsa-verify-8q's 8q run:** tokens 14/18.

## Peak memory and the OOMs

**What happened.** Tune group 3 (agent, long) failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` in a spec
prefill, 8 times across 8 processes.
- At the time, fseventsd held 30 GB (all compressed) and swap was 15-25 GB full.
- The ab_suite records show a per-prompt peak of 57.1 (baseline) and 58.5-58.6 GB (exact, exact+q8g64). The
  recommended working set is 55.7 GB.

**Probes.** Fresh process, one config, long/03-long3 (12,547 tokens). Each figure is `mx.get_peak_memory()` from a
reset counter over spec prefill plus 256 tokens. Run by `combined_test/peak_probe.py`, log `combined_test/peak.log`.
main and the branches are read-only `git archive` exports.

| code | head | command buffers | after load | peak |
|---|---|---|---|---|
| main | bf16 | spec (100 ops, no MB cap) | 26.04 | **55.71** |
| bcsa-verify-8q | bf16 | spec | 26.04 | 55.71 |
| decode-quick-wins | bf16 | spec | 26.04 | 57.19 |
| combined, baseline config | bf16 | spec | 26.04 | 55.71 |
| combined, exact | bf16 | spec | 26.04 | 57.19 |
| combined, exact+q8g64 | q8 | spec | 24.84 | **53.80** |
| combined, fused+copyidx (views off) | bf16 | spec | 26.04 | 56.64 |
| combined, unfused (decode_fused off, views on) | bf16 | spec | 26.04 | ≥ 56.81 (OOM at that point) |
| combined, ab_suite set-up (3 configs, bf16 + q8 heads), running exact+q8g64 | q8 | spec | 27.39 | 58.54 |
| main | bf16 | MLX default | 26.04 | **30.50** |
| combined, exact+q8g64 | q8 | MLX default | 24.84 | **27.07** |
| combined, plain greedy (no drafter), exact+q8g64 | q8 | MLX default | 22.96 | 25.04 |

**Cause.**
1. **The spec harness's command-buffer settings: ~25-28 GB, pre-existing, on main too.**
   - `specdec.py bench` / `check` and `ab_suite.py spec` set `MLX_MAX_OPS_PER_BUFFER=100` and
     `MLX_MAX_MB_PER_BUFFER=1000000`.
   - Measured: changing only these two variables, on the same code and prompt, moves main from 55.7 to 30.5 GB and
     combined from 53.8 to 27.1 GB.
   - Inferred, not traced: the arrays a command buffer uses stay alive until it completes. With 100 ops per buffer and
     no size cap, a 2048-token prefill chunk's large activations accumulate.
   - bcsa-verify-8q's own profile on main-equivalent code shows the same effect: 28.95 GB at 197 tokens, 37.7-40.5 at
     1.5K, 54.7-55.7 from 7.3K up (`spec/bcsa_verify_8q/*prof*.jsonl`). It grows with the chunk's length up to 2048,
     not with the context.
   - The phase-1 figures (24.9 / 29.1 GB; `bench/short-think.json` 24.85 and `bench/long-14k.json` 28.51 GB) come
     from `generate.py`: plain greedy, no drafter, and MLX's default buffers (generate.py, dump_logits.py,
     bench_decode.py and check_verify.py set neither variable).
2. **The merged code: +1.48 GB, under the spec buffer settings only. It comes from decode-quick-wins; the 8q verify
   adds nothing** (bcsa-verify-8q = main = 55.71).
   - It is not decode_fused's kernels alone (unfused already exceeds 56.8), nor the views alone (views off: 56.64).
   - Most likely a different count and order of prefill ops changes how many large intermediates each 100-op buffer
     holds. It is not isolated further. Under MLX's default buffers the merged code's peak is the low 27.1 GB.
   - With the 8-bit head, the merged code is 1.9 GB below main (53.8 against 55.7), since the head is 1.2 GB
     smaller.
3. **The A/B harness: +1.35 GB (exact) to +4.7 GB (exact+q8g64 against single-config) on top.** `Modes` keeps the
   bf16 head (2.54 GB) next to the q8 copy (1.35 GB) for the whole process; three configs' kernels are compiled. The
   drafter is in every spec row of the table (main included), so it is not the difference.

**For the user.** This is not a regression introduced by the merge; the user should still know:
- the frozen spec config already runs main at the 55.7 GB recommended working set on prompts of a few thousand tokens
  or more (54.7 GB at 7.3K, 55.7 GB at 12.5-23.6K), and OOMs when other apps hold ~30 GB;
- the merge adds 1.5 GB to that with the bf16 head and takes 1.9 GB off it with the 8-bit head.

The obvious lever is a size cap on the command buffers for the prefill, for example a finite
`MLX_MAX_MB_PER_BUFFER`, which MLX reads once at device init. It has not been measured here, and it needs a decode
speed A/B, because specdec.py's comment credits the large buffers with the drafter's async overlap.

## Canaries (TFLOP/s; idle 26-28; all ≥ 22.9, none below the 22 threshold)

| step | before / after |
|---|---|
| gates: test_volundr, test_decode_fused, test_bcsa_verify | 25.6 / 26.0, 25.6 / 25.2, 25.7 / 25.8 |
| gates: parity dump, check_verify, check_combined | 25.3 / 24.0, 22.9 / 24.9, 25.4 / 25.5 |
| tune group 1 (22:21-22:35) / group 2 (22:35-22:45) | 26.5 / 27.7, 28.2 / 28.2 |
| tune group 3: two OOM'd starts, then resume attempts 1-7 (attempt 7 completed it) | 27.5 / 26.6, 26.8 / 27.7; 28.3 / 27.7, 27.5 / 28.2, 27.8 / 27.8, 27.6 / 28.3, 27.8 / 27.5, 27.9 / 28.5, 28.2 / 27.2 |
| plain decode short / 19K | 28.0 / 27.3, 28.0 / 28.1 |
| memory probes (22:54, 23:57) | 27.9 / 27.5, 28.1 / 27.5 |
| peak probes (16 processes, 23:18-00:00) | 25.8-28.4 (the low 25.8 is after combined-exact) |
| held-out baseline / exact+q8g64 | 28.4 / 27.4, 27.2 / 27.9 |
| agent/02-03 repeat | 28.1 / 27.8 |

**Load average.**
- At the start of the tune run: 29 (bfs and node dev servers).
- From 22:35: 3-12.
- The tune A/B is interleaved per prompt, so load affects all three configs alike.

The sessions ran from 22:10 to 00:05, outside the 18:33-18:37 CPU stress window.

## The 8-bit lm_head default is the user's decision

`VOLUNDR_LMHEAD8` stays opt-in (off). On this stack:
- exact+q8g64 adds ×1.038 spec throughput over exact on tune (26.99 against 26.01), −4.3 ms/token in plain decode, and
  −1.2 GB of weights;
- decode-quick-wins measured the cost: KL 1.6e-4 against the bf16 head, top-1 239/240, 44/56 spec outputs identical.

Whether that trade becomes the default is for the user to decide; this branch does not change it.
