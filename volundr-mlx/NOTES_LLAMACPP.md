# Blockway's llama.cpp fork on Metal (2026-10-20)

Status: done, for two commits of the fork's `volundr` branch.

- **67e4001**, the first test: the mHC and Engram ops had no Metal kernels and ran on the CPU.
- **bb1c8bc**, the current tip:
  - Blockway added Metal kernels for `SINKHORN` and `DSV4_HC_COMB/PRE/POST` (3ea6ab6) and for `NGRAM_HASH` (bb1c8bc).
  - The 67e4001 binaries are kept in `~/personal-dev/llama.cpp-volundr/build-67e4001/bin`, with their rpath
    re-pointed; `build/` is now bb1c8bc.

Scripts, logs and raw results are in `llamacpp/`. The two `GGML_SCHED_DEBUG=2` placement dumps
(`logs/llama_diag*.server.log.gz`, 6 MB each) are not committed.

## Verdict

- **It works on Metal.** The fork (github.com/BlockWayz/llama.cpp, branch `volundr`) builds with the default Metal
  backend. It runs the Q4_K_M GGUF with `-ngl 99 --jinja`, and all 73 layers are offloaded.
- **The output is sane.** Replies are coherent in English, Swedish, Chinese, code and arithmetic. The 19K-token CUDA
  file is summarized correctly.
  - Greedy output is identical from run to run, and identical between the two commits.
  - The CPU backend gives the same replies as Metal, except at one near-tie.
- **On bb1c8bc, the whole graph runs on the GPU** apart from the token-embedding lookup. That is 2 graph splits in
  decode and 11 in prefill, all but one of them Metal to Metal, against 298 before.
- **bb1c8bc is faster, but still slower than our port everywhere:**
  - Decode: 9.8 tok/s short (+14%) and 7.6 at 19K (+43%).
  - Prefill: 294 tok/s short (+30%) and 137 at 19K (+17%).
  - Against our port: decode is 1.4x slower than our plain greedy and 2.8x slower than our speculative loop. At 19K
    context, prefill is 2.7x slower and decode 1.8x slower.
  - The fork has no speculative decoding.
- **Why it is still slower:** this is no longer CPU fallbacks. Decode takes 102 ms/token, against a weight-read floor
  of ~73 ms and our 72 ms step. Where the remaining ~30 ms goes was not profiled.
- **As a second route on a Mac,** it is usable for correctness cross-checks and is no longer crippled. It is still
  not a speed alternative.

## bb1c8bc against 67e4001

The same machine, settings, prompts and scripts as the first test. Every GPU job ran alone under `with_gpu_lock.sh`,
with a canary of 23.0–28.3 TFLOP/s before and after each measurement run.

| | 67e4001 (CPU fallbacks) | bb1c8bc (Metal kernels) | change | our MLX port |
|---|---|---|---|---|
| Graph splits, prefill / decode | 298 / 298 | 11 / 2 | | |
| Ops on the CPU per forward | 221 (all mHC, Engram hashes, embedding) | 1 (embedding `GET_ROWS`) | | |
| tg64, depth 0 | 8.57 ± 0.06 | 9.77 ± 0.33 | +14% | 13.8 plain, 27.6 speculative |
| tg64 @ d1024 | 8.13 ± 0.05 | 9.28 ± 0.49 | +14% | — |
| tg64 @ d8192 | 5.96 ± 0.19 | 7.38 ± 0.17 | +24% | — |
| Decode through llama-server, short prompts | 6.9–7.6 | 9.5–10.2 | ~+35% | 14.5–34.6 speculative (today) |
| Decode through llama-server, 19K prompt | 5.34 (187 ms/token) | 7.65 (131 ms/token) | +43% | 13.6 plain (73.4 ms), 13.5 speculative |
| pp512 | 225.5 ± 5.6 | 293.8 ± 4.0 | +30% | — |
| pp2048 | 225.0 ± 3.4 | 284.3 ± 9.8 | +26% | — |
| pp512 / pp2048 @ d1024 | 207.6 / 217.4 | 288.2 / 259.0 | +39% / +19% | — |
| pp512 / pp2048 @ d8192 | 127.4 / 120.4 | 124.1 / 134.7 | ~0 / +12% | — |
| Prefill, 19K prompt (llama-server) | 117 tok/s (162 s) | 137 tok/s (139 s) | +17% | 374 tok/s |
| Allocated memory at `-c 32768` | 31,314 MiB (32.8 GB) | 31,366 MiB (32.9 GB) | | ~29 GB at 19K |

All tok/s figures are from llama-bench with 3 repetitions (± is the standard deviation), unless the row says
llama-server.

- **The new kernels pass test-backend-ops on Metal:** 117/117 cases, against the CPU backend
  (`test_backend_ops_bb1c8bc.txt`).
  - `SINKHORN` 27, `DSV4_HC_COMB` 12, `DSV4_HC_PRE` 21, `DSV4_HC_POST` 18, `NGRAM_HASH` 39.
- **Op placement** (`GGML_SCHED_DEBUG=2`, `-lv 5`, `logs/llama_diag_bb1c8bc.server.log.gz`): the only node left on
  the CPU is the token-embedding `GET_ROWS`.
  - Decode has 2 splits: the embedding on the CPU, then the whole model on Metal.
  - Prefill has 11 splits. Beyond the embedding split, they are Metal to Metal: the scheduler caps a split at 30
    inputs, and the BCSA mask inputs (2–8 MB views at 512 tokens) fill those slots.
  - llama-bench now uses 19 s of CPU time across the matrix, against 796 s before.
- **Output, the six-prompt check:**
  - All six greedy replies are byte-identical to the 67e4001 Metal run.
  - Chosen-token logprobs match the 67e4001 Metal run to within 0.001.
  - Against the CPU backend, bb1c8bc reproduces the 67e4001 Metal comparison exactly: four of five identical, and
    `swedish` splits at the same near-tie at token 6 (' rä' 0.406 / ' r' 0.372).
  - So the Metal kernels compute the same thing as the CPU ops they replace.
  - The comparison with our MLX port is unchanged: the same first-divergence points as below.
- **Memory:** about the same as before.
  - Metal: 29,946 MiB (model 25,360, KV and recurrent state 2,760, compute 1,825).
  - Host: 1,420 MiB (token embedding 1,288, compute 132).
- **What 67e4001 got wrong in its explanation:**
  - Removing 296 of the 298 splits saves ~15 ms per decode token (117 → 102 ms), not the ~45 ms estimated before.
  - What is left is Metal-side and unprofiled.
- **Two side effects that look tied to the CPU fallbacks:**
  - llama-server now decodes as fast as llama-bench tg64 (~101 ms/token on short prompts). On 67e4001 it was ~15%
    slower.
  - Requesting top-5 logprobs no longer slows decode. On 67e4001, `fib` took 40 s with logprobs and 22 s without;
    now both take 17 s.
- **Prefill at 8K depth hardly moved:** it is dominated by BCSA prompt processing, which scores every query against
  every cached position (the fork's README: roughly quadratic).

## 67e4001 numbers

M5 Pro 64 GB. Every run was under `with_gpu_lock.sh`, one GPU job at a time, with a canary of 21.7–27.7 TFLOP/s
before and after.

- llama.cpp settings: `-ngl 99`, flash attention auto, 6 threads, ubatch 512.
- llama-bench: 3 repetitions, ± is the standard deviation.
- Our numbers come from the README, except where marked "today" (the `./serve.sh` run on the same prompts).

| | llama.cpp Metal, Q4_K_M, 67e4001 | our MLX port, INT4 |
|---|---|---|
| Decode, short context | 8.57 ± 0.06 tok/s (llama-bench tg64)<br>6.9–7.6 tok/s through llama-server | 13.8 plain greedy, 27.6 speculative<br>today 14.5–34.6 speculative on the five short prompts |
| Decode at ~1K | 8.13 ± 0.05 tok/s (tg64 @ d1024) | — |
| Decode at ~8K | 5.96 ± 0.19 tok/s (tg64 @ d8192) | — |
| Decode at 19K | 5.34 tok/s (187 ms/token, llama-server) | 73.4 ms/step plain (13.6 tok/s)<br>today 13.5 tok/s speculative |
| Prefill, short | pp512 225.5 ± 5.6, pp2048 225.0 ± 3.4 | — |
| Prefill after 1K / 8K | pp2048 217.4 ± 3.3 / 120.4 ± 8.9 | — |
| Prefill, 19K prompt | 117 tok/s (162 s for 19,027 tokens, llama-server) | 374 tok/s<br>today a 50.8 s TTFT for 19,045 tokens, about 375 tok/s |
| Memory | 31,314 MiB (32.8 GB) allocated at `-c 32768`; details below | ~25 GB short, ~29 GB at 19K<br>today the server peaked at 34.4 GB, including a 7.5 GB session cache |

- **Memory, llama.cpp:**
  - Metal: 29,814 MiB in total (model 25,360, KV and recurrent state 2,760, compute 1,693).
  - Host: 1,500 MiB (the token embedding, 1,288, which llama.cpp keeps on the CPU, plus 212 of compute).
  - The KV cache is preallocated for 32K positions.
  - Each context checkpoint takes 168 MiB more of host RAM.
  - `time -l` reports a max RSS of 21.5–25.1 GB, because the weights are mmap'd.
- **Short versus deep decode:** decode falls from 8.6 to 6.0 tok/s by 8K. Below 12,288 cached positions, the 17 BCSA
  layers and the one dense layer attend densely; `VOLUNDR_BCSA_SPARSE` switches to the sparse path above that.
- **Thread and ubatch sweep** (`bench2.txt`): 6 or 12 threads, with ubatch 512 or 2048.
  - Prefill does not move: pp2048 stays at 205–222 tok/s in all four settings.
  - Decode in that run was noisy (± up to 2.5 tok/s, and below the clean run in `bench1.txt`), so its tg figures are
    not used.
- **Server versus llama-bench:** llama-server decoded about 15% slower than llama-bench tg64 on short prompts. On
  bb1c8bc the gap is gone, so it was tied to the CPU fallbacks, not to sampling.

## 67e4001 CPU fallbacks

The README named two ops that had only CPU and CUDA kernels, `NGRAM_HASH` and `SINKHORN`. The `DSV4_HC_PRE` and
`DSV4_HC_POST` hyper-connection ops, which come from upstream's DeepSeek-V4 support, had no Metal kernel either.
bb1c8bc adds all of them (see above).

`GGML_SCHED_DEBUG=2` with `-lv 5` printed the op placement. It was the same for prefill (bs 512) and decode (bs 1):
3,339 compute nodes and 298 splits (149 CPU, 149 Metal). These ran on the CPU:

| op | count per forward | what it is |
|---|---|---|
| `SINKHORN` (hc_comb) | 72 | mHC residual mixing matrix, every layer |
| `DSV4_HC_PRE` (layer_inp) | 72 | mHC stream mix in, every layer |
| `DSV4_HC_POST` (l_out) | 72 | mHC stream mix out, every layer |
| `NGRAM_HASH` | 4 | Engram 2- and 3-gram hashes on the two Engram layers |
| `GET_ROWS` (inp_embd) | 1 | token embedding; llama.cpp keeps it on the host |

Everything else ran on Metal: the KDA (`GATED_DELTA_NET`, `SSM_CONV`), BCSA, top-k and the matmuls.

- **Decode:** the tensors crossing each split were tiny (~20 KB, the 4×5120 stream state).
  - Decode took 117 ms/token (tg64).
  - About 22 GB of weights are read per token: the 25.4 GB Metal buffer minus the 4.3 GB of Engram tables, which are
    only looked up. At 307 GB/s, that sets a floor of ~73 ms.
  - We put the ~45 ms above that floor down to the ~300 syncs per token. bb1c8bc shows the syncs cost only ~15 ms of
    it.
- **Prefill:** each crossing carried the hidden streams of the whole ubatch.

## Output check

Six prompts went through llama-server and `./serve.sh` with the same client (`llamacpp/chat.py`): greedy, thinking
off (`chat_template_kwargs.enable_thinking=false`), `prompts.json`. Prompt token counts match on all six (49, 41,
52, 36, 63, 19,045), so both chat templates render the same tokens. Both commits give the same results here.

- **Against the MLX port:** greedy replies diverge on all six prompts, between token 0 (`swedish`) and token 23
  (`arith`). At each divergence, MLX's token is in llama.cpp's top 5:
  - rank 2 in five prompts, rank 5 in `zh`;
  - near-ties in `sky` (0.48/0.29), `swedish` (0.41/0.23) and `long19k` (0.59/0.29);
  - not near-ties in `fib` (0.91/0.10) and `arith` (0.81/0.13).
- **What the divergence means:** two different 4-bit quantizations of the model.
  - Q4_K_M keeps the KDA, mHC, Engram projections, embedding and head at Q8_0.
  - Ours is Blockway's INT4 with a bf16 head.
  - This is not a parity verdict, since neither is the bf16 reference. That is still PARITY.md section 3.
- **Metal against llama.cpp's own CPU backend** (`--device none -ngl 0`, top-5 logprobs, run on 67e4001), on the five
  short prompts:
  - Four are byte-identical.
  - `swedish` splits at token 6, a near-tie: Metal ' rä' 0.406 / ' r' 0.372, CPU ' r' 0.396 / ' rä' 0.388.
  - The chosen token's logprob differs by 0.003–0.016 on average, 0.15 at most.
  - So the Metal path computes the same model as the reference CPU path.
  - bb1c8bc gives the same figures.
- **An observation on our side:** with thinking off, the MLX reply to `swedish` opens with reasoning-style prose ("The
  user wants a Swedish translation…") instead of the translation. llama.cpp gives that first token ("The") 0.23
  against 0.41 for "Den".

## Setup notes

- **Build:**
  `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server llama-cli llama-bench`
  (Metal, BLAS via Accelerate, AppleClang 21). Add `test-backend-ops` to the targets for the op tests.
  - The llama-server target provisions its web UI. The first build ran `npm install`, which I stopped.
  - Later builds fetched the prebuilt UI tarball from the HF bucket `ggml-org/llama-ui` instead. The tags for builds
    `b52` and `b54` were missing, and `latest` worked.
  - `-DLLAMA_BUILD_UI=OFF` on a reconfigure did not skip it.
  - I built at background priority (`taskpolicy -b`) so a running GPU benchmark wasn't disturbed.
  - The update to bb1c8bc was an incremental rebuild (ggml-metal and test-backend-ops).
- **Model:** `~/personal-dev/models/volundr-gguf/Agens-Volundr-32B-v1-Q4_K_M.gguf`, 26.6 GB (24.77 GiB, 6.60 BPW).
  Its SHA256 matches the repo's `SHA256SUMS`. The first load takes ~15–25 s.
- **Running the scripts:** run them from `llamacpp/` as `../with_gpu_lock.sh ./run_server.sh llama|mlx OUT.json
  [--only ...] [--logprobs 5]` and `../with_gpu_lock.sh ./bench_llama.sh [llama-bench flags]`.
  - Then `compare.py A.json B.json`.
  - Both scripts use `~/personal-dev/llama.cpp-volundr/build/bin`. Set `LL=.../build-67e4001/bin` to rerun the old
    commit.
  - Repeated llama-bench flags append to the defaults rather than replace them, so `bench_llama.sh` only uses its
    default matrix when it gets no arguments.
- **Shutdown:** llama-server ignores SIGINT while it loads the model, so `run_server.sh` stops it with TERM.
- **Raw results, 67e4001:** `llama_short.json`, `llama_long.json`, `llama_lp.json` (with logprobs),
  `llama_cpu.json` (CPU backend), `mlx_all.json`, `bench1.txt`, `bench2.txt`, the server logs in `logs/`, and
  `logs/llama_diag.server.log.gz` (kept locally only).
- **Raw results, bb1c8bc:** `bench_bb1c8bc.txt`, `test_backend_ops_bb1c8bc.txt`, `llama_short_bb1c8bc.json`,
  `llama_long_bb1c8bc.json`, `llama_lp_bb1c8bc.json`, `logs/llama_*_bb1c8bc.server.log`, and
  `logs/llama_diag_bb1c8bc.server.log.gz` (kept locally only).
