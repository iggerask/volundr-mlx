# Blockway's llama.cpp fork on Metal (2026-10-20)

Status: done. Scripts, logs and raw results are in `llamacpp/`. The `GGML_SCHED_DEBUG=2` placement dump
(`logs/llama_diag.server.log.gz`, 6 MB) is not committed.

## Verdict

- **It works on Metal.** Blockway's fork (github.com/BlockWayz/llama.cpp, branch `volundr`, 67e4001, build 52) builds
  with the default Metal backend. It runs the Q4_K_M GGUF with `-ngl 99 --jinja`, and all 73 layers are offloaded.
- **The output is sane.** Replies are coherent in English, Swedish, Chinese, code and arithmetic. The 19K-token CUDA
  file is summarized correctly.
  - Greedy output is identical from run to run.
  - The CPU backend gives the same replies as Metal, except at one near-tie.
- **It is slower than our port everywhere:**
  - Decode is 1.6x slower than our plain greedy and 3.2x slower than our speculative loop.
  - At 19K context, prefill is 3.2x slower and decode 2.5x slower.
  - The fork has no speculative decoding.
- **Why it is slower:** every mHC op runs on the CPU, so each forward pass has 298 graph splits, or about 4 GPU↔CPU
  round trips per layer.
- **As a second route on a Mac,** it is usable for correctness cross-checks, not for speed.

## Numbers

M5 Pro 64 GB. Every run was under `with_gpu_lock.sh`, one GPU job at a time, with a canary of 21.7–27.7 TFLOP/s
before and after.

- llama.cpp settings: `-ngl 99`, flash attention auto, 6 threads, ubatch 512.
- llama-bench: 3 repetitions, ± is the standard deviation.
- Our numbers come from the README, except where marked "today" (the `./serve.sh` run on the same prompts).

| | llama.cpp Metal, Q4_K_M | our MLX port, INT4 |
|---|---|---|
| Decode, short context | 8.57 ± 0.06 tok/s (llama-bench tg64)<br>6.9–7.6 tok/s through llama-server | 13.8 plain greedy, 27.6 speculative<br>today 14.5–34.6 speculative on the five short prompts |
| Decode at ~1K | 8.13 ± 0.05 tok/s (tg64 @ d1024) | — |
| Decode at ~8K | 5.96 ± 0.19 tok/s (tg64 @ d8192) | — |
| Decode at 19K | 5.34 tok/s (187 ms/token, llama-server) | 73.4 ms/step plain (13.6 tok/s)<br>today 13.5 tok/s speculative |
| Prefill, short | pp512 225.5 ± 5.6, pp2048 225.0 ± 3.4 | — |
| Prefill after 1K / 8K | pp2048 217.4 ± 3.3 / 120.4 ± 8.9 | — |
| Prefill, 19K prompt | 117 tok/s (162 s for 19,027 tokens, llama-server) | 374 tok/s<br>today a 50.8 s TTFT for 19,045 tokens, about 375 tok/s |
| Memory | 31.3 GB allocated at `-c 32768`; details below | ~25 GB short, ~29 GB at 19K<br>today the server peaked at 34.4 GB, including a 7.5 GB session cache |

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
- **Server versus llama-bench:** llama-server decodes about 15% slower than llama-bench tg64 on short prompts. This
  was not investigated. It may be per-token sampling over the 248K vocabulary plus the unused-token mask.

## CPU fallbacks

The README names two ops that have only CPU and CUDA kernels, `NGRAM_HASH` and `SINKHORN`. The `DSV4_HC_PRE` and
`DSV4_HC_POST` hyper-connection ops, which come from upstream's DeepSeek-V4 support, have no Metal kernel either.

`GGML_SCHED_DEBUG=2` with `-lv 5` printed the op placement. It is the same for prefill (bs 512) and decode (bs 1):
3,339 compute nodes and 298 splits (149 CPU, 149 Metal). These run on the CPU:

| op | count per forward | what it is |
|---|---|---|
| `SINKHORN` (hc_comb) | 72 | mHC residual mixing matrix, every layer |
| `DSV4_HC_PRE` (layer_inp) | 72 | mHC stream mix in, every layer |
| `DSV4_HC_POST` (l_out) | 72 | mHC stream mix out, every layer |
| `NGRAM_HASH` | 4 | Engram 2- and 3-gram hashes on the two Engram layers |
| `GET_ROWS` (inp_embd) | 1 | token embedding; llama.cpp keeps it on the host |

Everything else runs on Metal: the KDA (`GATED_DELTA_NET`, `SSM_CONV`), BCSA, top-k and the matmuls.

- **Decode:** the tensors crossing each split are tiny (~20 KB, the 4×5120 stream state). The cost is most likely
  the ~300 syncs per token, not data movement. This is an estimate, not profiled:
  - Decode takes 117 ms/token (tg64).
  - About 22 GB of weights are read per token: the 25.4 GB Metal buffer minus the 4.3 GB of Engram tables, which are
    only looked up. At 307 GB/s, that sets a floor of ~73 ms.
  - That leaves ~45 ms per token of overhead.
- **Prefill:** each crossing carries the hidden streams of the whole ubatch.
- **Possible fixes in the fork:** the hc_pre/hc_post kernels would be trivial to write for Metal, and so would a
  4×4 Sinkhorn. They would cut the graph down to a handful of splits. We have not tried.

## Output check

Six prompts went through llama-server and `./serve.sh` with the same client (`llamacpp/chat.py`): greedy, thinking
off (`chat_template_kwargs.enable_thinking=false`), `prompts.json`. Prompt token counts match on all six (49, 41,
52, 36, 63, 19,045), so both chat templates render the same tokens.

- **Against the MLX port:** greedy replies diverge on all six prompts, between token 0 (`swedish`) and token 23
  (`arith`). At each divergence, MLX's token is in llama.cpp's top 5:
  - rank 2 in five prompts, rank 5 in `zh`;
  - near-ties in `sky` (0.48/0.29), `swedish` (0.41/0.23) and `long19k` (0.59/0.29);
  - not near-ties in `fib` (0.91/0.10) and `arith` (0.81/0.13).
- **What the divergence means:** two different 4-bit quantizations of the model.
  - Q4_K_M keeps the KDA, mHC, Engram projections, embedding and head at Q8_0.
  - Ours is Blockway's INT4 with a bf16 head.
  - This is not a parity verdict, since neither is the bf16 reference. That is still PARITY.md section 3.
- **Metal against llama.cpp's own CPU backend** (`--device none -ngl 0`, top-5 logprobs), on the five short prompts:
  - Four are byte-identical.
  - `swedish` splits at token 6, a near-tie: Metal ' rä' 0.406 / ' r' 0.372, CPU ' r' 0.396 / ' rä' 0.388.
  - The chosen token's logprob differs by 0.003–0.016 on average, 0.15 at most.
  - So the Metal path and its CPU fallbacks compute the same model as the reference CPU path.
- **An observation on our side:** with thinking off, the MLX reply to `swedish` opens with reasoning-style prose ("The
  user wants a Swedish translation…") instead of the translation. llama.cpp gives that first token ("The") 0.23
  against 0.41 for "Den".

## Setup notes

- **Build:**
  `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server llama-cli llama-bench`
  (Metal, BLAS via Accelerate, AppleClang 21).
  - The llama-server target provisions its web UI. The first build ran `npm install`, which I stopped.
  - The rebuild fetched the prebuilt UI tarball from the HF bucket `ggml-org/llama-ui` instead. `b52` was missing and
    `latest` worked.
  - `-DLLAMA_BUILD_UI=OFF` on a reconfigure did not skip it.
  - I built at background priority (`taskpolicy -b`) so a running GPU benchmark wasn't disturbed.
- **Model:** `~/personal-dev/models/volundr-gguf/Agens-Volundr-32B-v1-Q4_K_M.gguf`, 26.6 GB (24.77 GiB, 6.60 BPW).
  Its SHA256 matches the repo's `SHA256SUMS`. The first load takes ~15–25 s.
- **Running the scripts:** run them from `llamacpp/` as `../with_gpu_lock.sh ./run_server.sh llama|mlx OUT.json
  [--only ...] [--logprobs 5]` and `../with_gpu_lock.sh ./bench_llama.sh [llama-bench flags]`.
  - Then `compare.py A.json B.json`.
  - Repeated llama-bench flags append to the defaults rather than replace them, so `bench_llama.sh` only uses its
    default matrix when it gets no arguments.
- **Shutdown:** llama-server ignores SIGINT while it loads the model, so `run_server.sh` stops it with TERM.
- **Raw results:** `llama_short.json`, `llama_long.json`, `llama_lp.json` (with logprobs), `llama_cpu.json`,
  `mlx_all.json`, `bench1.txt`, and the server logs in `logs/` (the `GGML_SCHED_DEBUG=2` placement dump, `llama_diag.server.log.gz`, is kept locally only).
