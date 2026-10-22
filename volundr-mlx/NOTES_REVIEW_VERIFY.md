# Volundr MLX: performance review, verify step first (2026-10-22)

Scope: the single-stream speculative loop (spec/specdec.py, spec/verify.py, nax_hook.py, smallm/qmm_nax.py,
dflash.py) and what the M5 Pro / MLX / Metal 4 / spec-decode literature says about each lever. Numbers below come
from the repo's own notes (NOTES_SPEC_ACCEPT, VERIFY_STEP, NOTES_decode, smallm/NOTES) unless marked.

## Where the step is today (short context, early@cv, bf16 head)

| part | ms | notes |
|---|---|---|
| verify matmul weight streaming | 64.6 | 15.9 GB at ~247 GB/s; decode streams the same bytes at ~293 |
| lm_head, 8 rows, bf16 | ~10 | 2.54 GB; q8 opt-in 7.2 ms (stock qmm at ~190 GB/s, not NAX) |
| non-matmul ops (norms, mHC, Engram, conv, recurrence) | ~12 | verify.py has its own _kda/_layer: decode_fused's KDA fusion does not reach it |
| T=8 attention/KDA extra + prep | ~6 | |
| commit (KDA replay) + materialize, GPU, serial | ~3.1 | before the verify can start |
| drafter (on ~90% of steps) | ~9 | 5-layer block 5.3 ms is ~2x its 0.8 GB streaming floor |
| GPU idle after the host sync | 1.5-2.5 | |
| **step** | **~93** | decode 66-72 |

Ideal with the current acceptance (2.6 tokens/step) and a verify at exactly decode cost would be ~2.6x; the loop
is at ~2.0x. So about a third of the remaining headroom is verify cost, the rest is acceptance.

## 1. Verify matmul kernel: 247 -> ~290 GB/s (worth ~8-11 ms, 9-12%)

What the research adds to smallm/NOTES "Limits":
- Apple's WWDC26 session 330 (Metal tensors/TensorOps) says explicitly: if you dequantize yourself, dequantize into a
  **cooperative tensor** (registers), not into threadgroup memory, and feed TensorOps from there. The device-operand
  `uint4b_format` path the kernel uses is the one whose load path tops out at 235-250 GB/s; plain 16-byte loads with
  the same tiling reach 283 GB/s (your own probe).
- Your own 2/3-bit prototype already does exactly this (lane unpack -> int8 cooperative tensor) and, with int8
  activations, sits at 1.05-1.07x of a same-width decode, i.e. at the streaming floor. For 4 bits the unpack is one
  shift and one mask per nibble, cheaper than the 2/3-bit case. The bf16-activation variant was compute-bound at
  2 bits (bf16 x int8 runs at ~23 TFLOP/s on the accelerator; 16-token full-model work is ~38 ms), but 4-bit
  streaming takes ~2x longer than 2-bit, so that compute fits under it with margin. Expected: the 1.19x matmul
  overhead drops towards 1.05-1.1x. **This is the single most promising untried kernel variant**: 4-bit weights
  via plain wide loads -> unpack to int8 cooperative tensor -> bf16 x int8 (or int8 x int8 with the i8 activation
  prep you already have, 3x the error, at 29 TFLOP/s).
- Third-party M5 microbenchmarks (scratchy issue #118) found "throughput through device memory depends heavily on
  tile shape" and that a (32,2,2) threadgroup raised MLX's own in-tree NAX int4 kernel from 12.7 to 14.3 TFLOP/s.
  Two knobs variants.py never had: (a) split-K across **threadgroups** (not only across the 8 simdgroups of one
  threadgroup) for the N<=6144 shapes, which are the slow ones (down 220 GB/s, out 230; gate_up with 544
  threadgroups is at 238). down/out/kda_qkv launch only 80-160 threadgroups on 20 cores, ~1 K threads per core,
  little to hide DRAM latency with. fp32 partials + a tiny reduce kernel, as bcsa_verify already does. (b) MT=32
  tiles: only 8/16 were probed. If a 32-row tile streams at the same rate (scratchy: 32-row W4A8 tiles still ran at
  17 TOPS), batched speculation can run B=4 at T=8 instead of T=4, and B up to 8.
- MLX 0.32.x has `qmv_wide` (PR #3764), a small-M (2-8) quantized matvec gated to gen-15+ GPUs, written for
  speculative verify. Your 0.32.4 stock bench still shows 2.5-2.9x at M=8, so either it is not being dispatched
  for 4-bit affine g128 on your shapes or it is slower than NAX. Worth one Metal capture to see which kernel name
  runs; it matters for the batched-decode rows (B=2-3) where stock beat the 16-row NAX tile.
- There is no MLX release after 0.32.4; the pin is current.

## 2. lm_head: a tiered head (worth ~7 ms/step spec, ~7 ms/token plain decode; 7-10%)

Greedy verify needs only each row's argmax. The bf16 head reads 2.54 GB per step (10 ms at 8 rows). Proposal:
1. Run the head on the **existing 4-bit head copy** (drafter.head, 0.66 GB, ~2.7 ms on the NAX kernel at 8 rows;
   for 1-row decode, stock qmv ~2.3 ms) and take the top-K (K=32..64) per row.
2. Gather those K rows of the bf16 head (K x 5120 x 2 B, a few hundred KB) and rescore exactly in bf16; argmax
   over the rescored logits.
Output equals the bf16 head's argmax whenever the true argmax is in the shortlist. That is not provable with a
usable bound (the worst-case dequant bound sum_g s_g/2 * |x_g|_1 is ~sqrt(K)=72x looser than the actual error),
but it is **measurable**: count positions on parity/prompts.json where the bf16 top-1 is outside the 4-bit top-K.
With the 4-bit head's error at ~0.3-0.5 logits RMS and K=64, expect zero misses. Compare with today's q8 opt-in,
which uses the q8 argmax directly (top-1 flips 1/240, 44/56 identical outputs) and still pays 1.35 GB: the tiered
head is both cheaper (0.66 GB) and closer to the bf16 model. The same trick serves top-k sampling (take the top-k
from the rescored shortlist). agens-engine does a related thing on the draft side (32K shortlist for its MTP chain).
Side fix regardless: nax_hook only converts 4-bit g128 linears, so with VOLUNDR_LMHEAD8 the verify head runs the
stock 8-row qmm at ~190 GB/s; qmm_nax_u8 already exists and saves ~2 ms there.

## 3. Shorten the serial GPU chain around the verify (worth ~3-4 ms, 3-4%)

- **Fold the KDA commit replay into the next verify's recurrence.** commit(n) re-reads and re-writes 54 x 6.3 MB
  of fp32 state (0.68 GB, ~2.3 ms) on the critical path. batch_spec.commit already uses neutral rows (decay 1,
  beta 0) to replay n rows in one call; take that further: the next verify's gated_delta_vec takes state0 and
  [n replay rows padded to T with neutral rows] + [T new rows] as one 2T-step scan. One state read+write instead
  of two, bit-identical (the neutral rows are exact no-ops), and shapes stay n-independent.
- **Materialize before the sync.** The drafter's context K/V depends only on captured features (layers 9..68) and
  can be computed for all T rows right after layer 68, overlapping layers 69-72 + head. Ring slots of rejected rows
  are never read (the next block stores its own K/V at P..P+7 before reading the window), so no masking is needed.
- Port decode_fused's KDA fusion (kda_conv_gates, bf16-writing recurrence) into verify._kda: it saved ~2 ms/step in
  decode and verify still runs the unfused op sequence. With shape-fixed commit (above) the compiled KDA layer can
  absorb it.
- Longer term the host sync itself (1.5-2.5 ms idle) can go: commit and materialize become n-agnostic with the two
  items above, and mx.fast.rope accepts an array offset, so only BCSA block-closing and the dense-cache offset keep
  Python ints. Big refactor for ~2%; last on the list.

## 4. Drafter (9 ms/step, ~10%): 2x above its floor

The 5-layer block streams 0.8 GB (2.7 ms) but takes 5.3 ms: ~300 small kernels, latency-bound. It cannot overlap
the verify (it needs layer 68's features and the bonus token), so the only lever is fewer, fatter kernels.
mx.compile of the whole hidden() is blocked today because cache.window(l, lo, P+8) slices by P (new graph per
step). Make it P-independent: attend over the full 4096-slot ring with a mask array computed outside (8 queries x
4096 keys x 5 layers is trivial), then compile the whole block: elementwise fusion alone should recover 1-2 ms.
PEARL-style draft-ahead does not apply: DFlash2 conditions on verified target features (confirmed in the DFlash
paper and the engine README).

## 5. Acceptance: the tree you already measured, implemented differently (worth +5-10% tokens/step net)

NOTES_SPEC_ACCEPT rejected a second chain because B=2 batched verify costs +9-13 ms (two BCSA passes, broadcast KDA
state, second dense pass). But the accept_log upper bounds were computed for *chains*; the literature (DDTree,
arXiv 2604.12989, built exactly for block-diffusion drafters; ddtree-mlx on a hybrid GatedDeltaNet 27B on M3 Ultra:
+10% over DFlash with a 4-node budget) uses a **tree in one forward**: 8 chain rows + up to 8 alternate rows in the
same <=16-row verify, which is free on the matmul side (T=16 vs 8 costs +6 ms total today). Your own data says the
walk's second choice at a slot is the target's token 20% of the time, and "alt at the first rejected slot" is worth
+18.7% tokens/step (with continuation) / "alt at slot 0" +6.7%. A tree with one alternate per slot (7 nodes) plus
short continuations for slots 0-1 realises most of the oracle number.
Cost model that makes it cheap on this model:
- KDA: a branch row at slot j is one extra rank-1 read-out from the chain's state S_j inside the same scan (state is
  already in registers); no broadcast state, no second pass. One kernel change: rows carry a "branch from slot j"
  index.
- BCSA 8q and dense: a per-row visible-prefix length instead of the causal rule; the 8q kernel already masks per
  query (ring age test), so this is a small mask change. 16 rows = 96 packed rows per KV head, still 6 NAX fragments.
- lm_head/argmax over 16 rows: covered by section 2.
Break-even: +6-7 ms/step (7%) against +10-18% tokens/step. Needs the drafter's per-slot top-2 (you have top-16 and
walk scores already). Budget selection per CaDDTree: pick nodes by walk-score prefix probability, 4-8 nodes.
Block verification (Sun et al.) is irrelevant for greedy; draft-length stopping (AdaEDL/SpecDec++) saves little
when verify costs only 1.25x.

## 6. MTP head: the engine's other +0.4-0.5 tokens/step

convert.py drops the release's MTP head. agens-engine chains it after a context copy: up to 4 slots with the
drafter, 7 without (0.22 GB, 4-bit), scoring a 32K shortlist of the head per chained slot. On this port each MTP
step is a tiny serial pass (one layer + a shortlisted head), ~1-2 ms per slot on the GPU, so it only pays where the
drafter's slots 1-4 are weak (chat/QA/writing at 2.1-2.4 tokens/step). Worth converting the head and measuring its
acceptance with accept_log.py before wiring anything: that is a one-day experiment with the existing harness.

## 7. Not worth pursuing (with reasons)

- Batch-invariant verify kernels: cost (no split-K, fixed reduction order) for a tie-only difference that the
  engine itself does not guarantee under TP. Keep as a correctness option, not a perf item.
- Fallback to plain decode on low acceptance: break-even 1.33 tokens/step, lowest prompt 1.40.
- 2/3-bit target weights: a model-quality decision; and at those widths the accelerator compute (not bandwidth)
  becomes the verify bottleneck per your own probes.
- Running anything on the CPU/ANE in parallel: the GPU already uses ~95% of the 307 GB/s in decode; there is no
  spare bandwidth to give another unit.
- Metal 4 command queues / argument tables: MLX does not use them and no primary source gives dispatch-cost
  numbers; the verify is GPU-bound now that the build is compiled.

## Suggested order

1. Tiered lm_head (section 2): ~7%, self-contained, measurable on the parity dumps in an afternoon.
2. Cooperative-tensor 4-bit NAX variant (section 1): up to ~10%, one kernel, the 2/3-bit prototype is the template.
3. Threadgroup split-K for N<=6144 and the MT=32 probe (section 1): small, informs batched spec too.
4. Commit-in-scan + early materialize + KDA fusion in verify (section 3): ~3-4%, bit-identical.
5. Tree verify with KDA branch rows (section 5): the largest acceptance lever; measure with accept_log first.
6. Drafter compile with a P-independent attention (section 4): 1-2%.
7. MTP head experiment (section 6).

Sources consulted by the research agents: Apple newsroom M5 Pro/Max (307 GB/s, NAX per GPU core); WWDC26 s330,
WWDC25 s262; scratchy issue #118 (M5 NAX rates: bf16 16.5 TFLOP/s per 10 cores, int8xint4 25-29 TOPS, tile-shape
sensitivity); ml-explore/mlx releases 0.31-0.32.4 and PR #3764 (qmv_wide), issues #3584/#3789; mlx-lm PR #990
(MTP with GatedDeltaNet state rollback); z-lab/dflash, DFlash arXiv 2602.06036, DDTree 2604.12989, CaDDTree
2606.01813, PEARL 2408.11850, AdaEDL 2410.18351, Block Verification 2403.10444; BlockWayz/agens-engine README;
llama.cpp PR #16634 / issue #30287 (M5 tensor path: prefill up, decode flat). arXiv, HF and Apple ML blog pages
were unreachable from the sandbox, so those items rest on abstracts and search snippets.
