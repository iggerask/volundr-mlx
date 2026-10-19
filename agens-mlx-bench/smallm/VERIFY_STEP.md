# NAX small-M kernel in the full Volundr verify step (2026-10-18)

Follow-up to NOTES.md "Next steps" 1 and 2. Run with `volundr_synth.py 4 128 <T>:short:<mode>[:nax|:nax8]`.

**Result: the 8-token verify step costs 1.25× a decode step, not 1.1×.** It is 83.3 ms against
144.5 ms on stock MLX, a 1.73× speedup. At 16 tokens it is 1.33× (89.3 ms against 203.4 ms).

## What changed

- `volundr_synth.py`: the `:nax` suffix sends every 4-bit g128 linear with T=2..16 through
  `qmm_nax_prepped`; `:nax8` also sends the 8-bit lm_head. T=1 decode stays on
  `mx.quantized_matmul`. One prep result is shared by all linears that read the same input
  (q/k/v/iq/ik, qkv/z/fdown, gate_up).
- `prep_nax.py`: each producer writes x padded to 16 rows plus its group sums in one
  `metal_kernel`. There are variants for RMSNorm, swiglu (feeds down), sigmoid gate + transpose
  (feeds o), per-head RMSNorm × silu(z) (feeds KDA out), and a plain strided copy (linears mode).
  Plain and RMSNorm are bit-exact with stock. The others differ by about 1 bf16 ulp, because they
  round once instead of at each step.
- `qmm_nax_u8.py`: the same kernel with MPP's `bfloat × uint8_t → float` operand, one 64-wide
  tensor op per group. Its error equals stock's (`check_prep.py`). It saves about 1.5–2 ms on the
  lm_head at T=8.
- Checks:
  - `check_prep.py`: each prep variant against stock ops + `prep_x`, and the u8 kernel.
  - `check_synth.py`: every linear's output in a 4-layer step, nax against stock. The relative RMS
    difference is ≤ 8e-3 and the argmax is identical.

## Numbers

Short context, ms/step, best of 3 interleaved reps (`synth_nax.log`). The GPU lock was held
throughout. The canary read 14.1 TFLOP/s before the run and 23.3 after.

| T | path | matmul-only (`linears`) | full step (`tuned`) | full step / decode |
|---|---|---|---|---|
| 1 | stock (decode) | 58.7 | 66.9 | 1.00 |
| 8 | stock | 132.2 | 144.5 | 2.16 |
| 8 | nax | 71.2 | 85.1 | 1.27 |
| 8 | nax8 | 69.7 | 83.3 | **1.25** |
| 16 | stock | 189.6 | 203.4 | 3.04 |
| 16 | nax | 72.5 | 90.8 | 1.36 |
| 16 | nax8 | 70.7 | 89.3 | **1.33** |

The `linears` mode includes about 4–5 ms of non-matmul ops (output sums, residual adds). Net of
those, decode matmuls take 54.5 ms, which matches the old 54.7 ms figure.

## Where the 1.25× comes from

`ablate_synth.py` (`ablate_synth.log`) re-ran the step with every quantized matmul replaced by one
trivial op on its input ("nomm"). That keeps the op count and the dependencies but drops the
weight traffic. A second run reproduced the table within 2% at T=8 and 5% at T=16.

T=8 nax8, 82.8 ms in that run:

| Part | ms | vs decode |
|---|---|---|
| Matmul weight streaming | 64.6 | decode 54.4 ms, 1.19× |
| Non-matmul ops (decode share) | 11.8 | same as decode |
| T=8 attention, KDA and 3-token replay | +4.7 | stock pays this too |
| Prep kernels, net of the ops they replace | +1.7 | |

- The matmul 1.19× is the MPP 4-bit load path. Over the 15.9 GB of weights, stock decode streams
  at about 293 GB/s in the real 72-layer chain. That is faster than the 242–275 GB/s the isolated
  bench measured, because back-to-back kernels overlap. The NAX kernel stays at about 247 GB/s,
  its known device-operand ceiling (NOTES.md "Limits"). The kernel-only 1.07× compared it with a
  slower isolated decode.
- The small-N linears (k, v, iq, ik, fdown) are not the problem. Moving them back to stock made the
  step 0.4–3.6 ms slower.
- The prep kernels cost about 2 ms. Further prep fusion is worth little.
- Graph build is 4.7 ms/step for nax against 2.5 for stock. That is CPU time, hidden by
  `async_eval`.

## Revised projection

The INT4 31/41/54 tok/s (chat/code/agent) assumed a verify step of about 1.1× decode. Scaling by the
measured 1.1/1.25 gives **about 27/36/48 tok/s** at T=8. This assumes cycle time scales with verify
cost and ignores drafter cost.

A 16-token verify (1.33×) costs only 7% more than 8 tokens. It pays off if a deeper draft or tree
raises the acceptance length by more than about 7%.

## What would close the gap to 1.1×

- The kernel's streaming rate: about 11 ms of the 16 ms excess at T=8. NOTES.md "Limits" lists
  what has already been tried.
- Verify-only attention/KDA work: about 5 ms.
- The int8-activation variant (`nax-i8`) measured 0.03–0.1× faster in isolation. Its error is
  about 3× higher, and it is not wired in here.
