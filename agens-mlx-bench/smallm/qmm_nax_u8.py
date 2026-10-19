"""qmm_nax for MLX affine 8-bit, group-64 weights (the Volundr lm_head), M = 1..16 tokens.

Same kernel as qmm_nax.py with MPP's plain uint8_t right operand (bfloat x uint8_t -> float) and a
64-wide reduction per tensor op, one per quantization group. x comes from a gs=64 prep
(prep_nax.*_prep(..., gs=64)), since the epilogue needs per-64-group sums.
"""
import mlx.core as mx
import qmm_nax as Q

MT, NT, KS, GS = 16, 64, 8, 64
_kernel = None


def qmm_nax_u8_prepped(px, w, scales, biases):
    global _kernel
    if _kernel is None:
        src = ("#define XINT8 0\n"
               + Q.SOURCE.replace("XT", "bfloat").replace("WF", "uint8_t").replace("DT", "float"))
        _kernel = mx.fast.metal_kernel(
            name="qmm_nax_u8", input_names=["x", "w", "scales", "biases", "xsum", "xscale"],
            output_names=["y"], source=src, header=Q.HEADER)
    M, xin, xsum, xscale = px
    K = xin.shape[1]
    N = w.shape[0]
    assert N % NT == 0 and K % GS == 0 and w.shape[1] * 4 == K and xsum.shape[1] == K // GS
    return _kernel(
        inputs=[xin, w, scales, biases, xsum, xscale],
        template=[("MT", MT), ("NT", NT), ("KS", KS), ("GS", GS), ("K", K), ("N", N), ("MROWS", M)],
        grid=(N // NT * KS * 32, 1, 1), threadgroup=(KS * 32, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
