"""Volundr-shaped decode/verify step with random weights, timed end to end on this Mac.

Same layer map, shapes, quantization and per-layer op sequence as kernels/m1/m1_model.cu
(72 layers: 54 KDA, 17 BCSA, 1 dense; mHC 4 streams; Engram at 2 and 22; 8-bit lm_head).
Numerics are meaningless; only the work and the data movement are real.

usage: volundr_synth.py <bits> <group> [variants...]
variant = <T>:<ctx>:<mode>[:nax|:nax8]   T = tokens per step (1 decode, 8 verify), ctx = short|long,
mode = naive (eager, Sinkhorn in plain ops) | tuned (Sinkhorn kernel + mx.compile pieces) | linears
nax  = for T = 2..16, every 4-bit g128 linear runs on the small-M neural-accelerator kernel
       (smallm/qmm_nax.py), its input prep fused into the producing op (smallm/prep_nax.py)
nax8 = nax, plus the 8-bit lm_head on the uint8 variant (smallm/qmm_nax_u8.py)
REPS=n runs the variants n times, interleaved, and also prints the best of each.
Timing waits for an idle GPU and holds the shared gpu.lock (smallm/common.py).
"""
import os, sys, time, math
SKIP = set(os.environ.get('SKIP', '').split(','))
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "smallm"))
import qmm_nax as Q
import prep_nax as P
from qmm_nax_u8 import qmm_nax_u8_prepped
from common import wait_gpu_idle

BF = mx.bfloat16
HID, INTER, VOCAB = 5120, 17408, 248320
NH, NKV, HD, ROT = 24, 4, 256, 64
KH, VH, KD = 16, 48, 128
CONV = 2 * KH * KD + VH * KD  # 10240
WIN, TOPK = 4096, 512
LONG_CTX = 219000
BCSA = {3, 8, 12, 17, 21, 26, 30, 31, 35, 40, 44, 49, 53, 58, 62, 63, 67}
DENSE = {71}
ENGRAM = {2, 22}

BITS, GROUP = (int(sys.argv[1]), int(sys.argv[2])) if __name__ == "__main__" else (4, 128)
VARIANTS = sys.argv[3:] or ["1:short:tuned"]


def ql(n, k, bits=None, group=None):
    bits, group = bits or BITS, group or GROUP
    w = mx.random.randint(0, 2**31 - 1, (n, k * bits // 32)).astype(mx.uint32)
    s = (mx.random.uniform(shape=(n, k // group)) * 1e-3).astype(BF)
    b = (-s * (2 ** (bits - 1))).astype(BF)
    mx.eval(w, s, b)
    return (w, s, b, bits, group)


def qmm(x, p):
    w, s, b, bits, group = p
    if isinstance(x, tuple):  # a prep_nax result: 2..16 tokens on the neural accelerators
        if (bits, group) in ((4, 128), (8, 64)):
            assert x[2].shape[1] * group == x[1].shape[1], "prep group size != weight group size"
            return (Q.qmm_nax_prepped if bits == 4 else qmm_nax_u8_prepped)(x, w, s, b)
        x = P.unprep(x)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits)


def rnd(*shape, dtype=BF, scale=0.02):
    a = (mx.random.normal(shape) * scale).astype(dtype)
    mx.eval(a)
    return a


_sk_src = """
    uint t = thread_position_in_grid.x;
    float m[16];
    float mx_ = -INFINITY;
    for (int i = 0; i < 16; i++) { m[i] = res[t * 16 + i]; mx_ = metal::max(mx_, m[i]); }
    for (int i = 0; i < 16; i++) m[i] = metal::exp(m[i] - mx_);
    for (int it = 0; it < ITERS; it++) {
        for (int r = 0; r < 4; r++) {
            float s = metal::max(m[r*4] + m[r*4+1] + m[r*4+2] + m[r*4+3], 1e-9f);
            for (int c = 0; c < 4; c++) m[r*4+c] /= s;
        }
        for (int c = 0; c < 4; c++) {
            float s = metal::max(m[c] + m[4+c] + m[8+c] + m[12+c], 1e-9f);
            for (int r = 0; r < 4; r++) m[r*4+c] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[t * 16 + i] = m[i];
"""
_sk = mx.fast.metal_kernel(name="sinkhorn4", input_names=["res"], output_names=["out"], source=_sk_src)


def sinkhorn_kernel(res):
    T = res.shape[0]
    return _sk(inputs=[res.astype(mx.float32)], template=[("ITERS", 10)], grid=(T, 1, 1),
               threadgroup=(1, 1, 1), output_shapes=[(T, 4, 4)], output_dtypes=[mx.float32])[0]


def sinkhorn_ops(res):
    m = res.astype(mx.float32).reshape(-1, 4, 4)
    m = mx.exp(m - m.max(axis=(-1, -2), keepdims=True))
    for _ in range(10):
        m = m / mx.maximum(m.sum(-1, keepdims=True), 1e-9)
        m = m / mx.maximum(m.sum(-2, keepdims=True), 1e-9)
    return m


# elementwise pieces, compiled in "tuned" mode
def _kda_gates(f, dt_bias, a_neg, b):
    g = mx.exp(a_neg * mx.logaddexp(f + dt_bias, 0.0))
    return g.astype(BF), mx.sigmoid(b)

def _mhc_pre(d, H, ps, pp, rs):
    pre = ps + d[:, :4]
    post = pp + d[:, 4:8]
    res = rs + d[:, 8:24]
    x_in = (pre[..., None] * H).sum(1)
    return x_in.astype(BF), post, res

def _mhc_post(M, H, post, y, x_in):
    Hn = (M.astype(BF) @ H) + post[..., None].astype(BF) * (y - x_in)[:, None, :]
    return Hn, Hn.mean(1)

def _swiglu(gu):
    g, u = mx.split(gu, 2, axis=-1)
    return nn_silu(g) * u

def nn_silu(x):
    return x * mx.sigmoid(x)

def _gated_out(o, z):
    return o * nn_silu(z)

def _sig_gate(o, gate):
    return o * mx.sigmoid(gate)

C = {}


def compiled(f):
    if f.__name__ not in C:
        C[f.__name__] = mx.compile(f)
    return C[f.__name__]


def build(nlayers=72):
    L = []
    for i in range(nlayers):
        lay = {"mhc_norm": rnd(HID), "mhc_dyn": rnd(24, HID), "ps": rnd(4, dtype=mx.float32),
               "pp": rnd(4, dtype=mx.float32), "rs": rnd(16, dtype=mx.float32),
               "n1": rnd(HID), "n2": rnd(HID),
               "gate_up": ql(2 * INTER, HID), "down": ql(HID, INTER)}
        if i in BCSA or i in DENSE:
            lay.update(q=ql(NH * 2 * HD, HID), k=ql(NKV * HD, HID), v=ql(NKV * HD, HID), o=ql(HID, NH * HD),
                       qn=rnd(HD), kn=rnd(HD))
            if i in BCSA:
                lay.update(iq=ql(4 * 128, HID), ik=ql(128, HID), idn=rnd(128))
        else:
            lay.update(qkv=ql(CONV, HID), z=ql(VH * KD, HID), b=rnd(VH, HID), fdown=ql(128, HID),
                       fup=rnd(VH * KD, 128), convw=rnd(CONV, 4, 1, dtype=BF), dt=rnd(VH * KD, dtype=mx.float32),
                       aneg=-mx.exp(rnd(VH, 1, dtype=mx.float32)), on=rnd(KD), out=ql(HID, VH * KD))
        if i in ENGRAM:
            lay.update(t2=rnd(65536, 512), t3=rnd(65536, 512), en=rnd(1024), ev=ql(HID, 1024),
                       eg=rnd(1, HID), egb=rnd(1))
        L.append(lay)
        if i % 12 == 11:
            mx.eval([v for v in lay.values() if isinstance(v, mx.array)])
    head = {"emb": rnd(4096, HID), "fn": rnd(HID), "lm": ql(VOCAB, HID, 8, 64)}
    return L, head


class Cache:
    def __init__(self, T, long):
        self.T, self.long = T, long
        self.win = WIN if (long or T > 64) else 1024
        extra = TOPK * T if long else 0
        self.nb = LONG_CTX // 4 if long else 0
        self.H = mx.zeros((T, 4, HID), dtype=BF)
        self.hmean = mx.zeros((T, HID), dtype=BF)
        self.st, self.conv, self.rk, self.rv, self.bk, self.pk, self.pv = {}, {}, {}, {}, {}, {}, {}
        for i in range(72):
            if i in BCSA:
                self.rk[i] = mx.zeros((1, NKV, self.win + extra, HD), dtype=BF)
                self.rv[i] = mx.zeros((1, NKV, self.win + extra, HD), dtype=BF)
                if long:
                    self.bk[i] = rnd(self.nb, 128)
                    self.pk[i] = rnd(self.nb, NKV, HD)
                    self.pv[i] = rnd(self.nb, NKV, HD)
            elif i in DENSE:
                n = LONG_CTX if long and 'dense' not in SKIP else max(1024, 2 * T)
                self.rk[i] = mx.zeros((1, NKV, n, HD), dtype=BF)
                self.rv[i] = mx.zeros((1, NKV, n, HD), dtype=BF)
            else:
                self.st[i] = mx.zeros((1, VH, KD, KD), dtype=mx.float32)
                self.conv[i] = mx.zeros((1, 3, CONV), dtype=BF)
        mx.eval(self.H, list(self.st.values()), list(self.conv.values()), list(self.rk.values()),
                list(self.rv.values()), list(self.bk.values()), list(self.pk.values()), list(self.pv.values()))
        self.pos = 0
        self.mask = self.dmask = None
        if T > 1:
            nk = self.win + extra
            self.mask = mx.random.uniform(shape=(1, 1, T, nk)) < 0.9
            nd = LONG_CTX if long and 'dense' not in SKIP else max(1024, 2 * T)
            self.dmask = mx.random.uniform(shape=(1, 1, T, nd)) < 0.99
            mx.eval(self.mask, self.dmask)


def step(L, head, c, tok, mode, nax=""):
    T = c.T
    tuned = mode == "tuned"
    cp = compiled if tuned else (lambda f: f)
    sink = sinkhorn_kernel if tuned else sinkhorn_ops
    # nax: 4-bit linears with 2..16 tokens take a prep_nax result (x padded to 16 rows + group
    # sums), written by the op that produces x; one result is shared by every linear reading it
    use = bool(nax) and 2 <= T <= P.MT
    use8 = use and nax == "nax8"
    pin = P.prep if use else (lambda x: x)
    norm_in = P.rms_prep if use else (lambda x, w: mx.fast.rms_norm(x, w, 1e-6))
    x = head["emb"][tok.reshape(-1) % 4096]                      # [T, HID]
    if mode == "linears":
        # every matmul output feeds the result, or MLX's lazy graph would skip it
        h, acc = x, mx.zeros((), dtype=mx.float32)
        for i, lay in enumerate(L):
            hp = pin(h)
            if i in BCSA or i in DENSE:
                qg = qmm(hp, lay["q"])
                acc = acc + qmm(hp, lay["k"]).sum() + qmm(hp, lay["v"]).sum()
                if i in BCSA:
                    acc = acc + qmm(hp, lay["iq"]).sum() + qmm(hp, lay["ik"]).sum()
                a = qmm(pin(qg[:, : NH * HD]), lay["o"])
            else:
                acc = acc + qmm(hp, lay["qkv"]).sum() + qmm(hp, lay["fdown"]).sum()
                a = qmm(pin(qmm(hp, lay["z"])), lay["out"])
            gu = qmm(pin(h + a), lay["gate_up"])
            h = h + qmm(pin(gu[:, :INTER]), lay["down"])
            if i in ENGRAM:
                acc = acc + qmm(pin(h[:, :1024]), lay["ev"]).sum()
        h = h + (acc * 0).astype(BF)
        h = h[-1:] if T > 64 else h
        return mx.argmax(qmm(P.prep(h, gs=64) if use8 else h, head["lm"]), axis=-1)
    H = mx.broadcast_to(x[:, None, :], (T, 4, HID)) if c.pos == 0 else c.H
    hmean = x if c.pos == 0 else c.hmean
    p = c.pos
    for i, lay in enumerate(L):
        d = (mx.fast.rms_norm(hmean, lay["mhc_norm"], 1e-6) @ lay["mhc_dyn"].T).astype(mx.float32)
        x_in, post, res = cp(_mhc_pre)(d, H, lay["ps"], lay["pp"], lay["rs"])
        M = sink(res)
        h = norm_in(x_in, lay["n1"])
        if i in BCSA or i in DENSE:
            qg = qmm(h, lay["q"]).reshape(1, T, NH, 2 * HD)
            q, gate = qg[..., :HD], qg[..., HD:]
            k = qmm(h, lay["k"]).reshape(1, T, NKV, HD)
            v = qmm(h, lay["v"]).reshape(1, T, NKV, HD)
            q = mx.fast.rms_norm(q, lay["qn"], 1e-6).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k, lay["kn"], 1e-6).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, ROT, traditional=False, base=1e7, scale=1.0, offset=p)
            k = mx.fast.rope(k, ROT, traditional=False, base=1e7, scale=1.0, offset=p)
            slot = p % c.rk[i].shape[2] if i in DENSE else p % c.win
            slot = min(slot, c.rk[i].shape[2] - T)
            c.rk[i][:, :, slot:slot + T, :] = k
            c.rv[i][:, :, slot:slot + T, :] = v.transpose(0, 2, 1, 3)
            mask = c.dmask if i in DENSE else c.mask
            if i in BCSA:
                iq = mx.fast.rms_norm(qmm(h, lay["iq"]).reshape(T, 4, 128), lay["idn"], 1e-6)
                ik = mx.fast.rms_norm(qmm(h, lay["ik"]), lay["idn"], 1e-6)
                if c.long and 'far' not in SKIP:
                    s = (iq @ c.bk[i].T).mean(axis=1)                         # [T, nb]
                    idx = mx.argpartition(-s, TOPK - 1, axis=-1)[:, :TOPK].reshape(-1)
                    fk = mx.take(c.pk[i], idx, axis=0).transpose(1, 0, 2)[None]  # [1,4,T*512,HD]
                    fv = mx.take(c.pv[i], idx, axis=0).transpose(1, 0, 2)[None]
                    c.rk[i][:, :, c.win:, :] = fk
                    c.rv[i][:, :, c.win:, :] = fv
            o = mx.fast.scaled_dot_product_attention(q, c.rk[i], c.rv[i], scale=1 / 16, mask=mask)
            if use:  # gate, transpose and prep in one kernel
                o = P.sgate_prep(o[0].transpose(1, 0, 2), gate[0])
            else:
                o = cp(_sig_gate)(o.transpose(0, 2, 1, 3), gate).reshape(T, NH * HD)
            a = qmm(o, lay["o"])
        else:
            qkv = qmm(h, lay["qkv"])[None]                                     # [1,T,CONV]
            z = qmm(h, lay["z"])
            b = (P.unprep(h) if use else h) @ lay["b"].T
            f = qmm(h, lay["fdown"]) @ lay["fup"].T
            xc = mx.concatenate([c.conv[i], qkv], axis=1)
            c.conv[i] = xc[:, -3:, :]
            xc = nn_silu(mx.conv1d(xc, lay["convw"], groups=CONV))               # [1,T,CONV]
            q = xc[..., : KH * KD].reshape(1, T, KH, KD)
            k = xc[..., KH * KD: 2 * KH * KD].reshape(1, T, KH, KD)
            v = xc[..., 2 * KH * KD:].reshape(1, T, VH, KD)
            q = mx.fast.rms_norm(q, None, 1e-6) * (1 / KD)
            k = mx.fast.rms_norm(k, None, 1e-6) * (1 / math.sqrt(KD))
            g, beta = cp(_kda_gates)(f.reshape(1, T, VH, KD).astype(mx.float32),
                                     lay["dt"].reshape(VH, KD), lay["aneg"], b.reshape(1, T, VH))
            o, st = gated_delta_kernel(q, k, v, g, beta.astype(BF), c.st[i])
            if T == 1 or T > 16:  # decode and prefill commit; verify replays
                c.st[i] = st
            else:  # verify does not commit; replay 3 accepted tokens from the old state
                _, c.st[i] = gated_delta_kernel(q[:, :3], k[:, :3], v[:, :3], g[:, :3],
                                                beta[:, :3].astype(BF), c.st[i])
            if use:  # head rms_norm, z gate and prep in one kernel
                o = P.kda_out_prep(o[0], lay["on"], z)
            else:
                o = mx.fast.rms_norm(o, lay["on"], 1e-6)
                o = cp(_gated_out)(o, z.reshape(1, T, VH, KD)).reshape(T, VH * KD)
            a = qmm(o, lay["out"])
        y1 = x_in + a
        gu = qmm(norm_in(y1, lay["n2"]), lay["gate_up"])
        y = y1 + qmm(P.swiglu_prep(gu) if use else cp(_swiglu)(gu), lay["down"])
        if i in ENGRAM:
            r = mx.random.randint(0, 65536, (2, T))
            e = mx.concatenate([lay["t2"][r[0]], lay["t3"][r[1]]], axis=-1)
            val = qmm(norm_in(e, lay["en"]), lay["ev"])
            y = y + mx.sigmoid(y @ lay["eg"].T + lay["egb"]) * val
        H, hmean = cp(_mhc_post)(M, H, post, y, x_in)
    c.H, c.hmean = H, hmean
    c.pos += T if T <= 64 else 0
    hf = H[-1:, 0, :] if T > 64 else H[:, 0, :]
    hf = P.rms_prep(hf, head["fn"], gs=64) if use8 else mx.fast.rms_norm(hf, head["fn"], 1e-6)
    return mx.argmax(qmm(hf, head["lm"]), axis=-1)


def run(L, head, T, ctx, mode, nax="", nsteps=None):
    nsteps = nsteps or (24 if T <= 64 else 4)
    c = Cache(T, ctx == "long")
    c.pos = (LONG_CTX - 64) if c.long else (0 if T > 64 else 512)
    tok0 = mx.zeros((T,), dtype=mx.uint32)
    inp = (lambda y: tok0) if T > 64 else (lambda y: y)
    tok = tok0
    for _ in range(3):  # warm up and compile
        tok = step(L, head, c, inp(tok), mode, nax)
        mx.eval(tok)
    wait_gpu_idle()
    mx.synchronize()
    build = 0.0
    t0 = time.perf_counter()
    b0 = time.perf_counter()
    y = step(L, head, c, inp(tok), mode, nax)
    build += time.perf_counter() - b0
    mx.async_eval(y)
    for _ in range(nsteps - 1):
        b0 = time.perf_counter()
        y2 = step(L, head, c, inp(y), mode, nax)
        build += time.perf_counter() - b0
        mx.async_eval(y2)
        mx.eval(y)
        y = y2
    mx.eval(y)
    dt = (time.perf_counter() - t0) / nsteps
    print(f"T={T} ctx={ctx:5s} mode={mode:7s} {nax or 'stock':5s}: {dt*1e3:6.1f} ms/step  "
          f"({T/dt:5.1f} positions/s)  graph build {build/nsteps*1e3:5.1f} ms/step", flush=True)
    del c
    return dt


def canary():
    a = mx.random.normal((8192, 8192)).astype(BF)
    mx.eval(a, a @ a)
    mx.synchronize()
    t = time.perf_counter()
    mx.eval([a @ a for _ in range(8)])
    mx.synchronize()
    print(f"canary {8 * 2 * 8192**3 / (time.perf_counter() - t) / 1e12:.1f} TFLOP/s", flush=True)


if __name__ == "__main__":
    wait_gpu_idle()  # held until exit, so building the weights does not disturb other benchmarks
    t = time.perf_counter()
    L, head = build()
    wbytes = sum(v[0].nbytes + v[1].nbytes + v[2].nbytes for lay in L for v in lay.values() if isinstance(v, tuple))
    print(f"w{BITS}g{GROUP}: quantized linears {wbytes/1e9:.2f} GB + lm_head "
          f"{sum(a.nbytes for a in head['lm'][:3])/1e9:.2f} GB, built in {time.perf_counter()-t:.0f} s", flush=True)
    canary()
    best = {}
    for _ in range(int(os.environ.get("REPS", "1"))):
        for vspec in VARIANTS:
            T, ctx, mode, *nax = vspec.split(":")
            dt = run(L, head, int(T), ctx, mode, *nax)
            best[vspec] = min(best.get(vspec, 1e9), dt)
    canary()
    if int(os.environ.get("REPS", "1")) > 1:
        print("best of reps:")
        for vspec, dt in best.items():
            print(f"  {vspec:24s} {dt*1e3:6.1f} ms/step")
    print(f"peak memory {mx.get_peak_memory()/1e9:.1f} GB")
