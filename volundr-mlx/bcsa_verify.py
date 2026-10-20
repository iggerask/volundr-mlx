"""BCSA attention for a speculative verify: T <= 16 queries against the window and their top-k far blocks, without
copying the window out of the ring and without per-head serial passes.

The prefill kernel (kernels.bcsa_attend) is built for chunks of hundreds of queries: one threadgroup per (32-query
block, head) walks the whole window plus the block's far-block union. For an 8-token verify that is 24 threadgroups
each streaming ~4K keys serially (3/4 of every query tile padding), behind a gather + concatenate of the window out
of the ring (NOTES_BCSA_VERIFY.md has the profile). Here instead:

- the G query heads of a KV head and the T queries are packed into one row block (rows g * T + t, 48 for T = 8,
  G = 6), so each key is loaded once per KV head, not once per head;
- the keys are read where they are: the ring slots of the cache (any order: softmax is permutation invariant, each
  slot's position is known from the offset), the T new keys, and the union of the T queries' selected far blocks
  (kernels.far_union) gathered from the pooled K/V; each (row, key) is masked to what that query sees (window
  (pos - W, pos], causal among the new keys, its own selected far blocks);
- the key list is split over NTG threadgroups per KV head (flash-decoding split-K), each leaving fp32 partials
  (unnormalized O, running max, sum) that a second kernel merges.

Same arithmetic as the prefill kernel (NAX tiles, bf16 inputs, fp32 scores and accumulation, exp2 online softmax,
one joint softmax over window and far keys); only the order in which keys are accumulated differs.
"""
import mlx.core as mx

import kernels

F32 = mx.float32

_MAIN_SRC = r"""
    using AccumType = float;
    constexpr short kU = 16;
    constexpr int TD = BD / kU;
    constexpr short TK = BK / kU;
    constexpr int TDh = TD / WN;
    constexpr int BDh = BD / WN;
    constexpr int R = G * NQ;          // rows per KV head: (head in group, query)
    constexpr int RP = WM * kU;        // rows padded to whole 16-row fragments
    static_assert(TD % WN == 0 && TDh % 2 == 0 && TK == 2 && BK == 32 && RP >= R && RP - kU < R, "tile shape");
    constexpr float NEG_INF = -INFINITY;
    constexpr float FMIN = -3.4028234663852886e38f;

    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const int P = params[0], WIN = params[1], UMAX = params[2], NTG = params[3];
    const int c = tid.x;               // split index within the KV head
    const int kvh = tid.y;
    const int RING = kr_shape[1];      // rows per KV head in the ring buffer (W ring slots + decode's far slots)
    const int CAP = pk_shape[1];
    const int UW = UMAX / 32;

    const short row_group = simd_group_id / WN;
    const short d_group = simd_group_id % WN;
    const short tm = kU * row_group;
    const float scale2 = scale[0] * 1.44269504089f;

    using otile_t = NAXTile<AccumType, 1, TDh>;
    using stile_t = NAXTile<AccumType, 1, TK>;
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
    const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;

    otile_t Otile;
    Otile.clear();
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = FMIN;
        sum_score[i] = 0.0f;
    }
    threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];

    // this simdgroup's 16 rows x BDh dims of the packed queries
    const device T* Q = q + (size_t(kvh) * R + tm) * BD + d_group * BDh;
    const short lim_rows_q = R - tm;
    NAXTile<T, 1, 1> Qtiles[TDh];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
        if (lim_rows_q < kU) {
            Qtiles[id].load_rows(Q + id * kU, BD, lim_rows_q);
        } else {
            Qtiles[id].load(Q + id * kU, BD);
        }
    }
    // query index of this thread's two rows (rows tm + sm and tm + sm + 8; padding rows mask like the last query)
    int tq[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const int r = tm + sm + 8 * i;
        tq[i] = r < R ? r % NQ : NQ - 1;
    }

    // key batches of 32: ring slots [0, nring), then the far union [0, ucount), then the NQ new keys
    const int nring = min(P, WIN);
    const int pm = P % WIN;            // slot of position P - WIN, the oldest the ring holds; slot s holds age d(s)
    const int NW = (nring + BK - 1) / BK;
    const int ucount = ucnt[0];
    const int NF = (ucount + BK - 1) / BK;
    const int total = NW + NF + 1;
    const int per = (total + NTG - 1) / NTG;
    const int b0 = c * per, b1 = min(total, b0 + per);
    const int age_min = max(NQ, WIN - P);   // a slot is visible to every query iff its age d >= age_min

    const device T* KR = kr + size_t(kvh) * RING * BD + d_group * BDh + sn;
    const device T* VR = vr + size_t(kvh) * RING * BD + d_group * BDh + sn;
    const device T* PK = pk + size_t(kvh) * CAP * BD + d_group * BDh + sn;
    const device T* PV = pv + size_t(kvh) * CAP * BD + d_group * BDh + sn;
    const device T* KN = kn + size_t(kvh) * NQ * BD + d_group * BDh + sn;
    const device T* VN = vn + size_t(kvh) * NQ * BD + d_group * BDh + sn;

    for (int b = b0; b < b1; b++) {
        const int kind = b < NW ? 0 : (b < NW + NF ? 1 : 2);
        const int base = kind == 0 ? b * BK : (kind == 1 ? (b - NW) * BK : 0);
        const device T* KB = kind == 0 ? KR : (kind == 1 ? PK : KN);
        const device T* VB = kind == 0 ? VR : (kind == 1 ? PV : VN);
        // this thread's key rows: batch keys r * 16 + sm (+ 8)
        int rows[TK][2];
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < TK; r++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const int j = base + r * kU + sm + 8 * i;
                rows[r][i] = kind == 0 ? j : (kind == 1 ? uidx[j] : min(j, NQ - 1));
            }
        }
        stile_t Stile;
        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDh; id++) {
            NAXTile<T, 2, 1> Ktile;
            STEEL_PRAGMA_UNROLL
            for (short r = 0; r < 2; r++) {
                thread auto& fk = Ktile.frag_at(r, 0);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const device T* src = KB + size_t(rows[r][i]) * BD + id * kU;
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) fk[i * 4 + j] = src[j];
                }
            }
            stile_t::NAXFrag_t::mma(
                Stile.frag_at(0, 0), Stile.frag_at(0, 1), Qtiles[id].frag_at(0, 0), metal::false_type{},
                Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
        // sum the WN partial scores in a fixed order
        threadgroup AccumType* slot = s_xchg[row_group][d_group];
        const short xb = short(simd_lane_id) * (TK * kEPF);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) slot[xb + ik * kEPF + i] = s[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) s[i] = s_xchg[row_group][0][xb + ik * kEPF + i];
            STEEL_PRAGMA_UNROLL
            for (short pg = 1; pg < WN; pg++) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < kEPF; i++) s[i] += s_xchg[row_group][pg][xb + ik * kEPF + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scale, and mask each (row, key) to what its query sees
        bool inside = false;
        uint mb[2] = {0u, 0u};
        if (kind == 0) {
            const int d0 = base >= pm ? base - pm : base - pm + WIN;
            inside = base + BK <= nring && d0 + BK <= WIN && d0 >= age_min;
        } else if (kind == 1) {
            mb[0] = umask[tq[0] * UW + (base >> 5)];
            mb[1] = umask[tq[1] * UW + (base >> 5)];
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            thread auto& fg = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < 2; ii++) {
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < 4; jj++) {
                    const auto loc = ii * 4 + jj;
                    const int kj = ik * kU + sn + jj;       // key within the batch
                    bool ok;
                    if (kind == 0) {
                        const int s = base + kj;
                        const int d = s >= pm ? s - pm : s - pm + WIN;
                        ok = inside || (s < nring && d > tq[ii] && d >= WIN - P);
                    } else if (kind == 1) {
                        ok = (mb[ii] >> kj) & 1u;
                    } else {
                        ok = kj <= tq[ii];
                    }
                    fg[loc] = ok ? fg[loc] * scale2 : NEG_INF;
                }
            }
        }

        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<VMaxOp>(new_max);
        Stile.template row_bin_op<VExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<VSumOp>(sum_score);
        Otile.template row_bin_op<VMulOp>(factor);
        simdgroup_barrier(mem_flags::mem_none);

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDh; id += 2) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                NAXTile<T, 1, 2> Vtile;
                STEEL_PRAGMA_UNROLL
                for (short cc = 0; cc < 2; cc++) {
                    thread auto& fv = Vtile.frag_at(0, cc);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 2; i++) {
                        const device T* src = VB + size_t(rows[ik][i]) * BD + (id + cc) * kU;
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++) fv[i * 4 + j] = src[j];
                    }
                }
                otile_t::NAXFrag_t::mma(
                    Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
                    Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
            }
        }
    }

    // partials: unnormalized O, running max (log2 units) and sum per row
    const size_t pb = (size_t(kvh) * NTG + c) * RP;
    Otile.store(opart + (pb + tm) * BD + d_group * BDh, BD);
    if (d_group == 0 && sn == 0) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const size_t row = pb + tm + sm + 8 * i;
            ml[row * 2] = max_score[i];
            ml[row * 2 + 1] = sum_score[i];
        }
    }
"""

_MERGE_SRC = r"""
    const int d = thread_position_in_grid.x;
    const int r = thread_position_in_grid.y;
    const int kvh = thread_position_in_grid.z;
    const int NTG = params[0];
    const device float* m0 = ml + (size_t(kvh) * NTG * RP + r) * 2;
    const device float* o0 = opart + (size_t(kvh) * NTG * RP + r) * BD + d;
    float M = -INFINITY;
    for (int c = 0; c < NTG; c++) M = metal::max(M, m0[size_t(c) * RP * 2]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < NTG; c++) {
        const float w = fast::exp2(m0[size_t(c) * RP * 2] - M);
        L += m0[size_t(c) * RP * 2 + 1] * w;
        acc += o0[size_t(c) * RP * BD] * w;
    }
    const int h = kvh * G + r / NQ, t = r % NQ;
    out[(size_t(h) * NQ + t) * BD + d] = static_cast<T>(acc / L);
"""

_main_kernels, _merge_kernels = {}, {}
# splits per KV head: 4-8 are equally fast at 1-24K context (spec/bench_bcsa_attend.py); more splits only add partials
NTG = 8


def attend(q, kn, vn, kr, vr, P, window, scale, pk=None, pv=None, uidx=None, ucnt=None, umask=None, ntg=None):
    """BCSA attention of T verify queries (B = 1).
    q [H, T, D] (post-RoPE, positions P..P+T-1); kn, vn [KVH, T, D] their keys / values; kr, vr [KVH, rows >= W, D]
    the cache's ring (slot = position % W, positions < P); far blocks (optional): pk, pv [KVH, cap, D] pooled K/V and
    kernels.far_union(sel, nv_max, 32) of the queries' selections -> out [H, T, D]."""
    H, T, D = q.shape
    KVH = kr.shape[0]
    G = H // KVH
    assert 1 <= T <= 16 and H == G * KVH and D % 64 == 0
    wm = (G * T + 15) // 16
    wn = 2
    if pk is None:  # window only: an empty union
        pk, pv = kr, vr
        uidx = mx.zeros((1, 32), mx.int32)
        ucnt = mx.zeros((1,), mx.int32)
        umask = mx.zeros((1, 32, 1), mx.uint32)
    umax = uidx.shape[1]
    if ntg is None:
        ntg = NTG
    key = (T, G, D, wm, wn)
    if key not in _main_kernels:
        hdr = kernels._win_header()
        _main_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_bcsa_verify_{T}_{G}_{D}", header=hdr, source=_MAIN_SRC,
            input_names=["q", "kn", "vn", "kr", "vr", "pk", "pv", "uidx", "ucnt", "umask", "scale", "params"],
            output_names=["opart", "ml"])
        _merge_kernels[key] = mx.fast.metal_kernel(
            name=f"volundr_bcsa_verify_merge_{T}_{G}_{D}", source=_MERGE_SRC,
            input_names=["opart", "ml", "params"], output_names=["out"])
    rp = wm * 16
    nth = wm * wn * 32
    params = mx.array([P, window, umax, ntg], dtype=mx.int32)
    tmpl = [("T", q.dtype), ("NQ", T), ("G", G), ("BD", D), ("BK", 32), ("WM", wm), ("WN", wn)]
    opart, ml = _main_kernels[key](
        inputs=[q, kn, vn, kr, vr, pk, pv, uidx, ucnt, umask, mx.array([scale], dtype=F32), params],
        template=tmpl, grid=(ntg * nth, KVH, 1), threadgroup=(nth, 1, 1),
        output_shapes=[(KVH, ntg, rp, D), (KVH, ntg, rp, 2)], output_dtypes=[F32, F32])
    return _merge_kernels[key](
        inputs=[opart, ml, params[3:]], template=[("T", q.dtype), ("NQ", T), ("G", G), ("BD", D), ("RP", rp)],
        grid=(D, G * T, KVH), threadgroup=(D, 1, 1), output_shapes=[(H, T, D)], output_dtypes=[q.dtype])[0]
