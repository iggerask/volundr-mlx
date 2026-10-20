"""Full-model gates for the session cache (session.py), on prompts built from this repo's files.

1. exact: prefill, snapshot, then the original state and a restored one each run the same continuation: a prefill
   chunk, plain decode steps, verify forwards with commits of random length, drafter proposals. Logits, drafts and
   every final cache array must be bit-identical, and the snapshot unchanged.
2. resume: a "turn 1" prompt A and a longer prompt B = A + more. Fresh: B prefilled from 0 (chunks of 2048). Resumed:
   A prefilled, snapshot at len(A) (not chunk-aligned), restore, the rest of B prefilled. Calibration: B prefilled
   in chunks of 4096 (PARITY.md's noise floor). Teacher-forced rows (the last prompt position, then 47 forced decode
   steps on fresh's greedy tokens): KL fresh vs resumed against fresh vs chunk-4096. Free-running: greedy tokens of
   resumed vs fresh (divergences must sit at a near-tie of fresh's top two logits), and speculative decoding from the
   restored state vs plain greedy from it (the spec gate as in spec/NOTES.md: identical up to ties).
3. engine: two-turn conversation through session.Engine (what worker.py runs): the second turn resumes from the
   first turn's reply snapshot and prefills only the new tokens; its tokens equal a cache-less engine's up to ties.

4. split: where gate 2's KL comes from (gate_split's docstring).

usage (from volundr-mlx, under ./with_gpu_lock.sh): python session_gate.py [--only exact,resume,engine,split]
      [--json out]
"""
import os
import sys

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spec"))
import session as SS  # noqa: E402
import specdec  # noqa: E402
import verify as VF  # noqa: E402


def chat(tok, text, question, think=True):
    return tok.apply_chat_template([{"role": "user", "content": f"{text}\n\n{question}"}], add_generation_prompt=True,
                                   enable_thinking=think)


def repo_text(names, limit_chars):
    s = ""
    for n in names:
        s += f"\n\n### File {n}\n\n```\n{(HERE / n).read_text()}\n```"
    return s[:limit_chars]


def kl_rows(p, q):
    p, q = p.astype(mx.float32), q.astype(mx.float32)
    lp = p - mx.logsumexp(p, axis=-1, keepdims=True)
    lq = q - mx.logsumexp(q, axis=-1, keepdims=True)
    return np.array((mx.exp(lp) * (lp - lq)).sum(-1))


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b, equal_nan=True).item()


def model_arrays(st):
    """The target model's cache arrays of a snapshot (without the drafter ring and logits)."""
    return SS.State(st.pos, st.layers).arrays()


def fresh(model, drafter, ids, chunk=2048):
    cache, dcache = model.make_cache(), drafter.make_cache()
    last = SS.prefill(model, drafter, cache, dcache, ids, 0, chunk)
    return cache, dcache, last


def forced_rows(model, cache, last, force, vv):
    """Logits of the last prompt position, then of a plain decode step on each of force[:-1]."""
    rows = [last[:vv]]
    for t in force[:-1]:
        lg, _ = SS.decode_step(model, cache, int(t))
        mx.eval(lg, [c.state for c in cache])
        rows.append(lg[:vv])
    return mx.stack(rows)


def greedy_margins(model, cache, last, n, vv, eos):
    out, mg, lg = [], [], last
    for _ in range(n):
        lv = lg[:vv].astype(mx.float32)
        top2 = mx.topk(lv, 2)
        y = mx.argmax(lv)
        mx.eval(y, top2)
        out.append(y.item())
        mg.append(abs(top2[1].item() - top2[0].item()))
        if out[-1] in eos:
            break
        lg, _ = SS.decode_step(model, cache, out[-1])
        mx.eval(lg, [c.state for c in cache])
    return out, mg


def first_div(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if d is None and len(a) != len(b):
        d = min(len(a), len(b))
    return d


# ----------------------------------------------------------------------------- gate 1
def gate_exact(model, drafter, tok, res):
    res["exact"] = {}
    # 6151: not a chunk or block multiple, far blocks active (> 4096); 1501: the rings and pooled blocks compacted
    for split in (6151, 1501):
        _exact_at(model, drafter, tok, res, split)
    res["exact"]["pass"] = all(v["pass"] for v in res["exact"].values())


def _exact_at(model, drafter, tok, res, split):
    vv = model.args.vocab_valid
    rng = random.Random(7)
    text = repo_text(["volundr.py", "README.md"], 40000)
    ids = chat(tok, text, "Explain how the BCSA cache works.")[:9000]
    cache, dcache, last = fresh(model, drafter, ids[:split])
    st = SS.capture(cache, dcache, last)
    before = [np.array(a.astype(mx.float32)) if a.dtype != mx.int32 else np.array(a) for a in st.arrays()]
    cont = ids[split:]
    blocks = [[int(t) for t in np.random.default_rng(s).integers(0, vv, 8)] for s in range(6)]
    commits = [rng.randint(1, 8) for _ in blocks]

    def run(cache, dcache):
        logits = [SS.prefill(model, drafter, cache, dcache, list(ids[:split]) + list(cont[:1500]), split)]
        P = split + 1500
        for t in cont[1500:1520]:  # plain decode steps
            lg, f = SS.decode_step(model, cache, int(t), drafter.capture_layers)
            drafter.materialize(dcache, f, P)
            mx.eval(lg, [c.state for c in cache], dcache.state)
            logits.append(lg)
            P += 1
        drafts = []
        for blk, n in zip(blocks, commits):  # verify + commit (KDA replay, BCSA partial append, KV trim, Engram)
            bonus = mx.array(int(blk[0]), dtype=mx.int32)
            d = drafter.propose(dcache, bonus, P)
            lg, cap, rec = VF.forward(model, mx.array([blk], dtype=mx.int32), cache, capture=drafter.capture_layers,
                                      verify=True, compiled=True)
            mx.eval(lg, d)
            logits.append(lg)
            drafts.append(d)
            VF.commit(cache, rec, n)
            drafter.materialize(dcache, cap[:n], P)
            mx.eval([c.state for c in cache], dcache.state)
            P += n
        return logits, drafts, SS.capture(cache, dcache).arrays()

    a = run(cache, dcache)
    out = {}
    for k in range(2):
        c2, d2 = SS.restore(st)
        b = run(c2, d2)
        out[f"restore{k}_logits_identical"] = all(same(x, y) for x, y in zip(a[0], b[0]))
        out[f"restore{k}_drafts_identical"] = all(same(x, y) for x, y in zip(a[1], b[1]))
        out[f"restore{k}_final_cache_identical"] = len(a[2]) == len(b[2]) and all(same(x, y) for x, y in zip(a[2], b[2]))
    after = [np.array(a_.astype(mx.float32)) if a_.dtype != mx.int32 else np.array(a_) for a_ in st.arrays()]
    out["snapshot_unchanged"] = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(before, after))
    out["snapshot_gb"] = st.nbytes / 1e9
    out["snapshot_pos"] = st.pos
    out["rows_compared"] = sum(int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1 for x in a[0])
    out["pass"] = all(v for k, v in out.items() if k.endswith(("identical", "unchanged")))
    res["exact"][f"split{split}"] = out
    print("exact:", split, json.dumps(out), flush=True)


# ----------------------------------------------------------------------------- gate 2
def gate_resume(model, drafter, tok, res, new=128):
    vv = model.args.vocab_valid
    eos = set(tok.eos_token_ids)
    cases = [
        ("code-9k", ["kernels.py"], 30000, 0.8, "Which Metal kernels does this file define, and what does each do?"),
        ("notes-14k", ["NOTES_SPEC_ACCEPT.md", "spec/NOTES.md", "NOTES_COMBINED.md", "PARITY.md"], 60000, 0.75,
         "Summarize what these notes measured, as a short list."),
    ]
    res["resume"] = []
    for name, files, chars, frac, q in cases:
        text = repo_text(files, chars)
        B = chat(tok, text, q)
        A = B[: int(len(B) * frac)]  # turn 1 ended here (a cut that no chunk boundary of B's fresh prefill hits)
        A = A[: len(A) - (len(A) % 2048 == 0)]
        t0 = time.perf_counter()
        cF, dF, lF = fresh(model, drafter, B)
        sF = SS.capture(cF, dF, lF)
        tf = time.perf_counter() - t0
        cA, dA, lA = fresh(model, drafter, A)
        sA = SS.capture(cA, dA, lA)
        del cA, dA
        t0 = time.perf_counter()
        cR, dR = SS.restore(sA)
        lR = SS.prefill(model, drafter, cR, dR, B, len(A))
        tr = time.perf_counter() - t0
        sR = SS.capture(cR, dR, lR)
        cC, dC, lC = fresh(model, drafter, B, chunk=4096)
        sC = SS.capture(cC, dC, lC)
        del cF, dF, cR, dR, cC, dC
        mx.clear_cache()
        # free-running greedy from fresh (with margins) and from resumed
        gF, mF = greedy_margins(model, SS.restore(sF)[0], lF, new, vv, eos)
        gR, _ = greedy_margins(model, SS.restore(sR)[0], lR, new, vv, eos)
        # teacher-forced rows on fresh's tokens
        force = gF[:48]
        rF = forced_rows(model, SS.restore(sF)[0], lF, force, vv)
        rR = forced_rows(model, SS.restore(sR)[0], lR, force, vv)
        rC = forced_rows(model, SS.restore(sC)[0], lC, force, vv)
        klR, klC = kl_rows(rF, rR), kl_rows(rF, rC)
        topR = float((np.array(mx.argmax(rF, -1)) == np.array(mx.argmax(rR, -1))).mean())
        # speculative decoding from the restored state (and from fresh), against plain greedy from the same state
        from lookup import Lookup

        c2, d2 = SS.restore(sR)
        spR, rec = specdec.spec_decode(model, drafter, c2, d2, B, lR, new, 8, eos, Lookup(), finalize=True)
        c3, d3 = SS.restore(sF)
        spF, _ = specdec.spec_decode(model, drafter, c3, d3, B, lF, new, 8, eos, Lookup())
        del c2, d2, c3, d3
        dRF = first_div(gF, gR)
        dSR = first_div(gR, spR)
        dSF = first_div(gF, spF)
        o = {"name": name, "len_A": len(A), "len_B": len(B), "prefill_fresh_s": round(tf, 2),
             "prefill_resumed_s": round(tr, 2), "resumed_prefilled_tokens": len(B) - len(A),
             "kl_resumed_mean": float(klR.mean()), "kl_resumed_p99": float(np.percentile(klR, 99)),
             "kl_resumed_max": float(klR.max()), "kl_chunk4096_mean": float(klC.mean()),
             "kl_chunk4096_p99": float(np.percentile(klC, 99)), "kl_chunk4096_max": float(klC.max()),
             "top1_resumed": topR, "rows": int(klR.shape[0]),
             "greedy_resumed_vs_fresh_first_div": dRF,
             "fresh_margin_at_div": None if dRF is None or dRF >= len(mF) else mF[dRF],
             "spec_vs_greedy_from_restored_first_div": dSR,
             "spec_vs_greedy_from_fresh_first_div": dSF, "spec_tokens_per_step": sum(rec["acc"]) / max(1, len(rec["acc"])),
             "n_greedy": len(gF), "snapshot_gb_B": sF.nbytes / 1e9}
        # a divergence of the spec run from greedy-from-the-same-state: the greedy margin there
        if dSR is not None and dSR < len(gR):
            _, mR = greedy_margins(model, SS.restore(sR)[0], lR, dSR + 1, vv, eos)
            o["restored_margin_at_spec_div"] = mR[dSR] if dSR < len(mR) else None
        if dSF is not None and dSF < len(mF):
            o["fresh_margin_at_spec_div"] = mF[dSF]
        res["resume"].append(o)
        print("resume:", json.dumps(o), flush=True)
        del sF, sA, sR, sC
        mx.clear_cache()


# ----------------------------------------------------------------------------- where the resume KL comes from
def gate_split(model, drafter, tok, res):
    """The resumed prefill is a prefill whose chunk boundaries sit at the resume point. So: (a) a fresh prefill split
    at the same point must equal the resumed one bit for bit (the snapshot adds nothing); (b) moving the split to a
    multiple of 32 (the BCSA prefill kernel's query block) shows how much of the KL against the 2048-chunk prefill
    is the in-chunk tiling of the attention; (c) a split at 16 mod 32 for contrast."""
    vv = model.args.vocab_valid
    eos = set(tok.eos_token_ids)
    B = chat(tok, repo_text(["kernels.py"], 30000), "Which Metal kernels does this file define, and what does each do?")
    A = int(len(B) * 0.8)
    A -= (A % 2048 == 0)

    def rows_of(cache, last, force):
        return forced_rows(model, cache, last, force, vv)

    cF, dF, lF = fresh(model, drafter, B)
    sF = SS.capture(cF, dF, lF)
    del cF, dF
    gF, _ = greedy_margins(model, SS.restore(sF)[0], lF, 48, vv, eos)
    rF = rows_of(SS.restore(sF)[0], lF, gF[:48])
    out = {"len_A": A, "len_B": len(B)}
    cA, dA, lA = fresh(model, drafter, B[:A])
    sA = SS.capture(cA, dA, lA)
    del cA, dA
    cR, dR = SS.restore(sA)
    lR = SS.prefill(model, drafter, cR, dR, B, A)
    sR = SS.capture(cR, dR, lR)
    del cR, dR
    rR = rows_of(SS.restore(sR)[0], lR, gF[:48])
    for name, at in (("split_at_A", A), ("split_32_aligned", A - A % 32), ("split_16_mod_32", A - A % 32 + 16)):
        cS, dS = model.make_cache(), drafter.make_cache()
        lS = SS.prefill(model, drafter, cS, dS, B, 0, 2048, points=[at])
        sS = SS.capture(cS, dS, lS)
        del cS, dS
        rS = rows_of(SS.restore(sS)[0], lS, gF[:48])
        kl = kl_rows(rF, rS)
        o = {"split": at, "kl_vs_fresh_mean": float(kl.mean()), "kl_vs_fresh_max": float(kl.max()),
             "rows_identical_to_fresh": bool(same(rF, rS)), "model_state_identical_to_fresh": all(
                 same(x, y) for x, y in zip(model_arrays(sF), model_arrays(sS)))}
        if name == "split_at_A":
            o["rows_identical_to_resumed"] = bool(same(rR, rS))
            o["state_identical_to_resumed"] = all(same(x, y) for x, y in zip(sR.arrays(), sS.arrays()))  # + drafter
        out[name] = o
        print("split:", name, json.dumps(o), flush=True)
        del sS
        mx.clear_cache()
    klR = kl_rows(rF, rR)
    out["resumed_kl_vs_fresh_mean"] = float(klR.mean())
    out["pass"] = out["split_at_A"]["rows_identical_to_resumed"] and out["split_at_A"]["state_identical_to_resumed"]
    res["split"] = out
    print("split:", json.dumps(out), flush=True)


# ----------------------------------------------------------------------------- gate 3
def gate_engine(model, drafter, tok, res):
    eos = list(tok.eos_token_ids)
    text = repo_text(["sampler.py"], 12000)
    msgs = [{"role": "system", "content": "You are a concise code reviewer."},
            {"role": "user", "content": f"Here is a file:\n{text}\nWhat does Budget do? One paragraph."}]
    g = SS.Sampling(top_k=1, max_new_tokens=96, stop_token_ids=eos)
    cached = SS.Engine(model, drafter, SS.SessionCache(budget_gb=6))
    plain = SS.Engine(model, drafter, None)
    out = {"turns": []}
    for turn in range(2):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        r1 = cached.generate(ids, g, session="gate")
        r0 = plain.generate(ids, g, session="gate")
        div = first_div(r0["tokens"], r1["tokens"])
        out["turns"].append({"prompt": len(ids), "cached_tokens": r1["cached_tokens"],
                             "prefilled_tokens": r1["prefilled_tokens"], "snapshot": r1["snapshot"],
                             "ttft_cached_s": r1["ttft_s"], "ttft_plain_s": r0["ttft_s"], "first_div": div,
                             "generated": len(r1["tokens"]), "cache_gb": cached.cache.nbytes / 1e9})
        reply = tok.decode(r1["tokens"], skip_special_tokens=False)
        content = reply.split("<|/think|>")[-1].replace("<|agens_end|>", "").strip()
        reasoning = reply.split("<|/think|>")[0].strip() if "<|/think|>" in reply else ""
        msgs += [{"role": "assistant", "content": content, "reasoning_content": reasoning},
                 {"role": "user", "content": "And what does Sampler.sample do with min_p? Two sentences."}]
    t2 = out["turns"][1]
    out["pass"] = (t2["cached_tokens"] >= out["turns"][0]["prompt"] and t2["prefilled_tokens"] < t2["prompt"] // 2
                   and all(t["first_div"] is None for t in out["turns"]))
    res["engine"] = out
    print("engine:", json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="exact,resume,engine")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    model, drafter, tok = specdec.load(4, 4, nax=True)
    w = tok.encode("Warm up the kernels, please.")
    specdec.warm_up(model, drafter, w)
    res = {}
    for g in args.only.split(","):
        t = time.perf_counter()
        {"exact": gate_exact, "resume": gate_resume, "engine": gate_engine, "split": gate_split}[g](
            model, drafter, tok, res)
        print(f"[{g}: {time.perf_counter() - t:.0f} s, peak {mx.get_peak_memory() / 1e9:.1f} GB]", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
