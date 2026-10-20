"""Greedy speculative decoding for Volundr MLX with the DFlash2 block drafter, and the plain greedy baseline.

One step: the drafter proposes 7 tokens after the bonus token in one parallel pass; the target verifies
[bonus, d1..d_{T-1}] in one forward; the longest prefix of drafts that matches the target's argmax is kept, plus the
target's next token (the new bonus). Then the caches keep those n tokens (verify.commit) and the drafter's context
K/V gets their target features.

usage (from volundr-mlx/spec):
  python specdec.py check  [--suite suite.json] [--split tune] [--per-cat 1] [--new 128]   greedy == spec?
  python specdec.py bench  [--suite suite.json] [--split tune] [--new 256] [--T 8] --out runs/x.jsonl [--resume]
                           [--config baseline]   (../ab_modes.py config instead of the env-selected defaults)
  python specdec.py base   [--suite suite.json] [--split tune] [--new 256] --out runs/base.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Speculative runs: larger Metal command buffers. MLX lets the CPU run at most ~10 command buffers ahead of the GPU, and
# the drafter's ~400 kernels span more than that at the default size, so async_eval of the draft blocked until it was
# nearly done (26 ms) and the verify graph was built with the GPU idle. With 100 ops per buffer it returns in ~1 ms.
# Plain greedy decoding is ~4% slower with it (its own async pipeline prefers small buffers), so `base` keeps MLX's
# default. Read by MLX at device init, hence before the import. The setting is process-wide and cannot change at run
# time, so spec_prefill evaluates its chunks layer by layer instead: unsynchronised, a 2048-token chunk under these
# buffers peaked at 50-57 GB (MLX default buffers: 27-30 GB); layer by layer it peaks at ~27.7 GB on a 12.5K prompt.
if len(sys.argv) > 1 and sys.argv[1] in ("bench", "check") and "MLX_MAX_OPS_PER_BUFFER" not in os.environ:
    os.environ["MLX_MAX_OPS_PER_BUFFER"] = "100"
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import generate as G  # noqa: E402
import verify as VF  # noqa: E402
from dflash import BLOCK, WINDOW, DFlash2  # noqa: E402

MODEL = HERE.parent / "models/volundr-int4-mlx"


def load(drafter_bits=0, lm_head_bits=0, nax=False):
    model, tok = G.load(MODEL, lm_head_q=False)
    G.set_wired_limit()
    drafter = DFlash2.load(bits=drafter_bits, lm_head_bits=lm_head_bits)
    drafter.attach(model, model.args.vocab_valid)
    # the opt-in 8-bit target head (VOLUNDR_LMHEAD8) only after attach, so the drafter's quantized head copy is still
    # made from the bf16 weights (with --head-bits 0 the drafter keeps the bf16 head it shares)
    G.apply_lm_head_q(model)
    if nax:
        import nax_hook

        nax_hook.enable(model)
        nax_hook.enable(drafter)
    return model, drafter, tok


PREFILL_LAYER_SYNC = os.environ.get("VOLUNDR_PREFILL_SYNC", "1") != "0"


def spec_prefill(model, drafter, cache, dcache, ids, chunk=2048, layer_sync=None):
    """Chunked prefill that also fills the drafter's context with the last WINDOW positions' features.
    layer_sync (default PREFILL_LAYER_SYNC, on; VOLUNDR_PREFILL_SYNC=0 turns it off): evaluate each chunk layer by layer
    (verify.forward), which bounds the prefill peak under the spec processes' large command buffers (see the top of this
    file) to the weights plus two layers' activations. Same values; the decode loop keeps the large buffers."""
    if layer_sync is None:
        layer_sync = PREFILL_LAYER_SYNC
    n = len(ids)
    arr = mx.array(ids, dtype=mx.int32)[None]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        want = e > n - WINDOW
        logits, cap, _ = VF.forward(model, arr[:, s:e], cache, capture=drafter.capture_layers if want else (),
                                    layer_sync=layer_sync)
        if want:
            lo = max(s, n - WINDOW)
            drafter.materialize(dcache, cap[lo - s :], lo)
        mx.eval(logits, [c.state for c in cache], dcache.state)
        mx.clear_cache()
    return logits[-1]


# Default loop: the early schedule with the compiled KDA verify layers (`early@cv`; bit-identical to the base loop,
# ×1.025 on the tune split, NOTES_SPEC_ACCEPT.md). VOLUNDR_SPEC_SCHED=base / VOLUNDR_SPEC_COMPILED=0, or the loop flag
# `base`, give the previous loop.
SCHED = os.environ.get("VOLUNDR_SPEC_SCHED", "early")
COMPILED = os.environ.get("VOLUNDR_SPEC_COMPILED", "1") != "0"


def loop_options(flags) -> dict:
    """spec_generate keyword arguments from loop flags (ab_suite's config@flag, specdec's --loop): early -> sched;
    base -> the previous loop (base schedule, eager verify)."""
    opts = {}
    for f in flags:
        if f == "base":
            opts["sched"], opts["compiled"] = "base", False
        elif f == "early":
            opts["sched"] = f
        elif f == "cv":  # compiled KDA verify layers (verify.forward compiled=True; same values)
            opts["compiled"] = True
        elif f == "nocv":  # eager verify
            opts["compiled"] = False
        elif f == "s2":  # early schedule, the target commit on a second GPU stream (concurrent with the draft)
            opts["sched"], opts["stream2"] = "early", True
        elif f.startswith("long"):  # long<L>[c][g<b>]: full lookup copies verified with L - 1 drafts (early loop)
            import re

            m = re.fullmatch(r"long(\d+)(c?)(?:g(\d+))?", f)
            assert m, f
            opts.update(sched="early", long=int(m[1]), long_cap=bool(m[2]), long_gate=int(m[3] or 0))
        else:
            raise ValueError(f"unknown spec loop flag {f!r}")
    return opts


def spec_generate(model, drafter, ids, max_new, T=BLOCK, eos=(), chunk=2048, lookup=None, sched=None, stream2=False,
                  long=0, long_cap=False, long_gate=0, compiled=None):
    """Greedy tokens (up to max_new, stopping after an eos) and a record of the run (per-step accepted counts and
    draft source). lookup: a lookup.Lookup; when its copy proposal fills all T - 1 draft rows the drafter pass is
    skipped, a shorter one overwrites the first drafter slots (the engine's apply_hint).
    Loop options (loop_options maps the flags). The default is `early@cv` (SCHED, COMPILED), which gives the same
    tokens as the frozen config's base loop; stream2 and long are off by default:
    - sched (default SCHED, env VOLUNDR_SPEC_SCHED): "base" or "early" (_spec_loop_early: the same ops and values, the
      host work after each step's sync reordered so the GPU restarts sooner); stream2: early, the target commit on a
      second GPU stream. Same tokens.
    - compiled (default COMPILED, env VOLUNDR_SPEC_COMPILED): the verify's 54 KDA layers through mx.compile
      (verify.forward); the same values, a cheaper graph build.
    - long (early loop only; 0 = off): when the lookup's copy fills the whole draft and runs on to long - 1 tokens,
      verify all of them (long rows, at most 16: the NAX and 8q BCSA verify limit) instead of T - 1; long_cap: count at
      most T - 1 accepted drafts of such a step in the Lookup's running means; long_gate: only for a copy whose match
      length falls in Lookup bucket >= long_gate (lookup.BUCKETS). Changes the drafts, so tokens differ from the base
      loop where the target's top two logits tie."""
    if compiled is None:
        compiled = COMPILED
    if (sched or SCHED) == "early":
        return _spec_loop_early(model, drafter, ids, max_new, T, eos, chunk, lookup, stream2, long, long_cap,
                                long_gate, compiled)
    assert 2 <= T <= BLOCK
    vv = model.args.vocab_valid
    cache, dcache = model.make_cache(), drafter.make_cache()
    t0 = time.perf_counter()
    last = spec_prefill(model, drafter, cache, dcache, ids, chunk)
    bonus = mx.argmax(last[:vv]).astype(mx.int32)
    out = [bonus.item()]
    t1 = time.perf_counter()
    P, acc, src = len(ids), [], []
    t_wait = 0.0  # time blocked on the GPU result; the rest of decode_s is CPU (graph building, host bookkeeping)
    seg, te = [0.0, 0.0, 0.0], time.perf_counter()  # CPU: bookkeeping + commit + draft build, async_eval, verify build
    if lookup is not None:
        lookup.reset(list(ids) + out)
    while len(out) < max_new and out[-1] not in eos:
        hint = lookup.propose() if lookup is not None else []
        if len(hint) >= T - 1:
            drafts = mx.array(hint[: T - 1], dtype=mx.int32)
        else:
            drafts = drafter.propose(dcache, bonus, P)
            if hint:
                drafts = mx.concatenate([mx.array(hint, dtype=mx.int32), drafts[len(hint) :]])
        inp = mx.concatenate([bonus.reshape(1), drafts[: T - 1]])[None]
        ta = time.perf_counter()
        mx.async_eval(inp)  # the GPU drafts (and commits the last step) while the CPU builds the verify graph
        tb = time.perf_counter()
        logits, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True, compiled=compiled)
        tok = mx.argmax(logits[:, :vv], axis=-1).astype(mx.int32)
        nacc = mx.cumprod((inp[0, 1:] == tok[:-1]).astype(mx.int32)).sum()
        tw = time.perf_counter()
        seg[0] += ta - te
        seg[1] += tb - ta
        seg[2] += tw - tb
        mx.eval(tok, nacc, inp, [c.state for c in cache], dcache.state)
        te = time.perf_counter()
        t_wait += te - tw
        n = nacc.item() + 1
        emitted = inp[0, 1:n].tolist() + [tok[n - 1].item()]
        acc.append(n)
        src.append(len(hint))
        stop = next((i for i, t in enumerate(emitted) if t in eos), None)
        if stop is not None:  # an eos inside the accepted run ends the reply there
            out += emitted[: stop + 1]
            break
        out += emitted
        if lookup is not None:
            lookup.record(emitted)
        VF.commit(cache, rec, n)
        drafter.materialize(dcache, cap[:n], P)
        bonus = tok[n - 1]
        P += n
    t2 = time.perf_counter()
    out = out[:max_new]
    return out, {"prefill_s": t1 - t0, "decode_s": t2 - t1, "wait_s": t_wait, "seg_s": seg, "steps": len(acc),
                 "acc": acc, "src": src}


_S2 = []


def _stream2():
    if not _S2:
        _S2.append(mx.new_stream(mx.gpu))
    return _S2[0]


def _long_copy(hint, lookup, T, long, long_gate):
    """Draft length of this step's lookup proposal under the long-copy option (spec_generate's long / long_gate)."""
    if long and len(hint) >= long - 1 and lookup.pending[1] >= long_gate:
        return long - 1
    return T - 1 if len(hint) >= T - 1 else len(hint)


def _spec_loop_early(model, drafter, ids, max_new, T, eos, chunk, lookup, stream2=False, long=0, long_cap=False,
                     long_gate=0, compiled=False):
    """spec_generate's loop with the same ops and values, scheduled so the GPU idles less between steps. In the base
    loop, after a step's sync the host does the bookkeeping, builds the target commit, the drafter-context update
    (materialize) and the next draft, and only then submits work, so the GPU waits for all of it. Here:
    1. right after the sync (and the accepted count), the materialize is built and submitted: the GPU starts on it;
    2. the bookkeeping, lookup and the draft build overlap it; the draft is submitted;
    3. only then the target commit is built (it is the verify's input, not the drafter's), while the GPU drafts. When
       the lookup filled the whole draft (no drafter pass to overlap), the commit is submitted on its own so the GPU
       runs it while the CPU builds the verify.
    GPU order is unchanged (materialize, draft, commit, verify); only when the CPU submits each part changes.
    long: see spec_generate (changes the drafts, so not the same tokens as the base loop)."""
    cache, dcache = model.make_cache(), drafter.make_cache()
    t0 = time.perf_counter()
    last = spec_prefill(model, drafter, cache, dcache, ids, chunk)
    return spec_decode(model, drafter, cache, dcache, ids, last, max_new, T, eos, lookup, stream2, long, long_cap,
                       long_gate, compiled, t0=t0)


def spec_decode(model, drafter, cache, dcache, ids, last, max_new, T=BLOCK, eos=(), lookup=None, stream2=False, long=0,
                long_cap=False, long_gate=0, compiled=None, t0=None, emit=None, finalize=False):
    """The early loop (_spec_loop_early) from a prefilled state: cache holds ids, dcache the drafter context of their
    last WINDOW positions, last = the logits of the last position (spec_prefill's return value). Used by the session
    cache (../session.py), which starts it from a restored snapshot. With emit=None and finalize=False it runs exactly
    the ops of _spec_loop_early.
    - emit(tokens): called with the first token, then with each step's new tokens (cut at an eos and at max_new);
      returning False stops after that step (cancellation).
    - finalize: on return the caches hold exactly ids + out[:-1]: the last step's commit and drafter-context update are
      applied for the tokens kept (the loop otherwise leaves them pending). rec gets "P" (that length) and
      "last_logits" (the logits that picked out[-1]), so the state can be snapshotted or continued.
    The loop runs SpecStepper one step at a time (the batched server steps it itself)."""
    st = SpecStepper(model, drafter, cache, dcache, ids, last, max_new, T, eos, lookup, stream2, long, long_cap,
                     long_gate, compiled, t0=t0, emit=emit)
    while st.step():
        pass
    return st.result(finalize)


class SpecStepper:
    """spec_decode's loop, one step per step() call (the same ops in the same order). The batched server
    (../scheduler.py) runs one step between its other work, and stops a stream in the middle of a reply to move it into
    the decode batch (result(finalize=True) leaves the caches holding ids + out[:-1]).
    first: resume a reply whose next token is already chosen and emitted (the caches hold ids, `first` follows them,
    `last` the logits that chose it); otherwise the first token is the argmax of `last` and is emitted here."""

    def __init__(self, model, drafter, cache, dcache, ids, last, max_new, T=BLOCK, eos=(), lookup=None, stream2=False,
                 long=0, long_cap=False, long_gate=0, compiled=None, t0=None, emit=None, first=None):
        assert 2 <= T <= BLOCK and long in (0,) + tuple(range(T + 1, 17))
        if compiled is None:
            compiled = COMPILED
        self.model, self.drafter, self.cache, self.dcache = model, drafter, cache, dcache
        self.ids, self.last, self.max_new, self.T, self.eos, self.lookup = ids, last, max_new, T, eos, lookup
        self.stream2, self.long, self.long_cap, self.long_gate = stream2, long, long_cap, long_gate
        self.compiled = compiled
        self.emit = emit
        self.vv = model.args.vocab_valid
        self.t0 = time.perf_counter() if t0 is None else t0
        if first is None:
            self.bonus = mx.argmax(last[: self.vv]).astype(mx.int32)
            self.out = [self.bonus.item()]
            self.t1 = time.perf_counter()
            self.stopped = emit is not None and emit(self.out[:1]) is False
        else:
            self.bonus = mx.array(int(first), dtype=mx.int32)
            self.out = [int(first)]
            self.t1 = time.perf_counter()
            self.stopped = False
        self.P, self.acc, self.src = len(ids), [], []
        self.t_wait = 0.0
        # CPU segments: bookkeeping + draft build, async_eval + commit build, verify build
        self.seg, self.te = [0.0, 0.0, 0.0], time.perf_counter()
        if lookup is not None:
            lookup.reset(list(ids) + self.out)
        self.pend = None  # (rec, n): the last step's target commit, built once the next draft is submitted
        self.last_step = None  # (rec, cap, P, logits, len(out) before the step): the step whose commit finalize applies
        self.s2 = _stream2() if stream2 else None
        self.over = False

    @property
    def done(self):
        return self.over or self.stopped or len(self.out) >= self.max_new or self.out[-1] in self.eos

    def step(self) -> bool:
        """One draft + verify step and its emit; False (nothing run) once the reply is over."""
        if self.done:
            return False
        model, drafter, cache, dcache, lookup = self.model, self.drafter, self.cache, self.dcache, self.lookup
        T, long, out, eos, emit, max_new = self.T, self.long, self.out, self.eos, self.emit, self.max_new
        bonus, P = self.bonus, self.P
        hint = (lookup.propose(long - 1) if long else lookup.propose()) if lookup is not None else []
        full = len(hint) >= T - 1
        if full:  # (a long copy keeps up to long - 1 tokens)
            hint = hint[: _long_copy(hint, lookup, T, long, self.long_gate)]
            drafts = mx.array(hint, dtype=mx.int32)
        else:
            drafts = drafter.propose(dcache, bonus, P)
            if hint:
                drafts = mx.concatenate([mx.array(hint, dtype=mx.int32), drafts[len(hint) :]])
        inp = mx.concatenate([bonus.reshape(1), drafts if len(hint) > T - 1 else drafts[: T - 1]])[None]
        ta = time.perf_counter()
        mx.async_eval(inp)
        if self.pend is not None:
            if self.stream2:  # on its own queue, so the GPU can run it next to the draft; the verify waits for it
                with mx.stream(self.s2):
                    VF.commit(cache, *self.pend)
                    mx.async_eval([c.state for c in cache])
            else:
                VF.commit(cache, *self.pend)
                if full:  # no draft on the GPU: let it run the commit while the verify graph is built
                    mx.async_eval([c.state for c in cache])
            self.pend = None
        tb = time.perf_counter()
        logits, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True,
                                      compiled=self.compiled)
        tok = mx.argmax(logits[:, : self.vv], axis=-1).astype(mx.int32)
        nacc = mx.cumprod((inp[0, 1:] == tok[:-1]).astype(mx.int32)).sum()
        tw = time.perf_counter()
        self.seg[0] += ta - self.te
        self.seg[1] += tb - ta
        self.seg[2] += tw - tb
        mx.eval(tok, nacc, inp, [c.state for c in cache], dcache.state)
        self.te = time.perf_counter()
        self.t_wait += self.te - tw
        n = nacc.item() + 1
        emitted = inp[0, 1:n].tolist() + [tok[n - 1].item()]
        self.acc.append(n)
        self.src.append(len(hint))
        self.last_step = (rec, cap, P, logits, len(out))
        stop = next((i for i, t in enumerate(emitted) if t in eos), None)
        if stop is not None:  # an eos inside the accepted run ends the reply there
            out += emitted[: stop + 1]
            if emit is not None:
                emit(emitted[: stop + 1][: max(0, max_new - self.last_step[4])])
            self.over = True
            return True
        out += emitted
        if emit is not None and emit(emitted[: max(0, max_new - self.last_step[4])]) is False:
            self.stopped = True
            return True
        if len(out) < max_new:  # (the base loop builds these at the end too, but never runs them)
            drafter.materialize(dcache, cap[:n], P)
            mx.async_eval(dcache.state)
        if lookup is not None:
            lookup.record(emitted, cap=T - 1 if self.long_cap else None)
        self.pend = (rec, n)
        self.bonus = tok[n - 1]
        self.P = P + n
        return True

    def result(self, finalize=False):
        """(out, record) as spec_decode returns them; finalize applies the pending commit (see spec_decode)."""
        t2 = time.perf_counter()
        out = self.out = self.out[: self.max_new]
        res = {"prefill_s": self.t1 - self.t0, "decode_s": t2 - self.t1, "wait_s": self.t_wait, "seg_s": self.seg,
               "steps": len(self.acc), "acc": self.acc, "src": self.src}
        if finalize:
            cache, dcache = self.cache, self.dcache
            res["last_logits"] = self.last
            P_end = len(self.ids)
            if self.last_step is not None:  # its commit (and the drafter-context update) is still pending: keep its
                rec, cap, P0, logits, m0 = self.last_step  # tokens in out
                kept = len(out) - m0
                VF.commit(cache, rec, kept)
                self.drafter.materialize(dcache, cap[:kept], P0)
                mx.eval([c.state for c in cache], dcache.state)
                res["last_logits"] = logits[kept - 1]
                P_end = P0 + kept
            assert P_end == len(self.ids) + len(out) - 1, (P_end, len(self.ids), len(out))
            res["P"] = P_end
            self.last_step = self.pend = None
        return out, res


def warm_up(model, drafter, ids, T=BLOCK, **loop):
    """Metal JIT for a spec config before timing: a short spec_generate, and with long copies also one verify (and
    commit) of `long` rows, which a short warm-up prompt need not reach."""
    spec_generate(model, drafter, ids, 24, T, **loop)
    if loop.get("long"):
        cache, dcache = model.make_cache(), drafter.make_cache()
        spec_prefill(model, drafter, cache, dcache, ids)
        inp = mx.array([(list(ids) * loop["long"])[: loop["long"]]], dtype=mx.int32)
        _, cap, rec = VF.forward(model, inp, cache, capture=drafter.capture_layers, verify=True,
                                 compiled=loop.get("compiled", COMPILED))
        VF.commit(cache, rec, 3)
        drafter.materialize(dcache, cap[:loop["long"] - 1], len(ids))
        mx.eval([c.state for c in cache], dcache.state)
        del cache, dcache
        mx.clear_cache()


def lookup_replay(lookup, ids, rec, eos=(), T=BLOCK, long=0, long_cap=False, long_gate=0, **_):
    """Feed a recorded spec_generate run of ids (rec: tokens, per-step accepted counts acc, draft sources src) into
    lookup exactly as that run did, so a resumed suite run continues the same Lookup history. Lookup is deterministic
    given its history, so every step's proposal is re-derived and checked against the recorded source. long /
    long_cap: the run's loop options (spec_generate)."""
    out = rec["tokens"]
    lookup.reset(list(ids) + out[:1])
    pos = 1
    for i, (n, s) in enumerate(zip(rec["acc"], rec["src"])):
        hint = lookup.propose(long - 1) if long else lookup.propose()
        k = _long_copy(hint, lookup, T, long, long_gate)
        assert k == s, f"{rec['name']}: step {i} proposes {k} tokens, the run used {s}"
        emitted = out[pos : pos + n]
        if any(t in eos for t in emitted):  # spec_generate stops on an eos before recording the step
            break
        # the last step may be cut at max_new: its length counts
        lookup.record(emitted + [-1] * (n - len(emitted)), cap=T - 1 if long_cap else None)
        pos += n


def greedy_generate(model, ids, max_new, eos=(), chunk=2048, margins=False):
    """Plain greedy decoding (generate.py's pipelined loop); margins=True records top1 - top2 logits per token
    instead (unpipelined, for the exactness check)."""
    vv = model.args.vocab_valid
    cache = model.make_cache()
    t0 = time.perf_counter()
    last = G.prefill(model, cache, ids, chunk)
    mx.eval(last)
    t1 = time.perf_counter()
    if not margins:
        out = list(G.greedy_tokens(model, cache, last, max_new, set(eos), vv))
        return out, {"prefill_s": t1 - t0, "decode_s": time.perf_counter() - t1}
    out, mg, logits = [], [], last
    for _ in range(max_new):
        lv = logits[:vv].astype(mx.float32)
        top2 = mx.topk(lv, 2)
        y = mx.argmax(lv).astype(mx.int32)
        mx.eval(y, top2)
        out.append(y.item())
        mg.append(abs(top2[1].item() - top2[0].item()))
        if out[-1] in eos:
            break
        logits = model(y.reshape(1, 1), cache=cache, last_only=True)[0, -1]
        mx.eval(logits, [c.state for c in cache])
    return out, {"prefill_s": t1 - t0, "margins": mg}


def suite(path, split, per_cat=0, cats=None):
    s = [e for e in json.loads(Path(path).read_text()) if split in (e["split"], "all")]
    if cats:
        s = [e for e in s if e["category"] in cats]
    if per_cat:
        seen = {}
        s = [e for e in s if seen.setdefault(e["category"], []).append(e) or len(seen[e["category"]]) <= per_cat]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "bench", "base"])
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--per-cat", type=int, default=0)
    ap.add_argument("--cats", default="")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--T", type=int, default=BLOCK)
    ap.add_argument("--drafter-bits", type=int, default=0)
    ap.add_argument("--head-bits", type=int, default=0, help="quantized copy of the lm_head for drafting")
    ap.add_argument("--nax", action="store_true", help="small-M verify matmuls on the NAX kernel (nax_hook.py)")
    ap.add_argument("--lookup", action="store_true", help="copy-from-context drafts (lookup.py)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="", help="an ../ab_modes.py config (e.g. baseline: main's decode path and "
                    "verify, which env vars cannot fully select); default: the env-selected defaults")
    ap.add_argument("--resume", action="store_true", help="skip prompts already in --out (bench: replaying their "
                    "Lookup history, see lookup_replay)")
    ap.add_argument("--loop", default="", help="spec loop flags, @-separated (loop_options; e.g. early)")
    args = ap.parse_args()
    loop = loop_options([f for f in args.loop.split("@") if f])

    model, drafter, tok = load(args.drafter_bits, args.head_bits, args.nax)
    import ab_modes

    if args.config:
        ab_modes.Modes(model, [args.config]).set(args.config)  # before the warm-up, so its kernels are JIT-ed too
    eos = set(tok.eos_token_ids)
    prompts = suite(args.suite, args.split, args.per_cat, [c for c in args.cats.split(",") if c])
    # warm up the kernels (Metal JIT) on a short prompt
    w = tok.encode("Warm up the kernels, please.")
    warm_up(model, drafter, w, args.T, **loop)
    greedy_generate(model, w, 8)
    look = None
    if args.lookup:
        from lookup import Lookup

        look = Lookup()
    done = {}
    if args.resume and args.out and Path(args.out).exists():  # skip prompts already in --out, replaying their Lookup
        done = {r["name"]: r for r in map(json.loads, filter(str.strip, Path(args.out).read_text().splitlines()))}
    fout = open(args.out, "a") if args.out else None
    for e in prompts:
        if e["name"] in done:
            if look is not None and args.cmd == "bench":
                lookup_replay(look, e["ids"], done[e["name"]], eos, **loop)
            continue
        rec = {"name": e["name"], "category": e["category"], "split": e["split"], "prompt_tokens": len(e["ids"]),
               "T": args.T, "drafter_bits": args.drafter_bits, "head_bits": args.head_bits, "nax": args.nax,
               "lookup": args.lookup, "target_head_q8": model.lm_head.group_size if "scales" in model.lm_head else 0,
               "config": args.config or None, "loop": loop, **ab_modes.describe()}
        if args.cmd == "check":
            ref, r0 = greedy_generate(model, e["ids"], args.new, eos, margins=True)
            got, r1 = spec_generate(model, drafter, e["ids"], args.new, args.T, eos, lookup=look, **loop)
            div = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
            if div is None and len(ref) != len(got):
                div = min(len(ref), len(got))
            rec.update(match=div is None, first_diverge=div, n_ref=len(ref), n_spec=len(got),
                       margin_at_div=None if div is None or div >= len(r0["margins"]) else r0["margins"][div],
                       min_margin=min(r0["margins"]), mean_acc=sum(r1["acc"]) / max(1, len(r1["acc"])))
        elif args.cmd == "bench":
            got, r1 = spec_generate(model, drafter, e["ids"], args.new, args.T, eos, lookup=look, **loop)
            rec.update(generated=len(got), prefill_s=r1["prefill_s"], decode_s=r1["decode_s"], wait_s=r1["wait_s"],
                       seg_s=r1["seg_s"], steps=r1["steps"],
                       acc=r1["acc"], src=r1["src"], mean_acc=sum(r1["acc"]) / max(1, len(r1["acc"])), tok_s=(len(got) - 1) / r1["decode_s"] if r1["decode_s"] > 0 else None,
                       tokens=got)
        else:
            got, r0 = greedy_generate(model, e["ids"], args.new, eos)
            rec.update(generated=len(got), prefill_s=r0["prefill_s"], decode_s=r0["decode_s"],
                       tok_s=(len(got) - 1) / r0["decode_s"] if r0["decode_s"] > 0 else None, tokens=got)
        mx.clear_cache()
        short = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("acc", "src", "tokens")}
        print(json.dumps(short), flush=True)
        if fout:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
