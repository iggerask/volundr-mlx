"""Continuous batching for the MLX worker (worker.py), after upstream agens-engine's server mode
(crates/agens-cli/src/serve.rs): one engine thread owns the GPU, and each pass of its loop
1. takes new requests and admits what fits (up to max_batch active requests, and a memory limit);
2. runs prefill work: one chunk of the oldest request still prefilling when nothing is decoding, else prefill
   layers until `slice_ms` have passed (session.prefill_iter yields between layers, so a long prompt never stalls
   the decoding streams for a whole chunk; a short prompt done early hands the rest of the slice to the next one);
3. runs ONE decode step for every decoding request together, and streams each request's tokens.

Policy for the decode step:
- one decoding request that is greedy (Sampling.speculative): one step of the speculative loop (specdec.SpecStepper:
  DFlash2 + prompt lookup + NAX verify, the bs = 1 path, ~28 tok/s);
- 2..bspec_max decoding requests, all greedy: one batched speculative step (batch_spec.py: each row's draft, one
  verify forward of B x T rows, T = bspec_rows // B, each row keeps its own accepted run);
- otherwise every decoding request is a row of the decode batch (batch.BatchCache: per-row positions, slot buffers)
  and the step is one batched plain decode forward, NAX small-M matmuls from `nax_min_batch` rows (stock MLX is
  faster at 2-3 rows). Greedy rows take the GPU argmax, sampled rows sampler.py on their logits row.
A speculative stream that gets company is finalized (the pending commit applied) and joins the batch; the last one
left in the batch, if greedy, leaves it and continues speculatively (its drafter context is filled from the features
the batch captured for it).

Sessions: the session cache (session.SessionCache) works as at bs = 1: a request restores the longest snapshot whose
tokens prefix its prompt into its own B = 1 caches, prefills the rest there (interval and prompt snapshots), and at
the end of its reply the caches (its batch row via BatchCache.capture_row, or its spec state) are snapshotted.
"""
import sys
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spec"))
import batch as BT  # noqa: E402
import session as SS  # noqa: E402

FEATS_FLUSH = 64  # batch rows' drafter features are materialized into their drafter context every N tokens


class Req:
    """One request in the engine."""

    def __init__(self, job, ids, sp: SS.Sampling, session):
        self.job, self.ids, self.sp, self.session = job, ids, sp, session
        self.n = len(ids)
        self.ids_np = np.asarray(ids, dtype=np.int32)
        self.out = []
        self.phase = "prefill"  # -> "decode" -> "done"
        self.mode = None  # decode: None (not started), "spec", "batch"
        self.cache = self.dcache = None  # B = 1 caches while prefilling / speculative / about to join the batch
        self.gen = None  # prefill generator
        self.last = None  # logits of the last fed position (they chose out[-1])
        self.feats, self.feat_p0 = [], 0  # batch mode: drafter features not yet in dcache, from position feat_p0
        self.lk, self.lk_len = None, 0  # batched speculative steps: the request's prompt lookup, len(out) it saw
        self.stepper = None
        self.sampler = None
        self.stop_ids = set() if sp.ignore_eos else set(sp.stop_token_ids)
        self.finish = None
        self.rec = {}
        self.t_admit = time.perf_counter()
        self.t_first = self.t_prefill = None
        self.spec_steps = self.batch_steps = self.batch_rows = 0
        self.switches = 0

    @property
    def cancelled(self):
        return self.job.cancel.is_set()


class BatchEngine:
    def __init__(self, model, drafter, cache: SS.SessionCache, T=8, chunk=2048, lookup=True, loop="",
                 think_ids=(None, None), max_batch=16, slice_ms=200.0, nax_min_batch=4, spec=True, max_gb=46.0,
                 pipeline=True, bspec=True, bspec_max=4, bspec_rows=16, clear_min=0, log=None):
        import specdec

        self.model, self.drafter, self.cache = model, drafter, cache
        self.T, self.chunk, self.lookup = T, chunk, lookup
        self.loop = specdec.loop_options([f for f in loop.split("@") if f])
        assert self.loop.get("sched", specdec.SCHED) == "early", "the engine runs the early spec loop"
        self.loop.pop("sched", None)
        self.think_ids = think_ids
        self.vv = model.args.vocab_valid
        self.max_batch, self.slice_s, self.nax_min_batch = max_batch, slice_ms / 1e3, nax_min_batch
        self.spec, self.max_gb, self.pipeline, self.bspec = spec, max_gb, pipeline, bspec
        # batched speculation for 2..bspec_max greedy rows, each drafting T = bspec_rows // B tokens (2 <= T <= self.T)
        self.bspec_max, self.bspec_rows = bspec_max, bspec_rows
        # prefill chunks shorter than clear_min keep MLX's buffer cache (session.prefill_iter). 0 (default): clear
        # after every chunk, as at bs = 1. Keeping it (1024) was tried: buffers of every batch size seen stay cached
        # (each join / leave changes the KDA state shapes); join steps then took 0.7-1.7 s, but a rerun with the
        # default had 0.5-1.0 s ones too while swap had grown to 27 GB (gate_prof), so it is not settled
        self.clear_min = clear_min
        self.bc = BT.BatchCache(model)
        self.waiting = deque()
        self.active = []
        self.requests = 0
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.on_done = None  # callback(req, rec) when a request ends (the worker sends the done frame)
        self.stats = {"steps_batch": 0, "steps_spec": 0, "steps_bspec": 0, "rows_batch": 0, "prefill_slices": 0,
                      "stall_s_max": 0.0}
        # wall seconds per part of the loop (admission, prefill work, decode steps by kind, joins, request ends)
        for k in ("admit", "prefill", "spec", "bspec", "batch", "join", "switch", "finish"):
            self.stats["t_" + k] = 0.0
        self._last_decode = None  # time of the last decode step (stall measurement)
        self.step_log = None  # a list to record every tick's decode step in (gate.py)

    # ------------------------------------------------------------------ admission
    def submit(self, job):
        self.waiting.append(job)

    @property
    def busy(self):
        return bool(self.active or self.waiting)

    def _admit(self):
        while self.waiting and len(self.active) < self.max_batch:
            if self.active and mx.get_active_memory() / 1e9 > self.max_gb:
                break  # memory: wait for a request to finish (one at a time always fits)
            job = self.waiting.popleft()
            if job.cancel.is_set():
                rec = {"finish": "cancelled", "cancelled": "queued", "queue_s": time.perf_counter() - job.t_recv,
                       "generated": 0, "prompt_tokens": len(job.msg.get("prompt_ids") or []),
                       "session_id": job.msg.get("session_id"), "t": time.time()}
                self.on_done(job, None, rec)
                continue
            try:
                r = self._start(job)
            except Exception as e:  # noqa: BLE001
                self.on_done(job, None, {"error": f"{type(e).__name__}: {e}"})
                continue
            self.active.append(r)

    def _start(self, job):
        msg = job.msg
        ids = [int(t) for t in msg["prompt_ids"]]
        if not ids:
            raise ValueError("empty prompt")
        sp = SS.Sampling.from_dict(msg.get("sampling"))
        ctx = job.context_length
        if len(ids) + sp.max_new_tokens > ctx + 1:
            sp.max_new_tokens = max(0, ctx - len(ids))
        r = Req(job, ids, sp, msg.get("session_id"))
        self.requests += 1
        r.rid = self.requests
        r.pure_greedy = sp.greedy and not sp.needs_logits(self.vv) and sp.reasoning_budget is None
        r.spec_ok = self.spec and sp.speculative(self.vv)
        if not r.pure_greedy:
            from sampler import Sampler

            r.sampler = Sampler(sp, prompt=ids, think_ids=self.think_ids, vocab_valid=self.vv, request_id=r.rid)
        snap = self.cache.find(r.ids_np, r.session) if self.cache is not None else None
        if snap is not None:
            r.cache, r.dcache = SS.restore(snap.state)
            start = snap.pos
            r.last = SS._ref(snap.state.logits) if snap.pos == r.n else None
        else:
            r.cache, r.dcache = self.model.make_cache(), self.drafter.make_cache()
            start = 0
        if r.dcache is None:  # (a snapshot without drafter context: speculation still works, it just drafts worse)
            r.dcache = self.drafter.make_cache()
        r.rec = {"prompt_tokens": r.n, "cached_tokens": start, "prefilled_tokens": r.n - start,
                 "snapshot": None if snap is None else {"id": snap.id, "kind": snap.kind, "session": snap.session,
                                                        "same_session": snap.session == r.session},
                 "queue_s": r.t_admit - job.t_recv}
        r.n_points = 0
        if start < r.n:
            pts = self.cache.points(start, r.n) if self.cache is not None else []

            def on_point(pos, logits, r=r):
                if self._snap(r.session, "interval", r.ids_np[:pos], r.cache, r.dcache, logits) is not None:
                    r.n_points += 1

            r.n_prompt = len(ids)
            r.gen = SS.prefill_iter(self.model, self.drafter, r.cache, r.dcache, ids, start, self.chunk, pts,
                                    on_point, yield_layers=True, clear_min=self.clear_min)
        return r

    def _snap(self, session, kind, tokens, cache, dcache, logits):
        if self.cache is None or not self.cache.enabled or self.cache.has(tokens):
            return None
        return self.cache.add(session, kind, tokens, SS.capture(cache, dcache, logits))

    def _snap_state(self, session, kind, tokens, state):
        if self.cache is None or not self.cache.enabled or self.cache.has(tokens):
            return None
        return self.cache.add(session, kind, tokens, state)

    # ------------------------------------------------------------------ the loop
    def tick(self):
        """One pass: cancellations, admission, prefill work, one decode step."""
        for r in list(self.active):
            if r.cancelled and r.phase != "done":
                self._finish(r, "cancelled")
        t0 = time.perf_counter()
        self._admit()
        t1 = time.perf_counter()
        self.stats["t_admit"] += t1 - t0
        decoding = [r for r in self.active if r.phase == "decode"]
        pres = [r for r in self.active if r.phase == "prefill"]
        pre = pres[0] if pres else None
        t_end = t1 + self.slice_s
        for p in pres:  # the oldest first; the next one only while the slice lasts
            try:
                if not self._prefill_work(p, bool(decoding), t_end):
                    break
            except Exception as e:  # noqa: BLE001
                self._fail([p], e)
                break
        if pres:
            self.stats["t_prefill"] += time.perf_counter() - t1
        decoding = [r for r in self.active if r.phase == "decode"]
        if decoding:
            t2 = time.perf_counter()
            try:
                self._decode_step(decoding)
            except Exception as e:  # noqa: BLE001
                self._fail(decoding, e, reset_batch=True)
            if self.step_log is not None:  # (rows, decode step s, prefill work s this tick)
                self.step_log.append((len(decoding), time.perf_counter() - t2, t2 - t1 if pre is not None else 0.0))
        if not self.active:
            self._last_decode = None

    def _fail(self, reqs, e, reset_batch=False):
        import traceback

        traceback.print_exc()
        for r in reqs:
            r.finish = "error"
            r.phase = "done"
            if r in self.active:
                self.active.remove(r)
            self.on_done(r.job, r, {"error": f"{type(e).__name__}: {e}"})
        if reset_batch:
            self.bc = BT.BatchCache(self.model)
            for r in self.active:
                if r.mode == "batch":  # (every batch row was in reqs)
                    r.mode = None
        mx.clear_cache()

    def _prefill_work(self, r, others, t_end):
        """Advance r's prefill: to the end of its current chunk when nothing is decoding, else until t_end. True when
        r's prefill ended while others decode and the slice has time left (the next request's prefill may go on)."""
        self.stats["prefill_slices"] += 1
        if r.gen is None:  # the snapshot covers the whole prompt (it carries the last logits)
            self._prefill_done(r, r.last)
            return others and time.perf_counter() < t_end
        try:
            while True:
                kind, v = next(r.gen)
                if r.cancelled:
                    self._finish(r, "cancelled")
                    return False
                if kind == "chunk" and v == r.n_prompt:  # the last chunk: the generator ends at the next call
                    continue
                if kind == "chunk" and not others:
                    return False
                if others and time.perf_counter() >= t_end:
                    return False
        except StopIteration as e:
            r.gen = None
            self._prefill_done(r, e.value)
            return others and time.perf_counter() < t_end

    def _prefill_done(self, r, last):
        r.last = last
        if r.rec["prefilled_tokens"]:
            self._snap(r.session, "prompt", r.ids_np, r.cache, r.dcache, last)
        r.t_prefill = time.perf_counter()
        r.rec.update(prefill_s=r.t_prefill - r.t_admit, interval_snapshots=r.n_points)
        r.phase = "decode"
        if r.sp.max_new_tokens <= 0:
            return self._finish(r, "length")
        t = int(mx.argmax(last[: self.vv]).item()) if r.pure_greedy else r.sampler.choose(last)
        self._emit(r, [t])
        if r.finish:
            self._finish(r, r.finish)

    def _emit(self, r, toks):
        """Stream tokens (cut at max_new_tokens and after a stop id); sets r.finish when the reply is over."""
        room = r.sp.max_new_tokens - len(r.out)
        toks = list(toks[: max(0, room)])
        stop = next((i for i, t in enumerate(toks) if t in r.stop_ids), None)
        if stop is not None:
            toks = toks[: stop + 1]
        if not toks:
            return True
        if r.t_first is None:
            r.t_first = time.perf_counter()
        r.out += toks
        ok = not r.cancelled and r.job.conn.send({"type": "tokens", "ids": toks})
        if not ok:
            r.finish = "cancelled"
        elif stop is not None:
            r.finish = "stop"
        elif len(r.out) >= r.sp.max_new_tokens:
            r.finish = "length"
        return ok

    def _decode_step(self, decoding):
        now = time.perf_counter()
        if self._last_decode is not None:
            self.stats["stall_s_max"] = max(self.stats["stall_s_max"], now - self._last_decode)
        if len(decoding) == 1 and decoding[0].spec_ok:
            r = decoding[0]
            if r.mode == "batch":
                self._batch_to_spec(r)
            elif r.mode is None:
                self._start_spec(r)
            t1 = time.perf_counter()
            self.stats["t_switch"] += t1 - now
            r.stepper.step()
            self.stats["t_spec"] += time.perf_counter() - t1
            r.spec_steps += 1
            self.stats["steps_spec"] += 1
            if r.stepper.done and not r.finish:
                r.finish = "stop" if r.out and r.out[-1] in r.stop_ids else "length"
            if r.finish:
                self._finish(r, r.finish)
        else:
            join = []
            for r in decoding:
                if r.mode == "spec":
                    self._spec_to_pending(r)
                if r.mode is None:
                    join.append(r)
            if join:
                self.bc.apply(join=[(r, r.cache) for r in join])
                for r in join:
                    r.mode = "batch"
                    r.cache = None  # the batch row holds it now
                    r.feats, r.feat_p0 = [], self.bc.pos[self.bc.row(r)]
            t1 = time.perf_counter()
            self.stats["t_join"] += t1 - now
            if self.bspec and 2 <= len(decoding) <= self.bspec_max and all(r.spec_ok for r in decoding):
                self._bspec_step()
                self.stats["t_bspec"] += time.perf_counter() - t1
            else:
                self._batch_step()
                self.stats["t_batch"] += time.perf_counter() - t1
        self._last_decode = time.perf_counter()

    # ------------------------------------------------------------------ speculative mode
    def _start_spec(self, r, first_from_batch=False):
        import specdec
        from lookup import Lookup

        P = SS.cache_pos(r.cache)
        assert P == r.n + len(r.out) - 1, (P, r.n, len(r.out))
        held = r.ids + r.out[:-1]
        r.stepper = specdec.SpecStepper(self.model, self.drafter, r.cache, r.dcache, held, r.last,
                                        r.sp.max_new_tokens - len(r.out) + 1, self.T, r.stop_ids,
                                        Lookup() if self.lookup else None, emit=lambda toks, r=r: self._emit(r, toks),
                                        first=r.out[-1], **self.loop)
        r.mode = "spec"

    def _spec_to_pending(self, r):
        """Finalize r's speculative state (the caches hold ids + out[:-1]) so it can join the batch."""
        _, res = r.stepper.result(finalize=True)
        assert res["P"] == r.n + len(r.out) - 1, (res["P"], r.n, len(r.out))
        r.last = res["last_logits"]
        r.stepper = None
        r.mode = None
        r.switches += 1

    def _batch_to_spec(self, r):
        """Take r's row out of the batch (B = 1 copies) and continue speculatively."""
        r.cache = self.bc.row_cache(r)
        self._flush_feats(r)
        self.bc.apply(leave=[r])
        r.switches += 1
        self._start_spec(r)

    # ------------------------------------------------------------------ batched mode
    def _flush_feats(self, r):
        if r.feats and r.dcache is not None:
            self.drafter.materialize(r.dcache, mx.concatenate(r.feats, axis=0), r.feat_p0)
            mx.eval(r.dcache.state)
            r.feat_p0 += len(r.feats)
            r.feats = []

    def _build_step(self, rows, toks):
        """Build one batched decode step's graph (not evaluated): (logits, greedy, feats, the caches' new arrays)."""
        import nax_hook

        bc = self.bc
        B = bc.B
        capl = self.drafter.capture_layers if any(r.want_feats for r in rows) else ()
        with nax_hook.min_rows(self.nax_min_batch if B < self.nax_min_batch else 2):
            logits, feats = BT.forward(self.model, bc, toks, capture=capl)
        greedy = mx.argmax(logits[:, : self.vv], axis=-1).astype(mx.int32)
        return logits, greedy, feats, bc.state

    def _batch_step(self):
        """One decode step of every batch row. Pipelined when every row is greedy: the next step's graph is built
        (toks = this step's argmax, still on the GPU) while this one runs, and used if the batch is unchanged then."""
        bc = self.bc
        B = bc.B
        rows = list(bc.keys)
        for r in rows:
            r.want_feats = r.dcache is not None and (r.spec_ok or (self.cache is not None and self.cache.enabled))
        key = tuple(map(id, rows))
        pre = bc.take_prebuilt(key)
        if pre is None:
            pre = self._build_step(rows, mx.array([r.out[-1] for r in rows], dtype=mx.int32))
        logits, greedy, feats, state = pre
        outs = [logits, greedy, [] if feats is None else feats, state]  # logits first (batch.py's write order)
        if self.pipeline and all(r.pure_greedy and len(r.out) + 1 < r.sp.max_new_tokens for r in rows):
            mx.async_eval(*outs)
            bc.prebuild(key, lambda: self._build_step(rows, greedy))
        mx.eval(*outs)
        del outs, state, pre
        g = greedy.tolist()
        self.stats["steps_batch"] += 1
        self.stats["rows_batch"] += B
        for i, r in enumerate(rows):
            r.batch_steps += 1
            r.batch_rows += B
            r.last = logits[i]
            if r.want_feats:
                r.feats.append(feats[i : i + 1])
                if len(r.feats) >= FEATS_FLUSH:
                    self._flush_feats(r)
            t = g[i] if r.pure_greedy else r.sampler.choose(logits[i])
            self._emit(r, [t])
        done = [r for r in rows if r.finish]
        for r in done:
            self._finish(r, r.finish, leave=False)
        if done:
            self.bc.apply(leave=done)

    def _bspec_step(self):
        """One batched speculative step of the B batch rows (2..bspec_max, all greedy): each row's draft (prompt lookup,
        else the DFlash2 drafter on its own context), one verify forward of B x T rows (batch_spec), each row keeps its
        own accepted run; the drafter contexts get the kept rows' features."""
        import batch_spec
        from lookup import Lookup

        bc = self.bc
        bc.rollback()
        rows = list(bc.keys)
        T = max(2, min(self.T, self.bspec_rows // len(rows)))
        drafts = []
        for b, r in enumerate(rows):
            self._flush_feats(r)
            if self.lookup and (r.lk is None or r.lk_len != len(r.out)):  # (re)entering: a fresh lookup
                r.lk = Lookup()
                r.lk.reset(r.ids + r.out)
            P = bc.pos[b]
            bonus = mx.array(r.out[-1], dtype=mx.int32)
            hint = r.lk.propose() if r.lk is not None else []
            if len(hint) >= T - 1:
                d = mx.array(hint[: T - 1], dtype=mx.int32)
            else:
                d = self.drafter.propose(r.dcache, bonus, P)[: T - 1]
                if hint:
                    d = mx.concatenate([mx.array(hint, dtype=mx.int32), d[len(hint) :]])
            drafts.append(mx.concatenate([bonus.reshape(1), d]))
        inp = mx.stack(drafts)  # [B, T]
        mx.async_eval(inp)
        logits, cap, rec = batch_spec.verify_forward(self.model, bc, inp, capture=self.drafter.capture_layers)
        tok = mx.argmax(logits[:, :, : self.vv], axis=-1).astype(mx.int32)
        nacc = mx.cumprod((inp[:, 1:] == tok[:, :-1]).astype(mx.int32), axis=1).sum(axis=1)
        mx.eval(tok, nacc, inp, bc.state)
        ns = (nacc + 1).tolist()
        inp_l, tok_l = inp.tolist(), tok.tolist()
        kept, P0 = [], list(bc.pos)
        for b, r in enumerate(rows):
            n = ns[b]
            em = inp_l[b][1:n] + [tok_l[b][n - 1]]
            room = r.sp.max_new_tokens - len(r.out)
            em_k = em[: max(0, room)]
            stop = next((i for i, t in enumerate(em_k) if t in r.stop_ids), None)
            if stop is not None:
                em_k = em_k[: stop + 1]
            kept.append((em, em_k))
        # each row keeps its emitted run: the caches then hold ids + out[:-1] (its last token is the next input)
        batch_spec.commit(bc, rec, [max(1, len(k)) for _, k in kept])
        for b, r in enumerate(rows):
            nk = max(1, len(kept[b][1]))
            self.drafter.materialize(r.dcache, cap[b, :nk], P0[b])
        mx.async_eval(bc.state, [r.dcache.state for r in rows])
        self.stats["steps_bspec"] += 1
        self.stats["rows_batch"] += len(rows)
        for b, r in enumerate(rows):
            em, em_k = kept[b]
            r.batch_steps += 1
            r.batch_rows += len(rows)
            r.bspec_steps = getattr(r, "bspec_steps", 0) + 1
            r.bspec_tokens = getattr(r, "bspec_tokens", 0) + len(em_k)
            r.last = logits[b, len(em_k) - 1] if em_k else r.last
            r.feats, r.feat_p0 = [], bc.pos[b]
            self._emit(r, em_k)
            if r.lk is not None:
                r.lk.record(em)
                r.lk_len = len(r.out)
        done = [r for r in rows if r.finish]
        for r in done:
            self._finish(r, r.finish, leave=False)
        if done:
            self.bc.apply(leave=done)

    # ------------------------------------------------------------------ the end of a request
    def _finish(self, r, reason, leave=True):
        """End r: its reply snapshot, its batch row (leave=False: the caller removes the row), the done record."""
        if r.phase == "done":
            return
        t_fin = time.perf_counter()
        try:
            self._finish_inner(r, reason, leave)
        finally:
            self.stats["t_finish"] += time.perf_counter() - t_fin

    def _finish_inner(self, r, reason, leave):
        r.finish = r.finish or reason
        t_end = time.perf_counter()
        rec = r.rec
        if r.phase == "prefill":
            if r.gen is not None:
                r.gen.close()
            rec.update(cancelled="prefill", interval_snapshots=r.n_points, prefill_s=t_end - r.t_admit)
        else:
            tokens = np.concatenate([r.ids_np, np.asarray(r.out[:-1], np.int32)]) if len(r.out) > 1 else None
            t0s = time.perf_counter()
            if r.mode == "spec":
                _, res = r.stepper.result(finalize=True)
                if tokens is not None:
                    self._snap(r.session, "reply", tokens, r.cache, r.dcache, res["last_logits"])
                r.stepper = None
            elif r.mode == "batch":
                if tokens is not None and self.cache is not None and self.cache.enabled and \
                        not self.cache.has(tokens):
                    self._flush_feats(r)
                    self._snap_state(r.session, "reply", tokens, self.bc.capture_row(r, r.dcache, r.last))
                if leave:
                    self.bc.apply(leave=[r])
            rec["snapshot_s"] = time.perf_counter() - t0s
        r.phase = "done"
        r.cache = r.dcache = r.gen = None
        r.feats = []
        if r in self.active:
            self.active.remove(r)
        ttft = None if r.t_first is None else r.t_first - r.t_admit
        dec_s = None if r.t_first is None else t_end - r.t_first
        n = len(r.out)
        rec.update(mode=("spec" if r.batch_steps == 0 else "batch" if r.spec_steps == 0 else "mixed")
                   if n > 1 else "none", generated=n, finish=r.finish, spec_steps=r.spec_steps,
                   batch_steps=r.batch_steps, mean_batch=(r.batch_rows / r.batch_steps) if r.batch_steps else None,
                   bspec_steps=getattr(r, "bspec_steps", 0), bspec_tokens=getattr(r, "bspec_tokens", 0),
                   switches=r.switches, ttft_s=ttft, decode_s=dec_s,
                   decode_tok_s=((n - 1) / dec_s) if dec_s and n > 1 else None, total_s=t_end - r.t_admit,
                   peak_gb=mx.get_peak_memory() / 1e9, active_gb=mx.get_active_memory() / 1e9,
                   active_requests=len(self.active))
        if self.cache is not None:
            rec["cache_bytes"] = self.cache.nbytes
            rec["cache_snapshots"] = len(self.cache.snaps)
        rec["tokens"] = r.out
        self.on_done(r.job, r, rec)
        if not self.active:
            mx.clear_cache()
