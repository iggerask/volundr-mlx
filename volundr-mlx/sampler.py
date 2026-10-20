"""Token sampling for the plain (non-speculative) decode path, a port of upstream crates/agens-cli/src/sampler.rs.

Order as sglang's sampler (and the port): logit bias, suppressed ids, penalties (repetition / frequency / presence
over the generated tokens), min-new-tokens (stop ids masked), then greedy argmax, or temperature, top-k, top-p, min-p
and a draw from the renormalised distribution with xorshift64* (seeded by the request's seed). The reasoning budget
(VolundrReasoningBudgetProcessor) forces `<|/think|>` after N thinking tokens. Rows that need none of this take the
GPU argmax.
"""
import time

import mlx.core as mx
import numpy as np

MASK = (1 << 64) - 1
SAMPLING_EPS = 1e-6


class Budget:
    """Count tokens since `<|think|>` opened (in the prompt's last 16 tokens, or generated); after n of them with no
    `<|/think|>`, force `<|/think|>` and mask `<|think|>` for the rest of the request; with a soft bias b, add b to the
    `<|/think|>` logit from 80% of the budget on."""

    def __init__(self, n, soft, prompt, start, end):
        tail = list(prompt[-16:])
        self.start, self.end, self.n, self.soft = start, end, int(n), float(soft)
        self.open = None
        for i in range(len(tail) - 1, -1, -1):
            if tail[i] == start:
                if end not in tail[i + 1 :]:
                    self.open = 0
                break
        self.closed = self.forced = False

    def accept(self, j, t):
        if t == self.start:
            if self.open is None and not self.closed:
                self.open = j + 1
        elif t == self.end and self.open is not None:
            self.open = None
            self.closed = True

    def act(self, produced):
        """-> ("none" | "force" | "bias", payload) before sampling output token `produced`."""
        if self.closed:
            return ("bias", [(self.start, -np.inf)]) if self.forced else ("none", None)
        if self.open is None:
            return ("none", None)
        k = produced - self.open
        if k >= self.n:
            self.forced = self.closed = True
            self.open = None
            return ("force", self.end)
        if self.soft != 0.0 and k >= 0.8 * self.n:
            return ("bias", [(self.end, self.soft)])
        return ("none", None)


class Sampler:
    def __init__(self, sp, prompt=(), think_ids=(None, None), vocab_valid=248077, request_id=0):
        self.sp, self.vv = sp, vocab_valid
        seed = sp.seed if sp.seed is not None else time.time_ns() ^ ((request_id * 0x9E3779B97F4A7C15) & MASK)
        self.rng = (int(seed) | 1) & MASK
        self.counts = {}
        self.produced = 0
        self.budget = None
        if sp.reasoning_budget and think_ids[0] is not None and think_ids[1] is not None:
            self.budget = Budget(sp.reasoning_budget, sp.reasoning_budget_soft_bias, prompt, *think_ids)

    def uniform(self):  # xorshift64*
        r = self.rng
        r ^= r >> 12
        r ^= (r << 25) & MASK
        r ^= r >> 27
        self.rng = r
        return (((r * 0x2545F4914F6CDD1D) & MASK) >> 11) / float(1 << 53)

    def accept(self, tok):
        self.counts[tok] = self.counts.get(tok, 0) + 1
        if self.budget is not None:
            self.budget.accept(self.produced, tok)
        self.produced += 1

    def choose(self, logits) -> int:
        """The next token from a logits row (mx.array [vocab]); records it."""
        extra = []
        if self.budget is not None:
            kind, v = self.budget.act(self.produced)
            if kind == "force":
                self.accept(v)
                return v
            if kind == "bias":
                extra = v
        if extra or self.sp.needs_logits(self.vv, self.produced):
            tok = self.sample(np.array(logits.astype(mx.float32)), extra)
        else:
            tok = int(mx.argmax(logits[: self.vv]).item())
        self.accept(tok)
        return tok

    def sample(self, x, extra=()):
        sp = self.sp
        x = np.array(x, dtype=np.float32, copy=True)
        V = x.shape[0]
        for t, d in extra:
            if t < V:
                x[t] += d
        for t, b in sp.logit_bias:
            if t < V:
                x[t] += np.float32(b)
        sup = [t for t in sp.suppress_tokens if t < V]
        if sup:
            x[np.asarray(sup)] = -np.inf
        if self.counts:
            ids = np.fromiter(self.counts.keys(), dtype=np.int64)
            ids = ids[ids < V]
            c = np.asarray([self.counts[int(i)] for i in ids], dtype=np.float32)
            v = x[ids]
            if sp.repetition_penalty != 1.0:
                r = np.float32(sp.repetition_penalty)
                v = np.where(v > 0, v / r, v * r)
            v = v - (np.float32(sp.frequency_penalty) * c + np.where(c > 0, np.float32(sp.presence_penalty), 0))
            x[ids] = v.astype(np.float32)
        if self.produced < sp.min_new_tokens and sp.stop_token_ids:
            st = [t for t in sp.stop_token_ids if t < V]
            if st:
                x[np.asarray(st)] = -np.inf
        if sp.greedy:
            return int(np.argmax(x))  # first max
        t = np.float32(sp.temperature)
        idx = np.nonzero(np.isfinite(x))[0]
        if idx.size == 0:
            return int(np.argmax(x))
        k = min(sp.top_k if sp.top_k else idx.size, idx.size)
        k = max(k, 1)
        if k < idx.size:
            idx = idx[np.argpartition(-x[idx], k - 1)[:k]]
        elif idx.size > 4096:  # whole vocabulary: drop tokens below exp(-40) of the top one (< 1e-12 of the mass)
            m = x[idx].max()
            idx = idx[x[idx] / t >= m / t - 40.0]
        idx = idx[np.lexsort((idx, -x[idx]))]  # logit descending, ties to the lower id
        xs = (x[idx] / t).astype(np.float32)
        pr = np.exp((xs - xs[0]).astype(np.float64))
        pr /= pr.sum()
        keep = pr.size
        if sp.top_p < 1.0:
            hit = np.nonzero(np.cumsum(pr) >= sp.top_p)[0]
            if hit.size:
                keep = int(hit[0]) + 1
        if sp.min_p > 0.0:
            below = np.nonzero(pr < pr[0] * sp.min_p)[0]
            cnt = int(below[0]) if below.size else pr.size
            keep = min(keep, max(cnt, 1))
        z = pr[:keep].sum()
        u = self.uniform() * z
        j = int(np.searchsorted(np.cumsum(pr[:keep]), u, side="left"))
        return int(idx[min(j, keep - 1)])
