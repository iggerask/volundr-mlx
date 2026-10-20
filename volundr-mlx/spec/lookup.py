"""Copy-from-context drafting (prompt lookup), a port of the engine's crates/agens-cli/src/lookup.rs.

The session's ids (prompt + everything emitted, the bonus token last) are indexed by bigram. Each step looks up the
longest earlier occurrence of the current suffix (the 24 most recent occurrences of the last bigram, extended
backwards up to 64) and proposes the up to 7 tokens that followed it. It is used instead of the drafter's block when
its expected accepted run beats the drafter's: running means of drafts accepted per step, kept for the drafter and
per match-length bucket, with the engine's priors. The means persist across prompts (they describe the drafter and
the workload); `reset` starts a new history.
"""
MAX_DRAFT, CANDIDATES, PRIOR_W, ALPHA = 7, 24, 4.0, 0.03
BUCKETS = (2, 3, 4, 6, 10, 16)
PRIORS = (0.6, 1.0, 1.8, 2.6, 3.6, 4.6)
DRAFTER_PRIOR = 2.0


class Mean:
    def __init__(self, v, n):
        self.v, self.n = v, n

    def add(self, x):
        self.n += 1.0
        self.v += max(1.0 / self.n, ALPHA) * (x - self.v)


def bucket(m):
    return max(i for i, b in enumerate(BUCKETS) if m >= b) if m >= BUCKETS[0] else 0


class Lookup:
    def __init__(self, on=True):
        self.on = on
        self.lk = [Mean(p, PRIOR_W) for p in PRIORS]
        self.dr = Mean(DRAFTER_PRIOR, PRIOR_W)
        self.used = self.steps = self.acc_lookup = self.acc_draft = 0
        self.reset([])

    def reset(self, ids):
        self.hist, self.idx, self.indexed, self.pending = list(ids), {}, 0, None

    def _index_upto(self, end):
        h = self.hist
        while self.indexed < end:
            p = self.indexed
            if p >= 1:
                self.idx.setdefault((h[p - 1], h[p]), []).append(p)
            self.indexed += 1

    def find(self, max_draft=MAX_DRAFT):
        h, n = self.hist, len(self.hist)
        if n < 3:
            return None
        self._index_upto(n - 1)  # never the current position itself
        cands = self.idx.get((h[n - 2], h[n - 1]))
        if not cands:
            return None
        best = None
        for p in reversed(cands[-CANDIDATES:]):
            if p + 1 >= n:
                continue
            m = 2
            while m < 64 and m <= p and h[p - m] == h[n - 1 - m]:
                m += 1
            if best is None or m > best[0]:
                best = (m, p)
        if best is None:
            return None
        m, p = best
        cont = h[p + 1 : min(p + 1 + max_draft, n)]
        return (cont, m) if cont else None

    def propose(self, long=0):
        """The copy proposal for this step if it is expected to beat the drafter, else [].
        long > MAX_DRAFT: a copy used with the full MAX_DRAFT tokens is returned with up to `long` tokens (the decision
        itself is the engine's: it sees at most MAX_DRAFT tokens)."""
        self.steps += 1
        self.pending = (False, 0)
        if not self.on:
            return []
        f = self.find(max(long, MAX_DRAFT))
        if f is None:
            return []
        cont, m = f
        b = bucket(m)
        if min(self.lk[b].v, len(cont[:MAX_DRAFT])) > self.dr.v:
            self.pending = (True, b)
            self.used += 1
            return cont if len(cont) > MAX_DRAFT else cont[:MAX_DRAFT]
        return []

    def record(self, emitted, cap=None):
        """The step's outcome: the emitted tokens (accepted drafts + the target's next token). cap: count at most that
        many accepted drafts in the running means (long copies: keep the means on the engine's 7-draft scale)."""
        acc = max(0, len(emitted) - 1)
        if cap is not None:
            acc = min(acc, cap)
        if self.pending is not None:
            used, b = self.pending
            if used:
                self.lk[b].add(acc)
                self.acc_lookup += acc
            else:
                self.dr.add(acc)
                self.acc_draft += acc
        self.pending = None
        self.hist.extend(emitted)
