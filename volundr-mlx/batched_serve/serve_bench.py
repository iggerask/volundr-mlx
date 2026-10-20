"""Server-mode measurements through agens-server (../../serve.sh) and the batching worker (NOTES_BATCHED_SERVE.md).

  server  upstream's server-mode run: 19 short prompts (written here, not the spec suite) x 256 new tokens, greedy
          (temperature 0, ignore_eos), over the OpenAI API at concurrency 1, 2, 4, 8, 16 (a pool of c clients, each
          sending the next prompt as soon as its last one ends). Per level: aggregate tok/s (completion tokens / wall
          time from the first request sent to the last token), per-request decode tok/s, TTFT p50 / p95, the worker's
          MLX peak memory, its engine counters; the worker log's token ids (worker.py --log-tokens) per prompt.
          The session cache is cleared before each level, so every level prefills its prompts.
  mixed   one long agent session (session cache on) and four short streams at once: the short streams decode while
          the agent's next turn prefills its new tokens from the reply snapshot; their inter-token gaps during that
          prefill are the stall. Run twice: prefill layers in slices of --slice-ms (the default), and whole chunks
          between decode steps (upstream's schedule, slice_ms set huge through the worker's config message).
  compare the server run's completions at each concurrency against concurrency 1 (worker log token ids)
  ties    for each completion that differs from its concurrency-1 run: the logits of both runs' tokens at the first
          difference, teacher-forced along the common prefix at B = 1 (loads the model; run with the server stopped)

usage (batched_serve/serve_bench.sh runs server + mixed under the GPU lock):
  python batched_serve/serve_bench.py server --port P --worker-log W --out batched_serve/server.jsonl
  python batched_serve/serve_bench.py mixed --port P --worker-log W --out batched_serve/mixed.jsonl
  python batched_serve/serve_bench.py compare --worker-log W --run batched_serve/server.jsonl
  python batched_serve/serve_bench.py ties --worker-log W --run batched_serve/server.jsonl --out ...
"""
import argparse
import http.client
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from session_e2e import FILES, SYSTEM, TOOLS, assistant_message, file_text, wq  # noqa: E402

PY = os.environ.get("PY", sys.executable)

# 19 short prompts, written for this run
PROMPTS = [
    "Explain in plain words how a hash map handles collisions, and when it needs to resize.",
    "Write a Python function that merges two sorted lists into one sorted list, with a short docstring.",
    "What are three practical differences between TCP and UDP? Give an example application for each.",
    "Plan a three-day walking trip in Lisbon for someone who likes architecture and good coffee.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps.",
    "Summarise the plot of a classic heist film in five sentences, without naming the film.",
    "Write a short, friendly email asking a colleague to review a pull request by Friday.",
    "Give a JSON object describing a book with title, author, year, genres (a list) and a rating out of 5.",
    "What does the Rust borrow checker prevent? Explain with a small example of code it rejects.",
    "List the steps to make a basic sourdough loaf, from feeding the starter to baking.",
    "Translate into French and German: 'The meeting is moved to Thursday afternoon because of the holiday.'",
    "Why is the sky blue during the day and red at sunset? Keep it under 150 words.",
    "Write a SQL query that returns the ten customers with the highest total order value in 2025.",
    "Compare renting and buying a flat in a big city: list the main financial trade-offs.",
    "Describe how a binary search works and what its time complexity is, then give pseudocode.",
    "Write a haiku sequence of three poems about the first snow in a city.",
    "What is the difference between a process and a thread? When would you use each?",
    "Give me five interview questions for a junior data analyst and what a good answer covers.",
    "Explain what a Kalman filter does to someone who knows basic statistics but not control theory.",
]


def stream(port, body, session=None, host="127.0.0.1", into=None):
    """One streamed chat completion. -> dict(t0, arrivals (perf_counter of every delta with text), usage, finish).
    into: a dict to fill while the stream runs (another thread can watch it)."""
    conn = http.client.HTTPConnection(host, port, timeout=3600)
    hdr = {"Content-Type": "application/json"}
    if session:
        hdr["X-Session-Id"] = session
    r = into if into is not None else {}
    r.update({"t0": time.perf_counter(), "arrivals": [], "usage": None, "finish": None, "content": "",
              "reasoning": "", "tool_calls": {}})
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), hdr)
        resp = conn.getresponse()
        if resp.status != 200:
            r["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return r
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                r["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                got = False
                for k, f in (("content", "content"), ("reasoning_content", "reasoning")):
                    if d.get(k):
                        r[f] += d[k]
                        got = True
                for tc in d.get("tool_calls") or []:
                    e = r["tool_calls"].setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    e["id"] = tc.get("id") or e["id"]
                    fn = tc.get("function") or {}
                    e["name"] += fn.get("name") or ""
                    e["arguments"] += fn.get("arguments") or ""
                    got = True
                if got:
                    r["arrivals"].append(time.perf_counter())
                if ch.get("finish_reason"):
                    r["finish"] = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        r["error"] = repr(e)
    finally:
        conn.close()
    r["t_end"] = time.perf_counter()
    r["tool_calls"] = [r["tool_calls"][k] for k in sorted(r["tool_calls"])]
    return r


def body_for(text, max_tokens, ignore_eos=True):
    return {"model": "volundr", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"enable_thinking": False}}


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def canary():
    try:
        out = subprocess.run([PY, str(ROOT.parent / "agens-mlx-bench/canary.py")], capture_output=True, text=True,
                             timeout=120).stdout
        return next((ln for ln in out.splitlines() if "canary" in ln), "").strip()
    except Exception as e:  # noqa: BLE001
        return f"canary failed: {e}"


def worker_records(path, t_wall):
    recs = []
    if path and Path(path).exists():
        for ln in Path(path).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("t", 0) >= t_wall:
                    recs.append(r)
    return recs


def run_level(port, c, idx, max_tokens):
    """Prompts idx through a pool of c concurrent clients. -> (results by prompt index, wall start, wall end)."""
    q = queue.Queue()
    for i in idx:
        q.put(i)
    res = {}

    def client():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            res[i] = stream(port, body_for(PROMPTS[i], max_tokens))

    t0 = time.perf_counter()
    ths = [threading.Thread(target=client) for _ in range(c)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return res, t0, time.perf_counter()


def summarize(res, t0, t1, c=0):
    toks, ttft, dec = 0, [], []
    for r in res.values():
        n = (r.get("usage") or {}).get("completion_tokens") or len(r["arrivals"])
        toks += n
        if r["arrivals"]:
            ttft.append(r["arrivals"][0] - r["t0"])
            if n > 1 and r["arrivals"][-1] > r["arrivals"][0]:
                dec.append((n - 1) / (r["arrivals"][-1] - r["arrivals"][0]))
    end = max((r["arrivals"][-1] for r in res.values() if r["arrivals"]), default=t1)
    # the full-batch window: from the c-th first token to the first request's last token (every client streaming)
    firsts = sorted(r["arrivals"][0] for r in res.values() if r["arrivals"])
    lasts = sorted(r["arrivals"][-1] for r in res.values() if r["arrivals"])
    full = None
    if c and len(firsts) >= c and lasts[0] > firsts[c - 1]:
        lo, hi = firsts[c - 1], lasts[0]
        n_in = sum(1 for r in res.values() for t in r["arrivals"] if lo <= t <= hi)
        full = round(n_in / (hi - lo), 2)
    return {"requests": len(res), "completion_tokens": toks, "wall_s": round(end - t0, 2), "agg_full_tok_s": full,
            "agg_tok_s": round(toks / (end - t0), 2), "per_request_tok_s_mean": round(sum(dec) / len(dec), 2),
            "per_request_tok_s_p50": round(pct(dec, 50), 2), "ttft_p50_s": round(pct(ttft, 50), 3),
            "ttft_p95_s": round(pct(ttft, 95), 3), "ttft_max_s": round(max(ttft), 3),
            "errors": sum(1 for r in res.values() if r.get("error"))}


def cmd_server(a):
    out = open(a.out, "a")

    def log(rec):
        print(json.dumps({k: v for k, v in rec.items() if k != "per_prompt"}), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()

    levels = [int(x) for x in a.levels.split(",")]
    # warm-up: batched kernels at 2..8 rows and the spec path, untimed
    run_level(a.port, 8, list(range(8)), 24)
    run_level(a.port, 1, [8], 24)
    for c in levels:
        can0 = canary() if a.canary else None
        wq({"type": "clear"}, a.socket)  # every level prefills its prompts (the prefix cache would hold them)
        wq({"type": "reset_peak"}, a.socket)
        t_wall = time.time()
        res, t0, t1 = run_level(a.port, c, list(range(len(PROMPTS))), a.max_tokens)
        st = wq({"type": "stats"}, a.socket)
        time.sleep(0.5)
        recs = worker_records(a.worker_log, t_wall)
        s = summarize(res, t0, t1, c)
        modes = {}
        for r in recs:
            modes[r.get("mode")] = modes.get(r.get("mode"), 0) + 1
        per = {i: {"finish": res[i]["finish"], "completion_tokens": (res[i].get("usage") or {}).get("completion_tokens"),
                   "ttft_s": (res[i]["arrivals"][0] - res[i]["t0"]) if res[i]["arrivals"] else None} for i in res}
        rec = {"kind": "level", "concurrency": c, **s, "peak_gb": round(st.get("peak_gb") or 0, 2),
               "engine": st.get("engine"), "worker_records": len(recs), "modes": modes,
               "mean_batch": round(sum((r.get("mean_batch") or 1) * r.get("batch_steps", 0) for r in recs) /
                                   max(1, sum(r.get("batch_steps", 0) for r in recs)), 2),
               "canary_before": can0, "canary_after": canary() if a.canary else None, "t_wall": t_wall,
               "t_end": time.time(),
               "per_prompt": per}
        log(rec)


def cmd_compare(a):
    """Token ids per prompt and level from the worker log; first difference against concurrency 1."""
    levels = [json.loads(ln) for ln in Path(a.run).read_text().splitlines() if ln.strip()]
    levels = [lv for lv in levels if lv.get("kind") == "level"]
    allrecs = [json.loads(ln) for ln in Path(a.worker_log).read_text().splitlines() if ln.strip()]
    by_level = {}
    for i, lv in enumerate(levels):
        t_next = levels[i + 1]["t_wall"] if i + 1 < len(levels) else lv.get("t_end", 1e18)
        recs = [r for r in allrecs if lv["t_wall"] <= r.get("t", 0) < t_next and r.get("prompt_ids")]
        by_level[lv["concurrency"]] = {tuple(r["prompt_ids"]): r for r in recs}
    base = by_level.get(1)
    rows = []
    for c, recs in by_level.items():
        same = 0
        divs = []
        for p, r in recs.items():
            b = base.get(p) if base else None
            if b is None:
                continue
            x, y = b["tokens"], r["tokens"]
            d = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), None)
            if d is None and len(x) == len(y):
                same += 1
            else:
                divs.append({"prompt_ids": list(p), "diverge_at": d, "a": x[d] if d is not None else None,
                             "b": y[d] if d is not None else None, "prefix": x[:d] if d is not None else None,
                             "mode_1": b.get("mode"), "mode_c": r.get("mode")})
        rows.append({"kind": "compare", "concurrency": c, "requests": len(recs), "identical_to_c1": same,
                     "diverged": divs})
        print(json.dumps({"concurrency": c, "requests": len(recs), "identical_to_c1": same,
                          "diverge_at": sorted(d["diverge_at"] for d in divs if d["diverge_at"] is not None)}))
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def cmd_ties(a):
    """Teacher-forced logits of both tokens at each first difference (B = 1, the model loaded here)."""
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "spec"))
    import session as S
    import specdec

    model, drafter, _ = specdec.load(4, 4, True)
    vv = model.args.vocab_valid
    rows = [json.loads(ln) for ln in Path(a.compare).read_text().splitlines() if ln.strip()]
    out = open(a.out, "w") if a.out else None
    seen = {}
    for row in rows:
        for d in row["diverged"]:
            if d["diverge_at"] is None:
                continue
            key = (tuple(d["prompt_ids"]), d["diverge_at"], d["a"], d["b"])
            if key not in seen:
                c = model.make_cache()
                lg = S.prefill(model, None, c, None, d["prompt_ids"])
                for t in d["prefix"]:
                    lg, _ = S.decode_step(model, c, t)
                mx.eval(lg)
                lf = lg.astype(mx.float32)
                top = mx.topk(lf[:vv], 2)
                seen[key] = {"logit_c1_token": float(lf[d["a"]].item()), "logit_other": float(lf[d["b"]].item()),
                             "top2_margin": float((top.max() - top.min()).item())}
                del c
                mx.clear_cache()
            rec = {"concurrency": row["concurrency"], "diverge_at": d["diverge_at"], "modes": [d["mode_1"],
                   d["mode_c"]], **seen[key]}
            print(json.dumps(rec), flush=True)
            if out:
                out.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------- mixed
AGENT_Q = ("Review how speculative decoding continues from a cached state in this repository. The files are below. "
           "Answer with the three riskiest places and why, briefly.")


def wait_record(path, t_wall, session, timeout=15.0):
    """The worker log's record of `session` written after t_wall (the worker writes it just after the last token)."""
    end = time.time() + timeout
    while True:
        recs = [r for r in worker_records(path, t_wall) if r.get("session_id") == session]
        if recs or time.time() > end:
            return recs[-1] if recs else None
        time.sleep(0.1)


def stall_stats(arr, lo, hi):
    """A stream's inter-token gaps against the window [lo, hi] (the agent's prefill): the longest gap overlapping it,
    tokens and rate inside it."""
    gaps = [(a, b) for a, b in zip(arr, arr[1:])]
    over = [b - a for a, b in gaps if b >= lo and a <= hi]
    n_in = sum(1 for t in arr if lo <= t <= hi)
    return (max(over) if over else None), n_in, n_in / max(1e-9, hi - lo)


def cmd_mixed(a):
    out = open(a.out, "a")

    def log(rec):
        print(json.dumps({k: v for k, v in rec.items() if k not in ("arrivals",)}), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()

    sid = "agent-mixed"
    first_files = FILES[:4]
    text = AGENT_Q + "\n\n" + "\n\n".join(f"{f}:\n```python\n{file_text(f)}\n```" for f in first_files)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    agent_body = lambda m: {"model": "volundr", "messages": m, "tools": TOOLS, "max_tokens": a.agent_tokens,
                            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                            "chat_template_kwargs": {"enable_thinking": False}}
    # turn 1 alone: the long prefill, the reply snapshot
    t_wall = time.time()
    r1 = stream(a.port, agent_body(msgs), session=sid)
    w1 = wait_record(a.worker_log, t_wall, sid)
    log({"kind": "agent_turn1_alone", "ttft_s": r1["arrivals"][0] - r1["t0"] if r1["arrivals"] else None,
         "worker": {k: w1.get(k) for k in ("prompt_tokens", "cached_tokens", "prefilled_tokens", "ttft_s",
                                           "decode_tok_s", "mode")} if w1 else None})
    base_msgs = msgs + [assistant_message({"content": r1["content"], "reasoning": r1["reasoning"],
                                           "tool_calls": r1["tool_calls"]})]
    # turn 2, twice from turn 1's reply snapshot (a different file of the same size each time, ~3.5K new tokens)
    for slice_ms, f in ((None, "kernels.py"), (1e7, "decode_fused.py")):
        if slice_ms is not None:
            wq({"type": "config", "slice_ms": slice_ms}, a.socket)
        m2 = base_msgs + [{"role": "user", "content": f"Here is {f} as well:\n```python\n{file_text(f)}\n```\n"
                                                      "Update your answer."}]
        short, ths = {}, []

        for j in range(a.short):
            def go(j=j):
                stream(a.port, body_for(PROMPTS[j], a.short_tokens), into=short.setdefault(j, {}))
            th = threading.Thread(target=go)
            th.start()
            ths.append(th)
        # wait until every short stream decodes (stream() fills its record while it runs), then a steady stretch
        deadline = time.perf_counter() + 300
        while time.perf_counter() < deadline and not (len(short) == a.short and
                                                      all(s["arrivals"] for s in short.values())):
            time.sleep(0.05)
        time.sleep(a.lead_s)
        t_wall = time.time()
        t_agent = time.perf_counter()
        ra = stream(a.port, agent_body(m2), session=sid)
        for th in ths:
            th.join()
        wa = wait_record(a.worker_log, t_wall, sid)
        if slice_ms is not None:
            wq({"type": "config", "slice_ms": a.slice_ms}, a.socket)
        t_af = ra["arrivals"][0] if ra.get("arrivals") else t_agent
        rows = []
        for j, s_ in sorted(short.items()):
            arr = s_["arrivals"]
            gmax, n_in, rate_in = stall_stats(arr, t_agent, t_af)
            pre = [t for t in arr if t < t_agent]
            rate_pre = (len(pre) - 1) / (pre[-1] - pre[0]) if len(pre) > 2 else None
            after = [t for t in arr if t_af + 1.0 < t]
            rate_after = (len(after) - 1) / (after[-1] - after[0]) if len(after) > 2 else None
            big = [(round(x - t_agent, 2), round(y - x, 3)) for x, y in zip(arr, arr[1:]) if y - x > 0.25]
            rows.append({"stream": j, "tokens": len(arr), "max_gap_during_prefill_s": round(gmax, 3) if gmax else None,
                         "tokens_during_prefill": n_in, "tok_s_during_prefill": round(rate_in, 2),
                         "tok_s_before": round(rate_pre, 2) if rate_pre else None,
                         "tok_s_after_join": round(rate_after, 2) if rate_after else None,
                         "gaps_over_250ms": big})
        log({"kind": "mixed", "schedule": "layer slices" if slice_ms is None else "whole prefill",
             "slice_ms": a.slice_ms if slice_ms is None else slice_ms, "file": f,
             "agent_ttft_s": round(t_af - t_agent, 2),
             "agent_worker": {k: wa.get(k) for k in ("prompt_tokens", "cached_tokens", "prefilled_tokens", "ttft_s",
                                                     "decode_tok_s", "mode", "mean_batch", "snapshot")} if wa else None,
             "short": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["server", "mixed", "compare", "ties"])
    ap.add_argument("--port", type=int, default=30123)
    ap.add_argument("--socket", default=None)
    ap.add_argument("--worker-log", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--short", type=int, default=4)
    ap.add_argument("--short-tokens", type=int, default=384)
    ap.add_argument("--agent-tokens", type=int, default=96)
    ap.add_argument("--lead-s", type=float, default=3.0)
    ap.add_argument("--slice-ms", type=float, default=200.0, help="the worker's slice_ms to restore")
    a = ap.parse_args()
    {"server": cmd_server, "mixed": cmd_mixed, "compare": cmd_compare, "ties": cmd_ties}[a.cmd](a)


if __name__ == "__main__":
    main()
