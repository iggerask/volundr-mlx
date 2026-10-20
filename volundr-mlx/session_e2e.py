"""End-to-end measurement of the session cache through agens-server (../serve.sh): a multi-turn agent session over
the OpenAI API (system prompt + tool definitions, tool results appended each turn, growing past ~20K tokens), its
per-turn time to first token and decode speed with the cache and without, the same conversation's greedy replies
with and without the cache, and concurrency checks (interleaved sessions, eviction under a small budget,
cancellation mid-stream, mid-prefill and while queued). The texts are this repository's files.

usage (server on --port, worker on --socket; session_cache/e2e.sh runs all of it under the GPU lock):
  python session_e2e.py run    --out session_cache/e2e/cached.json        live conversation, cache on
  python session_e2e.py replay --transcript session_cache/e2e/cached.json --out session_cache/e2e/nocache.json
                                                                           same requests, cache off (budget 0)
  python session_e2e.py checks --out session_cache/e2e/checks.json
  python session_e2e.py report session_cache/e2e/cached.json session_cache/e2e/nocache.json
"""
import argparse
import http.client
import json
import os
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SYSTEM = (
    "You are Smith, a careful code-review agent working in the volundr-mlx repository: an MLX port of a hybrid "
    "language model (linear-attention, sparse-attention and dense layers) for Apple silicon, with speculative "
    "decoding and a session cache. You work in turns. Each turn, call exactly one tool to read what you need next, "
    "and keep your thinking short. Track what you have learned: data structures, invariants, where state lives, and "
    "what could go wrong. When you have read enough, write the review: what the code does, the risks you see "
    "(correctness first, then performance), and concrete suggestions with file and function names. Never invent "
    "file contents; read them with the tools."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository (path relative to volundr-mlx/).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"},
                                                        "path": {"type": "string", "description": "directory"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]

FIRST = ("Review how speculative decoding continues from a cached state in this repository: the spec loop, the "
         "verify/commit step, the drafter context and the session snapshots. Start with spec/specdec.py.")
FILES = ["spec/specdec.py", "spec/verify.py", "session.py", "spec/dflash.py", "spec/lookup.py", "worker.py",
         "sampler.py", "generate.py", "spec/nax_hook.py"]
FILE_CHARS = 13000


def worker_socket():
    return os.environ.get("VOLUNDR_WORKER_SOCKET") or os.path.join(tempfile.gettempdir(), "volundr-mlx-worker.sock")


def wq(msg, sock_path=None):
    """One request/response with the worker."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path or worker_socket())
    body = json.dumps(msg).encode()
    s.sendall(struct.pack(">I", len(body)) + body)
    h = b""
    while len(h) < 4:
        h += s.recv(4 - len(h))
    (n,) = struct.unpack(">I", h)
    buf = b""
    while len(buf) < n:
        buf += s.recv(n - len(buf))
    s.close()
    return json.loads(buf)


def last_record():
    return (wq({"type": "stats"}).get("last") or [None])[-1]


class Chat:
    """Streaming /v1/chat/completions over http.client (so a request can be dropped mid-stream)."""

    def __init__(self, port, host="127.0.0.1"):
        self.host, self.port = host, port

    def stream(self, body, session=None, stop_after=None, close_after_s=None):
        """-> dict(content, reasoning, tool_calls, ttft_s, t_first, t_last, usage, finish, n_deltas, closed_early)."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=3600)
        hdr = {"Content-Type": "application/json"}
        if session:
            hdr["X-Session-Id"] = session
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body), hdr)
        if close_after_s is not None:
            threading.Timer(close_after_s, lambda: _hard_close(conn)).start()
        r = {"content": "", "reasoning": "", "tool_calls": {}, "ttft_s": None, "t_first": None, "t_last": None,
             "usage": None, "finish": None, "n_deltas": 0, "closed_early": False, "t0": t0}
        try:
            resp = conn.getresponse()
            if resp.status != 200:
                r["error"] = f"HTTP {resp.status}: {resp.read()[:500]!r}"
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
                    if d.get("content"):
                        r["content"] += d["content"]
                        got = True
                    if d.get("reasoning_content"):
                        r["reasoning"] += d["reasoning_content"]
                        got = True
                    for tc in d.get("tool_calls") or []:
                        e = r["tool_calls"].setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        e["id"] = tc.get("id") or e["id"]
                        f = tc.get("function") or {}
                        e["name"] += f.get("name") or ""
                        e["arguments"] += f.get("arguments") or ""
                        got = True
                    if got:
                        now = time.perf_counter()
                        if r["t_first"] is None:
                            r["t_first"], r["ttft_s"] = now, now - t0
                        r["t_last"] = now
                        r["n_deltas"] += 1
                    if ch.get("finish_reason"):
                        r["finish"] = ch["finish_reason"]
                if stop_after is not None and r["n_deltas"] >= stop_after:
                    r["closed_early"] = True
                    break
        except (OSError, http.client.HTTPException) as e:
            r["closed_early"] = True
            r["error"] = repr(e)
        finally:
            _hard_close(conn)
        r["tool_calls"] = [r["tool_calls"][k] for k in sorted(r["tool_calls"])]
        return r


def _hard_close(conn):
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def assistant_message(r):
    m = {"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]}
    if r["tool_calls"]:
        m["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                           for i, c in enumerate(r["tool_calls"])]
    return m


def file_text(rel):
    t = (HERE / rel).read_text()
    return t if len(t) <= FILE_CHARS else t[:FILE_CHARS] + f"\n... [truncated: {len(t) - FILE_CHARS} more characters]"


def requested_path(call):
    try:
        a = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        return None
    p = str(a.get("path") or "").strip().lstrip("./")
    if p.startswith("volundr-mlx/"):
        p = p[len("volundr-mlx/"):]
    f = HERE / p
    if call["name"] == "read_file" and p.endswith(".py") and f.is_file() and ".." not in p:
        return p
    return None


def turn_stats(r, w):
    dec = None
    if r["t_first"] and r["t_last"] and r["usage"] and r["usage"].get("completion_tokens", 0) > 1:
        dec = (r["usage"]["completion_tokens"] - 1) / max(1e-9, r["t_last"] - r["t_first"])
    out = {"prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
           "completion_tokens": (r["usage"] or {}).get("completion_tokens"), "ttft_client_s": r["ttft_s"],
           "decode_tok_s_client": dec, "finish": r["finish"], "tool_calls": [c["name"] for c in r["tool_calls"]]}
    if w:
        for k in ("cached_tokens", "prefilled_tokens", "ttft_s", "decode_tok_s", "tokens_per_step", "cache_bytes",
                  "cache_snapshots", "peak_gb", "active_gb", "prefill_s", "restore_s", "snapshot_s", "queue_s",
                  "mode", "generated", "interval_snapshots"):
            out["w_" + k] = w.get(k)
        out["w_snapshot"] = w.get("snapshot")
    return out


def cmd_run(a):
    chat = Chat(a.port)
    sid = a.session
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST}]
    served, turns = [], []
    queue = list(FILES)
    for t in range(a.max_turns):
        body = {"model": "volundr", "messages": msgs, "tools": TOOLS, "max_tokens": a.max_tokens, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, "reasoning_effort": "low"}
        r = chat.stream(body, session=sid)
        if r.get("error"):
            raise SystemExit(f"turn {t}: {r['error']}")
        w = last_record()
        st = turn_stats(r, w)
        st["turn"] = t
        turns.append({"body": json.loads(json.dumps(body)), "reply": {k: r[k] for k in ("content", "reasoning",
                      "tool_calls", "finish")}, "stats": st})
        print(json.dumps(st), flush=True)
        if (st["prompt_tokens"] or 0) >= a.target:
            break
        msgs = msgs + [assistant_message(r)]
        call = r["tool_calls"][0] if r["tool_calls"] else None
        want = requested_path(call) if call else None
        if want is None or want in served:
            want = next((f for f in queue if f not in served), None)
        if want is None:
            break
        served.append(want)
        text = file_text(want)
        if call:
            msgs.append({"role": "tool", "tool_call_id": call["id"] or "call_0",
                         "content": f"{want}:\n```python\n{text}\n```"})
        else:
            msgs.append({"role": "user", "content": f"Continue with one tool call per turn. Here is {want}:\n"
                                                    f"```python\n{text}\n```"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"session": sid, "turns": turns, "cache": wq({"type": "stats"})["cache"]},
                                      indent=1))


def cmd_replay(a):
    """The recorded requests again (cache disabled with budget 0 unless --keep-cache), replies compared."""
    chat = Chat(a.port)
    src = json.loads(Path(a.transcript).read_text())
    if not a.keep_cache:
        print(wq({"type": "budget", "gb": 0}), flush=True)
    turns = []
    for t, rec in enumerate(src["turns"]):
        r = chat.stream(rec["body"], session=src["session"] + "-replay")
        w = last_record()
        st = turn_stats(r, w)
        st["turn"] = t
        ref = rec["reply"]
        st["same_reply"] = (r["content"] == ref["content"] and r["reasoning"] == ref["reasoning"]
                            and [(c["name"], c["arguments"]) for c in r["tool_calls"]]
                            == [(c["name"], c["arguments"]) for c in ref["tool_calls"]])
        if not st["same_reply"]:
            x, y = ref["reasoning"] + "␞" + ref["content"], r["reasoning"] + "␞" + r["content"]
            st["first_diff_char"] = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            st["diff_context"] = [x[max(0, st["first_diff_char"] - 40): st["first_diff_char"] + 40],
                                  y[max(0, st["first_diff_char"] - 40): st["first_diff_char"] + 40]]
        turns.append({"reply": {k: r[k] for k in ("content", "reasoning", "tool_calls", "finish")}, "stats": st})
        print(json.dumps(st), flush=True)
    Path(a.out).write_text(json.dumps({"turns": turns}, indent=1))


# ----------------------------------------------------------------------------- checks
def _conv(system, text, q):
    return [{"role": "system", "content": system}, {"role": "user", "content": f"{text}\n\n{q}"}]


def _body(msgs, max_tokens, think=False):
    b = {"model": "volundr", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}}
    if not think:
        b["chat_template_kwargs"] = {"enable_thinking": False}
    return b


def _records_since(t):
    return [r for r in wq({"type": "stats"})["last"] if r.get("t", 0) >= t]


def cmd_checks(a):
    chat = Chat(a.port)
    res = {}
    # 1. two sessions interleaved, then concurrent
    convs = {"A": _conv("You review Python code. Be brief.", file_text("sampler.py")[:7000], "What does Budget do?"),
             "B": _conv("You explain build scripts. Be brief.", (HERE.parent / "serve.sh").read_text(),
                        "What does this script start?")}
    seq = []
    for turn in range(2):
        for s in ("A", "B"):
            t = time.time()
            r = chat.stream(_body(convs[s], 48), session=s)
            w = _records_since(t)[-1]
            seq.append({"session": s, "turn": turn, "prompt": w["prompt_tokens"], "cached": w["cached_tokens"],
                        "snapshot": w["snapshot"], "ttft_s": w["ttft_s"]})
            convs[s] = convs[s] + [assistant_message(r), {"role": "user", "content": "Say more, in two sentences."}]
    out = {}

    def go(s):
        out[s] = chat.stream(_body(convs[s], 48), session=s)

    t = time.time()
    th = [threading.Thread(target=go, args=(s,)) for s in ("A", "B")]
    for x in th:
        x.start()
        time.sleep(0.2)
    for x in th:
        x.join()
    conc = [{"session": r.get("session_id"), "prompt": r["prompt_tokens"], "cached": r["cached_tokens"],
             "snapshot": r["snapshot"], "queue_s": r["queue_s"], "ttft_s": r["ttft_s"]} for r in _records_since(t)]
    ok_inter = all(x["snapshot"] and x["snapshot"]["session"] == x["session"] for x in seq if x["turn"] == 1)
    ok_conc = len(conc) == 2 and all(x["snapshot"] and x["snapshot"]["session"] == x["session"] for x in conc) and \
        max(x["queue_s"] for x in conc) > 0.5
    res["interleaved"] = {"sequential": seq, "concurrent": conc, "pass": ok_inter and ok_conc}
    print("interleaved:", json.dumps(res["interleaved"]), flush=True)

    # 2. eviction under a small budget
    budget = a.small_budget
    print(wq({"type": "budget", "gb": budget}), flush=True)
    ev = []
    texts = {"C": "spec/verify.py", "D": "spec/dflash.py", "E": "decode_fused.py"}
    cc = {}
    for s, f in texts.items():
        cc[s] = _conv("You summarize code.", file_text(f), "Summarize in one sentence.")
        t = time.time()
        r = chat.stream(_body(cc[s], 24), session=s)
        cc[s] = cc[s] + [assistant_message(r), {"role": "user", "content": "And its main risk?"}]
        st = wq({"type": "stats"})["cache"]
        ev.append({"session": s, "prompt": _records_since(t)[-1]["prompt_tokens"], "cache_bytes": st["bytes"],
                   "snapshots": st["snapshots"], "evicted": st["evicted"],
                   "kept": [(i["session"], i["kind"], i["pos"]) for i in st["items"]]})
    t = time.time()
    chat.stream(_body(cc["E"], 24), session="E")  # E's are the newest: kept
    we = _records_since(t)[-1]
    t = time.time()
    chat.stream(_body(cc["C"], 24), session="C")  # C's snapshots are the least recently used: evicted
    wc = _records_since(t)[-1]
    res["eviction"] = {"budget_gb": budget, "steps": ev, "C_again": {"cached": wc["cached_tokens"],
                       "prompt": wc["prompt_tokens"], "snapshot": wc["snapshot"]},
                       "E_again": {"cached": we["cached_tokens"], "prompt": we["prompt_tokens"],
                                   "snapshot": we["snapshot"]},
                       "pass": all(e["cache_bytes"] <= budget * 1e9 for e in ev) and ev[-1]["evicted"] > 0
                       and (wc["snapshot"] is None or wc["snapshot"]["session"] != "C")
                       and we["snapshot"] is not None and we["snapshot"]["session"] == "E"}
    print("eviction:", json.dumps(res["eviction"]), flush=True)
    print(wq({"type": "budget", "gb": a.budget}), flush=True)

    # 3. cancellation: mid-stream, mid-prefill, while queued
    canc = {}
    msgs = _conv("You write long, detailed explanations.", file_text("spec/lookup.py"),
                 "Explain this file line by line, in detail.")
    t = time.time()
    r = chat.stream(_body(msgs, 1500), session="F", stop_after=30)
    r2 = chat.stream(_body(_conv("Be brief.", "Hello.", "Say hi."), 8), session="G")
    recs = _records_since(t)
    f = next(x for x in recs if x.get("session_id") == "F")
    g = next(x for x in recs if x.get("session_id") == "G")
    canc["mid_stream"] = {"deltas_read": r["n_deltas"], "F_finish": f["finish"], "F_generated": f["generated"],
                          "G_queue_s": g["queue_s"], "G_ttft_client_s": r2["ttft_s"],
                          "pass": f["finish"] == "cancelled" and f["generated"] < 200 and g["queue_s"] < 2.0}
    big = "\n\n".join(file_text(x) for x in ("kernels.py", "volundr.py", "convert.py", "bcsa_verify.py"))
    msgs = _conv("You are a meticulous Metal reviewer. " + str(time.time()), big, "List every kernel.")
    t = time.time()
    r = chat.stream(_body(msgs, 64), session="H", close_after_s=6.0)
    t_close = time.perf_counter()
    r2 = chat.stream(_body(_conv("Be brief.", "Hello again.", "Say hi."), 8), session="I")
    recs = _records_since(t)
    h = next(x for x in recs if x.get("session_id") == "H")
    i = next(x for x in recs if x.get("session_id") == "I")
    canc["mid_prefill"] = {"H_prompt": h["prompt_tokens"], "H_finish": h["finish"], "H_cancelled": h.get("cancelled"),
                           "H_cancelled_at": h.get("cancelled_at"), "H_total_s": h.get("total_s"),
                           "I_queue_s": i["queue_s"], "I_ttft_after_close_s": (r2["t_first"] - t_close)
                           if r2["t_first"] else None, "interval_snapshots": h.get("interval_snapshots"),
                           "pass": h["finish"] == "cancelled" and h.get("cancelled") == "prefill"
                           and h.get("cancelled_at", 1 << 30) < h["prompt_tokens"]}
    big2 = "\n\n".join(file_text(x) for x in ("spec/specdec.py", "spec/verify.py", "session.py"))
    long_msgs = _conv("You are a meticulous Python reviewer. " + str(time.time()), big2, "List every function.")
    t = time.time()
    hold = {}
    tj = threading.Thread(target=lambda: hold.setdefault("J", chat.stream(_body(long_msgs, 16), session="J")))
    tj.start()
    time.sleep(1.5)
    rk = chat.stream(_body(_conv("Be brief.", "Queued.", "Say hi."), 8), session="K", close_after_s=1.5)
    tj.join()
    recs = _records_since(t)
    j = next(x for x in recs if x.get("session_id") == "J")
    k = next(x for x in recs if x.get("session_id") == "K")
    canc["while_queued"] = {"J_finish": j["finish"], "J_prompt": j["prompt_tokens"], "K_finish": k["finish"],
                            "K_cancelled": k.get("cancelled"), "K_queue_s": k.get("queue_s"),
                            "K_deltas": rk["n_deltas"],
                            "pass": k["finish"] == "cancelled" and k.get("cancelled") == "queued"
                            and j["finish"] in ("stop", "length")}
    res["cancellation"] = canc
    print("cancellation:", json.dumps(canc), flush=True)
    res["cache_after"] = {k: v for k, v in wq({"type": "stats"})["cache"].items() if k != "items"}
    res["pass"] = res["interleaved"]["pass"] and res["eviction"]["pass"] and all(v["pass"] for v in canc.values())
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("checks pass:", res["pass"], flush=True)


def cmd_report(a):
    c = json.loads(Path(a.files[0]).read_text())["turns"]
    n = json.loads(Path(a.files[1]).read_text())["turns"] if len(a.files) > 1 else [None] * len(c)
    print("| turn | prompt tokens | cached / prefilled (snapshot) | TTFT cached | TTFT no cache | speed-up | decode tok/s "
          "cached / no cache | tokens/step | cache GB (snapshots) | peak GB | same reply |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for x, y in zip(c, n):
        s, u = x["stats"], (y or {}).get("stats", {})
        snap = s.get("w_snapshot")
        kind = f" ({snap['kind']})" if snap else ""
        tc, tn = s.get("ttft_client_s"), u.get("ttft_client_s")
        f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(f"| {s['turn']} | {s['prompt_tokens']} | {s.get('w_cached_tokens')} / {s.get('w_prefilled_tokens')}{kind} | "
              f"{f(tc)} s | {f(tn)} s | {f(tn / tc if tc and tn else None, 1)}x | {f(s.get('decode_tok_s_client'), 1)} / "
              f"{f(u.get('decode_tok_s_client'), 1)} | {f(s.get('w_tokens_per_step'))} | "
              f"{f((s.get('w_cache_bytes') or 0) / 1e9)} ({s.get('w_cache_snapshots')}) | {f(s.get('w_peak_gb'), 1)} | "
              f"{u.get('same_reply', '-')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "replay", "checks", "report"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--session", default="agent-1")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--target", type=int, default=20000, help="stop after the first turn with this many prompt tokens")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--small-budget", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=12.0, help="restored after the eviction check")
    a = ap.parse_args()
    {"run": cmd_run, "replay": cmd_replay, "checks": cmd_checks, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
