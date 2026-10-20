"""Volundr MLX worker: one model on the GPU behind a Unix socket, with the session / prefix cache (session.py).
agens-server (server/, the upstream crate with MlxBackend) talks to it; so can any client of the protocol below.

Protocol, version 1. Every message is a frame: a 4-byte big-endian length, then that many bytes of UTF-8 JSON (an
object with a "type"). One request per connection.

  client -> worker
    {"type": "generate", "prompt_ids": [int], "session_id": str | null,
     "sampling": {"temperature", "top_p", "top_k" (null = all), "min_p", "presence_penalty", "frequency_penalty",
                  "repetition_penalty", "min_new_tokens", "max_new_tokens", "seed" (null = random), "logit_bias":
                  [[id, bias]], "suppress_tokens": [id], "stop_token_ids": [id], "ignore_eos",
                  "reasoning_budget" (null = off), "reasoning_budget_soft_bias"}}
        (agens-server's SamplingParams; missing fields take those defaults: temperature 1, top_p 1, no top-k, ...)
    {"type": "info"}    -> {"type": "info", "protocol": 1, "model", "context_length", "vocab_valid", ...}
    {"type": "stats"}   -> {"type": "stats", "cache": {...}, "queue": n, "last": [recent request records]}
    {"type": "clear"}   -> {"type": "cleared"} (drops every snapshot; requests keep the copies they restored)
    {"type": "budget", "gb": float} -> {"type": "budget", "budget_bytes": n} (sets the cache budget, evicting down to
                           it; 0 disables the cache)
    {"type": "reset_peak"} -> {"type": "reset_peak", "peak_gb": float} (the MLX peak memory so far, then reset)
    {"type": "config", "slice_ms"?, "max_batch"?, "spec"?, "pipeline"?, "bspec"?, "bspec_max"?} -> the scheduler
                                                                                        settings
    {"type": "cancel"}  (optional, on a generate connection; closing the connection does the same)

  worker -> client, for generate
    {"type": "tokens", "ids": [int]}   one frame per decode step (several ids per step when speculative)
    {"type": "done", "finish": "stop" | "length" | "cancelled", "stats": {...}}
    {"type": "error", "message": str}

Requests are served with continuous batching (scheduler.py, after upstream's server mode): up to --max-batch
requests are active at once; each pass of the engine loop admits new requests, runs prefill work for the oldest one
still prefilling (layers of its current chunk, at most --slice-ms while others decode), then one decode step for every
decoding request together. One decoding greedy request (top_k == 1 or temperature 0, no penalties / biases / budget)
runs the speculative loop: DFlash2 drafter (4-bit linears and head copy), the NAX small-M verify, prompt lookup, the
default early@cv loop. Two or more decode as rows of one batched plain decode step (batch.py), sampled rows with
sampler.py. All resume from and leave snapshots in the session cache. --max-batch 1 serves one request at a time.

A request whose connection closes is cancelled: dropped while queued, stopped between prefill layers or after the
current decode step. The worker emits at most max_new_tokens ids and stops after a stop id (unless ignore_eos);
agens-server also enforces both, plus stop strings.

usage (from volundr-mlx):
  python worker.py [--socket PATH] [--cache-gb 12] [--interval 4096] [--max-batch 16] [--log FILE] [--no-warmup]
"""
import os
import sys

# Speculative decoding's Metal command-buffer setting (spec/specdec.py's top comment); read by MLX at device init.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spec"))
import session as SS  # noqa: E402

PROTOCOL = 1
MAX_FRAME = 256 << 20


def default_socket() -> str:
    return os.environ.get("VOLUNDR_WORKER_SOCKET") or os.path.join(tempfile.gettempdir(), "volundr-mlx-worker.sock")


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b:
            return None
        buf += b
    return bytes(buf)


def read_frame(sock):
    h = recv_exact(sock, 4)
    if h is None:
        return None
    (n,) = struct.unpack(">I", h)
    if n > MAX_FRAME:
        raise ValueError(f"frame of {n} bytes")
    body = recv_exact(sock, n)
    if body is None:
        return None
    return json.loads(body)


def write_frame(sock, obj):
    body = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(body)) + body)


class Conn:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.gone = threading.Event()

    def send(self, obj) -> bool:
        if self.gone.is_set():
            return False
        try:
            with self.lock:
                write_frame(self.sock, obj)
            return True
        except OSError:
            self.gone.set()
            return False

    def close(self):
        with self.lock:
            if getattr(self, "_closed", False):
                return
            self._closed = True
        self.gone.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class Job:
    def __init__(self, conn, msg, kind="generate"):
        self.conn, self.msg, self.kind = conn, msg, kind
        self.t_recv = time.perf_counter()
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.context_length = 262144


class LocalConn:
    """A connection for the worker's own requests (warm-up): keeps the frames."""

    def __init__(self):
        self.frames = []
        self.gone = threading.Event()

    def send(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass


class Worker:
    def __init__(self, args):
        import specdec

        t0 = time.perf_counter()
        self.model, self.drafter, self.tok = specdec.load(args.drafter_bits, args.head_bits, args.nax)
        self.load_s = time.perf_counter() - t0
        conv = getattr(self.tok, "convert_tokens_to_ids", None)
        think = (conv("<|think|>"), conv("<|/think|>")) if conv else (None, None)
        self.cache = SS.SessionCache(budget_gb=args.cache_gb, interval=args.interval)
        import scheduler

        self.engine = scheduler.BatchEngine(self.model, self.drafter, self.cache, T=args.T, chunk=args.chunk,
                                            lookup=not args.no_lookup, loop=args.loop, think_ids=think,
                                            max_batch=args.max_batch, slice_ms=args.slice_ms,
                                            nax_min_batch=args.nax_min_batch, spec=not args.no_spec,
                                            max_gb=args.max_active_gb, pipeline=not args.no_pipeline,
                                            bspec=not args.no_bspec, bspec_max=args.bspec_max)
        self.engine.on_done = self._done
        self.args = args
        self.jobs = queue.Queue()
        self.recent = deque(maxlen=64)
        self.log = open(args.log, "a") if args.log else None
        self.info = {"type": "info", "protocol": PROTOCOL, "model": str(specdec.MODEL), "context_length":
                     args.context_length or int(json.loads((specdec.MODEL / "config.json").read_text())
                                                    .get("max_position_embeddings", 262144)),
                     "vocab_valid": self.model.args.vocab_valid, "cache_budget_bytes": self.cache.budget,
                     "interval": self.cache.interval, "spec": {"T": args.T, "drafter_bits": args.drafter_bits,
                     "head_bits": args.head_bits, "nax": args.nax, "lookup": not args.no_lookup,
                     "loop": args.loop or "early@cv", "single_stream": not args.no_spec},
                     "batch": {"max_batch": args.max_batch, "slice_ms": args.slice_ms,
                               "nax_min_batch": args.nax_min_batch, "max_active_gb": args.max_active_gb,
                               "pipeline": not args.no_pipeline, "bspec": not args.no_bspec,
                               "bspec_max": args.bspec_max},
                     "load_s": round(self.load_s, 1)}

    # -- warm-up (Metal JIT for the prefill, verify and decode kernels, batched decode at 2-4 rows, far blocks)
    def _local(self, ids, sp: dict, session=None):
        job = Job(LocalConn(), {"type": "generate", "prompt_ids": list(ids), "session_id": session, "sampling": sp})
        job.context_length = self.info["context_length"]
        job.local = True
        self.engine.submit(job)
        return job

    def _drain(self):
        while self.engine.busy:
            self.engine.tick()

    def warm_up(self):
        t0 = time.perf_counter()
        saved, self.engine.cache = self.engine.cache, None
        try:
            w = self.tok.encode("Warm up the kernels, please. " * 4)
            g = {"top_k": 1, "max_new_tokens": 24}
            self._local(w, g)  # speculative, alone
            self._drain()
            self._local(w, {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "max_new_tokens": 8, "seed": 1})
            self._drain()  # sampled, alone (a batch of one)
            for k in range(4):  # batched, 2-4 rows at different lengths, joining and leaving
                self._local(w[: len(w) - 3 * k], {"top_k": 1, "max_new_tokens": 12 + 4 * k})
            self._drain()
            if not self.args.short_warmup:  # far blocks (> 4096 positions) in prefill, verify and batched decode
                text = (HERE / "README.md").read_text()
                ids = self.tok.encode(text)
                ids = (ids * (5000 // len(ids) + 1))[:5000]
                self._local(ids, {"top_k": 1, "max_new_tokens": 16})
                self._drain()
                self._local(ids, {"top_k": 1, "max_new_tokens": 16})
                self._local(w, {"top_k": 1, "max_new_tokens": 16})
                self._drain()
        finally:
            self.engine.cache = saved
            mx.clear_cache()
        return time.perf_counter() - t0

    # -- connections (threads; no MLX work here)
    def handle(self, sock):
        conn = Conn(sock)
        try:
            msg = read_frame(sock)
        except Exception as e:  # noqa: BLE001
            conn.send({"type": "error", "message": f"bad frame: {e}"})
            conn.close()
            return
        if msg is None:
            conn.close()
            return
        kind = msg.get("type")
        if kind == "info":
            conn.send(self.info)
            conn.close()
            return
        if kind == "stats":
            e = self.engine
            conn.send({"type": "stats", "cache": self._cache_stats(), "queue": self.jobs.qsize() + len(e.waiting),
                       "active": len(e.active), "batch_rows": e.bc.B, "engine": dict(e.stats),
                       "peak_gb": mx.get_peak_memory() / 1e9, "last": list(self.recent)[-16:]})
            conn.close()
            return
        if kind not in ("generate", "clear", "budget", "reset_peak", "config"):
            conn.send({"type": "error", "message": f"unknown message type {kind!r}"})
            conn.close()
            return
        job = Job(conn, msg, kind)
        job.context_length = self.info["context_length"]
        self.jobs.put(job)
        # watch the connection: EOF or a cancel message cancels the job
        while not job.done.is_set():
            try:
                m = read_frame(sock)
            except (OSError, ValueError):
                m = None
            if m is None or m.get("type") == "cancel":
                job.cancel.set()
                if m is None:
                    conn.gone.set()
                break
        job.done.wait()
        conn.close()

    def _cache_stats(self):
        try:
            return self.cache.stats()
        except Exception:  # noqa: BLE001 (read from another thread while a request runs)
            return {"bytes": None}

    def serve(self, path):
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        os.chmod(path, 0o600)
        srv.listen(64)

        def accept():
            while True:
                s, _ = srv.accept()
                threading.Thread(target=self.handle, args=(s,), daemon=True).start()

        threading.Thread(target=accept, daemon=True).start()
        print(f"worker: listening on {path}", file=sys.stderr, flush=True)
        while True:
            # requests in, one engine pass (all MLX work is on this thread)
            block = not self.engine.busy
            while True:
                try:
                    job = self.jobs.get(block=block)
                except queue.Empty:
                    break
                block = False
                if job.kind == "generate":
                    self.engine.submit(job)
                else:
                    try:
                        self.control(job)
                    finally:
                        job.done.set()
                        job.conn.close()
            if self.engine.busy:
                self.engine.tick()

    def control(self, job):
        msg, conn = job.msg, job.conn
        if job.kind == "clear":
            self.cache.clear()
            conn.send({"type": "cleared"})
        elif job.kind == "budget":
            self.cache.budget = max(0, int(float(msg.get("gb", 0)) * 1e9))
            self.cache.evict(self.cache.budget)
            self.info["cache_budget_bytes"] = self.cache.budget
            conn.send({"type": "budget", "budget_bytes": self.cache.budget, "bytes": self.cache.nbytes})
        elif job.kind == "reset_peak":
            peak = mx.get_peak_memory() / 1e9
            mx.reset_peak_memory()
            conn.send({"type": "reset_peak", "peak_gb": peak})
        elif job.kind == "config":  # scheduler settings at run time (for measurements)
            e = self.engine
            if msg.get("slice_ms") is not None:
                e.slice_s = float(msg["slice_ms"]) / 1e3
            if msg.get("max_batch") is not None:
                e.max_batch = max(1, int(msg["max_batch"]))
            if msg.get("spec") is not None:
                e.spec = bool(msg["spec"])
            if msg.get("pipeline") is not None:
                e.pipeline = bool(msg["pipeline"])
            if msg.get("bspec") is not None:
                e.bspec = bool(msg["bspec"])
            if msg.get("bspec_max") is not None:
                e.bspec_max = int(msg["bspec_max"])
            conn.send({"type": "config", "slice_ms": e.slice_s * 1e3, "max_batch": e.max_batch, "spec": e.spec,
                       "pipeline": e.pipeline, "bspec": e.bspec, "bspec_max": e.bspec_max})

    # -- the end of a request (engine thread): the done / error frame, the log, the connection
    def _done(self, job, req, rec):
        conn = job.conn
        try:
            if "error" in rec:
                conn.send({"type": "error", "message": rec["error"]})
                print(f"worker: error {rec['error']}", file=sys.stderr, flush=True)
                return
            if getattr(job, "local", False):
                return
            out_tokens = rec.pop("tokens", None)
            msg = job.msg
            sp = req.sp if req is not None else None
            rec["session_id"] = msg.get("session_id")
            rec["sampling"] = None if sp is None else ("greedy" if sp.greedy else "sampled")
            if rec.get("ttft_s") is not None:
                rec["ttft_from_receipt_s"] = rec["ttft_s"] + rec.get("queue_s", 0.0)
            rec["t"] = time.time()
            self.recent.append(rec)
            conn.send({"type": "done", "finish": rec.get("finish", "length"), "stats": rec})
            if self.log:
                if self.args.log_tokens:  # (for offline checks: the exact ids the server sent and the worker emitted)
                    rec = {**rec, "prompt_ids": msg.get("prompt_ids"), "tokens": out_tokens}
                self.log.write(json.dumps(rec) + "\n")
                self.log.flush()
            if rec.get("cancelled") == "queued":
                print(f"worker: cancelled while queued ({rec.get('queue_s', 0):.1f} s)", file=sys.stderr, flush=True)
                return
            snap = rec.get("snapshot")
            print(f"worker: {rec.get('finish')} prompt {rec.get('prompt_tokens')} cached {rec.get('cached_tokens')} "
                  f"({'-' if not snap else snap['kind']}) prefilled {rec.get('prefilled_tokens')} "
                  f"gen {rec.get('generated')} {rec.get('mode')} (batch {rec.get('mean_batch') or 0:.1f}) "
                  f"queue {rec.get('queue_s') or 0:.2f}s ttft {rec.get('ttft_s') or 0:.2f}s "
                  f"dec {rec.get('decode_tok_s') or 0:.1f} tok/s active {rec.get('active_requests')} "
                  f"cache {self.cache.nbytes / 1e9:.2f} GB ({len(self.cache.snaps)}) "
                  f"peak {rec.get('peak_gb', 0):.1f} GB", file=sys.stderr, flush=True)
        finally:
            job.done.set()
            conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--socket", default=default_socket())
    ap.add_argument("--cache-gb", type=float, default=None,
                    help="session cache budget in GB (default $VOLUNDR_SESSION_CACHE_GB or 12; 0 disables it)")
    ap.add_argument("--interval", type=int, default=4096, help="snapshot every N tokens inside a long prefill")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--max-batch", type=int, default=16, help="requests active at once (prefilling or decoding)")
    ap.add_argument("--slice-ms", type=float, default=200.0,
                    help="prefill time between two decode steps while other requests decode")
    ap.add_argument("--nax-min-batch", type=int, default=4, help="batched decode uses the NAX matmuls from N rows")
    ap.add_argument("--no-spec", action="store_true",
                    help="a lone greedy request decodes plainly too (batched path), not speculatively")
    ap.add_argument("--no-bspec", action="store_true",
                    help="greedy decoding requests decode plainly in the batch, never with batched speculation")
    ap.add_argument("--bspec-max", type=int, default=4,
                    help="batched speculation while 2..N greedy requests decode (each drafts 16 // N tokens)")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="batched steps one at a time (by default the next step's graph is built while one runs)")
    ap.add_argument("--max-active-gb", type=float, default=46.0,
                    help="admit no new request while MLX holds more than this (unless none is active)")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--head-bits", type=int, default=4)
    ap.add_argument("--no-nax", dest="nax", action="store_false")
    ap.add_argument("--no-lookup", action="store_true")
    ap.add_argument("--loop", default="", help="spec loop flags (specdec.loop_options), default early@cv")
    ap.add_argument("--context-length", type=int, default=0, help="default: the model's max_position_embeddings")
    ap.add_argument("--log", default=None, help="append one JSON record per request")
    ap.add_argument("--log-tokens", action="store_true", help="--log records include prompt_ids and output tokens")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--short-warmup", action="store_true", help="skip the 5K-token warm-up (far-block kernels)")
    args = ap.parse_args()
    w = Worker(args)
    print(f"worker: model loaded in {w.load_s:.1f} s; cache budget {w.cache.budget / 1e9:.1f} GB, "
          f"interval {w.cache.interval}", file=sys.stderr, flush=True)
    if not args.no_warmup:
        print(f"worker: warm-up {w.warm_up():.1f} s", file=sys.stderr, flush=True)
    mx.reset_peak_memory()
    w.serve(args.socket)


if __name__ == "__main__":
    main()
