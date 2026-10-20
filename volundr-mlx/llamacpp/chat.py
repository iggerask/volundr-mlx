#!/usr/bin/env python3
"""Send greedy, thinking-off chat completions to an OpenAI-compatible server (llama-server or ./serve.sh) and save
the replies with timings. Standard library only.

usage: chat.py --port 8080 --out llama.json [--prompts prompts.json] [--only name,...] [--repeat 1]
"""
import argparse, json, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ask(port, prompt, max_tokens, logprobs=0):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    if logprobs:
        body.update(logprobs=True, top_logprobs=logprobs)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    wall = time.perf_counter() - t0
    msg = out["choices"][0]["message"]
    return {"content": msg.get("content"), "reasoning_content": msg.get("reasoning_content"),
            "finish_reason": out["choices"][0].get("finish_reason"), "usage": out.get("usage"),
            "timings": out.get("timings"), "wall_s": round(wall, 3),
            "logprobs": (out["choices"][0].get("logprobs") or {}).get("content")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--logprobs", type=int, default=0, help="top-k logprobs per token (llama-server)")
    a = ap.parse_args()
    prompts = json.loads(Path(a.prompts).read_text())
    only = set(filter(None, a.only.split(",")))
    res = {}
    for p in prompts:
        if only and p["name"] not in only:
            continue
        text = p["prompt"] if "prompt" in p else (HERE / p["prompt_file"]).read_text()
        runs = [ask(a.port, text, p["max_tokens"], a.logprobs) for _ in range(a.repeat)]
        res[p["name"]] = runs
        r = runs[-1]
        t = r["timings"] or {}
        print(f"## {p['name']}: wall {r['wall_s']} s, usage {r['usage']}, "
              f"pp {t.get('prompt_per_second', 0):.1f} tok/s, tg {t.get('predicted_per_second', 0):.2f} tok/s")
        print((r["content"] or "")[:400].replace("\n", " | "))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
