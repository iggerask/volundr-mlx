"""Build the speculative-decoding evaluation suite: broad categories, a tune split and a disjoint held-out split.

Sources (all already in the local HF cache, read as data):
- nvidia/SPEED-Bench "qualitative": 11 categories x 80 prompts (coding, humanities, math, qa, rag, reasoning, stem,
  writing, multilingual, summarization, roleplay); the first user turn of each. Rows whose text is SPEED-Bench's
  "fetch from the source" stub are dropped; categories left short are topped up from local MMLU / ARC-Challenge
  questions (humanities, stem, reasoning) and BoolQ Wikipedia passages (summarization, and a "json" extraction
  category).
- SWE-bench/SWE-smith-trajectories: real agent sessions with bash / editor tool calls, cut after a tool result, so
  the model writes the next assistant turn (tool-call output) -> category "agent".
- Long context: bench/long_prompt.txt and the engine's m1_model.cu with questions -> category "long" (8-19K tokens).

Per category, a seeded sample: the first TUNE prompts go to "tune", the next TEST to "test" (never overlapping).
Thinking alternates on / off within each category so both reply styles are in both splits.

usage: python -I make_suite.py --model ../models/volundr-int4-mlx --out suite.json [--tune 4] [--test 6]
"""
import argparse
import ast
import glob
import json
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": "View, create or edit files: commands view, create, str_replace, insert, undo_edit.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "path": {"type": "string"}, "file_text": {"type": "string"},
            "old_str": {"type": "string"}, "new_str": {"type": "string"}, "insert_line": {"type": "integer"},
            "view_range": {"type": "array", "items": {"type": "integer"}}},
            "required": ["command", "path"]}}},
]


def _text(content):
    """SWE-smith content: a plain string, a [{'type': 'text', 'text': ...}] list, or that list stringified."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if isinstance(content, str) and content.startswith("[{"):
        try:
            parts = ast.literal_eval(content)
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (ValueError, SyntaxError):
            return content
    return content if isinstance(content, str) else str(content)


def _tool_calls(tc):
    if not tc:
        return None
    calls = ast.literal_eval(tc) if isinstance(tc, str) else tc
    out = []
    for c in calls:
        f = c["function"]
        args = f.get("arguments", "{}")
        out.append({"type": "function", "id": c.get("id"), "function": {
            "name": f["name"], "arguments": json.loads(args) if isinstance(args, str) else args}})
    return out


def chat_ids(tok, msgs, **kw):
    """apply_chat_template returns a BatchEncoding in this transformers version; we want the plain id list."""
    r = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    return list(r["input_ids"]) if hasattr(r, "keys") else list(r)


PLACEHOLDER = "SPECDEC_BENCH"  # SPEED-Bench ships some rows as a "fetch from the source" stub
MMLU_GROUPS = {
    "humanities": ["high_school_world_history", "high_school_us_history", "high_school_european_history",
                   "philosophy", "world_religions", "prehistory", "jurisprudence", "international_law"],
    "stem": ["college_physics", "college_chemistry", "college_biology", "astronomy", "electrical_engineering",
             "college_computer_science", "high_school_physics", "anatomy"],
    "reasoning": ["formal_logic", "logical_fallacies", "moral_disputes"],
}


def _mc(question, choices):
    letters = "ABCDEFGH"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{opts}\n\nExplain your reasoning step by step, then state the answer letter."


def fillers(rng):
    """Real prompts for the categories SPEED-Bench leaves as stubs: MMLU / ARC questions, BoolQ (Wikipedia) passages."""
    mmlu = pq.read_table(glob.glob(str(HUB / "datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))[0]).to_pylist()
    out = {}
    for cat, subjects in MMLU_GROUPS.items():
        rows = [r for r in mmlu if r["subject"] in subjects]
        rng.shuffle(rows)
        out[cat] = [(f"mmlu-{r['subject'][:10]}", _mc(r["question"], r["choices"])) for r in rows]
    arc = pq.read_table(glob.glob(str(HUB / "datasets--allenai--ai2_arc/snapshots/*/ARC-Challenge/test-*.parquet"))[0])
    arc = arc.to_pylist()
    rng.shuffle(arc)
    out["reasoning"] = [x for pair in zip(out["reasoning"], [(f"arc-{r['id'][-6:]}", _mc(r["question"],
                        r["choices"]["text"])) for r in arc]) for x in pair]
    boolq = pq.read_table(glob.glob(str(HUB / "datasets--aps--super_glue/snapshots/*/boolq/validation-*.parquet"))[0])
    passages = sorted({p for p in boolq.column("passage").to_pylist() if len(p.split()) > 150})
    rng.shuffle(passages)
    half = len(passages) // 2
    out["summarization"] = [(f"boolq{i}", f"Summarize the following passage in three sentences.\n\n{p}")
                            for i, p in enumerate(passages[:half])]
    out["json"] = [(f"boolq{i}", "Extract the information in the passage below as JSON with the keys \"topic\", "
                    "\"entities\" (a list of {\"name\", \"type\"}), \"dates\" and \"key_facts\" (a list of short "
                    f"strings). Output only the JSON.\n\n{p}") for i, p in enumerate(passages[half:])]
    return out


def speed_bench(rng, tok, n):
    f = glob.glob(str(HUB / "datasets--nvidia--SPEED-Bench/snapshots/*/qualitative/*.parquet"))[0]
    rows = pq.read_table(f, columns=["question_id", "category", "turns"]).to_pylist()
    by = {}
    for r in rows:
        if PLACEHOLDER not in r["turns"][0]:
            by.setdefault(r["category"], []).append((r["question_id"][:8], r["turns"][0]))
    fill = fillers(rng)
    out = {}
    for cat in sorted(set(by) | set(fill)):
        cand = by.get(cat, [])[:]
        rng.shuffle(cand)
        cand = [c for c in cand if len(tok.encode(c[1])) <= 6000]  # bounded prefill; "long" covers long context
        if len(cand) < n:  # top up from the filler source, then shuffle so both splits mix the sources
            cand = cand + fill.get(cat, [])[: n - len(cand)]
            rng.shuffle(cand)
        out[cat] = cand[:n]
        assert len(out[cat]) == n, (cat, len(out[cat]))
    return out


def agent(rng, tok, n):
    files = sorted(glob.glob(str(HUB / "datasets--SWE-bench--SWE-smith-trajectories/snapshots/*/data/tool-*.parquet")))
    t = pq.read_table(files[0], columns=["messages", "resolved", "traj_id"])
    rows = [r for r in t.to_pylist() if str(r["resolved"]) == "True"]
    rng.shuffle(rows)
    out = []
    for r in rows:
        msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        conv = []
        for m in msgs:
            role = m["role"]
            if role == "assistant":
                conv.append({"role": "assistant", "content": _text(m.get("content", "")),
                             **({"tool_calls": _tool_calls(m.get("tool_calls"))} if m.get("tool_calls") else {})})
            elif role in ("system", "user", "tool"):
                conv.append({"role": role, "content": _text(m.get("content", ""))})
        # cut after a tool result somewhere in the middle, keeping 2K-8K prompt tokens
        cuts = [i for i, m in enumerate(conv) if m["role"] == "tool" and 4 <= i < len(conv) - 2]
        rng.shuffle(cuts)
        for c in cuts:
            ids = chat_ids(tok, conv[: c + 1], tools=TOOLS, enable_thinking=len(out) % 2 == 0)
            if 2000 <= len(ids) <= 8000:
                out.append((r["traj_id"], conv[: c + 1], ids))
                break
        if len(out) == n:
            break
    return out


def long_ctx(tok, n):
    root = HERE.parent
    doc = (root / "bench/long_prompt.txt").read_text()
    code = (root / "ref/agens-engine/kernels/m1/m1_model.cu").read_text()
    qs = [
        (doc, "Summarize the document above in ten bullet points."),
        (code, "Explain what the main forward pass in the code above does, step by step, naming the functions."),
        (doc, "List every function or section heading that appears in the document above, with one line each."),
        (code, "Write a short design review of the code above: three strengths, three risks."),
        (doc, "Rewrite the first section of the document above as a concise technical note."),
        (code, "Which kernels in the code above are launched per layer during decode? Answer as a table."),
        (doc, "What are the five most important ideas in the document above? Explain each in two sentences."),
        (code, "Find likely performance bottlenecks in the code above and suggest fixes."),
        (doc, "Write a FAQ with six questions and answers based on the document above."),
        (code, "Document the public functions in the code above as a Markdown API reference."),
    ]
    out = []
    for i, (body, q) in enumerate(qs[:n]):
        body_ids = tok.encode(body)[: 8000 + 1500 * i]  # 8K..~22K tokens
        out.append((f"long{i}", tok.decode(body_ids) + "\n\n" + q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(HERE / "suite.json"))
    ap.add_argument("--tune", type=int, default=4)
    ap.add_argument("--test", type=int, default=6)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(20261018)
    n = args.tune + args.test
    suite = []

    def add(cat, i, name, ids):
        split = "tune" if i < args.tune else "test"
        suite.append({"name": f"{cat}/{i:02d}-{name}", "category": cat, "split": split, "ids": ids})

    for cat, picked in speed_bench(rng, tok, n).items():
        for i, (name, text) in enumerate(picked):
            think = i % 2 == 0
            ids = chat_ids(tok, [{"role": "user", "content": text}], enable_thinking=think)
            add(cat, i, f"{name}{'-think' if think else ''}", ids)
    for i, (tid, conv, ids) in enumerate(agent(rng, tok, n)):
        add("agent", i, f"{tid.split('.')[0][:24]}{'-think' if i % 2 == 0 else ''}", ids)
    for i, (name, text) in enumerate(long_ctx(tok, n)):
        think = i % 2 == 0
        ids = chat_ids(tok, [{"role": "user", "content": text}], enable_thinking=think)
        add("long", i, f"{name}{'-think' if think else ''}", ids)
    Path(args.out).write_text(json.dumps(suite))
    cats = sorted({s["category"] for s in suite})
    for c in cats:
        tl = [len(s["ids"]) for s in suite if s["category"] == c]
        print(f"{c:14s} tune {sum(s['category'] == c and s['split'] == 'tune' for s in suite)} "
              f"test {sum(s['category'] == c and s['split'] == 'test' for s in suite)}  "
              f"prompt tokens {min(tl)}..{max(tl)}", file=sys.stderr)
    print(f"{len(suite)} prompts -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
