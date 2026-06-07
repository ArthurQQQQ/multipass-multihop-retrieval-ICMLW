#!/usr/bin/env python3
"""lb2_eval_multipass_hier.py - Hierarchical: 1 best chunk + top-K multipass nodes."""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LB_DIR = REPO / "data" / "longbench_v2"
LB_EMB = LB_DIR / "embeddings"

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.cn/v1/chat/completions")
MODEL = "glm-4.7"
CONC = 24


def render_reader(question, choices, contexts, ctx_label="context"):
    parts = [f"[{ctx_label.title()} {i+1}] {c}" for i, c in enumerate(contexts)]
    ctx_str = "\n\n".join(parts)
    return f"""You answer multiple-choice questions using only the provided {ctx_label}.

{ctx_label.title()}:
{ctx_str}

Question: {question}

Choices:
A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Reply with EXACTLY one letter (A, B, C, or D). No explanation.
"""


async def call_glm(client, prompt, max_tokens=8):
    err = "max_retries"
    for attempt in range(4):
        try:
            r = await client.post(API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0.0, "enable_thinking": False},
                timeout=httpx.Timeout(180.0, connect=30.0))
            if r.status_code in (429,) or r.status_code >= 500:
                await asyncio.sleep(min(60.0, 2.0 ** (attempt + 1))); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip(), None
        except Exception as e:
            err = str(e)[:160]
            await asyncio.sleep(min(60.0, 2.0 ** (attempt + 1)))
    return "", err


def extract_letter(s):
    s = (s or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1) if m else (s[:1] if s and s[0] in "ABCD" else "")


async def read_one(sem, client, t):
    async with sem:
        choices = {"A": t["choice_A"], "B": t["choice_B"], "C": t["choice_C"], "D": t["choice_D"]}
        out, err = await call_glm(client,
            render_reader(t["question"], choices, t["contexts"]))
        return {**t, "predicted_raw": out, "predicted": extract_letter(out), "reader_error": err}


async def run_pool(items):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        return await asyncio.gather(*[read_one(sem, client, t) for t in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-nodes", type=int, default=120, help="multipass nodes to add after chunk")
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--char-budget", type=int, default=8000)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(LB_DIR / "sample.jsonl")]

    # Load multipass nodes + embeddings
    print("Loading multipass nodes + chunks ...")
    nodes = [json.loads(l) for l in open(LB_EMB / "nodes_multipass_index.jsonl")]
    n_emb = np.load(LB_EMB / "nodes_multipass.npy").astype(np.float32)
    n_emb = n_emb / np.maximum(np.linalg.norm(n_emb, axis=1, keepdims=True), 1e-9)

    chunks = [json.loads(l) for l in open(LB_EMB / "chunks_index.jsonl")]
    c_emb = np.load(LB_EMB / "chunks.npy").astype(np.float32)
    c_emb = c_emb / np.maximum(np.linalg.norm(c_emb, axis=1, keepdims=True), 1e-9)

    qa_index = [json.loads(l) for l in open(LB_EMB / "qa_index.jsonl")]
    q_emb = np.load(LB_EMB / "q.npy").astype(np.float32)
    q_emb = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-9)
    qa_to_idx = {q["qa_idx"]: i for i, q in enumerate(qa_index)}

    nodes_by_story = defaultdict(list)
    for i, n in enumerate(nodes):
        nodes_by_story[n["story_id"]].append(i)
    chunks_by_story = defaultdict(list)
    for i, c in enumerate(chunks):
        chunks_by_story[c["story_id"]].append(i)

    items = []
    for s in samples:
        story = f"qa{s['qa_idx']}"
        n_ix = nodes_by_story.get(story, [])
        c_ix = chunks_by_story.get(story, [])
        qi = qa_to_idx[s["qa_idx"]]
        qv = q_emb[qi]
        contexts = []
        # Top-N chunks first
        if c_ix:
            cos_c = c_emb[c_ix] @ qv
            top_c = np.argsort(-cos_c)[:args.n_chunks]
            for r in top_c:
                contexts.append(chunks[c_ix[int(r)]]["text"])
        # Then top-K multipass nodes
        if n_ix:
            cos_n = n_emb[n_ix] @ qv
            top_n = np.argsort(-cos_n)[:args.K_nodes]
            for r in top_n:
                contexts.append(nodes[n_ix[int(r)]]["text"])
        # Trim by char budget
        trimmed = []; used = 0
        for c in contexts:
            if used + len(c) > args.char_budget and trimmed: break
            trimmed.append(c); used += len(c)
        items.append({
            "qa_idx": s["qa_idx"], "question": s["question"],
            "choice_A": s["choice_A"], "choice_B": s["choice_B"],
            "choice_C": s["choice_C"], "choice_D": s["choice_D"],
            "answer": s["answer"], "contexts": trimmed,
            "n_contexts": len(trimmed), "ctx_chars": used,
        })

    avg_n = np.mean([it["n_contexts"] for it in items])
    avg_c = np.mean([it["ctx_chars"] for it in items])
    print(f"\n=== hier_mp ({args.n_chunks} chunk + {args.K_nodes} multipass nodes) budget={args.char_budget}c (n={len(items)}) ===")
    print(f"  avg n_contexts={avg_n:.1f} avg_chars={avg_c:.0f}")
    t0 = time.time()
    scored = asyncio.run(run_pool(items))
    print(f"  reader done in {time.time()-t0:.0f}s")
    n = len(scored)
    correct = sum(1 for r in scored if r["predicted"] == r["answer"]) / n
    print(f"  -> n={n} acc={correct:.3f}")
    out_path = LB_DIR / f"scored_hier_mp_c{args.n_chunks}_K{args.K_nodes}_b{args.char_budget}.jsonl"
    with open(out_path, "w") as f:
        for r in scored: f.write(json.dumps(r) + "\n")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
