#!/usr/bin/env python3
"""lb2_multipass_rlr.py - rlr_hier-style reranker on multipass nodes."""
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
import torch

REPO = Path(__file__).resolve().parents[2]
LB_DIR = REPO / "data" / "longbench_v2"
LB_EMB = LB_DIR / "embeddings"
RUNS = REPO / "data" / "mgt_runs"

sys.path.insert(0, str(REPO / "scripts" / "TEIE"))
from rlr_train_reranker import Reranker

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.cn/v1/chat/completions")
MODEL = "glm-4.7"
CONC = 24


def render_reader(question, choices, contexts, ctx_label="memories"):
    parts = [f"[{ctx_label.title()} {i+1}] {c}" for i, c in enumerate(contexts)]
    return f"""You answer multiple-choice questions using only the provided {ctx_label}.

{ctx_label.title()}:
{chr(10).join(parts)}

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
        out, err = await call_glm(client, render_reader(t["question"], choices, t["contexts"]))
        return {**t, "predicted_raw": out, "predicted": extract_letter(out), "reader_error": err}


async def run_pool(items):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        return await asyncio.gather(*[read_one(sem, client, t) for t in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=160)
    ap.add_argument("--POOL", type=int, default=200)
    ap.add_argument("--char-budget", type=int, default=8000)
    ap.add_argument("--with-chunk", type=int, default=0, help="add this many top chunks first")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(LB_DIR / "sample.jsonl")]

    print("Loading multipass nodes + embeddings ...")
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
    for i, n in enumerate(nodes): nodes_by_story[n["story_id"]].append(i)
    chunks_by_story = defaultdict(list)
    for i, c in enumerate(chunks): chunks_by_story[c["story_id"]].append(i)

    # Load reranker
    print("Loading reranker ...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(RUNS / "rlr_v1_best.pt", map_location=device)
    rlr = Reranker(d_in=1024, d_h=256).to(device)
    rlr.load_state_dict(ck["model"]); rlr.eval()
    n_emb_t = torch.from_numpy(n_emb).to(device)

    items = []
    for s in samples:
        story = f"qa{s['qa_idx']}"
        n_ix = nodes_by_story.get(story, [])
        c_ix = chunks_by_story.get(story, [])
        if not n_ix: continue
        qi = qa_to_idx[s["qa_idx"]]
        qv = q_emb[qi]
        # Stage 1: dense top-POOL
        cos = n_emb[n_ix] @ qv
        pool_local = np.argsort(-cos)[:min(args.POOL, len(n_ix))]
        cand = [n_ix[p] for p in pool_local]
        # Stage 2: rerank
        with torch.no_grad():
            ce = n_emb_t[cand].unsqueeze(0)
            qt = torch.from_numpy(qv).to(device).unsqueeze(0)
            scores = rlr(qt, ce).squeeze(0).cpu().numpy()
        new_order = np.argsort(-scores)

        contexts = []
        if args.with_chunk > 0 and c_ix:
            cos_c = c_emb[c_ix] @ qv
            top_c = np.argsort(-cos_c)[:args.with_chunk]
            for r in top_c:
                contexts.append(chunks[c_ix[int(r)]]["text"])
        for r in new_order[:args.K - len(contexts)]:
            contexts.append(nodes[cand[int(r)]]["text"])

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
    print(f"\n=== rlr_multipass POOL={args.POOL} K={args.K} chunk={args.with_chunk} budget={args.char_budget}c ===")
    print(f"  avg n_contexts={avg_n:.1f} avg_chars={avg_c:.0f}")
    t0 = time.time()
    scored = asyncio.run(run_pool(items))
    print(f"  reader done in {time.time()-t0:.0f}s")
    n = len(scored)
    correct = sum(1 for r in scored if r["predicted"] == r["answer"]) / n
    print(f"  -> n={n} acc={correct:.3f}")
    out_path = LB_DIR / f"scored_rlr_mp_POOL{args.POOL}_K{args.K}_c{args.with_chunk}_b{args.char_budget}.jsonl"
    with open(out_path, "w") as f:
        for r in scored: f.write(json.dumps(r) + "\n")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
