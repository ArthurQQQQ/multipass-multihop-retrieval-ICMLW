#!/usr/bin/env python3
"""longmemeval_ms_largek.py - Test if larger K helps multi-session aggregation.

Runs only multi-session questions (133) with K=30 BM25 chunks.
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from longmemeval_eval_methods import select_context, render_reader, lenient_score

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.cn/v1/chat/completions")
MODEL = "glm-4.7"
CONC = int(os.environ.get("LM_CONC", "8"))


async def call_glm(client, prompt, max_tokens=200):
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


JUDGE_PROMPT = """You are scoring a question-answering result against a gold reference.

Question: {question}
Gold reference answer: {gold}
Predicted answer: {pred}

Is the predicted answer essentially correct (paraphrase OK, but for numeric answers exact match required, "I don't know" is NO)?

Reply EXACTLY one token: YES or NO."""


async def evaluate_one(sem, client, q, K):
    async with sem:
        sessions = {"ids": q["haystack_session_ids"], "sessions": q["haystack_sessions"]}
        # Build with K=30 chunks budget=24000
        from longmemeval_eval_methods import select_context as sc
        # Inline import to set chunk_chars; we'll use bm25_chunks
        import longmemeval_eval_methods as lem
        lem.CHUNK_CHARS = 1200
        blocks = sc("bm25_chunks", q["question"], sessions, q["answer_session_ids"],
                    char_budget=24000, K=K)
        prompt = render_reader(q["question"], blocks, ctx_label="memory",
                              qdate=q.get("question_date", ""))
        out, err = await call_glm(client, prompt)
        # Run LLM judge
        judge_prompt = JUDGE_PROMPT.replace("{question}", q["question"])\
                                    .replace("{gold}", str(q["answer"]))\
                                    .replace("{pred}", out or "")
        jraw, _ = await call_glm(client, judge_prompt, max_tokens=8)
        jyes = (jraw or "").strip().upper().startswith("YES")
        return {"question_id": q["question_id"], "question_type": q["question_type"],
                "question": q["question"], "answer": q["answer"],
                "predicted": out, "judge_yes": jyes,
                "lenient_score": lenient_score(out, q["answer"]),
                "n_blocks": len(blocks), "ctx_chars": sum(len(b) for b in blocks)}


async def run(questions, K):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        return await asyncio.gather(*[evaluate_one(sem, client, q, K) for q in questions])


def main():
    LM = REPO / "data/longmemeval/cleaned"
    with open(LM / "longmemeval_oracle.json") as f:
        data = json.load(f)
    # Filter to multi-session only
    ms = [d for d in data if d["question_type"] == "multi-session"]
    print(f"Multi-session questions: {len(ms)}")
    print(f"Method: BM25 K=30 char_budget=24000  CONC={CONC}")
    t0 = time.time()
    results = asyncio.run(run(ms, K=30))
    print(f"  done {time.time()-t0:.0f}s")
    correct = sum(1 for r in results if r["judge_yes"])
    correct_lenient = sum(r["lenient_score"] for r in results)
    avg_chars = sum(r["ctx_chars"] for r in results) / len(results)
    print(f"  judge: {correct}/{len(results)} = {correct/len(results):.3f}")
    print(f"  lenient: {correct_lenient}/{len(results)} = {correct_lenient/len(results):.3f}")
    print(f"  avg_chars: {avg_chars:.0f}")
    print(f"  baseline (BM25 K=10): 0.481 lenient, 0.481 judge (≈0.481 lenient on this subset)")
    out = LM / "ms_bm25_K30_b24k.jsonl"
    with open(out, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
