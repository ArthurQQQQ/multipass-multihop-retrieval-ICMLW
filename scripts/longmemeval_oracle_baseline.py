#!/usr/bin/env python3
"""longmemeval_oracle_baseline.py - GLM-4.7 reader on LongMemEval oracle context.

Each question has answer_session_ids (gold sessions). Concatenate gold sessions and ask GLM
to answer free-form. Score with answer-string substring match (lenient).

Why this matters: gives the upper bound for any retrieval-based memory system on LongMemEval.
If our retrieval can match this, retrieval is solved. If far below, retrieval IS the bottleneck.

Output: data/longmemeval/oracle_baseline_{model}.jsonl
"""
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

REPO = Path(__file__).resolve().parent.parent
LM_DIR = REPO / "data" / "longmemeval" / "cleaned"

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.com/v1/chat/completions")
MODEL = os.environ.get("LM_MODEL", "glm-4.7")
CONC = int(os.environ.get("LM_CONC", "8"))  # low default to avoid stomping other jobs


def render_oracle(question, sessions, qdate=""):
    """sessions = list of [{role, content}, ...]"""
    blocks = []
    for i, s in enumerate(sessions):
        turns = "\n".join(f"  {t['role'].upper()}: {t['content']}" for t in s)
        blocks.append(f"=== Session {i+1} ===\n{turns}")
    ctx = "\n\n".join(blocks)
    return f"""You are answering a question based on prior conversation history.

Conversation history:
{ctx}

{('Question asked at: ' + qdate) if qdate else ''}
Question: {question}

Answer concisely (one sentence). If unanswerable from history, say "Cannot determine"."""


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


def lenient_score(pred, gold):
    """Lenient: ANY of the gold-acceptable answers (split by '.' for "X. Y is also acceptable" patterns)
    appears as substring in pred (case-insensitive, normalized whitespace)."""
    if not pred: return 0
    p = re.sub(r"\s+", " ", str(pred).lower().strip())
    # Convert gold to string (sometimes int)
    gold_str = str(gold).lower()
    candidates = [gold_str.strip()]
    # Common patterns: "X. Y is also acceptable" → check both "X" and "Y"
    for sep in [r"\.\s+", r";\s*", r"\s+or\s+"]:
        for c in list(candidates):
            parts = re.split(sep, c)
            for part in parts:
                part = part.strip()
                # Strip trailing punctuation/quotes
                part = part.strip("'\"`,.")
                if part and len(part) >= 2:
                    candidates.append(part)
    candidates = list(set(candidates))
    # Return 1 if any candidate found in pred
    for cand in candidates:
        cand_norm = re.sub(r"\s+", " ", cand)
        if cand_norm and cand_norm in p:
            return 1
    return 0


async def evaluate_one(sem, client, q):
    async with sem:
        # Filter to answer-bearing sessions only (oracle = ground truth context)
        ans_ids = set(q["answer_session_ids"])
        sessions_pairs = list(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        oracle_sessions = [s for sid, s in sessions_pairs if sid in ans_ids]
        if not oracle_sessions:
            oracle_sessions = q["haystack_sessions"]  # fallback
        prompt = render_oracle(q["question"], oracle_sessions, q.get("question_date", ""))
        # Truncate if too long
        if len(prompt) > 60000:
            prompt = prompt[:60000] + "\n[truncated]\n\nQuestion: " + q["question"]
        out, err = await call_glm(client, prompt)
        return {**{k: q.get(k) for k in ["question_id", "question_type", "question", "answer"]},
                "predicted": out, "lenient_score": lenient_score(out, q["answer"]), "error": err,
                "n_sessions": len(oracle_sessions),
                "ctx_chars": sum(len(t["content"]) for s in oracle_sessions for t in s)}


async def run(questions):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        coros = [evaluate_one(sem, client, q) for q in questions]
        return await asyncio.gather(*coros)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-name", default="oracle_glm47")
    ap.add_argument("--per-type", type=int, default=0, help="Sample this many per question_type (balanced)")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR GLM_API_KEY missing"); sys.exit(1)

    with open(LM_DIR / "longmemeval_oracle.json") as f:
        data = json.load(f)
    if args.per_type > 0:
        from collections import defaultdict as _dd
        by_t = _dd(list)
        for d in data: by_t[d["question_type"]].append(d)
        balanced = []
        for t in sorted(by_t):
            balanced.extend(by_t[t][:args.per_type])
        data = balanced
        print(f"  balanced sample: {args.per_type} per type → {len(data)} total")
    elif args.limit > 0:
        data = data[:args.limit]
    print(f"LongMemEval oracle: {len(data)} questions")
    print(f"  CONC={CONC}, model={MODEL}, url={API_URL}")

    t0 = time.time()
    results = asyncio.run(run(data))
    elapsed = time.time() - t0
    print(f"  done {elapsed:.0f}s ({len(data)/elapsed:.2f} q/s)")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["lenient_score"])
    print(f"\n{'type':<28} {'n':>4} {'lenient':>8}")
    print("-" * 46)
    for t in sorted(by_type):
        scores = by_type[t]
        print(f"  {t:<26} {len(scores):>4} {sum(scores)/len(scores):>8.3f}")
    overall = sum(r["lenient_score"] for r in results) / len(results)
    print(f"  {'OVERALL':<26} {len(results):>4} {overall:>8.3f}")

    out_path = LM_DIR / f"oracle_baseline_{args.out_name}.jsonl"
    with open(out_path, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
