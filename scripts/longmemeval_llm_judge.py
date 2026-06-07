#!/usr/bin/env python3
"""longmemeval_llm_judge.py - GLM-4.7 judge to fix lenient-substring scoring bug.

Runs YES/NO judge on existing eval_*_full_n500.jsonl files.
Output: data/longmemeval/cleaned/judged_{method}_full_n500.jsonl
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
LM_DIR = REPO / "data" / "longmemeval" / "cleaned"

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.cn/v1/chat/completions")
MODEL = "glm-4.7"
CONC = int(os.environ.get("LM_JUDGE_CONC", "12"))


JUDGE_PROMPT = """You are scoring a question-answering result against a gold reference.

Question: {question}
Gold reference answer: {gold}
Predicted answer: {pred}

Is the predicted answer essentially correct, considering:
- Minor wording or paraphrase is OK
- For numeric answers, exact match required (or within stated tolerance, e.g. "7 days. 8 days also acceptable")
- For preference questions, the predicted answer should reflect the user's actual preferences described in gold (specific brands, types, etc. matter)
- "I don't know" or empty answers are NO
- Answers that contradict gold are NO

Reply with EXACTLY one token: YES or NO."""


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


async def judge_one(sem, client, row):
    async with sem:
        prompt = JUDGE_PROMPT.replace("{question}", row["question"])\
                              .replace("{gold}", str(row["answer"]))\
                              .replace("{pred}", row["predicted"] or "")
        out, err = await call_glm(client, prompt)
        yes = (out or "").strip().upper().startswith("YES")
        return {**row, "judge_yes": yes, "judge_raw": out, "judge_error": err}


async def run(rows):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        return await asyncio.gather(*[judge_one(sem, client, r) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["full_context", "oracle", "bm25_chunks", "dense_chunks", "hybrid_rrf"])
    ap.add_argument("--out-name", default="full_n500")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR GLM_API_KEY missing"); sys.exit(1)

    src = LM_DIR / f"eval_{args.method}_{args.out_name}.jsonl"
    if not src.exists():
        print(f"ERROR: {src} missing"); sys.exit(1)

    rows = [json.loads(l) for l in open(src)]
    print(f"{len(rows)} rows from {src}")
    print(f"  CONC={CONC}")
    t0 = time.time()
    judged = asyncio.run(run(rows))
    print(f"  done {time.time()-t0:.0f}s")

    from collections import defaultdict
    by_type = defaultdict(list)
    for r in judged:
        by_type[r["question_type"]].append(int(r["judge_yes"]))
    overall_yes = sum(int(r["judge_yes"]) for r in judged) / len(judged)
    overall_lenient = sum(r.get("lenient_score", 0) for r in judged) / len(judged)
    print(f"\nOverall judge YES: {overall_yes:.3f}  (vs lenient {overall_lenient:.3f})")
    print(f"\n{'type':<28} {'n':>4} {'judge':>6} {'lenient':>8}")
    print("-" * 56)
    for t in sorted(by_type):
        scores = by_type[t]
        l_scores = [r.get("lenient_score", 0) for r in judged if r["question_type"] == t]
        print(f"  {t:<26} {len(scores):>4} {sum(scores)/len(scores):>6.3f} {sum(l_scores)/len(l_scores):>8.3f}")

    out_path = LM_DIR / f"judged_{args.method}_{args.out_name}.jsonl"
    with open(out_path, "w") as f:
        for r in judged: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
