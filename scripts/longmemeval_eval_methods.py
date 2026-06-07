#!/usr/bin/env python3
"""longmemeval_eval_methods.py - GLM-4.7 reader on LongMemEval with retrieval methods.

Methods:
  - oracle: only answer_session_ids sessions
  - full_context: all haystack_sessions concatenated (truncated to budget)
  - dense_chunks: BGE-M3 dense retrieval at chunk granularity
  - bm25_chunks: BM25 sparse retrieval
  - hybrid_rrf: dense + bm25 reciprocal rank fusion

Output: data/longmemeval/cleaned/eval_{method}_{out_name}.jsonl
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
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LM_DIR = REPO / "data" / "longmemeval" / "cleaned"

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.cn/v1/chat/completions")
MODEL = os.environ.get("LM_MODEL", "glm-4.7")
CONC = int(os.environ.get("LM_CONC", "8"))
CHUNK_CHARS = 1200  # set from args in main


def chunk_sessions(sessions, chunk_chars=1200):
    """Chunk a list of sessions into char-bounded chunks. Each chunk has session boundary markers."""
    chunks = []
    for sidx, sess in enumerate(sessions):
        # Linearize session
        text = ""
        for turn in sess:
            text += f"  {turn['role'].upper()}: {turn['content']}\n"
        # Chunk this session
        i = 0
        cidx = 0
        while i < len(text):
            end = min(i + chunk_chars, len(text))
            # Try to break on newline if near end
            if end < len(text):
                nl = text.rfind("\n", i, end)
                if nl > i + chunk_chars // 2:
                    end = nl
            chunks.append({
                "session_idx": sidx,
                "chunk_idx_in_session": cidx,
                "text": text[i:end],
                "char_start": i,
                "char_end": end,
            })
            i = end
            cidx += 1
    return chunks


def render_reader(question, ctx_blocks, ctx_label="memory", qdate=""):
    body = "\n\n".join(f"[{ctx_label.title()} {i+1}]\n{b}" for i, b in enumerate(ctx_blocks))
    return f"""You are answering a question based on prior conversation history.

{ctx_label.title()}:
{body}

{('Question asked at: ' + qdate) if qdate else ''}
Question: {question}

Answer concisely (one sentence). Provide a direct, specific answer."""


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
    if not pred: return 0
    p = re.sub(r"\s+", " ", str(pred).lower().strip())
    gold_str = str(gold).lower()
    candidates = [gold_str.strip()]
    for sep in [r"\.\s+", r";\s*", r"\s+or\s+"]:
        for c in list(candidates):
            for part in re.split(sep, c):
                part = part.strip().strip("'\"`,.")
                if part and len(part) >= 2:
                    candidates.append(part)
    candidates = list(set(candidates))
    for cand in candidates:
        cand_norm = re.sub(r"\s+", " ", cand)
        if cand_norm and cand_norm in p:
            return 1
    return 0


def bm25_score(query_terms, chunks):
    """Simple BM25 scoring. chunks is list of dicts with 'text'."""
    # Tokenize
    def tok(s):
        return [w for w in re.findall(r"[a-zA-Z0-9]+", s.lower()) if len(w) > 1]
    docs = [tok(c["text"]) for c in chunks]
    qtoks = [w for w in tok(query_terms) if w not in {"the","a","an","of","is","are","was","were","i","you","my","your","what","which","who","when","where","how","do","did","does"}]
    N = len(docs)
    if N == 0: return np.zeros(0)
    avgdl = sum(len(d) for d in docs) / N
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    k1, b = 1.5, 0.75
    scores = np.zeros(N)
    for i, d in enumerate(docs):
        score = 0.0
        d_len = len(d)
        from collections import Counter
        d_freq = Counter(d)
        for q in qtoks:
            if q not in df: continue
            idf = np.log((N - df[q] + 0.5) / (df[q] + 0.5) + 1)
            tf = d_freq.get(q, 0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * d_len / avgdl))
        scores[i] = score
    return scores


# Dense retrieval — lazy load BGE-M3 via sentence-transformers (DKMP convention)
_BGE = None
def get_bge():
    global _BGE
    if _BGE is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading BGE-M3 on {device}...", flush=True)
        _BGE = SentenceTransformer("BAAI/bge-m3", device=device)
        print("  Loaded.", flush=True)
    return _BGE


def dense_score(query, chunks):
    bge = get_bge()
    texts = [c["text"] for c in chunks]
    q_emb = bge.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    c_emb = bge.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    return (c_emb @ q_emb).flatten()


def select_context(method, question, sessions, answer_session_ids, char_budget=8000, K=10):
    """Return list of context blocks based on method."""
    sessions_pairs = list(zip(sessions["ids"], sessions["sessions"]))
    if method == "oracle":
        ans_set = set(answer_session_ids)
        oracle = [(sid, s) for sid, s in sessions_pairs if sid in ans_set]
        if not oracle: oracle = sessions_pairs
        blocks = []
        used = 0
        for sid, sess in oracle:
            text = "".join(f"  {t['role'].upper()}: {t['content']}\n" for t in sess)
            if used + len(text) > char_budget and blocks: break
            blocks.append(text); used += len(text)
        return blocks
    elif method == "full_context":
        text = "\n".join(
            f"=== Session {i+1} ===\n" + "".join(f"  {t['role'].upper()}: {t['content']}\n" for t in sess)
            for i, (_, sess) in enumerate(sessions_pairs)
        )
        return [text[:char_budget]]
    elif method in ("dense_chunks", "bm25_chunks", "hybrid_rrf"):
        all_chunks = chunk_sessions([sess for _, sess in sessions_pairs], chunk_chars=CHUNK_CHARS)
        if not all_chunks: return []
        if method == "bm25_chunks":
            scores = bm25_score(question, all_chunks)
            order = np.argsort(-scores)
        elif method == "dense_chunks":
            scores = dense_score(question, all_chunks)
            order = np.argsort(-scores)
        else:  # hybrid_rrf
            sb = bm25_score(question, all_chunks)
            sd = dense_score(question, all_chunks)
            ord_b = np.argsort(-sb); ord_d = np.argsort(-sd)
            rank_b = np.zeros(len(all_chunks)); rank_d = np.zeros(len(all_chunks))
            for r, i in enumerate(ord_b): rank_b[i] = r
            for r, i in enumerate(ord_d): rank_d[i] = r
            rrf = 1.0/(60+rank_b) + 1.0/(60+rank_d)
            order = np.argsort(-rrf)
        chosen = []
        used = 0
        for rank in order[:K*3]:  # over-fetch then char-budget trim
            c = all_chunks[int(rank)]
            if used + len(c["text"]) > char_budget and chosen: break
            chosen.append(c["text"]); used += len(c["text"])
            if len(chosen) >= K: break
        return chosen
    else:
        raise ValueError(f"unknown method {method}")


async def evaluate_one(sem, client, q, method, char_budget, K):
    async with sem:
        sessions = {"ids": q["haystack_session_ids"], "sessions": q["haystack_sessions"]}
        blocks = select_context(method, q["question"], sessions, q["answer_session_ids"],
                                char_budget=char_budget, K=K)
        prompt = render_reader(q["question"], blocks, ctx_label="memory" if method != "full_context" else "history",
                              qdate=q.get("question_date", ""))
        out, err = await call_glm(client, prompt)
        return {**{k: q.get(k) for k in ["question_id", "question_type", "question", "answer"]},
                "method": method,
                "predicted": out, "lenient_score": lenient_score(out, q["answer"]),
                "error": err,
                "n_blocks": len(blocks),
                "ctx_chars": sum(len(b) for b in blocks)}


async def run(questions, method, char_budget, K):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        return await asyncio.gather(*[evaluate_one(sem, client, q, method, char_budget, K) for q in questions])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["oracle", "full_context", "dense_chunks", "bm25_chunks", "hybrid_rrf"])
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--char-budget", type=int, default=8000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--chunk-chars", type=int, default=1200, help="Chunk size for chunking sessions")
    ap.add_argument("--out-name", default="balanced_n48")
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
        print(f"  balanced: {args.per_type} per type → {len(data)} total")

    global CHUNK_CHARS
    CHUNK_CHARS = args.chunk_chars
    print(f"Method={args.method}  K={args.K}  budget={args.char_budget}c  chunk={args.chunk_chars}c  CONC={CONC}")
    t0 = time.time()
    results = asyncio.run(run(data, args.method, args.char_budget, args.K))
    elapsed = time.time() - t0
    print(f"  done {elapsed:.0f}s")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["lenient_score"])
    print(f"\n{'type':<28} {'n':>4} {'lenient':>8} {'avg_chars':>10}")
    print("-" * 56)
    for t in sorted(by_type):
        scores = by_type[t]
        avgc = np.mean([r["ctx_chars"] for r in results if r["question_type"] == t])
        print(f"  {t:<26} {len(scores):>4} {sum(scores)/len(scores):>8.3f} {avgc:>10.0f}")
    overall = sum(r["lenient_score"] for r in results) / len(results)
    avgc_all = np.mean([r["ctx_chars"] for r in results])
    print(f"  {'OVERALL':<26} {len(results):>4} {overall:>8.3f} {avgc_all:>10.0f}")

    out_path = LM_DIR / f"eval_{args.method}_{args.out_name}.jsonl"
    with open(out_path, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
