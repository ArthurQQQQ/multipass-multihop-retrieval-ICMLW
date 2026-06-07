"""03_run_baselines.py - Run 5 DKMP baselines per (context, method).

Methods:
  M0 full_context  : feed entire context_text to GLM-4.7
  M1 dense_chunks  : 200-token sliding chunk -> BGE-M3 -> top-K=10 -> reader
  M2 bm25_chunks   : same chunks -> BM25 top-K=10 -> reader
  M3 hybrid_rrf    : dense top-30 ∪ bm25 top-30 -> RRF -> top-K=10 -> reader
  M5 oracle        : feed needle_text + question -> reader

For chunked methods, also record retrieved chunks and recall@k of needle.

Idempotent: skips qa_id+method combos already in output.
Output: data/dkmp/predicted_v0.jsonl (single combined file)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm import call_glm_async, configure, make_client  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONTEXTS_FILE = REPO / "data/dkmp/contexts_v0.jsonl"
OUTPUT = REPO / "data/dkmp/predicted_v0.jsonl"

CHUNK_TOKENS = 200
CHUNK_STRIDE = 150  # 50-token overlap
TOP_K = 10
TOP_DENSE = 30
TOP_BM25 = 30


def render_reader_prompt(question: str, context: str) -> str:
    # v0p1 prompt: less conservative, encourages causal/directional inference
    # from the text rather than defaulting to "I don't know".
    return f"""You answer a question using only the provided text.

Text:
{context}

Question: {question}

Answer in ONE short sentence (max 25 words), grounded in the text above. For causal or directional questions, identify the cause-and-effect relationship from the events described in the text (which event led to which). Always commit to your best answer based on the evidence given.
"""


# --------- Chunking ---------
def chunk_text(enc, text: str, chunk_size: int = CHUNK_TOKENS, stride: int = CHUNK_STRIDE) -> list[tuple[int, int, str]]:
    """Return list of (start_token, end_token, chunk_text)."""
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        end = min(i + chunk_size, len(toks))
        chunks.append((i, end, enc.decode(toks[i:end])))
        if end == len(toks):
            break
        i += stride
    return chunks


# --------- BM25 (lite, in-memory) ---------
class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1)
        self.df: dict[str, int] = {}
        for doc in corpus_tokens:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()}

    def score(self, doc: list[str], query: list[str]) -> float:
        from collections import Counter
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for q in query:
            if q not in tf:
                continue
            idf = self.idf.get(q, 0.0)
            num = tf[q] * (self.k1 + 1)
            den = tf[q] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * num / den
        return s

    def topk(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scores = [(i, self.score(d, query)) for i, d in enumerate(self.corpus)]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def simple_tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


# --------- Dense (BGE-M3 via sentence-transformers) ---------
_bge_model = None


def get_bge():
    global _bge_model
    if _bge_model is None:
        print("Loading BGE-M3...", flush=True)
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device: {device}", flush=True)
        _bge_model = SentenceTransformer("BAAI/bge-m3", device=device)
        print("Loaded.", flush=True)
    return _bge_model


def dense_topk(chunks_text: list[str], query: str, k: int) -> list[tuple[int, float]]:
    model = get_bge()
    chunk_emb = model.encode(chunks_text, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = (chunk_emb @ q_emb).tolist()
    scored = sorted(enumerate(sims), key=lambda x: -x[1])
    return scored[:k]


# --------- RRF fusion ---------
def rrf_fuse(rankings: list[list[int]], k_rrf: float = 60.0) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# --------- Recall metric ---------
def recall_of_needle(retrieved_chunks: list[str], needle_sentences: list[str]) -> float:
    """Fraction of needle sentences that appear in any retrieved chunk."""
    if not needle_sentences:
        return 0.0
    hits = 0
    for ns in needle_sentences:
        for ch in retrieved_chunks:
            if ns.strip() in ch:
                hits += 1
                break
    return hits / len(needle_sentences)


# --------- Per-context retrieval cache ---------
class RetrievalCache:
    """Cache chunks + dense embeddings + bm25 index per context to share across methods."""

    def __init__(self, enc):
        self.enc = enc
        self._cache: dict[str, dict] = {}

    def get(self, qa_id: str, context_text: str):
        if qa_id in self._cache:
            return self._cache[qa_id]
        chunks = chunk_text(self.enc, context_text)
        chunk_texts = [c[2] for c in chunks]
        chunk_tokens = [simple_tokenize(c) for c in chunk_texts]
        bm25 = BM25(chunk_tokens)
        # dense embeddings computed lazily
        e = {"chunks": chunks, "chunk_texts": chunk_texts, "bm25": bm25, "dense_emb": None}
        self._cache[qa_id] = e
        return e

    def get_dense_emb(self, qa_id: str):
        e = self._cache[qa_id]
        if e["dense_emb"] is None:
            model = get_bge()
            emb = model.encode(
                e["chunk_texts"],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            e["dense_emb"] = np.asarray(emb, dtype=np.float32)
        return e["dense_emb"]


# --------- Method runners ---------
async def run_M0(sem, client, ctx_row: dict) -> dict:
    """Full context."""
    async with sem:
        prompt = render_reader_prompt(ctx_row["question"], ctx_row["context_text"])
        t0 = time.time()
        text, err = await call_glm_async(client, prompt, max_tokens=120)
        dt = time.time() - t0
    return {
        "qa_id": ctx_row["qa_id"],
        "method": "M0",
        "predicted_answer": text,
        "reader_error": err,
        "retrieved_chunks": [],
        "recall_needle": 1.0,  # full context always has needle
        "retrieved_tokens": ctx_row["actual_length"],
        "latency_s": dt,
    }


async def run_M5(sem, client, ctx_row: dict) -> dict:
    """Oracle: feed only the needle."""
    async with sem:
        prompt = render_reader_prompt(ctx_row["question"], ctx_row["needle_text"])
        t0 = time.time()
        text, err = await call_glm_async(client, prompt, max_tokens=120)
        dt = time.time() - t0
    enc = tiktoken.get_encoding("cl100k_base")
    return {
        "qa_id": ctx_row["qa_id"],
        "method": "M5",
        "predicted_answer": text,
        "reader_error": err,
        "retrieved_chunks": [ctx_row["needle_text"]],
        "recall_needle": 1.0,
        "retrieved_tokens": len(enc.encode(ctx_row["needle_text"])),
        "latency_s": dt,
    }


async def run_chunked(sem, client, ctx_row: dict, cache: RetrievalCache, method: str) -> dict:
    """M1 dense, M2 bm25, M3 hybrid_rrf."""
    e = cache.get(ctx_row["qa_id"], ctx_row["context_text"])
    chunks_text = e["chunk_texts"]
    bm25 = e["bm25"]
    q = ctx_row["question"]
    q_tokens = simple_tokenize(q)

    if method == "M2":  # bm25 only
        ranked = bm25.topk(q_tokens, TOP_K)
        topk_idx = [i for i, _ in ranked]
    elif method == "M1":  # dense only
        emb = cache.get_dense_emb(ctx_row["qa_id"])
        model = get_bge()
        q_emb = np.asarray(model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)
        sims = (emb @ q_emb).tolist()
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])[:TOP_K]
        topk_idx = [i for i, _ in ranked]
    elif method == "M3":  # hybrid RRF
        emb = cache.get_dense_emb(ctx_row["qa_id"])
        model = get_bge()
        q_emb = np.asarray(model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)
        sims = (emb @ q_emb).tolist()
        dense_rank = [i for i, _ in sorted(enumerate(sims), key=lambda x: -x[1])][:TOP_DENSE]
        bm25_rank = [i for i, _ in bm25.topk(q_tokens, TOP_BM25)]
        fused = rrf_fuse([dense_rank, bm25_rank])
        topk_idx = [i for i, _ in fused[:TOP_K]]
    else:
        raise ValueError(method)

    retrieved = [chunks_text[i] for i in topk_idx]
    enc = tiktoken.get_encoding("cl100k_base")
    retrieved_text = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(retrieved))
    retrieved_tokens = len(enc.encode(retrieved_text))
    recall = recall_of_needle(retrieved, ctx_row["needle_sentences"])

    async with sem:
        prompt = render_reader_prompt(q, retrieved_text)
        t0 = time.time()
        text, err = await call_glm_async(client, prompt, max_tokens=120)
        dt = time.time() - t0
    return {
        "qa_id": ctx_row["qa_id"],
        "method": method,
        "predicted_answer": text,
        "reader_error": err,
        "retrieved_chunk_idx": topk_idx,
        "recall_needle": recall,
        "retrieved_tokens": retrieved_tokens,
        "latency_s": dt,
    }


def load_existing() -> set[tuple[str, str]]:
    if not OUTPUT.exists():
        return set()
    out = set()
    for line in OUTPUT.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            out.add((r["qa_id"], r["method"]))
        except Exception:
            continue
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="M5,M1,M2,M3,M0",
                    help="Comma-sep order. Default puts M0 last (slowest, longest contexts)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--smoke", type=int, default=0, help="Run only first N contexts")
    ap.add_argument("--lengths", default="", help="Filter to these lengths (comma-sep)")
    args = ap.parse_args()

    configure()
    enc = tiktoken.get_encoding("cl100k_base")
    methods = args.methods.split(",")
    contexts = [json.loads(l) for l in CONTEXTS_FILE.read_text().splitlines() if l.strip()]
    if args.lengths:
        ls = {int(x) for x in args.lengths.split(",")}
        contexts = [c for c in contexts if c["target_length"] in ls]
    if args.smoke:
        contexts = contexts[: args.smoke]
    print(f"Contexts: {len(contexts)} | methods: {methods}", flush=True)

    done = load_existing()
    print(f"Already done: {len(done)}", flush=True)

    cache = RetrievalCache(enc)
    sem = asyncio.Semaphore(args.concurrency)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fout = OUTPUT.open("a")

    async with make_client() as client:
        for method in methods:
            todo_ctxs = [c for c in contexts if (c["qa_id"], method) not in done]
            print(f"\n=== METHOD {method} | todo: {len(todo_ctxs)} ===", flush=True)
            if not todo_ctxs:
                continue

            if method == "M0":
                tasks = [run_M0(sem, client, c) for c in todo_ctxs]
            elif method == "M5":
                tasks = [run_M5(sem, client, c) for c in todo_ctxs]
            else:
                # Chunked methods: pre-warm dense embeddings sequentially per context
                # but reader call is async/concurrent
                if method in ("M1", "M3"):
                    for c in todo_ctxs:
                        cache.get(c["qa_id"], c["context_text"])
                        cache.get_dense_emb(c["qa_id"])
                tasks = [run_chunked(sem, client, c, cache, method) for c in todo_ctxs]

            n_ok, n_err = 0, 0
            t0 = time.time()
            for fut in asyncio.as_completed(tasks):
                row = await fut
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                if row.get("reader_error"):
                    n_err += 1
                else:
                    n_ok += 1
                if (n_ok + n_err) % 25 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{method}] {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={elapsed:.0f}s", flush=True)
            elapsed = time.time() - t0
            print(f"  [{method}] DONE {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={elapsed:.0f}s", flush=True)

    fout.close()
    print(f"\nAll done. Output: {OUTPUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
