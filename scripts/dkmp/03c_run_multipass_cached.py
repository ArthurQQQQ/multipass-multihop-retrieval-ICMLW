"""03c_run_multipass_cached.py - M6 multi-pass using pre-cached embeddings.

Loads data/dkmp/embeddings_cache_v0.npz (built by 00_cache_embeddings.py).
GLM-only at runtime — no BGE-M3 forward pass needed.

For each context:
  Pass 1: hybrid_rrf top-K=10 (cached emb + cached BM25)
  Bridge: GLM extracts intermediate entity from pass-1 chunks
  Pass 2: hybrid_rrf top-K=10 with bridge query (need to embed bridge query;
          single small embedding is fast)
  Final: GLM reads pass1 ∪ pass2 chunks and answers
"""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm import call_glm_async, configure, make_client  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONTEXTS_FILE = REPO / "data/dkmp/contexts_v0.jsonl"
EMB_FILE = REPO / "data/dkmp/embeddings_cache_v0.npz"
CHUNKS_FILE = REPO / "data/dkmp/chunks_cache_v0.jsonl"
OUTPUT = REPO / "data/dkmp/predicted_v0.jsonl"

TOP_K = 10
TOP_DENSE = 30
TOP_BM25 = 30


READER_PROMPT = """You answer a question using only the provided text.

Text:
{T}

Question: {Q}

Answer in ONE short sentence (max 25 words), grounded in the text above. For causal or directional questions, identify the cause-and-effect relationship from the events described in the text (which event led to which). Always commit to your best answer based on the evidence given.
"""

BRIDGE_PROMPT = """You are helping retrieve information for a multi-hop question. Given the question and the snippets retrieved so far, identify ONE specific intermediate entity (object, place, thing) mentioned in the snippets that would need a follow-up lookup to answer the question.

Question: {question}

Retrieved snippets:
{snippets}

Reply with ONE short noun phrase (≤ 6 words) — the entity to look up next. If the snippets already fully answer the question, reply: NONE.
"""


# -- BM25 lite --
class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1)
        df = {}
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - v + 0.5) / (v + 0.5) + 1.0) for t, v in df.items()}

    def topk(self, query_tokens, k):
        scores = []
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            dl = len(doc)
            s = 0.0
            for q in query_tokens:
                if q not in tf:
                    continue
                idf = self.idf.get(q, 0.0)
                num = tf[q] * (self.k1 + 1)
                den = tf[q] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


import re
def simple_tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rrf_fuse(rankings, k_rrf=60.0):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_topk(emb, bm25, q_emb, q_tokens, k=TOP_K, top_dense=TOP_DENSE, top_bm25=TOP_BM25):
    sims = (emb @ q_emb).tolist()
    dense_rank = [i for i, _ in sorted(enumerate(sims), key=lambda x: -x[1])][:top_dense]
    bm25_rank = [i for i, _ in bm25.topk(q_tokens, top_bm25)]
    fused = rrf_fuse([dense_rank, bm25_rank])
    return [i for i, _ in fused[:k]]


def recall_of_needle(retrieved_chunks, needle_sentences):
    if not needle_sentences:
        return 0.0
    hits = 0
    for ns in needle_sentences:
        if any(ns.strip() in ch for ch in retrieved_chunks):
            hits += 1
    return hits / len(needle_sentences)


def load_existing() -> set[str]:
    if not OUTPUT.exists():
        return set()
    out = set()
    for line in OUTPUT.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("method") == "M6":
                out.add(r["qa_id"])
        except Exception:
            continue
    return out


async def run_one(sem, client, ctx, cache, model, top_k):
    qa_id = ctx["qa_id"]
    chunks = cache[qa_id]["chunks"]
    bm25_tokens = cache[qa_id]["bm25_tokens"]
    emb = cache[qa_id]["emb"]
    bm25 = cache[qa_id]["bm25"]
    enc = tiktoken.get_encoding("cl100k_base")
    question = ctx["question"]

    t0 = time.time()
    # Pass 1
    q_emb = np.asarray(
        model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0],
        dtype=np.float32,
    )
    idx1 = hybrid_topk(emb, bm25, q_emb, simple_tokenize(question), k=top_k)
    retrieved1 = [chunks[i] for i in idx1]

    snippets_text = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(retrieved1))
    bridge_prompt = BRIDGE_PROMPT.format(question=question, snippets=snippets_text)
    async with sem:
        bridge, bridge_err = await call_glm_async(client, bridge_prompt, max_tokens=30)

    idx2: list[int] = []
    if bridge.strip().upper() != "NONE" and bridge.strip():
        b_emb = np.asarray(
            model.encode([bridge], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        idx2 = hybrid_topk(emb, bm25, b_emb, simple_tokenize(bridge), k=top_k)

    seen = set()
    merged_idx: list[int] = []
    for i in idx1 + idx2:
        if i not in seen:
            merged_idx.append(i)
            seen.add(i)
    merged_text = [chunks[i] for i in merged_idx]
    merged_concat = "\n\n".join(f"[{j}] {t}" for j, t in enumerate(merged_text))

    final_prompt = READER_PROMPT.format(T=merged_concat, Q=question)
    async with sem:
        ans, ans_err = await call_glm_async(client, final_prompt, max_tokens=120)
    dt = time.time() - t0

    return {
        "qa_id": qa_id,
        "method": "M6",
        "predicted_answer": ans,
        "reader_error": ans_err or bridge_err,
        "bridge_query": bridge,
        "retrieved_chunk_idx": merged_idx,
        "retrieved_chunk_idx_pass1": idx1,
        "retrieved_chunk_idx_pass2": idx2,
        "recall_needle": recall_of_needle(merged_text, ctx["needle_sentences"]),
        "retrieved_tokens": len(enc.encode(merged_concat)),
        "latency_s": dt,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--key_filter", default="")
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    args = ap.parse_args()

    configure()
    contexts = [json.loads(l) for l in CONTEXTS_FILE.read_text().splitlines() if l.strip()]
    if args.key_filter:
        keys = set(args.key_filter.split(","))
        contexts = [c for c in contexts if c["key_type"] in keys]
    if args.max_length:
        contexts = [c for c in contexts if c["target_length"] <= args.max_length]

    done = load_existing()
    todo = [c for c in contexts if c["qa_id"] not in done]
    print(f"Total: {len(contexts)} | done: {len(done)} | todo: {len(todo)}", flush=True)
    if not todo:
        return

    print("Loading embedding cache...", flush=True)
    npz = np.load(EMB_FILE, allow_pickle=True)
    emb_all = npz["emb"]
    offsets = npz["chunk_offsets"]
    qa_ids_arr = npz["qa_ids"]
    qa_to_idx = {str(q): i for i, q in enumerate(qa_ids_arr)}

    chunks_meta = {json.loads(l)["qa_id"]: json.loads(l) for l in CHUNKS_FILE.read_text().splitlines() if l.strip()}

    cache: dict[str, dict] = {}
    for c in todo:
        qa_id = c["qa_id"]
        if qa_id not in qa_to_idx:
            print(f"WARN: {qa_id} not in cache, skipping", flush=True)
            continue
        idx = qa_to_idx[qa_id]
        emb = emb_all[offsets[idx]:offsets[idx + 1]]
        meta = chunks_meta[qa_id]
        cache[qa_id] = {
            "chunks": meta["chunks"],
            "bm25_tokens": meta["bm25_tokens"],
            "emb": emb,
            "bm25": BM25(meta["bm25_tokens"]),
        }
    print(f"Loaded {len(cache)} contexts from cache", flush=True)

    print("Loading BGE-M3 (for query embedding only)...", flush=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("Loaded.", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    todo = [c for c in todo if c["qa_id"] in cache]
    fout = OUTPUT.open("a")
    async with make_client() as client:
        tasks = [run_one(sem, client, c, cache, model, args.top_k) for c in todo]
        n_ok = n_err = 0
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
                el = time.time() - t0
                print(f"  [M6] {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={el:.0f}s", flush=True)
        el = time.time() - t0
        print(f"  [M6] DONE {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={el:.0f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    asyncio.run(main())
