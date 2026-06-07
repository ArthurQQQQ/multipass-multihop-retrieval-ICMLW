"""03b_run_multipass.py - M6 multi-pass retrieval baseline.

Two-pass retrieval:
1. Pass 1: dense+BM25 hybrid_rrf top-K=10 with original question
2. Bridge: GLM extracts "what intermediate entity do we need to look up next?"
   from the question + top-10 chunks
3. Pass 2: hybrid_rrf top-K=10 with the new bridge query
4. Final answer: GLM reads (top-10 pass 1 ∪ top-10 pass 2) and answers question

This tests whether simple multi-pass closes the K6 multi-hop gap WITHOUT graph
structure. If yes, the graph memory's value is questioned. If no, graph
traversal provides additional value (since graph already encodes the chain).

Output: appends rows with method="M6" to predicted_v0.jsonl.
Idempotent: skips qa_ids already with method=M6.
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

# Re-use chunking, BM25, dense from 03_run_baselines
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rb", Path(__file__).resolve().parent / "03_run_baselines.py"
)
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)
chunk_text = _rb.chunk_text
BM25 = _rb.BM25
simple_tokenize = _rb.simple_tokenize
get_bge = _rb.get_bge
recall_of_needle = _rb.recall_of_needle
rrf_fuse = _rb.rrf_fuse
RetrievalCache = _rb.RetrievalCache
render_reader_prompt = _rb.render_reader_prompt
TOP_K = _rb.TOP_K
TOP_DENSE = _rb.TOP_DENSE
TOP_BM25 = _rb.TOP_BM25

REPO = Path(__file__).resolve().parents[2]
CONTEXTS_FILE = REPO / "data/dkmp/contexts_v0.jsonl"
OUTPUT = REPO / "data/dkmp/predicted_v0.jsonl"


BRIDGE_PROMPT = """You are helping retrieve information for a multi-hop question. Given the question and the snippets retrieved so far, identify ONE specific intermediate entity (object, place, thing) mentioned in the snippets that would need a follow-up lookup to answer the question.

Question: {question}

Retrieved snippets:
{snippets}

Reply with ONE short noun phrase (≤ 6 words) — the entity to look up next. If the snippets already fully answer the question, reply: NONE.
"""


def hybrid_rrf_topk(cache: RetrievalCache, qa_id: str, query: str, k: int = TOP_K) -> list[int]:
    """Hybrid RRF on cached retrieval data."""
    e = cache.get(qa_id, "")  # context_text not needed if cache populated
    if not e:
        return []
    chunks_text = e["chunk_texts"]
    bm25 = e["bm25"]
    emb = cache.get_dense_emb(qa_id)
    model = get_bge()
    q_emb = np.asarray(
        model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
        dtype=np.float32,
    )
    sims = (emb @ q_emb).tolist()
    dense_rank = [i for i, _ in sorted(enumerate(sims), key=lambda x: -x[1])][:TOP_DENSE]
    bm25_rank = [i for i, _ in bm25.topk(simple_tokenize(query), TOP_BM25)]
    fused = rrf_fuse([dense_rank, bm25_rank])
    return [i for i, _ in fused[:k]]


async def run_M6(sem, client, ctx_row, cache: RetrievalCache) -> dict:
    qa_id = ctx_row["qa_id"]
    question = ctx_row["question"]
    e = cache.get(qa_id, ctx_row["context_text"])
    chunks_text = e["chunk_texts"]
    enc = tiktoken.get_encoding("cl100k_base")

    t0 = time.time()
    async with sem:
        # Pass 1
        idx1 = hybrid_rrf_topk(cache, qa_id, question, TOP_K)
        retrieved1 = [chunks_text[i] for i in idx1]

        # Bridge: ask GLM for intermediate entity
        snippets_text = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(retrieved1))
        bridge_prompt = BRIDGE_PROMPT.format(question=question, snippets=snippets_text)
        bridge, bridge_err = await call_glm_async(client, bridge_prompt, max_tokens=30)

        # Pass 2 (skip if NONE)
        idx2: list[int] = []
        if bridge.strip().upper() != "NONE" and bridge.strip():
            idx2 = hybrid_rrf_topk(cache, qa_id, bridge, TOP_K)

        # Combine retrievals (dedup, preserve order)
        seen = set()
        merged_idx: list[int] = []
        for i in idx1 + idx2:
            if i not in seen:
                merged_idx.append(i)
                seen.add(i)
        merged_text = [chunks_text[i] for i in merged_idx]
        merged_concat = "\n\n".join(f"[{j}] {t}" for j, t in enumerate(merged_text))

        # Final answer
        final_prompt = render_reader_prompt(question, merged_concat)
        ans, ans_err = await call_glm_async(client, final_prompt, max_tokens=120)
    dt = time.time() - t0

    retrieved_tokens = len(enc.encode(merged_concat))
    recall = recall_of_needle(merged_text, ctx_row["needle_sentences"])
    return {
        "qa_id": qa_id,
        "method": "M6",
        "predicted_answer": ans,
        "reader_error": ans_err or bridge_err,
        "bridge_query": bridge,
        "retrieved_chunk_idx": merged_idx,
        "retrieved_chunk_idx_pass1": idx1,
        "retrieved_chunk_idx_pass2": idx2,
        "recall_needle": recall,
        "retrieved_tokens": retrieved_tokens,
        "latency_s": dt,
    }


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


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--key_filter", default="", help="Only run for these keys (comma-sep)")
    ap.add_argument("--max_length", type=int, default=0, help="Skip contexts with target_length > this")
    args = ap.parse_args()

    configure()
    enc = tiktoken.get_encoding("cl100k_base")
    contexts = [json.loads(l) for l in CONTEXTS_FILE.read_text().splitlines() if l.strip()]
    if args.key_filter:
        keys = set(args.key_filter.split(","))
        contexts = [c for c in contexts if c["key_type"] in keys]
    if args.max_length:
        contexts = [c for c in contexts if c["target_length"] <= args.max_length]

    done = load_existing()
    todo = [c for c in contexts if c["qa_id"] not in done]
    print(f"Total contexts: {len(contexts)} | done: {len(done)} | todo: {len(todo)}", flush=True)
    if not todo:
        return

    cache = RetrievalCache(enc)

    # Pre-warm dense embeddings with progress
    print(f"Warming up retrieval cache for {len(todo)} contexts...", flush=True)
    t_warm = time.time()
    for i, c in enumerate(todo):
        cache.get(c["qa_id"], c["context_text"])
        cache.get_dense_emb(c["qa_id"])
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_warm
            print(f"  warmed {i+1}/{len(todo)} elapsed={elapsed:.0f}s", flush=True)
    print(f"  warm-up complete ({time.time()-t_warm:.0f}s)", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    fout = OUTPUT.open("a")
    async with make_client() as client:
        tasks = [run_M6(sem, client, c, cache) for c in todo]
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
                print(f"  [M6] {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={elapsed:.0f}s", flush=True)
        elapsed = time.time() - t0
        print(f"  [M6] DONE {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={elapsed:.0f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    asyncio.run(main())
