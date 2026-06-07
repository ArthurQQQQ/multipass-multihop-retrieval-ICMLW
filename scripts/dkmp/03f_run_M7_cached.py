"""03f_run_M7_cached.py - M7: 3-pass multi-pass on K7 using cached embeddings.

Three-pass retrieval (cached emb, only GLM + small query embeddings at runtime):
1. Pass 1: hybrid_topk(question)
2. Bridge1: GLM extracts FIRST intermediate entity
3. Pass 2: hybrid_topk(bridge1)
4. Bridge2: GLM extracts SECOND intermediate entity (from pass1∪pass2)
5. Pass 3: hybrid_topk(bridge2)
6. Reader: GLM answers from union of all passes

Output: appends rows with method="M7" to predicted_v0.jsonl.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm import call_glm_async, configure, make_client  # noqa: E402

# Re-use BM25, hybrid_topk, recall_of_needle from 03c
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mpc", Path(__file__).resolve().parent / "03c_run_multipass_cached.py"
)
_mpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mpc)
BM25 = _mpc.BM25
hybrid_topk = _mpc.hybrid_topk
simple_tokenize = _mpc.simple_tokenize
recall_of_needle = _mpc.recall_of_needle
TOP_K = _mpc.TOP_K
READER_PROMPT = _mpc.READER_PROMPT

REPO = Path(__file__).resolve().parents[2]
CONTEXTS_FILE = REPO / "data/dkmp/contexts_v0.jsonl"
EMB_FILE = REPO / "data/dkmp/embeddings_cache_v0.npz"
CHUNKS_FILE = REPO / "data/dkmp/chunks_cache_v0.jsonl"
OUTPUT = REPO / "data/dkmp/predicted_v0.jsonl"


BRIDGE_PROMPT_1 = """You are helping retrieve information for a 3-hop chained-fact question. The question's answer requires linking through TWO intermediate entities. Identify the FIRST intermediate entity (the one most directly related to the question's anchor entity).

Question: {question}

Retrieved snippets:
{snippets}

Reply with ONE short noun phrase (≤ 6 words) — the FIRST intermediate entity to look up. If snippets already fully answer, reply: NONE.
"""

BRIDGE_PROMPT_2 = """You are helping retrieve information for a 3-hop chained-fact question. We have done 2 retrieval passes. Identify the SECOND intermediate entity needed (the one closer to the final answer; usually mentioned in pass-2 snippets).

Question: {question}

All retrieved snippets so far (pass 1 and pass 2):
{snippets}

Reply with ONE short noun phrase (≤ 6 words) — the SECOND intermediate entity to look up. If snippets already fully answer, reply: NONE.
"""


def load_existing() -> set[str]:
    if not OUTPUT.exists():
        return set()
    out = set()
    for line in OUTPUT.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("method") == "M7":
                out.add(r["qa_id"])
        except Exception:
            continue
    return out


async def run_one(sem, client, ctx, cache, model, top_k):
    qa_id = ctx["qa_id"]
    chunks = cache[qa_id]["chunks"]
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

    # Bridge 1
    snippets_text = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(retrieved1))
    b1_prompt = BRIDGE_PROMPT_1.format(question=question, snippets=snippets_text)
    async with sem:
        bridge1, b1_err = await call_glm_async(client, b1_prompt, max_tokens=30)

    # Pass 2
    idx2: list[int] = []
    if bridge1.strip().upper() != "NONE" and bridge1.strip():
        b1_emb = np.asarray(
            model.encode([bridge1.strip()], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        idx2 = hybrid_topk(emb, bm25, b1_emb, simple_tokenize(bridge1), k=top_k)

    # Bridge 2
    seen = set(); merged12: list[int] = []
    for i in idx1 + idx2:
        if i not in seen:
            merged12.append(i); seen.add(i)
    snippets_text2 = "\n\n".join(f"[{i+1}] {chunks[idx]}" for i, idx in enumerate(merged12))
    b2_prompt = BRIDGE_PROMPT_2.format(question=question, snippets=snippets_text2)
    async with sem:
        bridge2, b2_err = await call_glm_async(client, b2_prompt, max_tokens=30)

    # Pass 3
    idx3: list[int] = []
    if bridge2.strip().upper() != "NONE" and bridge2.strip():
        b2_emb = np.asarray(
            model.encode([bridge2.strip()], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        idx3 = hybrid_topk(emb, bm25, b2_emb, simple_tokenize(bridge2), k=top_k)

    # Combine all
    seen = set(); merged_idx: list[int] = []
    for i in idx1 + idx2 + idx3:
        if i not in seen:
            merged_idx.append(i); seen.add(i)
    merged_text = [chunks[i] for i in merged_idx]
    merged_concat = "\n\n".join(f"[{j}] {t}" for j, t in enumerate(merged_text))

    # Reader
    final_prompt = READER_PROMPT.format(T=merged_concat, Q=question)
    async with sem:
        ans, ans_err = await call_glm_async(client, final_prompt, max_tokens=120)
    dt = time.time() - t0

    return {
        "qa_id": qa_id,
        "method": "M7",
        "predicted_answer": ans,
        "reader_error": ans_err or b1_err or b2_err,
        "bridge_query": bridge1.strip(),
        "bridge_query_2": bridge2.strip(),
        "retrieved_chunk_idx": merged_idx,
        "retrieved_chunk_idx_pass1": idx1,
        "retrieved_chunk_idx_pass2": idx2,
        "retrieved_chunk_idx_pass3": idx3,
        "recall_needle": recall_of_needle(merged_text, ctx["needle_sentences"]),
        "retrieved_tokens": len(enc.encode(merged_concat)),
        "latency_s": dt,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--key_filter", default="K7")
    ap.add_argument("--top_k", type=int, default=TOP_K)
    args = ap.parse_args()

    configure()
    contexts = [json.loads(l) for l in CONTEXTS_FILE.read_text().splitlines() if l.strip()]
    if args.key_filter:
        keys = set(args.key_filter.split(","))
        contexts = [c for c in contexts if c["key_type"] in keys]

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
            "emb": emb,
            "bm25": BM25(meta["bm25_tokens"]),
        }
    print(f"Loaded {len(cache)} contexts from cache", flush=True)

    print("Loading BGE-M3 (for query embeddings)...", flush=True)
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
                print(f"  [M7] {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={el:.0f}s", flush=True)
        el = time.time() - t0
        print(f"  [M7] DONE {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={el:.0f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    asyncio.run(main())
