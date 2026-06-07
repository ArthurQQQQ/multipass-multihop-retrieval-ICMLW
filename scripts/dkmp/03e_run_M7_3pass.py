"""03e_run_M7_3pass.py - M7: 3-pass multi-pass retrieval (D-1=2 bridges) for K7 3-hop.

Three-pass retrieval:
1. Pass 1: hybrid_rrf top-K=10 with original question
2. Bridge1: GLM extracts first intermediate entity from pass-1 snippets
3. Pass 2: hybrid_rrf top-K=10 with bridge1
4. Bridge2: GLM extracts second intermediate entity from (pass1 ∪ pass2) snippets
5. Pass 3: hybrid_rrf top-K=10 with bridge2
6. Final answer: GLM reads (pass1 ∪ pass2 ∪ pass3) and answers

Tests if 3-pass closes the K7 3-hop gap. If yes, "agentic search" with D-1 passes
suffices for D-hop. If no, graph traversal still adds value.

Output: appends rows with method="M7" to predicted_v0.jsonl.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import tiktoken
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm import call_glm_async, configure, make_client  # noqa: E402

# Re-use everything from 03b
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mp", Path(__file__).resolve().parent / "03b_run_multipass.py"
)
_mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mp)
hybrid_rrf_topk = _mp.hybrid_rrf_topk
RetrievalCache = _mp.RetrievalCache
render_reader_prompt = _mp.render_reader_prompt
recall_of_needle = _mp.recall_of_needle
TOP_K = _mp.TOP_K

REPO = Path(__file__).resolve().parents[2]
CONTEXTS_FILE = REPO / "data/dkmp/contexts_v0.jsonl"
OUTPUT = REPO / "data/dkmp/predicted_v0.jsonl"


BRIDGE_PROMPT_1 = """You are helping retrieve information for a 3-hop question. The answer requires chaining through TWO intermediate entities. Identify the FIRST intermediate entity needed (the one most directly mentioned given the question's anchor).

Question: {question}

Retrieved snippets:
{snippets}

Reply with ONE short noun phrase (≤ 6 words) — the FIRST intermediate entity to look up. If the snippets already fully answer the question, reply: NONE.
"""

BRIDGE_PROMPT_2 = """You are helping retrieve information for a 3-hop question. We have already done 2 retrieval passes. Identify the SECOND intermediate entity needed (the one closer to the final answer, mentioned in the new snippets from pass 2).

Question: {question}

All retrieved snippets so far:
{snippets}

Reply with ONE short noun phrase (≤ 6 words) — the SECOND intermediate entity to look up. If the snippets already fully answer the question, reply: NONE.
"""


async def run_M7(sem, client, ctx_row, cache: RetrievalCache) -> dict:
    qa_id = ctx_row["qa_id"]
    question = ctx_row["question"]
    e = cache.get(qa_id, ctx_row["context_text"])
    chunks_text = e["chunk_texts"]
    enc = tiktoken.get_encoding("cl100k_base")

    t0 = time.time()
    bridge1 = bridge2 = ""
    bridge1_err = bridge2_err = None
    idx2: list[int] = []
    idx3: list[int] = []
    async with sem:
        # Pass 1
        idx1 = hybrid_rrf_topk(cache, qa_id, question, TOP_K)
        retrieved1 = [chunks_text[i] for i in idx1]

        # Bridge 1
        snippets_text = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(retrieved1))
        b1_prompt = BRIDGE_PROMPT_1.format(question=question, snippets=snippets_text)
        bridge1, bridge1_err = await call_glm_async(client, b1_prompt, max_tokens=30)

        # Pass 2
        if bridge1.strip().upper() != "NONE" and bridge1.strip():
            idx2 = hybrid_rrf_topk(cache, qa_id, bridge1.strip(), TOP_K)

        # Bridge 2 — uses pass1 + pass2 snippets
        seen = set(); merged12: list[int] = []
        for i in idx1 + idx2:
            if i not in seen:
                merged12.append(i); seen.add(i)
        merged12_text = [chunks_text[i] for i in merged12]
        snippets_text2 = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(merged12_text))
        b2_prompt = BRIDGE_PROMPT_2.format(question=question, snippets=snippets_text2)
        bridge2, bridge2_err = await call_glm_async(client, b2_prompt, max_tokens=30)

        # Pass 3
        if bridge2.strip().upper() != "NONE" and bridge2.strip():
            idx3 = hybrid_rrf_topk(cache, qa_id, bridge2.strip(), TOP_K)

        # Combine all
        seen = set(); merged_idx: list[int] = []
        for i in idx1 + idx2 + idx3:
            if i not in seen:
                merged_idx.append(i); seen.add(i)
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
        "method": "M7",
        "predicted_answer": ans,
        "reader_error": ans_err or bridge1_err or bridge2_err,
        "bridge_query": bridge1.strip(),
        "bridge_query_2": bridge2.strip(),
        "retrieved_chunk_idx": merged_idx,
        "retrieved_chunk_idx_pass1": idx1,
        "retrieved_chunk_idx_pass2": idx2,
        "retrieved_chunk_idx_pass3": idx3,
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
            if r.get("method") == "M7":
                out.add(r["qa_id"])
        except Exception:
            continue
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--key_filter", default="K7")
    args = ap.parse_args()

    configure()
    contexts = [json.loads(l) for l in CONTEXTS_FILE.read_text().splitlines() if l.strip()]
    if args.key_filter:
        keys = set(args.key_filter.split(","))
        contexts = [c for c in contexts if c["key_type"] in keys]

    done = load_existing()
    todo = [c for c in contexts if c["qa_id"] not in done]
    print(f"Total K7 contexts: {len(contexts)} | done: {len(done)} | todo: {len(todo)}", flush=True)
    if not todo:
        return

    enc = tiktoken.get_encoding("cl100k_base")
    cache = RetrievalCache(enc)

    print(f"Warming retrieval cache for {len(todo)} contexts...", flush=True)
    t_warm = time.time()
    for i, c in enumerate(todo):
        cache.get(c["qa_id"], c["context_text"])
        cache.get_dense_emb(c["qa_id"])
        if (i + 1) % 10 == 0:
            print(f"  warmed {i+1}/{len(todo)} elapsed={time.time()-t_warm:.0f}s", flush=True)
    print(f"  warm-up complete ({time.time()-t_warm:.0f}s)", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    fout = OUTPUT.open("a")
    async with make_client() as client:
        tasks = [run_M7(sem, client, c, cache) for c in todo]
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
                print(f"  [M7] {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)
        print(f"  [M7] DONE {n_ok+n_err}/{len(tasks)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    asyncio.run(main())
