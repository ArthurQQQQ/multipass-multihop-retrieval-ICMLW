"""00_cache_embeddings.py - Pre-compute BGE-M3 embeddings for all DKMP context chunks.

Flat batch encoding: chunks all 720 contexts, encodes ALL chunks in one big
sequence of large batches, saves to disk. Avoids per-context MPS overhead +
sporadic stalls observed during 03b warm-up.

Output: data/dkmp/embeddings_cache_v0.npz with keys:
  - emb: float32 (N_total_chunks, 1024)
  - chunk_offsets: int64 (N_contexts+1,) — chunks for ctx i are emb[chunk_offsets[i]:chunk_offsets[i+1]]
  - qa_ids: list[str] — qa_id per context, in order matching offsets

Plus: data/dkmp/chunks_cache_v0.jsonl — one row per context with chunked text +
BM25-tokenized chunks (avoids re-chunking later).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
CONTEXTS_FILE = REPO / "data/dkmp/contexts_v0.jsonl"
EMB_OUT = REPO / "data/dkmp/embeddings_cache_v0.npz"
CHUNKS_OUT = REPO / "data/dkmp/chunks_cache_v0.jsonl"

CHUNK_TOKENS = 200
CHUNK_STRIDE = 150
DENSE_DIM = 1024  # BGE-M3 output dim


def chunk_text(enc, text: str, chunk_size: int = CHUNK_TOKENS, stride: int = CHUNK_STRIDE):
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


def simple_tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--key_filter", default="", help="Only cache for these keys")
    ap.add_argument("--max_length", type=int, default=0)
    args = ap.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    contexts = [json.loads(l) for l in CONTEXTS_FILE.read_text().splitlines() if l.strip()]
    if args.key_filter:
        keys = set(args.key_filter.split(","))
        contexts = [c for c in contexts if c["key_type"] in keys]
    if args.max_length:
        contexts = [c for c in contexts if c["target_length"] <= args.max_length]
    print(f"Caching for {len(contexts)} contexts", flush=True)

    # Step 1: chunk all contexts and persist
    print("Chunking...", flush=True)
    t0 = time.time()
    chunks_per_ctx: list[list[str]] = []
    chunks_meta: list[dict] = []
    for c in contexts:
        chunks = chunk_text(enc, c["context_text"])
        chunk_texts = [t for _, _, t in chunks]
        chunks_per_ctx.append(chunk_texts)
        chunks_meta.append(
            {
                "qa_id": c["qa_id"],
                "key_type": c["key_type"],
                "target_length": c["target_length"],
                "chunks": chunk_texts,
                "bm25_tokens": [simple_tokenize(t) for t in chunk_texts],
            }
        )
    print(f"  chunked in {time.time()-t0:.0f}s; total chunks: {sum(len(c) for c in chunks_per_ctx):,}", flush=True)

    # Step 2: flat-batch encode all chunks
    print("Loading BGE-M3...", flush=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("Loaded.", flush=True)

    flat_chunks = [t for chunks in chunks_per_ctx for t in chunks]
    n_total = len(flat_chunks)
    chunk_offsets = np.zeros(len(chunks_per_ctx) + 1, dtype=np.int64)
    for i, chunks in enumerate(chunks_per_ctx):
        chunk_offsets[i + 1] = chunk_offsets[i] + len(chunks)

    print(f"Encoding {n_total:,} chunks at batch_size={args.batch_size}...", flush=True)
    t0 = time.time()
    all_emb = np.zeros((n_total, DENSE_DIM), dtype=np.float32)
    bs = args.batch_size
    for start in range(0, n_total, bs):
        end = min(start + bs, n_total)
        batch_emb = model.encode(
            flat_chunks[start:end],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=bs,
        )
        all_emb[start:end] = np.asarray(batch_emb, dtype=np.float32)
        if start % (bs * 50) == 0 and start > 0:
            elapsed = time.time() - t0
            rate = start / elapsed
            eta = (n_total - start) / rate
            print(f"  {start:>6,}/{n_total:,} elapsed={elapsed:.0f}s rate={rate:.0f}/s eta={eta:.0f}s", flush=True)
    print(f"  encoded in {time.time()-t0:.0f}s", flush=True)

    # Step 3: save
    qa_ids = [c["qa_id"] for c in contexts]
    np.savez_compressed(
        EMB_OUT,
        emb=all_emb,
        chunk_offsets=chunk_offsets,
        qa_ids=np.asarray(qa_ids, dtype=object),
    )
    print(f"Wrote {EMB_OUT} ({EMB_OUT.stat().st_size/1e6:.1f} MB)", flush=True)

    with CHUNKS_OUT.open("w") as f:
        for m in chunks_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Wrote {CHUNKS_OUT} ({CHUNKS_OUT.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
