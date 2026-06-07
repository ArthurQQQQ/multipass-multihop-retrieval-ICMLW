#!/usr/bin/env python3
"""multipass_extract_lb2.py - Dense X style multi-pass extraction for LongBench v2.

Adapted from multipass_extract.py. Adds LongBench v2 support.

Usage:
  python multipass_extract_lb2.py --limit-chunks 200 --out-name multipass_pilot
  python multipass_extract_lb2.py --out-name multipass_full   # all 18,618 chunks
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

REPO = Path(__file__).resolve().parent.parent.parent
LB_DIR = REPO / "data" / "longbench_v2"
LB_EMB = LB_DIR / "embeddings"

API_KEY = os.environ.get("GLM_API_KEY", "").strip()
API_URL = os.environ.get("GLM_URL", "https://www.dmxapi.com/v1/chat/completions")
MODEL = "glm-4.7"
CONC = int(os.environ.get("LB2_CONC", "32"))

# Reuse prompts from multipass_extract.py
sys.path.insert(0, str(REPO / "scripts" / "TEIE"))
from multipass_extract import (
    PROMPT_A_EVENTS, PROMPT_B_ATTRS, PROMPT_C_RELATIONS, PROMPT_D_CAUSAL,
    parse_json_list,
)


async def call_glm(client, prompt, max_tokens=700):
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


async def extract_chunk(sem, client, chunk):
    text = chunk["text"]
    chunk_id = chunk["chunk_id"]
    source_span = [chunk.get("char_start", 0), chunk.get("char_end", len(text))]
    async with sem:
        coros = [
            call_glm(client, PROMPT_A_EVENTS.replace("{text}", text)),
            call_glm(client, PROMPT_B_ATTRS.replace("{text}", text)),
            call_glm(client, PROMPT_C_RELATIONS.replace("{text}", text)),
            call_glm(client, PROMPT_D_CAUSAL.replace("{text}", text)),
        ]
        a_raw, b_raw, c_raw, d_raw = await asyncio.gather(*coros)

        nodes = []
        for raw_pair, kind in [(a_raw, "event"), (b_raw, "attribute"),
                                (c_raw, "relation"), (d_raw, "causal")]:
            data = parse_json_list(raw_pair[0])
            for n in data:
                if not isinstance(n, dict) or "text" not in n: continue
                node = {
                    "text": str(n.get("text", ""))[:250],
                    "subject": str(n.get("subject", n.get("cause", ""))),
                    "predicate": str(n.get("predicate", "")),
                    "object": str(n.get("object", n.get("effect", ""))),
                    "types": [kind],
                    "chunk_id": chunk_id,
                    "qa_idx": chunk.get("qa_idx", ""),
                    "story_id": chunk.get("story_id", chunk_id),
                    "chunk_idx": chunk.get("chunk_index", 0),
                    "source_span": source_span,
                    "pass": kind,
                }
                if node["text"]: nodes.append(node)
        return {"chunk_id": chunk_id, "nodes": nodes}


async def run_extraction(chunks):
    sem = asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONC + 30)) as client:
        coros = [extract_chunk(sem, client, c) for c in chunks]
        return await asyncio.gather(*coros)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--out-name", default="multipass")
    ap.add_argument("--start", type=int, default=0, help="start chunk index")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR GLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    chunks = [json.loads(l) for l in open(LB_EMB / "chunks_index.jsonl")]
    print(f"Total LongBench v2 chunks available: {len(chunks)}")

    if args.start > 0:
        chunks = chunks[args.start:]
    if args.limit_chunks > 0:
        chunks = chunks[:args.limit_chunks]

    out_path = LB_DIR / f"nodes_{args.out_name}.jsonl"
    print(f"Multi-pass extraction: {len(chunks)} chunks (4 passes each = {len(chunks)*4} GLM calls)")
    print(f"  CONC = {CONC}")
    print(f"  API URL = {API_URL}")
    print(f"  Output: {out_path}")

    t0 = time.time()
    results = asyncio.run(run_extraction(chunks))
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s ({len(chunks)*4 / elapsed:.1f} calls/sec)")

    n_nodes = 0
    pass_counts = defaultdict(int)
    chunks_with_zero_nodes = 0
    with open(out_path, "w") as f:
        for ck, r in zip(chunks, results):
            if not r["nodes"]:
                chunks_with_zero_nodes += 1
            for ni, n in enumerate(r["nodes"]):
                n["node_id"] = f"{ck['chunk_id']}_mp{ni:04d}"
                f.write(json.dumps(n) + "\n")
                n_nodes += 1
                pass_counts[n["pass"]] += 1
    print(f"  -> {out_path}  ({n_nodes} nodes)")
    print(f"  per pass: {dict(pass_counts)}")
    print(f"  avg nodes/chunk: {n_nodes/len(chunks):.1f}")
    print(f"  zero-node chunks: {chunks_with_zero_nodes}/{len(chunks)}")
    avg_chars = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"  avg chunk chars: {avg_chars:.0f}")
    print(f"  density: 1 node per {(avg_chars * len(chunks)) / max(1, n_nodes):.0f} chars")


if __name__ == "__main__":
    main()
