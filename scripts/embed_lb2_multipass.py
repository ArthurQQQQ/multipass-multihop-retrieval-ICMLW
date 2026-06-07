#!/usr/bin/env python3
"""embed_lb2_multipass.py - Embed multipass nodes for LongBench v2 with BGE-M3.

Inputs: data/longbench_v2/nodes_multipass_full.jsonl
Outputs: data/longbench_v2/embeddings/nodes_multipass.npy + nodes_multipass_index.jsonl
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NODES_IN = REPO / "data" / "longbench_v2" / "nodes_multipass_full.jsonl"
OUT_DIR = REPO / "data" / "longbench_v2" / "embeddings"
OUT_NPY = OUT_DIR / "nodes_multipass.npy"
OUT_IDX = OUT_DIR / "nodes_multipass_index.jsonl"


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    nodes = [json.loads(l) for l in open(NODES_IN)]
    print(f"loaded {len(nodes)} nodes")
    texts = [n["text"] for n in nodes]

    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("Embedding ...")
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    print(f"emb shape: {emb.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, emb)

    with open(OUT_IDX, "w") as f:
        for i, n in enumerate(nodes):
            f.write(json.dumps({
                "idx": i,
                "node_id": n.get("node_id"),
                "story_id": n.get("story_id"),
                "chunk_id": n.get("chunk_id"),
                "qa_idx": n.get("qa_idx"),
                "chunk_idx": n.get("chunk_idx", 0),
                "pass": n.get("pass"),
                "text": n["text"],
                "subject": n.get("subject"),
                "predicate": n.get("predicate"),
                "object": n.get("object"),
            }) + "\n")
    print(f"-> {OUT_NPY}")
    print(f"-> {OUT_IDX}")


if __name__ == "__main__":
    main()
