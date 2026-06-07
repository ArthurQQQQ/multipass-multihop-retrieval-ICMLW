"""02_build_contexts.py - Build DKMP contexts (distractor + needle insertion).

For each (needle, length N), produce one context_text of ~N tokens with the
needle inserted at a random non-edge position.

Distractor source: pool the OTHER stories' bodies (excluding the needle's
own story to avoid leaking story-specific context).

Output: data/dkmp/contexts_v0.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import tiktoken

REPO = Path(__file__).resolve().parents[2]
STORIES_FILE = REPO / "data/dkmp/stories_v0.json"
STORY_DIR = REPO / "data/narrativeqa/full_text"
NEEDLES_FILE = REPO / "data/dkmp/needles_v0.jsonl"
OUTPUT = REPO / "data/dkmp/contexts_v0.jsonl"


def encode(enc, text: str) -> list[int]:
    return enc.encode(text)


def decode(enc, toks: list[int]) -> str:
    return enc.decode(toks)


def load_story_tokens(enc) -> dict[str, list[int]]:
    sel = json.loads(STORIES_FILE.read_text())
    out = {}
    for r in sel:
        body = (STORY_DIR / f"{r['story_id']}.txt").read_text(errors="ignore")
        out[r["story_id"]] = encode(enc, body)
    return out


def load_needles() -> list[dict]:
    rows = []
    for line in NEEDLES_FILE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("validated") and "needle_sentences" in r:
            rows.append(r)
    return rows


def make_context(
    enc,
    needle_sentences: list[str],
    target_n: int,
    excluded_story: str,
    story_tokens: dict[str, list[int]],
    rng: random.Random,
    separate_inserts: bool = False,
) -> tuple[str, int, list[int]]:
    """
    Build a context of ~target_n tokens by sampling from non-excluded stories,
    then inserting the needle(s) at random non-edge position(s).

    If separate_inserts=True and multiple needle_sentences, each sentence is
    inserted at an independent random position (non-overlapping).

    Returns (context_text, actual_total_tokens, list_of_needle_positions_chars)
    """
    needles_toks = [encode(enc, s) for s in needle_sentences]
    total_needle_len = sum(len(t) for t in needles_toks)
    pool_keys = [k for k in story_tokens if k != excluded_story]
    rng.shuffle(pool_keys)

    budget = target_n - total_needle_len
    if budget < 0:
        joined = "\n\n".join(needle_sentences)
        return joined, total_needle_len, [0] * len(needle_sentences)

    distractor_toks: list[int] = []
    for k in pool_keys:
        if len(distractor_toks) >= budget:
            break
        st = story_tokens[k]
        max_start = max(0, len(st) - 2000)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        take = min(len(st) - start, budget - len(distractor_toks), 8000)
        distractor_toks.extend(st[start : start + take])
        if len(distractor_toks) < budget:
            distractor_toks.extend(encode(enc, "\n\n"))

    while len(distractor_toks) < budget:
        for k in pool_keys:
            if len(distractor_toks) >= budget:
                break
            st = story_tokens[k]
            max_start = max(0, len(st) - 1000)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            take = min(len(st) - start, budget - len(distractor_toks), 4000)
            distractor_toks.extend(st[start : start + take])

    distractor_toks = distractor_toks[:budget]
    sep_toks = encode(enc, "\n\n")

    if not separate_inserts or len(needle_sentences) == 1:
        # Single insert (joined) — original behavior
        joined_needle_toks = []
        for i, t in enumerate(needles_toks):
            if i > 0:
                joined_needle_toks.extend(encode(enc, " "))
            joined_needle_toks.extend(t)
        pos_min = max(1, int(0.05 * len(distractor_toks)))
        pos_max = int(0.95 * len(distractor_toks))
        pos = rng.randint(pos_min, pos_max) if pos_max > pos_min else len(distractor_toks) // 2
        full_toks = distractor_toks[:pos] + sep_toks + joined_needle_toks + sep_toks + distractor_toks[pos:]
        needle_pos_char = len(decode(enc, distractor_toks[:pos] + sep_toks))
        return decode(enc, full_toks), len(full_toks), [needle_pos_char]

    # Separate insertion at K random distinct positions
    K = len(needle_sentences)
    pos_min = max(1, int(0.05 * len(distractor_toks)))
    pos_max = int(0.95 * len(distractor_toks))
    if pos_max - pos_min < K * 200:
        # Distractor too small — fall back to single insert
        return make_context(enc, needle_sentences, target_n, excluded_story, story_tokens, rng, False)

    # Sample K positions, sort, then ensure min-spacing
    while True:
        positions = sorted(rng.sample(range(pos_min, pos_max), K))
        if all(positions[i+1] - positions[i] >= 200 for i in range(K-1)):
            break

    # Build context by interleaving needles at the sorted positions
    # Walk distractor_toks, inserting needles at appropriate offsets
    out_toks: list[int] = []
    needle_pos_chars: list[int] = []
    last = 0
    for i, p in enumerate(positions):
        out_toks.extend(distractor_toks[last:p])
        out_toks.extend(sep_toks)
        # Compute char position before adding needle
        needle_pos_chars.append(len(decode(enc, out_toks)))
        out_toks.extend(needles_toks[i])
        out_toks.extend(sep_toks)
        last = p
    out_toks.extend(distractor_toks[last:])
    return decode(enc, out_toks), len(out_toks), needle_pos_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="1000,8000,32000,128000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Build only 3 contexts")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing contexts file, skipping qa_ids already present")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    lengths = [int(x) for x in args.lengths.split(",")]
    needles = load_needles()
    if args.smoke:
        needles = needles[:3]

    existing_ids: set[str] = set()
    if args.append and OUTPUT.exists():
        for line in OUTPUT.read_text().splitlines():
            if not line.strip():
                continue
            try:
                existing_ids.add(json.loads(line)["qa_id"])
            except Exception:
                continue
        print(f"Existing contexts: {len(existing_ids)} (will skip)", flush=True)

    print(f"Needles: {len(needles)} | lengths: {lengths} | total contexts: {len(needles) * len(lengths)}", flush=True)

    story_tokens = load_story_tokens(enc)
    rng = random.Random(args.seed)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    n_written = 0
    with OUTPUT.open(mode) as f:
        for n in needles:
            needle_full = " ".join(n["needle_sentences"])
            # K6/K7/K8 = multi-hop: insert each needle sentence at a separate position
            separate = (n["key_type"] in ("K6", "K7", "K8")) and (len(n["needle_sentences"]) > 1)
            for N in lengths:
                qa_id_pre = f"dkmp_{n['story_id'][:8]}_{n['key_type']}_N{N}"
                if args.append and qa_id_pre in existing_ids:
                    continue
                ctx, actual_n, needle_pos = make_context(
                    enc=enc,
                    needle_sentences=n["needle_sentences"],
                    target_n=N,
                    excluded_story=n["story_id"],
                    story_tokens=story_tokens,
                    rng=rng,
                    separate_inserts=separate,
                )
                qa_id = f"dkmp_{n['story_id'][:8]}_{n['key_type']}_N{N}"
                row = {
                    "qa_id": qa_id,
                    "story_id": n["story_id"],
                    "key_type": n["key_type"],
                    "target_length": N,
                    "actual_length": actual_n,
                    "needle_position_chars": needle_pos[0] if len(needle_pos) == 1 else needle_pos,
                    "needle_separate_inserts": separate,
                    "needle_sentences": n["needle_sentences"],
                    "needle_text": needle_full,
                    "question": n["question"],
                    "gold_answer": n["gold_answer"],
                    "lexical_overlap": n.get("lexical_overlap"),
                    "context_text": ctx,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
        print(f"\nWrote {n_written} contexts to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
