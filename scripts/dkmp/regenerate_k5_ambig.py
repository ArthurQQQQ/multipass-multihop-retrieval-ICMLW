#!/usr/bin/env python3
"""regenerate_k5_ambig.py — Regenerate the 9 ambiguous K5 needles with patched prompt + entailment validator.

Outputs: data/dkmp/needles_v0_k5fix.jsonl (only the 9 fixed needles, full set merged later)
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dkmp"))
from _glm import call_glm_async, configure, make_client  # noqa: E402

NEEDLES_OLD = REPO / "data/dkmp/needles_v0.jsonl"
STORIES = REPO / "data/dkmp/stories_v0.json"
STORY_DIR = REPO / "data/narrativeqa/full_text"
OUT = REPO / "data/dkmp/needles_v0_k5fix.jsonl"

# 9 ambiguous K5 story IDs (from K5 audit)
AMBIG_SIDS = [
    "0e9c46e2aaab8f794dd9e636ee6e88f301a2ef78",
    "2584ef223f762658333799d37f593530f02ada28",
    "3add9dffaf9e59148643f2e6e2a3032ef36aad29",
    "3bbc2ac0b3ad1f68b4d7cf3d27e26a06268a80d1",
    "72ddeff1fe6bb9841d1c1da4b3031001bf6b1f58",
    "9562ea781e95c048df96f528a2a8272721cde3a7",
    "c9fab7d896ddf088d12ac7aaa448960789c73850",
    "e3f3a84e789ac0600c47997785bdc710cc3a0445",
    "fcf28eb74e5d50c402f34d32ce2370ce1bd12506",
]


PROMPT_K5_FIX = """You will read a fictional story and create a controlled causal-direction probe.

TASK: Invent TWO new fictional events E1 and E2 about brand-new entities, where E1 CAUSES E2.

CONSTRAINTS (strict):
- E1 < 22 words. E2 < 22 words.
- E2 MUST EXPLICITLY ATTRIBUTE its cause to E1, using one of:
  "caused by", "due to", "as a result of", "triggered by", "because of", "resulting from", "brought about by"
- Example E1: "The asteroid struck the orbital station Krellis-7 with violent force."
- Example E2: "The station's main reactor exploded due to the asteroid impact."
- BAD E2 (forbidden — implicit only): "The station's main reactor exploded."
  (No causal link mentioned in E2 itself — REJECT this format.)
- Question must be: "Did <E1 brief> cause <E2 brief>, or did <E2 brief> cause <E1 brief>?"
- Gold answer: "<E1 brief> caused <E2 brief>" — short, exactly using entities/events from E1 and E2.
- Use brand-new entity names not from the story.

Self-check before returning: Does E2 contain a causal verb that NAMES E1? If no, regenerate.

Return STRICT JSON only:
{"needle_e1": "...", "needle_e2": "...", "question": "...", "gold_answer": "...", "synthetic_entities": ["...", "..."]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

ENTAILMENT_PROMPT = """Two sentences are below. Determine whether they UNAMBIGUOUSLY support the gold claim.

Sentence 1: {e1}
Sentence 2: {e2}
Gold claim: {gold}

Strict criteria:
- Sentence 2 must contain a causal verb (caused, due to, as a result of, triggered by, because of, resulting from) explicitly naming Sentence 1's content.
- The causal direction in Sentence 2 must match the gold claim's direction.
- If Sentence 2 just describes a separate event without explicit causal attribution, the answer is NO.

Reply EXACTLY one word: YES or NO."""


def parse_json_loose(text: str):
    if not text: return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try: return json.loads(t)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m: return None
        try: return json.loads(m.group(0))
        except Exception: return None


def story_head(body: str, n_chars: int = 8000) -> str:
    return body[:n_chars]


async def gen_one_with_validate(client, story, max_retries=3):
    """Generate K5 needle, validate entailment, retry up to max_retries."""
    head = story_head(story["body"])
    prompt = PROMPT_K5_FIX.replace("{STORY_HEAD}", head)
    for attempt in range(max_retries):
        out, err = await call_glm_async(client, prompt, max_tokens=400)
        if err: continue
        d = parse_json_loose(out)
        if not d: continue
        e1 = d.get("needle_e1", "").strip()
        e2 = d.get("needle_e2", "").strip()
        q = d.get("question", "").strip()
        g = d.get("gold_answer", "").strip()
        if not (e1 and e2 and q and g): continue
        # Validate entailment
        v_prompt = ENTAILMENT_PROMPT.replace("{e1}", e1).replace("{e2}", e2).replace("{gold}", g)
        v_out, _ = await call_glm_async(client, v_prompt, max_tokens=8)
        v_yes = v_out.strip().upper().startswith("YES")
        result = {
            "story_id": story["story_id"],
            "key_type": "K5",
            "raw": out,
            "needle_sentences": [e1, e2],
            "question": q,
            "gold_answer": g,
            "entity": d.get("synthetic_entities", [""])[0] if d.get("synthetic_entities") else "",
            "lexical_overlap": None,
            "validated": True,
            "k5_fix_attempt": attempt,
            "entailment_check": v_out,
            "entailment_yes": v_yes,
        }
        if v_yes:
            return result
        # else loop and retry
    # If all attempts failed, return last (will be flagged)
    result["validated"] = False
    return result


async def main():
    configure()
    # Load stories - need to read body from STORY_DIR
    with open(STORIES) as f:
        meta = json.load(f)
    meta_by_id = {s["story_id"]: s for s in meta}
    todo = []
    for sid in AMBIG_SIDS:
        if sid not in meta_by_id: continue
        body_path = STORY_DIR / f"{sid}.txt"
        if not body_path.exists():
            print(f"  missing body: {sid}")
            continue
        body = body_path.read_text(errors="ignore")
        todo.append({"story_id": sid, "tokens": meta_by_id[sid]["tokens"], "body": body})
    print(f"Regenerating {len(todo)} ambiguous K5 needles with patched prompt + entailment validator")

    sem = asyncio.Semaphore(8)
    results = []
    async with make_client() as client:
        async def run_one(s):
            async with sem:
                t0 = time.time()
                r = await gen_one_with_validate(client, s)
                print(f"  {s['story_id'][:8]}: validated={r['entailment_yes']} ({time.time()-t0:.1f}s)")
                return r
        results = await asyncio.gather(*[run_one(s) for s in todo])

    n_validated = sum(1 for r in results if r["entailment_yes"])
    print(f"\n{n_validated}/{len(results)} needles passed entailment after up to 3 retries")

    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
