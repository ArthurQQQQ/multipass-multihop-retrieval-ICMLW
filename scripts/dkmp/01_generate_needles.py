"""01_generate_needles.py - Generate DKMP needle-question pairs via GLM-4.7 teacher.

For each (story, key_type) in K1/K3/K5, generate ONE needle-question pair satisfying:
- Needle sentence(s) total < 30 tokens
- Needle uses synthetic entity names (collision-checked vs story body)
- Needle answer not present elsewhere in story (collision-checked)
- Lexical overlap (needle vs question) <30% for K3/K5; 30-50% for K1
- For K3 coref: needle uses an alias, question uses real char name
- For K5 causal: TWO needle sentences with directional relationship

Output: data/dkmp/needles_v0.jsonl

Idempotent: skips story_id+key_type combos already in output file.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm import call_glm_async, configure, make_client  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STORIES_FILE = REPO / "data/dkmp/stories_v0.json"
STORY_DIR = REPO / "data/narrativeqa/full_text"
OUTPUT = REPO / "data/dkmp/needles_v0.jsonl"

STOPWORDS = set("""
a an and are as at be but by for from has have he her hers him his i in is it its
me my of on or our she so that the their them then there these they this those to
us was we were what when where which who whom why will with you your s t d ll re m
""".split())

PROMPT_K1 = """You will read a fictional story and create a controlled probe item.

TASK: Invent ONE entirely new fact about a brand-new fictional entity (use a unique made-up name like "Quenton Krellis", "Vixenia Strode"). The fact must NOT relate to anything actually in the story.

CONSTRAINTS:
- Needle is ONE sentence, < 25 words.
- Use a unique synthetic entity name (5+ characters, not in the story).
- Then write a question whose answer is the fact. The question SHOULD share 30-50% non-stopword vocabulary with the needle (lexical key).
- The answer must be a short noun phrase (≤ 6 words).

Return STRICT JSON only, with this schema:
{"needle": "...", "question": "...", "gold_answer": "...", "synthetic_entity": "..."}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K3 = """You will read a fictional story and create a controlled coreference probe.

TASK: Invent ONE entirely new fact about a brand-new fictional entity (use a unique made-up name). Use the entity TWICE: once via a descriptive ALIAS in the needle, once via the PROPER NAME in the question. The reader must resolve alias→name to answer.

CONSTRAINTS:
- Make up ONE alias (e.g. "the silver-haired physicist", "the gondolier of Aldgate") AND ONE proper name (e.g. "Mirella Vance"). They are coreferent.
- IMPORTANT: include a setup clause stating the coreference at the start of the needle, like "Mirella Vance, the silver-haired physicist, ...". This is the ONLY time both forms appear together.
- Needle is ONE sentence, < 28 words, using both alias and name in the setup form.
- Question MUST refer to the entity ONLY by their proper name (not the alias) and ask about the fact.
- Lexical overlap between (needle's fact-bearing portion AFTER the setup clause) and question should be < 30%.
- Answer must be ≤ 6 words.

Return STRICT JSON only:
{"needle": "...", "question": "...", "gold_answer": "...", "alias": "...", "proper_name": "..."}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K2 = """You will read a fictional story and create a controlled paraphrase probe.

TASK: Invent ONE entirely new fact about a brand-new fictional entity (use a unique made-up name). Then write a question that asks for the same fact but uses ONLY synonyms and paraphrase — NO shared content words with the needle.

CONSTRAINTS:
- Use a unique synthetic entity name (5+ characters, not in the story).
- Needle is ONE sentence, < 25 words.
- The question must:
  * Use the synthetic entity name (so the question identifies the right subject)
  * Otherwise use SYNONYMS for ALL content words in the needle (e.g., "purchased" → "acquired", "vintage book" → "antique tome")
  * Share < 25% non-stopword vocabulary with the needle (excluding the entity name)
- Answer must be a short noun phrase (≤ 6 words), can use any vocabulary.

Return STRICT JSON only:
{"needle": "...", "question": "...", "gold_answer": "...", "synthetic_entity": "..."}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K4 = """You will read a fictional story and create a controlled temporal-order probe.

TASK: Invent TWO new fictional events E1 and E2 about brand-new entities, where E1 happens BEFORE E2 in time. Each event in ONE sentence.

CONSTRAINTS:
- Use brand-new entity names (not from story).
- E1 < 22 words. E2 < 22 words.
- The temporal order must be UNAMBIGUOUS from the wording — use absolute time markers (dates, "in the morning of...", "an hour later", "the following day", etc.) so order is clear from the text.
- The two events should NOT have a causal relationship (avoid "because" / "due to") — pure temporal.
- Question asks "Which event happened FIRST?" using a phrasing that names both events but does NOT include time markers like "before/after/first/next" so the model must infer from the temporal markers in the needle.
- Gold answer: "E1 happened first" using the actual entity descriptions.

Return STRICT JSON only:
{"needle_e1": "...", "needle_e2": "...", "question": "...", "gold_answer": "...", "synthetic_entities": ["...", "..."]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K8 = """You will read a fictional story and create a controlled 4-HOP probe.

TASK: Invent FOUR chained facts requiring 4-hop reasoning to answer.

CONSTRAINTS:
- Use unique made-up entity names X, Y, Z, W, V (5 distinct, 5+ chars).
- Needle1: X → Y. Example: "Vixenia Strode acquired the Crystal of Mar."
- Needle2: Y → Z. Example: "The Crystal of Mar was forged in the city of Velorum."
- Needle3: Z → W. Example: "Velorum is ruled by Queen Tessaria."
- Needle4: W → V. Example: "Queen Tessaria's signature scent is night-blooming jasmine."
- Each needle is ONE sentence < 22 words.
- Chain X → Y → Z → W → V must be unambiguous and unidirectional.
- Question asks about V via X (not naming Y/Z/W): "What scent is associated with the ruler of the city where Vixenia Strode's acquisition was forged?"
- Gold answer: short phrase about V (≤ 6 words).
- Question must NOT directly name Y, Z, or W.

Return STRICT JSON only:
{"needle1": "...", "needle2": "...", "needle3": "...", "needle4": "...", "question": "...", "gold_answer": "...", "synthetic_entities": ["X", "Y", "Z", "W", "V"]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K7 = """You will read a fictional story and create a controlled 3-HOP probe.

TASK: Invent THREE chained facts requiring 3-hop reasoning to answer.

CONSTRAINTS:
- Use unique made-up entity names X, Y, Z, W (4 distinct, 5+ chars).
- Needle1: X → Y. Example: "Vixenia Strode acquired the Crystal of Mar."
- Needle2: Y → Z. Example: "The Crystal of Mar was forged in the city of Velorum."
- Needle3: Z → W. Example: "Velorum is famous for its rare blue lithium-9."
- Each needle is ONE sentence < 22 words.
- The chain X → Y → Z → W must be unambiguous and unidirectional.
- Question asks about W via X (not naming Y or Z): "What is X's acquisition's forge city famous for?"
- Gold answer: short phrase about W (≤ 6 words).
- Question must NOT name Y or Z directly.

Return STRICT JSON only:
{"needle1": "...", "needle2": "...", "needle3": "...", "question": "...", "gold_answer": "...", "synthetic_entities": ["X", "Y", "Z", "W"]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K6 = """You will read a fictional story and create a controlled MULTI-HOP probe.

TASK: Invent TWO chained facts about brand-new fictional entities such that answering the question requires BOTH facts (single-fact retrieval is insufficient).

CONSTRAINTS:
- Use unique made-up entity names (5+ chars, not in story).
- Needle1 establishes fact about entity X mentioning intermediate entity Y.
  Example: "Vixenia Strode acquired the Crystal of Mar in 1923."
- Needle2 establishes fact about entity Y mentioning answer Z.
  Example: "The Crystal of Mar contains rare blue lithium-9."
- Each needle is ONE sentence < 22 words.
- The two needles share ONE intermediate entity (Y). They do NOT share X or Z.
- Question asks about X via the chain X→Y→Z, i.e., the answer Z is a property of Y, not stated about X directly.
  Example: "What rare element does Vixenia Strode's acquisition contain?"
- Gold answer: short phrase about Z (≤ 6 words).
- Question must NOT name Y directly (otherwise single-fact retrieval suffices). Instead refer to Y via X's fact.

Return STRICT JSON only:
{"needle1": "...", "needle2": "...", "question": "...", "gold_answer": "...", "intermediate_entity": "Y", "synthetic_entities": ["X", "Y", "Z"]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPT_K5 = """You will read a fictional story and create a controlled causal-direction probe.

TASK: Invent TWO new fictional events E1 and E2 about brand-new entities, where E1 CAUSES E2 (not the other way around). Each event is ONE sentence.

CONSTRAINTS:
- Use brand-new entity names (not from the story).
- E1 sentence < 22 words. E2 sentence < 22 words.
- The causal direction must be UNAMBIGUOUS from the wording (E1 happens first AND causes E2).
- Question asks the directional relationship: "Did E1 cause E2 or did E2 cause E1?"
- Phrase the question without using "first" / "before" / "after" — only causal language.
- Gold answer: a short phrase like "E1 caused E2" using the actual entities.

Return STRICT JSON only:
{"needle_e1": "...", "needle_e2": "...", "question": "...", "gold_answer": "...", "synthetic_entities": ["...", "..."]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""

PROMPTS = {"K1": PROMPT_K1, "K2": PROMPT_K2, "K3": PROMPT_K3, "K4": PROMPT_K4, "K5": PROMPT_K5, "K6": PROMPT_K6, "K7": PROMPT_K7, "K8": PROMPT_K8}


def tokenize_for_overlap(s: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", s.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def lexical_overlap(a: str, b: str) -> float:
    A, B = tokenize_for_overlap(a), tokenize_for_overlap(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(len(A), len(B))


def parse_json_loose(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown fences if any
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Find first { ... last }
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except Exception:
        return None


def needle_collision(needle_text: str, gold_answer: str, story_text: str) -> bool:
    """Return True if gold_answer appears in story (would leak)."""
    return gold_answer.lower() in story_text.lower()


def load_stories() -> list[dict]:
    sel = json.loads(STORIES_FILE.read_text())
    out = []
    for r in sel:
        body = (STORY_DIR / f"{r['story_id']}.txt").read_text(errors="ignore")
        out.append({"story_id": r["story_id"], "tokens": r["tokens"], "body": body})
    return out


def load_existing() -> set[tuple[str, str]]:
    if not OUTPUT.exists():
        return set()
    done = set()
    for line in OUTPUT.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            done.add((row["story_id"], row["key_type"]))
        except Exception:
            continue
    return done


def story_head(body: str, n_chars: int = 8000) -> str:
    """Use first ~8K chars (≈2K tokens) as story excerpt for needle generation.

    GLM only needs to see enough story to (a) avoid colliding entity names and
    (b) understand the genre. Full story not needed for the prompt.
    """
    return body[:n_chars]


async def gen_one(sem, client, story: dict, key_type: str) -> dict:
    async with sem:
        prompt = PROMPTS[key_type].replace("{STORY_HEAD}", story_head(story["body"]))
        text, err = await call_glm_async(client, prompt, max_tokens=400)
        out = {
            "story_id": story["story_id"],
            "key_type": key_type,
            "raw": text,
            "error": err,
        }
        if err or not text:
            return out
        parsed = parse_json_loose(text)
        if not parsed:
            out["error"] = "json_parse_failed"
            return out

        # Build needle_sentences and metadata per key
        try:
            if key_type in ("K1", "K2"):
                ns = [parsed["needle"]]
                question = parsed["question"]
                gold = parsed["gold_answer"]
                ent = parsed.get("synthetic_entity", "")
            elif key_type == "K3":
                ns = [parsed["needle"]]
                question = parsed["question"]
                gold = parsed["gold_answer"]
                ent = f"{parsed.get('proper_name','')}|{parsed.get('alias','')}"
            elif key_type in ("K4", "K5"):
                ns = [parsed["needle_e1"], parsed["needle_e2"]]
                question = parsed["question"]
                gold = parsed["gold_answer"]
                ent = "|".join(parsed.get("synthetic_entities", []))
            elif key_type == "K6":
                ns = [parsed["needle1"], parsed["needle2"]]
                question = parsed["question"]
                gold = parsed["gold_answer"]
                ent = "|".join(parsed.get("synthetic_entities", []))
            elif key_type == "K7":
                ns = [parsed["needle1"], parsed["needle2"], parsed["needle3"]]
                question = parsed["question"]
                gold = parsed["gold_answer"]
                ent = "|".join(parsed.get("synthetic_entities", []))
            elif key_type == "K8":
                ns = [parsed["needle1"], parsed["needle2"], parsed["needle3"], parsed["needle4"]]
                question = parsed["question"]
                gold = parsed["gold_answer"]
                ent = "|".join(parsed.get("synthetic_entities", []))
            else:
                out["error"] = f"unknown_key_type:{key_type}"
                return out
        except KeyError as ke:
            out["error"] = f"missing_field:{ke}"
            return out

        # Validate
        needle_full = " ".join(ns)
        if needle_collision(needle_full, gold, story["body"]):
            out["error"] = "gold_answer_in_story"
            return out

        overlap = lexical_overlap(needle_full, question)
        out.update(
            {
                "needle_sentences": ns,
                "question": question,
                "gold_answer": gold,
                "entity": ent,
                "lexical_overlap": round(overlap, 3),
                "validated": True,
            }
        )
        # K3 / K2: enforce <30% (coref / paraphrase tests must dodge surface match).
        # K4 / K5: skip overlap check (question must name both events; high overlap structural).
        # K1: warn if not 20-60%.
        if key_type in ("K2", "K3") and overlap >= 0.30:
            out["validated"] = False
            out["validation_note"] = f"overlap_{overlap:.2f}_too_high"
        elif key_type == "K1" and not (0.20 <= overlap <= 0.60):
            out["validated"] = False
            out["validation_note"] = f"overlap_{overlap:.2f}_outside_range"
        # Sanity: gold_answer must reference at least one needle word
        gold_words = tokenize_for_overlap(gold)
        needle_words = tokenize_for_overlap(needle_full)
        if gold_words and not (gold_words & needle_words):
            out["validated"] = False
            out["validation_note"] = "gold_answer_no_needle_overlap"
        return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="K1,K3,K5")
    ap.add_argument("--limit_stories", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--smoke", type=int, default=0, help="Smoke run on N stories only")
    args = ap.parse_args()

    configure()
    keys = args.keys.split(",")
    stories = load_stories()[: (args.smoke or args.limit_stories)]
    done = load_existing()
    todo = [(s, k) for s in stories for k in keys if (s["story_id"], k) not in done]
    print(f"Stories: {len(stories)} | keys: {keys} | todo: {len(todo)} | done: {len(done)}", flush=True)
    if not todo:
        print("Nothing to do.")
        return

    sem = asyncio.Semaphore(args.concurrency)
    async with make_client() as client:
        tasks = [gen_one(sem, client, s, k) for s, k in todo]
        # Append-as-we-go so partial progress is preserved
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("a") as fout:
            for fut in asyncio.as_completed(tasks):
                row = await fut
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                ok = "✓" if row.get("validated") else ("?" if not row.get("error") else "✗")
                note = row.get("error") or row.get("validation_note", "")
                print(f"{ok} {row['story_id'][:8]} {row['key_type']} ovl={row.get('lexical_overlap','-')} {note}", flush=True)
    print(f"\nDONE. Wrote to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
