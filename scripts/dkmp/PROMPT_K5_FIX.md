# PROMPT_K5 Fix — Closing the 70% Ceiling

**Discovered**: 2026-05-08, K5 ambiguity audit on v0_origprompt scored data.

## Root cause

Original PROMPT_K5 says "causal direction must be UNAMBIGUOUS" but GLM produces TWO INDEPENDENT EVENT SENTENCES with implied co-occurrence. 9/30 K5 needles (30%) are causally ambiguous:

```
E1: "The heavy storm flooded the basement."
E2: "The electrical short ignited a fire."
Q:  "Did the heavy storm cause the electrical short or did the electrical short cause the heavy storm?"
GOLD: "The heavy storm caused the electrical short"
```

The needle does NOT contain the causation being asked. GLM correctly answers "I don't know" — judge marks this NO. **70% K5 ceiling is measurement bias, not model failure.**

## Patched PROMPT_K5

Replace the K5 prompt with this version that **forces explicit causal verbs in E2**:

```python
PROMPT_K5_FIX = """You will read a fictional story and create a controlled causal-direction probe.

TASK: Invent TWO new fictional events E1 and E2 about brand-new entities, where E1 CAUSES E2.

CONSTRAINTS (strict):
- E1 < 22 words. E2 < 22 words.
- E2 MUST explicitly attribute its cause to E1, using one of: "caused by", "due to", "as a result of", "triggered by", "because of".
- Example E1: "The asteroid struck the orbital station Krellis-7."
- Example E2: "The station's reactor exploded due to the asteroid impact."
- Bad E2 (forbidden): "The station's reactor exploded." [no causal link to E1]
- Question must be: "Did <E1 brief> cause <E2 brief>, or did <E2 brief> cause <E1 brief>?"
- Gold answer: "<E1 brief> caused <E2 brief>" — short, exactly using entities from E1 and E2.
- Use brand-new entity names not from the story.

Self-check before returning: Does E2 contain a causal verb naming E1? If no, regenerate.

Return STRICT JSON only:
{"needle_e1": "...", "needle_e2": "...", "question": "...", "gold_answer": "...", "synthetic_entities": ["...", "..."]}

STORY EXCERPT:
<<<
{STORY_HEAD}
>>>
"""
```

## Post-generation validator

After generation, run an entailment check:

```python
async def validate_k5(needle_e1, needle_e2, gold_answer, client):
    """Use GLM-4.7 to verify needle ⊨ gold_answer (entailment)."""
    prompt = f"""Given these two sentences:
1. {needle_e1}
2. {needle_e2}

Does this claim follow as an unambiguous fact: "{gold_answer}"?

Reply EXACTLY one word: YES or NO."""
    out, _ = await call_glm(client, prompt, max_tokens=8)
    return out.strip().upper().startswith("YES")
```

Reject any needle that fails entailment.

## Expected impact

- 30 K5 needles × 4 lengths × 5 methods = 600 K5 datapoints
- Currently 70% accuracy
- After fix: 21 needles already correct (estimated 90%+) → expected M5 K5 ceiling ≥ 90%
- M2 K5 should also rise from 0.80 → ≥0.90

## DKMP v1 plan delta

Add to `01_generate_needles.py`:
1. Replace PROMPT_K5 with PROMPT_K5_FIX
2. Add `validate_k5()` after generation
3. Reject + regenerate if validation fails (max 3 retries)
4. Log validation failure rate to detect prompt drift

Cost impact: 1 extra GLM call per K5 needle = 30 extra calls for v0 size, 100+ for v1.
