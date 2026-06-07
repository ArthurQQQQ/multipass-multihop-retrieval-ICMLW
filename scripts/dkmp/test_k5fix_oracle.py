#!/usr/bin/env python3
"""test_k5fix_oracle.py - Test M5 oracle on regenerated K5 needles to confirm ceiling rises."""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dkmp"))
from _glm import call_glm_async, configure, make_client

NEW_NEEDLES = REPO / "data/dkmp/needles_v0_k5fix.jsonl"

# Same reader prompt as v0p1 (no IDK fallback)
READER_PROMPT_V0P1 = """You are answering a reading-comprehension question.

Context: {context}

Question: {question}

Provide a direct, specific answer to the question."""

# Old reader prompt (for comparison)
READER_PROMPT_V0 = """You are answering a reading-comprehension question.

Context: {context}

Question: {question}

If the text does not contain the answer, say exactly: I don't know.
Otherwise, provide a direct, specific answer to the question."""


JUDGE_PROMPT = """You are scoring a reading-comprehension answer.

Question: {question}
Reference (gold) answer: {gold}
Predicted answer: {pred}

Is the predicted answer essentially correct (matches the meaning of the gold answer)? Minor wording or paraphrase is OK. Predictions like "I don't know" or empty answers are incorrect. For directional/causal questions, the direction MUST match.

Reply with exactly one token: YES or NO."""


async def predict(client, needle, prompt_template):
    context = " ".join(needle["needle_sentences"])
    prompt = prompt_template.replace("{context}", context).replace("{question}", needle["question"])
    out, err = await call_glm_async(client, prompt, max_tokens=200)
    return out, err


async def judge(client, question, gold, pred):
    prompt = JUDGE_PROMPT.replace("{question}", question).replace("{gold}", gold).replace("{pred}", pred)
    out, err = await call_glm_async(client, prompt, max_tokens=8)
    return out.strip().upper().startswith("YES"), out


async def main():
    configure()
    needles = [json.loads(l) for l in open(NEW_NEEDLES)]
    print(f"Loaded {len(needles)} regenerated K5 needles")
    validated = [n for n in needles if n["entailment_yes"]]
    print(f"  {len(validated)} validated by entailment")
    print()

    async with make_client() as client:
        # Test BOTH prompts (v0 with IDK, v0p1 without IDK)
        for label, template in [("v0_prompt", READER_PROMPT_V0), ("v0p1_prompt", READER_PROMPT_V0P1)]:
            print(f"=== Reader: {label} ===")
            tasks = [predict(client, n, template) for n in needles]
            preds = await asyncio.gather(*tasks)
            judge_tasks = [judge(client, n["question"], n["gold_answer"], p[0]) for n, p in zip(needles, preds)]
            judge_results = await asyncio.gather(*judge_tasks)

            n_correct = sum(1 for j, _ in judge_results if j)
            n_correct_validated = sum(1 for j, n in zip([j for j, _ in judge_results], needles)
                                      if j and n["entailment_yes"])
            n_total = len(needles)
            n_validated = sum(1 for n in needles if n["entailment_yes"])
            print(f"  All ({n_total}): {n_correct}/{n_total} = {n_correct/n_total:.0%}")
            print(f"  Validated only ({n_validated}): {n_correct_validated}/{n_validated} = {n_correct_validated/n_validated:.0%}")

            # Per-needle output
            for n, (pred, _), (jy, jraw) in zip(needles, preds, judge_results):
                icon = "✓" if jy else "✗"
                vmark = "[V]" if n["entailment_yes"] else "[ ]"
                print(f"    {icon} {vmark} {n['story_id'][:8]}: pred='{pred[:80]}'")
            print()


if __name__ == "__main__":
    asyncio.run(main())
