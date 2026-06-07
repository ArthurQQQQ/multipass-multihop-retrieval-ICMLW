"""04_judge.py - GLM-4.7 judge: YES/NO whether predicted answer matches gold.

Reads data/dkmp/predicted_v0.jsonl + data/dkmp/contexts_v0.jsonl
Writes data/dkmp/scored_v0.jsonl with extra fields: judge_yes (bool), judge_raw, judge_error

Idempotent: skips qa_id+method combos already in output.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm import call_glm_async, configure, make_client  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PREDICTED = REPO / "data/dkmp/predicted_v0.jsonl"
CONTEXTS = REPO / "data/dkmp/contexts_v0.jsonl"
OUTPUT = REPO / "data/dkmp/scored_v0.jsonl"


JUDGE_PROMPT = """You are scoring a reading-comprehension answer.

Question: {question}
Reference (gold) answer: {gold}
Predicted answer: {pred}

Is the predicted answer essentially correct (matches the meaning of the gold answer)? Minor wording or paraphrase is OK. Predictions like "I don't know" or empty answers are incorrect. For directional/causal questions, the direction MUST match.

Reply with exactly one token: YES or NO.
"""


def load_existing() -> set[tuple[str, str]]:
    if not OUTPUT.exists():
        return set()
    out = set()
    for line in OUTPUT.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            out.add((r["qa_id"], r["method"]))
        except Exception:
            continue
    return out


async def judge_one(sem, client, pred_row, ctx_row):
    async with sem:
        prompt = JUDGE_PROMPT.format(
            question=ctx_row["question"],
            gold=ctx_row["gold_answer"],
            pred=pred_row["predicted_answer"] or "(empty)",
        )
        text, err = await call_glm_async(client, prompt, max_tokens=10)
    yes = text.strip().upper().startswith("YES")
    return {**pred_row,
            "key_type": ctx_row["key_type"],
            "target_length": ctx_row["target_length"],
            "story_id": ctx_row["story_id"],
            "gold_answer": ctx_row["gold_answer"],
            "question": ctx_row["question"],
            "judge_yes": yes,
            "judge_raw": text,
            "judge_error": err}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    configure()
    contexts = {json.loads(l)["qa_id"]: json.loads(l) for l in CONTEXTS.read_text().splitlines() if l.strip()}
    predicted = [json.loads(l) for l in PREDICTED.read_text().splitlines() if l.strip()]
    done = load_existing()
    todo = [p for p in predicted if (p["qa_id"], p["method"]) not in done]
    print(f"Predicted: {len(predicted)} | done: {len(done)} | todo: {len(todo)}", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fout = OUTPUT.open("a")
    async with make_client() as client:
        tasks = [judge_one(sem, client, p, contexts[p["qa_id"]]) for p in todo]
        n_done = 0
        for fut in asyncio.as_completed(tasks):
            row = await fut
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done}/{len(todo)}", flush=True)
    fout.close()
    print(f"DONE. Output: {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
