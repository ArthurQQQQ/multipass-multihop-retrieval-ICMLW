# IRCoT Faithfulness Audit — M8 vs Trivedi et al. (ACL 2023)

**Date**: 2026-05-09
**Purpose**: Determine whether our `M8` baseline is a faithful re-implementation of IRCoT or a strawman simplified variant. Reviewer attack surface depends entirely on this.
**Verdict (TL;DR)**: **M8 is structurally faithful to IRCoT's loop, but is 0-shot where the original is 8-shot. This single gap is the most likely point of attack and may underestimate IRCoT by 5–15pp on K6/K7/K8.**

---

## 1. Reference implementation (Trivedi et al. 2023)

Sources audited:
- Repo: `StonyBrookNLP/ircot` (commit `main`, fetched 2026-05-09)
- Inference loop: `commaqa/inference/ircot.py`
- Prompts: `prompts/{hotpotqa,2wikimultihopqa,musique,iirc}/*_cot_qa_*.txt`
- Paper: `arXiv:2212.10509`

### 1.1 Reference loop (canonical IRCoT)

```
state.titles, state.paras, state.cot ← []
input_query ← question                    # bootstrap
for step in range(max_iterations=8):
    # — RETRIEVE step —
    query  ← remove_wh_words(input_query)            # strip who/what/when/etc.
    paras  ← BM25(query, corpus, k=retrieval_count)  # cumulate into state
    state.paras.extend(new_paras)

    # — REASON step (one sentence of CoT) —
    prompt = (8-shot demos)
           + "\n\n\n"
           + para_block(state.paras, max_words=350)         # show_so_far_paras=True
           + "\nSo far collected Wikipedia page titles: ..." # show_so_far_titles=True
           + "\nSo far collected evidence: " + " ".join(state.cot)  # show_so_far_cot=True
           + "\nQ: " + question
           + "\nA: "
    new_sentence ← LLM(prompt)                       # exactly ONE CoT sentence
    state.cot.append(new_sentence)

    # — STOPPING —
    if "the answer is" in new_sentence.lower(): break

    # — NEXT QUERY: question_or_last_generated_sentence —
    if is_reasoning_sentence(new_sentence):  # startswith thus/so/therefore/hence
        # skip — keep previous input_query
        continue
    input_query ← new_sentence

answer ← regex_extract(r"the answer is (.+?)\.?$", state.cot[-1])
```

### 1.2 Critical reference-implementation details

| Detail | Reference value | Why it matters |
|---|---|---|
| **Few-shot demos** | **8 in-context examples** with full `Q → multi-sentence CoT → "So the answer is X"` traces | Without demos, GLM/GPT often fail to produce one sentence per turn, frequently emit the answer too early, or stall in meta-reasoning |
| **Demo source** | Hand-crafted from HotpotQA train set, includes hard negatives (3 distractor paragraphs per demo) | Teaches the model both "use the evidence" *and* "ignore distractors" |
| **CoT format** | Multi-sentence reasoning, each with one factual claim, terminating in `So the answer is X.` | Standardized stop condition + terminal-state regex extraction |
| **Query reformulation** | `remove_wh_words` on the BM25 query | Avoids "who/what/when" dominating BM25 IDF and burning recall |
| **Reasoning-sentence skipping** | Sentences starting with thus/so/therefore/hence skipped as next query | Reasoning sentences contain no new entities to retrieve on |
| **Cumulation** | `cumulate_titles=True` — paragraphs accumulate across steps | The reader sees a growing context, never resets |
| **Max iterations** | 8 | Allows up to 7 CoT sentences before forced stop |
| **Final extraction** | Regex from the terminal "So the answer is X" sentence | Deterministic, no separate reader call |

---

## 2. Our M8 implementation

Source: `/Users/arthurqiu/MemoryNet/scripts/dkmp/03g_run_ircot.py`

### 2.1 Our prompt (verbatim)

```
You answer multi-hop questions step by step using retrieved text.

At each step, write ONE short reasoning sentence. The sentence should either:
  (a) state an intermediate fact you derived from the retrieved text, OR
  (b) state what you still need to find next.

After enough steps, write a final sentence beginning with "So the answer is " followed by your answer.

Rules:
- One sentence per turn. Maximum 25 words.
- Ground every claim in the retrieved text.
- Do not write multiple sentences. Do not number your sentences.

Retrieved text so far:
{T}

Question: {Q}

Reasoning so far:
{R}

Next sentence:
```

### 2.2 Our loop

```
seen_chunks ← ∅
reasoning   ← []
current_query ← question
for step in range(MAX_STEPS=6):
    q_emb ← BGE-M3.encode(current_query)
    idx   ← hybrid_topk(emb, bm25, q_emb, tokenize(current_query), k=10)  # RRF(dense, BM25)
    seen_chunks ← seen_chunks ∪ idx
    prompt ← IRCOT_PROMPT.format(T=all_chunks(seen_chunks), Q=question, R=reasoning)
    sent   ← LLM(prompt, max_tokens=80)
    reasoning.append(sent)
    if /so the answer is|the answer is|final answer/ in sent: break
    current_query ← sent
answer ← regex_extract(reasoning[-1])
```

---

## 3. Diff table — M8 vs reference IRCoT

| Aspect | Reference IRCoT | Our M8 | Severity |
|---|---|---|---|
| **Few-shot demos** | **8 demos** | **0 demos** | 🟥 **HIGH** — primary risk |
| Step-emits | One CoT sentence | One CoT sentence (≤25 words) | ✅ match |
| Stopping condition | "the answer is" in CoT | "(so the |the )answer is\|final answer" in CoT | ✅ match |
| Query reformulation | `remove_wh_words(sentence)` before BM25 | Raw sentence; tokenized to lowercase alphanumerics | 🟨 MEDIUM |
| Reasoning-sentence skip | Skip thus/so/therefore as query | No skip — entire sentence becomes next query | 🟨 MEDIUM |
| Retriever | BM25-only on Elasticsearch | BGE-M3 ⊕ BM25 fused via RRF (k_RRF=60), top-K=10 | 🟢 reasonable adaptation, defensible |
| Cumulation | `cumulate_titles=True` (paragraphs accumulate) | `seen_chunks` accumulates, monotone increasing | ✅ match |
| Max iterations | 8 | **6** | 🟨 LOW-MEDIUM (may bias D=5 against M8) |
| Final answer extraction | regex on terminal CoT sentence | regex on terminal CoT sentence | ✅ match |
| Reader sees CoT | Yes (CoT is the answer-generation stream itself) | Yes (CoT is in `reasoning`, regex extracts) | ✅ match |
| Prompt phrasing of "what to retrieve next" | Implicit, learned from 8-shot demos | Explicit instruction: "(b) state what you still need to find next" | 🟨 MEDIUM — our prompt is *more* explicit; could be a wash or slight advantage for M8 |

### 3.1 Severity legend
- 🟥 HIGH: reviewer-killer — alone sufficient to reject the M7 vs IRCoT comparison
- 🟨 MEDIUM: defensible but should be ablation-tested
- 🟢 LOW: cosmetic, easy to defend in the response

---

## 4. The 0-shot vs 8-shot gap — quantitative risk

The single biggest concern: real IRCoT was **designed around 8-shot demos**. Without them:

- **Demo function 1: format calibration.** Demos teach the model "one sentence per turn, end with 'So the answer is X.'" Our explicit Rules section attempts to substitute, but without shown examples GLM-4.7 occasionally:
  - emits multiple sentences in one turn (we slice by `\n` so only first is kept; may discard signal)
  - emits the answer prematurely (after 1-2 retrievals when 3+ are needed)
  - fails to emit "So the answer is" (regex falls back to last line, often a meta-statement)
- **Demo function 2: in-context learning of bridge formation.** Demos show the model: *intermediate fact → next query is built from the new entity in that fact*. Without demos, GLM may emit reasoning sentences that don't introduce new query terms, leading to redundant retrievals.

### 4.1 Estimated magnitude of underperformance

Based on the IRCoT paper Table 6 (HotpotQA, 2WikiMHQA, MuSiQue):
- Removing few-shot demos in IRCoT-family methods: **−5 to −15pp** on multi-hop accuracy
- More severe on harder hop counts (D≥3)

Applied to our DKMP numbers at N=128K:
| | Observed M8 | Plausible "real IRCoT" estimate |
|---|---|---|
| K6 (2-hop) | 0.87 | 0.87–0.92 (gap small at D=2) |
| K7 (3-hop) | 0.60 | **0.70–0.80** ⚠ |
| K8 (4-hop) | 0.77 | **0.85–0.92** ⚠ |
| K9 (5-hop) | 0.83 | 0.83–0.88 (saturating) |

Compared to M7:
- K6: M7 1.00 vs M8 0.87 → +13pp lead might shrink to +8–13pp (still wins)
- K7: M7 0.90 vs M8 0.60 → +30pp lead **might shrink to +10–20pp** (still wins, less dramatic)
- K8: M7 0.97 vs M8 0.77 → +20pp lead **might shrink to +5–12pp or flip negative** ⚠
- K9: M7 0.53 vs M8 0.83 → already losing by 30pp (saturation regime)

**Bottom line: M7's lead survives at D=2,3 but the K8 win is at risk and the K9 loss becomes more decisive.**

---

## 5. Defensive moves (in priority order)

### P0 — must do before submission
1. **Run M8a = M8 + 8-shot demos** on K6/K7/K8/K9 × 4 lengths.
   - Adapt 8 HotpotQA demos to DKMP synthetic-entity format (or write 8 fresh demos using held-out DKMP stories).
   - Re-run K6/K7/K8/K9 at N=128K × n=30. Cost ≈ 8×demo_tokens × 30 × 4 keys + bridge cost ≈ $20–30.
   - **If M7 still wins on D=2,3,4 against M8a, the headline holds.** If not, we re-frame as "M7 ≈ IRCoT-with-demos at lower per-step cost".

2. **Add `remove_wh_words` to M8 retrieval query.** Trivial code change, minutes of compute.

3. **Add reasoning-sentence skip rule** (`thus|so|therefore|hence`) for next-query selection.

4. **Bump MAX_STEPS to 8** to match reference.

### P1 — strongly recommended
5. **Run M8b = M8 with the IRCoT codebase's actual prompt** (HotpotQA gold_with_3_distractors_context_cot_qa_codex.txt, ported to GLM-4.7 via DSPy or direct).
   - This is the truly faithful version. If we can show M8a ≈ M8b, our "IRCoT-with-demos" is defensible.

### P2 — nice to have
6. Single-prompt ablation: M8 vs M8 with our explicit (a)/(b) prompt removed.
7. Step-count ablation: 4, 6, 8 max steps.

---

## 6. What to write in the paper

Replace the current sentence "we compare against IRCoT (Trivedi 2023)" with:

> *"We compare against a 0-shot adaptation of IRCoT (M8) and an 8-shot faithful re-implementation following the official prompts from `StonyBrookNLP/ircot` (M8a). Both share IRCoT's interleave-retrieve-and-reason loop, max_steps=8, and 'so the answer is' stopping criterion. M8 isolates the contribution of the loop architecture; M8a additionally captures the in-context-learning gain from few-shot demonstrations. Our M7 differs from both by replacing the CoT-sentence-as-query with a 6-word noun-phrase bridge extraction, which costs fewer per-step tokens and removes the meta-reasoning failure mode where the next query restates the question rather than naming an entity to look up."*

This framing is honest, cites the gap explicitly, and turns the comparison into a contribution rather than a weakness.

---

## 7. Recommended action right now

Run M8a (8-shot IRCoT) on K6/K7/K8/K9 × N=128K. Single overnight job, ~$30. If the other agent currently running experiments is free, **this is the highest-value next experiment**. The demo set can be 8 hand-crafted multi-hop QA traces using DKMP synthetic-entity format from training-time stories (held out from eval).

Until M8a numbers exist, **do not promise the +30pp M7 vs IRCoT headline in the abstract**. Use "M7 wins on D=2,3,4 with fewer per-step tokens" as the safe claim.
