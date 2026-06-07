# MULTIPASS — bounded multi-hop retrieval for long-context memory (ICMLW)

**What this code does.** It diagnoses and repairs a concrete failure of
single-pass retrieval over long-context / agentic memory, which we call
**bridge starvation**: a query names entity *X* but the answer needs a chain
*X → Y → Z → …*. The first retrieval pass finds the *X* fact, but the bridge
entities (*Y*, *Z*) never appear in the query, so the remaining links are never
retrieved and the reader composes the wrong answer.

This repo gives you two things:

- **DKMP** — a controlled trigger that *makes the failure happen on demand*. It
  embeds synthetic-entity needle chains (depth 2–5 hops) into real NarrativeQA
  distractor contexts at {1K, 8K, 32K, 128K} tokens, so you can measure exactly
  where single-pass retrieval breaks.
- **MULTIPASS** — a **bounded repair** that reuses your existing retriever. Between
  retrieval passes it asks the LLM to emit one short bridge entity (≤ 6 words),
  re-retrieves with it, and loops until the bridge call returns `NONE`. No new
  index, no graph, no retraining.

Plus trace diagnostics that separate *retrieval coverage* from *reader
composition*, so you can tell which stage is failing.

## When to use this

Use **DKMP** if you want a reproducible multi-hop stress test for any
long-context retriever. Use **MULTIPASS** if you have a single-pass retrieval
loop that fails on chained/multi-hop questions and you want a drop-in repair that
doesn't touch your index.

### Headline results (N = 128K, GLM-4.7 LLM-judged, n = 30/cell)

| Hop | Single-pass Hybrid-RRF | MULTIPASS | Oracle (gold needles) |
|----:|:----------------------:|:---------:|:---------------------:|
| 2   | 0.40 | **1.00** | 1.00 |
| 3   | 0.47 | **0.90** | 0.93 |
| 4   | 0.57 | **0.97** | 1.00 |
| 5   | 0.13 | 0.53 | 0.76 |

MULTIPASS hits oracle on 2-hop and is within 3pp on 3/4-hop, above IRCoT, ReAct,
Self-Ask, and a passage-graph PPR variant under the same reader and index. The
repair is intentionally **bounded**: it does not solve 5-hop chains, does not help
on short-context MuSiQue (the trigger doesn't fire), and does not target
LongMemEval's cross-session aggregation failure.

## Install

```bash
pip install -r requirements.txt
```

The reader and judge use GLM-4.7 over an HTTP API. Set credentials via env vars
or a local `.env` (git-ignored). Start from the template:

```bash
cp .env.example .env
```

## Run

```bash
# --- DKMP multi-hop trigger (after data/ is populated + embeddings cached) ---
python scripts/dkmp/03_run_baselines.py        # single-pass: FullCtx / BM25 / Hybrid-RRF / oracle
python scripts/dkmp/03c_run_multipass_cached.py # MULTIPASS, 1 bridge
python scripts/dkmp/03f_run_M7_cached.py        # MULTIPASS, D-pass
python scripts/dkmp/04_judge.py                 # fresh-instance GLM-4.7 YES/NO judge
python scripts/dkmp/05_compute_L90.py           # accuracy grid + bootstrap CIs

# --- LongMemEval N=500 same-reader fairness check ---
python scripts/longmemeval_eval_methods.py --method hybrid_rrf --char-budget 8000 --K 10 --out-name full_n500
python scripts/longmemeval_llm_judge.py --method hybrid_rrf --out-name full_n500
```

## What's in here

```
scripts/
  dkmp/                 DKMP trigger pipeline (00 cache → 05 grid) + MULTIPASS
  longmemeval_*.py      LongMemEval same-reader fairness check
  lb2_* / *_lb2.py      LongBench v2 multipass extraction & eval
results/                released predictions, scores, per-cell grids
  dkmp/                 needles, predicted, scored, L90 grids
  longmemeval/          eval_* and judged_* for N=500
  longbench_v2/         multipass scored
data/                   (empty) — see "Data" below
```

### DKMP pipeline (`scripts/dkmp/`)

| Script | Step |
|---|---|
| `01_generate_needles.py` | generate D-hop synthetic chains (entailment-validated) |
| `02_build_contexts.py` | insert needles into distractor contexts at {1K,8K,32K,128K} |
| `00_cache_embeddings.py` | cache BGE-M3 chunk embeddings |
| `03_run_baselines.py` | single-pass baselines (FullCtx / BM25 / Hybrid-RRF / oracle) |
| `03c_run_multipass_cached.py` | MULTIPASS, 1 bridge |
| `03f_run_M7_cached.py` | MULTIPASS, D-pass |
| `04_judge.py` | fresh-instance GLM-4.7 YES/NO judge |
| `05_compute_L90.py` | accuracy grid + bootstrap CIs (10K resamples, seed 42) |

## Data

`data/` is intentionally empty. The DKMP needles and stories are released under
`results/dkmp/` (`needles_v0.jsonl`, `stories_v0.json`); the per-context corpus
(`data/dkmp/contexts_v0.jsonl`) is rebuilt with `02_build_contexts.py`, and the
embedding cache (`data/dkmp/embeddings_cache_v0.npz`, ~386 MB) with
`00_cache_embeddings.py`. NarrativeQA / LongMemEval / LongBench v2 source corpora
are **not** redistributed; place them under `data/` before re-running.

## Reproducibility notes

- Retrieval index: BGE-M3 dense + BM25 fused via reciprocal rank fusion (same
  index for every method).
- Accuracies are LLM-judge accuracies from a fresh GLM-4.7 instance with a YES/NO
  prompt disjoint from the reader prompt.
- Bootstrap CIs: 10K resamples, seed 42.
- Total compute for the released runs: ~12 h on a single laptop (M-series MPS for
  BGE-M3 embeddings).

From *"Bridge Starvation: A Reproducible Failure Mode of Single-Pass Memory
Retrieval, with a Bounded Repair"* (ICML 2026 workshop).
