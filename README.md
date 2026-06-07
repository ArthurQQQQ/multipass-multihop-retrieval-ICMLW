# Bridge Starvation

Code, results, and paper source for

> **Bridge Starvation: A Reproducible Failure Mode of Single-Pass Memory
> Retrieval, with a Bounded Repair** (ICML 2026).

**Bridge starvation** is a retrieval-layer failure of single-pass agentic
memory: a query names entity *X* but the answer requires a chain
*X → Y → Z → …*; the first retrieval pass finds the *X* fact, but the bridge
entities are absent from the query, so the remaining links are never retrieved.

This repo provides:

- **DKMP** — a controlled trigger that varies chain depth (2–5 hops) and context
  length (1K–128K tokens) over synthetic-entity needles embedded in real
  NarrativeQA distractor contexts.
- **MULTIPASS** — a bounded repair that reuses the same retriever but asks the
  LLM to emit one short bridge entity (≤ 6 words) between retrieval passes,
  looping until the bridge call emits `NONE`.
- **Trace diagnostics** that separate retrieval coverage from reader composition.

## Headline results (N = 128K, GLM-4.7 LLM-judged, n = 30/cell)

| Hop | Single-pass Hybrid-RRF | MULTIPASS | Oracle (gold needles) |
|----:|:----------------------:|:---------:|:---------------------:|
| 2   | 0.40 | **1.00** | 1.00 |
| 3   | 0.47 | **0.90** | 0.93 |
| 4   | 0.57 | **0.97** | 1.00 |
| 5   | 0.13 | 0.53 | 0.76 |

MULTIPASS reaches oracle on 2-hop and is within 3pp of oracle on 3/4-hop, above
IRCoT, ReAct, Self-Ask, and a passage-graph PPR variant under the same reader and
index. The repair is intentionally **bounded**: it does not solve 5-hop chains
(coverage falls to 0.75 and reader composition becomes a second failure mode), it
does not help on short-context MuSiQue (the trigger does not fire), and it does
not target LongMemEval's cross-session aggregation failure.

## Layout

```
Bridge Starvation/
├── README.md
├── paper/                  # ICML 2026 submission source (paper.tex / paper.pdf) + figures
├── scripts/                # all code
│   ├── dkmp/               # DKMP synthetic trigger pipeline (00–05 + helpers)
│   ├── longmemeval_*.py    # LongMemEval same-reader fairness check
│   └── lb2_* / *_lb2.py    # LongBench v2 multipass extraction & eval
├── results/                # released predictions, scores, and per-cell grids
│   ├── dkmp/               # needles, predicted, scored, L90 grids
│   ├── longmemeval/        # eval_* and judged_* for N=500
│   └── longbench_v2/       # multipass scored
├── docs/                   # detailed reports, dossiers, and the D-pass theory note
└── data/                   # (empty) — see "Data" below
```

## DKMP pipeline (`scripts/dkmp/`)

| Script | Step |
|---|---|
| `01_generate_needles.py` | Generate D-hop synthetic-entity chains (validated by entailment) |
| `02_build_contexts.py` | Insert needles into distractor contexts at {1K, 8K, 32K, 128K} |
| `00_cache_embeddings.py` | Cache BGE-M3 chunk embeddings for the contexts |
| `03_run_baselines.py` | Single-pass baselines (FullCtx / BM25 / Hybrid-RRF / oracle) |
| `03c_run_multipass_cached.py` | MULTIPASS, 1 bridge |
| `03f_run_M7_cached.py` | MULTIPASS, D-pass |
| `04_judge.py` | Fresh-instance GLM-4.7 YES/NO judge |
| `05_compute_L90.py` | Accuracy grid + bootstrap CIs (10K resamples, seed 42) |

## Reproducing

All paths are relative to the repo root. Re-running requires the source data and
embedding caches (see **Data**); the released `results/` already contain the
predictions and scores behind every table.

```bash
# DKMP multi-hop (after data/ is populated and embeddings cached)
python scripts/dkmp/03_run_baselines.py
python scripts/dkmp/03c_run_multipass_cached.py
python scripts/dkmp/03f_run_M7_cached.py
python scripts/dkmp/04_judge.py
python scripts/dkmp/05_compute_L90.py

# LongMemEval N=500 same-reader check
python scripts/longmemeval_eval_methods.py --method hybrid_rrf --char-budget 8000 --K 10 --out-name full_n500
python scripts/longmemeval_llm_judge.py --method hybrid_rrf --out-name full_n500
```

## Data

The `data/` directory is intentionally empty in this archive. The DKMP needles
and stories are released under `results/dkmp/` (`needles_v0.jsonl`,
`stories_v0.json`); the per-context corpus (`data/dkmp/contexts_v0.jsonl`) can be
rebuilt with `02_build_contexts.py`, and embedding caches
(`data/dkmp/embeddings_cache_v0.npz`, ~386 MB) with `00_cache_embeddings.py`.

NarrativeQA source text and the LongMemEval / LongBench v2 corpora are **not
redistributed** here; place them under `data/` (NarrativeQA full text,
LongMemEval cleaned JSON, LongBench v2 nodes) before re-running the pipelines.

## Configuration

The reader and judge use GLM-4.7 via an API gateway. Set credentials through
environment variables (or a local `.env`, which is git-ignored):

```
GLM_URL=...
GLM_API_KEY=...
# GPT/OpenAI cross-reader slice:
OPENAI_URL=...
OPENAI_API_KEY=...
```

No keys are committed to this repository.

## Reproducibility notes

- Retrieval index: BGE-M3 dense + BM25 fused via reciprocal rank fusion (same
  index for every method).
- All accuracies are LLM-judge accuracies from a fresh GLM-4.7 instance with a
  YES/NO prompt disjoint from the reader prompt.
- Bootstrap CIs: 10K resamples, seed 42.
- Total compute for the released runs: ~12 hours on a single laptop (M-series MPS
  for BGE-M3 embeddings).
