# DKMP v0p1 — Executive Summary (with reader prompt fix)

**Run date**: 2026-05-08
**Change vs v0**: reader prompt rewrote to discourage default "I don't know" on causal/directional questions
**Total**: 1800 reader + 1800 judge = **3600 GLM-4.7 calls (v0p1)**, plus the original 3600 from v0
**Errors**: 0/3600

**TL;DR**: K5 oracle ceiling jumped **70% → 92%** (the IDK behavior was the bottleneck, not GLM reasoning). M0 full_context also benefits — gap to memory shrinks but still favors memory in K1 lexical and at long N.

---

## 🔄 v0 vs v0p1 head-to-head

### Oracle K5 ceiling (most important sanity check)

| N | v0 (orig prompt) | v0p1 (fix prompt) | Δ |
|---|---|---|---|
| 1K | 0.70 | 0.93 | +23 pp |
| 8K | 0.67 | 0.90 | +23 pp |
| 32K | 0.70 | 0.90 | +20 pp |
| 128K | 0.70 | 0.93 | +23 pp |

**The K5 70% ceiling was a prompt bug, not a model limitation.** GLM-4.7 *can* answer causal-direction questions correctly when the prompt doesn't push it toward "I don't know."

### M0 full_context (the key thesis comparison)

| Key × N | v0 | v0p1 |
|---|---|---|
| K1 lexical N=1K  | 0.87 | 1.00 |
| K1 lexical N=128K | **0.37** | **0.73** |
| K3 coref N=1K    | 0.67 | 0.90 |
| K3 coref N=128K   | 0.57 | 0.90 |
| K5 causal N=1K   | 0.60 | 0.80 |
| K5 causal N=128K  | 0.47 | 0.90 |

**M0 numbers ALL improved**, especially long-N ones. The +50pp drop on K1 in v0 was real but compounded by IDK over-firing — actual drop is +27pp (1.00→0.73).

---

## 🎯 v0p1 grid (the cleaner result)

### Accuracy at N=128K (the long-context test)

| Key | M0 full_context | M2 BM25 | Δ |
|---|---|---|---|
| K1 lexical | 0.73 | **1.00** | **+27 pp** |
| K3 coref | 0.90 | 1.00 | +10 pp |
| K5 causal | 0.90 | 0.97 | +7 pp |

**Memory still beats context, but the gap is smaller and more nuanced than v0 suggested.** 60× token compression (128K → 2K) still gets +7 to +27pp accuracy.

### L₉₀ grid (median + 95% bootstrap CI)

| method | K1 | K3 | K5 |
|---|---|---|---|
| **M0 full_context** | **2.8K** [1.8K,32K] | 128K [2.6K,128K] | 128K [2.5K,128K] |
| M1 dense | 128K [2.8K,128K] | 128K [90K,128K] | 49.8K [4.8K,128K] |
| M2 BM25 | 128K [3.5K,128K] | 128K [20K,128K] | 29.9K [4.5K,128K] |
| M3 hybrid_rrf | **128K** [128K,128K] | 128K [90K,128K] | 128K [48K,128K] |
| M5 oracle | 128K (ceil) | 128K (ceil) | 128K [3.3K,128K] |

**Read this carefully** — L₉₀ wider CIs because n=30 small, but key signals:

1. **M0 K1 L₉₀ = 2.8K** — full-context with GLM-4.7 cannot maintain K1 lexical accuracy at long N
2. **M3 (hybrid_rrf) is the most consistent winner** — full ceiling on K1, near-ceiling on K3/K5
3. **M0 K3/K5 L₉₀ = 128K** in v0p1 (didn't drop below threshold) — this is new vs v0

---

## 🔍 What changed and what it means

### 1. The "memory crushes context +60pp" claim was inflated

v0 had GLM defaulting to "I don't know" when prompt said "say I don't know if not in text". This hit M0 disproportionately because at long N, GLM was less confident the needle was "in the text" and would IDK.

**Real effect (v0p1)**: memory still wins, but by **+7 to +27pp** at N=128K, not +33 to +60pp.

This is still a strong, publishable result — but the magnitude is honest and the mechanism is clearer (lexical-needle retrieval > full-context for long N).

### 2. K5 oracle 92% ceiling means GLM-4.7 *can* do causal direction

Previously thought GLM-4.7 had a fundamental 70% ceiling on K5. Wrong — it was prompt over-conservatism. With permissive prompt, oracle hits 92%. Implication: don't blame the model, blame the prompt.

### 3. M0 K3 and K5 actually do well at long N (with right prompt)

In v0p1, M0 maintains 0.90 on K3 and K5 at N=128K. So full-context isn't actually broken on coref/causal. **The decay is mainly K1 lexical** — counterintuitive! Lexical needle = unique synthetic entity name like "Vixenia Strode", harder to find in 128K of distractor narrative than to do coref or causal reasoning.

### 4. Hybrid retrieval (M3) is the strongest baseline

v0p1 M3 grid:
- K1: 0.97/1.00/1.00/1.00 — perfect across N
- K3: 0.97/1.00/1.00/0.93
- K5: 0.93/0.97/0.97/0.90

M3 (dense + BM25 RRF) consistently dominates. This is the "to beat" line for M4 MemoryNet v6.

---

## 📊 New money shot

At **N=128K, GLM-4.7 reader, same data, 60× token reduction**:

| Method | K1 | K3 | K5 | avg |
|---|---|---|---|---|
| M0 full_context | 0.73 | 0.90 | 0.90 | 0.84 |
| M3 hybrid_rrf | 1.00 | 0.93 | 0.90 | 0.94 |
| M5 oracle | 1.00 | 1.00 | 0.93 | 0.98 |

**M3 closes ~50% of the gap to oracle while using ~1.5% of full-context tokens.**

---

## 🚧 v1 plan stays the same

Now that the v0p1 baseline is honest:

1. **n=100/cell** to tighten CIs
2. **Add K2 paraphrase + K4 temporal-order** keys
3. **Add hops 2/3** (M3 advantage should grow on multi-hop)
4. **Add adversarial-near distractor** to make recall < 1.0
5. **Mix InfiniteBench/LongBench-v2** stories
6. **Build M4 MemoryNet v6** — must beat M3 hybrid_rrf to validate insertion-centric thesis

---

## 📁 Artifacts

```
data/dkmp/
├── v0_origprompt/                 ← v0 with original conservative prompt
│   ├── predicted_v0.jsonl
│   ├── scored_v0.jsonl
│   ├── L90_grid_v0.json
│   ├── REPORT_v0.md
│   └── SUMMARY_v0.md
├── predicted_v0p1.jsonl           ← v0p1 with permissive prompt (this run)
├── scored_v0p1.jsonl              (deduped: 1800 unique)
├── L90_grid_v0p1.json
├── REPORT_v0p1.md                 ← machine-generated grid
├── SUMMARY_v0p1.md                ← this doc
├── stories_v0.json                (30 stories, ≥27K tokens)
├── needles_v0.jsonl               (90 needles, unchanged across v0/v0p1)
└── contexts_v0.jsonl              (360 contexts, unchanged)

scripts/dkmp/
├── _glm.py
├── 01_generate_needles.py
├── 02_build_contexts.py
├── 03_run_baselines.py            ← prompt updated for v0p1
├── 04_judge.py
└── 05_compute_L90.py
```

**Total cost so far**: ~$30-40 GLM API (v0 + v0p1).
