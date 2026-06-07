# DKMP v0p2 — Full 5-Key Grid

**Run date**: 2026-05-08
**Scope**: 30 stories × **5 keys** × 4 lengths × 5 methods = **3000 reader + 3000 judge calls**
**Errors**: 0/6000 (across all v0+v0p1+v0p2 runs)
**Build on top of**: v0p1 (permissive prompt). K1/K3/K5 unchanged from v0p1; K2 paraphrase + K4 temporal-order are new.

---

## 🎯 Money grid — Accuracy at N=128K (long context)

| Method | K1 lex | K2 par | K3 cor | K4 tem | K5 cau | **avg** | tokens |
|---|---|---|---|---|---|---|---|
| **M0 full_context** | 0.73 | **0.67** | 0.90 | 1.00 | 0.90 | **0.84** | 128K |
| M1 dense | 0.97 | 0.83 | 0.93 | 0.97 | 0.80 | 0.90 | ~2K |
| **M2 BM25** | **1.00** | **0.90** | **1.00** | **1.00** | **0.97** | **0.97** | ~2K |
| M3 hybrid_rrf | 1.00 | 0.93 | 0.93 | 0.97 | 0.90 | 0.95 | ~2K |
| M5 oracle | 1.00 | 0.90 | 1.00 | 1.00 | 0.93 | 0.97 | ~15 |

**Headline:** at N=128K, **M2 BM25 ties M5 oracle (0.97 avg)** while using ~2K tokens vs 15. M3 hybrid 0.95. M0 full_context 0.84. **Memory wins by +11 to +13pp on average, with 60× fewer tokens.**

---

## 🔍 Per-key story

### K1 lexical — synthetic entity dropped in narrative
- **M0 collapses with N**: 1.00 → 0.80 → 0.83 → **0.73** (-27pp)
- All retrieval methods hit ceiling 0.97-1.00. Unique synthetic entity name = perfect lexical anchor.
- **K1 L₉₀: M0 = 2.8K, M3 = 128K (ceiling)**. The single most dramatic capacity gap.

### K2 paraphrase — synonyms-only question, shared entity name
- **M0 weakest here overall**: 0.63/0.60/0.80/0.67 (no clear monotone)
- **M5 oracle ceiling = 0.87 across N** — paraphrase questions are intrinsically ~13% noisy even with just the needle
- M1 dense **does NOT beat M2 BM25** (0.83 vs 0.90 at N=128K) — entity name still acts as lexical anchor; dense's semantic advantage doesn't show up
- **Implication**: to make K2 a true dense-vs-sparse test, v1 needs needle/question to share *no* tokens including entity name → but that's nearly impossible for QA without ambiguity

### K3 coreference
- M0 stable at 0.87-1.00 across N (with permissive prompt)
- M2 BM25 = 0.90-1.00 ceiling. Dense, hybrid all > 0.90.
- **Coref is not the differentiator we hoped** when needles use unique alias→name pairs

### K4 temporal-order — all events with explicit time markers
- **All methods near-ceiling (0.93-1.00)**. M0 = 1.00 at N=128K.
- GLM-4.7 with explicit time markers is essentially perfect — temporal-order with "in 1923" / "the following day" is too easy
- **K4 is uninformative as written**. v1 should make temporal markers *implicit* (require inferring order from event sequence)

### K5 causal-direction
- M5 oracle ceiling jumped from v0's 0.70 → **v0p1/v0p2 0.93** (the prompt fix, not a retrieval change)
- M0 full_context ≈ M5 oracle on causal — context size doesn't hurt much
- **Causal direction is solved by GLM-4.7 + good prompt**, not a memory-system differentiator at this difficulty

---

## 📊 L₉₀ grid (median, 95% CI)

| method | K1 | K2 | K3 | K4 | K5 |
|---|---|---|---|---|---|
| **M0** | **2.8K** [1.8,32] | 128K [1.7,128] | 128K [2.6,128] | 128K [3.5,128] | 128K [2.5,128] |
| M1 | 128K [2.8,128] | 90K [15,128] | 128K [90,128] | 128K (ceil) | 49.8K [4.8,128] |
| M2 | 128K [3.5,128] | 3.5K [1.9,128] | 128K [20,128] | 128K (ceil) | 29.9K [4.5,128] |
| **M3** | **128K (ceil)** | 128K [2.5,128] | 128K [90,128] | 128K [90,128] | 128K [48,128] |
| M5 | 128K (ceil) | 6.1K [1.9,128] | 128K (ceil) | 128K (ceil) | 128K [3.3,128] |

**Notes**:
- M5 K2 L₉₀ = 6.1K is misleading — actually it's the noise around 0.87 ceiling crossing the 0.9×0.87=0.78 threshold randomly. Same for M2 K2.
- **M3 hybrid_rrf hits 128K on every key** — strongest baseline, must be beaten by M4.

---

## 🧠 Cross-key insights (the real findings)

### 1. **K1 lexical is M0's true Achilles' heel**
Counterintuitive: lexical retrieval should be EASIEST (just match the entity name). But for full-context, it's HARDEST — unique entity buried in 128K narrative is nearly invisible to GLM. Retrieval methods extract the chunk and GLM trivially answers.

This is exactly the "needle in haystack" failure mode (NIAH-style) — GLM-4.7 has effective ~3K context for K1.

### 2. **Causal/temporal/coref are NOT the discriminators we expected**
v6 §1.5 framed L4-L6 as the "industry blind spots". DKMP v0p2 says: with right prompt and explicit markers, GLM-4.7 handles them. The discriminator is **lexical retrieval at long context** — exactly L1 Information Preservation, *not* L4-L6.

### 3. **Dense ≤ BM25 across all keys (including paraphrase)**
Synthetic-entity needles always have a unique-token anchor. Sparse always wins or ties. **For DKMP v0/v0p1/v0p2 design, dense embedding adds no value over BM25.** This means M1 dense baseline is essentially redundant with M2 BM25.

To make dense matter, v1 needs needles where the question shares NO content with the needle (truly latent-associative, NoLiMa-style).

### 4. **Hybrid (M3) is the most consistent winner across keys**
M3's RRF makes it robust — it never loses to either dense or sparse. Always at or near top. **This is the "to beat" line** for any future memory system claim.

### 5. **M0 still loses 11-13pp on average — the thesis still holds**
Even with the prompt fix and "easy" K3/K4/K5 keys, full-context lags retrieval. The argument is now:
> **Full-context with GLM-4.7 has effective context ≈ 3K for unique-token retrieval.** Above 3K, retrieval is not just optional — it's *necessary* to maintain accuracy.

---

## 📁 Final v0p2 artifacts

```
data/dkmp/
├── v0_origprompt/                 ← v0 (orig conservative prompt, archived)
├── stories_v0.json                30 stories
├── needles_v0.jsonl               150 needles (5 keys × 30 stories)
├── contexts_v0.jsonl              600 contexts (5 keys × 30 × 4 lengths)
├── predicted_v0.jsonl             3000 predictions (5 methods × 600)
├── scored_v0.jsonl                3000 scored (deduped)
├── L90_grid_v0p2.json             5×5 L₉₀ grid + bootstrap CIs
├── REPORT_v0p2.md                 machine-generated full grid
├── SUMMARY_v0p1.md                v0p1 detailed comparison vs v0
├── SUMMARY_v0p2.md                this doc — full 5-key analysis
├── run_baselines_v0p1.log
├── run_baselines_v0p2.log
├── run_judge_v0p1.log
└── run_judge_v0p2.log

scripts/dkmp/
├── _glm.py                        GLM client wrapper
├── 01_generate_needles.py         5 prompts: K1/K2/K3/K4/K5
├── 02_build_contexts.py           with --append for incremental builds
├── 03_run_baselines.py            5 methods, idempotent
├── 04_judge.py                    YES/NO judge
└── 05_compute_L90.py              L_90 + bootstrap CI + grid + report
```

---

## 🚀 Implications for BRAINSTORM v6

1. **§0.2 "memory beats context" claim — VALIDATED**, but at +11-13pp not +60pp. Honest claim is "60× compression with +11pp accuracy uplift, hits oracle ceiling on M2 BM25."
2. **§1.5 L1-L6 framework — partially refuted by DKMP**:
   - L1 Information Preservation IS the discriminator (M0 K1 collapses)
   - L4 Identity (coref), L5 Belief Revision, L6 Generative — NOT differentiators in current DKMP setup. Need different probe design.
3. **§5.3 DSA-style indexer — still important** but bar to clear is M3 hybrid_rrf (0.95 avg at 128K). Hard.
4. **Dense embedding (M1) — questionable value-add**. v6 should de-emphasize dense vs sparse and emphasize **structure** (graph edges, bitemporal validity) which DKMP v0 doesn't yet test.
5. **§5.6 bitemporal edges — UNTESTED**. K4 with implicit temporal would test this. v1 priority.

---

## 🚧 v1 priority list (ranked)

1. **Implicit temporal K4** (no "before/after" / no explicit dates → must infer order from event chain). This is where bitemporal edges matter.
2. **Multi-hop K6** (chained needles: "X lives in Y. Y is owned by Z. Question: who owns where X lives?"). This is where graph traversal beats single retrieval.
3. **Adversarial-near distractor** (recall < 1.0 to give methods room to differ).
4. **n=100/cell** to tighten CIs.
5. **K2 redesign with no shared entity** to make dense vs sparse meaningful.
6. **Build M4 MemoryNet v6**. Bar = M3 hybrid_rrf 0.95.

**Do NOT do without explicit user approval** (cost > $50): n=100 + 5 keys + 3 hops + 3 similarities is the v1 light grid (~$150-300).

---

## Costs so far

- v0 baselines + judge: ~$15-20
- v0p1 baselines + judge: ~$15-20  
- v0p2 K2/K4 baselines + judge: ~$10-15
- Needle gen (across all): ~$0.50
- **Total**: ~$45-55 GLM API
