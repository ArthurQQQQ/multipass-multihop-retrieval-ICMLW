# Revision Change Log

Revision pass per FAGEN reviewer guidance. Goal: reframe as fault-injection
harness + measured repair, not "retrieval method beats IRCoT".

## 1. Framing changes

| Before | After |
|---|---|
| "MULTIPASS: A new retrieval method that beats IRCoT/ReAct" | "A controlled fault-injection harness for agentic-memory retrieval, plus a minimal repair with measured cost and known boundaries" |
| Abstract leads with "sharp failure mode... we propose MULTIPASS" | Abstract leads with operational definition (bridge starvation), trigger (DKMP), trace diagnostic, repair, and explicit boundary |
| "Production agentic-memory systems all use single-pass top-K" | "Many proposed agentic-memory systems expose a single-pass retrieval interface at inference time; the exact pipelines differ" |
| "MULTIPASS dominates / wins / oracle-level" (unconditional) | "MULTIPASS within 3pp of oracle on 2/3/4-hop"; explicit scope conditions on every comparison |
| Section 7: defensive ("looks like a defeat... two observations recover the picture") | Section 7: cold ("MULTIPASS does not solve 5-hop. Coverage diagnostics show both retrieval and reader contribute") |
| LongMemEval framed as "production systems may overstate contribution" | LongMemEval framed as "same-reader fairness check; does not refute originals" |
| 6 separate "honest negative result" / "we document this rather than tune" phrases | Reduced to one boundary statement per result |

## 2. New diagnostics added (FAGEN topic 2 + 3)

- **Boxed failure definition** (Section 3): triggering preconditions and explicit non-trigger boundaries.
- **Minimal repair trace table** (Section 5, schematic + concrete): per-pass retrieval, bridge entity, chain coverage, outcome.
- **Chain-coverage diagnostic table** (Section 5): for 2/3/4/5-hop at N=128K, reports avg gold-needle recall, final answer accuracy, oracle accuracy, and the (coverage=1, wrong-answer) vs (coverage<1, correct) breakdown.
- **Bridge-entity accuracy** (Section 5): of all non-NONE bridges emitted by MULTIPASS, fraction that name a gold chain entity. Computed at all four hop depths.
- **Bootstrap 95% CIs** (Table 1): n=30/cell, 10K resamples.

## 3. Citations fixed/removed

| Citation | Action |
|---|---|
| `mem0blog2026` (Mem0 Team blog) | Kept as @misc; cited only as one example of single-pass interface |
| `memoryos2025emnlp` | Kept (verified EMNLP 2025 oral) |
| `licomemory` (preprint) | Removed; could not verify |
| `evermemos2026` (preprint) | Removed; could not verify |
| `xu2025amem` (A-Mem NeurIPS 2025) | Kept |
| `wu2024longmemeval` | Kept (LongMemEval ICLR 2025) |
| `trivedi2023ircot`, `yao2023react`, `press2023selfask`, `jiang2023flare`, `gutierrez2024hipporag`, `liu2024lostmiddle`, `musique` | Kept (peer-reviewed) |
| "GLM-4.7's published rate" | Removed; cost given in API calls and tokens, not USD |
| "EverMemOS 0.83, LiCoMemory 0.738 on LongMemEval" | Removed (unverifiable) |

## 4. Numeric claims corrected

| Old text | Old number | New text | New number |
|---|---|---|---|
| "MULTIPASS recovers oracle-level accuracy on 2/3/4-hop" | "oracle-level" | "MULTIPASS recovers near-oracle accuracy on 2/3/4-hop (within 3pp of oracle)" | within 3pp |
| "2.6× latency gap vs IRCoT" | 2.6× | "≈3× latency gap vs IRCoT" (470/156=3.0) | 3.0× |
| "MULTIPASS dominates 4 prior-art baselines simultaneously on accuracy and cost" (unconditional) | unconditional | "MULTIPASS dominates 4 prior-art baselines on the targeted bridge-starvation regime (2/3/4-hop, N=128K)" | scoped |
| "5-hop oracle drops to 0.76 → reader is bottleneck" | reader-only | "5-hop M7 chain-coverage = 0.75; oracle = 0.76. Both retrieval and reader contribute" | both |
| Table 5 MuSiQue "MULTIPASS 0.490" framed against ReAct 0.493 | implicit win | "MULTIPASS, FullCtx, ReAct cluster around 0.49 (tie)" | tie |
| "$0.137 / $0.020 / $0.015 per correct" | dollar costs | Removed; reported as tokens/calls/latency only | removed |

## 5. Style edits (AI-generated phrasing reduced)

Removed or rewritten:
- "sharp failure" → "deterministic failure under the specified preconditions"
- "uncomfortable but worth stating" → removed
- "recovers the picture" → removed
- "one line of code" → removed (the implementation is more than one line)
- "dominates" (unconditional) → "wins on 2/3/4-hop" (scoped)
- "oracle-level" → "near-oracle (within 3pp)" or quoted exact gaps
- "production systems" (universal) → "deployed or proposed systems often"
- "we document this rather than tune it away" → removed (implicit in section structure)
- "the implication is uncomfortable" → removed
- "recovers oracle-level accuracy" → "approaches oracle accuracy"

## 6. Structure changes

Section order reorganized to match FAGEN-suggested layout:
1. Introduction
2. Related Work
3. Failure Definition and DKMP Trigger (with boxed definition + non-trigger boundaries)
4. MULTIPASS Repair (algorithm + cost model + targeted regime)
5. Trace Diagnostics (minimal trace + chain coverage + bridge accuracy)
6. Main Results (Table 1 + length curve + cost-Pareto)
7. Boundary and Negative Results (5-hop with diagnostic; MuSiQue tie; LongMemEval as same-reader sanity)
8. Limitations
9. Conclusion

## 7. Remaining limitations / honest gaps

- n=30/cell (CIs ±10-15pp). Larger n in preparation but not yet run.
- Two readers tested (GLM-4.7 main, GPT-4o-mini at n=30). Claude/Llama not run.
- Synthetic chains. Natural multi-hop at long context (HotpotQA-long, custom MuSiQue extension) not yet built.
- HippoRAG-style PPR uses chunk-cosine edges, not OpenIE entity edges.
- IRCoT 8-shot demonstrations are hand-crafted DKMP-style, not directly transferred from the original IRCoT HotpotQA prompts.
- No theoretical bound for D-pass = D-hop; empirical only.
- Bridge-drift error analysis is by sampling; full taxonomy left for future work.
