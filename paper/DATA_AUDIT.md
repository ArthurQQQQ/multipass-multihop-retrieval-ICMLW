# Data-Consistency Audit

| Textual claim (old) | Source | Consistent with table? | Action |
|---|---|---|---|
| Abstract: "47--67 accuracy points" gap on multi-hop | Table 1: 2-hop hybrid 0.40 vs oracle 1.00 = 60pp; 3-hop 0.47 vs oracle 0.93 = 46pp; 4-hop 0.57 vs oracle 1.00 = 43pp | Range 43--60, not 47--67 | Updated abstract to "40--60 accuracy points below oracle" |
| Abstract: "MULTIPASS recovers oracle-level accuracy on 2/3/4-hop" | Table 1: M7 1.00/0.90/0.97 vs Oracle 1.00/0.93/1.00 → gaps 0/3/3 pp | Not exactly oracle; within 3pp on 3/4-hop | Changed to "matches oracle on 2-hop and is within 3pp on 3/4-hop" |
| Intro / abstract: "beats four prior-art baselines simultaneously on accuracy and cost" | Table 1: M7 wins 2/3/4-hop. Loses 5-hop to IRCoT 0-shot. Cost: M7 has fewer LLM calls than IRCoT 8-shot, ReAct, Self-Ask but slower latency than FullCtx | Mixed; needs scope condition | Reworded: "wins on the targeted bridge-starvation regime (2/3/4-hop, N=128K)" |
| Sec 6: "MULTIPASS upper-left on both axes" (Figure 1 caption) | M7 156s vs FullCtx 65.8s — FullCtx is faster on latency | Mixed | Reworded: "MULTIPASS is upper-left on both axes among iterative-retrieval methods (IRCoT/ReAct/Self-Ask). FullCtx is faster but uses 22× more reader tokens and is less accurate at long N." |
| Sec 6: "2.6× latency gap vs IRCoT" | M7 156s, IRCoT 8-shot 470s. 470/156 = 3.01 | Wrong (off by 0.4×) | Changed to "≈3×" |
| Sec 6: "Cost per correct: $0.015 / $0.020 / $0.137" | Depends on GLM-4.7 published rate (not directly verifiable) | Unsupportable | Removed dollar costs; report calls + tokens + latency |
| Sec 7: "5-hop oracle 0.76 → reader is the bottleneck" | New diagnostic: M7 5-hop chain coverage = 0.75; M7 acc = 0.53; oracle = 0.76. Of 30 cases: 12 (40%) have full coverage; of those 12, 11 are correct (0.92). Of 18 partial-coverage cases, 5 are correct (0.28). | Reader-only framing is wrong; both retrieval and reader contribute | Rewrote: "MULTIPASS does not solve 5-hop. Coverage diagnostic shows only 40% of 5-hop cases reach full chain coverage; the remaining 60% reflect bridge-stage retrieval misses, not bridge-LLM failure (bridge-entity accuracy = 0.98). Among full-coverage cases, reader accuracy is 11/12 = 0.92, close to oracle's 0.76. Both retrieval and reader contribute, with retrieval the larger drop." |
| Tab 1 vs abstract: 5-hop "regime change where reader becomes bottleneck" | New diagnostic refutes single-cause framing | Inconsistent | Reframed as boundary case with two-mode failure |
| Sec 8 Cross-Reader: "MULTIPASS beats single-pass by +33--57pp on either reader" | GPT-4o-mini M7: 0.97/0.80/0.90; GLM Hybrid-RRF: 0.40/0.47/0.57 → gaps 57/33/33 pp | Consistent | Kept; range 33--57 is correct |
| Sec 9 MuSiQue: "MULTIPASS 0.490, ReAct 0.493, FullCtx 0.493" | Tab 5 | Tie | Reworded "MULTIPASS, ReAct, FullCtx tie at ≈0.49; oracle is 0.75" |
| Sec 10 LongMemEval: "Hybrid-RRF beats Mem0-style by +5.2pp" | Tab 6: 0.614 vs 0.562 → 5.2 pp | Consistent | Kept; framing softened to "same-reader sanity check" |
| Sec 10 LongMemEval: cited "EverMemOS 0.83, LiCoMemory 0.738" | Both citations removed | N/A | Numeric mentions removed |
| Sec 10: "+25.4pp paired bootstrap, P=1.000 (10K resamples)" | This was computed in earlier RESULTS_DOSSIER for hybrid_rrf vs full_context; n=500 | Consistent (verified in earlier docs) | Kept |
| Appendix: max_passes ablation accuracies (lenient match) | data/dkmp/ablation_max_hops.jsonl | Consistent with computed table | Kept |
| Appendix: bridge-prompt ablation | data/dkmp/ablation_bridge_prompt.jsonl | Consistent | Kept |
| Appendix: top-K ablation | data/dkmp/ablation_topk.jsonl | Consistent | Kept |

## New diagnostic numbers added (with sources)

| Claim | Source | Verified value |
|---|---|---|
| M7 chain coverage at 2-hop, N=128K | data/dkmp/predicted_v0.jsonl, recall_needle field, M7 + K6 + N=128K | 1.00 (n=30) |
| M7 chain coverage at 3-hop | same, K7 | 0.98 |
| M7 chain coverage at 4-hop | same, K8 | 0.97 |
| M7 chain coverage at 5-hop | same, K9 | 0.75 |
| Bridge-entity accuracy (non-NONE bridges hitting gold entity), avg | data/dkmp/predicted_v0.jsonl, bridges field, vs needles_v0 entity field | 95--100% across all hop depths |
| 2-hop bootstrap 95% CI for M7 | n=30, 10K resamples | 1.00 [1.00, 1.00] |
| 3-hop bootstrap 95% CI for M7 | n=30 | 0.90 [0.80, 1.00] |
| 4-hop bootstrap 95% CI for M7 | n=30 | 0.97 [0.90, 1.00] |
| 5-hop bootstrap 95% CI for M7 | n=30 | 0.53 [0.37, 0.70] |

## Numbers held to old definition (no change)

- Length curve at K7 (Table 2): all values match scored_v0.jsonl verified.
- Cost (Table 3): calls and latency match cost-Pareto data; tokens reflect retrieved_tokens field.
- LongMemEval (Table 6): all numbers match judged_*_full_n500.jsonl.
- MuSiQue (Table 5): all numbers match musique_judged_val_n300.jsonl.
