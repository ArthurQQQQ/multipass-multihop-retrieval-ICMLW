# Citation Audit

| Citation key | Claim supported in text | Verified? | Action |
|---|---|---|---|
| `trivedi2023ircot` | IRCoT interleaves retrieval with single CoT sentences, originally 8-shot | ✓ ACL 2023 (Trivedi, Balasubramanian, Khot, Sabharwal); reference impl `StonyBrookNLP/ircot` is publicly available | Kept |
| `yao2023react` | ReAct alternates Thought/Action(Search)/Observation | ✓ ICLR 2023 (Yao et al.) | Kept |
| `press2023selfask` | Self-Ask decomposes into follow-ups | ✓ EMNLP 2023 Findings (Press et al.) | Kept |
| `jiang2023flare` | FLARE triggers retrieval on uncertainty | ✓ EMNLP 2023 (Jiang et al.) | Kept; mentioned only in related work |
| `gutierrez2024hipporag` | HippoRAG uses personalized PageRank over OpenIE-derived passage graph | ✓ NeurIPS 2024 (Gutiérrez, Shu, Gu, Yasunaga, Su) | Kept; we use a simplified passage-cosine variant and label it as such |
| `liu2024lostmiddle` | Lost-in-the-middle phenomenon | ✓ TACL 2024 (Liu, Lin, Hewitt, ..., Liang) | Kept |
| `wu2024longmemeval` | LongMemEval benchmark | ✓ ICLR 2025 (Wu, Wang, Yu, Zhang, Chang, Yu) | Kept |
| `xu2025amem` | A-Mem agentic memory; we cite it for note-linking design | ✓ NeurIPS 2025 (Xu, Zhao, Wang, Zhang, Liu, Wang) — confirmed via web search | Kept |
| `musique` | MuSiQue benchmark | ✓ TACL 2022 (Trivedi, Balasubramanian, Khot, Sabharwal) | Kept |
| `mem0blog2026` (Mem0 Team blog) | Cited only as one publicly visible example of single-pass interface | Public blog at mem0.ai/blog (verified via search); not peer-reviewed | Kept as `@misc`; cited softly with "for example" framing |
| `memoryos2025emnlp` (BAI-LAB) | Cited as another example | EMNLP 2025 oral confirmed by web search; GitHub `BAI-LAB/MemoryOS` | Kept |
| ~~`licomemory`~~ | "LiCoMemory ... 73.8% on LongMemEval" | Could not verify a peer-reviewed reference; only secondary mentions | Removed; corresponding numeric claim removed |
| ~~`evermemos2026`~~ | "EverMemOS ... 0.83 on LongMemEval" | Could not verify a peer-reviewed reference; only secondary mentions | Removed; corresponding numeric claim removed |
| GLM-4.7 / dmxapi.cn | Reader model used throughout | Public API; we report calls and tokens, not USD cost | Cost section now reports tokens/calls/latency only |
| GPT-4o-mini | Cross-reader validation | Public OpenAI API; accessed via dmxapi.cn proxy | Reported in §Cross-Reader |

## Removed numeric claims that depended on unverifiable citations

- "LongMemEval reported numbers for EverMemOS (0.83) and LiCoMemory (0.738)" — removed
- "$0.015 / $0.020 / $0.137 per correct answer" — removed (requires GLM-4.7 published rate, which we cannot verify; replaced with tokens/calls/latency)
- "9× cheaper than full-context per correct answer" — removed (depends on $/token claim)

## Production-systems claim (revised)

| Old | New |
|---|---|
| "Production agentic-memory systems --- Mem0, MemoryOS, A-Mem, LiCoMemory, EverMemOS --- ... at the bottom they all retrieve top-K items in one pass" | "Many proposed agentic-memory systems expose a single-pass retrieval interface at inference time: given a query or state, retrieve a small set of memory items and pass them to a reader. Examples include Mem0~\cite{mem0blog2026}, MemoryOS~\cite{memoryos2025emnlp}, and A-Mem~\cite{xu2025amem}; the exact extraction and consolidation pipelines differ substantially. We do not claim universality." |
