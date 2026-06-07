#!/usr/bin/env python3
"""dedup_and_analyze.py - Dedupe scored_v0.jsonl (concurrent judges produced duplicates)
and produce v0_origprompt vs v0p1 comparison.

Usage: python dedup_and_analyze.py
"""
import json
from pathlib import Path
from collections import defaultdict, OrderedDict
import sys

REPO = Path("/Users/arthurqiu/MemoryNet")
SCORED_V0P1 = REPO / "data/dkmp/scored_v0.jsonl"
SCORED_V0 = REPO / "data/dkmp/v0_origprompt/scored_v0.jsonl"
NEEDLES = REPO / "data/dkmp/needles_v0.jsonl"


def dedup_file(path):
    """Keep last occurrence of each (qa_id, method) pair."""
    if not path.exists():
        print(f"  {path} missing")
        return []
    rows = [json.loads(l) for l in open(path)]
    seen = OrderedDict()
    for r in rows:
        seen[(r["qa_id"], r["method"])] = r
    return list(seen.values())


def main():
    v0p1 = dedup_file(SCORED_V0P1)
    v0 = [json.loads(l) for l in open(SCORED_V0)]
    print(f"v0: {len(v0)}  v0p1 dedup'd: {len(v0p1)}")

    # Save dedup'd
    if SCORED_V0P1.exists() and len(v0p1) < sum(1 for _ in open(SCORED_V0P1)):
        backup = SCORED_V0P1.with_suffix(".jsonl.preDedup")
        SCORED_V0P1.rename(backup)
        with open(SCORED_V0P1, "w") as f:
            for r in v0p1: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  dedup'd → {SCORED_V0P1} (backup at {backup})")

    # Build per-cell accuracy
    def grid(rows):
        g = defaultdict(list)
        for r in rows:
            g[(r["method"], r["key_type"], r["target_length"])].append(r["judge_yes"])
        return {k: (sum(v) / len(v), len(v)) for k, v in g.items()}

    a = grid(v0)
    b = grid(v0p1)

    methods = ["M0", "M1", "M2", "M3", "M5"]
    keys = ["K1", "K3", "K5"]
    lens = [1000, 8000, 32000, 128000]

    print(f"\n{'='*80}")
    print(f"  v0 (orig prompt with 'I don't know') vs v0p1 (no-IDK prompt)")
    print(f"  Format: v0_acc → v0p1_acc (Δ)")
    print(f"  {'='*80}\n")
    for k in keys:
        print(f"--- {k} ---")
        print(f"{'method':<5} {'1K':>15} {'8K':>15} {'32K':>15} {'128K':>15}")
        for m in methods:
            line = f"{m:<5}"
            for n in lens:
                a_acc = a.get((m, k, n), (None, 0))[0]
                b_acc = b.get((m, k, n), (None, 0))[0]
                if a_acc is None or b_acc is None:
                    line += f" {'--':>15}"
                else:
                    delta = b_acc - a_acc
                    line += f" {a_acc:.2f}→{b_acc:.2f}({delta:+.2f})"
            print(line)
        print()

    # K5 stratified
    needles = list(json.loads(l) for l in open(NEEDLES))
    v0_m5_k5 = defaultdict(list)
    for r in v0:
        if r["method"] == "M5" and r["key_type"] == "K5":
            v0_m5_k5[r["story_id"]].append(r["judge_yes"])
    clean_sids = {sid for sid, outs in v0_m5_k5.items() if sum(outs) >= len(outs) * 0.5}
    ambig_sids = {sid for sid, outs in v0_m5_k5.items() if sum(outs) < len(outs) * 0.5}
    print(f"\n=== K5 stratified by needle quality ===\n")
    print(f"CLEAN (n={len(clean_sids)}) / AMBIG (n={len(ambig_sids)})\n")

    for label, sids in [("CLEAN", clean_sids), ("AMBIG", ambig_sids)]:
        print(f"--- K5 {label} (n={len(sids)} needles × 4 lengths) ---")
        print(f"{'method':<5} {'1K':>15} {'8K':>15} {'32K':>15} {'128K':>15}")
        for m in methods:
            line = f"{m:<5}"
            for n in lens:
                a_rows = [r for r in v0 if r["method"] == m and r["key_type"] == "K5"
                          and r["target_length"] == n and r["story_id"] in sids]
                b_rows = [r for r in v0p1 if r["method"] == m and r["key_type"] == "K5"
                          and r["target_length"] == n and r["story_id"] in sids]
                if not a_rows or not b_rows:
                    line += f" {'--':>15}"
                    continue
                a_acc = sum(r["judge_yes"] for r in a_rows) / len(a_rows)
                b_acc = sum(r["judge_yes"] for r in b_rows) / len(b_rows)
                line += f" {a_acc:.2f}→{b_acc:.2f}({b_acc-a_acc:+.2f})"
            print(line)
        print()

    # Headline number
    print("\n=== HEADLINE: M2 vs M0 at N=128K, BOTH versions ===\n")
    for k in keys:
        a_m0 = a.get(("M0", k, 128000), (0, 0))[0]
        a_m2 = a.get(("M2", k, 128000), (0, 0))[0]
        b_m0 = b.get(("M0", k, 128000), (0, 0))[0]
        b_m2 = b.get(("M2", k, 128000), (0, 0))[0]
        print(f"  {k}: M0 {a_m0:.2f}→{b_m0:.2f}  M2 {a_m2:.2f}→{b_m2:.2f}  "
              f"Δ_v0={a_m2-a_m0:+.2f}  Δ_v0p1={b_m2-b_m0:+.2f}")


if __name__ == "__main__":
    main()
