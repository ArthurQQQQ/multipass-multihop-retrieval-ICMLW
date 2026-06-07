#!/usr/bin/env python3
"""compare_v0_vs_v0p1.py — Show v0 (orig prompt) vs v0p1 (no-IDK prompt) per cell."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path("/Users/arthurqiu/MemoryNet")
V0 = REPO / "data/dkmp/v0_origprompt/scored_v0.jsonl"
V0P1 = REPO / "data/dkmp/scored_v0.jsonl"


def load(p):
    return [json.loads(l) for l in open(p)]


def grid(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["key_type"], r["target_length"])].append(r["judge_yes"])
    return {k: (sum(v) / len(v), len(v)) for k, v in g.items()}


def main():
    a = grid(load(V0))
    b = grid(load(V0P1))
    methods = ["M0", "M1", "M2", "M3", "M5"]
    keys = ["K1", "K3", "K5"]
    lens = [1000, 8000, 32000, 128000]

    print(f"\n=== v0 (orig 'I don't know' prompt) vs v0p1 (no-IDK prompt) ===\n")
    for k in keys:
        print(f"--- {k} ---")
        print(f"{'method':<5} {'1K':>14} {'8K':>14} {'32K':>14} {'128K':>14}")
        for m in methods:
            line = f"{m:<5}"
            for n in lens:
                a_acc, a_n = a.get((m, k, n), (None, 0))
                b_acc, b_n = b.get((m, k, n), (None, 0))
                if a_acc is None or b_acc is None:
                    line += f" {'--':>14}"
                else:
                    delta = b_acc - a_acc
                    line += f" {a_acc:.2f}→{b_acc:.2f} ({delta:+.2f})"
            print(line)
        print()

    # Stratified clean-needle K5
    needles = list(load(REPO / "data/dkmp/needles_v0.jsonl"))
    needles_by_sk = {(n["story_id"], n["key_type"]): n for n in needles}

    # Classify K5 needles by v0 M5 majority
    v0_m5_k5 = defaultdict(list)
    for r in load(V0):
        if r["method"] == "M5" and r["key_type"] == "K5":
            v0_m5_k5[r["story_id"]].append(r["judge_yes"])
    clean_sids = {sid for sid, outs in v0_m5_k5.items() if sum(outs) >= len(outs) * 0.5}
    ambig_sids = {sid for sid, outs in v0_m5_k5.items() if sum(outs) < len(outs) * 0.5}
    print(f"\nK5 needles: {len(clean_sids)} CLEAN / {len(ambig_sids)} AMBIG\n")

    print(f"=== K5 CLEAN-only (n={len(clean_sids)}) ===")
    print(f"{'method':<5} {'1K':>14} {'8K':>14} {'32K':>14} {'128K':>14}")
    for m in methods:
        line = f"{m:<5}"
        for n in lens:
            a_rows = [r for r in load(V0) if r["method"] == m and r["key_type"] == "K5"
                      and r["target_length"] == n and r["story_id"] in clean_sids]
            b_rows = [r for r in load(V0P1) if r["method"] == m and r["key_type"] == "K5"
                      and r["target_length"] == n and r["story_id"] in clean_sids]
            if not a_rows or not b_rows:
                line += f" {'--':>14}"
                continue
            a_acc = sum(r["judge_yes"] for r in a_rows) / len(a_rows)
            b_acc = sum(r["judge_yes"] for r in b_rows) / len(b_rows)
            line += f" {a_acc:.2f}→{b_acc:.2f} ({b_acc-a_acc:+.2f})"
        print(line)

    print(f"\n=== K5 AMBIG-only (n={len(ambig_sids)}) ===")
    print(f"{'method':<5} {'1K':>14} {'8K':>14} {'32K':>14} {'128K':>14}")
    for m in methods:
        line = f"{m:<5}"
        for n in lens:
            a_rows = [r for r in load(V0) if r["method"] == m and r["key_type"] == "K5"
                      and r["target_length"] == n and r["story_id"] in ambig_sids]
            b_rows = [r for r in load(V0P1) if r["method"] == m and r["key_type"] == "K5"
                      and r["target_length"] == n and r["story_id"] in ambig_sids]
            if not a_rows or not b_rows:
                line += f" {'--':>14}"
                continue
            a_acc = sum(r["judge_yes"] for r in a_rows) / len(a_rows)
            b_acc = sum(r["judge_yes"] for r in b_rows) / len(b_rows)
            line += f" {a_acc:.2f}→{b_acc:.2f} ({b_acc-a_acc:+.2f})"
        print(line)

    # Headline
    print("\n=== HEADLINE: M2 vs M0 at N=128K, v0 → v0p1 ===")
    for k in keys:
        a_m0 = a.get(("M0", k, 128000), (0, 0))[0]
        a_m2 = a.get(("M2", k, 128000), (0, 0))[0]
        b_m0 = b.get(("M0", k, 128000), (0, 0))[0]
        b_m2 = b.get(("M2", k, 128000), (0, 0))[0]
        print(f"  {k}: M0 {a_m0:.2f}→{b_m0:.2f}  M2 {a_m2:.2f}→{b_m2:.2f}  "
              f"Δ_v0={a_m2-a_m0:+.2f}  Δ_v0p1={b_m2-b_m0:+.2f}")


if __name__ == "__main__":
    main()
