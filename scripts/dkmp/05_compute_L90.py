"""05_compute_L90.py - L_90 metric + bootstrap CI per (method, key_type).

L_90 = max N s.t. accuracy(N) >= 0.9 * accuracy(N=N_min)
Pin: N_min = 1000 (the shortest length).
Interp: log-linear between two points bracketing the threshold.
CI: 1000x bootstrap resample of items.

Output: data/dkmp/L90_grid_v0.json + data/dkmp/REPORT_v0.md
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCORED = REPO / "data/dkmp/scored_v0.jsonl"
GRID_OUT = REPO / "data/dkmp/L90_grid_v0.json"
REPORT_OUT = REPO / "data/dkmp/REPORT_v0.md"


def load_scored() -> list[dict]:
    return [json.loads(l) for l in SCORED.read_text().splitlines() if l.strip()]


def L90(items: list[dict], n_levels: list[int], min_score_for_pin: float = 0.05) -> tuple[float, float]:
    """
    items: list with keys 'target_length' and 'judge_yes'.
    Returns (L90, baseline_score) where baseline_score = accuracy at smallest N.
    """
    by_n = defaultdict(list)
    for it in items:
        by_n[it["target_length"]].append(1 if it["judge_yes"] else 0)
    accs = [(N, np.mean(by_n[N]) if by_n[N] else float("nan")) for N in n_levels]
    accs_clean = [(N, a) for N, a in accs if not math.isnan(a)]
    if not accs_clean:
        return float("nan"), float("nan")
    base_n, base_acc = accs_clean[0]
    if base_acc < min_score_for_pin:
        return float("nan"), base_acc
    threshold = 0.9 * base_acc

    # Find first N where acc < threshold; interp between that N and previous
    above_pairs = []
    below_pair = None
    for N, a in accs_clean:
        if a >= threshold:
            above_pairs.append((N, a))
        else:
            below_pair = (N, a)
            break
    if not above_pairs:
        # Even N_min is below threshold (shouldn't happen since base_acc * 0.9)
        return float(accs_clean[0][0]) / 2.0, base_acc
    if below_pair is None:
        # Never dropped below — return the largest N tested
        return float(accs_clean[-1][0]), base_acc

    N_above, a_above = above_pairs[-1]
    N_below, a_below = below_pair
    if a_above == a_below:
        return float(N_above), base_acc
    # Linear interp on log(N)
    frac = (a_above - threshold) / (a_above - a_below)
    log_L90 = math.log(N_above) + frac * (math.log(N_below) - math.log(N_above))
    return math.exp(log_L90), base_acc


def L90_with_CI(items: list[dict], n_levels: list[int], n_boot: int = 1000, seed: int = 42):
    """Bootstrap items WITHIN each N (resample with replacement)."""
    by_n = defaultdict(list)
    for it in items:
        by_n[it["target_length"]].append(it)
    rng = np.random.default_rng(seed)
    L90s = []
    for _ in range(n_boot):
        sampled_items = []
        for N in n_levels:
            pool = by_n[N]
            if not pool:
                continue
            idx = rng.integers(0, len(pool), size=len(pool))
            sampled_items.extend([pool[i] for i in idx])
        l, _ = L90(sampled_items, n_levels)
        if not math.isnan(l):
            L90s.append(l)
    if not L90s:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(L90s)), float(np.percentile(L90s, 2.5)), float(np.percentile(L90s, 97.5))


def make_grid(scored: list[dict]) -> dict:
    methods = sorted({r["method"] for r in scored})
    keys = sorted({r["key_type"] for r in scored})
    n_levels = sorted({r["target_length"] for r in scored})

    grid = {}
    for m in methods:
        grid[m] = {}
        for k in keys:
            items = [r for r in scored if r["method"] == m and r["key_type"] == k]
            l90, base_acc = L90(items, n_levels)
            l90_med, l90_lo, l90_hi = L90_with_CI(items, n_levels)
            # Per-N accuracies
            by_n = defaultdict(list)
            for it in items:
                by_n[it["target_length"]].append(1 if it["judge_yes"] else 0)
            accs = {N: float(np.mean(by_n[N])) if by_n[N] else None for N in n_levels}
            # Per-N recall and tokens
            recall_by_n = {N: [] for N in n_levels}
            tokens_by_n = {N: [] for N in n_levels}
            latency_by_n = {N: [] for N in n_levels}
            for it in items:
                recall_by_n[it["target_length"]].append(it.get("recall_needle", 1.0))
                tokens_by_n[it["target_length"]].append(it.get("retrieved_tokens", 0))
                latency_by_n[it["target_length"]].append(it.get("latency_s", 0))
            recall_avg = {N: float(np.mean(v)) if v else None for N, v in recall_by_n.items()}
            tokens_avg = {N: float(np.mean(v)) if v else None for N, v in tokens_by_n.items()}
            latency_avg = {N: float(np.mean(v)) if v else None for N, v in latency_by_n.items()}
            grid[m][k] = {
                "L90_point": l90,
                "L90_median": l90_med,
                "L90_ci_lo": l90_lo,
                "L90_ci_hi": l90_hi,
                "baseline_acc_at_min_N": base_acc,
                "n_items_per_N": {N: len(by_n[N]) for N in n_levels},
                "accuracy_by_N": accs,
                "recall_by_N": recall_avg,
                "retrieved_tokens_by_N": tokens_avg,
                "latency_s_by_N": latency_avg,
            }
    return {"methods": methods, "keys": keys, "n_levels": n_levels, "grid": grid}


def fmt_L90(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if v >= 1000:
        return f"{v/1000:.1f}K"
    return f"{v:.0f}"


def write_report(grid: dict):
    methods = grid["methods"]
    keys = grid["keys"]
    n_levels = grid["n_levels"]
    g = grid["grid"]

    lines = [
        "# DKMP v0 Smoke — Report",
        "",
        f"**Scored items**: {sum(g[m][k]['n_items_per_N'][n] for m in methods for k in keys for n in n_levels)} (= 5 methods × 3 keys × 4 lengths × 30 stories at full)",
        f"**Lengths**: {n_levels}",
        f"**Keys**: {keys}",
        f"**Methods**: {methods}",
        "",
        "## L₉₀ Grid (median with 95% bootstrap CI)",
        "",
    ]
    # Header
    header = "| method | " + " | ".join(keys) + " |"
    sep = "|---|" + "|".join("---" for _ in keys) + "|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        row = [m]
        for k in keys:
            cell = g[m][k]
            row.append(f"{fmt_L90(cell['L90_median'])} [{fmt_L90(cell['L90_ci_lo'])},{fmt_L90(cell['L90_ci_hi'])}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Accuracy by N (per method × key)")
    lines.append("")
    for m in methods:
        lines.append(f"### {m}")
        head = "| key | " + " | ".join(f"N={n}" for n in n_levels) + " |"
        sep2 = "|---|" + "|".join("---" for _ in n_levels) + "|"
        lines.append(head)
        lines.append(sep2)
        for k in keys:
            cell = g[m][k]
            row = [k]
            for n in n_levels:
                a = cell["accuracy_by_N"][n]
                row.append("-" if a is None else f"{a:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Recall@10 of needle (retrieval methods only)")
    lines.append("")
    for m in methods:
        if m in ("M0", "M5"):
            continue
        lines.append(f"### {m}")
        head = "| key | " + " | ".join(f"N={n}" for n in n_levels) + " |"
        lines.append(head)
        lines.append("|---|" + "|".join("---" for _ in n_levels) + "|")
        for k in keys:
            cell = g[m][k]
            row = [k]
            for n in n_levels:
                v = cell["recall_by_N"][n]
                row.append("-" if v is None else f"{v:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Retrieved tokens by N (rate axis)")
    lines.append("")
    for m in methods:
        lines.append(f"### {m}")
        head = "| key | " + " | ".join(f"N={n}" for n in n_levels) + " |"
        lines.append(head)
        lines.append("|---|" + "|".join("---" for _ in n_levels) + "|")
        for k in keys:
            cell = g[m][k]
            row = [k]
            for n in n_levels:
                v = cell["retrieved_tokens_by_N"][n]
                row.append("-" if v is None else f"{v:,.0f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    REPORT_OUT.write_text("\n".join(lines))
    print(f"Wrote {REPORT_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    scored = load_scored()
    print(f"Scored rows: {len(scored)}")
    grid = make_grid(scored)
    GRID_OUT.write_text(json.dumps(grid, indent=2))
    print(f"Wrote {GRID_OUT}")
    write_report(grid)


if __name__ == "__main__":
    main()
