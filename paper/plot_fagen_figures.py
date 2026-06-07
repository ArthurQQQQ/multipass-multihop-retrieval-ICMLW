from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent

# Values are the audited DKMP GLM-4.7 judged results reported in the paper.
hops = np.array([2, 3, 4, 5])
hybrid = np.array([0.40, 0.47, 0.57, 0.13])
ircot8 = np.array([0.70, 0.63, 0.77, 0.53])
multipass = np.array([1.00, 0.90, 0.97, 0.53])
oracle = np.array([1.00, 0.93, 1.00, 0.76])

lengths = np.array([1, 8, 32, 128])
fullctx_len = np.array([0.97, 0.83, 0.83, 0.50])
hybrid_len = np.array([0.90, 0.80, 0.57, 0.47])
ircot8_len = np.array([0.90, 0.80, 0.67, 0.63])
multipass_len = np.array([0.93, 0.93, 0.90, 0.90])
oracle_len = np.array([0.93, 0.93, 0.87, 0.93])

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)

ax = axes[0]
ax.plot(hops, hybrid, marker="s", color="#d62728", linewidth=1.7, label="single-pass Hybrid-RRF")
ax.plot(hops, ircot8, marker="^", color="#9467bd", linewidth=1.7, label="IRCoT 8-shot")
ax.plot(hops, multipass, marker="o", color="#1f77b4", linewidth=2.0, label="MultiPass")
ax.plot(hops, oracle, marker="D", color="#2ca02c", linewidth=1.7, label="gold-support oracle")
ax.axvspan(1.8, 4.2, color="#1f77b4", alpha=0.07, lw=0)
ax.axvspan(4.8, 5.2, color="#d62728", alpha=0.07, lw=0)
ax.text(3.0, 0.08, "targeted repair regime", ha="center", va="center", color="#1f77b4", fontsize=7)
ax.text(5.0, 0.08, "boundary", ha="center", va="center", color="#d62728", fontsize=7)
ax.set_title("(a) DKMP at 128K context")
ax.set_xlabel("chain depth")
ax.set_ylabel("answer accuracy")
ax.set_xticks(hops)
ax.set_ylim(0.0, 1.05)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.75)
ax.legend(frameon=False, loc="lower left")

ax = axes[1]
ax.plot(lengths, fullctx_len, marker="D", color="#7f7f7f", linewidth=1.6, label="FullCtx")
ax.plot(lengths, hybrid_len, marker="s", color="#d62728", linewidth=1.7, label="single-pass Hybrid-RRF")
ax.plot(lengths, ircot8_len, marker="^", color="#9467bd", linewidth=1.7, label="IRCoT 8-shot")
ax.plot(lengths, multipass_len, marker="o", color="#1f77b4", linewidth=2.0, label="MultiPass")
ax.plot(lengths, oracle_len, marker="D", color="#2ca02c", linewidth=1.4, linestyle="--", label="oracle")
ax.set_xscale("log", base=2)
ax.set_xticks(lengths)
ax.set_xticklabels(["1K", "8K", "32K", "128K"])
ax.set_title("(b) 3-hop length stress test")
ax.set_xlabel("context length")
ax.set_ylabel("answer accuracy")
ax.set_ylim(0.35, 1.02)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.75)
ax.legend(frameon=False, loc="lower left")

for suffix in ("pdf", "png"):
    fig.savefig(OUT / f"fagen_dkmp_summary.{suffix}", bbox_inches="tight", dpi=300)

plt.close(fig)

# One-column version for the main paper. The length-stress numbers are already
# reported in a table, so the compact figure focuses on the 128K trigger.
fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)

ax.plot(hops, hybrid, marker="s", color="#d62728", linewidth=1.6, label="Hybrid-RRF")
ax.plot(hops, ircot8, marker="^", color="#9467bd", linewidth=1.6, label="IRCoT 8-shot")
ax.plot(hops, multipass, marker="o", color="#1f77b4", linewidth=1.9, label="MultiPass")
ax.plot(hops, oracle, marker="D", color="#2ca02c", linewidth=1.5, label="oracle")
ax.axvspan(1.8, 4.2, color="#1f77b4", alpha=0.06, lw=0)
ax.axvspan(4.8, 5.2, color="#d62728", alpha=0.06, lw=0)
ax.set_xlabel("chain depth")
ax.set_ylabel("answer accuracy")
ax.set_xticks(hops)
ax.set_ylim(0.0, 1.05)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.75)
ax.legend(frameon=False, loc="lower left", handlelength=1.2)

for suffix in ("pdf", "png"):
    fig.savefig(OUT / f"dkmp_128k_singlecol.{suffix}", bbox_inches="tight", dpi=300)

plt.close(fig)

# Compact one-column two-panel version for the main text.
plt.rcParams.update(
    {
        "font.size": 5.2,
        "axes.titlesize": 6.0,
        "axes.labelsize": 5.7,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "legend.fontsize": 4.6,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55), constrained_layout=True)

ax = axes[0]
ax.plot(hops, hybrid, marker="s", color="#d62728", linewidth=0.9, markersize=2.6, label="Hybrid")
ax.plot(hops, ircot8, marker="^", color="#9467bd", linewidth=0.9, markersize=2.8, label="IRCoT")
ax.plot(hops, multipass, marker="o", color="#1f77b4", linewidth=1.1, markersize=2.8, label="MultiPass")
ax.plot(hops, oracle, marker="D", color="#2ca02c", linewidth=0.9, markersize=2.8, label="oracle")
ax.axvspan(1.8, 4.2, color="#1f77b4", alpha=0.055, lw=0)
ax.axvspan(4.8, 5.2, color="#d62728", alpha=0.055, lw=0)
ax.set_title("(a) 128K depth")
ax.set_xlabel("chain depth")
ax.set_ylabel("accuracy")
ax.set_xticks(hops)
ax.set_ylim(0.0, 1.05)
ax.grid(True, axis="y", linestyle=":", linewidth=0.35, alpha=0.75)
ax.legend(frameon=False, loc="lower left", handlelength=1.1, borderaxespad=0.15)

ax = axes[1]
ax.plot(lengths, fullctx_len, marker="D", color="#7f7f7f", linewidth=0.8, markersize=2.4, label="FullCtx")
ax.plot(lengths, hybrid_len, marker="s", color="#d62728", linewidth=0.9, markersize=2.5, label="Hybrid")
ax.plot(lengths, ircot8_len, marker="^", color="#9467bd", linewidth=0.9, markersize=2.7, label="IRCoT")
ax.plot(lengths, multipass_len, marker="o", color="#1f77b4", linewidth=1.1, markersize=2.7, label="MultiPass")
ax.plot(lengths, oracle_len, marker="D", color="#2ca02c", linewidth=0.8, linestyle="--", markersize=2.5, label="oracle")
ax.set_xscale("log", base=2)
ax.set_xticks(lengths)
ax.set_xticklabels(["1K", "8K", "32K", "128K"])
ax.set_title("(b) 3-hop length")
ax.set_xlabel("context length")
ax.set_ylim(0.35, 1.02)
ax.grid(True, axis="y", linestyle=":", linewidth=0.35, alpha=0.75)
ax.legend(frameon=False, loc="lower left", handlelength=1.0, borderaxespad=0.15)

for suffix in ("pdf", "png"):
    fig.savefig(OUT / f"dkmp_summary_singlecol.{suffix}", bbox_inches="tight", dpi=300)
