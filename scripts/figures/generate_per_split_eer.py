#!/usr/bin/env python3
"""
Generates per_split_eer.pdf/png for the ArcDoc paper.

Figure 6: Per-split EER breakdown on LA-CDIP (splits 0-4).
Grouped bar chart with a broken y-axis:
  - Lower panel (0-9%): competitive methods
  - Upper panel (22-50%): InternVL-3-2B (unadapted, VLM-prompted), which fails badly

Data sources:
  ArcDoc: sprint3_epoch_analysis.html — per-split best (Sub-Center ArcFace)
  Others: results/vlm_metric/*/summary.csv, results/emb_baseline_lacdip/jina-v4/summary.csv

Run from repo root:
    python scripts/figures/generate_per_split_eer.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

splits = [0, 1, 2, 3, 4]
split_labels = [f"Split {s}" for s in splits]

data = {
    "ArcDoc (ours)":              [2.14, 0.17, 0.35, 0.75, 0.65],
    "Qwen3-VL-8B":                [1.16, 0.56, 0.00, 3.62, 0.26],
    "InternVL-3-14B":             [0.58, 0.46, 0.42, 4.86, 0.21],
    "Jina-v4":                    [3.57, 1.28, 1.80, 6.29, 2.47],
    "InternVL-3-2B (unadap.)":    [27.90, 26.69, 24.90, 43.61, 26.86],
}
colors = {
    "ArcDoc (ours)":           "#54A24B",
    "Qwen3-VL-8B":             "#4C78A8",
    "InternVL-3-14B":          "#72B7B2",
    "Jina-v4":                 "#F58518",
    "InternVL-3-2B (unadap.)": "#BAB0AC",
}

n_methods = len(data)
width     = 0.15
x         = np.arange(len(splits))
offsets   = np.linspace(
    -(n_methods - 1) / 2 * width,
     (n_methods - 1) / 2 * width,
    n_methods,
)

# ── Broken-axis figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 5.2))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.10)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

for (label, eer_vals), offset in zip(data.items(), offsets):
    c = colors[label]
    for ax in (ax_top, ax_bot):
        ax.bar(x + offset, eer_vals, width, label=label,
               color=c, edgecolor="white", linewidth=0.4)

# Y-axis ranges
ax_top.set_ylim(22, 50)
ax_bot.set_ylim(0, 9)

# Hide the boundary spines between the two panels
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(bottom=False, labelbottom=False)

# Diagonal break marks (standard "cut" indicator)
d = 0.018
kw = dict(color="k", clip_on=False, lw=1.2)
ax_top.plot((-d, +d), (-d*3, +d*3), transform=ax_top.transAxes, **kw)
ax_top.plot((1-d, 1+d), (-d*3, +d*3), transform=ax_top.transAxes, **kw)
ax_bot.plot((-d, +d), (1-d*3, 1+d*3), transform=ax_bot.transAxes, **kw)
ax_bot.plot((1-d, 1+d), (1-d*3, 1+d*3), transform=ax_bot.transAxes, **kw)

# Axis labels and title
ax_top.set_title("Per-Split EER Breakdown on LA-CDIP")
ax_bot.set_xlabel("Cross-Validation Split")
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(split_labels)

# Shared y-label centred across both panels
fig.text(0.02, 0.5, "EER (%)", va="center", rotation="vertical", fontsize=11)

ax_top.text(0.99, 0.95, "lower is better",
            transform=ax_top.transAxes, fontsize=9, color="gray",
            ha="right", va="top")

# Grid lines only on bot panel (low range)
ax_bot.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_bot.set_axisbelow(True)
ax_top.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_top.set_axisbelow(True)

for ax in (ax_top, ax_bot):
    ax.spines["right"].set_visible(False)

# Single legend above the figure
handles, labels = ax_bot.get_legend_handles_labels()
# Deduplicate (bars are drawn twice)
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
fig.legend(
    seen.values(), seen.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=True,
    framealpha=0.9,
    fontsize=9,
)

fig.savefig(OUTPUT_DIR / "per_split_eer.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "per_split_eer.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved per_split_eer → {OUTPUT_DIR}")
