#!/usr/bin/env python3
"""
Generates ablation_heatmap.pdf/png for the ArcDoc paper.

Two-column figure: per-split cumulative-best EER (%) for each loss function.
"Cumulative best" = min EER seen across all 15 training epochs (Phases 1 and 2
combined, with and without teacher mining).

Rows  = loss functions (4)
Columns = Split 0–4 + Mean (6)
Colour encodes EER: green (low/good) → white → orange (high/bad).
A black border marks the best loss in each column.

Run from repo root:
    python scripts/figures/generate_ablation_heatmap.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
})

# Raw per-split best EER (%) from sprint3_epoch_analysis.html
# p1 = best epoch 1-10; p2_on/p2_off = best epoch 11-15 only
_raw = {
    "Sub-Center ArcFace": dict(
        p1    = [2.14, 0.17, 0.35, 0.75, 0.65],
        p2_on = [2.52, 0.35, 0.53, 0.75, 1.31],
        p2_off= [2.16, 0.35, 0.53, 0.75, 0.98],
    ),
    "Sub-Center CosFace": dict(
        p1    = [1.97, 1.91, 0.18, 1.81, 0.65],
        p2_on = [1.97, 1.91, 0.18, 1.81, 0.98],
        p2_off= [1.97, 1.91, 0.00, 1.96, 0.98],
    ),
    "Triplet": dict(
        p1    = [2.33, 1.73, 0.17, 1.51, 0.49],
        p2_on = [2.16, 2.44, 0.52, 1.66, 0.49],
        p2_off= [3.77, 1.91, 0.17, 1.81, 0.49],
    ),
    "Contrastive": dict(
        p1    = [3.58, 1.20, 0.70, 3.93, 1.31],
        p2_on = [3.96, 1.20, 1.05, 3.32, 1.47],
        p2_off= [4.31, 1.73, 0.70, 3.63, 1.31],
    ),
}

losses = list(_raw.keys())

def cum_best(d):
    return [min(d["p1"][i], d["p2_on"][i], d["p2_off"][i]) for i in range(5)]

per_split = {l: cum_best(_raw[l]) for l in losses}
means     = {l: float(np.mean(per_split[l])) for l in losses}

# 2-D array: rows = losses, cols = split0..4 + mean
mat = np.array([[*per_split[l], means[l]] for l in losses])

col_labels = ["Split 0", "Split 1", "Split 2", "Split 3", "Split 4", "Mean"]
row_labels  = losses
n_rows, n_cols = mat.shape

# Colour map: low EER (good) = ArcDoc green; high EER = orange
cmap = mcolors.LinearSegmentedColormap.from_list(
    "eer", ["#54A24B", "#F7F7F7", "#F58518"], N=256
)

fig, ax = plt.subplots(figsize=(10.5, 2.8))
ax.set_aspect("auto")

vmin, vmax = 0.0, 4.2
im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Annotate every cell
for i in range(n_rows):
    for j in range(n_cols):
        val = mat[i, j]
        # Pick text colour for readability against background
        norm_val = (val - vmin) / (vmax - vmin)
        bg_rgb = np.array(cmap(norm_val)[:3])
        luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        txt_color = "white" if luminance < 0.45 else "#1a1a1a"
        weight = "bold" if j == n_cols - 1 else "normal"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color=txt_color, fontweight=weight)

# Black border on best (min) loss per column
for j in range(n_cols):
    best_i = int(np.argmin(mat[:, j]))
    rect = mpatches.FancyBboxPatch(
        (j - 0.495, best_i - 0.495), 0.99, 0.99,
        linewidth=2.2, edgecolor="black", facecolor="none",
        boxstyle="square,pad=0", clip_on=False,
    )
    ax.add_patch(rect)

# Dashed separator before the Mean column
ax.axvline(n_cols - 1 - 0.5, color="black", linewidth=1.2,
           linestyle="--", alpha=0.55)

# Axes labels — column labels on top
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", which="both", length=0, pad=4)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=10)
ax.tick_params(axis="y", which="both", length=0, pad=4)

ax.set_title(
    "Per-Split Cumulative Best EER (%) — Loss Function Ablation on LA-CDIP",
    pad=28, fontsize=11,
)

# Colour bar
cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.015, aspect=18)
cbar.set_label("EER (%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([0, 1, 2, 3, 4])

# Remove outer spines for a cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(
    1.065, -0.06, "lower is better",
    transform=ax.transAxes, fontsize=8, color="gray",
    ha="right", va="top",
)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ablation_heatmap.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "ablation_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved ablation_heatmap → {OUTPUT_DIR}")
