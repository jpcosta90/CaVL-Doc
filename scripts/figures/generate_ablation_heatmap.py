#!/usr/bin/env python3
"""
Generates ablation_heatmap.pdf/png for the ArcDoc paper.

Two-column figure: per-split cumulative-best EER (%) for each loss function.
"Cumulative best" = min EER seen across Phase 1, Phase 2 with teacher,
and Phase 2 without teacher (full evaluation on complete validation set).

Config: Attention Pooler (MQAP, nq=1), Sub-Center k=3, base prompt.

Rows  = loss functions (4), ordered best → worst by mean EER
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

# Full-evaluation per-split EER (%) — FullEval_Sprint3b runs
# p1      = Phase 1 best epoch EER
# p2_on   = Phase 2 (Teacher ON)  best epoch EER
# p2_off  = Phase 2 (Teacher OFF) best epoch EER
_raw = {
    "Sub-Center CosFace": dict(
        p1    = [1.35, 1.80, 0.46, 2.76, 0.98],
        p2_on = [1.64, 1.33, 0.33, 1.52, 0.21],
        p2_off= [1.25, 1.64, 0.58, 2.29, 0.46],
    ),
    "Sub-Center ArcFace": dict(
        p1    = [1.16, 1.85, 0.42, 2.00, 0.46],
        p2_on = [1.35, 2.05, 0.50, 1.90, 0.41],
        p2_off= [2.12, 2.57, 0.79, 2.00, 0.82],
    ),
    "Triplet": dict(
        p1    = [2.22, 1.59, 0.42, 2.67, 0.93],
        p2_on = [2.90, 4.00, 1.09, 2.57, 1.55],
        p2_off= [4.25, 1.54, 0.67, 4.10, 0.57],
    ),
    "Contrastive": dict(
        p1    = [3.28, 1.69, 1.71, 1.90, 1.96],
        p2_on = [4.73, 3.85, 2.76, 1.62, 2.27],
        p2_off= [3.57, 2.57, 5.72, 2.86, 1.91],
    ),
}

# Losses ordered best → worst by cumulative mean EER
losses = ["Sub-Center CosFace", "Sub-Center ArcFace", "Triplet", "Contrastive"]


def cum_best(d):
    # For each teacher strategy, cumulative best = min(Phase1, Phase2)
    on_vals  = [min(d["p1"][i], d["p2_on"][i])  for i in range(5)]
    off_vals = [min(d["p1"][i], d["p2_off"][i]) for i in range(5)]
    # Pick the strategy with the better mean — consistent with HTML's
    # "Melhor EER Acumulado" which evaluates each teacher strategy separately
    if float(np.mean(on_vals)) <= float(np.mean(off_vals)):
        return on_vals
    return off_vals


per_split = {l: cum_best(_raw[l]) for l in losses}
means     = {l: float(np.mean(per_split[l])) for l in losses}

# 2-D array: rows = losses, cols = split0..4 + mean
mat = np.array([[*per_split[l], means[l]] for l in losses])

col_labels = ["Split 0", "Split 1", "Split 2", "Split 3", "Split 4", "Mean"]
n_rows, n_cols = mat.shape

# Colour map: low EER (good) = ArcDoc green; high EER = orange
cmap = mcolors.LinearSegmentedColormap.from_list(
    "eer", ["#54A24B", "#F7F7F7", "#F58518"], N=256
)

fig, ax = plt.subplots(figsize=(10.5, 2.8))
ax.set_aspect("auto")

vmin, vmax = 0.0, 4.0
im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Annotate every cell
for i in range(n_rows):
    for j in range(n_cols):
        val = mat[i, j]
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

# Column labels on top
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", which="both", length=0, pad=4)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(losses, fontsize=10)
ax.tick_params(axis="y", which="both", length=0, pad=4)

ax.set_title(
    "Per-Split Cumulative Best EER (%) — Loss Function Ablation on LA-CDIP\n"
    r"Config: Attention Pooler (MQAP, $n_q{=}1$), base prompt — best across Teacher ON/OFF",
    pad=14, fontsize=11,
)

# Colour bar
cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.015, aspect=18)
cbar.set_label("EER (%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([0, 1, 2, 3, 4])

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
