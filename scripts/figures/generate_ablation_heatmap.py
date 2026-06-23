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
    "font.size": 10,
    "axes.titlesize": 11,
    "mathtext.fontset": "cm",
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


def cum_best_with_phase(d):
    """Returns (values, phase_labels, teacher_strategy)."""
    on_vals  = [min(d["p1"][i], d["p2_on"][i])  for i in range(5)]
    off_vals = [min(d["p1"][i], d["p2_off"][i]) for i in range(5)]
    if float(np.mean(on_vals)) <= float(np.mean(off_vals)):
        vals, strategy = on_vals, "on"
    else:
        vals, strategy = off_vals, "off"
    # Phase label per split
    labels = []
    for i in range(5):
        p2 = d["p2_on"][i] if strategy == "on" else d["p2_off"][i]
        if d["p1"][i] <= p2:
            labels.append("P1")
        else:
            labels.append("P2+M" if strategy == "on" else "P2")
    return vals, labels


per_split_data = {l: cum_best_with_phase(_raw[l]) for l in losses}
per_split = {l: per_split_data[l][0] for l in losses}
phase_labels = {l: per_split_data[l][1] for l in losses}
means     = {l: float(np.mean(per_split[l])) for l in losses}

# 2-D array: rows = losses, cols = split0..4 + mean
mat = np.array([[*per_split[l], means[l]] for l in losses])

col_labels = ["Split 0", "Split 1", "Split 2", "Split 3", "Split 4", "Mean"]
n_rows, n_cols = mat.shape

# Grayscale: medium gray (best) → very light (worst) — dark text readable throughout
cmap = mcolors.LinearSegmentedColormap.from_list(
    "eer_gray", ["#888888", "#F2F2F2"], N=256
)

fig, ax = plt.subplots(figsize=(10.5, 2.8))
ax.set_aspect("auto")

# Per-column ordinal normalisation: 1st/2nd/3rd/4th place get equal visual spacing
# regardless of how close the actual EER values are.
mat_norm = np.zeros_like(mat)
for j in range(n_cols):
    col = mat[:, j]
    ranks = np.argsort(np.argsort(col)).astype(float)   # 0 = best (lowest EER)
    mat_norm[:, j] = ranks / (len(col) - 1)             # normalise to [0, 1]

im = ax.imshow(mat_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

# Row with best mean EER — entire row gets bold text
best_row = int(np.argmin([means[l] for l in losses]))

# Annotate every cell: EER value + phase label (P1 / P2+T)
phase_color_p1 = "#555555"
phase_color_p2 = "#1a6b2e"   # dark green for Phase 2 + Teacher
for i in range(n_rows):
    loss = losses[i]
    for j in range(n_cols):
        val = mat[i, j]
        norm_val = mat_norm[i, j]
        txt_color = "#1a1a1a"
        weight = "bold" if j == n_cols - 1 else "normal"
        if j < n_cols - 1:   # split columns only, not Mean
            phase = phase_labels[loss][j]
            p_color = phase_color_p2 if "P2" in phase else phase_color_p1
            ax.text(j, i - 0.20, f"{val:.2f}", ha="center", va="center",
                    fontsize=9.5, color=txt_color, fontweight=weight)
            ax.text(j, i + 0.24, phase, ha="center", va="center",
                    fontsize=8.5, color=p_color, fontstyle="italic")
        else:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9.5, color=txt_color, fontweight=weight)

# Dark border on the best (min EER) cell per column
for j in range(n_cols):
    best_i = int(np.argmin(mat[:, j]))
    rect = mpatches.FancyBboxPatch(
        (j - 0.495, best_i - 0.495), 0.99, 0.99,
        linewidth=1.5, edgecolor="#111111", facecolor="none",
        boxstyle="square,pad=0", clip_on=False,
    )
    ax.add_patch(rect)

# Dashed separator before the Mean column
ax.axvline(n_cols - 1 - 0.5, color="#555555", linewidth=0.9,
           linestyle="--", alpha=0.60)

# Column labels on top
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", which="both", length=0, pad=4)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(losses, fontsize=10)
ax.tick_params(axis="y", which="both", length=0, pad=6)


# Colour bar — relative rank per split (not absolute EER)
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.018, aspect=20)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["best", "worst"], fontsize=8)
cbar.set_label("rank\n(per split)", fontsize=8, labelpad=4)
cbar.ax.tick_params(length=2)
cbar.outline.set_linewidth(0.6)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.text(
    0.99, -0.04, "lower is better",
    transform=ax.transAxes, fontsize=8.5, color="#888888",
    ha="right", va="top",
)
ax.text(
    0.01, -0.04,
    r"$\it{P1}$ = best in Phase 1 (epochs 1–10)   "
    r"$\it{P2{+}M}$ / $\it{P2}$ = best in Phase 2 (epochs 11–15), w/ mining / w/o mining",
    transform=ax.transAxes, fontsize=7.5, color="#555555",
    ha="left", va="top",
)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ablation_heatmap.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "ablation_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved ablation_heatmap → {OUTPUT_DIR}")
