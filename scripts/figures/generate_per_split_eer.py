#!/usr/bin/env python3
"""
Generates per_split_eer.pdf/png for the ArcDoc paper.

Figure 6: Per-split EER breakdown on LA-CDIP (splits 0-4).
Line chart showing EER per cross-validation split for key methods.

Run from repo root:
    python scripts/figures/generate_per_split_eer.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# EER (%) per split. Sources:
#   ArcDoc: sprint3_epoch_analysis.html per-split best for Sub-Center ArcFace
#   Qwen3-VL-8B, InternVL-3-14B, InternVL-3-2B: results/vlm_metric/*/summary.csv
#   Jina-v4: results/emb_baseline_lacdip/jina-v4/summary.csv
data = {
    "ArcDoc (Sub-Center ArcFace, ours)": [2.14, 0.17, 0.35, 0.75, 0.65],
    "Qwen3-VL-8B (zero-shot, 8B)":       [1.16, 0.56, 0.00, 3.62, 0.26],
    "InternVL-3-14B (zero-shot, 14B)":   [0.58, 0.46, 0.42, 4.86, 0.21],
    "Jina-v4 (embedding)":               [3.57, 1.28, 1.80, 6.29, 2.47],
    "InternVL-3-2B (unadapted)":         [27.90, 26.69, 24.90, 43.61, 26.86],
}

styles = {
    "ArcDoc (Sub-Center ArcFace, ours)": dict(color="#54A24B", marker="o",  lw=2.0, ls="-",  zorder=5),
    "Qwen3-VL-8B (zero-shot, 8B)":       dict(color="#4C78A8", marker="s",  lw=1.5, ls="--", zorder=4),
    "InternVL-3-14B (zero-shot, 14B)":   dict(color="#72B7B2", marker="^",  lw=1.5, ls="--", zorder=3),
    "Jina-v4 (embedding)":               dict(color="#F58518", marker="D",  lw=1.5, ls="-.", zorder=3),
    "InternVL-3-2B (unadapted)":         dict(color="#BAB0AC", marker="x",  lw=1.5, ls=":",  zorder=2),
}

fig, ax = plt.subplots(figsize=(7.5, 4))

for label, eer_vals in data.items():
    st = styles[label]
    ax.plot(splits, eer_vals, label=label, **st)

ax.set_xlabel("Cross-Validation Split")
ax.set_ylabel("EER (%)")
ax.set_title("Per-Split EER Breakdown on LA-CDIP")
ax.set_xticks(splits)
ax.set_xticklabels([f"Split {s}" for s in splits])
ax.set_ylim(bottom=0)

# "lower is better" annotation
ax.text(
    0.99, 0.98, "lower is better",
    transform=ax.transAxes, fontsize=9, color="gray",
    ha="right", va="top",
)

# Legend in upper right — avoids covering the low-EER region of the plot
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "per_split_eer.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "per_split_eer.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved per_split_eer → {OUTPUT_DIR}")
