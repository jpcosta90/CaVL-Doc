#!/usr/bin/env python3
"""
Generates comparison_lacdip.pdf/png for the ArcDoc paper.

Figure 5: Method comparison on LA-CDIP (average EER, splits 0-4).
Horizontal bar chart with colour-coded method families.

Run from repo root:
    python scripts/figures/generate_comparison_lacdip.py
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
    "legend.fontsize": 10,
})

# Average EER (%) across splits 0-4, sorted best→worst.
methods = [
    "ArcDoc (Sub-Center ArcFace)",
    "Qwen3-VL-8B (zero-shot)",
    "InternVL-3-14B (zero-shot)",
    "InternVL-3-8B (zero-shot)",
    "Qwen3-VL-4B (zero-shot)",
    "Jina-v4",
    "InternVL3-out",
    "InternVL3-in",
    "InternVL-3-2B",
    "Pixel (L2)",
    "Pixel (Cosine)",
]
eer_vals = [0.82, 1.12, 1.30, 2.13, 2.67, 3.08, 4.78, 7.62, 29.99, 31.20, 32.30]

colors = {
    "ArcDoc (Sub-Center ArcFace)": "#54A24B",
    "Qwen3-VL-8B (zero-shot)":     "#4C78A8",
    "InternVL-3-14B (zero-shot)":  "#4C78A8",
    "InternVL-3-8B (zero-shot)":   "#4C78A8",
    "Qwen3-VL-4B (zero-shot)":     "#4C78A8",
    "Jina-v4":                     "#F58518",
    "InternVL3-out":               "#F58518",
    "InternVL3-in":                "#F58518",
    "InternVL-3-2B":               "#4C78A8",
    "Pixel (L2)":                  "#BAB0AC",
    "Pixel (Cosine)":              "#BAB0AC",
}

fig, ax = plt.subplots(figsize=(8, 5.5))

y_pos = np.arange(len(methods))[::-1]  # top-to-bottom: best at top
bar_colors = [colors[m] for m in methods]
bars = ax.barh(y_pos, eer_vals, color=bar_colors, height=0.65,
               edgecolor="white", linewidth=0.4)

for bar, val in zip(bars, eer_vals):
    ax.text(
        bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center", ha="left", fontsize=9,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel("Average EER (%)")
ax.set_title("Method Comparison on LA-CDIP (Avg. EER, Splits 0–4)")
ax.set_xlim(0, 40)

ax.text(
    0.98, -0.08, "lower is better",
    transform=ax.transAxes, fontsize=9, color="gray",
    ha="right", va="top",
)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#54A24B", label="ArcDoc (ours)"),
    Patch(facecolor="#4C78A8", label="Zero-shot VLM baseline"),
    Patch(facecolor="#F58518", label="Embedding baseline"),
    Patch(facecolor="#BAB0AC", label="Pixel baseline"),
]
# Legend at upper right — bars are densest in the lower-right quadrant
ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "comparison_lacdip.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "comparison_lacdip.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved comparison_lacdip → {OUTPUT_DIR}")
