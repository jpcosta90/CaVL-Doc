#!/usr/bin/env python3
"""
Generates comparison_lacdip.pdf/png for the ArcDoc paper.

Horizontal bar chart with colour-coded method families.
Sorted best → worst EER (lower is better).

Run from repo root:
    python scripts/figures/generate_comparison_lacdip.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

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

C_OURS  = "#54A24B"   # green
C_VLM   = "#4C78A8"   # blue
C_EMB   = "#F58518"   # orange
C_FEMB  = "#E45756"   # red — fine-tuned embedding
C_PIXEL = "#BAB0AC"   # gray

# (label, eer, color) — sorted best → worst
entries = [
    ("ArcDoc (ours)",              0.91,  C_OURS),
    ("Qwen3-VL-8B (zero-shot)",    1.12,  C_VLM),
    ("InternVL-3-14B (zero-shot)", 1.30,  C_VLM),
    ("InternVL-3-8B (zero-shot)",  2.13,  C_VLM),
    ("Jina-v4 (LoRA fine-tuned)",  2.23,  C_FEMB),
    ("Qwen3-VL-4B (zero-shot)",    2.67,  C_VLM),
    ("Jina-v4",                    3.08,  C_EMB),
    ("InternVL3 (output layer)",   4.78,  C_EMB),
    ("InternVL3 (input layer)",    7.62,  C_EMB),
    ("InternVL-3-2B (zero-shot)",  29.99, C_VLM),
    ("Pixel (L2)",                 31.20, C_PIXEL),
    ("Pixel (Cosine)",             32.30, C_PIXEL),
]

methods   = [e[0] for e in entries]
eer_vals  = [e[1] for e in entries]
bar_colors= [e[2] for e in entries]

fig, ax = plt.subplots(figsize=(8, 6.0))

y_pos = np.arange(len(methods))[::-1]   # best at top
bars  = ax.barh(y_pos, eer_vals, color=bar_colors, height=0.65,
                edgecolor="white", linewidth=0.4)

for bar, val in zip(bars, eer_vals):
    ax.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center", ha="left", fontsize=9,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel("Average EER (%)")
ax.set_title("Method Comparison on LA-CDIP (Avg. EER, Splits 0–4)")
ax.set_xlim(0, 42)

ax.text(0.98, -0.07, "lower is better",
        transform=ax.transAxes, fontsize=9, color="gray",
        ha="right", va="top")

legend_elements = [
    Patch(facecolor=C_OURS,  label="ArcDoc (ours)"),
    Patch(facecolor=C_VLM,   label="Zero-shot VLM"),
    Patch(facecolor=C_FEMB,  label="Fine-tuned embedding"),
    Patch(facecolor=C_EMB,   label="Embedding (unadapted)"),
    Patch(facecolor=C_PIXEL, label="Pixel baseline"),
]
ax.legend(handles=legend_elements, loc="lower right",
          frameon=True, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    p = OUTPUT_DIR / f"comparison_lacdip.{ext}"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    print(f"saved → {p}")
plt.close(fig)
