#!/usr/bin/env python3
"""
Generates ablation_loss.pdf/png for the ArcDoc paper.

Figure 4: Loss function ablation on LA-CDIP.
Average EER across splits 0-4 for each loss × training phase.
Data sourced from sprint3_epoch_analysis per-split summaries.

Run from repo root:
    python scripts/figures/generate_ablation_loss.py
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
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

# Average EER (%) across splits 0-4, per loss × phase.
# Source: results/sprint3_epoch_analysis.html per-split tables.
losses = ["Sub-Center\nArcFace", "Sub-Center\nCosFace", "Triplet", "Contrastive"]

phase1     = [0.812, 1.304, 1.246, 2.144]   # Phase 1 (no mining, epoch best)
phase2_on  = [1.092, 1.370, 1.454, 2.200]   # Phase 2 with teacher mining
phase2_off = [0.954, 1.364, 1.630, 2.336]   # Phase 2 without mining
best       = [0.812, 1.268, 1.212, 2.022]   # Best epoch across all phases

x = np.arange(len(losses))
width = 0.20

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(x - 1.5 * width, phase1,     width, label="Phase 1 (no mining)",  color="#4C78A8")
ax.bar(x - 0.5 * width, phase2_on,  width, label="Phase 2 with mining",  color="#F58518")
ax.bar(x + 0.5 * width, phase2_off, width, label="Phase 2 no mining",    color="#72B7B2")
ax.bar(x + 1.5 * width, best,       width, label="Best (all phases)",     color="#54A24B")

ax.set_ylabel("Average EER (%)")
ax.set_title("Loss Function Ablation on LA-CDIP (Avg. Splits 0–4)")
ax.set_xticks(x)
ax.set_xticklabels(losses)
ax.set_ylim(0, 5)

# "lower is better" rotated label
ax.text(
    -0.08, 0.5, "lower is better",
    fontsize=9, color="gray", rotation=90,
    ha="center", va="center",
    transform=ax.transAxes,
)

# Annotate global best
best_val = min(best)
best_idx = best.index(best_val)
ax.annotate(
    f"Best\n{best_val:.2f}%",
    xy=(x[best_idx] + 1.5 * width, best_val),
    xytext=(x[best_idx] + 2.3 * width, best_val + 0.9),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9,
    ha="left",
)

# Legend in upper right — avoids covering bars
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ablation_loss.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "ablation_loss.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved ablation_loss → {OUTPUT_DIR}")
