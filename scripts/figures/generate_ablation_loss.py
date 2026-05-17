#!/usr/bin/env python3
"""
Generates ablation_loss.pdf/png for the ArcDoc paper.

Figure 4: Loss function ablation on LA-CDIP.
Three bars per loss function show the cumulative best EER:
  - Phase 1 only (epochs 1-10, no mining)
  - Phase 1+2 with teacher mining (epochs 1-15, cumulative best)
  - Phase 1+2 without mining (epochs 1-15, cumulative best)

Cumulative best: min(best_phase1, best_phase2_only) per split, then averaged.
Data: results/sprint3_epoch_analysis.html per-split tables.

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

# Raw per-split best EER (%) from sprint3_epoch_analysis.html
# p1 = best epoch among 1-10 (Phase 1 only)
# p2_on / p2_off = best epoch among 11-15 only (Phase 2, with/without teacher)
_raw = {
    "Sub-Center\nArcFace": dict(
        p1    = [2.14, 0.17, 0.35, 0.75, 0.65],
        p2_on = [2.52, 0.35, 0.53, 0.75, 1.31],
        p2_off= [2.16, 0.35, 0.53, 0.75, 0.98],
    ),
    "Sub-Center\nCosFace": dict(
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

def _cum(p1, p2):
    """Cumulative best: the best epoch seen across all 15 epochs."""
    return [min(a, b) for a, b in zip(p1, p2)]

def _avg(vals):
    return sum(vals) / len(vals)

losses = list(_raw.keys())
phase1     = [_avg(_raw[l]["p1"])                           for l in losses]
cum_on     = [_avg(_cum(_raw[l]["p1"], _raw[l]["p2_on"]))   for l in losses]
cum_off    = [_avg(_cum(_raw[l]["p1"], _raw[l]["p2_off"]))  for l in losses]

x = np.arange(len(losses))
width = 0.26

fig, ax = plt.subplots(figsize=(7.5, 4))

ax.bar(x - width, phase1,  width, label="Phase 1 only (ep. 1–10)",         color="#4C78A8")
ax.bar(x,         cum_on,  width, label="Phase 1+2, teacher ON (ep. 1–15)", color="#F58518")
ax.bar(x + width, cum_off, width, label="Phase 1+2, teacher OFF (ep. 1–15)",color="#72B7B2")

ax.set_ylabel("Average EER (%)")
ax.set_title("Loss Function Ablation on LA-CDIP (Avg. Splits 0–4)")
ax.set_xticks(x)
ax.set_xticklabels(losses)
ax.set_ylim(0, 4)

ax.text(
    -0.08, 0.5, "lower is better",
    fontsize=9, color="gray", rotation=90,
    ha="center", va="center",
    transform=ax.transAxes,
)

# Annotate global best
all_vals = phase1 + cum_on + cum_off
best_val = min(all_vals)
# Find which bar it belongs to (phase1 for ArcFace)
best_idx = phase1.index(best_val)
ax.annotate(
    f"Best\n{best_val:.2f}%",
    xy=(x[best_idx] - width, best_val),
    xytext=(x[best_idx] + 0.5, best_val + 0.8),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=9, ha="left",
)

ax.legend(loc="upper right", frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ablation_loss.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "ablation_loss.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved ablation_loss → {OUTPUT_DIR}")
