#!/usr/bin/env python3
"""
Generates ablation_loss.pdf/png for the ArcDoc paper.

Figure 7: Loss function ablation on LA-CDIP.
Three bars per loss function show the cumulative best EER:
  - Phase 1 only (epochs 1-10, no mining)
  - Phase 1+2 with teacher mining (epochs 1-15, cumulative best)
  - Phase 1+2 without mining (epochs 1-15, cumulative best)

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
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",
})

# ── Muted academic palette ────────────────────────────────────────────────────
C_P1  = "#5B8DB8"   # steel blue  — Phase 1
C_ON  = "#B85840"   # muted brick — Teacher ON
C_OFF = "#5A9070"   # sage green  — Teacher OFF

# ── Data ─────────────────────────────────────────────────────────────────────
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
    return [min(a, b) for a, b in zip(p1, p2)]

def _avg(vals):
    return sum(vals) / len(vals)

losses  = list(_raw.keys())
phase1  = [_avg(_raw[l]["p1"])                          for l in losses]
cum_on  = [_avg(_cum(_raw[l]["p1"], _raw[l]["p2_on"]))  for l in losses]
cum_off = [_avg(_cum(_raw[l]["p1"], _raw[l]["p2_off"])) for l in losses]

x     = np.arange(len(losses))
width = 0.24

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.8))

bars1 = ax.bar(x - width, phase1,  width, label="Phase 1 only (ep. 1–10)",
               color=C_P1,  edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x,          cum_on,  width, label=r"Phase 1+2, Teacher ON (ep. 1–15)",
               color=C_ON,  edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + width,  cum_off, width, label="Phase 1+2, Teacher OFF (ep. 1–15)",
               color=C_OFF, edgecolor="white", linewidth=0.5)

# Value labels on top of each bar
for bars in (bars1, bars2, bars3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="#333333")

# Annotate global best (Phase 1 ArcFace bar)
best_val = min(phase1 + cum_on + cum_off)
best_li  = phase1.index(best_val)
ax.annotate(
    f"best: {best_val:.2f}%",
    xy=(x[best_li] - width, best_val),
    xytext=(x[best_li] + 0.25, best_val + 1.1),
    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9),
    fontsize=8.5, color="#444444", ha="left",
)

ax.set_ylabel("Average EER (%)")
ax.set_title("Loss Function Ablation on LA-CDIP (Avg. Splits 0–4)")
ax.set_xticks(x)
ax.set_xticklabels(losses)
ax.set_ylim(0, 2.8)

ax.text(0.99, 0.98, "lower is better",
        transform=ax.transAxes, fontsize=8, color="#888888",
        ha="right", va="top", style="italic")

ax.legend(loc="upper left", frameon=True, framealpha=0.92,
          edgecolor="#cccccc", handlelength=1.4, handletextpad=0.5)

ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

fig.tight_layout()
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    p = OUTPUT_DIR / f"ablation_loss.{ext}"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    print(f"saved → {p}")
plt.close(fig)
