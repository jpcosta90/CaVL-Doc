#!/usr/bin/env python3
"""
Generates pooler_prompt_ablation.pdf/png for the ArcDoc paper.

Boxplots (5 splits per box) for each pooler × prompt combination,
with individual split points overlaid. Positions the legend where
it does not overlap data.

Run from repo root:
    python scripts/figures/generate_pooler_prompt_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
})

# ── Data (EER % per split, Sub-Center CosFace, cumulative best) ──────────────
GROUPS = [
    ("Mean Pool",       r"MQAP ($n_q{=}1$)", "Cross-Modal"),
    ("mean",             "mqap",               "cross"),
]
POOLER_LABELS = ["Mean Pool", r"MQAP ($n_q{=}1$)", "Cross-Modal"]
POOLER_KEYS   = ["mean",       "mqap",               "cross"]

vals = {
    "mean": {
        "P0": [0.87, 1.23, 0.21, 0.86, 0.21],
        "Pr": [0.87, 1.23, 0.17, 0.76, 0.10],
        "delta": -0.05,
    },
    "mqap": {
        "P0": [1.25, 1.33, 0.33, 1.52, 0.21],
        "Pr": [0.97, 0.77, 0.42, 1.71, 0.36],
        "delta": -0.09,
    },
    "cross": {
        "P0": [1.35, 0.92, 0.17, 3.62, 0.52],
        "Pr": [0.87, 0.77, 0.04, 2.76, 0.10],
        "delta": -0.41,
    },
}

C_P0   = "#4C78A8"   # steel blue
C_PR   = "#F58518"   # orange
C_PT   = 0.25        # box fill alpha

# x positions: two boxes per group, with gap between groups
#   Mean Pool: 1.0, 1.7   MQAP: 3.3, 4.0   Cross-Modal: 5.6, 6.3
GAP_INTRA = 0.7   # between P0 and Pr within a group
GAP_INTER = 1.6   # gap between groups (extra space)
WIDTH     = 0.55

pos_p0, pos_pr = [], []
for i in range(3):
    base = i * (GAP_INTRA + WIDTH + GAP_INTER)
    pos_p0.append(base)
    pos_pr.append(base + GAP_INTRA)

group_centers = [(p0 + pr) / 2 for p0, pr in zip(pos_p0, pos_pr)]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.2))

def draw_box(ax, pos, data, color, width=WIDTH):
    bp = ax.boxplot(
        data, positions=[pos], widths=width,
        patch_artist=True,
        medianprops=dict(color="#111111", linewidth=2),
        boxprops=dict(facecolor=color, alpha=0.35, linewidth=1.2, edgecolor=color),
        whiskerprops=dict(color=color, linewidth=1.3, linestyle="--"),
        capprops=dict(color=color, linewidth=1.5),
        flierprops=dict(marker="o", color=color, markersize=4, alpha=0.6),
        showfliers=False,  # individual points shown separately
    )
    # Individual split points (jittered slightly so they don't stack)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.07, 0.07, size=len(data))
    ax.scatter([pos + j for j in jitter], data,
               color=color, s=28, zorder=4, alpha=0.85, linewidths=0)
    # Mean marker (diamond)
    ax.scatter([pos], [np.mean(data)],
               marker="D", s=38, color=color, zorder=5,
               edgecolors="white", linewidths=0.8)
    return bp

for i, key in enumerate(POOLER_KEYS):
    draw_box(ax, pos_p0[i], vals[key]["P0"], C_P0)
    draw_box(ax, pos_pr[i], vals[key]["Pr"], C_PR)

    # Δ annotation above the pair
    delta = vals[key]["delta"]
    top = max(max(vals[key]["P0"]), max(vals[key]["Pr"])) + 0.18
    ax.annotate(
        f"Δ −{abs(delta):.2f} pp",
        xy=(group_centers[i], top),
        ha="center", va="bottom", fontsize=9,
        color="#2d6a2d", fontweight="bold",
    )

# Group labels on x-axis
ax.set_xticks(group_centers)
ax.set_xticklabels(POOLER_LABELS, fontsize=11)
ax.tick_params(axis="x", length=0)

ax.set_xlim(pos_p0[0] - 0.7, pos_pr[-1] + 0.7)
ax.set_ylim(-0.05, 4.3)
ax.set_ylabel("EER (%) — splits 0–4")
ax.set_title(
    r"Prompt Conditioning over Pooling Architecture — LA-CDIP"
    "\n(Sub-Center CosFace; ◆ = mean, line = median, points = splits)",
    pad=6,
)

# Subtle vertical separators between groups
for i in range(len(POOLER_KEYS) - 1):
    sep_x = (pos_pr[i] + pos_p0[i + 1]) / 2
    ax.axvline(sep_x, color="#cccccc", linewidth=1, linestyle="--", zorder=0)

ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(-0.055, 0.5, "lower is better",
        transform=ax.transAxes, fontsize=8.5, color="gray",
        rotation=90, ha="right", va="center")

# Legend — upper left, where Mean Pool boxes are small, well away from data
patch_p0 = mpatches.Patch(facecolor=C_P0, alpha=0.55, edgecolor=C_P0,
                           label=r"Base prompt $\mathcal{P}_0$ (5 tokens)")
patch_pr = mpatches.Patch(facecolor=C_PR, alpha=0.55, edgecolor=C_PR,
                           label=r"Rich prompt $\mathcal{P}_r$ (63 tokens)")
ax.legend(handles=[patch_p0, patch_pr],
          loc="upper left", frameon=True, framealpha=0.93,
          fontsize=10, borderpad=0.7)

fig.tight_layout()
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    p = OUTPUT_DIR / f"pooler_prompt_ablation.{ext}"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    print(f"saved → {p}")
plt.close(fig)
