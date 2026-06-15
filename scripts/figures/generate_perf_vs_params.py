#!/usr/bin/env python3
"""
Generates perf_vs_params.pdf/png for the ArcDoc paper.

Scatter plot: Average EER (%) vs. Active Parameters (B) on LA-CDIP.
Shows the efficiency frontier — ArcDoc (2B) beats zero-shot VLMs up to 14B.
Both axes are log-scaled to handle the 0.63%–30% EER range.

Jina-v4: 4B parameters (unadapted and LoRA fine-tuned).
Pixel baselines excluded (no learnable parameters).

Run from repo root:
    python scripts/figures/generate_perf_vs_params.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8,
    "mathtext.fontset": "cm",
})

# (params_B, avg_eer, display_label, family, (dx_pts, dy_pts), in_trend)
# in_trend: whether to include in the VLM scaling-trend fit
# InternVL-3-2B excluded from trend — ~30% EER reflects prompting failure, not scale.
methods = [
    # Ours
    (2.0,  0.91,  "ArcDoc (ours)",         "ours",   ( 8,  10), False),
    # Zero-shot VLM
    (8.0,  1.12,  "Qwen3-VL-8B",          "vlm",    ( 6,   8), True),
    (14.0, 1.30,  "InternVL-3-14B",       "vlm",    (-6,  -8), True),
    (8.0,  2.13,  "InternVL-3-8B",        "vlm",    ( 6,  -8), True),
    (4.0,  2.67,  "Qwen3-VL-4B",          "vlm",    ( 6,  -8), True),
    (2.0,  29.99, "InternVL-3-2B",        "vlm",    ( 6,   6), False),
    # Fine-tuned embedding
    (4.0,  2.23,  "Jina-v4 (LoRA)",       "ft_emb", ( 6,  12), False),
    # Embedding baselines (unadapted)
    (4.0,  3.08,  "Jina-v4",              "emb",    (-6, -10), False),
    (2.0,  4.78,  "InternVL3 (out)",      "emb",    ( 6,  -6), False),
    (2.0,  7.62,  "InternVL3 (in)",       "emb",    ( 6,   6), False),
]

family_style = {
    "ours":   dict(color="#54A24B", marker="*", s=320, zorder=5),
    "vlm":    dict(color="#4C78A8", marker="o", s=90,  zorder=4),
    "ft_emb": dict(color="#E45756", marker="D", s=80,  zorder=4),
    "emb":    dict(color="#F58518", marker="s", s=90,  zorder=4),
}

legend_labels = {
    "ours":   "ArcDoc (ours)",
    "vlm":    "Zero-shot VLM",
    "ft_emb": "Fine-tuned embedding",
    "emb":    "Embedding (unadapted)",
}

fig, ax = plt.subplots(figsize=(4.0, 3.5))

legend_done = set()
for params, eer, label, fam, (dx, dy), _ in methods:
    sty = {k: v for k, v in family_style[fam].items()}
    s   = sty.pop("s")
    lbl = legend_labels[fam] if fam not in legend_done else None
    ax.scatter(params, eer, s=s, label=lbl, **sty)
    if lbl:
        legend_done.add(fam)
    ha = "right" if dx < 0 else "left"
    ax.annotate(
        label, xy=(params, eer),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=6.5, ha=ha, va="center",
        color=family_style[fam]["color"],
    )

# VLM scaling trend: log-linear fit through competitive zero-shot VLMs
# (InternVL-3-2B excluded — its ~30% EER reflects prompting failure, not scale)
trend_pts = [(p, e) for p, e, *_, it in methods if it]
vlm_log_p = np.log([p for p, e in trend_pts])
vlm_log_e = np.log([e for p, e in trend_pts])
coeffs    = np.polyfit(vlm_log_p, vlm_log_e, 1)
trend_x   = np.linspace(1.2, 16, 100)
trend_y   = np.exp(np.polyval(coeffs, np.log(trend_x)))
ax.plot(trend_x, trend_y, "--", color="#4C78A8", linewidth=0.9,
        alpha=0.45, zorder=2, label="VLM scaling trend")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Active Parameters (B)")
ax.set_ylabel("Average EER (%)")

ax.set_xticks([2, 4, 8, 14])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_yticks([1, 2, 5, 10, 30])
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

ax.set_xlim(1.2, 20)
ax.set_ylim(0.50, 60)

ax.text(0.01, 0.01, "lower-left is better",
        transform=ax.transAxes, fontsize=7.5, color="#888888",
        ha="left", va="bottom", style="italic")

ax.legend(loc="upper right", frameon=True, framealpha=0.92,
          edgecolor="#cccccc", handlelength=1.4, handletextpad=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.25)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "perf_vs_params.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUTPUT_DIR / "perf_vs_params.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved perf_vs_params → {OUTPUT_DIR}")
