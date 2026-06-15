#!/usr/bin/env python3
"""
Generates architecture_with_teacher.pdf/png for the ArcDoc paper (Figure 1).

Final ArcDoc configuration:
  - Rich prompt P_r conditioning the frozen LVLM backbone
  - Bidirectional Cross-Modal Attention pooler
  - Residual MLP projection head
  - Sub-Center CosFace loss  (k=3 sub-centres)
  - RL Teacher agent (Phase 2, optional)

Adapted from analysis/paper_figures.ipynb (Cell 4).

Run from repo root:
    python scripts/figures/generate_architecture_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import os

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

# ── Colours ──────────────────────────────────────────────────────────────────
C_TOK    = "#cce5ff"
C_BACK   = "#a3c2e0"
C_POOL   = "#C8E6C9";  C_POOL2  = "#A5D6A7"
C_PROJ   = "#ffdbcc";  C_PROJ2  = "#ffbf80"
C_LOSS   = "#F8D7DA";  C_LOSS2  = "#f5b7b1"
C_TEACH  = "#ffcccc";  C_POLICY = "#ff9999"
C_PROM   = "#FFF3CD"
C_TRAIN  = "#fdfdfd"


def draw_doc_icon(ax, xy, width=1.0, height=1.3, label_top="", label_bot=""):
    doc = FancyBboxPatch(
        (xy[0] - width / 2, xy[1] - height / 2), width, height,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        ec="black", fc="#f8f9fa", lw=1.5, zorder=10,
    )
    ax.add_patch(doc)
    line_w = width * 0.6
    start_y = xy[1] + height * 0.25
    for i in range(3):
        ly = start_y - i * height * 0.2
        lw_i = line_w * 0.7 if i == 0 else line_w
        ax.plot([xy[0] - lw_i / 2, xy[0] + lw_i / 2], [ly, ly],
                color="#a0a0a0", lw=1.5, zorder=11)
    if label_top:
        ax.text(xy[0], xy[1] + height / 2 + 0.25, label_top,
                ha="center", va="bottom", fontweight="bold", fontsize=12)
    if label_bot:
        ax.text(xy[0], xy[1] - height / 2 - 0.25, label_bot,
                ha="center", va="top", fontsize=11, style="italic")


def draw_functional_block(ax, xy, w, h, title, color, sub_text=None, sub_color="white"):
    box = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        ec="black", fc=color, lw=1.5, zorder=15,
    )
    ax.add_patch(box)
    y_off = h / 4 if sub_text else 0
    ax.text(xy[0], xy[1] + y_off, title,
            ha="center", va="center", fontweight="bold", fontsize=12, zorder=16)
    if sub_text:
        sh, sw = h * 0.4, w * 0.85
        sub_box = FancyBboxPatch(
            (xy[0] - sw / 2, xy[1] - h / 3.5 - sh / 2), sw, sh,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            ec="black", fc=sub_color, lw=1, zorder=16,
        )
        ax.add_patch(sub_box)
        ax.text(xy[0], xy[1] - h / 3.5, sub_text,
                ha="center", va="center", fontsize=10, fontweight="bold", zorder=17)


def draw_arrow(ax, p1, p2, style="->", lw=1.5, connectionstyle="arc3", color="black"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=5, shrinkB=5,
                                connectionstyle=connectionstyle),
                zorder=5)


def plot_architecture_with_teacher():
    fig, ax = plt.subplots(figsize=(22, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11.5)
    ax.axis("off")

    # ── Layout ────────────────────────────────────────────────────────────────
    y_top, y_bot, y_mid = 6.0, 2.0, 4.0
    y_teacher = 10.3
    y_prom    = 8.4   # rich prompt badge — above backbone, below teacher

    x_doc     = 1.0
    x_tok     = 3.2
    x_back    = 7.5
    x_pool    = 11.0
    x_proj    = 14.2
    x_loss    = 17.8
    x_teacher = 11.5  # centred above pipeline

    # ── Trainable region (pooler + head) ──────────────────────────────────────
    tr_x0 = x_back + 1.2
    tr_x1 = x_proj + 1.5
    tr_y0 = y_bot - 1.8
    tr_y1 = y_top + 2.2
    train_frame = FancyBboxPatch(
        (tr_x0, tr_y0), tr_x1 - tr_x0, tr_y1 - tr_y0,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        ec="#555555", fc=C_TRAIN, lw=2, ls="--", zorder=1,
    )
    ax.add_patch(train_frame)
    ax.text((tr_x0 + tr_x1) / 2, tr_y1 - 0.3, "Trainable Architecture",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#444444", zorder=2)

    # ── Teacher block ──────────────────────────────────────────────────────────
    draw_functional_block(
        ax, (x_teacher, y_teacher), 4.2, 1.8,
        'The "Teacher"\n(RL Agent)', C_TEACH,
        "Policy π(Action|Loss)", C_POLICY,
    )

    # ── Document icons ─────────────────────────────────────────────────────────
    draw_doc_icon(ax, (x_doc, y_top), label_top="Image A", label_bot="(Support)")
    draw_doc_icon(ax, (x_doc, y_bot), label_top="Image B", label_bot="(Query)")

    # ── Tokens blocks ──────────────────────────────────────────────────────────
    draw_functional_block(ax, (x_tok, y_top), 1.7, 1.2, "Tokens\n(Image Patches)", C_TOK)
    draw_functional_block(ax, (x_tok, y_bot), 1.7, 1.2, "Tokens\n(Image Patches)", C_TOK)

    # ── Rich Prompt P_r badge (shared, enters backbone from above) ────────────
    badge = FancyBboxPatch(
        (x_back - 1.5, y_prom - 0.55), 3.0, 1.1,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        ec="#BB8800", fc=C_PROM, lw=1.8, zorder=15,
    )
    ax.add_patch(badge)
    ax.text(x_back, y_prom + 0.15, r"Rich Prompt  $\mathcal{P}_r$",
            ha="center", va="center", fontsize=13, color="#885500",
            fontweight="bold", zorder=16)
    ax.text(x_back, y_prom - 0.22, "(63 tokens — conditions backbone attention)",
            ha="center", va="center", fontsize=9, color="#885500",
            style="italic", zorder=16)

    # ── Frozen backbone (spans both streams) ───────────────────────────────────
    back_h = y_top - y_bot + 2.0  # referenced also in Prompt arrow below
    draw_functional_block(ax, (x_back, y_mid), 2.8, back_h, "LVLM Backbone", C_BACK)
    ax.text(x_back, y_mid - back_h / 2 + 0.5, "(frozen)",
            ha="center", va="center", fontsize=10, color="#336699",
            style="italic", zorder=17)

    # ── Cross-Modal Attention pooler ───────────────────────────────────────────
    draw_functional_block(
        ax, (x_pool, y_top), 2.5, 1.6,
        "Cross-Modal\nAttn Pooler", C_POOL,
        sub_text="qv  x  qt  .  alpha", sub_color=C_POOL2,
    )
    draw_functional_block(
        ax, (x_pool, y_bot), 2.5, 1.6,
        "Cross-Modal\nAttn Pooler", C_POOL,
        sub_text="qv  x  qt  .  alpha", sub_color=C_POOL2,
    )

    # ── Projection head ────────────────────────────────────────────────────────
    draw_functional_block(
        ax, (x_proj, y_top), 2.5, 1.6,
        "Projection\nHead", C_PROJ,
        sub_text="MLP  (D->d, L2)", sub_color=C_PROJ2,
    )
    draw_functional_block(
        ax, (x_proj, y_bot), 2.5, 1.6,
        "Projection\nHead", C_PROJ,
        sub_text="MLP  (D->d, L2)", sub_color=C_PROJ2,
    )

    # ── Sub-Center CosFace loss ────────────────────────────────────────────────
    draw_functional_block(
        ax, (x_loss, y_mid), 2.5, 2.0,
        "Sub-Center\nCosFace", C_LOSS,
        sub_text="k=3 sub-centres", sub_color=C_LOSS2,
    )

    # ── Data-flow arrows ───────────────────────────────────────────────────────
    # Doc → Tokens
    for y_s in (y_top, y_bot):
        draw_arrow(ax, (x_doc + 0.6, y_s), (x_tok - 0.85, y_s))

    # Tokens → Backbone (direct)
    for y_s in (y_top, y_bot):
        draw_arrow(ax, (x_tok + 0.85, y_s), (x_back - 1.4, y_s))

    # Prompt → Backbone (from above)
    draw_arrow(ax, (x_back, y_prom - 0.55), (x_back, y_mid + back_h / 2))

    # Backbone → Pooler
    draw_arrow(ax, (x_back + 1.4, y_top), (x_pool - 1.25, y_top))
    draw_arrow(ax, (x_back + 1.4, y_bot), (x_pool - 1.25, y_bot))

    # Pooler → Head
    draw_arrow(ax, (x_pool + 1.25, y_top), (x_proj - 1.25, y_top))
    draw_arrow(ax, (x_pool + 1.25, y_bot), (x_proj - 1.25, y_bot))

    # Head → Loss (converging)
    draw_arrow(ax, (x_proj + 1.25, y_top), (x_loss - 1.25, y_mid + 0.3),
               connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, (x_proj + 1.25, y_bot), (x_loss - 1.25, y_mid - 0.3),
               connectionstyle="arc3,rad=-0.1")

    # ── Teacher connections ────────────────────────────────────────────────────
    # Loss → Teacher (state/reward) — right-angle path
    draw_arrow(ax, (x_loss, y_mid + 1.1),
               (x_teacher + 2.1, y_teacher),
               connectionstyle="angle,angleA=90,angleB=0")
    ax.text(16.5, 9.55, "Loss Feedback\n(State/Reward)",
            ha="center", fontsize=11, fontweight="bold", color="#555555")

    # Teacher → Input (hard negative selection) — right-angle path
    draw_arrow(ax, (x_teacher - 2.1, y_teacher),
               (x_doc, y_top + 0.8),
               connectionstyle="angle,angleA=180,angleB=90")
    ax.text(4.2, 9.55, "Hard Negative\nSelection",
            ha="center", fontsize=11, fontweight="bold", color="#555555")

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    fname = "architecture_with_teacher"
    for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
        p = OUTPUT_DIR / f"{fname}.{ext}"
        plt.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
        print(f"saved → {p}")
    plt.close()


if __name__ == "__main__":
    plot_architecture_with_teacher()
