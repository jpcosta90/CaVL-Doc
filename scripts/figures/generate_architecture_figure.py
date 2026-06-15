#!/usr/bin/env python3
"""
Generates architecture_with_teacher.pdf/png for the ArcDoc paper (Figure 1).

Final ArcDoc configuration:
  - Rich prompt P_r conditioning the frozen LVLM backbone (from above)
  - Bidirectional Cross-Modal Attention pooler
  - Residual MLP projection head
  - Sub-Center CosFace loss (k=3 sub-centres)
  - RL Teacher agent (Phase 2, optional)

Style mirrors the other paper figures: serif/CM math, muted academic palette,
thin coordinated borders, no nested sub-boxes.

Adapted from analysis/paper_figures.ipynb (Cell 4).

Run from repo root:
    python scripts/figures/generate_architecture_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
})

# ── Academic colour palette ───────────────────────────────────────────────────
C_TOK    = "#D3E8F5"; B_TOK    = "#4A8AB8"   # pale / border blue
C_BACK   = "#B8D4E8"; B_BACK   = "#5B8DB8"   # muted light blue
C_POOL   = "#C8E6D0"; B_POOL   = "#3A7A50"   # sage green
C_PROJ   = "#F5E3CB"; B_PROJ   = "#A86830"   # warm peach
C_LOSS   = "#F5D3CC"; B_LOSS   = "#A03828"   # coral
C_TEACH  = "#EDE0F5"; B_TEACH  = "#7040A8"   # lavender
C_PROM   = "#FFF8D6"; B_PROM   = "#B89000"   # amber

C_ARROW  = "#404040"  # arrow colour
C_TRAIN  = "#F5F5F5"  # trainable-region fill

# ── Drawing helpers ───────────────────────────────────────────────────────────

def block(ax, cx, cy, w, h,
          title, fill, border,
          subtitle=None,
          lw=1.2, ls="-",
          title_color="#1a1a1a", sub_color="#555555",
          title_fs=10.5, sub_fs=8.5,
          text_color=None):
    """Single-layer rounded block — title + optional italic subtitle."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.07,rounding_size=0.18",
        facecolor=fill, edgecolor=border, linewidth=lw, linestyle=ls, zorder=15,
    )
    ax.add_patch(rect)
    tc = text_color or title_color
    y_off = 0.18 if subtitle else 0
    ax.text(cx, cy + y_off, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=tc, zorder=16)
    if subtitle:
        ax.text(cx, cy - 0.22, subtitle,
                ha="center", va="center",
                fontsize=sub_fs, style="italic", color=sub_color, zorder=16)


def doc_icon(ax, cx, cy, label_top="", label_bot="", w=0.95, h=1.25):
    """Vectorial document icon (white card with ruled lines)."""
    card = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#F9F9F9", edgecolor="#555555", linewidth=1.1, zorder=10,
    )
    ax.add_patch(card)
    ly0 = cy + h * 0.22
    for i in range(3):
        lw_i = w * 0.45 if i == 0 else w * 0.58
        ax.plot([cx - lw_i / 2, cx + lw_i / 2],
                [ly0 - i * h * 0.18, ly0 - i * h * 0.18],
                color="#AAAAAA", lw=1.3, zorder=11)
    if label_top:
        ax.text(cx, cy + h / 2 + 0.20, label_top,
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    if label_bot:
        ax.text(cx, cy - h / 2 - 0.20, label_bot,
                ha="center", va="top", fontsize=9, style="italic", color="#555")


def arrow(ax, x0, y0, x1, y1, cs="arc3", lw=1.3, color=C_ARROW,
          style="->", shrink=5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=shrink, shrinkB=shrink,
                                connectionstyle=cs),
                zorder=6)


# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11.5)
ax.axis("off")

# ── Coordinate grid ───────────────────────────────────────────────────────────
y_top, y_bot, y_mid = 6.0, 2.0, 4.0
y_teacher = 10.3
y_prom    = 8.5   # rich prompt — above backbone, below teacher

x_doc  = 1.1
x_tok  = 3.2
x_back = 7.5
x_pool = 11.0
x_proj = 14.4
x_loss = 17.8
x_tchr = 11.5   # teacher centred above pipeline

back_h = y_top - y_bot + 2.2   # height of backbone block

# ── Trainable region (subtle dotted outline) ──────────────────────────────────
tr_x0 = x_back + 1.2
tr_x1 = x_proj + 1.55
tr_rect = FancyBboxPatch(
    (tr_x0, y_bot - 1.5), tr_x1 - tr_x0, y_top + 1.5 - (y_bot - 1.5),
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=C_TRAIN, edgecolor="#999999", linewidth=1.0, linestyle=":",
    zorder=1,
)
ax.add_patch(tr_rect)
ax.text((tr_x0 + tr_x1) / 2, y_top + 1.25, "Trainable Architecture",
        ha="center", va="center",
        fontsize=9.5, color="#666666", style="italic", zorder=2)

# ── Teacher block ─────────────────────────────────────────────────────────────
block(ax, x_tchr, y_teacher, 5.0, 1.5,
      r'The "Teacher"  (RL Agent)',
      subtitle=r"Policy $\pi(\text{Action}|\text{Loss})$",
      fill=C_TEACH, border=B_TEACH,
      lw=1.4, ls="--",
      title_fs=11, sub_fs=9)

# ── Document icons ─────────────────────────────────────────────────────────────
doc_icon(ax, x_doc, y_top, label_top="Image A", label_bot="(Support)")
doc_icon(ax, x_doc, y_bot, label_top="Image B", label_bot="(Query)")

# ── Tokens blocks ─────────────────────────────────────────────────────────────
block(ax, x_tok, y_top, 1.8, 1.1,
      "Tokens", C_TOK, B_TOK,
      subtitle="(Image Patches)", sub_fs=8)
block(ax, x_tok, y_bot, 1.8, 1.1,
      "Tokens", C_TOK, B_TOK,
      subtitle="(Image Patches)", sub_fs=8)

# ── Rich Prompt badge (enters backbone from above) ────────────────────────────
prom_rect = FancyBboxPatch(
    (x_back - 1.65, y_prom - 0.55), 3.3, 1.1,
    boxstyle="round,pad=0.08,rounding_size=0.18",
    facecolor=C_PROM, edgecolor=B_PROM, linewidth=1.4, linestyle="--", zorder=15,
)
ax.add_patch(prom_rect)
ax.text(x_back, y_prom + 0.12,
        r"Rich Prompt $\mathcal{P}_r$",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="#7A6000", zorder=16)
ax.text(x_back, y_prom - 0.22,
        r"63 tokens — conditions backbone attention",
        ha="center", va="center",
        fontsize=8, style="italic", color="#9A8020", zorder=16)

# ── Frozen backbone (spans both streams) ──────────────────────────────────────
block(ax, x_back, y_mid, 3.0, back_h,
      "LVLM Backbone\n(InternVL3-2B)", C_BACK, B_BACK,
      subtitle="frozen — no gradient",
      lw=1.5, title_fs=11, sub_fs=8.5)

# ── Cross-Modal Attention pooler ───────────────────────────────────────────────
for y_s in (y_top, y_bot):
    block(ax, x_pool, y_s, 2.6, 1.55,
          "Cross-Modal\nAttn Pooler", C_POOL, B_POOL,
          subtitle=r"$q_V \!\times\! q_T \cdot \alpha$",
          title_fs=10, sub_fs=9)

# ── Projection head ────────────────────────────────────────────────────────────
for y_s in (y_top, y_bot):
    block(ax, x_proj, y_s, 2.6, 1.55,
          "Projection Head", C_PROJ, B_PROJ,
          subtitle=r"MLP $(D{\to}d)$,  $\ell_2$-norm",
          title_fs=10, sub_fs=9)

# ── Sub-Center CosFace loss ────────────────────────────────────────────────────
block(ax, x_loss, y_mid, 2.6, 2.1,
      "Sub-Center\nCosFace",
      C_LOSS, B_LOSS,
      subtitle=r"$k{=}3$ sub-centres",
      lw=1.5, title_fs=11, sub_fs=9)

# ── Arrows: main data flow ─────────────────────────────────────────────────────
# Doc → Tokens
for y_s in (y_top, y_bot):
    arrow(ax, x_doc + 0.55, y_s, x_tok - 0.9, y_s)

# Tokens → Backbone
for y_s in (y_top, y_bot):
    arrow(ax, x_tok + 0.9, y_s, x_back - 1.5, y_s)

# Prompt → Backbone top (dashed amber arrow)
arrow(ax, x_back, y_prom - 0.55, x_back, y_mid + back_h / 2,
      color=B_PROM, lw=1.2)

# Backbone → Pooler
arrow(ax, x_back + 1.5, y_top, x_pool - 1.3, y_top)
arrow(ax, x_back + 1.5, y_bot, x_pool - 1.3, y_bot)

# Pooler → Head
arrow(ax, x_pool + 1.3, y_top, x_proj - 1.3, y_top)
arrow(ax, x_pool + 1.3, y_bot, x_proj - 1.3, y_bot)

# Head → Loss (converging arcs)
arrow(ax, x_proj + 1.3, y_top, x_loss - 1.3, y_mid + 0.4,
      cs="arc3,rad=0.12")
arrow(ax, x_proj + 1.3, y_bot, x_loss - 1.3, y_mid - 0.4,
      cs="arc3,rad=-0.12")

# ── Arrows: Teacher loop ────────────────────────────────────────────────────────
# Loss → Teacher (state/reward), right-angle
arrow(ax, x_loss, y_mid + 1.15, x_tchr + 2.5, y_teacher,
      cs="angle,angleA=90,angleB=0", color=B_TEACH, lw=1.1)
ax.text(16.8, 9.8, "Loss Feedback\n(State / Reward)",
        ha="center", fontsize=9, color="#5A3A88")

# Teacher → Doc input (hard negative selection), right-angle
arrow(ax, x_tchr - 2.5, y_teacher, x_doc, y_top + 0.75,
      cs="angle,angleA=180,angleB=90", color=B_TEACH, lw=1.1)
ax.text(4.0, 9.8, "Hard Negative\nSelection",
        ha="center", fontsize=9, color="#5A3A88")

# ── Save ────────────────────────────────────────────────────────────────────────
fig.tight_layout()
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    p = OUTPUT_DIR / f"architecture_with_teacher.{ext}"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.1, **kw)
    print(f"saved → {p}")
plt.close()
