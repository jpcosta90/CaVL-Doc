#!/usr/bin/env python3
"""at_once_figure.pdf — Figure 2: At-Once Aggregation vs. Autoregressive Decoding.

Two-panel comparison:
  (A) Generative LLM  — autoregressive decoding, sequential token output
  (B) ArcDoc encoder  — single forward pass, all tokens at once → Pooler → embedding

Style matches the other paper figures (serif/CM math, muted academic palette,
thin lines, no heavy nested boxes).

Run from repo root:
    python scripts/figures/generate_at_once_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
})

# ── Academic colour palette (consistent with architecture_figure) ─────────────
C_LLM   = {"face": "#E8E0F0", "edge": "#6A58A0"}  # muted lavender
C_ENC   = {"face": "#C8E6D0", "edge": "#3A7A50"}  # sage green (frozen backbone)
C_TOK   = {"face": "#FFF8D6", "edge": "#B89000"}  # pale amber (tokens)
C_POOL  = {"face": "#D3E8F5", "edge": "#4A8AB8"}  # pale blue  (pooler)
C_EMBD  = {"face": "#F5E3CB", "edge": "#A86830"}  # warm peach (embedding)

C_A_BG  = "#FEFAF9"; C_A_BD  = "#C8B0A8"   # Panel A: very pale warm
C_B_BG  = "#F8FBF8"; C_B_BD  = "#90B8A0"   # Panel B: very pale cool
C_ARR   = "#444444"                          # arrow colour

FS      = 10    # base font size

# ── Drawing helpers ───────────────────────────────────────────────────────────

def fbox(ax, x, y, w, h, c, label, sub=None, fs=None, zo=2):
    if fs is None:
        fs = FS
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.07,rounding_size=0.15",
        lw=1.1, edgecolor=c["edge"], facecolor=c["face"], zorder=zo,
    ))
    cy_lbl = y + h / 2 + (0.13 if sub else 0)
    ax.text(x + w / 2, cy_lbl, label,
            ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1a1a1a", zorder=zo + 1)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18, sub,
                ha="center", va="center",
                fontsize=fs - 1.5, style="italic", color="#555555", zorder=zo + 1)


def arr(ax, x0, y0, x1, y1, color=C_ARR, lw=1.2, cs="arc3"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                shrinkA=3, shrinkB=3,
                                connectionstyle=cs),
                zorder=6)


def doc_stack(ax, x, y, w=1.2, h=1.2, label="Input Docs"):
    dw, dh = w * 0.80, h * 0.90
    sx, sy = x + (w - dw) / 2, y + (h - dh) / 2
    for i in range(3):
        o = i * 0.07
        ax.add_patch(Rectangle(
            (sx + o, sy + o), dw, dh,
            lw=0.9, edgecolor="#888888", facecolor="#FAFAFA", zorder=2 + i,
        ))
        for yl in [0.75, 0.55, 0.35, 0.15]:
            ax.plot([sx + o + 0.10, sx + o + dw - 0.10],
                    [sy + o + yl * dh] * 2,
                    color="#CCCCCC", lw=0.9, zorder=3 + i)
    ax.text(x + w / 2, y - 0.22, label,
            ha="center", fontsize=FS - 0.5, color="#555555")


# ── Canvas ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8))
plt.subplots_adjust(hspace=0.38, top=0.96, bottom=0.04, left=0.02, right=0.98)

for ax in (ax1, ax2):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

# ── Shared geometry ────────────────────────────────────────────────────────────
X_START = 1.5; W_DOC = 1.2; GAP_1 = 1.0; W_MODEL = 2.8; GAP_2 = 0.8
X_OUT   = X_START + W_DOC + GAP_1 + W_MODEL + GAP_2   # 7.3
YC      = 1.6

x_doc_right   = X_START + W_DOC           # 2.7
x_model_left  = x_doc_right + GAP_1       # 3.7
x_model_right = x_model_left + W_MODEL    # 6.5

TOKS  = [r"$t_1$", r"$t_2$", r"$t_3$", r"$\cdots$", r"$t_n$"]
TOK_W, TOK_H = 1.30, 0.28
TOK_Y = [YC + 0.50 - i * 0.38 for i in range(len(TOKS))]


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL A — Generative / Autoregressive
# ══════════════════════════════════════════════════════════════════════════════
ax1.add_patch(FancyBboxPatch(
    (0.15, 0.15), 11.7, 2.6,
    boxstyle="round,pad=0.08,rounding_size=0.2",
    lw=0.9, edgecolor=C_A_BD, facecolor=C_A_BG, zorder=0,
))
ax1.text(0.38, 3.05, "(A)", fontsize=11, fontweight="bold", color="#552222")
ax1.text(0.72, 3.05,
         r"Generative Model — Autoregressive Decoding",
         fontsize=11, fontweight="bold", color="#333333")

doc_stack(ax1, X_START, YC - 0.6)
arr(ax1, x_doc_right + 0.05, YC, x_model_left - 0.05, YC)
fbox(ax1, x_model_left, YC - 0.70, W_MODEL, 1.40, C_LLM,
     "Multimodal LLM", sub="(autoregressive decoder)")
arr(ax1, x_model_right + 0.05, YC, X_OUT - 0.05, YC)

# Sequential tokens with chain arrows (down)
for i, tok in enumerate(TOKS):
    fbox(ax1, X_OUT, TOK_Y[i], TOK_W, TOK_H, C_TOK, tok, fs=FS)
    if i < len(TOKS) - 1:
        arr(ax1, X_OUT + TOK_W / 2, TOK_Y[i],
                 X_OUT + TOK_W / 2, TOK_Y[i + 1] + TOK_H, lw=1.0)

ax1.text(X_OUT + TOK_W + 0.28, YC,
         "one token\nat a time",
         ha="left", va="center", fontsize=9, style="italic", color="#774444")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL B — ArcDoc: At-Once Aggregation
# ══════════════════════════════════════════════════════════════════════════════
ax2.add_patch(FancyBboxPatch(
    (0.15, 0.15), 11.7, 2.6,
    boxstyle="round,pad=0.08,rounding_size=0.2",
    lw=0.9, edgecolor=C_B_BD, facecolor=C_B_BG, zorder=0,
))
ax2.text(0.38, 3.05, "(B)", fontsize=11, fontweight="bold", color="#224422")
ax2.text(0.72, 3.05,
         r"ArcDoc — At-Once Aggregation",
         fontsize=11, fontweight="bold", color="#333333")

doc_stack(ax2, X_START, YC - 0.6)
arr(ax2, x_doc_right + 0.05, YC, x_model_left - 0.05, YC)
fbox(ax2, x_model_left, YC - 0.70, W_MODEL, 1.40, C_ENC,
     "LVLM Encoder", sub="(single forward pass, frozen)")
arr(ax2, x_model_right + 0.05, YC, X_OUT - 0.12, YC)

# "At-once" dashed group box
PAD    = 0.12
BOX_X  = X_OUT - PAD
BOX_Y  = TOK_Y[-1] - PAD
BOX_W  = TOK_W + 2 * PAD
BOX_H  = (TOK_Y[0] + TOK_H + PAD) - BOX_Y
BOX_CX = BOX_X + BOX_W / 2
BOX_RIGHT = BOX_X + BOX_W

ax2.add_patch(FancyBboxPatch(
    (BOX_X, BOX_Y), BOX_W, BOX_H,
    boxstyle="round,pad=0.06,rounding_size=0.12",
    lw=1.2, linestyle="--",
    edgecolor=C_ENC["edge"], facecolor="#EEF8EE", zorder=1,
))

# Tokens inside box (all at once — no chain arrows)
for tok in zip(TOKS, TOK_Y):
    fbox(ax2, X_OUT, tok[1], TOK_W, TOK_H, C_TOK, tok[0], fs=FS, zo=4)

# Label above group box
ax2.text(BOX_CX, BOX_Y + BOX_H + 0.06,
         r"$\leftarrow$  simultaneously  $\rightarrow$",
         ha="center", va="bottom", fontsize=8.5,
         style="italic", color=C_ENC["edge"], zorder=5)

# Arrow: group box → Pooler
POOL_X, POOL_W, POOL_H = BOX_RIGHT + 0.35, 1.10, 0.70
arr(ax2, BOX_RIGHT, YC, POOL_X - 0.05, YC)
fbox(ax2, POOL_X, YC - POOL_H / 2, POOL_W, POOL_H, C_POOL, "Pooler")

# Arrow: Pooler → embedding
EMBD_X, EMBD_W, EMBD_H = POOL_X + POOL_W + 0.30, 0.80, 0.55
arr(ax2, POOL_X + POOL_W, YC, EMBD_X - 0.05, YC)
fbox(ax2, EMBD_X, YC - EMBD_H / 2, EMBD_W, EMBD_H, C_EMBD, r"$e$", fs=12)

ax2.text(EMBD_X + EMBD_W / 2, YC - EMBD_H / 2 - 0.20,
         r"metric embedding  $\in \mathbb{R}^{d}$",
         ha="center", va="top", fontsize=8.5, color="#555555")


# ── Save ─────────────────────────────────────────────────────────────────────
fig.tight_layout()
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    p = OUTPUT_DIR / f"at_once_figure.{ext}"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    print(f"saved → {p}")
plt.close(fig)
