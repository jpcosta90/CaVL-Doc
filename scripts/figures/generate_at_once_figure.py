#!/usr/bin/env python3
"""at_once_figure.pdf — adapted from teaser_fig style.

Two-row horizontal layout (same proportions as teaser_fig):
  (A) Generative LLM  — sequential token output  (chain arrows)
  (B) ArcDoc encoder  — at-once token extraction  (dashed group box)
                        → Pooler → metric embedding

Run from repo root:
    python scripts/figures/generate_at_once_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

plt.rcParams.update({'font.family': 'serif', 'font.size': 14})

OUT = Path(__file__).resolve().parents[2] / 'docs' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
C_BG_A  = '#fff5f5';  C_BD_A  = '#e6b8af'   # Panel A  warm red
C_BG_B  = '#f0f9f0';  C_BD_B  = '#b6d7a8'   # Panel B  cool green
C_LLM   = {'face': '#e1d5e7', 'edge': '#9673a6'}
C_ENC   = {'face': '#d5e8d4', 'edge': '#82b366'}
C_TOK   = {'face': '#fff2cc', 'edge': '#d6b656'}
C_POOL  = {'face': '#dae8fc', 'edge': '#6c8ebf'}
C_EMBD  = {'face': '#ffe6cc', 'edge': '#d79b00'}

FS = 13   # base font size

# ── Helpers (same style as teaser_fig) ────────────────────────────────────
def fbox(ax, x, y, w, h, c, label, sub=None, fs=None, zo=2):
    if fs is None: fs = FS
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.1',
        lw=1.5, edgecolor=c['edge'], facecolor=c['face'], zorder=zo))
    cy = y + h/2 + (0.14 if sub else 0)
    ax.text(x+w/2, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='#333333', zorder=zo+1)
    if sub:
        ax.text(x+w/2, y+h/2-0.20, sub, ha='center', va='center',
                fontsize=fs-2, color='#555555', style='italic', zorder=zo+1)

def arr(ax, x0, y0, x1, y1, color='#555555', lw=1.5):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color))

def doc_stack(ax, x, y, w=1.2, h=1.2, label='Input Docs'):
    dw, dh = w*0.80, h*0.90
    sx, sy = x+(w-dw)/2, y+(h-dh)/2
    for i in range(3):
        o = i*0.08
        ax.add_patch(patches.Rectangle(
            (sx+o, sy+o), dw, dh,
            lw=1, edgecolor='#999', facecolor='white', zorder=2+i))
        for yl in [0.75, 0.55, 0.35, 0.15]:
            ax.plot([sx+o+0.1, sx+o+dw-0.1],
                    [sy+o+yl*dh]*2, color='#ccc', lw=1, zorder=3+i)
    ax.text(x+w/2, y-0.25, label, ha='center', fontsize=FS-1, color='#555')

# ── Canvas — same proportions as teaser_fig ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
plt.subplots_adjust(hspace=0.40, top=0.95, bottom=0.05, left=0.02, right=0.98)

for ax in (ax1, ax2):
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis('off')

# ── Shared x/y constants ───────────────────────────────────────────────────
X_START = 1.5; W_DOC = 1.2; GAP_1 = 1.0; W_MODEL = 2.8; GAP_2 = 0.8
X_OUT   = X_START + W_DOC + GAP_1 + W_MODEL + GAP_2   # 7.3
YC      = 1.6

# x helpers
x_doc_right  = X_START + W_DOC                          # 2.7
x_model_left = x_doc_right + GAP_1                      # 3.7
x_model_right= x_model_left + W_MODEL                   # 6.5

# Token column (shared geometry: 5 tokens, same y for both panels)
TOKS   = [r'$t_1$', r'$t_2$', r'$t_3$', r'$\cdots$', r'$t_n$']
TOK_W, TOK_H = 1.30, 0.28
# y-bottom of each token, evenly spaced, centred on YC
TOK_Y  = [YC + 0.50 - i * 0.38 for i in range(len(TOKS))]
# → [2.10, 1.72, 1.34, 0.96, 0.58]


# ══════════════════════════════════════════════════════════════════════════
#  PANEL A — Generative / Sequential
# ══════════════════════════════════════════════════════════════════════════
ax1.add_patch(patches.FancyBboxPatch(
    (0.2, 0.2), 11.6, 2.5, boxstyle='round,pad=0.1',
    lw=1.5, edgecolor=C_BD_A, facecolor=C_BG_A, zorder=0))
ax1.set_title('(A)  Generative Model  —  Autoregressive Decoding',
              loc='left', fontsize=15, fontweight='bold', pad=5)

doc_stack(ax1, X_START, YC-0.6)
arr(ax1, x_doc_right+0.05, YC, x_model_left-0.05, YC)
fbox(ax1, x_model_left, YC-0.70, W_MODEL, 1.40, C_LLM,
     'Multimodal LLM', '(autoregressive decoder)', fs=13)
arr(ax1, x_model_right+0.05, YC, X_OUT-0.05, YC)

# Sequential tokens with DOWN chain arrows
for i, tok in enumerate(TOKS):
    fbox(ax1, X_OUT, TOK_Y[i], TOK_W, TOK_H, C_TOK, tok, fs=12)
    if i < len(TOKS) - 1:
        arr(ax1, X_OUT + TOK_W/2, TOK_Y[i],
                 X_OUT + TOK_W/2, TOK_Y[i+1] + TOK_H)

ax1.text(X_OUT + TOK_W + 0.30, YC,
         'One token\nat a time',
         ha='left', va='center', fontsize=FS-1, color='#666666')


# ══════════════════════════════════════════════════════════════════════════
#  PANEL B — ArcDoc: At-Once Aggregation
# ══════════════════════════════════════════════════════════════════════════
ax2.add_patch(patches.FancyBboxPatch(
    (0.2, 0.2), 11.6, 2.5, boxstyle='round,pad=0.1',
    lw=1.5, edgecolor=C_BD_B, facecolor=C_BG_B, zorder=0))
ax2.set_title('(B)  ArcDoc  —  At-Once Aggregation',
              loc='left', fontsize=15, fontweight='bold', pad=5)

doc_stack(ax2, X_START, YC-0.6)
arr(ax2, x_doc_right+0.05, YC, x_model_left-0.05, YC)
fbox(ax2, x_model_left, YC-0.70, W_MODEL, 1.40, C_ENC,
     'LVLM Encoder', '(single forward pass)', fs=13)

# "At-once" dashed group box (replaces the Metric Proximity circle)
PAD    = 0.12
BOX_X  = X_OUT - PAD
BOX_Y  = TOK_Y[-1] - PAD                               # ≈ 0.46
BOX_W  = TOK_W + 2*PAD                                 # ≈ 1.54
BOX_H  = (TOK_Y[0] + TOK_H + PAD) - BOX_Y             # ≈ 2.04
BOX_CX = BOX_X + BOX_W / 2
BOX_RIGHT = BOX_X + BOX_W                              # ≈ 8.72

# Arrow: encoder → group box left edge
arr(ax2, x_model_right+0.05, YC, BOX_X-0.05, YC)

# Dashed bounding box
ax2.add_patch(patches.FancyBboxPatch(
    (BOX_X, BOX_Y), BOX_W, BOX_H,
    boxstyle='round,pad=0.08', lw=1.5, linestyle='--',
    edgecolor='#5a9e57', facecolor='#edf7ed', zorder=1))

# Tokens inside box (no chain arrows)
for i, tok in enumerate(TOKS):
    fbox(ax2, X_OUT, TOK_Y[i], TOK_W, TOK_H, C_TOK, tok, fs=12, zo=4)

# "← simultaneously →" label just above the group box
ax2.text(BOX_CX, BOX_Y + BOX_H + 0.04,
         r'$\leftarrow$ simultaneously $\rightarrow$',
         ha='center', va='bottom', fontsize=9,
         color='#3a6e39', style='italic', zorder=5)

# Arrow: group box → Pooler
POOL_X, POOL_W, POOL_H = BOX_RIGHT + 0.35, 1.10, 0.70
arr(ax2, BOX_RIGHT, YC, POOL_X - 0.05, YC)
fbox(ax2, POOL_X, YC - POOL_H/2, POOL_W, POOL_H, C_POOL, 'Pooler', fs=13)

# Arrow: Pooler → embedding
EMBD_X, EMBD_W, EMBD_H = POOL_X + POOL_W + 0.30, 0.80, 0.55
arr(ax2, POOL_X + POOL_W, YC, EMBD_X - 0.05, YC)
fbox(ax2, EMBD_X, YC - EMBD_H/2, EMBD_W, EMBD_H, C_EMBD, r'$e$', fs=14)

ax2.text(EMBD_X + EMBD_W/2, YC - EMBD_H/2 - 0.22,
         r'metric embedding  $\in\mathbb{R}^{d}$',
         ha='center', va='top', fontsize=10, color='#666666')


# ── Save ───────────────────────────────────────────────────────────────────
for ext, kw in [('pdf', {}), ('png', {'dpi': 300})]:
    p = OUT / f'at_once_figure.{ext}'
    fig.savefig(p, bbox_inches='tight', pad_inches=0.05, **kw)
    print(f'saved → {p}')
