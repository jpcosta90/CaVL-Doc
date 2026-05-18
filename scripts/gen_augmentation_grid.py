"""
Gera painel 2×2 com 4 augmentações selecionadas (par original→aumentado).
Salva: docs/assets/augmentation_grid.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path

plt.rcParams.update({"font.family": "serif", "font.size": 7.5})

SRC = Path("docs/assets/augmentation_examples")
OUT = Path("docs/assets/augmentation_grid.pdf")

# ── 4 augmentações com maior contraste visual ─────────────────────────────────
SELECTED = [
    ("Rotação",        "Rotação  ±15°"),
    ("Amarelamento",   "Amarelamento (aging)"),
    ("Perspectiva",    "Perspectiva  2–7%"),
    ("Salt & Pepper",  "Salt & Pepper  0.1–0.6%"),
]

orig = Image.open(SRC / "Rotação_orig.jpg")

fig = plt.figure(figsize=(7.5, 5.0))
gs = gridspec.GridSpec(2, 4, figure=fig,
                       hspace=0.08, wspace=0.05,
                       left=0.01, right=0.99,
                       top=0.91, bottom=0.01)

def show(ax, img, label, is_orig=False):
    ax.imshow(np.array(img), aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    c_border = "#1565C0" if is_orig else "#616161"
    lw = 2.2 if is_orig else 1.0
    for sp in ax.spines.values():
        sp.set_linewidth(lw); sp.set_edgecolor(c_border)
    ax.set_title(label, fontsize=7, pad=2, color="white",
                 bbox=dict(facecolor=c_border, edgecolor="none",
                           boxstyle="square,pad=0.22"))

# Cada coluna: original (topo) + aumentada (baixo)
for col, (stem, label) in enumerate(SELECTED):
    fname = SRC / f"{stem}_aug.jpg"
    aug = Image.open(fname)

    ax_o = fig.add_subplot(gs[0, col])
    ax_a = fig.add_subplot(gs[1, col])

    show(ax_o, orig, "Original" if col == 0 else "", is_orig=True)
    show(ax_a, aug, label)

    # Seta entre os dois
    ax_o.annotate("", xy=(0.5, -0.04), xycoords="axes fraction",
                  xytext=(0.5, -0.01), textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="#757575",
                                  lw=0.9, mutation_scale=7))

fig.savefig(OUT, bbox_inches="tight", dpi=180)
print(f"Salvo: {OUT}")
