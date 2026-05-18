"""
Gera visualização das funções de perda para slides do seminário ArcDoc.
Salva: docs/assets/loss_functions_viz.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Estilo geral ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

C_BLUE  = "#1565C0"
C_RED   = "#C62828"
C_GREEN = "#2E7D32"
C_GOLD  = "#F9A825"
C_GRAY  = "#9E9E9E"
C_ORANGE = "#E65100"

fig = plt.figure(figsize=(15, 6.0))
gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.18, left=0.02, right=0.98,
                       top=0.88, bottom=0.14)

axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
titles = ["Contrastive", "Triplet", "CosFace", "ArcFace", "Sub-Center\nArcFace (k=3)"]
for ax, title in zip(axes, titles):
    ax.set_title(title, pad=5)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTRASTIVE LOSS
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.axis("off")

np.random.seed(7)
# Mesmo classe (pares positivos) – dois grupos azuis próximos
pA1 = np.array([1.3, 3.8])
pA2 = np.array([2.0, 3.2])
pB1 = np.array([3.6, 1.4])

ax.scatter(*pA1, c=C_BLUE,  s=70, zorder=5, edgecolors="navy",    lw=0.8)
ax.scatter(*pA2, c=C_BLUE,  s=70, zorder=5, edgecolors="navy",    lw=0.8, marker="s")
ax.scatter(*pB1, c=C_RED,   s=70, zorder=5, edgecolors="darkred", lw=0.8, marker="^")

# Seta attract (par positivo)
ax.annotate("", xy=pA2, xytext=pA1,
            arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.4))
ax.text(1.3, 2.85, "atrair", fontsize=6.5, color=C_BLUE, ha="center")

# Seta repel (par negativo)
ax.annotate("", xy=pB1, xytext=pA2,
            arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.4,
                            connectionstyle="arc3,rad=0.15"))
ax.text(3.3, 2.7, "repelir\nse d < m", fontsize=6, color=C_RED, ha="center")

# Círculo de margem m em torno de pA2
m_c = 1.8
circle = Circle(pA2, m_c, fill=False, color=C_GRAY, ls=":", lw=0.9, alpha=0.65)
ax.add_patch(circle)
ax.text(2.0, 1.1, "m", fontsize=7, color=C_GRAY, ha="center")

ax.text(0.6, 4.6, "Classe A", fontsize=6.5, color=C_BLUE)
ax.text(4.0, 0.8, "Classe B", fontsize=6.5, color=C_RED)

formula = r"$\mathcal{L} = y\,d^2 + (1{-}y)\,[m{-}d]_+^2$"
ax.text(2.5, -0.72, formula, fontsize=11, ha="center", transform=ax.transData,
        color="#333")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRIPLET LOSS
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.axis("off")

anc = np.array([2.5, 2.5])
pos = np.array([3.3, 3.6])
neg = np.array([0.7, 1.1])

d_pos = np.linalg.norm(pos - anc)
d_neg = np.linalg.norm(neg - anc)
margin = 0.55

# Círculos de distância
for r, color, ls in [(d_pos, C_BLUE, "-"), (d_neg - margin, C_ORANGE, "--"), (d_neg, C_RED, "-")]:
    circ = Circle(anc, r, fill=False, color=color, ls=ls, lw=0.9, alpha=0.55)
    ax.add_patch(circ)

# Região de margem (anel entre d_neg-m e d_neg)
theta = np.linspace(0, 2 * np.pi, 300)
r_inner, r_outer = d_neg - margin, d_neg
ax.fill(
    np.concatenate([r_outer * np.cos(theta) + anc[0],
                    r_inner * np.cos(theta[::-1]) + anc[0]]),
    np.concatenate([r_outer * np.sin(theta) + anc[1],
                    r_inner * np.sin(theta[::-1]) + anc[1]]),
    alpha=0.10, color=C_ORANGE, lw=0,
)
ax.text(anc[0] + d_neg * 0.62, anc[1] + 0.1, "m", fontsize=7, color=C_ORANGE)

# Linhas A-P e A-N
ax.plot([anc[0], pos[0]], [anc[1], pos[1]], color=C_BLUE,   lw=1.4)
ax.plot([anc[0], neg[0]], [anc[1], neg[1]], color=C_RED,    lw=1.4, ls="--")

# Pontos
ax.scatter(*anc, c=C_GOLD,  s=140, zorder=6, marker="*", edgecolors=C_ORANGE, lw=0.8)
ax.scatter(*pos, c=C_BLUE,  s=70,  zorder=5, edgecolors="navy",    lw=0.8)
ax.scatter(*neg, c=C_RED,   s=70,  zorder=5, marker="^",  edgecolors="darkred", lw=0.8)

ax.text(anc[0] + 0.15, anc[1] - 0.3, "A", fontsize=8, color=C_ORANGE)
ax.text(pos[0] + 0.12, pos[1] + 0.1, "P", fontsize=8, color=C_BLUE)
ax.text(neg[0] - 0.35, neg[1] - 0.1, "N", fontsize=8, color=C_RED)

formula = r"$\mathcal{L} = [d_+ - d_- + m]_+$"
ax.text(2.5, -0.72, formula, fontsize=11, ha="center", transform=ax.transData, color="#333")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: hipersfera (círculo unitário 2-D)
# ══════════════════════════════════════════════════════════════════════════════
def draw_unit_circle(ax, lim=1.28):
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="#BDBDBD", lw=0.8, zorder=0)
    ax.plot([-lim, lim], [0, 0], color="#E0E0E0", lw=0.5, zorder=0)
    ax.plot([0, 0], [-lim, lim], color="#E0E0E0", lw=0.5, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")

def arrow_from_origin(ax, angle, length, color, lw=1.8, alpha=1.0):
    ax.annotate(
        "", xy=(length * np.cos(angle), length * np.sin(angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        mutation_scale=9, alpha=alpha),
    )

CLASS_ANGLES = [np.radians(25), np.radians(145), np.radians(265)]
CLASS_COLORS = [C_BLUE, C_RED, C_GREEN]

# Ângulos de fronteira de decisão (sem margem) — bissetriz entre classes
BOUNDARIES = [
    (CLASS_ANGLES[0] + CLASS_ANGLES[1]) / 2,
    (CLASS_ANGLES[1] + CLASS_ANGLES[2]) / 2,
    (CLASS_ANGLES[2] + CLASS_ANGLES[0] + 2 * np.pi) / 2,
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. CosFace — margem no cosseno: cos(θ) − m
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
draw_unit_circle(ax)

m_cos = 0.30  # margem no cosseno

for angle, color in zip(CLASS_ANGLES, CLASS_COLORS):
    arrow_from_origin(ax, angle, 0.93, color)

    # Fronteira de decisão sem margem (bissetriz)
    # Fronteira com margem: o ponto onde cos(θ) - m = cos(θ_boundary)
    # Shading: zona proibida angular em torno de cada centro
    half_zone = np.arccos(np.clip(np.cos(angle) - m_cos, -1, 1)) - angle
    w = Wedge((0, 0), 0.91, np.degrees(angle) - abs(np.degrees(half_zone)) * 0.6,
              np.degrees(angle) + abs(np.degrees(half_zone)) * 0.6,
              color=color, alpha=0.18, lw=0, zorder=1)
    ax.add_patch(w)

# Fronteiras de decisão com margem (linhas tracejadas deslocadas)
for b in BOUNDARIES:
    ax.plot([0, 1.15 * np.cos(b)], [0, 1.15 * np.sin(b)],
            "--", color="#9E9E9E", lw=0.75, alpha=0.7, zorder=2)

# Anotação: margem no cosseno
ax.annotate("", xy=(np.cos(CLASS_ANGLES[0]) * 0.72,
                     np.sin(CLASS_ANGLES[0]) * 0.72),
            xytext=(np.cos(CLASS_ANGLES[0] + 0.32) * 0.72,
                    np.sin(CLASS_ANGLES[0] + 0.32) * 0.72),
            arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=0.9))
ax.text(0.45, 0.82, "m", fontsize=7, color=C_BLUE)

formula = r"$s\,(\cos\theta - m)$"
ax.text(0, -1.22, formula, fontsize=11, ha="center", color="#333")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ArcFace — margem no ângulo: cos(θ + m)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[3]
draw_unit_circle(ax)

m_arc = np.radians(30)  # margem angular

for angle, color in zip(CLASS_ANGLES, CLASS_COLORS):
    arrow_from_origin(ax, angle, 0.93, color)

    # Zona de margem angular (wedge)
    w = Wedge((0, 0), 0.91,
              np.degrees(angle) - np.degrees(m_arc),
              np.degrees(angle) + np.degrees(m_arc),
              color=color, alpha=0.18, lw=0, zorder=1)
    ax.add_patch(w)

    # Limites da margem (linhas finas tracejadas)
    for sign in (-1, 1):
        tb = angle + sign * m_arc
        ax.plot([0, 0.88 * np.cos(tb)], [0, 0.88 * np.sin(tb)],
                "--", color=color, lw=0.65, alpha=0.45, zorder=2)

# Anotação: arco de margem m no ângulo
ang = CLASS_ANGLES[0]
r_ann = 0.60
ax.annotate("", xy=(r_ann * np.cos(ang + m_arc), r_ann * np.sin(ang + m_arc)),
            xytext=(r_ann * np.cos(ang), r_ann * np.sin(ang)),
            arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=0.9,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(0.72, 0.60, "m", fontsize=7, color=C_BLUE)

formula = r"$s\,\cos(\theta + m)$"
ax.text(0, -1.22, formula, fontsize=11, ha="center", color="#333")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Sub-Center ArcFace (k=3) — múltiplos centróides por classe
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[4]
draw_unit_circle(ax)

spread = np.radians(22)
CLASS_ANGLES_2 = [np.radians(35), np.radians(195)]

for center, color in zip(CLASS_ANGLES_2, CLASS_COLORS[:2]):
    sub_angles = [center - spread, center, center + spread]

    # Wedge cobrindo os sub-centros (margem total da classe)
    w = Wedge((0, 0), 0.91,
              np.degrees(center - spread - np.radians(14)),
              np.degrees(center + spread + np.radians(14)),
              color=color, alpha=0.13, lw=0, zorder=1)
    ax.add_patch(w)

    # Arco suave conectando os sub-centros
    t_range = np.linspace(sub_angles[0], sub_angles[-1], 60)
    r_arc = 0.80
    ax.plot(r_arc * np.cos(t_range), r_arc * np.sin(t_range),
            "--", color=color, lw=0.9, alpha=0.50, zorder=2)

    # Setas dos sub-centros
    for j, sub_angle in enumerate(sub_angles):
        is_main = (j == 1)
        length = 0.90 if is_main else 0.78
        lw_val = 1.9 if is_main else 1.0
        alpha_val = 1.0 if is_main else 0.55
        arrow_from_origin(ax, sub_angle, length, color, lw=lw_val, alpha=alpha_val)

    # Rótulo k=3
    cx = (np.cos(center) * 1.10)
    cy = (np.sin(center) * 1.10)
    ax.text(cx, cy, "k=3", fontsize=6, color=color, ha="center", va="center")

# Destaque: sub-centros capturam variações de layout (ex: invoice com/sem logo)
ax.text(0, 0.35, "layout\nvariations", fontsize=5.5, ha="center",
        color="#555", style="italic")

formula = r"$\cos\tilde\theta_y = \max_k\,\cos\theta_y^{(k)}$"
ax.text(0, -1.22, formula, fontsize=11, ha="center", color="#333")


# ══════════════════════════════════════════════════════════════════════════════
# Rótulos de grupo
# ══════════════════════════════════════════════════════════════════════════════
# Separador vertical entre os dois grupos
fig.add_artist(plt.Line2D(
    [0.405, 0.405], [0.06, 0.97],
    transform=fig.transFigure, color="#BDBDBD", lw=0.8, ls="--"
))

fig.text(0.21, 0.97, "Baseadas em Distância (espaço Euclidiano)",
         ha="center", va="top", fontsize=11, color="#555",
         style="italic",
         bbox=dict(facecolor="#F5F5F5", edgecolor="none", pad=3))

fig.text(0.70, 0.97, "Margem Angular (hipersfera unitária)",
         ha="center", va="top", fontsize=11, color="#2E7D32",
         style="italic",
         bbox=dict(facecolor="#E8F5E9", edgecolor="none", pad=3))

# ══════════════════════════════════════════════════════════════════════════════
# Salvar
# ══════════════════════════════════════════════════════════════════════════════
out = Path(__file__).parent.parent / "docs" / "assets" / "loss_functions_viz.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Salvo: {out}")
