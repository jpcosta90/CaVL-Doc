"""
Didactic figure (paper-style): metric learning losses in token embedding space.
White background, serif font, subfigure labels, LaTeX-style math.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(42)

# ── Palette (academic / printer-friendly) ────────────────────────────────────
BG     = 'white'
PANEL  = 'white'
GRID   = '#cccccc'
FG     = '#111111'
CLS_C  = ['#d62728', '#1f77b4', '#2ca02c']   # red / blue / green
CLS_N  = ['Classe A', 'Classe B', 'Classe C']
GREEN  = '#2ca02c'
RED    = '#d62728'
ORANGE = '#e07b00'

plt.rcParams.update({
    'font.family'      : 'serif',
    'mathtext.fontset' : 'dejavuserif',
    'font.size'        : 9,
})

# ── Synthetic token embeddings ────────────────────────────────────────────────
def cluster(center, n=35, spread=0.26):
    return center + np.random.randn(n, 3) * spread

def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

CENTERS = np.array([[1.3, 0.4, 0.7],
                    [-0.7, 1.4, 0.5],
                    [0.2, -1.1, 1.4]])

raw_pts  = np.vstack([cluster(c) for c in CENTERS])
lbls     = np.repeat([0, 1, 2], 35)
norm_pts = normalize(raw_pts)
norm_c   = normalize(CENTERS)

# ── Unit-sphere mesh ──────────────────────────────────────────────────────────
u = np.linspace(0, 2 * np.pi, 36)
v = np.linspace(0,     np.pi, 22)
SPH = (np.outer(np.cos(u), np.sin(v)),
       np.outer(np.sin(u), np.sin(v)),
       np.outer(np.ones_like(u), np.cos(v)))

# ── Helpers ───────────────────────────────────────────────────────────────────
VIEW = (22, -55)   # consistent viewpoint for all panels

def style_ax(ax, lim=None):
    ax.view_init(*VIEW)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor(GRID)
    ax.tick_params(colors=FG, labelsize=5.5, pad=1)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.5)
    ax.set_xlabel(r'$h_1$', fontsize=8, labelpad=-2)
    ax.set_ylabel(r'$h_2$', fontsize=8, labelpad=-2)
    ax.set_zlabel(r'$h_3$', fontsize=8, labelpad=-2)
    if lim:
        if isinstance(lim, (list, tuple)):
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[0], lim[1])
            ax.set_zlim(lim[0], lim[1])
        else:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)

def scatter_tokens(ax, pts, alpha=0.72, size=18):
    for i in range(3):
        m = lbls == i
        p = pts[m]
        ax.scatter(p[:,0], p[:,1], p[:,2],
                   c=CLS_C[i], alpha=alpha, s=size, edgecolors='none', zorder=3)

def formula_box(ax, text):
    ax.text2D(0.50, -0.04, text,
              transform=ax.transAxes, ha='center', va='top',
              fontsize=8.5, color=FG,
              bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                        edgecolor='#aaaaaa', linewidth=0.9))

def norm_label(ax):
    ax.text2D(0.03, 0.94, r'$\|\mathbf{h}_t\|=1$',
              transform=ax.transAxes, fontsize=8, color=FG,
              bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                        edgecolor='#bbbbbb', linewidth=0.7))

# ════════════════════════════════════════════════════════════════════════════
# 3D axes in matplotlib have large internal padding — we overlap the bounding
# boxes intentionally so the visible content (not whitespace) fills the figure.
# W is intentionally > 0.5: the 3D bbox has ~17% internal padding on each side,
# so overlapping bboxes by 0.20 brings the visible plots ~1 cm apart.
W, H = 0.70, 0.46
fig = plt.figure(figsize=(13, 10), facecolor=BG)

# ── (a) Contrastive Loss ──────────────────────────────────────────────────────
ax1 = fig.add_axes([-0.10, 0.52, W, H], projection='3d', facecolor=PANEL)
scatter_tokens(ax1, raw_pts)

rng = np.random.default_rng(1)
for i in range(3):
    p = raw_pts[lbls == i]
    for j in range(0, 8, 2):
        ax1.plot([p[j,0], p[j+1,0]], [p[j,1], p[j+1,1]], [p[j,2], p[j+1,2]],
                 color=GREEN, lw=0.9, alpha=0.55)

for c0, c1 in [(0, 1), (1, 2), (0, 2)]:
    p0, p1 = raw_pts[lbls == c0], raw_pts[lbls == c1]
    for _ in range(3):
        a = p0[rng.integers(len(p0))]
        b = p1[rng.integers(len(p1))]
        ax1.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]],
                 color=RED, lw=0.9, alpha=0.35, linestyle='--')

# Annotate a few token embeddings explicitly
for j, idx in enumerate([0, 4, 9]):
    p = raw_pts[lbls == 0][idx]
    ax1.text(p[0]+0.06, p[1]+0.03, p[2]+0.06,
             rf'$h_{{t_{j+1}}}$', color=CLS_C[0], fontsize=7)

ax1.set_title('(a) Contrastive Loss', fontsize=11, fontweight='bold',
              color=FG, pad=10, loc='left')
formula_box(ax1, r'$\mathcal{L} = \frac{1}{2}\left[y\,d^2 + (1-y)\max(m-d,0)^2\right]$')
style_ax(ax1, lim=(-1.3, 2.0))

# ── (b) Triplet Loss ──────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.36, 0.52, W, H], projection='3d', facecolor=PANEL)
scatter_tokens(ax2, raw_pts, alpha=0.15, size=12)

anchor   = raw_pts[lbls == 0][0]
positive = raw_pts[lbls == 0][5]
negative = raw_pts[lbls == 1][2]

ax2.scatter(*anchor,   c='black',     s=75, zorder=6, edgecolors=ORANGE, linewidths=1.6)
ax2.scatter(*positive, c=CLS_C[0],    s=75, zorder=6, edgecolors='black', linewidths=1.2)
ax2.scatter(*negative, c=CLS_C[1],    s=75, zorder=6, edgecolors='black', linewidths=1.2)

ax2.plot([anchor[0], positive[0]], [anchor[1], positive[1]], [anchor[2], positive[2]],
         color=GREEN, lw=2.0, zorder=4)
ax2.plot([anchor[0], negative[0]], [anchor[1], negative[1]], [anchor[2], negative[2]],
         color=RED, lw=2.0, linestyle='--', zorder=4)

def midpt(a, b, off=(0, 0, 0)):
    return (a + b) / 2 + np.array(off)

ax2.text(*midpt(anchor, positive, [0.0, 0.12, 0.10]),
         r'$d_+$', color=GREEN, fontsize=9, fontweight='bold')
ax2.text(*midpt(anchor, negative, [0.05, 0.0, 0.12]),
         r'$d_-$', color=RED,   fontsize=9, fontweight='bold')

ax2.text(*(anchor   + [-0.14, -0.06,  0.14]), r'$\mathbf{h}^{(a)}$', color='black',   fontsize=8.5)
ax2.text(*(positive + [ 0.09,  0.00,  0.00]), r'$\mathbf{h}^{(p)}$', color=CLS_C[0], fontsize=8.5)
ax2.text(*(negative + [ 0.09,  0.00,  0.00]), r'$\mathbf{h}^{(n)}$', color=CLS_C[1], fontsize=8.5)

ax2.set_title('(b) Triplet Loss', fontsize=11, fontweight='bold',
              color=FG, pad=10, loc='left')
formula_box(ax2, r'$\mathcal{L} = \max(d_+ - d_- + m,\; 0)$')
style_ax(ax2, lim=(-1.3, 2.0))

# ── (c) ArcFace — Angular Margin ─────────────────────────────────────────────
ax3 = fig.add_axes([-0.10, 0.07, W, H], projection='3d', facecolor=PANEL)
ax3.plot_wireframe(*SPH, color=GRID, alpha=0.18, linewidth=0.25)
scatter_tokens(ax3, norm_pts)

for i, (c, col) in enumerate(zip(norm_c, CLS_C)):
    ax3.quiver(0, 0, 0, c[0], c[1], c[2],
               color=col, linewidth=2.1, arrow_length_ratio=0.13, zorder=5)
    ax3.text(c[0]*1.21, c[1]*1.21, c[2]*1.21,
             rf'$\mathbf{{W}}_{i+1}$', color=col, fontsize=9.5, fontweight='bold')

# Great-circle arc = angular margin between prototypes 0 and 1
t   = np.linspace(0.04, 0.96, 70)
arc = normalize(np.outer(1 - t, norm_c[0]) + np.outer(t, norm_c[1]))
ax3.plot(arc[:,0], arc[:,1], arc[:,2], color=ORANGE, lw=2.0, linestyle=':', alpha=0.9)
mid = arc[len(arc) // 2]
ax3.text(mid[0]*1.14, mid[1]*1.14, mid[2]*1.08,
         r'$\theta_y + m$', color=ORANGE, fontsize=8.5)

norm_label(ax3)
ax3.set_title('(c) ArcFace — Angular Margin Softmax', fontsize=11,
              fontweight='bold', color=FG, pad=10, loc='left')
formula_box(ax3,
    r'$\mathcal{L} = -\log\frac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)}+\sum_{j\neq y}e^{s\cos\theta_j}}$')
style_ax(ax3, lim=1.35)

# ── (d) CosFace — Cosine Margin ───────────────────────────────────────────────
ax4 = fig.add_axes([0.36, 0.07, W, H], projection='3d', facecolor=PANEL)
ax4.plot_wireframe(*SPH, color=GRID, alpha=0.18, linewidth=0.25)
scatter_tokens(ax4, norm_pts)

for i, (c, col) in enumerate(zip(norm_c, CLS_C)):
    ax4.quiver(0, 0, 0, c[0], c[1], c[2],
               color=col, linewidth=2.1, arrow_length_ratio=0.13, zorder=5)
    ax4.text(c[0]*1.21, c[1]*1.21, c[2]*1.21,
             rf'$\mathbf{{W}}_{i+1}$', color=col, fontsize=9.5, fontweight='bold')

for c_vec, col in zip(norm_c, CLS_C):
    perp1 = np.cross(c_vec, [0, 0, 1])
    if np.linalg.norm(perp1) < 0.01:
        perp1 = np.cross(c_vec, [0, 1, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(c_vec, perp1)
    perp2 /= np.linalg.norm(perp2)
    th   = np.linspace(0, 2 * np.pi, 90)
    ring = normalize(c_vec + 0.19 * (np.outer(np.cos(th), perp1) +
                                      np.outer(np.sin(th), perp2)))
    lw_r = 2.0 if col == CLS_C[0] else 1.0
    al_r = 0.90 if col == CLS_C[0] else 0.38
    ax4.plot(ring[:,0], ring[:,1], ring[:,2],
             color=ORANGE, lw=lw_r, linestyle='--', alpha=al_r)

# Label the highlighted margin ring
c0v   = norm_c[0]
perp1 = np.cross(c0v, [0, 0, 1]); perp1 /= np.linalg.norm(perp1)
perp2 = np.cross(c0v, perp1);    perp2 /= np.linalg.norm(perp2)
th    = np.linspace(0, 2 * np.pi, 90)
ring0 = normalize(c0v + 0.19 * (np.outer(np.cos(th), perp1) +
                                  np.outer(np.sin(th), perp2)))
ax4.text(ring0[10, 0], ring0[10, 1], ring0[10, 2] + 0.10,
         r'$\cos\theta_y - m$', color=ORANGE, fontsize=8.5)

norm_label(ax4)
ax4.set_title('(d) CosFace — Cosine Margin Softmax', fontsize=11,
              fontweight='bold', color=FG, pad=10, loc='left')
formula_box(ax4,
    r'$\mathcal{L} = -\log\frac{e^{s(\cos\theta_y - m)}}{e^{s(\cos\theta_y-m)}+\sum_{j\neq y}e^{s\cos\theta_j}}$')
style_ax(ax4, lim=1.35)

# ── Global legend ─────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=CLS_C[i],
                   label=fr'Token embeddings $\mathbf{{h}}_t$ — {CLS_N[i]}')
    for i in range(3)
] + [
    Line2D([0], [0], color=GREEN,  lw=1.5,
           label=r'Par positivo / $d_+$  (mesma classe)'),
    Line2D([0], [0], color=RED,    lw=1.5, linestyle='--',
           label=r'Par negativo / $d_-$ (classes distintas)'),
    Line2D([0], [0], color=ORANGE, lw=1.8, linestyle=':',
           label=r'Margem angular $\theta_y + m$ (ArcFace)'),
    Line2D([0], [0], color=ORANGE, lw=1.8, linestyle='--',
           label=r'Margem coseno $\cos\theta_y - m$ (CosFace)'),
    Line2D([0], [0], color='gray', lw=0, marker=r'$\mathbf{W}$', markersize=10,
           label=r'Protótipos de classe $\mathbf{W}_i$ (esfera unitária)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           facecolor='white', labelcolor=FG, fontsize=8.5,
           framealpha=1.0, edgecolor='#aaaaaa',
           bbox_to_anchor=(0.5, 0.0))

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.005,
    r'Fig. X — 3D projection of token embedding spaces $\mathbf{h}_t \in \mathbb{R}^d$ under four metric learning objectives. '
    r'Each point represents a document token embedding.'
    '\n'
    r'(a–b) Pair/triplet methods enforce constraints via Euclidean distances with no geometric anchor. '
    r'(c–d) Face-recognition losses project embeddings onto the unit hypersphere ($\|\mathbf{h}_t\|=1$) and '
    r'enforce inter-class angular or cosine margins between fixed class prototypes $\mathbf{W}_i$.',
    ha='center', va='top', fontsize=7.8, color='#333333', style='italic',
    wrap=True
)

out = 'docs/assets/losses_embedding_space.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
plt.show()
