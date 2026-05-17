#!/usr/bin/env python3
"""Generates teaser_fig.pdf/png for the ArcDoc paper.

Run from repo root:
    python scripts/figures/generate_teaser_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# --- Global Configuration for Paper Figures ---
# Set font sizes and family for IEEE/Academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Output directory
from pathlib import Path
OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / 'docs' / 'assets') + '/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Output directory set to: {os.path.abspath(OUTPUT_DIR)}")

# --- Figure 3: Teaser Figure (Aligned) ---

def draw_aligned_figure():
    # Vertical Alignment: 2 Rows, 1 Column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7)) # Increased size
    plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.02, right=0.98)

    # Styles
    COLOR_BG_GEN = '#fff5f5'   # Faint Red/Warm
    COLOR_BG_OURS = '#f0f9f0'  # Faint Green/Cool
    COLOR_BORDER_GEN = '#e6b8af'
    COLOR_BORDER_OURS = '#b6d7a8'
    
    COLOR_MODEL_GEN = {'face': '#e1d5e7', 'edge': '#9673a6'} # Purple
    COLOR_MODEL_OURS = {'face': '#d5e8d4', 'edge': '#82b366'} # Green
    
    FONT_SIZE_BODY = 14 # Increased font size from 12

    def draw_box(ax, x, y, w, h, text, style, subtext=None, alpha=1.0, zorder=2, fontsize=None):
        if fontsize is None: fontsize = FONT_SIZE_BODY
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     linewidth=1.5, edgecolor=style['edge'], facecolor=style['face'], 
                                     alpha=alpha, zorder=zorder)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', color='#333333', zorder=zorder+1)
        if subtext:
             ax.text(x + w/2, y + h/2 - 0.35, subtext, ha='center', va='top', 
                     fontsize=fontsize-2, color='#555555', zorder=zorder+1)
        return box

    def draw_arrow(ax, start, end, color='#555555'):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color=color))

    def draw_doc_stack(ax, x, y, w, h):
        # Draw a stack of 3 documents within rectangular bounds w,h
        # Center it visually around x + w/2
        doc_w, doc_h = w * 0.8, h * 0.9
        start_x = x + (w - doc_w)/2
        start_y = y + (h - doc_h)/2
        
        for i in range(3):
            offset = i * 0.08
            rect = patches.Rectangle((start_x + offset, start_y + offset), doc_w, doc_h, linewidth=1, edgecolor='#999', facecolor='white', zorder=2+i)
            ax.add_patch(rect)
            for line_y in [0.75, 0.55, 0.35, 0.15]:
                ax.plot([start_x + offset + 0.1, start_x + offset + doc_w - 0.1], 
                        [start_y + offset + line_y*doc_h, start_y + offset + line_y*doc_h], color='#ccc', lw=1, zorder=3+i)
        ax.text(x + w/2, y - 0.25, "Input Docs", ha='center', fontsize=13, color='#555') # Increased from 11

    # ==========================================
    # Layout Logic
    # ==========================================
    # We define X standard positions to ensure perfect alignments
    # Total width approx 12 units
    
    X_START = 1.5
    W_DOC = 1.2
    GAP_1 = 1.0
    W_MODEL = 2.8
    GAP_2 = 1.0
    X_OUTPUT = X_START + W_DOC + GAP_1 + W_MODEL + GAP_2 # Start of output section
    
    Y_CENTER_A = 1.6
    Y_CENTER_B = 1.6
    
    # ==========================================
    # PANEL A: Generative Model
    # ==========================================
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 3.2)
    ax1.axis('off')
    
    # Container
    container_a = patches.FancyBboxPatch((0.2, 0.2), 11.6, 2.5, boxstyle="round,pad=0.1", 
                                         linewidth=1.5, edgecolor=COLOR_BORDER_GEN, facecolor=COLOR_BG_GEN, zorder=0)
    ax1.add_patch(container_a)
    ax1.set_title("(A) Multimodal Generative Model", loc='left', fontsize=18, fontweight='bold', color='#000000', pad=5) # Increased from 14

    # 1. Docs
    draw_doc_stack(ax1, X_START, Y_CENTER_A - 0.6, W_DOC, 1.2)

    # Arrow 1
    x_arrow1_start = X_START + W_DOC + 0.1 # Small padding
    x_arrow1_end = X_START + W_DOC + GAP_1 - 0.1
    draw_arrow(ax1, (x_arrow1_start, Y_CENTER_A), (x_arrow1_end, Y_CENTER_A))

    # 2. Model
    draw_box(ax1, X_START + W_DOC + GAP_1, Y_CENTER_A - 0.7, W_MODEL, 1.4, 
             "Multimodal\nLLM", COLOR_MODEL_GEN, subtext="(Autoregressive)", fontsize=13) # Increased from 11
             
    # Arrow 2
    x_arrow2_start = X_START + W_DOC + GAP_1 + W_MODEL + 0.1
    x_arrow2_end = X_OUTPUT - 0.1
    draw_arrow(ax1, (x_arrow2_start, Y_CENTER_A), (x_arrow2_end, Y_CENTER_A))

    # 3. Sequential Output
    # We draw tokens starting exactly at X_OUTPUT
    tokens = ["<class>", "Invoice", "<EOS>"]
    for i, t in enumerate(tokens):
        y_pos = Y_CENTER_A + 0.4 - i * 0.55
        draw_box(ax1, X_OUTPUT, y_pos, 1.3, 0.30, t, {'face': '#fff2cc', 'edge': '#d6b656'}, fontsize=12) # Increased from 10
        # Connectors
        if i < len(tokens) - 1:
             draw_arrow(ax1, (X_OUTPUT + 0.65, y_pos), (X_OUTPUT + 0.65, y_pos - 0.2))
             
    ax1.text(X_OUTPUT + 1.6, Y_CENTER_A, "Sequential Output", ha='left', va='center', fontsize=13, color='#666666') # Increased from 11


    # ==========================================
    # PANEL B: ArcDoc Approach
    # ==========================================
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 3.2)
    ax2.axis('off')

    # Container
    container_b = patches.FancyBboxPatch((0.2, 0.2), 11.6, 2.5, boxstyle="round,pad=0.1", 
                                         linewidth=1.5, edgecolor=COLOR_BORDER_OURS, facecolor=COLOR_BG_OURS, zorder=0)
    ax2.add_patch(container_b)
    ax2.set_title("(B) ArcDoc (Ours)", loc='left', fontsize=16, fontweight='bold', color='#000000', pad=5) # Increased from 14

    # 1. Docs
    draw_doc_stack(ax2, X_START, Y_CENTER_B - 0.6, W_DOC, 1.2)

    # Arrow 1
    draw_arrow(ax2, (x_arrow1_start, Y_CENTER_B), (x_arrow1_end, Y_CENTER_B))

    # 2. Model
    draw_box(ax2, X_START + W_DOC + GAP_1, Y_CENTER_B - 0.7, W_MODEL, 1.4, 
             "Multimodal\nEncoder", COLOR_MODEL_OURS, subtext="(LVLM Embeddings)", fontsize=13) # Increased from 11

    # Arrow 2
    draw_arrow(ax2, (x_arrow2_start, Y_CENTER_B), (x_arrow2_end, Y_CENTER_B))

    # 3. Metric Space Output
    # Circle centered at X_OUTPUT + radius
    radius = 1.0
    center_x = X_OUTPUT + radius
    circle = patches.Circle((center_x, Y_CENTER_B), radius, facecolor='white', edgecolor='#999', linestyle='--', zorder=1)
    ax2.add_patch(circle)
    
    # Points inside circle
    ax2.scatter([center_x - 0.3, center_x + 0.3, center_x - 0.1], 
                [Y_CENTER_B + 0.2, Y_CENTER_B + 0.1, Y_CENTER_B - 0.3], c='#ccc', s=30)
    ax2.scatter([center_x + 0.1], [Y_CENTER_B], c='#d79b00', edgecolors='black', s=80, marker='*', zorder=5)

    ax2.text(center_x + radius + 0.2, Y_CENTER_B, "Metric Proximity", ha='left', va='center', fontsize=13, color='#666666') # Increased from 11

    plt.tight_layout()
    
    # Save
    filename = 'teaser_fig'
    
    # Save PDF
    pdf_path = os.path.join(OUTPUT_DIR, f'{filename}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
    
    # Save PNG (High Res)
    png_path = os.path.join(OUTPUT_DIR, f'{filename}.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    
    print(f"Saved {filename} to {OUTPUT_DIR}")
    plt.show()

draw_aligned_figure()

draw_aligned_figure()
