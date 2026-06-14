#!/usr/bin/env python3
"""
Gera duas figuras de prompt conditioning para o paper:

  1. prompt_conditioning_visual.{pdf,png}
     Layout 2×4 — visual activation ||h_v^{-1}||_2 por patch espacial.
     Linhas: base prompt / rich prompt.
     Colunas: 4 documentos (2 por classe).
     Mensagem: a ativação visual é determinada pela classe do documento,
     não pelo prompt — evidenciando o sinal discriminativo já presente
     nas representações congeladas.

  2. prompt_conditioning_attention.{pdf,png}
     Layout 2×2 — atenção texto→visual na camada -1.
     Linhas: base prompt / rich prompt.
     Colunas: 1 documento por classe.
     Mensagem: o prompt rico redireciona a atenção para regiões de
     layout discriminativo (cabeçalhos, bordas, campos de formulário).

Run:
  python scripts/figures/generate_prompt_conditioning_figure.py [--gpu 0]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   8,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MAX_NUM  = 12
IMG_SIZE = 448

BASE_PROMPT = "<image> Analyze this document"

RICH_PROMPT = (
    "<image> Analyze the provided document image and give me its visual description"
    " based on: Shapes and Elements: presence of graphical components, tables,"
    " sections, headers, and any other visual elements. Layout Consistency: Evaluate"
    " the spatial arrangement of text blocks, margins, and alignments. Content Type:"
    " Ensure the document types of content (e.g., tables, forms, paragraphs),"
    " regardless of specific wording."
)

DOCUMENT_CLASSES = [
    {
        "label": "Philip Morris Letter",
        "dir":   "/mnt/data/la-cdip/data/philip_morris_letter",
        "n_docs": 2,
    },
    {
        "label": "Lorillard Invoice",
        "dir":   "/mnt/data/la-cdip/data/lorillard_invoice",
        "n_docs": 2,
    },
]

PANEL_LABELS = "abcdefghijklmnop"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading InternVL3-2B…")
    model = AutoModelForCausalLM.from_pretrained(
        "OpenGVLab/InternVL3-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    ).eval()
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-2B", trust_remote_code=True, use_fast=False
    )
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    return model, tok


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_inputs(model, tokenizer, image: Image.Image, prompt: str):
    from cavl_doc.data.transforms import dynamic_preprocess, build_transform, find_closest_aspect_ratio
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    transform = build_transform(IMG_SIZE)
    blocks    = dynamic_preprocess(
        image, max_num=MAX_NUM, image_size=IMG_SIZE, use_thumbnail=True
    )
    pv = torch.stack([transform(b) for b in blocks]).to(torch.bfloat16).to(
        next(model.parameters()).device
    )
    w, h = image.size
    ratios = sorted(
        {(i, j) for n in range(1, MAX_NUM + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= MAX_NUM},
        key=lambda r: r[0] * r[1],
    )
    cols, rows = find_closest_aspect_ratio(w / h, ratios, w, h, IMG_SIZE)
    inp        = prepare_inputs_for_multimodal_embedding(model, tokenizer, pv, prompt)
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    seq        = inp["input_ids"][0].cpu()
    img_pos    = (seq == img_ctx_id).nonzero(as_tuple=True)[0]
    text_pos   = (seq != img_ctx_id).nonzero(as_tuple=True)[0]
    return inp, img_pos, text_pos, rows, cols, model.num_image_token


# ---------------------------------------------------------------------------
# Combined extraction: visual activation + text attention in one forward pass
# ---------------------------------------------------------------------------

def extract_maps(model, tokenizer, image: Image.Image, prompt: str):
    """
    Returns (vis_act, text_attn) both as (rows, cols) ndarrays in [0,1].
    vis_act  : ||h_v^{-1}||_2 per spatial patch (prompt-invariant signal)
    text_attn: mean text-to-visual attention at last LM layer (prompt-sensitive)
    """
    inp, img_pos, text_pos, rows, cols, n_tok = prepare_inputs(
        model, tokenizer, image, prompt
    )
    lm_layers = model.language_model.model.layers
    last_li   = len(lm_layers) - 1

    attn_accum = torch.zeros(len(img_pos), dtype=torch.float32)
    attn_count = [0]

    orig_fwd = lm_layers[last_li].self_attn.forward
    def patched_fwd(*a, **kw):
        kw["output_attentions"] = True
        return orig_fwd(*a, **kw)
    lm_layers[last_li].self_attn.forward = patched_fwd

    def hook_fn(module, inp_, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn = output[1][0].float().cpu()   # [heads, seq, seq]
            attn_accum.add_(attn[:, text_pos, :][:, :, img_pos].mean(dim=(0, 1)))
            attn_count[0] += 1

    hook = lm_layers[last_li].self_attn.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, return_dict=True)
    finally:
        hook.remove()
        lm_layers[last_li].self_attn.forward = orig_fwd

    n_spatial = rows * cols

    # Visual activation — 5th–99th percentile clip
    h       = out.hidden_states[-1][0].float().cpu()
    norms   = h[img_pos].norm(dim=-1)
    vis_m   = norms[:n_spatial * n_tok].reshape(n_spatial, n_tok).mean(dim=1).reshape(rows, cols).numpy()
    lo, hi  = np.percentile(vis_m, 5), np.percentile(vis_m, 99)
    vis_act = np.clip((vis_m - lo) / (hi - lo + 1e-8), 0, 1)

    # Text attention — 50th–99th percentile clip
    if attn_count[0] > 0:
        attn_accum /= attn_count[0]
    attn_m    = attn_accum[:n_spatial * n_tok].reshape(n_spatial, n_tok).mean(dim=1).reshape(rows, cols).numpy()
    lo, hi    = np.percentile(attn_m, 50), np.percentile(attn_m, 99)
    text_attn = np.clip((attn_m - lo) / (hi - lo + 1e-8), 0, 1)

    return vis_act, text_attn


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay(image: Image.Image, heat: np.ndarray, alpha: float, cmap: str) -> np.ndarray:
    gray     = np.array(image.convert("L").convert("RGB")).astype(np.float32)
    h, w     = gray.shape[:2]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat_np  = np.array(heat_img) / 255.0
    heat_rgb = plt.get_cmap(cmap)(heat_np)[:, :, :3] * 255
    return np.clip(gray * (1 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

def _add_class_headers(fig, gs, class_labels: list[str], n_docs_per_class: int,
                        top_m: float, fig_h: float):
    """Draw class name labels and underlines above the panel columns."""
    for cls_idx, cls_label in enumerate(class_labels):
        c_start = cls_idx * n_docs_per_class
        c_end   = c_start + n_docs_per_class - 1
        pos_l   = gs[0, c_start].get_position(fig)
        pos_r   = gs[0, c_end].get_position(fig)
        mid_x   = (pos_l.x0 + pos_r.x1) / 2
        fig.text(mid_x, 1.0 - top_m * 0.32 / fig_h,
                 cls_label,
                 ha="center", va="top", fontsize=8.5,
                 fontweight="bold", style="italic",
                 transform=fig.transFigure)
        fig.add_artist(matplotlib.lines.Line2D(
            [pos_l.x0, pos_r.x1],
            [1.0 - top_m * 0.58 / fig_h] * 2,
            transform=fig.transFigure,
            color="#aaaaaa", linewidth=0.7,
        ))


def _add_class_separator(fig, gs, col_left: int, col_right: int,
                          bot_m: float, fig_h: float, top_m: float):
    """Dashed vertical line between two class groups."""
    pos_l = gs[0, col_left].get_position(fig)
    pos_r = gs[0, col_right].get_position(fig)
    div_x = (pos_l.x1 + pos_r.x0) / 2
    fig.add_artist(matplotlib.lines.Line2D(
        [div_x, div_x],
        [bot_m / fig_h, 1.0 - top_m * 0.58 / fig_h],
        transform=fig.transFigure,
        color="#999999", linewidth=0.8, linestyle="--",
    ))


def _add_row_labels(fig, gs, labels: list[str], left_m: float, fig_w: float):
    for r, label in enumerate(labels):
        pos   = gs[r, 0].get_position(fig)
        mid_y = (pos.y0 + pos.y1) / 2
        fig.text(left_m * 0.72 / fig_w, mid_y, label,
                 ha="center", va="center", fontsize=7.5, style="italic",
                 transform=fig.transFigure)


def _add_colorbar(fig, cbar_ax, cmap: str, label: str):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "", "High"])
    cb.ax.tick_params(labelsize=6, length=2, pad=2)
    cb.set_label(label, fontsize=7, labelpad=4)
    cb.outline.set_linewidth(0.5)


def _save(fig, output_path: str, dpi: int):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("pdf", {}), ("png", {"dpi": dpi})]:
        fig.savefig(f"{out}.{ext}", bbox_inches="tight",
                    facecolor="white", edgecolor="none", **kw)
        print(f"Saved → {out}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Visual Activation  (1 row × 4 cols)
# Usa apenas o base prompt (ativação é invariante ao prompt — basta mostrar uma vez)
# ---------------------------------------------------------------------------

def make_visual_figure(
    doc_images:   list[Image.Image],   # 4 docs: [cls0_d0, cls0_d1, cls1_d0, cls1_d1]
    vis_maps:     list[np.ndarray],    # mapa de ativação (base prompt)
    class_labels: list[str],
    output_path:  str,
    alpha: float, cmap: str, dpi: int,
):
    n_rows, n_cols = 1, 4

    col_w  = 1.55
    row_h  = 2.10
    left_m = 0.18   # sem label de linha — só header de classe
    top_m  = 0.68
    bot_m  = 0.10
    cbar_w = 0.13

    fig_w = left_m + n_cols * col_w + 0.26 + cbar_w + 0.08
    fig_h = top_m + n_rows * row_h + bot_m

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        left   = left_m / fig_w,
        right  = 1.0 - 0.08 / fig_w,
        top    = 1.0 - top_m / fig_h,
        bottom = bot_m / fig_h,
        wspace = 0.018,
        hspace = 0.025,
        width_ratios=[1] * n_cols + [cbar_w / col_w],
    )

    cbar_ax = fig.add_subplot(gs[:, n_cols])

    _add_class_headers(fig, gs, class_labels, n_docs_per_class=2, top_m=top_m, fig_h=fig_h)
    _add_class_separator(fig, gs, col_left=1, col_right=2, bot_m=bot_m, fig_h=fig_h, top_m=top_m)

    for c, (img, hmap) in enumerate(zip(doc_images, vis_maps)):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(overlay(img, hmap, alpha, cmap))
        ax.axis("off")
        ax.text(0.03, 0.97, f"({PANEL_LABELS[c]})",
                transform=ax.transAxes,
                fontsize=7, color="white", va="top", ha="left",
                fontweight="bold", fontfamily="serif",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.45))

    _add_colorbar(fig, cbar_ax, cmap, r"$\|\mathbf{h}_v^{(-1)}\|_2$")
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Figure 2 — Text→Visual Attention  (1 row × 4 cols)
# Layout: [base_cls0 | rich_cls0 || base_cls1 | rich_cls1]
# Mostra prompt vs classes: dentro de cada par de colunas vê-se o efeito do prompt;
# entre os dois pares vê-se a diferença de classe.
# ---------------------------------------------------------------------------

def make_attention_figure(
    # 1 doc per class: [cls0_d0, cls1_d0]
    att_base_per_class: list[np.ndarray],   # [att_base_cls0, att_base_cls1]
    att_rich_per_class: list[np.ndarray],   # [att_rich_cls0, att_rich_cls1]
    img_per_class:      list[Image.Image],  # [img_cls0, img_cls1]
    class_labels:       list[str],
    output_path:        str,
    alpha: float, cmap: str, dpi: int,
):
    # Column order: base_cls0, rich_cls0, base_cls1, rich_cls1
    panels = [
        (img_per_class[0], att_base_per_class[0]),
        (img_per_class[0], att_rich_per_class[0]),
        (img_per_class[1], att_base_per_class[1]),
        (img_per_class[1], att_rich_per_class[1]),
    ]
    col_prompts = ["Base\nPrompt", "Rich\nPrompt", "Base\nPrompt", "Rich\nPrompt"]

    n_rows, n_cols = 1, 4

    col_w  = 1.55
    row_h  = 2.10
    left_m = 0.18
    top_m  = 0.95   # extra space for two header rows (class + prompt)
    bot_m  = 0.10
    cbar_w = 0.13

    fig_w = left_m + n_cols * col_w + 0.26 + cbar_w + 0.08
    fig_h = top_m + n_rows * row_h + bot_m

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        left   = left_m / fig_w,
        right  = 1.0 - 0.08 / fig_w,
        top    = 1.0 - top_m / fig_h,
        bottom = bot_m / fig_h,
        wspace = 0.018,
        hspace = 0.025,
        width_ratios=[1] * n_cols + [cbar_w / col_w],
    )

    cbar_ax = fig.add_subplot(gs[:, n_cols])

    # Class headers spanning 2 cols each (base+rich per class)
    _add_class_headers(fig, gs, class_labels, n_docs_per_class=2, top_m=top_m, fig_h=fig_h)
    _add_class_separator(fig, gs, col_left=1, col_right=2, bot_m=bot_m, fig_h=fig_h, top_m=top_m)

    # Prompt sub-headers below class headers
    for c, prompt_label in enumerate(col_prompts):
        pos = gs[0, c].get_position(fig)
        fig.text(pos.x0 + pos.width / 2,
                 1.0 - top_m * 0.72 / fig_h,
                 prompt_label,
                 ha="center", va="top", fontsize=7, style="italic",
                 transform=fig.transFigure)

    # Panels
    for c, (img, hmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(overlay(img, hmap, alpha, cmap))
        ax.axis("off")
        ax.text(0.03, 0.97, f"({PANEL_LABELS[c]})",
                transform=ax.transAxes,
                fontsize=7, color="white", va="top", ha="left",
                fontweight="bold", fontfamily="serif",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.45))

    _add_colorbar(fig, cbar_ax, cmap, "Attention")
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-vis", default=str(ROOT / "docs/assets/prompt_conditioning_visual"))
    p.add_argument("--out-att", default=str(ROOT / "docs/assets/prompt_conditioning_attention"))
    p.add_argument("--alpha",    type=float, default=0.55)
    p.add_argument("--vis-cmap", default="inferno")
    p.add_argument("--att-cmap", default="plasma")
    p.add_argument("--dpi",      type=int,   default=300)
    p.add_argument("--gpu",      type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(device)

    doc_images:   list[Image.Image] = []
    vis_base:     list[np.ndarray]  = []
    vis_rich:     list[np.ndarray]  = []
    att_base:     list[np.ndarray]  = []
    att_rich:     list[np.ndarray]  = []
    class_labels: list[str]         = []

    for cls in DOCUMENT_CLASSES:
        cls_dir = Path(cls["dir"])
        tifs    = sorted(cls_dir.glob("*.tif"))[: cls["n_docs"]]
        if not tifs:
            tifs = sorted(cls_dir.glob("*.jpg"))[: cls["n_docs"]]
        class_labels.append(cls["label"])
        print(f"\n=== {cls['label']} ===")

        for tif in tifs:
            image = Image.open(tif).convert("RGB")
            doc_images.append(image)

            print(f"  {tif.name}  — base prompt…")
            va_b, ta_b = extract_maps(model, tokenizer, image, BASE_PROMPT)
            torch.cuda.empty_cache()

            print(f"  {tif.name}  — rich prompt…")
            va_r, ta_r = extract_maps(model, tokenizer, image, RICH_PROMPT)
            torch.cuda.empty_cache()

            vis_base.append(va_b)
            vis_rich.append(va_r)
            att_base.append(ta_b)
            att_rich.append(ta_r)

    # Figure 1: visual activation — 1 row × 4 cols, base prompt only
    print("\n--- Generating visual activation figure (1×4) ---")
    make_visual_figure(
        doc_images   = doc_images,
        vis_maps     = vis_base,          # invariant to prompt — show once
        class_labels = class_labels,
        output_path  = args.out_vis,
        alpha        = args.alpha,
        cmap         = args.vis_cmap,
        dpi          = args.dpi,
    )

    # Figure 2: text attention — 1 row × 4 cols [base_cls0|rich_cls0||base_cls1|rich_cls1]
    print("\n--- Generating text attention figure (1×4) ---")
    make_attention_figure(
        att_base_per_class = [att_base[0], att_base[2]],   # first doc of each class
        att_rich_per_class = [att_rich[0], att_rich[2]],
        img_per_class      = [doc_images[0], doc_images[2]],
        class_labels       = class_labels,
        output_path        = args.out_att,
        alpha              = args.alpha,
        cmap               = args.att_cmap,
        dpi                = args.dpi,
    )


if __name__ == "__main__":
    main()
