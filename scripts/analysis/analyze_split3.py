#!/usr/bin/env python3
"""
Análise Diagnóstica do Split 3 — Por que o split 3 é o mais difícil?

Split 3 contém: memo, presentation, questionnaire, resume
— todas classes text-heavy, com layouts visualmente similares.

Seções:
  1. Visão Geral do Split 3
  2. Comparação Cross-Modelo — EER por Split (heatmap)
  3. Análise de Erros por Par de Classes
  4. Curva de Treino — Subset val/eer vs Full Eval (W&B)
  5. Distribuição de Similarity Scores — Split 3 vs Outros

Uso:
    python scripts/analysis/analyze_split3.py
    python scripts/analysis/analyze_split3.py --output results/split3_analysis.html
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = WORKSPACE_ROOT / "results" / "split3_analysis.html"

ENTITY               = "jpcosta1990-university-of-brasilia"
PROJECT_SPRINT3B     = "CaVL-Doc_LA-CDIP_Sprint3b_s32_k3"
PROJECT_FULL_EVAL    = "CaVL-Doc_LA-CDIP_FullEval"

EMB_METHODS = [
    "pixel-l2",
    "pixel-cosine",
    "jina-v4",
    "internvl3-in",
    "internvl3-out",
]
VLM_METHODS = [
    "gemma4-e2b",
    "gemma4-e4b",
    "internvl3-2b",
    "internvl3-8b",
    "internvl3-14b",
    "qwen3vl-2b",
    "qwen3vl-4b",
    "qwen3vl-8b",
]

# Embeddings avaliados no LA-CDIP (diretório correto)
EMB_LACDIP_DIR = WORKSPACE_ROOT / "results" / "emb_baseline_lacdip"
# Pares VLM (LA-CDIP) — usados para análise por classe
VLM_PAIRS_DIR  = WORKSPACE_ROOT / "results" / "vlm_metric"
# Classes LA-CDIP do split 3 são extraídas dinamicamente dos pares VLM

LOSS_LABELS = {
    "subcenter_cosface":  "Sub-Center CosFace",
    "subcenter_arcface":  "Sub-Center ArcFace",
    "contrastive":        "Contrastive",
    "cosface":            "CosFace",
    "arcface":            "ArcFace",
    "triplet":            "Triplet",
    "circle":             "Circle",
}
KNOWN_LOSSES = sorted(LOSS_LABELS.keys(), key=len, reverse=True)

POOLER_LABELS: dict[str, str] = {
    "":                           "Atenção nq=1 (baseline)",
    "richprompt_cor":             "Atenção + Rich Prompt",
    "cross_modal":                "Cross-Modal",
    "cross_modal_richprompt_cor": "Cross-Modal + Rich Prompt",
    "mean":                       "Mean Pooler",
    "mean_richprompt_cor":        "Mean Pooler + Rich Prompt",
}

# Epoch offset for fase2 runs
FASE1_TOTAL_EPOCHS = 10

# Imagens originais LA-CDIP (usadas no treino/avaliação real)
# Tenta primeiro o NAS do servidor de compute; fallback para final_split3 (augmentado, só para visualização)
_NAS_LACDIP = Path("/mnt/nas/joaopaulo/LA-CDIP/data")
_VAL_CSV_S3 = WORKSPACE_ROOT / "data" / "generated_splits" / "sprint3_zsl_val_3_train_excl_5" / "validation_pairs.csv"
SPLIT3_IMAGES_DIR  = _NAS_LACDIP if _NAS_LACDIP.exists() else (WORKSPACE_ROOT / "data" / "generated_splits" / "final_split3" / "images_val")
SPLIT3_IMAGES_ORIGINAL = _NAS_LACDIP.exists()  # True = imagens originais disponíveis

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa;
}
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 12px; }
h3 { color: #5d6d7e; }
.section {
    background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
th { padding: 6px 10px; background: #f0f0f0; border-bottom: 2px solid #ccc; text-align: center; }
td { padding: 5px 10px; text-align: center; }
img { max-width: 100%; }
.note { color: #666; font-size: 0.88em; font-style: italic; margin-top: 8px; }
.split3-header { background: #fff3cd !important; border: 2px solid #f0ad4e !important;
                 font-weight: bold; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str, style: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="{style}" />'


def _compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute EER using sklearn roc_curve."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        idx = np.argmin(np.abs(fnr - fpr))
        return float(fpr[idx])
    except Exception:
        return float("nan")


def _eer_pct(labels, scores) -> float:
    return _compute_eer(np.array(labels), np.array(scores)) * 100


def _load_pairs(base_dir: Path, split: int) -> Optional[pd.DataFrame]:
    path = base_dir / f"split{split}_pairs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["class_a"] = df["file_a_path"].str.split("/").str[0]
    df["class_b"] = df["file_b_path"].str.split("/").str[0]
    return df


def _load_summary(base_dir: Path) -> Optional[pd.DataFrame]:
    path = base_dir / "summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _class_thumbnail_b64(class_name: str, size: tuple[int, int] = (100, 130)) -> str | None:
    """Return base64 PNG thumbnail of the first example image for a split-3 LA-CDIP class."""
    class_dir = SPLIT3_IMAGES_DIR / class_name
    if not class_dir.exists():
        return None
    imgs = sorted(class_dir.glob("*.tif"))
    if not imgs:
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(imgs[0]) as im:
            im = im.convert("RGB")
            im.thumbnail(size, PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _parse_fulleval_s3(name: str) -> tuple[str | None, str | None]:
    """Parse (variant, loss) from a FullEval_Sprint3b_S3_* run name.
    Returns ("", loss) for baseline, (variant, loss) for other variants, (None, None) on mismatch.
    """
    nl = name.lower()
    if not nl.startswith("fulleval_sprint3b_s3_"):
        return None, None
    for loss in KNOWN_LOSSES:  # sorted longest-first prevents sub-string collision
        prefix = f"fulleval_sprint3b_s3_{loss}_"
        if nl.startswith(prefix):
            rest = nl[len(prefix):]
            rest = re.sub(r"_?fase\d.*$", "", rest).strip("_")
            return rest, loss
    return None, None


def _eer_color(eer_pct: float) -> str:
    """Return background color: green (0%) → yellow (30%) → red (60%+)."""
    v = min(max(eer_pct / 60.0, 0.0), 1.0)
    r = int(255 * v)
    g = int(255 * (1 - v * 0.7))
    b = int(255 * 0.2 * (1 - v))
    text_color = "#fff" if v > 0.65 else "#222"
    return f"background:rgb({r},{g},{b});color:{text_color};"


# ---------------------------------------------------------------------------
# Section 1: Visão Geral do Split 3
# ---------------------------------------------------------------------------

def build_section1() -> str:
    print("[Sec 1] Carregando visão geral do split 3...")

    # Load VLM pairs (LA-CDIP classes) for split 3 — use qwen3vl-8b as reference
    vlm_ref = VLM_PAIRS_DIR / "qwen3vl-8b" / "split3_pairs.csv"
    if vlm_ref.exists():
        df_vlm = pd.read_csv(vlm_ref)
        df_vlm["class_a"] = df_vlm["file_a_path"].str.split("/").str[0]
        df_vlm["class_b"] = df_vlm["file_b_path"].str.split("/").str[0]
        pos_counts = df_vlm[df_vlm["is_equal"] == 1].groupby("class_a").size().sort_values(ascending=False)
        total_pos  = int((df_vlm["is_equal"] == 1).sum())
        total_neg  = int((df_vlm["is_equal"] == 0).sum())
        classes_la = pos_counts.index.tolist()
    else:
        pos_counts, classes_la, total_pos, total_neg = {}, [], 0, 0

    # Class table (LA-CDIP classes)
    rows = ""
    for cls in classes_la:
        n_pos = pos_counts.get(cls, 0)
        label = cls.replace("_", " ")
        rows += (f"<tr><td style='text-align:left;font-weight:bold'>{label}</td>"
                 f"<td>{n_pos}</td></tr>")

    class_table = f"""
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th style="text-align:left">Classe (LA-CDIP)</th>
        <th>Pares positivos</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """

    # Bar chart: Jina-v4 EER across splits (LA-CDIP — emb_baseline_lacdip)
    jina_summary = EMB_LACDIP_DIR / "jina-v4" / "summary.csv"
    s_df = pd.read_csv(jina_summary) if jina_summary.exists() else pd.DataFrame()
    s_df = s_df[s_df["split"].isin([0, 1, 2, 3, 4])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    # Chart 1: Jina-v4
    ax = axes[0]
    if not s_df.empty:
        splits = s_df["split"].tolist()
        eers   = (s_df["eer"] * 100).tolist()
        colors = ["#E45756" if sp == 3 else "#4C78A8" for sp in splits]
        bars = ax.bar([f"Split {s}" for s in splits], eers, color=colors)
        ax.set_ylabel("EER (%)")
        ax.set_title("Jina-v4 (LA-CDIP) — EER por Split", fontsize=10)
        ax.set_ylim(0, max(eers) * 1.25)
        for bar, eer in zip(bars, eers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{eer:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Chart 2: qwen3vl-8b
    qwen_summary = VLM_PAIRS_DIR / "qwen3vl-8b" / "summary.csv"
    s_df2 = pd.read_csv(qwen_summary) if qwen_summary.exists() else pd.DataFrame()
    s_df2 = s_df2[s_df2["split"].isin([0, 1, 2, 3, 4])].copy()
    ax2 = axes[1]
    if not s_df2.empty:
        splits2 = s_df2["split"].tolist()
        eers2   = (s_df2["eer"] * 100).tolist()
        colors2 = ["#E45756" if sp == 3 else "#54A24B" for sp in splits2]
        bars2 = ax2.bar([f"Split {s}" for s in splits2], eers2, color=colors2)
        ax2.set_ylabel("EER (%)")
        ax2.set_title("Qwen3-VL-8B (LA-CDIP) — EER por Split", fontsize=10)
        ax2.set_ylim(0, max(eers2) * 1.25)
        for bar, eer in zip(bars2, eers2):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{eer:.1f}%", ha="center", va="bottom", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    b64 = _b64_png(fig)

    html = f"""
    <div class="section">
      <h2>1. Visão Geral — Split 3</h2>

      <h3>Classes do Split 3 (LA-CDIP)</h3>
      <p>O split 3 da base LA-CDIP contém <strong>{len(classes_la)} classes</strong>
         de documentos da indústria tabagista — formas específicas de relatório, estimativas,
         formulários e cartas. São documentos de aparência visual muito semelhante entre si,
         com texto denso, pouca variação de layout e sem elementos gráficos diferenciadores.</p>
      <div style="display:flex;gap:24px;align-items:flex-start">
        <div style="flex:0 0 auto">{class_table}</div>
        <div style="flex:1">
          <p class="note">Total split 3: {total_pos} pares positivos + {total_neg} negativos
             = {total_pos + total_neg} pares avaliados por método VLM.</p>
          <p style="margin-top:12px">As classes com mais pares (ted_bates_production_estimate,
             borriston_laboratories_inc_final_report) são formulários padronizados com
             estrutura de tabela — visualmente quase idênticos a outros tipos de formulário
             no mesmo split.</p>
        </div>
      </div>

      <h3>EER por Split — dois métodos representativos</h3>
      <p>Split 3 em vermelho. Note como ambos — um embedding semântico (Jina-v4) e
         um VLM de métrica (Qwen3-VL-8B) — apresentam desempenho visivelmente pior no split 3.</p>
      {_img_tag(b64, "max-width:800px;display:block;margin:12px auto;")}
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# Section 2: Heatmap EER por Split × Método
# ---------------------------------------------------------------------------

def build_section2() -> str:
    print("[Sec 2] Construindo heatmap EER cross-modelo...")

    records = []

    # Embedding baselines (LA-CDIP evaluation)
    for method in EMB_METHODS:
        base = EMB_LACDIP_DIR / method
        summ = _load_summary(base)
        if summ is None:
            continue
        for _, row in summ.iterrows():
            split = int(row["split"])
            if split > 4:
                continue
            records.append({
                "method": method,
                "category": "Embedding",
                "split": split,
                "eer_pct": float(row["eer_pct"]),
            })

    # VLM metric
    for method in VLM_METHODS:
        base = WORKSPACE_ROOT / "results" / "vlm_metric" / method
        summ = _load_summary(base)
        if summ is None:
            continue
        for _, row in summ.iterrows():
            split = int(row["split"])
            if split > 4:
                continue
            records.append({
                "method": method,
                "category": "VLM",
                "split": split,
                "eer_pct": float(row["eer_pct"]),
            })

    if not records:
        return '<div class="section"><h2>2. Heatmap EER</h2><p>Sem dados.</p></div>'

    df = pd.DataFrame(records)

    # Sort by EER in split 3 (ascending)
    split3_eer = df[df["split"] == 3].set_index("method")["eer_pct"]
    methods_sorted = split3_eer.sort_values().index.tolist()
    # Add any methods not in split3
    for m in df["method"].unique():
        if m not in methods_sorted:
            methods_sorted.append(m)

    all_splits = sorted(df["split"].unique())

    # Build HTML table
    header_cells = "<th style='text-align:left'>Método</th><th>Categoria</th>"
    for s in all_splits:
        if s == 3:
            header_cells += f"<th class='split3-header'>Split {s} ★</th>"
        else:
            header_cells += f"<th>Split {s}</th>"

    body_rows = ""
    for method in methods_sorted:
        cat = df[df["method"] == method]["category"].iloc[0]
        row_html = f"<td style='text-align:left;font-weight:bold'>{method}</td><td>{cat}</td>"
        for s in all_splits:
            cell = df[(df["method"] == method) & (df["split"] == s)]
            if cell.empty:
                row_html += "<td style='color:#ccc'>—</td>"
            else:
                val = cell["eer_pct"].iloc[0]
                style = _eer_color(val)
                if s == 3:
                    style += "border-left:3px solid #f0ad4e;border-right:3px solid #f0ad4e;"
                row_html += f"<td style='{style}'>{val:.1f}%</td>"
        body_rows += f"<tr>{row_html}</tr>"

    table_html = f"""
    <div style="overflow-x:auto">
    <table>
      <thead><tr style="background:#f0f0f0">{header_cells}</tr></thead>
      <tbody>{body_rows}</tbody>
    </table>
    </div>
    """

    # Legend
    legend_html = """
    <div style="margin-top:10px;display:flex;align-items:center;gap:16px;font-size:0.85em">
      <span>Gradiente:</span>
      <span style="background:rgb(0,255,51);color:#222;padding:2px 8px;border-radius:3px">0% (ótimo)</span>
      <span style="background:rgb(127,186,102);color:#222;padding:2px 8px;border-radius:3px">30%</span>
      <span style="background:rgb(255,77,26);color:#fff;padding:2px 8px;border-radius:3px">60%+ (ruim)</span>
      <span style="margin-left:16px">★ Split 3 (coluna destacada)</span>
    </div>
    """

    return f"""
    <div class="section">
      <h2>2. Comparação Cross-Modelo — EER por Split</h2>
      <p>Tabela ordenada por EER no split 3 (menor → melhor). Apenas splits 0–4
         (split 5 = conjunto de teste, excluído). A coluna do split 3 é destacada
         com borda amarela.</p>
      {table_html}
      {legend_html}
      <p class="note">Células cinzas (—) indicam que o método não foi avaliado naquele split.</p>
    </div>
    """


# ---------------------------------------------------------------------------
# Section 3: Análise de Erros por Par de Classes
# ---------------------------------------------------------------------------

def build_section3() -> str:
    print("[Sec 3] Analisando erros por par de classes no split 3...")

    key_methods = [
        ("vlm_metric", "internvl3-14b"),
        ("vlm_metric", "qwen3vl-8b"),
    ]

    # Pre-load thumbnails for all LA-CDIP split-3 classes
    if SPLIT3_IMAGES_ORIGINAL and _VAL_CSV_S3.exists():
        # On compute server: get class list from validation CSV to avoid iterating the whole NAS
        _val_df = pd.read_csv(_VAL_CSV_S3)
        all_classes = sorted(set(_val_df["class_a_name"].tolist() + _val_df["class_b_name"].tolist()))
    elif SPLIT3_IMAGES_DIR.exists():
        all_classes = sorted(d.name for d in SPLIT3_IMAGES_DIR.iterdir() if d.is_dir())
    else:
        all_classes = []
    thumbnails: dict[str, str | None] = {c: _class_thumbnail_b64(c, size=(90, 120)) for c in all_classes}

    jina_note = """
    <div class="note" style="margin-bottom:16px">
      <strong>Nota:</strong> Jina-v4 LA-CDIP EER no split 3 = <strong>6,29%</strong>
      (via W&amp;B / <code>emb_baseline_lacdip</code>). Sem CSV de pares LA-CDIP disponível —
      análise por classe feita apenas para os dois VLMs abaixo.
    </div>
    """
    sections_html = jina_note

    for category, method in key_methods:
        base = WORKSPACE_ROOT / "results" / category / method
        df = _load_pairs(base, split=3)
        if df is None:
            sections_html += f"<p class='note'>{method}: dados não disponíveis.</p>"
            continue

        # Global EER threshold for this method
        all_labels = df["is_equal"].tolist()
        all_scores = df["similarity_score"].tolist()
        try:
            fpr_g, tpr_g, thr_g = roc_curve(all_labels, all_scores)
            fnr_g = 1 - tpr_g
            eer_thr = float(thr_g[np.argmin(np.abs(fnr_g - fpr_g))])
        except Exception:
            eer_thr = float(np.median(all_scores))

        # Count FP errors per inter-class pair (negative pairs above EER threshold)
        neg_df = df[df["is_equal"] == 0].copy()
        neg_df["is_error"] = neg_df["similarity_score"] >= eer_thr

        # Group by (class_a, class_b) — normalise order so (A,B) == (B,A)
        neg_df["pair_key"] = neg_df.apply(
            lambda r: tuple(sorted([r["class_a"], r["class_b"]])), axis=1
        )
        pair_stats = (
            neg_df.groupby("pair_key")
            .agg(n_errors=("is_error", "sum"), n_pairs=("is_error", "count"),
                 avg_score=("similarity_score", "mean"))
            .reset_index()
        )
        pair_stats = pair_stats[pair_stats["n_errors"] > 0].copy()
        pair_stats = pair_stats.sort_values("n_errors", ascending=False).reset_index(drop=True)

        # Build error cards
        cards_html = ""
        for _, row in pair_stats.iterrows():
            ca, cb = row["pair_key"]
            n_err  = int(row["n_errors"])
            n_tot  = int(row["n_pairs"])
            avg_sc = float(row["avg_score"])
            pct    = n_err / n_tot * 100 if n_tot else 0

            b64a = thumbnails.get(ca)
            b64b = thumbnails.get(cb)
            label_a = ca.replace("_", " ")
            label_b = cb.replace("_", " ")

            def _thumb(b64: str | None, label: str) -> str:
                if b64:
                    return (f'<div style="text-align:center;width:110px">'
                            f'<img src="data:image/png;base64,{b64}" '
                            f'style="width:100px;height:auto;border:1px solid #ddd" />'
                            f'<div style="font-size:0.65em;color:#555;margin-top:3px;'
                            f'word-break:break-word;line-height:1.2">{label}</div></div>')
                return (f'<div style="width:100px;height:130px;background:#eee;display:flex;'
                        f'align-items:center;justify-content:center;font-size:0.65em;'
                        f'text-align:center;color:#666">{label[:30]}</div>')

            err_color = "#c0392b" if pct > 50 else ("#e67e22" if pct > 20 else "#f39c12")
            cards_html += f"""
            <div style="display:flex;align-items:center;gap:14px;padding:10px 14px;
                        border:1px solid #e0e0e0;border-radius:6px;background:#fafafa;
                        margin-bottom:8px;flex-wrap:wrap">
              {_thumb(b64a, label_a)}
              <div style="font-size:1.5em;color:#aaa">↔</div>
              {_thumb(b64b, label_b)}
              <div style="margin-left:auto;text-align:center;min-width:120px">
                <div style="font-size:1.6em;font-weight:bold;color:{err_color}">{n_err}</div>
                <div style="font-size:0.8em;color:#666">erros de {n_tot} pares</div>
                <div style="font-size:0.75em;color:#888;margin-top:2px">({pct:.0f}% | score médio {avg_sc:.1f})</div>
              </div>
            </div>"""

        total_errors = int(pair_stats["n_errors"].sum()) if not pair_stats.empty else 0
        global_eer = _eer_pct(all_labels, all_scores)
        sections_html += f"""
        <h3>{method} — EER global split 3: {global_eer:.2f}% (threshold: {eer_thr:.1f})</h3>
        <p style="font-size:0.9em;color:#555">
          {len(pair_stats)} pares de classes com pelo menos 1 erro (FP acima do limiar EER).
          Total de erros: {total_errors}.
        </p>
        <div style="max-width:700px">{cards_html if cards_html else "<p class='note'>Nenhum erro encontrado.</p>"}</div>
        """

    return f"""
    <div class="section">
      <h2>3. Erros por Par de Classes — Split 3</h2>
      <p>Para cada par de classes LA-CDIP com pelo menos um erro de falso positivo
         (pares negativos com score ≥ limiar EER global), mostramos as imagens de
         exemplo das duas classes e a contagem de erros, ordenadas do mais confuso
         ao menos confuso.</p>
      {sections_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Section 4: Training Curves vs Full Eval (W&B)
# ---------------------------------------------------------------------------

def build_section4() -> str:
    print("[Sec 4] Buscando dados de treino no W&B (por pooler)...")

    try:
        import wandb
    except ImportError:
        return '<div class="section"><h2>4. Curva de Treino vs Full Eval</h2><p>wandb não instalado.</p></div>'

    try:
        api = wandb.Api(timeout=120)
    except Exception as e:
        return f'<div class="section"><h2>4. Curva de Treino vs Full Eval</h2><p>Erro W&B: {e}</p></div>'

    # --- 1. FullEval: best (loss, EER) per pooler variant for split 3 ---
    print(f"  Buscando FullEval em {ENTITY}/{PROJECT_FULL_EVAL}...")
    try:
        full_eval_runs = list(api.runs(f"{ENTITY}/{PROJECT_FULL_EVAL}"))
        print(f"  Encontradas {len(full_eval_runs)} runs de full eval.")
    except Exception as e:
        full_eval_runs = []
        print(f"  Erro: {e}")

    # variant → {"eer": float, "best_loss": str}
    variant_best: Dict[str, Dict] = {}

    for r in full_eval_runs:
        name = r.name or ""
        variant, loss = _parse_fulleval_s3(name)
        if variant is None or loss is None:
            continue
        if variant not in POOLER_LABELS:
            continue  # skip variants not in our display set
        try:
            eer_val = float(dict(r.summary).get("val/eer", float("nan"))) * 100
        except (TypeError, ValueError):
            continue
        if np.isnan(eer_val):
            continue
        if variant not in variant_best or eer_val < variant_best[variant]["eer"]:
            variant_best[variant] = {"eer": eer_val, "best_loss": loss}

    # --- 2. Training history for each variant's best loss ---
    print(f"  Buscando treino em {ENTITY}/{PROJECT_SPRINT3B}...")
    try:
        training_runs = list(api.runs(f"{ENTITY}/{PROJECT_SPRINT3B}"))
        print(f"  Encontradas {len(training_runs)} runs de treino.")
    except Exception as e:
        training_runs = []
        print(f"  Erro: {e}")

    # loss → list of (epoch, eer_pct) for split 3
    loss_epoch_eer: Dict[str, List[Tuple[float, float]]] = {}

    needed_losses = {info["best_loss"] for info in variant_best.values()}

    for r in training_runs:
        name = r.name or ""
        nl   = name.lower()
        if not name.startswith("Sprint3b_"):
            continue
        if "_noinit_" in nl:
            continue
        m_sp = re.search(r"_S(\d+)_", name)
        if not m_sp or int(m_sp.group(1)) != 3:
            continue

        loss = next((l for l in KNOWN_LOSSES if l in nl), None)
        if not loss or loss not in needed_losses:
            continue

        is_fase2  = "fase2" in nl
        ep_offset = FASE1_TOTAL_EPOCHS if is_fase2 else 0

        try:
            hist = r.history(keys=["val/eer", "epoch"], pandas=True)
        except Exception as exc:
            print(f"    Aviso: {name[:60]} — {exc}")
            continue

        if hist is None or hist.empty or "val/eer" not in hist.columns:
            continue
        hist = hist.dropna(subset=["val/eer"]).reset_index(drop=True)
        if hist.empty:
            continue

        if "epoch" in hist.columns and hist["epoch"].notna().any():
            ep_series = hist["epoch"].ffill().astype(float) + ep_offset + 1
        else:
            ep_series = pd.Series([ep_offset + i + 1 for i in range(len(hist))], dtype=float)

        if loss not in loss_epoch_eer:
            loss_epoch_eer[loss] = []
        for ep, eer in zip(ep_series, hist["val/eer"]):
            loss_epoch_eer[loss].append((float(ep), float(eer) * 100))

    # --- 3. Plot ---
    variants_ordered = [v for v in POOLER_LABELS if v in variant_best]

    if not variants_ordered:
        html_content = '<p class="note">Nenhum dado de FullEval encontrado para o split 3.</p>'
    else:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        max_ep_global = 0.0

        for color, variant in zip(PALETTE, variants_ordered):
            info      = variant_best[variant]
            fe_eer    = info["eer"]
            best_loss = info["best_loss"]
            label     = POOLER_LABELS[variant]
            loss_label = LOSS_LABELS.get(best_loss, best_loss)

            if best_loss in loss_epoch_eer:
                pairs  = sorted(loss_epoch_eer[best_loss], key=lambda x: x[0])
                epochs = [p[0] for p in pairs]
                eers   = [p[1] for p in pairs]
                max_ep_global = max(max_ep_global, max(epochs, default=0))
                ax.plot(epochs, eers, "o-", color=color, linewidth=1.8,
                        markersize=3, alpha=0.7,
                        label=f"{label}  [{loss_label}]")
            else:
                max_ep_global = max(max_ep_global, float(FASE1_TOTAL_EPOCHS + 5))

            ax.axhline(fe_eer, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
            ax.text(max_ep_global + 0.2, fe_eer, f" {fe_eer:.1f}%",
                    color=color, va="center", fontsize=8)

        ax.axvline(FASE1_TOTAL_EPOCHS + 0.5, color="#888", linestyle=":", linewidth=1)
        ylim = ax.get_ylim()
        ax.text(FASE1_TOTAL_EPOCHS + 0.6, ylim[1] * 0.97, "↑ Fase 2",
                fontsize=8, color="#888", va="top")
        ax.set_xlabel("Época (Fase 1: 1–10, Fase 2: 11+)")
        ax.set_ylabel("Subset val/eer (%)")
        ax.set_title(
            "Split 3 — Curva de Treino por Pooler Final\n"
            "(curva sólida = subset val; tracejado = FullEval EER | [colchetes] = melhor loss)",
            fontsize=11,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        b64 = _b64_png(fig)

        # Summary table
        gap_rows = ""
        for variant in variants_ordered:
            info       = variant_best[variant]
            fe_eer     = info["eer"]
            best_loss  = info["best_loss"]
            loss_label = LOSS_LABELS.get(best_loss, best_loss)
            best_sub   = (min(e for _, e in loss_epoch_eer[best_loss])
                          if best_loss in loss_epoch_eer else None)
            if best_sub is not None:
                gap = fe_eer - best_sub
                gap_color = "#155724" if gap < 0 else "#721c24"
                gap_str = f'<span style="color:{gap_color}">{"+" if gap >= 0 else ""}{gap:.2f} pp</span>'
                sub_str = f"{best_sub:.2f}%"
            else:
                gap_str = "—"
                sub_str = "—"
            gap_rows += (
                f"<tr><td style='text-align:left'>{POOLER_LABELS[variant]}</td>"
                f"<td style='text-align:left;color:#555'>{loss_label}</td>"
                f"<td>{sub_str}</td><td><strong>{fe_eer:.2f}%</strong></td>"
                f"<td>{gap_str}</td></tr>"
            )

        gap_table = f"""
        <table style="max-width:800px;margin-top:12px">
          <thead><tr>
            <th style="text-align:left">Pooler</th>
            <th style="text-align:left">Melhor Loss</th>
            <th>Melhor Subset val</th>
            <th>Full Eval EER</th>
            <th>Gap (Full − Subset)</th>
          </tr></thead>
          <tbody>{gap_rows}</tbody>
        </table>
        <p class="note">Gap positivo = full eval é pior que subset val.
           A mesma curva de treino pode corresponder a vários poolers que compartilham a melhor loss.</p>
        """

        html_content = f"""
        {_img_tag(b64, "max-width:950px;display:block;margin:12px auto;")}
        <h3>Tabela: Melhor Loss por Pooler — Split 3</h3>
        {gap_table}
        """

    aug_warning = """
    <div style="background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;
                padding:12px 16px;margin-bottom:16px;font-size:0.9em">
      <strong>✓ Protocolo de dados verificado</strong> — todos os splits (incluindo S3)
      foram treinados e avaliados exclusivamente com imagens originais LA-CDIP
      (via <code>run_sprint3b_split5_staged_lacdip.py --base-image-dir /mnt/nas/joaopaulo/LA-CDIP/data</code>
      e CSVs <code>sprint3_zsl_val_{N}_train_excl_5</code>). O dataset aumentado
      (<code>final_split3</code>) foi criado para um experimento distinto e
      <strong>não foi usado no treino nem na avaliação definitiva</strong> deste paper.
      Os EERs finais são comparáveis entre todos os splits.
    </div>
    """

    return f"""
    <div class="section">
      <h2>4. Curva de Treino — por Pooler Final (Split 3)</h2>
      {aug_warning}
      <p>Para cada pooler final, a curva sólida mostra o histórico de treino da loss
         que produziu o menor FullEval EER para aquele pooler no split 3.
         A linha tracejada mostra o EER final no conjunto completo de validação
         (imagens originais).</p>
      {html_content}
    </div>
    """


# ---------------------------------------------------------------------------
# Section 5: Similarity Score Distribution
# ---------------------------------------------------------------------------

def build_section5() -> str:
    print("[Sec 5] Analisando distribuições de similarity score...")

    focus_methods = [
        ("emb_baseline", "pixel-cosine"),
        ("emb_baseline", "internvl3-out"),
    ]

    html_parts = []

    for category, method in focus_methods:
        base = WORKSPACE_ROOT / "results" / category / method

        # Load split 3 data
        df3 = _load_pairs(base, split=3)

        # Load splits 0–2 combined as "outros"
        others = []
        for s in [0, 1, 2]:
            df_s = _load_pairs(base, split=s)
            if df_s is not None:
                df_s["split_id"] = s
                others.append(df_s)

        if df3 is None and not others:
            html_parts.append(f"<p class='note'>{method}: dados não disponíveis.</p>")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{method} — Distribuição de Similarity Scores", fontsize=12, y=1.01)

        datasets = []
        if df3 is not None:
            datasets.append(("Split 3 (memo/pres/quest/resume)", df3))
        if others:
            df_others = pd.concat(others, ignore_index=True)
            datasets.append(("Splits 0–2 (outros)", df_others))

        for col_idx, (label, df) in enumerate(datasets):
            pos_scores = df[df["is_equal"] == 1]["similarity_score"].dropna().values
            neg_scores = df[df["is_equal"] == 0]["similarity_score"].dropna().values

            # Histogram
            ax = axes[0, col_idx]
            all_scores = np.concatenate([pos_scores, neg_scores])
            bins = np.linspace(all_scores.min(), all_scores.max(), 50) if len(all_scores) > 0 else 50
            ax.hist(pos_scores, bins=bins, alpha=0.6, color="#2ecc71", label="Positivos (mesma classe)", density=True)
            ax.hist(neg_scores, bins=bins, alpha=0.6, color="#e74c3c", label="Negativos (classes diferentes)", density=True)
            eer = _eer_pct(
                [1] * len(pos_scores) + [0] * len(neg_scores),
                list(pos_scores) + list(neg_scores)
            ) if (len(pos_scores) > 0 and len(neg_scores) > 0) else float("nan")
            ax.set_title(f"{label}\nEER={eer:.1f}%", fontsize=10)
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Densidade")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # KDE-like overlapping region
            ax2 = axes[1, col_idx]
            if len(pos_scores) > 1 and len(neg_scores) > 1:
                from scipy import stats as scipy_stats
                try:
                    kde_pos = scipy_stats.gaussian_kde(pos_scores)
                    kde_neg = scipy_stats.gaussian_kde(neg_scores)
                    x_range = np.linspace(all_scores.min(), all_scores.max(), 300)
                    p_pos = kde_pos(x_range)
                    p_neg = kde_neg(x_range)
                    ax2.plot(x_range, p_pos, color="#2ecc71", linewidth=2, label="Positivos (KDE)")
                    ax2.plot(x_range, p_neg, color="#e74c3c", linewidth=2, label="Negativos (KDE)")
                    overlap = np.minimum(p_pos, p_neg)
                    ax2.fill_between(x_range, overlap, alpha=0.4, color="#9b59b6", label="Sobreposição")
                    overlap_area = np.trapezoid(overlap, x_range) if hasattr(np, "trapezoid") else np.trapz(overlap, x_range)
                    ax2.set_title(f"KDE — Área de sobreposição: {overlap_area:.3f}", fontsize=10)
                except Exception:
                    ax2.text(0.5, 0.5, "KDE não disponível", ha="center", va="center",
                             transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, "Dados insuficientes para KDE",
                         ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xlabel("Similarity Score")
            ax2.set_ylabel("Densidade")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

        fig.tight_layout()
        b64 = _b64_png(fig)
        html_parts.append(f"""
        <h3>{method}</h3>
        {_img_tag(b64, "max-width:950px;display:block;margin:12px auto;")}
        """)

    content = "\n".join(html_parts) if html_parts else "<p>Sem dados disponíveis.</p>"

    return f"""
    <div class="section">
      <h2>5. Distribuição de Similarity Scores — Split 3 vs Outros</h2>
      <p>Linha superior: histograma de scores para pares positivos (verde) e negativos
         (vermelho). Linha inferior: estimativa de densidade (KDE) com área de sobreposição
         destacada em roxo. Quanto maior a sobreposição, maior o EER.</p>
      {content}
    </div>
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Análise diagnóstica do split 3")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Caminho do HTML de saída")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 60)
    print("Análise Diagnóstica — Split 3")
    print("=" * 60)

    sections = [
        build_section1(),
        build_section2(),
        build_section3(),
        build_section4(),
        build_section5(),
    ]

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Análise do Split 3 — CaVL-Doc LA-CDIP</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>Análise Diagnóstica do Split 3 — CaVL-Doc LA-CDIP</h1>
  <p style="color:#888;font-size:0.9em">
    Gerado em {now} · Por que o split 3 é o mais difícil?
  </p>
  <div style="background:#fff3cd;border:1px solid #f0ad4e;border-radius:6px;
              padding:12px 16px;margin-bottom:24px;font-size:0.9em">
    <strong>Contexto:</strong> Split 3 da base LA-CDIP contém
    <strong>24 classes</strong> de documentos tabagistas (formulários, relatórios,
    estimativas de produção, etc.) com layouts visualmente muito semelhantes entre si.
    Este relatório investiga por que este split apresenta os maiores EERs na maioria
    dos métodos avaliados.
  </div>
  {"".join(sections)}
  <p style="color:#bbb;font-size:0.8em;text-align:center;margin-top:40px">
    CaVL-Doc · Análise gerada automaticamente · {now}
  </p>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\nSalvo em: {output_path}")
    print("Concluído.")


if __name__ == "__main__":
    main()
