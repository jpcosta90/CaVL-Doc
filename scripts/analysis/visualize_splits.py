#!/usr/bin/env python3
"""
Visualiza a estrutura dos ZSL splits: classes, contagens de imagens e resultados
de baselines por split — para apoiar a decisão de configuração do treino definitivo.

Uso:
  python scripts/analysis/visualize_splits.py \
    --splits-dir data/generated_splits \
    --baseline-dir results/emb_baseline \
    --output split_structure.html
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ── Leitura ─────────────────────────────────────────────────────────────────

def read_split(split_dir: Path):
    """Retorna {class: set(image_paths)} para um split."""
    csv_path = split_dir / "train_pairs.csv"
    if not csv_path.exists():
        return {}
    by_class = defaultdict(set)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_class[row["class_a_name"]].add(row["file_a_path"])
            by_class[row["class_b_name"]].add(row["file_b_path"])
    return dict(by_class)


RESULT_CATEGORIES = {
    "emb_baseline_lacdip": "Embedding Baseline (LA-CDIP)",
    "vlm_metric":          "VLM Metric (LA-CDIP)",
}


def _compute_eer_from_pairs(pairs_csv: Path) -> dict:
    """Calcula EER a partir de um arquivo de pares com similarity_score."""
    import numpy as np
    scores, labels = [], []
    with open(pairs_csv) as f:
        for row in csv.DictReader(f):
            scores.append(float(row["similarity_score"]))
            labels.append(int(row["is_equal"]))
    if not scores:
        return {}
    scores, labels = np.array(scores), np.array(labels)
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_eer, best_thr = 1.0, thresholds[0]
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tp = ((preds == 1) & (labels == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        eer = (fpr + fnr) / 2
        if eer < best_eer:
            best_eer, best_thr = eer, thr
    return {"eer_pct": round(best_eer * 100, 2), "n_pairs": len(scores)}


def read_baselines(results_dir: Path):
    """Lê vlm_metric (LA-CDIP). Para modelos sem summary.csv, computa EER dos pair files.
    Retorna lista de dicts {category, model, data: {split_id: {eer_pct, n_pairs}}}."""
    entries = []
    for subdir_name, category_label in RESULT_CATEGORIES.items():
        category_dir = results_dir / subdir_name
        if not category_dir.exists():
            continue
        for model_dir in sorted(category_dir.iterdir()):
            model_data = {}
            summary = model_dir / "summary.csv"
            if summary.exists():
                with open(summary) as f:
                    for row in csv.DictReader(f):
                        raw = row["split"]
                        try:
                            sid = int(raw)
                        except ValueError:
                            continue
                        model_data[sid] = {
                            "eer_pct": float(row["eer_pct"]),
                            "n_pairs": int(row["n_pairs"]),
                        }
            else:
                for pair_file in sorted(model_dir.glob("split*_pairs.csv")):
                    name = pair_file.stem  # e.g. "split0_pairs"
                    try:
                        sid = int(name.replace("split", "").replace("_pairs", ""))
                    except ValueError:
                        continue
                    result = _compute_eer_from_pairs(pair_file)
                    if result:
                        model_data[sid] = result
            if model_data:
                entries.append({
                    "category": category_label,
                    "model": model_dir.name,
                    "data": model_data,
                })
    return entries


# ── HTML ─────────────────────────────────────────────────────────────────────

COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
LIGHT  = ["#fdecea", "#fef0e6", "#fefce8", "#eafaf1", "#eaf4fb", "#f5eef8"]
N_SPLITS = 6


def _eer_color(eer_pct: float) -> str:
    if eer_pct < 25:
        return "#27ae60"
    if eer_pct < 35:
        return "#e67e22"
    return "#e74c3c"


def render_html(splits: dict, baselines: dict, n_splits: int) -> str:
    all_classes = sorted(set.union(*[set(s.keys()) for s in splits.values()]))
    absent_from = {}
    for cls in all_classes:
        for i in range(n_splits):
            if cls not in splits.get(i, {}):
                absent_from[cls] = i
                break

    classes_sorted = sorted(all_classes, key=lambda c: (absent_from.get(c, -1), c))

    # ── Baseline section ────────────────────────────────────────────────────
    baseline_rows = ""
    prev_cat = None
    for entry in baselines:
        if entry["category"] != prev_cat:
            baseline_rows += (
                f"<tr><td colspan='{n_splits + 1}' class='bl-section-label'>"
                f"{entry['category']}</td></tr>\n"
            )
            prev_cat = entry["category"]
        data = entry["data"]
        cells = f"<td class='model-name'>{entry['model']}</td>"
        for i in range(n_splits):
            if i in data:
                eer = data[i]["eer_pct"]
                color = _eer_color(eer)
                cells += f"<td style='color:{color};font-weight:600'>{eer:.1f}%</td>"
            else:
                cells += "<td style='color:#adb5bd'>—</td>"
        baseline_rows += f"<tr>{cells}</tr>\n"

    baseline_headers = "".join(
        f"<th style='background:{COLORS[i]};color:white'>Split {i}</th>"
        for i in range(n_splits)
    )

    # ── Class table ─────────────────────────────────────────────────────────
    class_rows = ""
    prev_absent = None
    for cls in classes_sorted:
        ab = absent_from.get(cls, -1)
        if ab != prev_absent:
            class_rows += (
                f"<tr><td colspan='{n_splits * 2 + 1}' class='section-label' "
                f"style='color:{COLORS[ab]}'>"
                f"⭐ Novel do Split {ab} — 24 classes ausentes do treino desse split"
                f"</td></tr>\n"
            )
            prev_absent = ab

        cells = f"<td class='cls-name' title='{cls}'>{cls}</td>"
        for i in range(n_splits):
            split_data = splits.get(i, {})
            if cls in split_data:
                n_img = len(split_data[cls])
                cells += (
                    f"<td class='present'><span class='check'>✓</span>"
                    f"<span class='img-count'>{n_img}</span></td>"
                )
            else:
                cells += (
                    f"<td class='novel-cell' style='background:{LIGHT[i]};"
                    f"border-left:3px solid {COLORS[ab]}'>⭐</td>"
                )
        class_rows += f"<tr>{cells}</tr>\n"

    col_headers = "".join(
        f"<th style='background:{COLORS[i]};color:white'>"
        f"Split {i}<br><small>{len(splits.get(i,{}))} classes · "
        f"{sum(len(v) for v in splits.get(i,{}).values())} imgs</small></th>"
        for i in range(n_splits)
    )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>ZSL Splits — Estrutura e Baselines</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f8f9fa; color: #212529; font-size: 13px; }}
  .page {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}

  h1 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin: 0 0 12px; }}
  .subtitle {{ color: #6c757d; font-size: .85rem; margin-bottom: 24px; }}

  .card {{ background: white; border: 1px solid #dee2e6; border-radius: 8px;
           padding: 16px 20px; margin-bottom: 20px; }}

  .insight ul {{ padding-left: 18px; line-height: 1.9; color: #495057; }}
  .insight strong {{ color: #212529; }}

  .summary-grid {{ display: grid; grid-template-columns: repeat({n_splits}, 1fr);
                   gap: 10px; margin-bottom: 20px; }}
  .summary-card {{ border-radius: 8px; padding: 12px; color: white; text-align: center; }}
  .summary-card .num {{ font-size: 1.6rem; font-weight: 700; }}
  .summary-card .lbl {{ font-size: .72rem; opacity: .9; margin-top: 2px; line-height: 1.4; }}

  /* baseline table */
  .bl-wrap {{ overflow-x: auto; }}
  .bl-table {{ border-collapse: collapse; width: 100%; font-size: .82rem; }}
  .bl-table th {{ padding: 8px 12px; text-align: center; font-size: .78rem; }}
  .bl-table th:first-child {{ text-align: left; background: #343a40; color: white; min-width: 160px; }}
  .bl-table td {{ padding: 7px 12px; border-bottom: 1px solid #f1f3f5; text-align: center; }}
  .bl-table td.model-name {{ text-align: left; font-weight: 500; font-family: monospace; }}
  .bl-table tr:hover td {{ background: #f8f9fa; }}
  .bl-section-label {{ font-size: .68rem; font-weight: 700; letter-spacing: .07em;
                        text-transform: uppercase; color: #6c757d; padding: 8px 12px 3px;
                        background: #f8f9fa; border-bottom: 1px solid #dee2e6; }}

  /* class table */
  .tbl-wrap {{ overflow-x: auto; border-radius: 8px;
               box-shadow: 0 1px 4px rgba(0,0,0,.08); max-height: 70vh; overflow-y: auto; }}
  table.cls-table {{ border-collapse: collapse; width: 100%; background: white; font-size: .78rem; }}
  table.cls-table thead {{ position: sticky; top: 0; z-index: 2; }}
  table.cls-table thead th {{ padding: 9px 6px; text-align: center; font-size: .73rem; }}
  table.cls-table thead th:first-child {{ text-align: left; background: #343a40;
                                          color: white; min-width: 200px; padding-left: 10px; }}
  table.cls-table tbody td {{ border-bottom: 1px solid #f1f3f5; vertical-align: middle; padding: 4px 6px; }}
  td.cls-name {{ font-family: monospace; color: #495057; max-width: 200px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-left: 10px; }}
  td.present {{ text-align: center; }}
  .check {{ color: #28a745; font-weight: 700; margin-right: 2px; }}
  .img-count {{ color: #6c757d; font-size: .72rem; }}
  td.novel-cell {{ text-align: center; font-size: .8rem; }}
  tr:hover td {{ background: #f1f3f5 !important; }}
  .section-label {{ font-size: .68rem; font-weight: 700; letter-spacing: .07em;
                    text-transform: uppercase; padding: 6px 10px 3px;
                    background: #f8f9fa; border-bottom: 1px solid #dee2e6; }}
</style>
</head>
<body>
<div class="page">
  <h1>ZSL Splits — Estrutura de Classes e Resultados de Baseline</h1>
  <p class="subtitle">
    {len(all_classes)} classes únicas · {n_splits} splits ·
    cada classe presente em exatamente 5 splits · 24 classes "novel" por split
  </p>

  <div class="summary-grid">
    {"".join(
        f'<div class="summary-card" style="background:{COLORS[i]}">'
        f'<div class="num">{len(splits.get(i,{}))}</div>'
        f'<div class="lbl">classes em treino<br>Split {i}<br>'
        f'{sum(len(v) for v in splits.get(i,{}).values())} imgs únicas</div>'
        f'</div>'
        for i in range(n_splits)
    )}
  </div>

  <div class="card insight">
    <h2>📌 Como interpretar os splits</h2>
    <ul>
      <li>O dataset tem <strong>{len(all_classes)} classes únicas</strong> no total.</li>
      <li>Cada split contém <strong>120 classes para treino</strong>,
          excluindo 24 classes "novel" diferentes — marcadas com ⭐.</li>
      <li>Cada classe aparece em <strong>exatamente 5 dos {n_splits} splits</strong>.</li>
      <li><strong>Combinar splits aumenta pares/imagens por classe</strong>, não o número de classes.</li>
      <li>Para validação ZSL real num modelo definitivo, as classes "novel" do split escolhido
          <strong>não podem aparecer em nenhum split de treino</strong>.</li>
      <li>Se todos os {n_splits} splits forem usados para treino,
          <strong>não sobra nenhuma classe novel</strong> para validação ZSL.</li>
    </ul>
  </div>

  <div class="card">
    <h2>📊 EER por Split — Baselines (sem fine-tuning)</h2>
    <div class="bl-wrap">
      <table class="bl-table">
        <thead><tr><th>Modelo</th>{baseline_headers}</tr></thead>
        <tbody>{baseline_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="tbl-wrap">
    <table class="cls-table">
      <thead>
        <tr>
          <th>Classe</th>{col_headers}
        </tr>
      </thead>
      <tbody>
        {class_rows}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-dir",   default="data/generated_splits",
                   help="Diretório com zsl_split_0 … zsl_split_N")
    p.add_argument("--baseline-dir", default="results",
                   help="Diretório raiz de resultados (contém emb_baseline, vlm_baseline, vlm_metric)")
    p.add_argument("--output",       default="split_structure.html",
                   help="Arquivo HTML de saída")
    p.add_argument("--n-splits",     type=int, default=6)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    splits_dir   = (root / args.splits_dir).resolve()
    baseline_dir = (root / args.baseline_dir).resolve()
    output       = (root / args.output).resolve()

    print("Lendo splits...")
    splits = {}
    for i in range(args.n_splits):
        d = splits_dir / f"zsl_split_{i}"
        splits[i] = read_split(d)
        print(f"  Split {i}: {len(splits[i])} classes, "
              f"{sum(len(v) for v in splits[i].values())} imgs únicas")

    print("Lendo baselines...")
    baselines = read_baselines(baseline_dir)
    for entry in baselines:
        print(f"  [{entry['category']}] {entry['model']}: splits {sorted(entry['data'].keys())}")

    print("Gerando HTML...")
    html = render_html(splits, baselines, args.n_splits)
    output.write_text(html, encoding="utf-8")
    print(f"✅ Salvo em: {output}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
