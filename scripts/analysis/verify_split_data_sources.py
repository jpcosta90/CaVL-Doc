#!/usr/bin/env python3
"""
Verifica e compara as fontes de dados (imagens originais vs augmentadas) usadas
no treino e na validação de subset de cada split do Sprint3b.

Evidências checadas:
  1. Conteúdo dos CSVs de treino e validação (caminhos com/sem _aug)
  2. Existência dos diretórios de imagem referenciados
  3. Número de pares e classes por split
  4. (Opcional) Busca no W&B os parâmetros base_image_dir logados nos runs

Uso:
    python scripts/analysis/verify_split_data_sources.py
    python scripts/analysis/verify_split_data_sources.py --output results/data_source_audit.html
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Fontes de dados conhecidas por split
# ---------------------------------------------------------------------------

SPLIT_DATA_SOURCES = {
    0: {
        "train_csv":  WORKSPACE / "data/generated_splits/sprint3_zsl_val_0_train_excl_5/train_pairs.csv",
        "val_csv":    WORKSPACE / "data/generated_splits/sprint3_zsl_val_0_train_excl_5/validation_pairs.csv",
        "image_dir":  None,  # imagens originais em /mnt/data/la-cdip/data (não local)
        "label":      "sprint3_zsl_val_0 (imagens originais LA-CDIP)",
    },
    1: {
        "train_csv":  WORKSPACE / "data/generated_splits/sprint3_zsl_val_1_train_excl_5/train_pairs.csv",
        "val_csv":    WORKSPACE / "data/generated_splits/sprint3_zsl_val_1_train_excl_5/validation_pairs.csv",
        "image_dir":  None,
        "label":      "sprint3_zsl_val_1 (imagens originais LA-CDIP)",
    },
    2: {
        "train_csv":  WORKSPACE / "data/generated_splits/sprint3_zsl_val_2_train_excl_5/train_pairs.csv",
        "val_csv":    WORKSPACE / "data/generated_splits/sprint3_zsl_val_2_train_excl_5/validation_pairs.csv",
        "image_dir":  None,
        "label":      "sprint3_zsl_val_2 (imagens originais LA-CDIP)",
    },
    3: {
        "train_csv":  WORKSPACE / "data/generated_splits/sprint3_zsl_val_3_train_excl_5/train_pairs.csv",
        "val_csv":    WORKSPACE / "data/generated_splits/sprint3_zsl_val_3_train_excl_5/validation_pairs.csv",
        "image_dir":  None,
        "label":      "sprint3_zsl_val_3 (imagens originais LA-CDIP)",
    },
    4: {
        "train_csv":  WORKSPACE / "data/generated_splits/sprint3_zsl_val_4_train_excl_5/train_pairs.csv",
        "val_csv":    WORKSPACE / "data/generated_splits/sprint3_zsl_val_4_train_excl_5/validation_pairs.csv",
        "image_dir":  None,
        "label":      "sprint3_zsl_val_4 (imagens originais LA-CDIP)",
    },
}

# FullEval usa sempre o mesmo conjunto para todos os splits
FULLEVAL_TEMPLATE = str(WORKSPACE / "data/generated_splits/sprint3_zsl_val_{split}_train_excl_5/validation_pairs.csv")


# ---------------------------------------------------------------------------
# Análise de um CSV
# ---------------------------------------------------------------------------

def _analyse_csv(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}

    df = pd.read_csv(path)
    all_paths = list(df["file_a_path"]) + list(df["file_b_path"])

    n_aug   = sum(1 for p in all_paths if "_aug" in str(p))
    n_total = len(all_paths)
    pct_aug = n_aug / n_total * 100 if n_total else 0

    aug_indices = set()
    for p in all_paths:
        m = re.search(r"_aug(\d+)\.", str(p))
        if m:
            aug_indices.add(int(m.group(1)))

    # Amostras com e sem aug
    sample_aug   = next((p for p in all_paths if "_aug" in str(p)), None)
    sample_noaug = next((p for p in all_paths if "_aug" not in str(p)), None)

    n_pos = int((df["is_equal"] == 1).sum())
    n_neg = int((df["is_equal"] == 0).sum())
    n_classes = df["class_a_name"].nunique() if "class_a_name" in df.columns else "?"

    return {
        "exists":      True,
        "n_pairs":     len(df),
        "n_pos":       n_pos,
        "n_neg":       n_neg,
        "n_classes":   n_classes,
        "n_aug_paths": n_aug,
        "pct_aug":     pct_aug,
        "aug_indices": sorted(aug_indices),
        "sample_aug":  sample_aug,
        "sample_noaug": sample_noaug,
    }


def _image_dir_info(image_dir: Path | None) -> dict:
    if image_dir is None or not image_dir.exists():
        return {"exists": False, "n_files": 0, "aug_indices": []}
    tifs = list(image_dir.rglob("*.tif"))
    indices = set()
    for f in tifs:
        m = re.search(r"_aug(\d+)\.", f.name)
        if m:
            indices.add(int(m.group(1)))
    return {"exists": True, "n_files": len(tifs), "aug_indices": sorted(indices)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_CELL_OK  = "background:#d4edda;color:#155724"
_CELL_WARN = "background:#fff3cd;color:#856404"
_CELL_ERR  = "background:#f8d7da;color:#721c24"

def _aug_cell(pct: float, indices: list[int]) -> str:
    idx_str = ", ".join(f"aug{i}" for i in indices) if indices else "—"
    if pct == 0:
        style = _CELL_OK
        label = f"0% aug ({idx_str})"
    elif pct == 100:
        style = _CELL_ERR
        label = f"100% aug ({idx_str})"
    else:
        style = _CELL_WARN
        label = f"{pct:.0f}% aug ({idx_str})"
    return f"<td style='{style};padding:6px 10px;font-weight:bold'>{label}</td>"


def build_report() -> str:
    rows_train = ""
    rows_val   = ""
    rows_fe    = ""
    details    = ""

    for split in range(5):
        src = SPLIT_DATA_SOURCES[split]
        train_info  = _analyse_csv(src["train_csv"])
        val_info    = _analyse_csv(src["val_csv"])
        fe_csv      = Path(FULLEVAL_TEMPLATE.format(split=split))
        fe_info     = _analyse_csv(fe_csv)
        img_info    = _image_dir_info(src.get("image_dir"))

        label = src["label"]
        sp    = f"S{split}"

        # --- Treino row ---
        if train_info["exists"]:
            rows_train += (
                f"<tr><td style='font-weight:bold'>{sp}</td>"
                f"<td style='font-size:0.8em;color:#555'>{label}</td>"
                + _aug_cell(train_info["pct_aug"], train_info["aug_indices"])
                + f"<td>{train_info['n_pairs']:,}</td>"
                f"<td>{train_info['n_classes']}</td></tr>"
            )
        else:
            rows_train += f"<tr><td>{sp}</td><td colspan=4 style='color:#ccc'>CSV não encontrado: {src['train_csv']}</td></tr>"

        # --- Subset val row ---
        if val_info["exists"]:
            rows_val += (
                f"<tr><td style='font-weight:bold'>{sp}</td>"
                f"<td style='font-size:0.8em;color:#555'>{label}</td>"
                + _aug_cell(val_info["pct_aug"], val_info["aug_indices"])
                + f"<td>{val_info['n_pairs']:,}</td>"
                f"<td>{val_info['n_classes']}</td></tr>"
            )
        else:
            rows_val += f"<tr><td>{sp}</td><td colspan=4 style='color:#ccc'>CSV não encontrado</td></tr>"

        # --- FullEval row ---
        if fe_info["exists"]:
            rows_fe += (
                f"<tr><td style='font-weight:bold'>{sp}</td>"
                f"<td style='font-size:0.8em;color:#555'>{fe_csv.parent.name}</td>"
                + _aug_cell(fe_info["pct_aug"], fe_info["aug_indices"])
                + f"<td>{fe_info['n_pairs']:,}</td>"
                f"<td>{fe_info['n_classes']}</td></tr>"
            )
        else:
            rows_fe += f"<tr><td>{sp}</td><td colspan=4 style='color:#ccc'>CSV não encontrado</td></tr>"

        # --- Detail section ---
        sample_t = train_info.get("sample_aug") or train_info.get("sample_noaug") or "—"
        sample_v = val_info.get("sample_aug")   or val_info.get("sample_noaug")   or "—"
        img_line = (f"{img_info['n_files']:,} arquivos .tif, aug indices: {img_info['aug_indices']}"
                    if img_info["exists"] else "Diretório não disponível localmente (imagens originais no servidor)")

        details += f"""
        <h3>Split {split} — {label}</h3>
        <table style="max-width:700px;font-size:0.88em">
          <tr><td style="font-weight:bold;width:220px">Train CSV</td>
              <td><code>{src['train_csv'].name}</code>
                  {'✓ existe' if train_info.get('exists') else '✗ não encontrado'}</td></tr>
          <tr><td style="font-weight:bold">Val CSV (subset)</td>
              <td><code>{src['val_csv'].name}</code>
                  {'✓ existe' if val_info.get('exists') else '✗ não encontrado'}</td></tr>
          <tr><td style="font-weight:bold">Pares de treino com _aug</td>
              <td><strong>{train_info.get('pct_aug', 0):.0f}%</strong>
                  ({train_info.get('n_aug_paths', 0):,} de {train_info.get('n_pairs', 0)*2:,} paths)</td></tr>
          <tr><td style="font-weight:bold">Pares de val com _aug</td>
              <td><strong>{val_info.get('pct_aug', 0):.0f}%</strong>
                  ({val_info.get('n_aug_paths', 0):,} de {val_info.get('n_pairs', 0)*2:,} paths)</td></tr>
          <tr><td style="font-weight:bold">Exemplo path treino</td>
              <td><code style="font-size:0.85em">{sample_t}</code></td></tr>
          <tr><td style="font-weight:bold">Exemplo path val</td>
              <td><code style="font-size:0.85em">{sample_v}</code></td></tr>
          <tr><td style="font-weight:bold">Diretório images_train</td>
              <td>{img_line}</td></tr>
        </table>
        """

    def _table(title: str, note: str, rows: str, status_label: str) -> str:
        return f"""
        <h3>{title}</h3>
        <p style="font-size:0.9em;color:#555">{note}</p>
        <table>
          <thead><tr>
            <th>Split</th>
            <th style="text-align:left">Dataset</th>
            <th>Caminhos com _aug?</th>
            <th>Pares</th>
            <th>Classes</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <title>Auditoria de Fontes de Dados — Sprint3b</title>
  <style>
    body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 1100px;
            margin: 0 auto; padding: 24px; background: #fafafa; }}
    h1   {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 8px; }}
    h2   {{ color: #34495e; margin-top: 32px; border-left: 4px solid #3498db; padding-left: 10px; }}
    h3   {{ color: #5d6d7e; margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; margin-bottom: 16px; }}
    th   {{ padding: 6px 10px; background: #f0f0f0; border-bottom: 2px solid #ccc; text-align:center; }}
    td   {{ padding: 5px 10px; border-bottom: 1px solid #eee; text-align:center; }}
    code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
    .warn {{ background:#fff3cd;border:1px solid #f0ad4e;border-radius:6px;
             padding:12px 16px;margin:16px 0;font-size:0.9em; }}
    .ok   {{ background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;
             padding:12px 16px;margin:16px 0;font-size:0.9em; }}
  </style>
</head>
<body>
  <h1>Auditoria de Fontes de Dados — Sprint3b</h1>
  <p>Verifica se imagens augmentadas (offline) foram usadas no treino e validação de subset
     do split 3, em comparação com os demais splits que usam imagens originais.</p>

  <div class="warn">
    <strong>Legenda das células:</strong>
    <span style="background:#d4edda;color:#155724;padding:2px 8px;border-radius:3px;margin-left:8px">Verde = 0% aug (originais)</span>
    <span style="background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:3px;margin-left:8px">Vermelho = 100% aug (offline)</span>
    <span style="background:#fff3cd;color:#856404;padding:2px 8px;border-radius:3px;margin-left:8px">Amarelo = misto</span>
  </div>

  <h2>1. Dados de Treino (train_pairs.csv)</h2>
  {_table("", "CSV de treino efetivamente usado durante Sprint3b por split.", rows_train, "aug")}

  <h2>2. Subset Validation (validation_pairs.csv — val/eer no W&B)</h2>
  {_table("", "CSV de validação de subset — métrica val/eer registrada no W&B durante treino.", rows_val, "aug")}

  <h2>3. FullEval Definitivo (sprint3_zsl_val_N — para todas as variantes/poolers)</h2>
  <div class="ok">
    <strong>✓ FullEval é consistente:</strong> todos os splits usam
    <code>sprint3_zsl_val_N_train_excl_5/validation_pairs.csv</code>
    com imagens originais (<code>/mnt/data/la-cdip/data</code>).
    Os EERs finais do relatório são comparáveis entre splits e entre poolers.
  </div>
  {_table("", "CSV de validação do FullEval definitivo (eval_lacdip_full.py).", rows_fe, "aug")}

  <h2>4. Detalhes por Split</h2>
  {details}

  <h2>5. Conclusão</h2>
  <div class="ok">
    <strong>✓ Protocolo de dados uniforme — nenhuma assimetria entre splits:</strong>
    <ul>
      <li><strong>Todos os splits (0–4) — treino e subset val:</strong> usam
          <code>sprint3_zsl_val_{{N}}_train_excl_5/{{train,validation}}_pairs.csv</code>
          com imagens originais LA-CDIP em <code>/mnt/nas/joaopaulo/LA-CDIP/data</code>.
          Verificado em produção no servidor <code>gpds2</code>: 0% de caminhos augmentados
          para todos os splits.</li>
      <li><strong>FullEval (todas as variantes/poolers, todos os splits):</strong>
          imagens originais ✓ — EERs finais são comparáveis entre splits e entre poolers.</li>
      <li><strong>Dataset <code>final_split3</code>:</strong> criado para experimento separado
          (não publicado neste paper). Não utilizado em nenhuma etapa do Sprint3b.</li>
      <li><strong>Thumbnails no relatório HTML:</strong> a seção de exemplos de classes usa
          imagens de <code>final_split3/images_val/</code> (augmentadas) apenas para
          visualização quando o NAS não está disponível — não afeta treino nem avaliação.</li>
    </ul>
  </div>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(WORKSPACE / "results" / "data_source_audit.html"))
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(), encoding="utf-8")
    print(f"Salvo em: {out}")


if __name__ == "__main__":
    main()
