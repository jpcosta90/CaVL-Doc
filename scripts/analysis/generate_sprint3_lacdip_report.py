#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = WORKSPACE_ROOT / "data" / "generated_splits"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "results" / "sprint3_lacdip_report.html"
DEFAULT_SPRINT2_PROJECT = "CaVL-Doc_LA-CDIP_Sprint2_TeacherSweep"
DEFAULT_WANDB_ENTITY = "jpcosta1990-university-of-brasilia"


@dataclass
class TeacherConfig:
    run_name: str
    run_id: str
    best_eer: float
    professor_lr: float
    baseline_alpha: float
    entropy_coeff: float
    candidate_pool_size: int


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _best_eer_from_summary(summary: Dict[str, object]) -> Optional[float]:
    for key in ["val/best_eer", "best_eer", "metrics/best_eer", "eer_best", "best_val_eer"]:
        if key in summary:
            try:
                return float(summary[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_from_name(run_name: str) -> Dict[str, float]:
    extracted: Dict[str, float] = {}
    patterns = {
        "professor_lr": r"_plr([0-9eE+\-.]+)",
        "candidate_pool_size": r"_pool(\d+)",
        "baseline_alpha": r"_ba([0-9eE+\-.]+)",
        "entropy_coeff": r"_ent([0-9eE+\-.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, run_name or "")
        if not match:
            continue
        try:
            extracted[key] = float(match.group(1))
        except ValueError:
            continue
    return extracted


def _extract_float(config: Dict[str, object], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in config:
            try:
                return float(config[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_int(config: Dict[str, object], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        if key in config:
            try:
                return int(float(config[key]))
            except (TypeError, ValueError):
                continue
    return None


def _extract_loss_name(run_name: str, config: Dict[str, object]) -> str:
    loss = config.get("loss_type")
    if isinstance(loss, str) and loss.strip():
        return loss.strip()
    match = re.match(r"^Sprint2_(.+?)_from_", run_name or "")
    if match:
        return match.group(1)
    return ""


def _load_sprint2_rows(entity: str, project: str) -> List[Dict[str, object]]:
    import wandb

    api = wandb.Api()
    rows: List[Dict[str, object]] = []

    for run in api.runs(f"{entity}/{project}"):
        run_name = run.name or ""
        config = dict(run.config) if run.config else {}
        summary = dict(run.summary) if run.summary else {}
        best_eer = _best_eer_from_summary(summary)
        if best_eer is None:
            continue

        loss_type = _extract_loss_name(run_name, config)
        if not loss_type:
            continue

        parsed = _extract_from_name(run_name)
        professor_lr = _extract_float(config, ["professor-lr", "professor_lr"])
        baseline_alpha = _extract_float(config, ["baseline-alpha", "baseline_alpha"])
        entropy_coeff = _extract_float(config, ["entropy-coeff", "entropy_coeff"])
        candidate_pool_size = _extract_int(config, ["candidate-pool-size", "candidate_pool_size"])

        if professor_lr is None:
            professor_lr = parsed.get("professor_lr")
        if baseline_alpha is None:
            baseline_alpha = parsed.get("baseline_alpha")
        if entropy_coeff is None:
            entropy_coeff = parsed.get("entropy_coeff")
        if candidate_pool_size is None and "candidate_pool_size" in parsed:
            candidate_pool_size = int(parsed["candidate_pool_size"])

        if professor_lr is None or baseline_alpha is None or entropy_coeff is None or candidate_pool_size is None:
            continue

        rows.append(
            {
                "run_name": run_name,
                "run_id": run.id,
                "loss_type": loss_type,
                "best_eer": float(best_eer),
                "professor_lr": float(professor_lr),
                "baseline_alpha": float(baseline_alpha),
                "entropy_coeff": float(entropy_coeff),
                "candidate_pool_size": int(candidate_pool_size),
            }
        )

    return rows


def _select_teacher(rows: List[Dict[str, object]]) -> Optional[TeacherConfig]:
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["best_eer"]))
    return TeacherConfig(
        run_name=str(best["run_name"]),
        run_id=str(best["run_id"]),
        best_eer=float(best["best_eer"]),
        professor_lr=float(best["professor_lr"]),
        baseline_alpha=float(best["baseline_alpha"]),
        entropy_coeff=float(best["entropy_coeff"]),
        candidate_pool_size=int(best["candidate_pool_size"]),
    )


def _select_losses(rows: List[Dict[str, object]]) -> List[str]:
    if not rows:
        return []
    best_by_loss: Dict[str, Dict[str, object]] = {}
    for row in rows:
        loss_type = str(row["loss_type"])
        current = best_by_loss.get(loss_type)
        if current is None or float(row["best_eer"]) < float(current["best_eer"]):
            best_by_loss[loss_type] = row
    ranked = sorted(best_by_loss.items(), key=lambda item: float(item[1]["best_eer"]))
    selected = [name for name, _ in ranked[:2]]
    if "contrastive" not in selected:
        selected.append("contrastive")
    deduped: List[str] = []
    for loss in selected:
        if loss not in deduped:
            deduped.append(loss)
    return deduped


def _load_split_stats(splits_dir: Path, split_names: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_name in split_names:
        split_dir = splits_dir / split_name
        train_csv = split_dir / "train_pairs.csv"
        val_csv = split_dir / "validation_pairs.csv"
        if not train_csv.exists() or not val_csv.exists():
            continue

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        train_classes = set(train_df.get("class_a_name", pd.Series(dtype=str)).astype(str)).union(
            set(train_df.get("class_b_name", pd.Series(dtype=str)).astype(str))
        )
        val_classes = set(val_df.get("class_a_name", pd.Series(dtype=str)).astype(str)).union(
            set(val_df.get("class_b_name", pd.Series(dtype=str)).astype(str))
        )
        overlap = sorted(train_classes & val_classes)

        rows.append(
            {
                "split": split_name,
                "train_pairs": len(train_df),
                "val_pairs": len(val_df),
                "train_pos": int(train_df["is_equal"].sum()) if "is_equal" in train_df.columns else None,
                "train_neg": int((1 - train_df["is_equal"]).sum()) if "is_equal" in train_df.columns else None,
                "val_pos": int(val_df["is_equal"].sum()) if "is_equal" in val_df.columns else None,
                "val_neg": int((1 - val_df["is_equal"]).sum()) if "is_equal" in val_df.columns else None,
                "train_classes": len(train_classes),
                "val_classes": len(val_classes),
                "class_overlap": len(overlap),
                "train_pairs_per_class": int(train_df.groupby("class_a_name").size().mode().iloc[0]) if "class_a_name" in train_df.columns else None,
                "val_pairs_per_class_mean": round(val_df.groupby("class_a_name").size().mean(), 2) if "class_a_name" in val_df.columns else None,
                "val_pairs_per_class_min": int(val_df.groupby("class_a_name").size().min()) if "class_a_name" in val_df.columns else None,
                "val_pairs_per_class_max": int(val_df.groupby("class_a_name").size().max()) if "class_a_name" in val_df.columns else None,
            }
        )

    return pd.DataFrame(rows)


def _load_sprint3_protocol_split_stats(
    raw_data_root: Path,
    eval_splits: Sequence[int],
    exclude_train_splits: Sequence[int],
) -> pd.DataFrame:
    splits_csv = raw_data_root / "splits.csv"
    protocol_csv = raw_data_root / "protocol.csv"
    if not splits_csv.exists() or not protocol_csv.exists():
        raise FileNotFoundError(
            f"Dados brutos inválidos em {raw_data_root}. Esperado splits.csv e protocol.csv."
        )

    split_meta = pd.read_csv(splits_csv)
    protocol_df = pd.read_csv(protocol_csv)
    zsl = protocol_df[protocol_df["split_mode"] == "zsl_split"].copy()
    all_splits = sorted(zsl["split_number"].dropna().astype(int).unique().tolist())

    rows: List[Dict[str, object]] = []
    for val_split in eval_splits:
        train_source_splits = [s for s in all_splits if s != int(val_split) and s not in exclude_train_splits]
        if not train_source_splits:
            continue

        train_df = zsl[zsl["split_number"].astype(int).isin(train_source_splits)].copy()
        val_df = zsl[zsl["split_number"].astype(int) == int(val_split)].copy()

        train_classes = split_meta[split_meta["zsl_split"].astype(int).isin(train_source_splits)]["class_name"].nunique()
        val_classes = split_meta[split_meta["zsl_split"].astype(int) == int(val_split)]["class_name"].nunique()

        rows.append(
            {
                "split": f"zsl_val_{int(val_split)}",
                "train_source_splits": ",".join(str(s) for s in train_source_splits),
                "train_pairs": len(train_df),
                "val_pairs": len(val_df),
                "train_pos": int(train_df["is_equal"].sum()),
                "train_neg": int((1 - train_df["is_equal"]).sum()),
                "val_pos": int(val_df["is_equal"].sum()),
                "val_neg": int((1 - val_df["is_equal"]).sum()),
                "train_classes": int(train_classes),
                "val_classes": int(val_classes),
                "class_overlap": 0,
            }
        )

    return pd.DataFrame(rows)


def _load_class_diagnostics(raw_data_root: Path, split_col: str = "zsl_split") -> Tuple[pd.DataFrame, pd.DataFrame]:
    splits_csv = raw_data_root / "splits.csv"
    if not splits_csv.exists():
        raise FileNotFoundError(f"splits.csv não encontrado em: {splits_csv}")

    splits = pd.read_csv(splits_csv)
    if split_col not in splits.columns:
        raise KeyError(f"Coluna {split_col!r} não encontrada em {splits_csv}")

    matrix = pd.pivot_table(
        splits,
        index="class_name",
        columns=split_col,
        values="doc_id",
        aggfunc="count",
        fill_value=0,
    ).sort_index()

    matrix = matrix.reindex(sorted(matrix.columns), axis=1, fill_value=0)
    matrix["doc_count"] = matrix.sum(axis=1)
    matrix["assigned_split"] = matrix.idxmax(axis=1)
    matrix = matrix.reset_index()

    totals = (
        splits.groupby("class_name")
        .size()
        .reset_index(name="doc_count")
        .sort_values(["doc_count", "class_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return matrix, totals


def _load_protocol_diagnostics(raw_data_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_csv = raw_data_root / "splits.csv"
    protocol_csv = raw_data_root / "protocol.csv"
    if not splits_csv.exists():
        raise FileNotFoundError(f"splits.csv não encontrado em: {splits_csv}")
    if not protocol_csv.exists():
        raise FileNotFoundError(f"protocol.csv não encontrado em: {protocol_csv}")

    splits = pd.read_csv(splits_csv)
    protocol = pd.read_csv(protocol_csv)
    name_to_class = splits.set_index("doc_id")["class_name"].to_dict()
    name_to_split = splits.set_index("doc_id")["zsl_split"].to_dict()

    zsl = protocol[protocol["split_mode"] == "zsl_split"].copy()
    zsl["class_a"] = zsl["file_a_name"].map(name_to_class)
    zsl["class_b"] = zsl["file_b_name"].map(name_to_class)
    zsl["split_number"] = zsl["split_number"].astype(int)

    # Matriz classe x split_number usando a soma de ocorrências das extremidades dos pares
    endpoint_counts = pd.concat(
        [zsl[["class_a", "split_number"]].rename(columns={"class_a": "class_name"}),
         zsl[["class_b", "split_number"]].rename(columns={"class_b": "class_name"})],
        ignore_index=True,
    )
    protocol_matrix = (
        endpoint_counts.groupby(["class_name", "split_number"]).size().unstack(fill_value=0).sort_index()
    )
    split_cols = [int(col) for col in sorted(protocol_matrix.columns)]
    protocol_matrix = protocol_matrix.reindex(split_cols, axis=1, fill_value=0)
    protocol_matrix["endpoint_total"] = protocol_matrix.sum(axis=1)
    protocol_matrix["dominant_split"] = protocol_matrix[split_cols].idxmax(axis=1)
    protocol_matrix = protocol_matrix.reset_index()

    pair_counts = (
        zsl.groupby("split_number")
        .agg(
            pairs=("is_equal", "size"),
            pos=("is_equal", "sum"),
        )
        .assign(neg=lambda d: d["pairs"] - d["pos"])
        .reset_index()
        .sort_values("split_number")
        .reset_index(drop=True)
    )

    class_occurrence = (
        endpoint_counts.groupby("class_name")
        .size()
        .reset_index(name="endpoint_occurrences")
        .sort_values(["endpoint_occurrences", "class_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return protocol_matrix, pair_counts, class_occurrence


def _render_chart(stats_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    splits = stats_df["split"].tolist()
    x = list(range(len(splits)))
    width = 0.38

    ax.bar([i - width / 2 for i in x], stats_df["train_pairs"], width=width, label="Train pairs", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], stats_df["val_pairs"], width=width, label="Validation pairs", color="#F58518")

    for idx, value in enumerate(stats_df["train_pairs"]):
        ax.text(idx - width / 2, value + 25, str(int(value)), ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(stats_df["val_pairs"]):
        ax.text(idx + width / 2, value + 25, str(int(value)), ha="center", va="bottom", fontsize=8)

    ax.set_title("LA-CDIP splits: pares de treino e validação por split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Número de pares")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_class_count_chart(class_counts_df: pd.DataFrame, title: str = "Documentos por classe") -> str:
    if class_counts_df.empty:
        return ""

    ordered = class_counts_df.sort_values(["doc_count", "class_name"], ascending=[False, True]).reset_index(drop=True)
    height = max(8.0, len(ordered) * 0.16)
    fig, ax = plt.subplots(figsize=(14, height), dpi=140)

    y = list(range(len(ordered)))
    ax.barh(y, ordered["doc_count"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["class_name"], fontsize=5)
    ax.set_xlabel("Número de documentos")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

    for idx, value in enumerate(ordered["doc_count"]):
        ax.text(value + 1, idx, str(int(value)), va="center", fontsize=5)

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_protocol_class_chart(class_occurrence_df: pd.DataFrame, title: str = "Ocorrências de classe no protocolo ZSL") -> str:
    if class_occurrence_df.empty:
        return ""

    ordered = class_occurrence_df.sort_values(["endpoint_occurrences", "class_name"], ascending=[False, True]).reset_index(drop=True)
    height = max(8.0, len(ordered) * 0.16)
    fig, ax = plt.subplots(figsize=(14, height), dpi=140)

    y = list(range(len(ordered)))
    ax.barh(y, ordered["endpoint_occurrences"], color="#F58518")
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["class_name"], fontsize=5)
    ax.set_xlabel("Ocorrências como endpoint em pares do protocolo")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

    for idx, value in enumerate(ordered["endpoint_occurrences"]):
        ax.text(value + 1, idx, str(int(value)), va="center", fontsize=5)

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _html_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="table", border=0, escape=True)


def build_report(
    splits_df: pd.DataFrame,
    class_matrix_df: pd.DataFrame,
    class_counts_df: pd.DataFrame,
    protocol_matrix_df: pd.DataFrame,
    protocol_pair_counts_df: pd.DataFrame,
    protocol_class_occurrence_df: pd.DataFrame,
    teacher: Optional[TeacherConfig],
    selected_losses: Sequence[str],
    output_path: Path,
) -> str:
    chart_b64 = _render_chart(splits_df)
    class_chart_b64 = _render_class_count_chart(class_counts_df)
    protocol_chart_b64 = _render_protocol_class_chart(protocol_class_occurrence_df)
    total_train = int(splits_df["train_pairs"].sum())
    total_val = int(splits_df["val_pairs"].sum())
    avg_val = float(splits_df["val_pairs"].mean()) if not splits_df.empty else 0.0
    avg_train = float(splits_df["train_pairs"].mean()) if not splits_df.empty else 0.0
    total_classes = int(class_counts_df.shape[0]) if not class_counts_df.empty else 0

    if teacher is not None:
        teacher_html = f"""
        <div class="cards">
          <div class="card"><div class="label">Run fonte</div><div class="value">{_html_escape(teacher.run_name)}</div><div class="muted">{_html_escape(teacher.run_id)}</div></div>
          <div class="card"><div class="label">Best EER</div><div class="value">{teacher.best_eer:.6f}</div><div class="muted">Menor EER entre as runs válidas</div></div>
          <div class="card"><div class="label">Professor LR</div><div class="value">{teacher.professor_lr:.6g}</div></div>
          <div class="card"><div class="label">Baseline alpha</div><div class="value">{teacher.baseline_alpha:.6g}</div></div>
          <div class="card"><div class="label">Entropy coeff</div><div class="value">{teacher.entropy_coeff:.6g}</div></div>
          <div class="card"><div class="label">Candidate pool</div><div class="value">{teacher.candidate_pool_size}</div></div>
        </div>
        """
    else:
        teacher_html = "<div class='alert'>Não foi possível carregar a configuração do professor via W&B.</div>"

    losses_html = "<ul>" + "".join(f"<li>{_html_escape(loss)}</li>" for loss in selected_losses) + "</ul>"

    summary_table = _table_html(splits_df)
    class_matrix_table = _table_html(class_matrix_df)
    protocol_matrix_table = _table_html(protocol_matrix_df)
    protocol_pairs_table = _table_html(protocol_pair_counts_df)

    organization_html = f"""
    <ul>
            <li>A Sprint 3 usa composição <strong>protocol-based</strong> do ZSL: para cada split de validação, o treino é a união dos demais splits elegíveis.</li>
            <li>O split <strong>5</strong> é mantido como holdout e excluído do treino em todos os cenários (usado para teste depois).</li>
            <li>A validação é sempre o split alvo e permanece balanceada 50/50 entre positivos e negativos no protocolo.</li>
            <li>Como o treino vem do protocolo completo, as quantidades variam por split de validação e deixam de ficar fixas em 2400.</li>
            <li>O limite por época no treino continua controlado por <strong>max-steps-per-epoch</strong>.</li>
    </ul>
    """

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Relatório Sprint 3 LA-CDIP</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --accent: #38bdf8;
      --border: rgba(148,163,184,.18);
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
      color: var(--text);
      font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.45;
    }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{ padding: 20px 0 8px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 2rem; }}
    h2 {{ margin-top: 30px; font-size: 1.35rem; }}
    .subtitle {{ color: var(--muted); max-width: 900px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card, .panel {{ background: rgba(17,24,39,.92); border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,.22); }}
    .card {{ padding: 14px 16px; }}
    .label {{ color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; }}
    .value {{ font-size: 1.2rem; font-weight: 700; margin-top: 6px; word-break: break-word; }}
    .muted {{ color: var(--muted); font-size: .9rem; margin-top: 4px; }}
    .panel {{ padding: 18px; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: .95rem; overflow: hidden; border-radius: 12px; }}
    .table th, .table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
    .table th {{ color: #fff; background: rgba(56,189,248,.08); }}
    .table tr:hover td {{ background: rgba(148,163,184,.04); }}
    .chart {{ width: 100%; height: auto; border-radius: 14px; border: 1px solid var(--border); background: #fff; }}
    ul {{ margin: 10px 0 0 22px; }}
    li {{ margin: 6px 0; }}
    .kicker {{ color: var(--accent); font-weight: 700; letter-spacing: .06em; text-transform: uppercase; font-size: .8rem; }}
    .alert {{ padding: 14px 16px; border-radius: 12px; background: rgba(244, 114, 182, 0.12); border: 1px solid rgba(244, 114, 182, 0.3); }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <div class="kicker">Sprint 3 · LA-CDIP</div>
      <h1>Relatório HTML de splits e configuração do professor</h1>
      <p class="subtitle">Resumo dos pares já preparados nos splits ZSL, com foco na organização por classe, balanceamento e nas configurações do professor que serão carregadas para a execução da Sprint 3.</p>
    </section>

    <section class="grid">
      <div class="card"><div class="label">Splits analisados</div><div class="value">{len(splits_df)}</div></div>
      <div class="card"><div class="label">Total treino</div><div class="value">{total_train}</div><div class="muted">média {avg_train:.0f} por split</div></div>
      <div class="card"><div class="label">Total validação</div><div class="value">{total_val}</div><div class="muted">média {avg_val:.0f} por split</div></div>
      <div class="card"><div class="label">Losses planejadas</div><div class="value">{len(selected_losses)}</div></div>
            <div class="card"><div class="label">Classes totais</div><div class="value">{total_classes}</div><div class="muted">na amostra bruta</div></div>
    </section>

    <section class="panel">
      <h2>Como os pares estão organizados</h2>
      {organization_html}
    </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>Distribuição de pares por split</h2>
      <img class="chart" src="data:image/png;base64,{chart_b64}" alt="Gráfico de pares por split" />
    </section>

        <section class="panel" style="margin-top: 18px;">
            <h2>Matriz de classes por split</h2>
            <p class="subtitle">Cada linha mostra a classe e quantos documentos ela possui em cada split ZSL; a coluna <strong>assigned_split</strong> indica o split dominante da classe.</p>
            {class_matrix_table}
        </section>

        <section class="panel" style="margin-top: 18px;">
            <h2>Documentos por classe</h2>
            <img class="chart" src="data:image/png;base64,{class_chart_b64}" alt="Gráfico de documentos por classe" />
        </section>

        <section class="panel" style="margin-top: 18px;">
            <h2>Composição das amostras do protocolo ZSL</h2>
            <p class="subtitle">Em cada split do protocolo, os pares pertencem exclusivamente às classes de validação do split correspondente; as demais classes ficam no treino.</p>
            <img class="chart" src="data:image/png;base64,{protocol_chart_b64}" alt="Gráfico de ocorrências de classe no protocolo ZSL" />
        </section>

        <section class="panel" style="margin-top: 18px;">
            <h2>Matriz classe × split do protocolo ZSL</h2>
            <p class="subtitle">A matriz conta quantas vezes cada classe aparece como extremidade de pares no protocolo, por split_number.</p>
            {protocol_matrix_table}
        </section>

        <section class="panel" style="margin-top: 18px;">
            <h2>Resumo dos pares do protocolo por split</h2>
            {protocol_pairs_table}
        </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>Detalhe dos splits</h2>
      {summary_table}
    </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>Configurações do professor que vão rodar</h2>
      {teacher_html}
      <h3 style="margin-top:16px;">Losses selecionadas</h3>
      {losses_html}
    </section>
  </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um relatório HTML para a Sprint 3 da LA-CDIP")
    parser.add_argument("--raw-data-root", default="/mnt/data/la-cdip")
    parser.add_argument("--splits", default="0,1,2,3,4")
    parser.add_argument(
        "--exclude-train-splits",
        default="5",
        help="Splits excluídos do treino em todos os cenários (ex.: 5).",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--sprint2-project", default=DEFAULT_SPRINT2_PROJECT)
    parser.add_argument("--no-wandb", action="store_true", help="Gera o relatório sem consultar o W&B")
    args = parser.parse_args()

    eval_splits = [int(item) for item in _parse_csv_list(args.splits)]
    exclude_train_splits = [int(item) for item in _parse_csv_list(args.exclude_train_splits)]
    splits_df = _load_sprint3_protocol_split_stats(
        raw_data_root=Path(args.raw_data_root),
        eval_splits=eval_splits,
        exclude_train_splits=exclude_train_splits,
    )
    if splits_df.empty:
        raise RuntimeError("Nenhum split válido encontrado para gerar o relatório.")

    class_matrix_df, class_counts_df = _load_class_diagnostics(Path(args.raw_data_root), split_col="zsl_split")
    protocol_matrix_df, protocol_pair_counts_df, protocol_class_occurrence_df = _load_protocol_diagnostics(Path(args.raw_data_root))

    teacher = None
    selected_losses: List[str] = []
    if not args.no_wandb:
        try:
            rows = _load_sprint2_rows(args.wandb_entity, args.sprint2_project)
            teacher = _select_teacher(rows)
            selected_losses = _select_losses(rows)
        except Exception as exc:
            print(f"[WARN] Não foi possível consultar W&B para carregar o professor: {exc}")

    if not selected_losses:
        selected_losses = ["subcenter_cosface", "subcenter_arcface", "contrastive"]

    output_path = Path(args.output).expanduser().resolve()
    result = build_report(
        splits_df=splits_df,
        class_matrix_df=class_matrix_df,
        class_counts_df=class_counts_df,
        protocol_matrix_df=protocol_matrix_df,
        protocol_pair_counts_df=protocol_pair_counts_df,
        protocol_class_occurrence_df=protocol_class_occurrence_df,
        teacher=teacher,
        selected_losses=selected_losses,
        output_path=output_path,
    )
    print(result)


if __name__ == "__main__":
    main()
