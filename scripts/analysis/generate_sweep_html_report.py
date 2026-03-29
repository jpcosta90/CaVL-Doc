#!/usr/bin/env python3
"""
Gera relatório HTML detalhado dos sweeps (coarse + fine) para RVL-CDIP e LA-CDIP.

Inclui:
- Critérios usados na seleção
- Avaliação por loss e por loss+k
- Tabelas de finalistas do coarse
- Tabelas de desempenho no fine search
- Comparação coarse vs fine por grupo

Uso:
    python scripts/analysis/generate_sweep_html_report.py \
        --entity <wandb_entity>

Saída padrão:
    analysis/sweep_report/sweep_report.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    coarse_project: str
    fine_project: str
    eer_ceiling: float
    top_k: int


DATASETS = [
    DatasetSpec(
        key="rvlcdip",
        display_name="RVL-CDIP",
        coarse_project="CaVL-Doc_RVL-CDIP_InternVL3-2B_Sweeps",
        fine_project="CaVL-Doc_RVL-CDIP_FineSearch",
        eer_ceiling=0.35,
        top_k=10,
    ),
    DatasetSpec(
        key="lacdip",
        display_name="LA-CDIP",
        coarse_project="CaVL-Doc_LA-CDIP_InternVL3-2B_Sweeps",
        fine_project="CaVL-Doc_LA-CDIP_FineSearch",
        eer_ceiling=0.05,
        top_k=10,
    ),
]


def cfg_get(cfg: dict[str, Any], key: str, default=None):
    return cfg.get(key, cfg.get(key.replace("-", "_"), default))


def loss_label(loss_type: str | None, k: int | None) -> str:
    if loss_type is None:
        return "unknown"
    return f"{loss_type}_k{k}" if k is not None and not pd.isna(k) else str(loss_type)


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    try:
        val = a.corr(b)
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def fetch_runs(api, entity: str | None, project: str, phase: str, dataset: str) -> pd.DataFrame:
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": {"$in": ["finished", "crashed"]}})

    records: list[dict[str, Any]] = []
    for run in runs:
        cfg = run.config
        run_loss = cfg_get(cfg, "loss-type")
        lr = cfg_get(cfg, "student-lr")
        margin = cfg_get(cfg, "margin")
        scale = cfg_get(cfg, "scale")
        k = cfg_get(cfg, "num-sub-centers")

        if run_loss is None:
            continue

        try:
            hist = run.history(keys=["epoch", "val/eer"], pandas=True)
        except Exception:
            hist = pd.DataFrame()

        eer_series = None
        if not hist.empty and "val/eer" in hist.columns:
            hist = hist.dropna(subset=["val/eer"]).sort_values("epoch").reset_index(drop=True)
            if not hist.empty:
                eer_series = hist["val/eer"].to_numpy(dtype=float)

        if eer_series is None or len(eer_series) == 0:
            summary_eer = run.summary.get("val/eer", None)
            if summary_eer is None:
                continue
            eer_series = np.array([float(summary_eer)], dtype=float)

        eer_final = float(np.min(eer_series))
        eer_epoch1 = float(eer_series[0])
        eer_drop = float(eer_epoch1 - eer_final)
        eer_stability = float(np.std(eer_series))
        diverged = bool(len(eer_series) >= 2 and eer_series[-1] > eer_series[0])

        k_int = None
        if k is not None and not pd.isna(k):
            try:
                k_int = int(k)
            except Exception:
                k_int = None

        records.append(
            {
                "dataset": dataset,
                "phase": phase,
                "project": project,
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "loss_type": str(run_loss),
                "k": k_int,
                "loss_label": loss_label(str(run_loss), k_int),
                "lr": float(lr) if lr is not None else np.nan,
                "margin": float(margin) if margin is not None else np.nan,
                "scale": float(scale) if scale is not None else np.nan,
                "eer_final": eer_final,
                "eer_epoch1": eer_epoch1,
                "eer_drop": eer_drop,
                "eer_stability": eer_stability,
                "diverged": diverged,
                "n_points": int(len(eer_series)),
                "sweep_url": run.url,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "dataset",
                "phase",
                "project",
                "run_id",
                "run_name",
                "state",
                "loss_type",
                "k",
                "loss_label",
                "lr",
                "margin",
                "scale",
                "eer_final",
                "eer_epoch1",
                "eer_drop",
                "eer_stability",
                "diverged",
                "n_points",
                "sweep_url",
            ]
        )

    df = pd.DataFrame(records)
    df["score"] = df["eer_final"] - 0.5 * df["eer_drop"] + df["eer_stability"]
    return df


def compute_coarse_selection(df_coarse: pd.DataFrame, eer_ceiling: float, top_k: int):
    if df_coarse.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_coarse.copy()
    df["toxic"] = df["diverged"] | (df["eer_final"] >= eer_ceiling)
    survivors = df[~df["toxic"]].copy()

    inert_records: list[dict[str, Any]] = []
    for label, grp in survivors.groupby("loss_label"):
        for param in ["margin", "scale"]:
            vals = grp[param].dropna()
            if vals.nunique() < 2:
                inert_records.append(
                    {
                        "loss_label": label,
                        "param": param,
                        "pearson_abs": np.nan,
                        "is_inert": False,
                        "fixed_value": np.nan,
                        "reason": "nunique<2",
                    }
                )
                continue
            corr = safe_corr(grp[param], grp["eer_final"])
            if corr is None:
                inert_records.append(
                    {
                        "loss_label": label,
                        "param": param,
                        "pearson_abs": np.nan,
                        "is_inert": False,
                        "fixed_value": np.nan,
                        "reason": "corr_nan",
                    }
                )
                continue

            is_inert = abs(corr) < 0.3
            fixed_val = float(vals.median()) if is_inert else np.nan
            inert_records.append(
                {
                    "loss_label": label,
                    "param": param,
                    "pearson_abs": abs(float(corr)),
                    "is_inert": is_inert,
                    "fixed_value": fixed_val,
                    "reason": "|r|<0.3" if is_inert else "active",
                }
            )

    inert_df = pd.DataFrame(inert_records)

    finalists = (
        survivors.sort_values(["loss_label", "score"], ascending=[True, True])
        .groupby("loss_label", as_index=False)
        .head(top_k)
        .copy()
    )

    return df, survivors, finalists, inert_df


def summarize_by_loss(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "runs",
                "best_eer",
                "mean_eer",
                "std_eer",
                "mean_drop",
                "mean_stability",
                "best_score",
            ]
        )

    out = (
        df.groupby("loss_type", dropna=False)
        .agg(
            runs=("run_id", "count"),
            best_eer=("eer_final", "min"),
            mean_eer=("eer_final", "mean"),
            std_eer=("eer_final", "std"),
            mean_drop=("eer_drop", "mean"),
            mean_stability=("eer_stability", "mean"),
            best_score=("score", "min"),
        )
        .reset_index()
        .sort_values("best_eer")
    )
    return out


def summarize_by_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "loss_label",
                "runs",
                "best_eer",
                "mean_eer",
                "best_score",
                "mean_score",
            ]
        )

    out = (
        df.groupby("loss_label", dropna=False)
        .agg(
            runs=("run_id", "count"),
            best_eer=("eer_final", "min"),
            mean_eer=("eer_final", "mean"),
            best_score=("score", "min"),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values("best_eer")
    )
    return out


def compare_coarse_fine(df_coarse_survivors: pd.DataFrame, df_fine: pd.DataFrame) -> pd.DataFrame:
    coarse_best = (
        df_coarse_survivors.groupby("loss_label", as_index=False)
        .agg(coarse_best_eer=("eer_final", "min"), coarse_best_score=("score", "min"))
    )

    if df_fine.empty:
        out = coarse_best.copy()
        out["fine_best_eer"] = np.nan
        out["fine_best_score"] = np.nan
        out["delta_eer_fine_minus_coarse"] = np.nan
        out["improved"] = False
        return out.sort_values("coarse_best_eer")

    fine_best = (
        df_fine.groupby("loss_label", as_index=False)
        .agg(fine_best_eer=("eer_final", "min"), fine_best_score=("score", "min"))
    )

    out = coarse_best.merge(fine_best, on="loss_label", how="outer")
    out["delta_eer_fine_minus_coarse"] = out["fine_best_eer"] - out["coarse_best_eer"]
    out["improved"] = out["delta_eer_fine_minus_coarse"] < 0
    return out.sort_values(["delta_eer_fine_minus_coarse", "coarse_best_eer"], na_position="last")


def format_df(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Sem dados para esta seção.</p>"

    view = df.copy()
    if max_rows is not None and len(view) > max_rows:
        view = view.head(max_rows)

    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

    return view.to_html(index=False, classes="table table-striped", border=0, escape=False)


def section_header(title: str, subtitle: str = "") -> str:
    sub = f"<p class='muted'>{html.escape(subtitle)}</p>" if subtitle else ""
    return f"<section><h2>{html.escape(title)}</h2>{sub}"


def build_dataset_html(spec: DatasetSpec,
                       df_coarse_all: pd.DataFrame,
                       df_coarse_survivors: pd.DataFrame,
                       df_coarse_finalists: pd.DataFrame,
                       df_fine_all: pd.DataFrame,
                       inert_df: pd.DataFrame) -> str:
    coarse_loss = summarize_by_loss(df_coarse_all)
    coarse_label = summarize_by_label(df_coarse_survivors)
    fine_loss = summarize_by_loss(df_fine_all)
    fine_label = summarize_by_label(df_fine_all)
    cmp_df = compare_coarse_fine(df_coarse_survivors, df_fine_all)

    coarse_total = len(df_coarse_all)
    coarse_survivors = len(df_coarse_survivors)
    fine_total = len(df_fine_all)

    best_coarse = df_coarse_survivors.nsmallest(1, "eer_final") if not df_coarse_survivors.empty else pd.DataFrame()
    best_fine = df_fine_all.nsmallest(1, "eer_final") if not df_fine_all.empty else pd.DataFrame()

    html_parts = []
    html_parts.append(section_header(
        f"{spec.display_name}",
        f"Coarse: {spec.coarse_project} | Fine: {spec.fine_project}"
    ))

    summary_df = pd.DataFrame(
        [
            {
                "dataset": spec.display_name,
                "eer_ceiling": spec.eer_ceiling,
                "top_k": spec.top_k,
                "coarse_runs": coarse_total,
                "coarse_survivors": coarse_survivors,
                "coarse_survival_rate": (coarse_survivors / coarse_total) if coarse_total else np.nan,
                "fine_runs": fine_total,
            }
        ]
    )
    html_parts.append("<h3>Resumo</h3>")
    html_parts.append(format_df(summary_df))

    html_parts.append("<h3>Critérios aplicados</h3>")
    html_parts.append(
        """
        <ul>
                    <li><b>Objetivo metodológico</b>: coarse (exploração barata) → fine (validação em regime mais fiel/exigente).</li>
          <li><b>Métrica principal</b>: val/eer (menor é melhor).</li>
          <li><b>Métricas derivadas</b>: eer_final = min(val/eer), eer_drop = eer_epoch1 - eer_final, eer_stability = std(val/eer).</li>
          <li><b>Tóxico</b>: diverged = (eer_última &gt; eer_primeira) ou eer_final &gt;= ceiling.</li>
          <li><b>Regra de inércia</b>: parâmetro inerte quando |corr(param, eer_final)| &lt; 0.3; nesse caso é fixado na mediana.</li>
          <li><b>Score composto</b>: score = eer_final - 0.5 × eer_drop + eer_stability.</li>
          <li><b>Seleção coarse</b>: top-k por grupo (loss+k) ordenado por score.</li>
                    <li><b>Fine Search</b>: espaço reduzido (lr + margin, scale fixa) com bayes e run_cap; interpretação principal é validação de transferência coarse→fine.</li>
        </ul>
        """
    )

    html_parts.append("<h3>Coarse — avaliação por loss</h3>")
    html_parts.append(format_df(coarse_loss))

    html_parts.append("<h3>Coarse — avaliação por loss+k (após filtro tóxico)</h3>")
    html_parts.append(format_df(coarse_label))

    html_parts.append("<h3>Coarse — parâmetros inertes detectados</h3>")
    html_parts.append(format_df(inert_df.sort_values(["loss_label", "param"])) )

    html_parts.append("<h3>Coarse — finalistas (top-k por grupo)</h3>")
    finalists_cols = [
        "loss_label",
        "run_name",
        "lr",
        "margin",
        "scale",
        "eer_final",
        "eer_drop",
        "eer_stability",
        "score",
    ]
    html_parts.append(format_df(df_coarse_finalists[finalists_cols].sort_values(["loss_label", "score"]) if not df_coarse_finalists.empty else df_coarse_finalists))

    html_parts.append("<h3>Fine Search — avaliação por loss</h3>")
    html_parts.append(format_df(fine_loss))

    html_parts.append("<h3>Fine Search — avaliação por loss+k</h3>")
    html_parts.append(format_df(fine_label))

    html_parts.append("<h3>Transferência Coarse → Fine (melhor EER por grupo)</h3>")
    html_parts.append(format_df(cmp_df))

    html_parts.append("<h3>Melhores runs (global)</h3>")
    html_parts.append("<h4>Melhor coarse</h4>")
    html_parts.append(format_df(best_coarse[["loss_label", "run_name", "eer_final", "score", "sweep_url"]] if not best_coarse.empty else best_coarse))
    html_parts.append("<h4>Melhor fine</h4>")
    html_parts.append(format_df(best_fine[["loss_label", "run_name", "eer_final", "score", "sweep_url"]] if not best_fine.empty else best_fine))

    html_parts.append("</section>")
    return "\n".join(html_parts)


def build_html_page(title: str, body_sections: str) -> str:
    return f"""
<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1220;
      --card: #171a2b;
      --text: #e9ecf1;
      --muted: #aab2c0;
      --line: #2a3047;
      --accent: #6ea8fe;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      line-height: 1.45;
    }}
    h1, h2, h3, h4 {{ margin: 12px 0; }}
    h1 {{ color: var(--accent); }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin: 18px 0;
    }}
    .muted {{ color: var(--muted); }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 16px;
      font-size: 13px;
    }}
    .table th, .table td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .table th {{ background: #1f2440; }}
    a {{ color: #8ec5ff; }}
    code {{ background: #1d2238; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class=\"muted\">Gerado em: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <section>
    <h2>Escopo do Relatório</h2>
    <ul>
      <li>Dois datasets: RVL-CDIP e LA-CDIP.</li>
            <li>Duas fases: coarse sweep (exploração) e fine search (validação em regime mais fiel).</li>
            <li>Análise por loss, por loss+k, critérios de seleção e transferência coarse → fine.</li>
      <li>Fonte dos dados: WandB (runs finished/crashed com histórico de val/eer).</li>
    </ul>
  </section>

  {body_sections}
</body>
</html>
"""


def parse_args():
    p = argparse.ArgumentParser(description="Gera relatório HTML de sweeps coarse/fine via WandB.")
    p.add_argument("--entity", default=None, help="Entity WandB (user/org).")
    p.add_argument(
        "--output-dir",
        default="analysis/sweep_report",
        help="Diretório de saída do HTML e CSVs auxiliares.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout da API WandB em segundos (default: 60).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb não instalado. Rode: pip install wandb")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "dataframes"
    csv_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=args.timeout)

    sections: list[str] = []

    for spec in DATASETS:
        df_coarse = fetch_runs(api, args.entity, spec.coarse_project, "coarse", spec.display_name)
        df_fine = fetch_runs(api, args.entity, spec.fine_project, "fine", spec.display_name)

        df_coarse_all, df_coarse_survivors, df_coarse_finalists, inert_df = compute_coarse_selection(
            df_coarse, spec.eer_ceiling, spec.top_k
        )

        # Salva dataframes auxiliares para inspeção no notebook/Excel
        df_coarse_all.to_csv(csv_dir / f"{spec.key}_coarse_all.csv", index=False)
        df_coarse_survivors.to_csv(csv_dir / f"{spec.key}_coarse_survivors.csv", index=False)
        df_coarse_finalists.to_csv(csv_dir / f"{spec.key}_coarse_finalists.csv", index=False)
        df_fine.to_csv(csv_dir / f"{spec.key}_fine_all.csv", index=False)
        inert_df.to_csv(csv_dir / f"{spec.key}_inert_params.csv", index=False)

        sections.append(
            build_dataset_html(
                spec=spec,
                df_coarse_all=df_coarse_all,
                df_coarse_survivors=df_coarse_survivors,
                df_coarse_finalists=df_coarse_finalists,
                df_fine_all=df_fine,
                inert_df=inert_df,
            )
        )

    report_title = "CaVL-Doc — Análise Detalhada de Sweeps (Coarse + Fine)"
    html_text = build_html_page(report_title, "\n".join(sections))

    output_file = out_dir / "sweep_report.html"
    output_file.write_text(html_text, encoding="utf-8")

    print("=" * 72)
    print("Relatório gerado com sucesso")
    print("=" * 72)
    print(f"HTML: {output_file}")
    print(f"CSVs: {csv_dir}")


if __name__ == "__main__":
    main()
