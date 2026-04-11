#!/usr/bin/env python3
"""
Gera relatório consolidado da Sprint 2 (LA-CDIP) com foco em:
- losses-base vindas da Sprint 1
- comparação Bayes (professor ativo) vs baseline (professor off)
- métricas de validação registradas no W&B

Saídas:
- analysis/sprint2_report/sprint2_runs_raw.csv
- analysis/sprint2_report/sprint2_summary_by_loss_variant.csv
- analysis/sprint2_report/sprint2_paired_seed_comparison.csv
- analysis/sprint2_report/sprint2_loss_report.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_PROJECT = "CaVL-Doc_LA-CDIP_Sprint2_TeacherSweep"
DEFAULT_OUTPUT_DIR = "analysis/sprint2_report"

RUN_NAME_PATTERN = re.compile(
    r"^Sprint2_(?P<loss>[a-zA-Z0-9_\-]+)_from_(?P<run_id>[a-zA-Z0-9]+)_(?P<tail>.+?)_seed(?P<seed>\d+)(?:_.+)?$"
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
        return None
    except (TypeError, ValueError):
        return None


def _cfg_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, cfg.get(key.replace("-", "_"), default))


def _parse_run_name(run_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "loss_type_from_name": None,
        "seed_from_name": None,
        "baseline_from_name": None,
    }
    match = RUN_NAME_PATTERN.match(run_name or "")
    if not match:
        return out

    tail = match.group("tail")
    baseline_flag: Optional[bool] = None
    if "prof_off" in tail:
        baseline_flag = True
    elif "prof_on" in tail:
        baseline_flag = False

    out.update(
        {
            "loss_type_from_name": match.group("loss"),
            "seed_from_name": int(match.group("seed")),
            "baseline_from_name": baseline_flag,
        }
    )
    return out


def _extract_eer_metrics(run) -> tuple[Optional[float], Optional[float], int]:
    try:
        hist = run.history(keys=["epoch", "val/eer"], pandas=True)
    except Exception:
        hist = pd.DataFrame()

    if not hist.empty and "val/eer" in hist.columns:
        hist = hist.dropna(subset=["val/eer"]).sort_values("epoch")
        if not hist.empty:
            series = hist["val/eer"].astype(float)
            return float(series.min()), float(series.iloc[-1]), int(series.shape[0])

    candidate_summary_keys = ["val/eer", "val/best_eer", "best_eer", "eer"]
    for key in candidate_summary_keys:
        value = _safe_float(run.summary.get(key))
        if value is not None:
            return value, value, 1

    return None, None, 0


def fetch_sprint2_runs(entity: Optional[str], project: str, states: list[str]) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": {"$in": states}})

    records: list[dict[str, Any]] = []
    for run in runs:
        cfg = run.config or {}
        parsed = _parse_run_name(run.name or "")

        loss_type = _cfg_get(cfg, "loss-type", parsed["loss_type_from_name"])
        seed = _cfg_get(cfg, "seed", parsed["seed_from_name"])
        professor_lr = _safe_float(_cfg_get(cfg, "professor-lr"))
        baseline_alpha = _safe_float(_cfg_get(cfg, "baseline-alpha"))
        entropy_coeff = _safe_float(_cfg_get(cfg, "entropy-coeff"))
        candidate_pool_size = _cfg_get(cfg, "candidate-pool-size")

        baseline_mode = parsed["baseline_from_name"]
        if baseline_mode is None and professor_lr is not None:
            baseline_mode = abs(professor_lr) < 1e-12

        best_eer, last_eer, n_val_points = _extract_eer_metrics(run)

        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": str(getattr(run, "created_at", "")),
                "url": run.url,
                "loss_type": str(loss_type) if loss_type is not None else None,
                "seed": int(float(seed)) if _safe_float(seed) is not None else None,
                "baseline_mode": baseline_mode,
                "baseline_mode_label": "baseline_off" if baseline_mode is True else "bayes_prof_on" if baseline_mode is False else "unknown",
                "professor_lr": professor_lr,
                "baseline_alpha": baseline_alpha,
                "entropy_coeff": entropy_coeff,
                "candidate_pool_size": int(candidate_pool_size) if _safe_float(candidate_pool_size) is not None else None,
                "best_eer": best_eer,
                "last_eer": last_eer,
                "n_val_points": n_val_points,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_name",
                "state",
                "created_at",
                "url",
                "loss_type",
                "seed",
                "baseline_mode",
                "baseline_mode_label",
                "professor_lr",
                "baseline_alpha",
                "entropy_coeff",
                "candidate_pool_size",
                "best_eer",
                "last_eer",
                "n_val_points",
            ]
        )

    df = pd.DataFrame(records)
    return df.sort_values(["loss_type", "baseline_mode_label", "seed", "best_eer"], na_position="last")


def summarize_by_loss_variant(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["loss_type", "best_eer"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "baseline_mode_label",
                "runs",
                "unique_seeds",
                "best_eer_min",
                "best_eer_mean",
                "best_eer_std",
                "best_eer_median",
                "best_eer_max",
            ]
        )

    summary = (
        valid.groupby(["loss_type", "baseline_mode_label"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            unique_seeds=("seed", "nunique"),
            best_eer_min=("best_eer", "min"),
            best_eer_mean=("best_eer", "mean"),
            best_eer_std=("best_eer", "std"),
            best_eer_median=("best_eer", "median"),
            best_eer_max=("best_eer", "max"),
        )
        .reset_index()
        .sort_values(["best_eer_mean", "best_eer_min"], na_position="last")
    )
    return summary


def paired_seed_comparison(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["loss_type", "seed", "best_eer", "baseline_mode"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "paired_seeds",
                "baseline_off_better_count",
                "bayes_prof_on_better_count",
                "ties_count",
                "delta_off_minus_on_mean",
                "delta_off_minus_on_median",
            ]
        )

    per_seed = (
        valid.groupby(["loss_type", "seed", "baseline_mode"], as_index=False)
        .agg(best_eer=("best_eer", "min"))
    )

    pivot = per_seed.pivot_table(
        index=["loss_type", "seed"],
        columns="baseline_mode",
        values="best_eer",
        aggfunc="first",
    ).reset_index()

    if True not in pivot.columns or False not in pivot.columns:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "paired_seeds",
                "baseline_off_better_count",
                "bayes_prof_on_better_count",
                "ties_count",
                "delta_off_minus_on_mean",
                "delta_off_minus_on_median",
            ]
        )

    paired = pivot.dropna(subset=[True, False]).copy()
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "paired_seeds",
                "baseline_off_better_count",
                "bayes_prof_on_better_count",
                "ties_count",
                "delta_off_minus_on_mean",
                "delta_off_minus_on_median",
            ]
        )

    paired["delta_off_minus_on"] = paired[True] - paired[False]

    out = (
        paired.groupby("loss_type", as_index=False)
        .agg(
            paired_seeds=("seed", "count"),
            baseline_off_better_count=("delta_off_minus_on", lambda s: int((s < 0).sum())),
            bayes_prof_on_better_count=("delta_off_minus_on", lambda s: int((s > 0).sum())),
            ties_count=("delta_off_minus_on", lambda s: int((s == 0).sum())),
            delta_off_minus_on_mean=("delta_off_minus_on", "mean"),
            delta_off_minus_on_median=("delta_off_minus_on", "median"),
        )
        .sort_values("delta_off_minus_on_mean", ascending=True)
    )
    return out


def _format_df_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_Sem dados._"
    display = df.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    return display.to_markdown(index=False)


def build_markdown_report(
    project: str,
    entity: Optional[str],
    df_raw: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_paired: pd.DataFrame,
) -> str:
    total_runs = len(df_raw)
    valid_runs = int(df_raw["best_eer"].notna().sum()) if not df_raw.empty else 0

    bayes_summary = df_summary[df_summary["baseline_mode_label"] == "bayes_prof_on"].copy()
    off_summary = df_summary[df_summary["baseline_mode_label"] == "baseline_off"].copy()

    best_bayes = bayes_summary.nsmallest(1, "best_eer_mean") if not bayes_summary.empty else pd.DataFrame()
    best_off = off_summary.nsmallest(1, "best_eer_mean") if not off_summary.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Sprint 2 — Relatório de Comparação de Losses (LA-CDIP)")
    lines.append("")
    lines.append("## Escopo")
    lines.append(f"- Projeto W&B: `{project}`")
    lines.append(f"- Entity: `{entity or 'default do wandb login'}`")
    lines.append(f"- Runs coletados: `{total_runs}` (com EER válido: `{valid_runs}`)")
    lines.append("")

    if not best_bayes.empty:
        row = best_bayes.iloc[0]
        lines.append("## Melhor Bayes com professor ativo")
        lines.append(
            f"- Loss: `{row['loss_type']}` | mean best EER: `{row['best_eer_mean']:.6f}` | runs: `{int(row['runs'])}`"
        )
        lines.append("")

    if not best_off.empty:
        row = best_off.iloc[0]
        lines.append("## Melhor baseline com professor desligado")
        lines.append(
            f"- Loss: `{row['loss_type']}` | mean best EER: `{row['best_eer_mean']:.6f}` | runs: `{int(row['runs'])}`"
        )
        lines.append("")

    lines.append("## Tabela — Loss x Variante")
    lines.append(_format_df_markdown(df_summary, max_rows=50))
    lines.append("")

    lines.append("## Comparação pareada por seed (OFF - ON)")
    lines.append("- Interpretação: `delta_off_minus_on > 0` favorece professor ON; `< 0` favorece OFF.")
    lines.append(_format_df_markdown(df_paired, max_rows=50))
    lines.append("")

    lines.append("## Próximos passos sugeridos")
    lines.append("- Fixar a melhor loss observada e repetir a análise com 3 seeds.")
    lines.append("- Comparar Bayes com baseline OFF usando a mesma janela de épocas.")
    lines.append("- Gerar um sweep report consolidado após finalizar as runs.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera relatório Sprint 2 comparando Bayes e baseline OFF")
    parser.add_argument("--entity", default=None, help="W&B entity (opcional)")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Projeto W&B da Sprint 2")
    parser.add_argument(
        "--states",
        default="finished,crashed",
        help="Estados aceitos separados por vírgula (ex.: finished,crashed,running)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Diretório de saída")
    args = parser.parse_args()

    states = [item.strip() for item in args.states.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = fetch_sprint2_runs(entity=args.entity, project=args.project, states=states)
    df_summary = summarize_by_loss_variant(df_raw)
    df_paired = paired_seed_comparison(df_raw)

    raw_csv = output_dir / "sprint2_runs_raw.csv"
    summary_csv = output_dir / "sprint2_summary_by_loss_variant.csv"
    paired_csv = output_dir / "sprint2_paired_seed_comparison.csv"
    report_md = output_dir / "sprint2_loss_report.md"

    df_raw.to_csv(raw_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    df_paired.to_csv(paired_csv, index=False)

    report_content = build_markdown_report(
        project=args.project,
        entity=args.entity,
        df_raw=df_raw,
        df_summary=df_summary,
        df_paired=df_paired,
    )
    report_md.write_text(report_content, encoding="utf-8")

    print("=" * 80)
    print("Sprint 2 report gerado com sucesso")
    print("=" * 80)
    print(f"Runs raw: {raw_csv}")
    print(f"Resumo loss+variante: {summary_csv}")
    print(f"Comparação pareada por seed: {paired_csv}")
    print(f"Relatório markdown: {report_md}")


if __name__ == "__main__":
    main()
