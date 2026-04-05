#!/usr/bin/env python3
"""
Gera relatório de comparação da Sprint 1 (LA-CDIP) entre losses,
com foco na ablação do professor nas últimas 5 épocas.

Saídas:
- analysis/sprint1_report/sprint1_runs_raw.csv
- analysis/sprint1_report/sprint1_summary_by_loss_variant.csv
- analysis/sprint1_report/sprint1_paired_seed_comparison.csv
- analysis/sprint1_report/sprint1_loss_report.md

Uso:
    python scripts/analysis/generate_sprint1_loss_report.py \
        --entity <wandb_entity>
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_PROJECT = "CaVL-Doc_LA-CDIP_Sprint1_Top5Validation"
DEFAULT_OUTPUT_DIR = "analysis/sprint1_report"

RUN_NAME_PATTERN = re.compile(
    r"^Sprint1_(?P<loss>[a-zA-Z0-9_\-]+)_k(?P<k>\d+)_(?P<tail>.+?)_seed(?P<seed>\d+)(?:_.+)?$"
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
        "k_from_name": None,
        "seed_from_name": None,
        "professor_last5_from_name": None,
    }
    match = RUN_NAME_PATTERN.match(run_name or "")
    if not match:
        return out

    tail = match.group("tail")
    prof_flag: Optional[bool] = None
    if "prof_last5_on" in tail:
        prof_flag = True
    elif "prof_last5_off" in tail:
        prof_flag = False

    out.update(
        {
            "loss_type_from_name": match.group("loss"),
            "k_from_name": int(match.group("k")),
            "seed_from_name": int(match.group("seed")),
            "professor_last5_from_name": prof_flag,
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

    candidate_summary_keys = [
        "val/eer",
        "best_val/eer",
        "eval/eer",
        "eer",
        "val_eer",
    ]
    for key in candidate_summary_keys:
        value = _safe_float(run.summary.get(key))
        if value is not None:
            return value, value, 1

    return None, None, 0


def fetch_sprint1_runs(entity: Optional[str], project: str, states: list[str]) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": {"$in": states}})

    records: list[dict[str, Any]] = []
    for run in runs:
        cfg = run.config or {}
        parsed = _parse_run_name(run.name or "")

        loss_type = _cfg_get(cfg, "loss-type", parsed["loss_type_from_name"])
        k_val = _cfg_get(cfg, "num-sub-centers", parsed["k_from_name"])
        seed = _cfg_get(cfg, "seed", parsed["seed_from_name"])

        professor_last5 = parsed["professor_last5_from_name"]
        if professor_last5 is None:
            warmup_steps = _safe_float(_cfg_get(cfg, "professor-warmup-steps"))
            epochs = _safe_float(_cfg_get(cfg, "epochs"))
            max_steps = _safe_float(_cfg_get(cfg, "max-steps-per-epoch"))
            if warmup_steps is not None and epochs is not None and max_steps is not None:
                full_train_steps = epochs * max_steps
                professor_last5 = warmup_steps < full_train_steps

        best_eer, last_eer, n_val_points = _extract_eer_metrics(run)

        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": str(getattr(run, "created_at", "")),
                "url": run.url,
                "loss_type": str(loss_type) if loss_type is not None else None,
                "k": int(float(k_val)) if _safe_float(k_val) is not None else None,
                "seed": int(float(seed)) if _safe_float(seed) is not None else None,
                "professor_last5": professor_last5,
                "professor_last5_label": "on" if professor_last5 is True else "off" if professor_last5 is False else "unknown",
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
                "k",
                "seed",
                "professor_last5",
                "professor_last5_label",
                "best_eer",
                "last_eer",
                "n_val_points",
            ]
        )

    df = pd.DataFrame(records)
    return df.sort_values(["loss_type", "professor_last5_label", "seed", "best_eer"], na_position="last")


def summarize_by_loss_variant(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["loss_type", "best_eer"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "professor_last5_label",
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
        valid.groupby(["loss_type", "professor_last5_label"], dropna=False)
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
    valid = df.dropna(subset=["loss_type", "seed", "best_eer", "professor_last5"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "paired_seeds",
                "on_better_count",
                "off_better_count",
                "ties_count",
                "delta_off_minus_on_mean",
                "delta_off_minus_on_median",
            ]
        )

    per_seed = (
        valid.groupby(["loss_type", "seed", "professor_last5"], as_index=False)
        .agg(best_eer=("best_eer", "min"))
    )

    pivot = per_seed.pivot_table(
        index=["loss_type", "seed"],
        columns="professor_last5",
        values="best_eer",
        aggfunc="first",
    ).reset_index()

    if True not in pivot.columns or False not in pivot.columns:
        return pd.DataFrame(
            columns=[
                "loss_type",
                "paired_seeds",
                "on_better_count",
                "off_better_count",
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
                "on_better_count",
                "off_better_count",
                "ties_count",
                "delta_off_minus_on_mean",
                "delta_off_minus_on_median",
            ]
        )

    paired["delta_off_minus_on"] = paired[False] - paired[True]

    out = (
        paired.groupby("loss_type", as_index=False)
        .agg(
            paired_seeds=("seed", "count"),
            on_better_count=("delta_off_minus_on", lambda s: int((s > 0).sum())),
            off_better_count=("delta_off_minus_on", lambda s: int((s < 0).sum())),
            ties_count=("delta_off_minus_on", lambda s: int((s == 0).sum())),
            delta_off_minus_on_mean=("delta_off_minus_on", "mean"),
            delta_off_minus_on_median=("delta_off_minus_on", "median"),
        )
        .sort_values("delta_off_minus_on_mean", ascending=False)
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

    on_summary = df_summary[df_summary["professor_last5_label"] == "on"].copy()
    off_summary = df_summary[df_summary["professor_last5_label"] == "off"].copy()

    best_on = on_summary.nsmallest(1, "best_eer_mean") if not on_summary.empty else pd.DataFrame()
    best_off = off_summary.nsmallest(1, "best_eer_mean") if not off_summary.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Sprint 1 — Relatório de Comparação de Losses (LA-CDIP)")
    lines.append("")
    lines.append("## Escopo")
    lines.append(f"- Projeto W&B: `{project}`")
    lines.append(f"- Entity: `{entity or 'default do wandb login'}`")
    lines.append(f"- Runs coletados: `{total_runs}` (com EER válido: `{valid_runs}`)")
    lines.append("")

    if not best_on.empty:
        row = best_on.iloc[0]
        lines.append("## Melhor com professor (últimas 5 épocas)")
        lines.append(
            f"- Loss: `{row['loss_type']}` | mean best EER: `{row['best_eer_mean']:.6f}` | runs: `{int(row['runs'])}`"
        )
        lines.append("")

    if not best_off.empty:
        row = best_off.iloc[0]
        lines.append("## Melhor sem professor (últimas 5 épocas)")
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
    lines.append("- Rodar Sprint 1 sem professor em todas as épocas para a melhor loss observada.")
    lines.append("- Fazer sweep dedicado da rede professor mantendo fixa a melhor loss do aluno.")
    lines.append("- Repetir com 3 seeds fixas para confirmar estabilidade da decisão final.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera relatório Sprint 1 comparando losses e professor ON/OFF")
    parser.add_argument("--entity", default=None, help="W&B entity (opcional)")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Projeto W&B da Sprint 1")
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

    df_raw = fetch_sprint1_runs(entity=args.entity, project=args.project, states=states)
    df_summary = summarize_by_loss_variant(df_raw)
    df_paired = paired_seed_comparison(df_raw)

    raw_csv = output_dir / "sprint1_runs_raw.csv"
    summary_csv = output_dir / "sprint1_summary_by_loss_variant.csv"
    paired_csv = output_dir / "sprint1_paired_seed_comparison.csv"
    report_md = output_dir / "sprint1_loss_report.md"

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
    print("Sprint 1 report gerado com sucesso")
    print("=" * 80)
    print(f"Runs raw: {raw_csv}")
    print(f"Resumo loss+variante: {summary_csv}")
    print(f"Comparação pareada por seed: {paired_csv}")
    print(f"Relatório markdown: {report_md}")


if __name__ == "__main__":
    main()
