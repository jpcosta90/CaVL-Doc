#!/usr/bin/env python3
"""
Compara hipoteticamente schedulers de learning rate usando histórico real do W&B.

A ideia é pegar uma curva real de validação (por exemplo, val/eer) de uma run
já executada na Sprint 1 e simular como dois schedulers se comportariam sob a
mesma trajetória observada:
- CosineAnnealingLR
- ReduceLROnPlateau

Isso não re-treina o modelo; é uma comparação de agenda de LR guiada pelo histórico real.

Saídas padrão:
- analysis/scheduler_comparison_wandb/scheduler_wandb_comparison.csv
- analysis/scheduler_comparison_wandb/scheduler_wandb_comparison.md
- analysis/scheduler_comparison_wandb/scheduler_wandb_comparison.png

Exemplo:
    python scripts/analysis/generate_scheduler_comparison_from_wandb.py \
        --entity jpcosta1990-university-of-brasilia \
        --project CaVL-Doc_LA-CDIP_Sprint1_Top5Validation \
        --run-id n3bso47h
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_PROJECT = "CaVL-Doc_LA-CDIP_Sprint1_Top5Validation"
DEFAULT_OUTPUT_DIR = "analysis/scheduler_comparison_wandb"
DEFAULT_RUN_IDS = [
    "n3bso47h",  # contrastive seed42 (exemplo já disponível no resumo)
]


@dataclass(frozen=True)
class SchedulerPoint:
    epoch: int
    metric: float
    cosine_lr: float
    plateau_lr: float
    plateau_reduced: bool


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _parse_metric_name(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    if ":" in raw:
        x, y = raw.split(":", 1)
        return x.strip(), y.strip()
    if raw.count(".") >= 1:
        return raw, raw.split(".")[-1]
    return raw, raw


def _cosine_lr(initial_lr: float, epoch_idx: int, epochs: int) -> float:
    if epochs <= 1:
        return initial_lr
    t = min(epoch_idx, epochs)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * t / epochs))


def _plateau_lrs(
    initial_lr: float,
    metrics: list[float],
    factor: float,
    patience: int,
    min_lr: float,
    threshold: float,
) -> list[tuple[float, bool]]:
    best = None
    bad_epochs = 0
    lr = initial_lr
    out: list[tuple[float, bool]] = []

    for metric in metrics:
        reduced = False
        if best is None or metric < (best - threshold):
            best = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                new_lr = max(lr * factor, min_lr)
                if new_lr < lr:
                    lr = new_lr
                    reduced = True
                bad_epochs = 0
        out.append((lr, reduced))

    return out


def _fetch_run_history(entity: Optional[str], project: str, run_id: str, metric_key: str) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    run = api.run(f"{path}/{run_id}")

    try:
        hist = run.history(keys=["epoch", metric_key], pandas=True)
    except Exception:
        hist = pd.DataFrame()

    if hist.empty:
        # Fallback: tenta alguns nomes comuns de métrica
        candidates = ["val/eer", "val/loss", "val_mean_loss", "val/eer_score", "val/eer"]
        for candidate in candidates:
            try:
                hist = run.history(keys=["epoch", candidate], pandas=True)
            except Exception:
                hist = pd.DataFrame()
            if not hist.empty and candidate in hist.columns:
                metric_key = candidate
                break

    if hist.empty:
        raise ValueError(f"Não foi possível obter histórico da run {run_id} em {path}")

    if "epoch" not in hist.columns:
        # Se não houver coluna epoch, tenta usar o índice como eixo temporal
        hist = hist.reset_index().rename(columns={"index": "epoch"})

    hist = hist.dropna(subset=[metric_key]).sort_values("epoch").reset_index(drop=True)
    if hist.empty:
        raise ValueError(f"Histórico sem valores válidos de {metric_key} na run {run_id}")

    return hist[["epoch", metric_key]].rename(columns={metric_key: "metric"})


def _build_points(hist: pd.DataFrame, initial_lr: float, factor: float, patience: int, min_lr: float, threshold: float) -> list[SchedulerPoint]:
    metrics = hist["metric"].astype(float).tolist()
    epochs = len(metrics)

    plateau = _plateau_lrs(initial_lr, metrics, factor, patience, min_lr, threshold)
    points: list[SchedulerPoint] = []
    for idx, metric in enumerate(metrics):
        points.append(
            SchedulerPoint(
                epoch=idx + 1,
                metric=metric,
                cosine_lr=_cosine_lr(initial_lr, idx, epochs - 1),
                plateau_lr=plateau[idx][0],
                plateau_reduced=plateau[idx][1],
            )
        )
    return points


def _write_csv(path: Path, points: list[SchedulerPoint], source_run: str, metric_key: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_run", "metric_key", "epoch", "metric", "cosine_lr", "plateau_lr", "plateau_reduced"])
        for p in points:
            writer.writerow([
                source_run,
                metric_key,
                p.epoch,
                f"{p.metric:.6f}",
                f"{p.cosine_lr:.12g}",
                f"{p.plateau_lr:.12g}",
                int(p.plateau_reduced),
            ])


def _write_markdown(path: Path, points: list[SchedulerPoint], source_run: str, metric_key: str, initial_lr: float, factor: float, patience: int) -> None:
    reductions = sum(1 for p in points if p.plateau_reduced)
    cosine_end = points[-1].cosine_lr
    plateau_end = points[-1].plateau_lr
    min_metric = min(p.metric for p in points)
    min_metric_epoch = min(points, key=lambda p: p.metric).epoch

    lines = []
    lines.append("# Comparação de schedulers com histórico real do W&B")
    lines.append("")
    lines.append("## Fonte")
    lines.append(f"- Run: `{source_run}`")
    lines.append(f"- Métrica analisada: `{metric_key}`")
    lines.append(f"- Épocas observadas: `{len(points)}`")
    lines.append("")
    lines.append("## Configuração da simulação")
    lines.append(f"- LR inicial: `{initial_lr:.12g}`")
    lines.append(f"- Plateau factor: `{factor}`")
    lines.append(f"- Plateau patience: `{patience}`")
    lines.append("")
    lines.append("## Resumo")
    lines.append(f"- Melhor métrica observada: `{min_metric:.6f}` na época `{min_metric_epoch}`")
    lines.append(f"- Cosine termina em LR `{cosine_end:.12g}`")
    lines.append(f"- Plateau termina em LR `{plateau_end:.12g}` com `{reductions}` reduções")
    lines.append("")
    lines.append("## Leitura prática")
    lines.append("- `cosine` reduz a LR de forma previsível, mesmo se a validação oscilar.")
    lines.append("- `plateau` reage diretamente ao histórico: segura LR enquanto houver melhoria e reduz ao estagnar.")
    lines.append("- Se a run real melhora cedo e depois piora, `plateau` costuma ser mais coerente para preservar o ponto bom.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, points: list[SchedulerPoint], title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception:
        print("[WARN] matplotlib indisponível; plot não será salvo.")
        return

    epochs = [p.epoch for p in points]
    cosine = [p.cosine_lr for p in points]
    plateau = [p.plateau_lr for p in points]
    metric = [p.metric for p in points]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(epochs, cosine, marker="o", linewidth=2, label="Cosine LR")
    ax1.plot(epochs, plateau, marker="s", linewidth=2, label="Plateau LR")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Learning rate")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, metric, color="black", linestyle="--", alpha=0.75, label="Métrica real")
    ax2.set_ylabel("Métrica real")
    ax2.invert_yaxis()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparação de schedulers usando histórico real do W&B")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Run ID do W&B. Pode repetir para comparar várias runs",
    )
    parser.add_argument(
        "--metric-key",
        default="val/eer",
        help="Chave da métrica a extrair do histórico (ex.: val/eer, val/loss)",
    )
    parser.add_argument("--initial-lr", type=float, default=1e-5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de saída",
    )
    args = parser.parse_args()

    run_ids = args.run_id or DEFAULT_RUN_IDS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids:
        hist = _fetch_run_history(args.entity, args.project, run_id, args.metric_key)
        points = _build_points(
            hist,
            initial_lr=args.initial_lr,
            factor=args.factor,
            patience=args.patience,
            min_lr=args.min_lr,
            threshold=args.threshold,
        )

        safe_run = run_id.replace("/", "-")
        csv_path = out_dir / f"{safe_run}_scheduler_wandb_comparison.csv"
        md_path = out_dir / f"{safe_run}_scheduler_wandb_comparison.md"
        png_path = out_dir / f"{safe_run}_scheduler_wandb_comparison.png"

        _write_csv(csv_path, points, run_id, args.metric_key)
        _write_markdown(md_path, points, run_id, args.metric_key, args.initial_lr, args.factor, args.patience)
        _write_plot(png_path, points, f"Schedulers com histórico real: {run_id}")

        print("=" * 80)
        print(f"Comparação gerada para run {run_id}")
        print("=" * 80)
        print(f"CSV: {csv_path}")
        print(f"Markdown: {md_path}")
        print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
