#!/usr/bin/env python3
"""
Gera uma comparação hipotética entre schedulers de learning rate:
- CosineAnnealingLR
- ReduceLROnPlateau

A ideia é visualizar como cada scheduler se comporta quando a métrica
val_loss/val_eer melhora no começo e depois entra em plateau/piora.

Saídas padrão:
- analysis/scheduler_comparison/scheduler_comparison.csv
- analysis/scheduler_comparison/scheduler_comparison.md
- analysis/scheduler_comparison/scheduler_comparison.png

Uso exemplo:
    python scripts/analysis/generate_scheduler_comparison_report.py \
        --epochs 10 \
        --initial-lr 1e-5 \
        --factor 0.5 \
        --patience 1
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_METRIC_SEQ = [
    0.92,
    0.88,
    0.84,
    0.83,
    0.835,
    0.845,
    0.86,
    0.87,
    0.875,
    0.88,
]


@dataclass(frozen=True)
class SchedulerPoint:
    epoch: int
    metric: float
    cosine_lr: float
    plateau_lr: float
    plateau_reduced: bool


def _parse_float_list(raw: str | None) -> list[float]:
    if not raw:
        return []
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def cosine_lr(initial_lr: float, epoch_idx: int, epochs: int) -> float:
    if epochs <= 1:
        return initial_lr
    # CosineAnnealingLR (forma fechada): lr = eta_min + 0.5*(base_lr-eta_min)*(1 + cos(pi * t/T_max))
    # Aqui eta_min = 0.
    t = min(epoch_idx, epochs)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * t / epochs))


def plateau_lrs(
    initial_lr: float,
    metrics: Iterable[float],
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

        is_improvement = False
        if best is None:
            is_improvement = True
        else:
            # Considera melhoria apenas se a métrica cai mais do que threshold.
            is_improvement = metric < (best - threshold)

        if is_improvement:
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


def build_points(
    epochs: int,
    initial_lr: float,
    factor: float,
    patience: int,
    metric_seq: list[float],
    min_lr: float,
    threshold: float,
) -> list[SchedulerPoint]:
    if not metric_seq:
        metric_seq = DEFAULT_METRIC_SEQ[:epochs]
    if len(metric_seq) < epochs:
        metric_seq = metric_seq + [metric_seq[-1]] * (epochs - len(metric_seq))
    metric_seq = metric_seq[:epochs]

    plateau = plateau_lrs(
        initial_lr=initial_lr,
        metrics=metric_seq,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        threshold=threshold,
    )

    points: list[SchedulerPoint] = []
    for idx in range(epochs):
        points.append(
            SchedulerPoint(
                epoch=idx + 1,
                metric=metric_seq[idx],
                cosine_lr=cosine_lr(initial_lr, idx, epochs - 1),
                plateau_lr=plateau[idx][0],
                plateau_reduced=plateau[idx][1],
            )
        )
    return points


def write_csv(path: Path, points: list[SchedulerPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "metric", "cosine_lr", "plateau_lr", "plateau_reduced"])
        for p in points:
            writer.writerow([
                p.epoch,
                f"{p.metric:.6f}",
                f"{p.cosine_lr:.12g}",
                f"{p.plateau_lr:.12g}",
                int(p.plateau_reduced),
            ])


def write_markdown(path: Path, points: list[SchedulerPoint], initial_lr: float, factor: float, patience: int) -> None:
    cosine_end = points[-1].cosine_lr
    plateau_end = points[-1].plateau_lr
    reductions = sum(1 for p in points if p.plateau_reduced)

    lines = []
    lines.append("# Comparação hipotética de schedulers")
    lines.append("")
    lines.append("## Configuração")
    lines.append(f"- Learning rate inicial: `{initial_lr:.12g}`")
    lines.append(f"- ReduceLROnPlateau: factor=`{factor}` | patience=`{patience}`")
    lines.append(f"- Épocas simuladas: `{len(points)}`")
    lines.append("")
    lines.append("## Leitura rápida")
    lines.append(f"- Cosine termina em `{cosine_end:.12g}`.")
    lines.append(f"- Plateau termina em `{plateau_end:.12g}` com `{reductions}` reduções.")
    lines.append("")
    lines.append("## Interpretação")
    lines.append("- `cosine` faz uma queda suave e previsível ao longo das épocas.")
    lines.append("- `plateau` mantém a LR até a métrica estagnar e só então reduz por fator multiplicativo.")
    lines.append("- Em cenários com melhoria inicial seguida de piora, `plateau` reage mais diretamente ao sinal de validação.")
    lines.append("")
    lines.append("## Próxima decisão prática")
    lines.append("- Se a validação oscila e piora cedo, `plateau` tende a ser mais adaptativo.")
    lines.append("- Se você quer uma agenda determinística sem depender da métrica, `cosine` é mais previsível.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, points: list[SchedulerPoint], title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except ImportError:
        print("[WARN] matplotlib não está disponível; plot PNG não será gerado.")
        return

    epochs = [p.epoch for p in points]
    cosine = [p.cosine_lr for p in points]
    plateau = [p.plateau_lr for p in points]
    metric = [p.metric for p in points]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ax1.plot(epochs, cosine, marker="o", label="Cosine LR", linewidth=2)
    ax1.plot(epochs, plateau, marker="s", label="Plateau LR", linewidth=2)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Learning rate")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, metric, color="black", linestyle="--", alpha=0.7, label="Métrica hipotética")
    ax2.set_ylabel("Métrica hipotética")
    ax2.invert_yaxis()

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparação hipotética entre CosineAnnealingLR e ReduceLROnPlateau")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--initial-lr", type=float, default=1e-5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument(
        "--metric-seq",
        default=",".join(str(v) for v in DEFAULT_METRIC_SEQ),
        help="Sequência hipotética de métrica (val_loss ou val_eer) separada por vírgulas",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/scheduler_comparison",
        help="Diretório de saída",
    )
    args = parser.parse_args()

    metric_seq = _parse_float_list(args.metric_seq)
    points = build_points(
        epochs=args.epochs,
        initial_lr=args.initial_lr,
        factor=args.factor,
        patience=args.patience,
        metric_seq=metric_seq,
        min_lr=args.min_lr,
        threshold=args.threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scheduler_comparison.csv"
    md_path = out_dir / "scheduler_comparison.md"
    png_path = out_dir / "scheduler_comparison.png"

    write_csv(csv_path, points)
    write_markdown(md_path, points, args.initial_lr, args.factor, args.patience)
    write_plot(png_path, points, "Comparação hipotética: Cosine vs Plateau")

    print("=" * 80)
    print("Comparação hipotética de schedulers gerada com sucesso")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
