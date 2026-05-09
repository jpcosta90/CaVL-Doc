#!/usr/bin/env python3
"""
Análise de época da Sprint 3 — melhor EER por loss/split/fase.

Para cada combinação (split × loss), busca o histórico por época no W&B e mostra:
  - Em qual época o melhor EER foi atingido
  - Curva de EER ao longo do treino (fase 1 + fase 2)
  - Qual loss foi melhor em cada split

Uso:
    python scripts/analysis/generate_sprint3_epoch_analysis.py
    python scripts/analysis/generate_sprint3_epoch_analysis.py --output results/sprint3_epoch_analysis.html
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
import numpy as np
import pandas as pd
import wandb

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT  = WORKSPACE_ROOT / "results" / "sprint3_epoch_analysis.html"
ENTITY          = "jpcosta1990-university-of-brasilia"
PROJECT         = "CaVL-Doc_LA-CDIP_Sprint3_Staged5x5"

LOSS_LABELS = {
    "subcenter_cosface":  "Sub-Center CosFace",
    "subcenter_arcface":  "Sub-Center ArcFace",
    "contrastive":        "Contrastive",
    "cosface":            "CosFace",
    "arcface":            "ArcFace",
    "triplet":            "Triplet",
    "circle":             "Circle",
}
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#FF9DA6", "#9D755D"]

KNOWN_LOSSES = sorted(LOSS_LABELS.keys(), key=len, reverse=True)

ALL_SPLITS = [0, 1, 2, 3, 4]
FASE1_EPOCHS = 10


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _api() -> wandb.Api:
    return wandb.Api(timeout=120)


def _parse_run(name: str) -> Optional[dict]:
    """Extrai split, loss, phase do nome da run Sprint3."""
    m = re.match(r"Sprint3_S(\d+)_(.+?)_seed\d+_(fase\d+.*?)(?:_E\d+)?$", name)
    if not m:
        return None
    split_idx = int(m.group(1))
    loss_raw  = m.group(2)
    phase_raw = m.group(3).lower()

    loss = next((l for l in KNOWN_LOSSES if l == loss_raw), loss_raw)

    if "fase1" in phase_raw:
        phase = "fase1"
    elif "fase2" in phase_raw and "profon" in phase_raw:
        phase = "fase2_on"
    elif "fase2" in phase_raw and "profoff" in phase_raw:
        phase = "fase2_off"
    else:
        return None

    return {"split": split_idx, "loss": loss, "phase": phase}


def _fetch_history(run) -> List[dict]:
    """Retorna lista de {epoch, eer} da run, usando scan_history."""
    rows = []
    try:
        for row in run.scan_history(keys=["val/eer", "epoch"]):
            eer   = row.get("val/eer")
            epoch = row.get("epoch")
            if eer is None or epoch is None:
                continue
            try:
                rows.append({"epoch": int(float(epoch)), "eer": float(eer)})
            except (TypeError, ValueError):
                continue
    except Exception as e:
        print(f"    [WARN] scan_history falhou: {e}")
    return rows


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data() -> pd.DataFrame:
    print(f"Buscando runs em {ENTITY}/{PROJECT}...")
    runs = list(_api().runs(f"{ENTITY}/{PROJECT}"))
    print(f"  {len(runs)} runs encontradas.")

    records = []
    for run in runs:
        meta = _parse_run(run.name or "")
        if meta is None:
            continue

        print(f"  → {run.name}")
        history = _fetch_history(run)
        if not history:
            # Fallback: usa só o summary
            try:
                s = dict(run.summary)
                eer = None
                for key in ["val/best_eer", "best_eer", "val/eer"]:
                    if key in s:
                        try:
                            eer = float(s[key])
                            break
                        except (TypeError, ValueError):
                            pass
                if eer is not None:
                    history = [{"epoch": 0, "eer": eer}]
            except Exception:
                pass

        for h in history:
            # Para fase2, desloca época para continuar após fase1
            epoch_offset = FASE1_EPOCHS if meta["phase"] != "fase1" else 0
            records.append({
                "split":    meta["split"],
                "loss":     meta["loss"],
                "phase":    meta["phase"],
                "epoch":    h["epoch"] + epoch_offset,
                "eer":      h["eer"],
                "run_name": run.name,
            })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _b64_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _eer_curve_chart(df_split: pd.DataFrame, split_idx: int) -> str:
    """Curva EER × época para todas as losses de um split (fase1 + fase2_on)."""
    losses = sorted(df_split["loss"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Linha vertical separando fase 1 e fase 2
    ax.axvline(x=FASE1_EPOCHS, color="#aaa", linestyle="--", linewidth=0.8, label="Início Fase 2")

    for i, loss in enumerate(losses):
        color = PALETTE[i % len(PALETTE)]
        label = LOSS_LABELS.get(loss, loss)

        # Fase 1
        sub1 = df_split[(df_split["loss"] == loss) & (df_split["phase"] == "fase1")].sort_values("epoch")
        if not sub1.empty:
            ax.plot(sub1["epoch"], sub1["eer"] * 100, color=color, linewidth=1.5,
                    label=label, marker="o", markersize=3)
            # Marca o melhor
            best_row = sub1.loc[sub1["eer"].idxmin()]
            ax.scatter(best_row["epoch"], best_row["eer"] * 100,
                       color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5)

        # Fase 2 ON (continuação)
        sub2 = df_split[(df_split["loss"] == loss) & (df_split["phase"] == "fase2_on")].sort_values("epoch")
        if not sub2.empty:
            # Conecta ao último ponto da fase 1 se possível
            if not sub1.empty:
                connect = pd.concat([sub1.tail(1), sub2])
                ax.plot(connect["epoch"], connect["eer"] * 100,
                        color=color, linewidth=1.5, linestyle="--")
            else:
                ax.plot(sub2["epoch"], sub2["eer"] * 100,
                        color=color, linewidth=1.5, linestyle="--")
            best_row2 = sub2.loc[sub2["eer"].idxmin()]
            ax.scatter(best_row2["epoch"], best_row2["eer"] * 100,
                       color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5,
                       marker="D")

    ax.set_xlabel("Época (global)")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"Split {split_idx} — Curva EER por Loss (● melhor fase 1 / ◆ melhor fase 2)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64_png(fig)


def _best_epoch_chart(summary_df: pd.DataFrame, split_idx: int) -> str:
    """Gráfico de barras: melhor EER por loss (fase1 vs fase2_on), com anotação de época."""
    losses = summary_df["loss"].unique()
    x      = np.arange(len(losses))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(losses) * 1.4), 4))

    for j, (phase, label, hatch) in enumerate([("fase1", "Fase 1 (10 ép.)", ""),
                                                ("fase2_on", "Fase 2 — com teacher", "//")]):
        vals, epochs = [], []
        for loss in losses:
            row = summary_df[(summary_df["loss"] == loss) & (summary_df["phase"] == phase)]
            if row.empty:
                vals.append(0); epochs.append(None)
            else:
                vals.append(float(row["best_eer"].iloc[0]) * 100)
                epochs.append(int(row["best_epoch"].iloc[0]))

        bars = ax.bar(x + j * width, vals, width, label=label,
                      color=PALETTE[j], hatch=hatch, edgecolor="white", linewidth=0.5)
        for bar, eer_v, ep in zip(bars, vals, epochs):
            if ep is not None and eer_v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f"{eer_v:.1f}%\né{ep}", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Melhor EER (%)")
    ax.set_title(f"Split {split_idx} — Melhor EER por Loss e Fase")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([LOSS_LABELS.get(l, l) for l in losses], fontsize=8, rotation=15, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _b64_png(fig)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Georgia', serif; background: #f7f7f7; color: #222; padding: 40px; max-width: 1200px; margin: auto; }
h1 { font-size: 1.5rem; margin-bottom: 4px; color: #1a1a2e; }
.subtitle { color: #666; font-size: 0.87rem; margin-bottom: 32px; }
h2 { font-size: 1.05rem; margin: 0 0 6px; color: #1a1a2e; border-bottom: 2px solid #4C78A8; padding-bottom: 4px; }
h3 { font-size: 0.9rem; margin: 14px 0 5px; color: #444; }
.section { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 20px 24px; margin-bottom: 24px; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-bottom: 16px; }
th { background: #1a1a2e; color: #fff; padding: 6px 10px; text-align: left; }
td { padding: 5px 10px; border-bottom: 1px solid #e8e8e8; }
tr:nth-child(even) td { background: #f4f4f9; }
tr:hover td { background: #eef2ff; }
.best { font-weight: bold; color: #1a6e37; }
.worst { color: #999; }
.epoch-tag { font-size: 0.72rem; color: #777; margin-left: 4px; }
.chart { margin: 12px 0 18px; }
.chart img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.no-data { color: #999; font-style: italic; font-size: 0.83rem; padding: 8px 0; }
.winner-badge { display: inline-block; background: #d4edda; color: #155724; border: 1px solid #c3e6cb;
    border-radius: 4px; font-size: 0.7rem; padding: 1px 7px; margin-left: 6px; vertical-align: middle; }
footer { font-size: 0.72rem; color: #aaa; margin-top: 32px; text-align: center; }
"""


def _chart_html(b64: str) -> str:
    if not b64:
        return ""
    return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="chart"></div>'


def _split_table_html(summary_df: pd.DataFrame) -> str:
    """Tabela: loss × fase com (melhor EER, época)."""
    phases = [("fase1", "Fase 1 (10 ép.)"), ("fase2_on", "Fase 2 — com teacher"), ("fase2_off", "Fase 2 — sem teacher")]
    losses = sorted(summary_df["loss"].unique(), key=lambda l: summary_df[summary_df["loss"] == l]["best_eer"].min())

    # Determina o melhor EER global para destacar
    global_best = summary_df["best_eer"].min()

    rows_html = []
    for loss in losses:
        label  = LOSS_LABELS.get(loss, loss)
        # Melhor entre todas as fases para este loss
        loss_best = summary_df[summary_df["loss"] == loss]["best_eer"].min()
        is_winner = abs(loss_best - global_best) < 1e-6

        badge = '<span class="winner-badge">★ melhor</span>' if is_winner else ""
        cells = [f"<td>{label}{badge}</td>"]
        for phase, _ in phases:
            row = summary_df[(summary_df["loss"] == loss) & (summary_df["phase"] == phase)]
            if row.empty:
                cells.append("<td>—</td>")
            else:
                eer = float(row["best_eer"].iloc[0])
                ep  = int(row["best_epoch"].iloc[0])
                eer_pct = f"{eer * 100:.2f}%"
                css_cls = "best" if abs(eer - global_best) < 1e-6 else ""
                cells.append(f'<td class="{css_cls}">{eer_pct} <span class="epoch-tag">época {ep}</span></td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    header = "<tr><th>Loss</th>" + "".join(f"<th>{ph}</th>" for _, ph in phases) + "</tr>"
    return f"<table>{header}{''.join(rows_html)}</table>"


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

def build_html(df: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = []

    if df.empty:
        sections.append('<div class="section"><p class="no-data">Nenhum dado encontrado no W&B.</p></div>')
    else:
        # Calcula summary: melhor EER e época por (split, loss, phase)
        summary = (
            df.groupby(["split", "loss", "phase"])
            .apply(lambda g: pd.Series({
                "best_eer":   g["eer"].min(),
                "best_epoch": int(g.loc[g["eer"].idxmin(), "epoch"]),
                "n_epochs":   g["epoch"].nunique(),
            }))
            .reset_index()
        )

        # Seção global: melhor loss por split (resumo)
        winner_rows = []
        for split_idx in ALL_SPLITS:
            s = summary[summary["split"] == split_idx]
            if s.empty:
                continue
            best_row = s.loc[s["best_eer"].idxmin()]
            winner_rows.append({
                "Split":        split_idx,
                "Melhor Loss":  LOSS_LABELS.get(best_row["loss"], best_row["loss"]),
                "Fase":         best_row["phase"].replace("_", " "),
                "Melhor EER":   f'{best_row["best_eer"] * 100:.2f}%',
                "Época":        int(best_row["best_epoch"]),
            })
        if winner_rows:
            winner_df = pd.DataFrame(winner_rows)
            sections.append(f"""
<div class="section">
  <h2>Resumo — Melhor Loss por Split</h2>
  <p style="font-size:.85rem;color:#555;margin-bottom:12px;">
    Para cada split, a loss e época que atingiram o menor EER considerando todas as fases.
  </p>
  {winner_df.to_html(index=False, classes="table", border=0, escape=True)}
</div>""")

        # Seção por split
        for split_idx in ALL_SPLITS:
            df_s  = df[df["split"] == split_idx]
            sum_s = summary[summary["split"] == split_idx]
            if df_s.empty:
                continue

            curve_chart = _eer_curve_chart(df_s, split_idx)
            bar_chart   = _best_epoch_chart(sum_s, split_idx)
            table_html  = _split_table_html(sum_s)

            sections.append(f"""
<div class="section">
  <h2>Split {split_idx}</h2>
  <h3>Melhor EER por Loss e Fase</h3>
  {table_html}
  <h3>Barras: EER com anotação de época</h3>
  {_chart_html(bar_chart)}
  <h3>Curva de EER por época</h3>
  {_chart_html(curve_chart)}
</div>""")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Sprint 3 — Análise de Épocas</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>Sprint 3 LA-CDIP — Melhor Época por Loss/Split</h1>
  <p class="subtitle">Gerado em {now} · {ENTITY}/{PROJECT}</p>
  {''.join(sections)}
  <footer>generate_sprint3_epoch_analysis.py · CaVL-Doc</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = collect_data()

    print("Gerando HTML...")
    html = build_html(df)
    output.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
