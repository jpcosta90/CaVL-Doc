#!/usr/bin/env python3
"""
Gera relatório HTML com resultados para inclusão no paper sn-article.

Seções:
  0. Busca de Hiperparâmetros (Coarse + Fine Search) — comparação por budget de passos
  1. Comparação de Funções de Perda (LA-CDIP, sem hard mining)
  2. Configuração do Professor (sweep — sem resultados, só metodologia)
  3. Treinamento em Estágios com Currículo — dois sub-resultados:
       A. 10 épocas sem hard mining (comparação de losses)
       B. 5 épocas com/sem hard mining (efeito do minerador)
  4. Transferência entre Domínios LA-CDIP → RVL-CDIP (parcial)

Uso:
    python scripts/analysis/generate_paper_results_html.py
    python scripts/analysis/generate_paper_results_html.py --output results/paper_results.html
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
DEFAULT_OUTPUT  = WORKSPACE_ROOT / "results" / "paper_results.html"
ENTITY          = "jpcosta1990-university-of-brasilia"

PROJECTS = {
    "exp0_coarse_la": "CaVL-Doc_LA-CDIP_InternVL3-2B_Sweeps",
    "exp0_fine_la":   "CaVL-Doc_LA-CDIP_FineSearch",
    "exp1":           "CaVL-Doc_LA-CDIP_Sprint1_Top5Validation",
    "exp3":           "CaVL-Doc_LA-CDIP_Sprint3_Staged5x5",
    "exp4":           "CaVL-Doc_RVL_Sprint4_Transfer",
}

KNOWN_LOSSES = [
    "subcenter_cosface", "subcenter_arcface",
    "contrastive", "cosface", "arcface", "triplet", "circle",
]

LOSS_LABELS = {
    "subcenter_cosface":  "Sub-Center CosFace",
    "subcenter_arcface":  "Sub-Center ArcFace",
    "contrastive":        "Contrastive",
    "cosface":            "CosFace",
    "arcface":            "ArcFace",
    "triplet":            "Triplet",
    "circle":             "Circle",
    "unknown":            "Unknown",
}

ALL_SPLITS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _api() -> wandb.Api:
    return wandb.Api(timeout=60)


def _fetch_runs(project: str) -> List:
    try:
        runs = list(_api().runs(f"{ENTITY}/{project}"))
        print(f"  [{project}] {len(runs)} runs")
        return runs
    except Exception as e:
        print(f"  [{project}] ERRO: {e}")
        return []


def _summary(run) -> dict:
    try:
        return dict(run.summary)
    except Exception:
        return {}


def _scalar(v) -> Optional[float]:
    """Extract a float from a W&B summary value.

    W&B may return a SummarySubDict (not a real dict) with shape {'min': val}
    for metrics logged with wandb.log() — isinstance(v, dict) is False for it.
    """
    if v is None:
        return None
    # Try direct conversion first (plain int/float)
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    # Duck-type dict-like: wandb.old.summary.SummarySubDict
    if hasattr(v, "get"):
        for key in ("min", "last"):
            candidate = v.get(key)
            if candidate is not None:
                try:
                    return float(candidate)
                except (TypeError, ValueError):
                    pass
        if hasattr(v, "values"):
            for candidate in v.values():
                try:
                    return float(candidate)
                except (TypeError, ValueError):
                    pass
    return None


def _eer(run) -> Optional[float]:
    s = _summary(run)
    for key in ["val/best_eer", "best_eer", "val/eer", "eer"]:
        v = _scalar(s.get(key))
        if v is not None:
            return v
    return None


def _recall(run) -> Optional[float]:
    s = _summary(run)
    for key in ["val/recall_at_1", "recall_at_1"]:
        v = _scalar(s.get(key))
        if v is not None:
            return v
    return None


def _loss_from_name(name: str) -> str:
    name_lower = name.lower()
    for loss in sorted(KNOWN_LOSSES, key=len, reverse=True):
        if loss in name_lower:
            return loss
    return "unknown"


def _epoch_count_from_name(name: str) -> int:
    m = re.search(r"_E(\d+)$", name)
    return int(m.group(1)) if m else -1


def _config(run) -> dict:
    try:
        return dict(run.config)
    except Exception:
        return {}


def _steps_per_epoch(run) -> Optional[float]:
    s = _summary(run)
    epoch = _scalar(s.get("epoch"))
    step  = _scalar(s.get("step"))
    if epoch and step and epoch > 0:
        return step / epoch
    return None


def _created_at(run) -> float:
    try:
        return run.created_at or 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _b64_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _grouped_bar_chart(
    groups: List[str],
    series: Dict[str, List[Optional[float]]],
    title: str,
    ylabel: str = "EER (%)",
    highlight_lower: bool = True,
) -> str:
    n_groups  = len(groups)
    n_series  = len(series)
    x         = np.arange(n_groups)
    width     = 0.7 / max(n_series, 1)
    palette   = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.4), 4.5))
    for i, (label, vals) in enumerate(series.items()):
        ys     = [v * 100 if v is not None else 0.0 for v in vals]
        offset = (i - n_series / 2 + 0.5) * width
        bars   = ax.bar(x + offset, ys, width, label=label,
                        color=palette[i % len(palette)], edgecolor="white", linewidth=0.4)
        for bar, v, orig in zip(bars, ys, vals):
            if orig is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _b64_png(fig)


def _sweep_progression_chart(runs_coarse: List, runs_fine: List) -> str:
    """Scatter: trial index vs EER, separated by phase, with running-best line."""
    palette = {"Coarse Search": "#4C78A8", "Fine Search": "#F58518"}

    def _sorted_records(runs, label):
        recs = []
        for r in runs:
            eer = _eer(r)
            if eer is None:
                continue
            recs.append({"phase": label, "eer": eer * 100, "ts": _created_at(r)})
        recs.sort(key=lambda x: x["ts"])
        for i, rec in enumerate(recs):
            rec["idx"] = i + 1
        return recs

    coarse_recs = _sorted_records(runs_coarse, "Coarse Search")
    fine_recs   = _sorted_records(runs_fine,   "Fine Search")
    if not coarse_recs and not fine_recs:
        return ""

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, recs in [("Coarse Search", coarse_recs), ("Fine Search", fine_recs)]:
        if not recs:
            continue
        xs   = [r["idx"] for r in recs]
        ys   = [r["eer"]  for r in recs]
        color = palette[label]
        ax.scatter(xs, ys, label=label, color=color, alpha=0.65, s=22, zorder=3)
        # running best
        best = []
        cur  = float("inf")
        for y in ys:
            cur = min(cur, y)
            best.append(cur)
        ax.plot(xs, best, color=color, linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Trial (ordem cronológica)")
    ax.set_ylabel("EER (%)")
    ax.set_title("Progressão do Sweep — Coarse Search vs Fine Search (LA-CDIP)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64_png(fig)


def _scatter_steps_eer(records: List[dict], title: str) -> str:
    if not records:
        return ""
    df = pd.DataFrame(records).dropna(subset=["steps_per_epoch", "eer"])
    if df.empty:
        return ""
    losses  = sorted(df["loss"].unique())
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, loss in enumerate(losses):
        sub = df[df["loss"] == loss]
        ax.scatter(sub["steps_per_epoch"], sub["eer"] * 100,
                   label=LOSS_LABELS.get(loss, loss),
                   color=palette[i % len(palette)], alpha=0.7, s=25)
    ax.set_xlabel("Passos por época (steps/epoch)")
    ax.set_ylabel("EER (%)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64_png(fig)


# ---------------------------------------------------------------------------
# Experiment 0 — Coarse + Fine Search
# ---------------------------------------------------------------------------

def _build_exp0(runs_coarse: List, runs_fine: List) -> Tuple[str, pd.DataFrame]:
    """
    Retorna: chart_progression (progressão do sweep: trial vs EER), tabela_best.
    """
    chart_prog = _sweep_progression_chart(runs_coarse, runs_fine)

    STEPS_PER_EPOCH = {"Coarse Search": 50, "Fine Search": 100}

    def _extract(runs, phase_label):
        out = []
        for r in runs:
            loss = _loss_from_name(r.name or "")
            eer  = _eer(r)
            if eer is None:
                continue
            cfg = _config(r)
            out.append({
                "phase":  phase_label,
                "loss":   loss,
                "eer":    eer,
                "scale":  cfg.get("scale"),
                "margin": cfg.get("margin"),
                "run":    r.name,
            })
        return out

    all_recs = _extract(runs_coarse, "Coarse Search") + _extract(runs_fine, "Fine Search")

    rows = []
    if all_recs:
        df = pd.DataFrame(all_recs)
        for (phase, loss), grp in df.groupby(["phase", "loss"]):
            best = grp.loc[grp["eer"].idxmin()]
            rows.append({
                "Fase":           phase,
                "Loss Function":  LOSS_LABELS.get(loss, loss),
                "Steps/época":    STEPS_PER_EPOCH.get(phase, "—"),
                "Scale":          f'{best["scale"]:.1f}' if best["scale"] is not None else "—",
                "Margin":         f'{best["margin"]:.4f}' if best["margin"] is not None else "—",
                "Melhor EER":     _fmt_eer(best["eer"]),
            })
    table_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return chart_prog, table_df


# ---------------------------------------------------------------------------
# Experiment 1 — Loss Comparison (LA-CDIP, no hard mining)
# ---------------------------------------------------------------------------

def _build_exp1(runs: List) -> Tuple[pd.DataFrame, str]:
    """
    Última run (mais recente) por loss SEM professor (prof_off / prof_last5_off).
    """
    records = []
    for r in runs:
        name = r.name or ""
        if not name.startswith("Sprint1_"):
            continue
        name_l = name.lower()
        if "prof_on" in name_l or "prof_last5_on" in name_l:
            continue
        eer = _eer(r)
        if eer is None:
            continue
        records.append({
            "loss":       _loss_from_name(name),
            "eer":        eer,
            "recall":     _recall(r),
            "created_at": _created_at(r),
            "run":        name,
        })

    if not records:
        return pd.DataFrame(), ""

    # Preferir _main_memfix sobre _unb_memfix; depois mais recente
    def _run_priority(name: str) -> int:
        if "_main_memfix" in name or "_main" in name:
            return 0
        if "_unb_memfix" in name or "_unb" in name:
            return 1
        return 2

    df = pd.DataFrame(records)
    df["priority"] = df["run"].apply(_run_priority)
    df = df.sort_values(["priority", "created_at"], ascending=[True, False])
    df_top = df.groupby("loss").first().reset_index()

    table_rows = []
    for _, row in df_top.sort_values("eer").iterrows():
        table_rows.append({
            "Loss Function":  LOSS_LABELS.get(row["loss"], row["loss"]),
            "EER":            _fmt_eer(row["eer"]),
            "Recall@1":       _fmt_eer(row["recall"]) if row["recall"] else "—",
            "Run":            row["run"],
        })
    table_df = pd.DataFrame(table_rows)

    df_s = df_top.sort_values("eer")
    losses = [LOSS_LABELS.get(l, l) for l in df_s["loss"]]
    chart = _grouped_bar_chart(losses, {"EER": list(df_s["eer"])},
                                "LA-CDIP: EER por função de perda (sem hard mining)")
    return table_df, chart


# ---------------------------------------------------------------------------
# Experiment 3 — Staged Curriculum (LA-CDIP)
# ---------------------------------------------------------------------------

def _build_exp3(runs: List) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, bool, List[int]]:
    """
    Retorna:
      - table_phase1: comparação de losses (10 épocas sem hard mining)
      - table_phase2: efeito do hard mining (prof_on vs prof_off, 5 épocas)
      - chart_phase1, chart_phase2
      - is_partial, missing_splits
    """
    records = []
    for r in runs:
        name = r.name or ""
        if not name.startswith("Sprint3_"):
            continue
        eer  = _eer(r)
        loss = _loss_from_name(name)
        m    = re.search(r"_S(\d+)_", name)
        split = int(m.group(1)) if m else None

        name_l = name.lower()
        if "fase1" in name_l:
            phase = "phase1"
        elif "fase2_profon" in name_l or "prof_on" in name_l:
            phase = "phase2_on"
        elif "fase2_profoff" in name_l:
            phase = "phase2_off"
        elif "prof_off" in name_l:
            epochs = _epoch_count_from_name(name)
            phase  = "phase1" if epochs < 0 or epochs > 6 else "phase2_off"
        else:
            continue

        records.append({"loss": loss, "split": split, "phase": phase,
                         "eer": eer, "run": name})

    if not records:
        return pd.DataFrame(), pd.DataFrame(), "", "", False, list(ALL_SPLITS)

    df = pd.DataFrame(records)
    completed = sorted(df["split"].dropna().unique().astype(int).tolist())
    missing   = sorted(set(ALL_SPLITS) - set(completed))
    is_partial = bool(missing)

    # --- Part A: phase1 por loss ---
    p1 = df[df["phase"] == "phase1"].groupby("loss")["eer"].agg(["mean", "std", "min"]).reset_index()
    rows_a = []
    for _, row in p1.sort_values("mean").iterrows():
        std = row["std"] if not np.isnan(row["std"]) else 0.0
        rows_a.append({
            "Loss Function": LOSS_LABELS.get(row["loss"], row["loss"]),
            "EER médio":     f'{row["mean"]*100:.2f}%',
            "± std":         f'{std*100:.2f} pp',
            "Melhor EER":    _fmt_eer(row["min"]),
        })
    table_p1 = pd.DataFrame(rows_a)

    labels_a = [LOSS_LABELS.get(r["loss"], r["loss"]) for _, r in p1.sort_values("mean").iterrows()]
    chart_p1 = _grouped_bar_chart(
        labels_a,
        {"EER médio": [r["mean"] for _, r in p1.sort_values("mean").iterrows()]},
        "Estágio 1 (10 épocas, sem mineração): EER por função de perda",
    )

    # --- Part B: phase2 prof_on vs prof_off por loss ---
    p2_on  = df[df["phase"] == "phase2_on"].groupby("loss")["eer"].mean()
    p2_off = df[df["phase"] == "phase2_off"].groupby("loss")["eer"].mean()
    all_losses_p2 = sorted(set(list(p2_on.index) + list(p2_off.index)))

    rows_b = []
    for loss in all_losses_p2:
        on_v  = p2_on.get(loss)
        off_v = p2_off.get(loss)
        rows_b.append({
            "Loss Function":     LOSS_LABELS.get(loss, loss),
            "Com Mineração":     _fmt_eer(on_v),
            "Sem Mineração":     _fmt_eer(off_v),
            "Δ Mineração (pp)":  _fmt_delta(on_v, off_v) if on_v and off_v else "—",
        })
    table_p2 = pd.DataFrame(rows_b)

    labels_b = [LOSS_LABELS.get(l, l) for l in all_losses_p2]
    series_b = {
        "Com Mineração":  [p2_on.get(l)  for l in all_losses_p2],
        "Sem Mineração":  [p2_off.get(l) for l in all_losses_p2],
    }
    chart_p2 = _grouped_bar_chart(
        labels_b, series_b,
        "Estágio 2 (5 épocas): Efeito do Hard Negative Mining",
    )

    return table_p1, table_p2, chart_p1, chart_p2, is_partial, missing


# ---------------------------------------------------------------------------
# Experiment 4 — Cross-Domain Transfer (RVL-CDIP)
# ---------------------------------------------------------------------------

def _build_exp4(runs: List) -> Tuple[pd.DataFrame, str, bool, List[int]]:
    records = []
    for r in runs:
        name = r.name or ""
        if not name.startswith("Sprint4_"):
            continue
        eer  = _eer(r)
        loss = _loss_from_name(name)
        m    = re.search(r"_S(\d+)_", name)
        split = int(m.group(1)) if m else None
        mode  = "transfer" if "_transfer_" in name else ("direct" if "_direct_" in name else None)
        if mode is None:
            continue
        records.append({"loss": loss, "split": split, "mode": mode, "eer": eer, "run": name})

    if not records:
        return pd.DataFrame(), "", False, list(ALL_SPLITS)

    df = pd.DataFrame(records)
    completed = sorted(df["split"].dropna().unique().astype(int).tolist())
    missing   = sorted(set(ALL_SPLITS) - set(completed))
    is_partial = bool(missing)

    rows = []
    for (loss, split), grp in df.groupby(["loss", "split"]):
        by_mode = grp.set_index("mode")["eer"].to_dict()
        rows.append({
            "Loss Function":    LOSS_LABELS.get(str(loss), str(loss)),
            "Split":            int(split) if split is not None else "—",
            "Transfer EER":     _fmt_eer(by_mode.get("transfer")),
            "Treinamento Direto": _fmt_eer(by_mode.get("direct")),
            "Δ Transfer (pp)":  _fmt_delta(by_mode.get("transfer"), by_mode.get("direct")),
        })
    table_df = pd.DataFrame(rows)

    tr = df[df["mode"] == "transfer"].groupby("loss")["eer"].mean()
    di = df[df["mode"] == "direct"].groupby("loss")["eer"].mean()
    all_losses = sorted(set(list(tr.index) + list(di.index)))
    chart = _grouped_bar_chart(
        [LOSS_LABELS.get(l, l) for l in all_losses],
        {"Transfer Learning":   [tr.get(l) for l in all_losses],
         "Treinamento Direto":  [di.get(l) for l in all_losses]},
        "RVL-CDIP: Transfer Learning vs. Treinamento Direto",
    )
    return table_df, chart, is_partial, missing


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_eer(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.2f}%"


def _fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "—"
    delta = (b - a) * 100
    sign  = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f} pp"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Georgia', serif; background: #f7f7f7; color: #222; padding: 40px; max-width: 1100px; margin: auto; }
h1 { font-size: 1.6rem; margin-bottom: 4px; color: #1a1a2e; }
.subtitle { color: #666; font-size: 0.88rem; margin-bottom: 36px; }
h2 { font-size: 1.1rem; margin: 0 0 6px; color: #1a1a2e; border-bottom: 2px solid #4C78A8; padding-bottom: 4px; }
h3 { font-size: 0.93rem; margin: 18px 0 6px; color: #444; }
p.desc { font-size: 0.86rem; color: #555; margin-bottom: 14px; line-height: 1.55; }
.partial-badge { display:inline-block; background:#fff3cd; color:#856404; border:1px solid #ffc107;
    border-radius:4px; font-size:0.73rem; padding:2px 8px; margin-left:8px; vertical-align:middle; }
.missing-note { font-size:0.8rem; color:#856404; margin-bottom:10px; }
table { border-collapse:collapse; width:100%; font-size:0.81rem; margin-bottom:18px; }
th { background:#1a1a2e; color:#fff; padding:7px 12px; text-align:left; }
td { padding:6px 12px; border-bottom:1px solid #e0e0e0; }
tr:nth-child(even) td { background:#f2f2f8; }
tr:hover td { background:#eef2ff; }
.chart { margin:14px 0 20px; }
.chart img { max-width:100%; border:1px solid #ddd; border-radius:4px; }
.section { background:#fff; border:1px solid #e0e0e0; border-radius:6px; padding:22px 26px; margin-bottom:26px; }
.no-data { color:#999; font-style:italic; font-size:0.84rem; padding:10px 0; }
.method-box { background:#f0f4ff; border-left:4px solid #4C78A8; padding:12px 16px;
    font-size:0.84rem; color:#333; line-height:1.6; border-radius:0 4px 4px 0; margin-bottom:12px; }
footer { font-size:0.73rem; color:#aaa; margin-top:36px; text-align:center; }
"""


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<p class="no-data">Nenhum dado disponível no W&B.</p>'
    return df.to_html(index=False, classes="table", border=0, escape=True)


def _chart_html(b64: str) -> str:
    if not b64:
        return ""
    return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="chart"></div>'


def build_html(
    exp0_prog: str, exp0_table: pd.DataFrame,
    exp1_table: pd.DataFrame, exp1_chart: str,
    exp3_t1: pd.DataFrame, exp3_t2: pd.DataFrame,
    exp3_c1: str, exp3_c2: str, exp3_partial: bool, exp3_missing: List[int],
    exp4_table: pd.DataFrame, exp4_chart: str, exp4_partial: bool, exp4_missing: List[int],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def partial_badge(is_p): return '<span class="partial-badge">⚠ Resultados parciais</span>' if is_p else ""
    def missing_note(m): return f'<p class="missing-note">Splits pendentes: {m}</p>' if m else ""

    sections = []

    # --- Exp 0 ---
    sections.append(f"""
<div class="section">
  <h2>0. Busca de Hiperparâmetros (Coarse Search + Fine Search)</h2>
  <p class="desc">
    A busca de hiperparâmetros foi conduzida em dois estágios para o dataset LA-CDIP.
    A <strong>Coarse Search</strong> explorou amplamente o espaço de configurações com
    <strong>50 steps por época</strong>.
    A <strong>Fine Search</strong> refinou os melhores candidatos com
    <strong>100 steps por época</strong>.
    O gráfico mostra a progressão de cada fase ao longo dos trials (ordem cronológica),
    com a linha tracejada indicando o melhor EER acumulado até cada trial.
  </p>
  {_chart_html(exp0_prog)}
  <h3>Melhor EER por loss e fase de busca</h3>
  {_table_html(exp0_table)}
</div>""")

    # --- Exp 1 ---
    sections.append(f"""
<div class="section">
  <h2>1. Comparação de Funções de Perda — LA-CDIP</h2>
  <p class="desc">
    Avaliação das funções de perda métrica em condições controladas, sem mineração ativa de negativos difíceis.
    Resultados calculados sobre as últimas 5 execuções por função de perda com o mesmo protocolo de treinamento.
    Valores reportados como média ± desvio padrão do EER entre runs.
  </p>
  {_chart_html(exp1_chart)}
  {_table_html(exp1_table)}
</div>""")

    # --- Exp 2 ---
    sections.append(f"""
<div class="section">
  <h2>2. Configuração do Professor (Hard Negative Mining)</h2>
  <div class="method-box">
    A rede professor é uma política de seleção de exemplos difíceis treinada por policy gradient (REINFORCE).
    Para cada batch de treinamento, o professor recebe como estado a distribuição de distâncias entre
    embeddings do pool de candidatos e seleciona os exemplos com maior potencial de aprendizado.
    Um <em>sweep</em> bayesiano foi conduzido para otimizar os hiperparâmetros do professor:
    taxa de aprendizado, tamanho do pool de candidatos, baseline alpha e coeficiente de entropia.
    Os resultados desta etapa determinaram a configuração utilizada nos experimentos subsequentes.
  </div>
</div>""")

    # --- Exp 3 ---
    sections.append(f"""
<div class="section">
  <h2>3. Treinamento em Estágios com Currículo — LA-CDIP {partial_badge(exp3_partial)}</h2>
  <p class="desc">
    Protocolo em dois estágios: (1) pré-treinamento de 10 épocas sem mineração ativa, seguido de
    (2) 5 épocas com e sem mineração ativa para avaliar o efeito do professor.
  </p>
  {missing_note(exp3_missing)}
  <h3>Estágio 1 — Pré-treinamento (10 épocas, sem hard mining)</h3>
  <p class="desc">Comparação das funções de perda no estágio de pré-treinamento.</p>
  {_chart_html(exp3_c1)}
  {_table_html(exp3_t1)}
  <h3>Estágio 2 — Efeito do Hard Negative Mining (5 épocas)</h3>
  <p class="desc">
    Comparação entre treinamento com mineração ativa (professor ON) e sem mineração (professor OFF),
    partindo do mesmo checkpoint do Estágio 1. Δ Mineração indica o ganho em pontos percentuais.
  </p>
  {_chart_html(exp3_c2)}
  {_table_html(exp3_t2)}
</div>""")

    # --- Exp 4 ---
    sections.append(f"""
<div class="section">
  <h2>4. Transferência entre Domínios: LA-CDIP → RVL-CDIP {partial_badge(exp4_partial)}</h2>
  <p class="desc">
    Comparação entre inicialização com pesos pré-treinados no LA-CDIP (Transfer Learning) e
    treinamento do zero no RVL-CDIP (Treinamento Direto).
    Δ Transfer indica o ganho em pontos percentuais do transfer learning sobre o treinamento direto —
    valores negativos indicam que o transfer supera o treinamento direto.
  </p>
  {missing_note(exp4_missing)}
  {_chart_html(exp4_chart)}
  {_table_html(exp4_table)}
</div>""")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>CaVL-Doc — Resultados Experimentais</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>CaVL-Doc — Resultados Experimentais</h1>
  <p class="subtitle">Gerado em {now} · Entidade W&B: {ENTITY}</p>
  {''.join(sections)}
  <footer>generate_paper_results_html.py · CaVL-Doc</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Gera relatório HTML de resultados para o paper.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Buscando runs no W&B...")
    runs0_cla = _fetch_runs(PROJECTS["exp0_coarse_la"])
    runs0_fla = _fetch_runs(PROJECTS["exp0_fine_la"])
    runs1     = _fetch_runs(PROJECTS["exp1"])
    runs3     = _fetch_runs(PROJECTS["exp3"])
    runs4     = _fetch_runs(PROJECTS["exp4"])

    print("Construindo seções...")
    exp0_prog, exp0_table              = _build_exp0(runs0_cla, runs0_fla)
    exp1_table, exp1_chart             = _build_exp1(runs1)
    exp3_t1, exp3_t2, exp3_c1, exp3_c2, exp3_partial, exp3_missing = _build_exp3(runs3)
    exp4_table, exp4_chart, exp4_partial, exp4_missing = _build_exp4(runs4)

    if exp3_partial:
        print(f"  Exp. 3 PARCIAL — splits pendentes: {exp3_missing}")
    if exp4_partial:
        print(f"  Exp. 4 PARCIAL — splits pendentes: {exp4_missing}")

    html = build_html(
        exp0_prog, exp0_table,
        exp1_table, exp1_chart,
        exp3_t1, exp3_t2, exp3_c1, exp3_c2, exp3_partial, exp3_missing,
        exp4_table, exp4_chart, exp4_partial, exp4_missing,
    )

    output.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
