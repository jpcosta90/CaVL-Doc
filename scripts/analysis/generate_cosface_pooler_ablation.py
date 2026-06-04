#!/usr/bin/env python3
"""
Ablação de arquitetura de pooler para Sub-Center CosFace.

Compara: Mean Pool, 1 Query, 2 Queries, 4 Queries.
Para cada split e pooler mostra lado a lado:
  - Val (treino): EER do subset de validação durante o treino
  - Full Eval:    EER calculado sobre os pares completos (eval_lacdip_full.py)

Projetos consultados:
  - Treino Val (attention): CaVL-Doc_LA-CDIP_Sprint3b_s32_k3
  - Treino Val (mean pool): CaVL-Doc_LA-CDIP_Ablation_MeanPool
  - Full Eval (attention):  CaVL-Doc_LA-CDIP_FullEval
  - Full Eval (mean pool):  CaVL-Doc_LA-CDIP_FullEval_MeanPool

Uso:
    python scripts/analysis/generate_cosface_pooler_ablation.py
    python scripts/analysis/generate_cosface_pooler_ablation.py --output results/cosface_pooler_ablation.html
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from generate_paper_results_html import (
    ENTITY, CSS,
    _fetch_runs, _scalar, _to_scalar, _eer, _fmt_eer, _fmt_delta,
    _table_html, _chart_html, _grouped_bar_chart, _b64_png,
    _loss_from_name, _epoch_count_from_name,
)

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT  = WORKSPACE_ROOT / "results" / "cosface_pooler_ablation.html"

PROJECT_SPRINT3B           = "CaVL-Doc_LA-CDIP_Sprint3b_s32_k3"
PROJECT_ABLATION_MEANPOOL  = "CaVL-Doc_LA-CDIP_Ablation_MeanPool"
PROJECT_FULL_EVAL          = "CaVL-Doc_LA-CDIP_FullEval"
PROJECT_FULL_EVAL_MEANPOOL = "CaVL-Doc_LA-CDIP_FullEval_MeanPool"

# Ordem canônica de exibição
POOLER_CONFIGS = ["Mean Pool", "1 Query", "2 Queries", "4 Queries"]
ALL_SPLITS     = [0, 1, 2, 3, 4]


def _pooler_label(pooler_type: str, num_queries) -> str:
    if str(pooler_type).strip() == "mean":
        return "Mean Pool"
    nq = int(num_queries or 1)
    return f"{nq} {'Query' if nq == 1 else 'Queries'}"


# ---------------------------------------------------------------------------
# Extração de records
# ---------------------------------------------------------------------------

def _extract_training_val(runs: List, force_pooler: Optional[str] = None) -> pd.DataFrame:
    """
    Extrai EER de validação (subset de treino) de runs Sprint3b/Ablation.
    Filtra apenas subcenter_cosface. Prefere runs noinit quando disponíveis.
    """
    records = []
    for r in runs:
        name = r.name or ""
        eer  = _eer(r)
        if eer is None:
            continue

        loss = _loss_from_name(name)
        if loss != "subcenter_cosface":
            continue

        if force_pooler:
            label = force_pooler
        else:
            cfg = {}
            try:
                cfg = dict(r.config)
            except Exception:
                pass
            label = _pooler_label(cfg.get("pooler_type", "attention"),
                                   cfg.get("num_queries", 1))

        m = re.search(r"_S(\d+)_", name)
        split = int(m.group(1)) if m else None

        nl = name.lower()
        if "fase2_profon" in nl:
            phase = "phase2_on"
        elif "fase2_profoff" in nl:
            phase = "phase2_off"
        elif "fase1" in nl:
            phase = "phase1"
        elif "prof_off" in nl:
            epochs = _epoch_count_from_name(name)
            phase  = "phase1" if (epochs < 0 or epochs > 6) else "phase2_off"
        else:
            continue

        records.append({
            "pooler":     label,
            "split":      split,
            "phase":      phase,
            "eer":        eer,
            "is_noinit":  "noinit" in nl,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Descartar versões sem noinit quando há versões noinit para o mesmo pooler
    poolers_noinit = set(df[df["is_noinit"]]["pooler"].unique())
    if poolers_noinit:
        df = df[~((df["pooler"].isin(poolers_noinit)) & (~df["is_noinit"]))].reset_index(drop=True)

    return df[["pooler", "split", "phase", "eer"]]


def _extract_full_eval(runs: List, force_pooler: Optional[str] = None) -> pd.DataFrame:
    """
    Extrai EER de avaliação completa de runs FullEval_.
    Filtra apenas subcenter_cosface.
    """
    records = []
    for r in runs:
        name = r.name or ""
        if not name.startswith("FullEval_"):
            continue

        s = {}
        try:
            s = dict(r.summary)
        except Exception:
            pass
        eer = _scalar(s.get("val/eer"))
        if eer is None:
            continue

        cfg = {}
        try:
            cfg = dict(r.config)
        except Exception:
            pass

        loss = cfg.get("loss") or _loss_from_name(name)
        if loss != "subcenter_cosface":
            continue

        if force_pooler:
            label = force_pooler
        else:
            label = _pooler_label(cfg.get("pooler_type", "attention"),
                                   cfg.get("num_queries", 1))

        split = cfg.get("split")
        if split is None:
            m = re.search(r"_S(\d+)_", name)
            split = int(m.group(1)) if m else None

        raw_phase = cfg.get("phase", "")
        nl = name.lower()
        if "fase2_profon" in nl or raw_phase == "fase2_profON":
            phase = "phase2_on"
        elif "fase2_profoff" in nl or raw_phase == "fase2_profOFF":
            phase = "phase2_off"
        else:
            phase = "phase1"

        records.append({"pooler": label, "split": split, "phase": phase, "eer": eer})

    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------

def _build_pooler_ablation(
    runs_sprint3b:          List,
    runs_ablation_meanpool: List,
    runs_full_eval:         List,
    runs_full_eval_meanpool:List,
) -> Tuple:
    """
    Retorna:
      t1, t2_on, t2_off, t3  — tabelas por estágio/melhor
      c1, c2_on, c2_off, c3  — gráficos correspondentes
      is_partial, missing
    """
    df_val = pd.concat([
        _extract_training_val(runs_sprint3b),
        _extract_training_val(runs_ablation_meanpool, force_pooler="Mean Pool"),
    ], ignore_index=True)

    df_full = pd.concat([
        _extract_full_eval(runs_full_eval),
        _extract_full_eval(runs_full_eval_meanpool, force_pooler="Mean Pool"),
    ], ignore_index=True)

    if df_val.empty and df_full.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, "", "", "", "", False, list(ALL_SPLITS)

    found_splits = sorted(set(
        (df_val["split"].dropna().unique().astype(int).tolist() if not df_val.empty else []) +
        (df_full["split"].dropna().unique().astype(int).tolist() if not df_full.empty else [])
    ))
    missing    = sorted(set(ALL_SPLITS) - set(found_splits))
    is_partial = bool(missing)

    # ---- helpers ----

    def _cell(df: pd.DataFrame, pooler: str, split: int,
               phase: Optional[str]) -> Optional[float]:
        """EER mínimo para (pooler, split[, phase])."""
        if df.empty:
            return None
        mask = (df["pooler"] == pooler) & (df["split"] == split)
        if phase:
            mask = mask & (df["phase"] == phase)
        sub = df[mask]["eer"]
        return float(sub.min()) if not sub.empty else None

    def _joint_table(phase: Optional[str]) -> pd.DataFrame:
        """Tabela: pooler × split, com Val e Full Eval por split."""
        rows = []
        for pooler in POOLER_CONFIGS:
            row: dict = {"Pooler": pooler}
            val_vals, full_vals = [], []
            for sp in found_splits:
                v_val  = _cell(df_val,  pooler, sp, phase)
                v_full = _cell(df_full, pooler, sp, phase)
                row[f"S{sp} Val"]  = _fmt_eer(v_val)
                row[f"S{sp} Full"] = _fmt_eer(v_full)
                if v_val  is not None: val_vals.append(v_val)
                if v_full is not None: full_vals.append(v_full)
            row["Média Val"]  = _fmt_eer(np.mean(val_vals)  if val_vals  else None)
            row["Média Full"] = _fmt_eer(np.mean(full_vals) if full_vals else None)
            rows.append(row)
        return pd.DataFrame(rows)

    def _joint_chart(phase: Optional[str], title: str) -> str:
        val_means, full_means = [], []
        for pooler in POOLER_CONFIGS:
            vv = [_cell(df_val,  pooler, sp, phase) for sp in found_splits]
            fv = [_cell(df_full, pooler, sp, phase) for sp in found_splits]
            vv = [x for x in vv if x is not None]
            fv = [x for x in fv if x is not None]
            val_means.append(np.mean(vv)  if vv  else None)
            full_means.append(np.mean(fv) if fv else None)
        return _grouped_bar_chart(
            POOLER_CONFIGS,
            {"Val (treino)": val_means, "Full Eval": full_means},
            title,
        )

    table_p1    = _joint_table("phase1")
    chart_p1    = _joint_chart("phase1",    "CosFace — Estágio 1 (fase1)")
    table_p2_on = _joint_table("phase2_on")
    chart_p2_on = _joint_chart("phase2_on", "CosFace — Estágio 2 (com mineração)")
    table_p2_off= _joint_table("phase2_off")
    chart_p2_off= _joint_chart("phase2_off","CosFace — Estágio 2 (sem mineração)")
    table_best  = _joint_table(None)
    chart_best  = _joint_chart(None,        "CosFace — Melhor EER acumulado (todos os estágios)")

    return (table_p1, table_p2_on, table_p2_off, table_best,
            chart_p1, chart_p2_on, chart_p2_off, chart_best,
            is_partial, missing)


# ---------------------------------------------------------------------------
# Convergência por época — gráfico de linhas
# ---------------------------------------------------------------------------

def _build_convergence(
    runs_sprint3b: List,
    runs_ablation_meanpool: List,
) -> Tuple[str, pd.DataFrame]:
    """
    Busca o histórico epoch-a-epoch de val/eer para as runs de fase1 cosface.
    Retorna (chart_linhas, tabela_resumo).
    chart_linhas: linha por pooler, média dos splits, estrela na melhor época.
    tabela_resumo: pooler, n_splits, EER época 1, melhor EER, melhor época, Δ.
    """

    def _get_histories(runs: List, force_pooler: Optional[str] = None) -> Dict:
        """Retorna {pooler: {split: [eer_e1, eer_e2, ...]}}."""
        # Primeiro pass: quais poolers têm runs noinit?
        noinit_poolers: set = set()
        for r in runs:
            name = r.name or ""
            if _loss_from_name(name) != "subcenter_cosface":
                continue
            nl = name.lower()
            if "fase2" in nl or "fase1" not in nl:
                continue
            if "noinit" in nl:
                if force_pooler:
                    noinit_poolers.add(force_pooler)
                else:
                    cfg: dict = {}
                    try:
                        cfg = dict(r.config)
                    except Exception:
                        pass
                    noinit_poolers.add(
                        _pooler_label(cfg.get("pooler_type", "attention"),
                                      cfg.get("num_queries", 1))
                    )

        data: Dict = {}
        for r in runs:
            name = r.name or ""
            if _loss_from_name(name) != "subcenter_cosface":
                continue
            nl = name.lower()
            if "fase2" in nl or "fase1" not in nl:
                continue

            if force_pooler:
                label = force_pooler
            else:
                cfg = {}
                try:
                    cfg = dict(r.config)
                except Exception:
                    pass
                label = _pooler_label(cfg.get("pooler_type", "attention"),
                                      cfg.get("num_queries", 1))

            # Prefere noinit quando disponível
            if label in noinit_poolers and "noinit" not in nl:
                continue

            m = re.search(r"_S(\d+)_", name)
            split = int(m.group(1)) if m else -1

            try:
                hist = r.history(keys=["val/eer"], pandas=True)
                if hist.empty or "val/eer" not in hist.columns:
                    continue
                curve = hist["val/eer"].dropna().tolist()
                if not curve:
                    continue
            except Exception:
                continue

            data.setdefault(label, {})[split] = curve

        return data

    hist_att  = _get_histories(runs_sprint3b)
    hist_mean = _get_histories(runs_ablation_meanpool, force_pooler="Mean Pool")
    all_hist  = {**hist_mean, **hist_att}

    # Média por época entre splits
    mean_curves: Dict[str, List[float]] = {}
    for pooler, split_data in all_hist.items():
        series = list(split_data.values())
        if not series:
            continue
        min_len = min(len(s) for s in series)
        if min_len == 0:
            continue
        mean_curves[pooler] = [
            float(np.mean([s[i] for s in series if i < len(s)]))
            for i in range(min_len)
        ]

    # ---- gráfico de linhas ----
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    any_curve = False
    for i, pooler in enumerate(POOLER_CONFIGS):
        curve = mean_curves.get(pooler)
        if not curve:
            continue
        any_curve = True
        color   = palette[i % len(palette)]
        n_split = len(all_hist.get(pooler, {}))
        epochs  = list(range(1, len(curve) + 1))
        eer_pct = [e * 100 for e in curve]

        ax.plot(epochs, eer_pct, marker="o", markersize=4, linewidth=2,
                label=f"{pooler} (n={n_split})", color=color)

        # Marca a melhor época com estrela e anotação
        best_i   = int(np.argmin(eer_pct))
        best_val = eer_pct[best_i]
        best_ep  = epochs[best_i]
        ax.scatter([best_ep], [best_val], s=130, zorder=6, color=color,
                   marker="*", edgecolors="white", linewidths=0.8)
        ax.annotate(
            f"{best_val:.2f}%\né{best_ep}",
            xy=(best_ep, best_val),
            xytext=(6, -16), textcoords="offset points",
            fontsize=7, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
        )

    if not any_curve:
        plt.close(fig)
        return "", pd.DataFrame()

    n_epochs = max((len(c) for c in mean_curves.values()), default=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("EER (%)")
    ax.set_title("Fase 1 — Convergência por Pooler (média dos splits)", fontsize=11)
    ax.set_xticks(range(1, n_epochs + 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.1)
    fig.tight_layout()
    chart = _b64_png(fig)

    # ---- tabela resumo ----
    rows = []
    for pooler in POOLER_CONFIGS:
        curve = mean_curves.get(pooler)
        if not curve:
            rows.append({
                "Pooler": pooler, "Splits": "—",
                "EER Época 1": "—", "Melhor EER": "—",
                "Melhor Época": "—", "Δ (pp)": "—",
            })
            continue
        n_sp    = len(all_hist.get(pooler, {}))
        first   = curve[0]
        best    = min(curve)
        best_ep = curve.index(best) + 1
        delta   = (best - first) * 100
        rows.append({
            "Pooler":       pooler,
            "Splits":       n_sp,
            "EER Época 1":  f"{first * 100:.2f}%",
            "Melhor EER":   f"{best  * 100:.2f}%",
            "Melhor Época": best_ep,
            "Δ (pp)":       f"{delta:+.2f}",
        })

    return chart, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def build_html(
    t1: pd.DataFrame, t2_on: pd.DataFrame, t2_off: pd.DataFrame, t3: pd.DataFrame,
    c1: str, c2_on: str, c2_off: str, c3: str,
    c_conv: str, t_conv: pd.DataFrame,
    is_partial: bool, missing: List[int],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    partial_badge = (
        '<span class="partial-badge">⚠ Resultados parciais</span>' if is_partial else ""
    )
    missing_note = (
        f'<p class="missing-note">Splits pendentes: {missing}</p>' if missing else ""
    )

    col_note = (
        '<p class="desc" style="font-size:0.8rem;color:#666;">'
        '<strong>S<em>n</em> Val</strong> = EER do subset de validação durante o treino · '
        '<strong>S<em>n</em> Full</strong> = EER sobre pares completos (eval_lacdip_full.py)'
        '</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>CosFace — Ablação de Pooler</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>CosFace — Ablação de Arquitetura de Pooler</h1>
  <p class="subtitle">Gerado em {now} · W&B: {ENTITY}</p>

  <div class="section">
    <h2>Ablação: Mean Pool · 1 Query · 2 Queries · 4 Queries {partial_badge}</h2>
    <p class="desc">
      Loss <strong>Sub-Center CosFace</strong> (s=32, k=3), backbone InternVL3-2B congelado.
      Cada célula mostra o EER mínimo para aquele split: <em>Val</em> = subset de validação
      durante o treino; <em>Full</em> = avaliação sobre todos os pares CSV da LA-CDIP.
      Arquiteturas sem dados aparecem com "—".
    </p>
    {missing_note}
    {col_note}

    <h3>Estágio 1 — Pré-treinamento (fase1)</h3>
    {_chart_html(c1)}
    {_table_html(t1)}

    <h3>Estágio 2 — Com Mineração (fase2_profON)</h3>
    {_chart_html(c2_on)}
    {_table_html(t2_on)}

    <h3>Estágio 2 — Sem Mineração (fase2_profOFF)</h3>
    {_chart_html(c2_off)}
    {_table_html(t2_off)}

    <h3>Melhor EER Acumulado — min(fase1, fase2)</h3>
    <p class="desc">
      Para cada pooler e split: menor EER observado ao longo de todos os estágios.
    </p>
    {_chart_html(c3)}
    {_table_html(t3)}
  </div>

  <div class="section">
    <h2>Convergência — Fase 1 (por época)</h2>
    <p class="desc">
      EER de validação a cada época do Estágio 1 (pré-treinamento, 10 épocas),
      calculado como média entre os splits disponíveis para cada pooler.
      A <strong>estrela ★</strong> marca a época com melhor EER; a anotação indica
      o valor e a época exata. Permite comparar velocidade de convergência e
      ponto de partida vs. melhor resultado alcançado.
    </p>
    {_chart_html(c_conv)}
    <h3>Resumo</h3>
    {_table_html(t_conv)}
  </div>

  <footer>generate_cosface_pooler_ablation.py · CaVL-Doc</footer>
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

    print("Buscando runs no W&B...")
    runs_sprint3b           = _fetch_runs(PROJECT_SPRINT3B)
    runs_ablation_meanpool  = _fetch_runs(PROJECT_ABLATION_MEANPOOL)
    runs_full_eval          = _fetch_runs(PROJECT_FULL_EVAL)
    runs_full_eval_meanpool = _fetch_runs(PROJECT_FULL_EVAL_MEANPOOL)

    print("Construindo ablação de pooler (CosFace)...")
    (t1, t2_on, t2_off, t3,
     c1, c2_on, c2_off, c3,
     is_partial, missing) = _build_pooler_ablation(
        runs_sprint3b, runs_ablation_meanpool,
        runs_full_eval, runs_full_eval_meanpool,
    )

    print("Buscando histórico de épocas para gráfico de convergência...")
    c_conv, t_conv = _build_convergence(runs_sprint3b, runs_ablation_meanpool)

    if is_partial:
        print(f"  Parcial — splits pendentes: {missing}")

    html = build_html(
        t1, t2_on, t2_off, t3,
        c1, c2_on, c2_off, c3,
        c_conv, t_conv,
        is_partial, missing,
    )
    output.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
