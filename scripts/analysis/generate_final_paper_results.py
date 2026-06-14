#!/usr/bin/env python3
"""
Gera paper_results_final.html — versão curada para o paper.

Seções:
  1. Full Eval — Comparação de Losses (Sprint3b, attention q=1, pares completos)
  2. Pooler Query Ablation (q=1 vs q=2)
  3. Prompt Effect Ablation (mean pool / q=1 / cross-modal × P₀ / Pᵣ)
  4. Baselines — Embedding
  5. Baselines — VLM

Uso:
    python scripts/analysis/generate_final_paper_results.py
    python scripts/analysis/generate_final_paper_results.py --output results/paper_results_final.html
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from generate_paper_results_html import (
    ENTITY, PROJECTS, CSS, ALL_SPLITS,
    POOLER_VARIANT_LABELS, LOSS_LABELS,
    _fetch_runs,
    _build_baselines_embedding, _build_baselines_vlm,
    _pooler_variant_from_wandb_name,
    _summary, _scalar,
    _grouped_bar_chart,
    _table_html, _chart_html, _fmt_eer, _fmt_delta,
    _loss_from_name, _to_scalar,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = WORKSPACE_ROOT / "results" / "paper_results_final.html"

# Variants excluded from the loss comparison (handled in other sections)
from generate_paper_results_html import ABLATION_VARIANTS

# All known losses — longest first so regex matches greedily correct
_KNOWN_LOSSES_RE = sorted(
    ["subcenter_cosface", "subcenter_arcface", "contrastive",
     "cosface", "arcface", "triplet", "circle"],
    key=len, reverse=True,
)

def _variant_from_any_loss_run(name: str) -> Optional[str]:
    """Extracts pooler variant from a FullEval run name for ANY loss.
    Returns variant string ("" for baseline, e.g. "mean" for mean pooler) or None.

    _pooler_variant_from_wandb_name only handles subcenter_cosface; this version
    covers all losses so ArcFace/triplet/etc. ablation runs are also filtered.
    """
    nl = name.lower()
    for loss in _KNOWN_LOSSES_RE:
        m = re.match(
            rf"fulleval_sprint3b_s\d+_{re.escape(loss)}_(.+)_fase\d",
            nl,
        )
        if m:
            return m.group(1).strip("_")
    return None  # baseline (old format or no variant tag)

# ---------------------------------------------------------------------------
# Loss comparison (replaces _build_full_eval with tighter Part C)
# ---------------------------------------------------------------------------

def _build_loss_comparison(runs: List):
    """Like _build_full_eval but Part C (Melhor EER Acumulado) is restricted to
    (loss, split) pairs that have runs in BOTH fase1 AND fase2.
    This prevents orphan fase1-only or fase2-only runs from contaminating Part C.
    Returns (table_p1, table_p2, table_p3, chart_p1, chart_p2, chart_p3,
             is_partial, missing, run_infos).
    """
    records: list[dict] = []
    run_infos: list[dict] = []

    for r in runs:
        name = r.name or ""
        if not name.startswith("FullEval_"):
            continue
        # Use general variant detector (not cosface-only) so ArcFace/triplet
        # ablation runs (e.g. mean pooler trained with arcface) are also excluded.
        v = _variant_from_any_loss_run(name)
        if v is not None and v in ABLATION_VARIANTS:
            continue
        s   = _summary(r)
        eer = _scalar(s.get("val/eer"))
        if eer is None:
            continue
        cfg   = dict(r.config) if hasattr(r, "config") else {}
        loss  = cfg.get("loss") or _loss_from_name(name)
        split = cfg.get("split")
        if split is None:
            m2 = re.search(r"_S(\d+)_", name)
            split = int(m2.group(1)) if m2 else None
        else:
            try:
                split = int(split)
            except (TypeError, ValueError):
                split = None
        nl = name.lower()
        raw_phase = cfg.get("phase", "")
        if "fase2_profon" in nl or raw_phase == "fase2_profON":
            phase = "phase2_on"
        elif "fase2_profoff" in nl or raw_phase == "fase2_profOFF":
            phase = "phase2_off"
        else:
            phase = "phase1"
        records.append({"loss": loss, "split": split, "phase": phase, "eer": eer})
        run_infos.append({
            "name": name, "url": _run_url(r),
            "split": split, "phase": phase, "eer": eer,
            "variant": loss,
        })

    if not records:
        empty = pd.DataFrame()
        return empty, empty, empty, "", "", "", False, list(ALL_SPLITS), [], [], [], []

    df = pd.DataFrame(records)
    completed  = sorted(df["split"].dropna().unique().astype(int).tolist())
    missing    = sorted(set(ALL_SPLITS) - set(completed))
    is_partial = bool(missing)

    def _stats_row(label, series):
        if series.empty:
            return {"Loss Function": label, "Média": "—", "Std": "—", "Mediana": "—", "Mín": "—", "Máx": "—"}
        return {
            "Loss Function": label,
            "Média":   _fmt_eer(series.mean()),
            "Std":     f"{series.std()*100:.2f} pp" if len(series) > 1 else "—",
            "Mediana": _fmt_eer(series.median()),
            "Mín":     _fmt_eer(series.min()),
            "Máx":     _fmt_eer(series.max()),
        }

    # Part A — fase1
    p1_df  = df[df["phase"] == "phase1"]
    p1_grp = p1_df.groupby("loss")["eer"].mean().sort_values()
    table_p1 = pd.DataFrame([
        _stats_row(LOSS_LABELS.get(l, l), p1_df[p1_df["loss"] == l]["eer"])
        for l in p1_grp.index
    ])
    chart_p1 = _grouped_bar_chart(
        [LOSS_LABELS.get(l, l) for l in p1_grp.index],
        {"EER médio": list(p1_grp.values)},
        "Full Eval — Fase 1 (pares completos)",
    )

    # Part B — fase2 profON vs profOFF
    p2on_df  = df[df["phase"] == "phase2_on"]
    p2off_df = df[df["phase"] == "phase2_off"]
    losses_b = sorted(set(p2on_df["loss"]) | set(p2off_df["loss"]))
    rows_b = []
    for l in losses_b:
        on_s  = p2on_df[p2on_df["loss"] == l]["eer"]
        off_s = p2off_df[p2off_df["loss"] == l]["eer"]
        on_m  = on_s.mean()  if not on_s.empty  else None
        off_m = off_s.mean() if not off_s.empty else None
        rows_b.append({
            "Loss Function": LOSS_LABELS.get(l, l),
            "Com Min. Média": _fmt_eer(on_m),  "Com Min. Std": f"{on_s.std()*100:.2f} pp" if len(on_s) > 1 else "—",
            "Sem Min. Média": _fmt_eer(off_m), "Sem Min. Std": f"{off_s.std()*100:.2f} pp" if len(off_s) > 1 else "—",
            "Δ média (pp)": _fmt_delta(on_m, off_m) if on_m and off_m else "—",
        })
    table_p2 = pd.DataFrame(rows_b)
    chart_p2 = _grouped_bar_chart(
        [LOSS_LABELS.get(l, l) for l in losses_b],
        {"Com Mineração": [p2on_df[p2on_df["loss"] == l]["eer"].mean() if l in set(p2on_df["loss"]) else None for l in losses_b],
         "Sem Mineração": [p2off_df[p2off_df["loss"] == l]["eer"].mean() if l in set(p2off_df["loss"]) else None for l in losses_b]},
        "Full Eval — Fase 2 efeito do professor (pares completos)",
    )

    # Part C — Melhor EER Acumulado
    # RESTRICTED: only (loss, split) pairs present in BOTH fase1 AND fase2.
    # This prevents orphan runs (only fase1 or only fase2) from appearing here.
    df_p1   = df[df["phase"] == "phase1"].set_index(["loss", "split"])["eer"]
    df_p2on = df[df["phase"] == "phase2_on"].set_index(["loss", "split"])["eer"]
    df_p2off= df[df["phase"] == "phase2_off"].set_index(["loss", "split"])["eer"]

    # Only losses that have BOTH fase1 entries AND fase2 entries
    losses_with_p1   = set(df[df["phase"] == "phase1"]["loss"].dropna())
    losses_with_p2   = set(df[df["phase"].isin(["phase2_on", "phase2_off"])]["loss"].dropna())
    losses_c = sorted(losses_with_p1 & losses_with_p2)

    all_splits_c = sorted(df["split"].dropna().unique().astype(int))
    best_on: dict[str, list] = {}
    best_off: dict[str, list] = {}
    for loss in losses_c:
        on_v, off_v = [], []
        for sp in all_splits_c:
            idx  = (loss, sp)
            p1e  = _to_scalar(df_p1.get(idx))
            one  = _to_scalar(df_p2on.get(idx))
            offe = _to_scalar(df_p2off.get(idx))
            if one  is not None: on_v.append(min(p1e, one)  if p1e is not None else one)
            if offe is not None: off_v.append(min(p1e, offe) if p1e is not None else offe)
            if one is None and offe is None and p1e is not None:
                on_v.append(p1e); off_v.append(p1e)
        best_on[loss]  = on_v
        best_off[loss] = off_v

    rows_c = []
    for loss in losses_c:
        on_s  = pd.Series(best_on[loss],  dtype=float)
        off_s = pd.Series(best_off[loss], dtype=float)
        on_m  = on_s.mean()  if not on_s.empty  else None
        off_m = off_s.mean() if not off_s.empty else None
        rows_c.append({
            "Loss Function": LOSS_LABELS.get(loss, loss),
            "Com Min. Média": _fmt_eer(on_m),  "Com Min. Std": f"{on_s.std()*100:.2f} pp" if len(on_s) > 1 else "—",
            "Sem Min. Média": _fmt_eer(off_m), "Sem Min. Std": f"{off_s.std()*100:.2f} pp" if len(off_s) > 1 else "—",
            "Δ média (pp)": _fmt_delta(on_m, off_m) if on_m and off_m else "—",
        })
    table_p3 = pd.DataFrame(rows_c)

    sort_key_c = {l: (np.mean(best_on[l]) if best_on[l] else 1.0) for l in losses_c}
    losses_c_sorted = sorted(losses_c, key=lambda l: sort_key_c[l])
    chart_p3 = _grouped_bar_chart(
        [LOSS_LABELS.get(l, l) for l in losses_c_sorted],
        {"Com Mineração": [np.mean(best_on[l])  if best_on[l]  else None for l in losses_c_sorted],
         "Sem Mineração": [np.mean(best_off[l]) if best_off[l] else None for l in losses_c_sorted]},
        "Full Eval — Melhor EER Acumulado (fase1 + fase2, pares completos)",
    )

    # Per-split rows for highlight tables ----------------------------------------
    # 1A — Fase 1: one row per loss
    split_rows_p1: list[dict] = []
    for l in p1_grp.index:
        rd: dict = {"Loss": LOSS_LABELS.get(l, l)}
        for sp in ALL_SPLITS:
            v_ = p1_df[(p1_df["loss"] == l) & (p1_df["split"] == sp)]["eer"]
            rd[f"S{sp}"] = float(v_.iloc[0]) if not v_.empty else None
        sub = p1_df[p1_df["loss"] == l]["eer"]
        rd["Média"]   = float(sub.mean())   if not sub.empty else None
        rd["Mediana"] = float(sub.median()) if not sub.empty else None
        rd["Std"]     = f"{sub.std()*100:.2f} pp" if len(sub) > 1 else "—"
        split_rows_p1.append(rd)

    # 1B — Fase 2: two rows per loss (Com / Sem Mineração)
    split_rows_p2: list[dict] = []
    for l in losses_b:
        for tipo, ph_df in [("Com Mineração", p2on_df), ("Sem Mineração", p2off_df)]:
            rd = {"Loss": LOSS_LABELS.get(l, l), "Tipo": tipo}
            for sp in ALL_SPLITS:
                v_ = ph_df[(ph_df["loss"] == l) & (ph_df["split"] == sp)]["eer"]
                rd[f"S{sp}"] = float(v_.iloc[0]) if not v_.empty else None
            sub = ph_df[ph_df["loss"] == l]["eer"]
            rd["Média"]   = float(sub.mean())   if not sub.empty else None
            rd["Mediana"] = float(sub.median()) if not sub.empty else None
            rd["Std"]     = f"{sub.std()*100:.2f} pp" if len(sub) > 1 else "—"
            split_rows_p2.append(rd)

    # 1C — Melhor Acumulado: two rows per loss (Com / Sem Mineração)
    split_rows_p3: list[dict] = []
    for l in losses_c_sorted:
        for tipo, vals in [("Com Mineração", best_on[l]), ("Sem Mineração", best_off[l])]:
            rd = {"Loss": LOSS_LABELS.get(l, l), "Tipo": tipo}
            val_by_sp = {int(all_splits_c[i]): float(vals[i]) for i in range(len(vals))}
            for sp in ALL_SPLITS:
                rd[f"S{sp}"] = val_by_sp.get(sp)
            s = pd.Series(list(val_by_sp.values()), dtype=float)
            rd["Média"]   = float(s.mean())   if not s.empty else None
            rd["Mediana"] = float(s.median()) if not s.empty else None
            rd["Std"]     = f"{s.std()*100:.2f} pp" if len(s) > 1 else "—"
            split_rows_p3.append(rd)
    # -------------------------------------------------------------------------

    return table_p1, table_p2, table_p3, chart_p1, chart_p2, chart_p3, is_partial, missing, \
           sorted(run_infos, key=lambda x: x["name"]), \
           split_rows_p1, split_rows_p2, split_rows_p3


# ---------------------------------------------------------------------------
# Training progression (epoch 1 EER + epoch of best, averaged across splits)
# ---------------------------------------------------------------------------

_FASE1_TOTAL_EPOCHS = 10   # fase2 epochs are offset by this


def _build_training_progression(training_runs: List) -> str:
    """Returns HTML table showing, per loss:
      - EER at epoch 1 of fase1 (average across splits)
      - Best EER across the full 15-epoch curve (fase1 + fase2, average across splits)
      - Which epoch (1–15) that best was reached (average across splits)
    Uses runs from the exp3b TRAINING project (Sprint3b_* prefix), not FullEval.
    """
    print("  Buscando histórico por época (pode demorar alguns segundos)...")

    # (loss, split) -> {"initial_eer": float|None, "best_eer": float, "best_epoch": float}
    per_ls: dict[tuple, dict] = {}

    for r in training_runs:
        name = r.name or ""
        nl   = name.lower()
        if not name.startswith("Sprint3b_"):
            continue
        # Skip ablation variant runs (they contain _noinit_ in the name)
        if "_noinit_" in nl:
            continue

        # Parse loss from run name
        loss = None
        for l in _KNOWN_LOSSES_RE:
            if l in nl:
                loss = l
                break
        if not loss:
            continue

        m_sp = re.search(r"_S(\d+)_", name)
        split = int(m_sp.group(1)) if m_sp else None
        if split is None:
            continue

        is_fase2  = "fase2" in nl
        ep_offset = _FASE1_TOTAL_EPOCHS if is_fase2 else 0

        try:
            hist = r.history(keys=["val/eer", "epoch"], pandas=True)
        except Exception as exc:
            print(f"    Aviso: {name[:60]} — {exc}")
            continue

        if hist is None or hist.empty or "val/eer" not in hist.columns:
            continue
        hist = hist.dropna(subset=["val/eer"]).reset_index(drop=True)
        if hist.empty:
            continue

        # Build epoch series (1-based), offset for fase2
        if "epoch" in hist.columns and hist["epoch"].notna().any():
            ep_series = hist["epoch"].fillna(method="ffill").astype(float) + ep_offset
        else:
            ep_series = pd.Series(
                [ep_offset + i + 1 for i in range(len(hist))], dtype=float
            )

        best_pos  = int(hist["val/eer"].argmin())
        best_eer  = float(hist["val/eer"].iloc[best_pos])
        best_ep   = float(ep_series.iloc[best_pos])
        init_eer  = float(hist["val/eer"].iloc[0]) if not is_fase2 else None

        key = (loss, split)
        if key not in per_ls:
            per_ls[key] = {"initial_eer": None, "best_eer": float("inf"), "best_epoch": None}

        # Update best across all phases
        if best_eer < per_ls[key]["best_eer"]:
            per_ls[key]["best_eer"]   = best_eer
            per_ls[key]["best_epoch"] = best_ep
        # Initial EER comes from fase1 only
        if init_eer is not None:
            per_ls[key]["initial_eer"] = init_eer

    if not per_ls:
        return '<p style="color:#aaa;font-style:italic;">Histórico por época não disponível.</p>'

    # Aggregate across splits, per loss
    loss_agg: dict[str, dict] = {}
    for (loss, _split), d in per_ls.items():
        if d["initial_eer"] is None or d["best_epoch"] is None:
            continue
        if loss not in loss_agg:
            loss_agg[loss] = {"ini": [], "best": [], "ep": []}
        loss_agg[loss]["ini"].append(d["initial_eer"])
        loss_agg[loss]["best"].append(d["best_eer"])
        loss_agg[loss]["ep"].append(d["best_epoch"])

    if not loss_agg:
        return '<p style="color:#aaa;font-style:italic;">Sem dados de época suficientes.</p>'

    sorted_losses = sorted(loss_agg, key=lambda l: np.mean(loss_agg[l]["best"]))

    # Render table
    def th(text, align="center"):
        return (
            f'<th style="padding:5px 10px;text-align:{align};'
            f'border-bottom:2px solid #ccc;white-space:nowrap;">{text}</th>'
        )

    header = (
        th("Loss", "left")
        + th("EER Inicial (média, época 1)")
        + th("Melhor EER (média)")
        + th("Melhora")
        + th("Época do melhor (média, 1–15)")
    )

    body = ""
    for i, loss in enumerate(sorted_losses):
        d    = loss_agg[loss]
        ini  = np.mean(d["ini"])
        best = np.mean(d["best"])
        ep   = np.mean(d["ep"])
        impr = (ini - best) * 100
        bg   = "#fafafa" if i % 2 == 0 else "#ffffff"
        impr_color = "#155724" if impr > 0 else "#721c24"
        impr_str   = f"−{impr:.2f} pp" if impr > 0 else f"+{abs(impr):.2f} pp"
        n    = len(d["ini"])
        body += (
            f"<tr>"
            f'<td style="padding:5px 10px;font-weight:bold;background:{bg};">'
            f"{LOSS_LABELS.get(loss, loss)}</td>"
            f'<td style="padding:5px 10px;text-align:center;background:{bg};">{_fmt_eer(ini)}</td>'
            f'<td style="padding:5px 10px;text-align:center;background:{bg};">{_fmt_eer(best)}</td>'
            f'<td style="padding:5px 10px;text-align:center;background:{bg};'
            f"color:{impr_color};font-weight:bold;\">{impr_str}</td>"
            f'<td style="padding:5px 10px;text-align:center;background:{bg};">'
            f"{ep:.1f} <span style='color:#888;font-size:0.85em;'>({n} splits)</span></td>"
            f"</tr>"
        )

    table = (
        '<div style="overflow-x:auto;margin-top:12px;">'
        '<table style="border-collapse:collapse;width:100%;font-size:0.9em;">'
        f'<thead><tr style="background:#f0f0f0;">{header}</tr></thead>'
        f"<tbody>{body}</tbody>"
        "</table></div>"
        '<p style="font-size:0.82em;color:#888;margin-top:6px;">'
        "Fase 1 = épocas 1–10 · Fase 2 = épocas 11–15 · "
        "Melhor = mínimo entre fase 1, fase 2 c/ prof. e fase 2 s/ prof.</p>"
    )

    return (
        '<div style="margin-top:20px;">'
        '<h4 style="margin-bottom:4px;">Progressão de Treino por Loss</h4>'
        f"{table}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_url(r) -> str:
    try:
        return f"https://wandb.ai/{ENTITY}/{r.project}/runs/{r.id}"
    except Exception:
        return "#"


def _collect_full_eval_run_infos(runs: List) -> list[dict]:
    """Returns sorted run info dicts for baseline-only Sprint3b runs (loss comparison)."""
    infos = []
    for r in runs:
        name = r.name or ""
        if not name.startswith("FullEval_Sprint3b_"):
            continue
        v = _variant_from_any_loss_run(name)
        if v is not None and v in ABLATION_VARIANTS:
            continue
        s = _summary(r)
        eer = _scalar(s.get("val/eer"))
        if eer is None:
            continue
        m = re.search(r"_S(\d+)_", name)
        split = int(m.group(1)) if m else None
        nl = name.lower()
        phase = "phase2_on" if "fase2_profon" in nl else ("phase2_off" if "fase2_profoff" in nl else "phase1")
        infos.append({
            "name": name, "url": _run_url(r),
            "split": split, "phase": phase, "eer": eer,
            "variant": "" if v is None else v,
        })
    return sorted(infos, key=lambda x: x["name"])


def _parse_full_eval_run(r) -> Optional[dict]:
    """Returns parsed dict for a FullEval Sprint3b subcenter_cosface run, or None."""
    name = r.name or ""
    if not name.startswith("FullEval_Sprint3b_"):
        return None
    if "subcenter_cosface" not in name.lower():
        return None
    s = _summary(r)
    eer = _scalar(s.get("val/eer"))
    if eer is None:
        return None
    m = re.search(r"_S(\d+)_", name)
    split = int(m.group(1)) if m else None
    nl = name.lower()
    if "fase2_profon" in nl:
        phase = "phase2_on"
    elif "fase2_profoff" in nl:
        phase = "phase2_off"
    else:
        phase = "phase1"
    v = _pooler_variant_from_wandb_name(name)
    # _pooler_variant_from_wandb_name returns None for baseline runs (both old
    # format "...cosface_fase1" and new format "...cosface__fase1") — map to "".
    variant = "" if v is None else v
    return {
        "variant": variant, "split": split, "phase": phase,
        "eer": eer, "name": name, "url": _run_url(r),
    }



def _best_by_variant_split(records: list[dict]) -> pd.DataFrame:
    """Returns DataFrame with best EER per (variant, split) across phases."""
    df = pd.DataFrame(records)
    return df.groupby(["variant", "split"])["eer"].min().reset_index()


def _variant_stats_table(best: pd.DataFrame, variant_keys: list[str]) -> pd.DataFrame:
    """Builds summary table: rows=variants, cols=split0..4 + mean + min."""
    rows = []
    for v in variant_keys:
        label = POOLER_VARIANT_LABELS.get(v, v or "Baseline")
        subset = best[best["variant"] == v]
        if subset.empty:
            continue
        row: dict = {"Variante": label}
        for sp in ALL_SPLITS:
            sp_val = subset[subset["split"] == sp]["eer"]
            row[f"S{sp}"] = _fmt_eer(sp_val.iloc[0]) if not sp_val.empty else "—"
        row["Média"] = _fmt_eer(subset["eer"].mean())
        row["Mín"]   = _fmt_eer(subset["eer"].min())
        rows.append(row)
    return pd.DataFrame(rows)


def _split_highlight_table_html(
    rows: list[dict],
    label_cols: list[str],
    splits: list[int] | None = None,
    extra_fmt_cols: list[str] | None = None,
) -> str:
    """Renders an HTML table with per-split EER columns.
    Minimum (best) EER in each split column is highlighted in green.

    rows: each dict has label_cols keys (str), "S0".."S4" (float|None),
          "Média" (float|None), and optional extra_fmt_cols (pre-formatted str).
    """
    if not rows:
        return '<p style="color:#888;font-style:italic;">Sem dados disponíveis.</p>'
    sp_list   = splits if splits is not None else list(ALL_SPLITS)
    data_cols = [f"S{sp}" for sp in sp_list] + ["Média", "Mediana"]
    extra     = extra_fmt_cols or []

    # Find best (min) row indices per data column — all ties are highlighted.
    # Round to 2 decimal places in % (same as display) to avoid float precision issues.
    def _eer_key(v: float) -> float:
        return round(v * 100, 2)

    best_idx: dict[str, set[int]] = {}
    for col in data_cols:
        col_vals = [(i, row[col]) for i, row in enumerate(rows) if row.get(col) is not None]
        if col_vals:
            best_val = min(_eer_key(v) for _, v in col_vals)
            best_idx[col] = {i for i, v in col_vals if _eer_key(v) == best_val}

    # Header
    def th(text, align="center"):
        return (f'<th style="padding:5px 10px;text-align:{align};'
                f'border-bottom:2px solid #ccc;white-space:nowrap;">{text}</th>')

    header = "".join(th(lc, "left") for lc in label_cols)
    header += "".join(th(dc) for dc in data_cols)
    header += "".join(th(ec) for ec in extra)

    # Body
    body = ""
    for i, row in enumerate(rows):
        bg = "#fafafa" if i % 2 == 0 else "#ffffff"
        cells = ""
        for lc in label_cols:
            cells += (f'<td style="padding:5px 10px;font-weight:bold;'
                      f'background:{bg};white-space:nowrap;">{row.get(lc, "—")}</td>')
        for col in data_cols:
            val = row.get(col)
            if i in best_idx.get(col, set()):
                style = "padding:5px 10px;text-align:center;background:#d4edda;font-weight:bold;color:#155724;"
            else:
                style = f"padding:5px 10px;text-align:center;background:{bg};"
            text = _fmt_eer(val) if val is not None else "—"
            cells += f'<td style="{style}">{text}</td>'
        for ec in extra:
            cells += (f'<td style="padding:5px 10px;text-align:center;background:{bg};">'
                      f'{row.get(ec, "—")}</td>')
        body += f"<tr>{cells}</tr>"

    return (
        '<div style="overflow-x:auto;margin-top:8px;margin-bottom:16px;">'
        '<table style="border-collapse:collapse;width:100%;font-size:0.9em;">'
        f'<thead><tr style="background:#f0f0f0;">{header}</tr></thead>'
        f'<tbody>{body}</tbody>'
        "</table></div>"
    )


# ---------------------------------------------------------------------------
# Section 2 — Pooler Query Ablation (q=1 vs q=2)
# ---------------------------------------------------------------------------

QUERY_VARIANTS = ["", "nq2"]

def _build_query_ablation(runs: List) -> tuple[pd.DataFrame, str, bool, list[int], list[dict]]:
    records = []
    run_infos = []
    for r in runs:
        row = _parse_full_eval_run(r)
        if row is None or row["variant"] not in QUERY_VARIANTS:
            continue
        records.append(row)
        run_infos.append(row)

    if not records:
        return pd.DataFrame(), "", False, list(ALL_SPLITS), [], []

    best = _best_by_variant_split(records)
    completed = sorted(best["split"].dropna().unique().astype(int).tolist())
    missing   = sorted(set(ALL_SPLITS) - set(completed))

    table = _variant_stats_table(best, QUERY_VARIANTS)

    # Consolidated chart: one bar per variant, mean EER across splits
    chart = _grouped_bar_chart(
        [POOLER_VARIANT_LABELS.get(v, v or "Baseline") for v in QUERY_VARIANTS],
        {"EER médio": [
            float(best[best["variant"] == v]["eer"].mean())
            if not best[best["variant"] == v].empty else None
            for v in QUERY_VARIANTS
        ]},
        "Query Ablation — nq=1 vs nq=2 (subcenter_cosface, full pairs)",
    )

    # Per-split rows for highlight table
    split_rows: list[dict] = []
    for v in QUERY_VARIANTS:
        label = POOLER_VARIANT_LABELS.get(v, v or "Baseline")
        rd: dict = {"Variante": label}
        for sp in ALL_SPLITS:
            v_ = best[(best["variant"] == v) & (best["split"] == sp)]["eer"]
            rd[f"S{sp}"] = float(v_.iloc[0]) if not v_.empty else None
        sub = best[best["variant"] == v]["eer"]
        rd["Média"]   = float(sub.mean())   if not sub.empty else None
        rd["Mediana"] = float(sub.median()) if not sub.empty else None
        rd["Std"]     = f"{sub.std()*100:.2f} pp" if len(sub) > 1 else "—"
        split_rows.append(rd)

    return table, chart, bool(missing), missing, \
           sorted(run_infos, key=lambda x: x["name"]), split_rows


# ---------------------------------------------------------------------------
# Section 3 — Prompt Effect Ablation
# ---------------------------------------------------------------------------

# (baseline_variant, richprompt_variant, pooler_label)
PROMPT_EFFECT_GROUPS = [
    ("",            "richprompt_cor",             "Attention q=1"),
    ("cross_modal", "cross_modal_richprompt_cor",  "Cross-Modal"),
    ("mean",        "mean_richprompt_cor",          "Mean Pool"),
]
PROMPT_EFFECT_VARIANTS = {v for grp in PROMPT_EFFECT_GROUPS for v in grp[:2]}

PROMPT_LABELS = {
    "default": "Prompt padrão (P₀)",
    "rich":    "Rich Prompt (Pᵣ)",
}


def _build_prompt_effect_ablation(runs: List) -> tuple[pd.DataFrame, str, bool, list[int], list[dict]]:
    records = []
    run_infos = []
    for r in runs:
        row = _parse_full_eval_run(r)
        if row is None or row["variant"] not in PROMPT_EFFECT_VARIANTS:
            continue
        records.append(row)
        run_infos.append(row)

    if not records:
        return pd.DataFrame(), "", False, list(ALL_SPLITS), [], []

    best = _best_by_variant_split(records)
    completed = sorted(best["split"].dropna().unique().astype(int).tolist())
    missing   = sorted(set(ALL_SPLITS) - set(completed))

    # Table: one row per (pooler, prompt), cols = S0..S4 + Média + Δ
    table_rows = []
    for v_default, v_rich, pooler_label in PROMPT_EFFECT_GROUPS:
        for prompt_key, variant_key in [("default", v_default), ("rich", v_rich)]:
            subset = best[best["variant"] == variant_key]
            if subset.empty:
                continue
            row: dict = {
                "Pooler": pooler_label,
                "Prompt": PROMPT_LABELS[prompt_key],
            }
            for sp in ALL_SPLITS:
                sp_val = subset[subset["split"] == sp]["eer"]
                row[f"S{sp}"] = _fmt_eer(sp_val.iloc[0]) if not sp_val.empty else "—"
            row["Média"] = _fmt_eer(subset["eer"].mean())
            table_rows.append({"pooler_label": pooler_label, "prompt_key": prompt_key,
                                "mean_eer": subset["eer"].mean(), **row})

    table_df = pd.DataFrame(table_rows)

    # Δ column: rich − default (percentage points)
    delta_rows = []
    for _, _, pooler_label in PROMPT_EFFECT_GROUPS:
        default_row = table_df[(table_df["pooler_label"] == pooler_label) & (table_df["prompt_key"] == "default")]
        rich_row    = table_df[(table_df["pooler_label"] == pooler_label) & (table_df["prompt_key"] == "rich")]
        if default_row.empty or rich_row.empty:
            continue
        d = default_row["mean_eer"].iloc[0]
        r = rich_row["mean_eer"].iloc[0]
        delta = (r - d) * 100
        sign  = "+" if delta > 0 else ""
        delta_rows.append((pooler_label, "default", "—"))
        delta_rows.append((pooler_label, "rich",    f"{sign}{delta:.2f} pp"))

    delta_map = {(pl, pk): dv for pl, pk, dv in delta_rows}
    display_cols = ["Pooler", "Prompt"] + [f"S{sp}" for sp in ALL_SPLITS] + ["Média", "Δ vs P₀"]
    table_df["Δ vs P₀"] = table_df.apply(
        lambda row_: delta_map.get((row_["pooler_label"], row_["prompt_key"]), "—"), axis=1
    )
    display_df = table_df[display_cols].reset_index(drop=True)

    # Chart: grouped bars — groups=poolers, series=P₀/Pᵣ
    groups = [pl for _, _, pl in PROMPT_EFFECT_GROUPS]
    series_default = [
        table_df[(table_df["pooler_label"] == pl) & (table_df["prompt_key"] == "default")]["mean_eer"].iloc[0]
        if not table_df[(table_df["pooler_label"] == pl) & (table_df["prompt_key"] == "default")].empty else None
        for _, _, pl in PROMPT_EFFECT_GROUPS
    ]
    series_rich = [
        table_df[(table_df["pooler_label"] == pl) & (table_df["prompt_key"] == "rich")]["mean_eer"].iloc[0]
        if not table_df[(table_df["pooler_label"] == pl) & (table_df["prompt_key"] == "rich")].empty else None
        for _, _, pl in PROMPT_EFFECT_GROUPS
    ]
    chart = _grouped_bar_chart(
        groups,
        {PROMPT_LABELS["default"]: series_default, PROMPT_LABELS["rich"]: series_rich},
        "Prompt Effect — P₀ vs Pᵣ por pooler (subcenter_cosface, full pairs)",
    )
    # Per-split rows for highlight table (with Δ vs P₀)
    split_rows: list[dict] = []
    for v_default, v_rich, pooler_label in PROMPT_EFFECT_GROUPS:
        for prompt_key, variant_key in [("default", v_default), ("rich", v_rich)]:
            subset = best[best["variant"] == variant_key]
            rd: dict = {
                "Pooler": pooler_label,
                "Prompt": PROMPT_LABELS[prompt_key],
            }
            for sp in ALL_SPLITS:
                v_ = subset[subset["split"] == sp]["eer"]
                rd[f"S{sp}"] = float(v_.iloc[0]) if not v_.empty else None
            rd["Média"]   = float(subset["eer"].mean())   if not subset.empty else None
            rd["Mediana"] = float(subset["eer"].median()) if not subset.empty else None
            rd["Std"]     = f"{subset['eer'].std()*100:.2f} pp" if len(subset) > 1 else "—"
            rd["Δ vs P₀"] = "—"
            split_rows.append(rd)

    # Fill Δ for rich rows — based on mean (matches chart metric)
    for i in range(0, len(split_rows), 2):
        d_med = split_rows[i]["Média"]
        r_med = split_rows[i + 1]["Média"] if i + 1 < len(split_rows) else None
        if d_med is not None and r_med is not None:
            delta = (r_med - d_med) * 100
            split_rows[i + 1]["Δ vs P₀"] = f"{'+'if delta>0 else''}{delta:.2f} pp"

    return display_df, chart, bool(missing), missing, \
           sorted(run_infos, key=lambda x: x["name"]), split_rows


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _runs_debug_html(run_infos: list[dict], section_label: str) -> str:
    if not run_infos:
        return (
            f'<p class="missing-note">⚠ Nenhuma run encontrada para: <b>{section_label}</b></p>'
        )
    header = (
        '<tr style="background:#f0f0f0;">'
        '<th style="padding:3px 8px;text-align:left;">Run name</th>'
        '<th style="padding:3px 8px;">Variant</th>'
        '<th style="padding:3px 8px;">Split</th>'
        '<th style="padding:3px 8px;">Phase</th>'
        '<th style="padding:3px 8px;">EER</th>'
        '</tr>'
    )
    rows = "".join(
        f'<tr>'
        f'<td style="padding:2px 8px;"><a href="{ri["url"]}" target="_blank">'
        f'<code style="font-size:0.8em;">{ri["name"]}</code></a></td>'
        f'<td style="padding:2px 8px;text-align:center;"><code>{ri.get("variant","?") or "baseline"}</code></td>'
        f'<td style="padding:2px 8px;text-align:center;">{ri.get("split","?")}</td>'
        f'<td style="padding:2px 8px;text-align:center;">{ri.get("phase","?")}</td>'
        f'<td style="padding:2px 8px;text-align:center;">{ri.get("eer",float("nan"))*100:.2f}%</td>'
        f'</tr>'
        for ri in run_infos
    )
    table = (
        f'<table style="border-collapse:collapse;width:100%;font-size:0.82em;">'
        f'{header}{rows}</table>'
    )
    return (
        f'<details style="margin-top:8px;margin-bottom:16px;">'
        f'<summary style="cursor:pointer;color:#555;font-size:0.85em;">'
        f'Runs consideradas ({len(run_infos)})</summary>'
        f'<div style="overflow-x:auto;margin-top:6px;">{table}</div>'
        f'</details>'
    )


def build_final_html(
    full_eval_t1, full_eval_t2, full_eval_t3,
    full_eval_c1, full_eval_c2, full_eval_c3,
    full_eval_partial, full_eval_missing, full_eval_run_names,
    full_eval_sr1, full_eval_sr2, full_eval_sr3,
    training_prog_html,
    query_abl_t, query_abl_c, query_abl_partial, query_abl_missing, query_abl_run_names,
    query_abl_sr,
    prompt_abl_t, prompt_abl_c, prompt_abl_partial, prompt_abl_missing, prompt_abl_run_names,
    prompt_abl_sr,
    emb_cv, emb_chart_cv,
    vlm_cv, vlm_chart_cv,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def partial_badge(is_p):
        return '<span class="partial-badge">⚠ Parcial</span>' if is_p else ""

    def missing_note(m):
        return f'<p class="missing-note">Splits pendentes: {m}</p>' if m else ""

    sections = []

    # --- 1. Full Eval — Loss Comparison ---
    sections.append(f"""
<div class="section">
  <h2>1. Comparação de Losses — Full Eval {partial_badge(full_eval_partial)}</h2>
  <p class="desc">
    Sprint3b · Attention pooler (nq=1) · Sub-Center (s=32, k=3), inicialização aleatória.
    EER calculado nos CSVs de validação completos (sem subset). Diretamente comparável
    com baselines de embedding e VLM.
  </p>
  {missing_note(full_eval_missing)}
  {_runs_debug_html(full_eval_run_names, "Full Eval baseline")}
  <h3>Fase 1 — Pré-treinamento</h3>
  {_chart_html(full_eval_c1)}
  {_split_highlight_table_html(full_eval_sr1, label_cols=["Loss"], extra_fmt_cols=["Std"])}
  <h3>Fase 2 — Efeito do Professor</h3>
  {_chart_html(full_eval_c2)}
  {_split_highlight_table_html(full_eval_sr2, label_cols=["Loss", "Tipo"], extra_fmt_cols=["Std"])}
  <h3>Melhor EER Acumulado</h3>
  {_chart_html(full_eval_c3)}
  {_split_highlight_table_html(full_eval_sr3, label_cols=["Loss", "Tipo"], extra_fmt_cols=["Std"])}
  {training_prog_html}
</div>""")

    # --- 2. Query Ablation ---
    sections.append(f"""
<div class="section">
  <h2>2. Pooler Query Ablation — nq=1 vs nq=2 {partial_badge(query_abl_partial)}</h2>
  <p class="desc">
    Efeito do número de queries no attention pooler (subcenter_cosface, full pairs).
    Melhor EER por split entre fase 1 e fase 2.
  </p>
  {missing_note(query_abl_missing)}
  {_runs_debug_html(query_abl_run_names, "Query ablation")}
  {_chart_html(query_abl_c)}
  {_split_highlight_table_html(query_abl_sr, label_cols=["Variante"], extra_fmt_cols=["Std"])}
</div>""")

    # --- 3. Prompt Effect Ablation ---
    sections.append(f"""
<div class="section">
  <h2>3. Efeito do Prompt — P₀ vs Pᵣ por Pooler {partial_badge(prompt_abl_partial)}</h2>
  <p class="desc">
    Comparação do prompt padrão (P₀) com o rich prompt (Pᵣ) para três arquiteturas
    de pooler (attention q=1, cross-modal, mean pool). subcenter_cosface · full pairs.
    Δ = EER(Pᵣ) − EER(P₀) em pp — positivo = degradação, negativo = melhora.
  </p>
  {missing_note(prompt_abl_missing)}
  {_runs_debug_html(prompt_abl_run_names, "Prompt effect ablation")}
  {_chart_html(prompt_abl_c)}
  {_split_highlight_table_html(prompt_abl_sr, label_cols=["Pooler", "Prompt"], extra_fmt_cols=["Std", "Δ vs P₀"])}
</div>""")

    # --- 4. Baselines Embedding ---
    sections.append(f"""
<div class="section">
  <h2>4. Baselines — Similaridade por Embedding</h2>
  <p class="desc">
    Pixel bruto, Jina-v4, InternVL3-2B (camadas de entrada/saída, prompt padrão e rich)
    e Jina-v4 finetuned (LoRA). Validação cruzada splits 0–4.
  </p>
  {_chart_html(emb_chart_cv)}
  {_table_html(emb_cv)}
</div>""")

    # --- 5. Baselines VLM ---
    sections.append(f"""
<div class="section">
  <h2>5. Baselines — VLM com Métrica Numérica</h2>
  <p class="desc">
    Modelos VLM solicitados a retornar similaridade 0–100. Splits 0–4.
  </p>
  {_chart_html(vlm_chart_cv)}
  {_table_html(vlm_cv)}
</div>""")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>ArcDoc — Resultados para o Paper</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>ArcDoc — Resultados para o Paper</h1>
  <p class="subtitle">Gerado em {now} · W&B: {ENTITY}</p>
  {''.join(sections)}
  <footer>generate_final_paper_results.py · CaVL-Doc</footer>
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
    runs_full_eval = _fetch_runs(PROJECTS["full_eval"])
    runs_train_3b  = _fetch_runs(PROJECTS["exp3b"])
    runs_emb       = _fetch_runs(PROJECTS["emb_baseline"])
    runs_jina_lora = _fetch_runs(PROJECTS["jina_lora"])
    runs_vlm       = _fetch_runs(PROJECTS["vlm_metric"])

    print("Construindo seções...")
    # Filtra apenas Sprint3b — exclui runs antigas de Sprint3
    runs_full_eval_3b = [r for r in runs_full_eval if (r.name or "").startswith("FullEval_Sprint3b_")]

    full_eval_t1, full_eval_t2, full_eval_t3, \
        full_eval_c1, full_eval_c2, full_eval_c3, \
        full_eval_partial, full_eval_missing, full_eval_run_names, \
        full_eval_sr1, full_eval_sr2, full_eval_sr3 = \
        _build_loss_comparison(runs_full_eval_3b)

    query_abl_t, query_abl_c, query_abl_partial, query_abl_missing, query_abl_run_names, \
        query_abl_sr = _build_query_ablation(runs_full_eval_3b)

    prompt_abl_t, prompt_abl_c, prompt_abl_partial, prompt_abl_missing, prompt_abl_run_names, \
        prompt_abl_sr = _build_prompt_effect_ablation(runs_full_eval_3b)

    emb_cv, _, emb_chart_cv, _ = _build_baselines_embedding(runs_emb, extra_runs=runs_jina_lora)
    vlm_cv, _, vlm_chart_cv, _ = _build_baselines_vlm(runs_vlm, "Baselines VLM")

    print("Buscando progressão de treino por época...")
    training_prog_html = _build_training_progression(runs_train_3b)

    html = build_final_html(
        full_eval_t1, full_eval_t2, full_eval_t3,
        full_eval_c1, full_eval_c2, full_eval_c3,
        full_eval_partial, full_eval_missing, full_eval_run_names,
        full_eval_sr1, full_eval_sr2, full_eval_sr3,
        training_prog_html,
        query_abl_t, query_abl_c, query_abl_partial, query_abl_missing, query_abl_run_names,
        query_abl_sr,
        prompt_abl_t, prompt_abl_c, prompt_abl_partial, prompt_abl_missing, prompt_abl_run_names,
        prompt_abl_sr,
        emb_cv, emb_chart_cv,
        vlm_cv, vlm_chart_cv,
    )

    output.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
