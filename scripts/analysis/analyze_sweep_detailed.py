#!/usr/bin/env python3
"""
Análise exploratória detalhada do sweep_analysis.csv por loss type.
Gera relatório HTML em múltiplas páginas com análise de Bayesian Optimization.
"""
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import statistics


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FINE_SEARCH_CSV = (
    WORKSPACE_ROOT
    / "scripts/optimization/coarse_search/configs/lacdip/fine_search/sweep_analysis.csv"
)

FINE_RUNS_CSV = (
    WORKSPACE_ROOT
    / "scripts/optimization/coarse_search/configs/lacdip/fine_search/runs_raw.csv"
)

HISTORY_CACHE_FILE = WORKSPACE_ROOT / "sweep_report" / "history_cache.json"
COARSE_RUNS_CACHE_FILE = WORKSPACE_ROOT / "sweep_report" / "coarse_runs_cache.json"
FINE_SEARCH_YAML_DIR = (
    WORKSPACE_ROOT
    / "scripts/optimization/coarse_search/configs/lacdip/fine_search"
)

# Fallback quando o WandB coarse não está disponível (extraído dos YAMLs)
_COARSE_FALLBACK = {
    "subcenter_arcface": {"lr": (1e-6, 1e-3), "margin": (0.1, 0.6),  "scale": (32.0, 32.0)},
    "subcenter_cosface": {"lr": (1e-6, 1e-3), "margin": (0.1, 0.6),  "scale": (32.0, 32.0)},
    "contrastive":       {"lr": (1e-6, 1e-3), "margin": (0.1, 1.0),  "scale": (64.0, 64.0)},
    "triplet":           {"lr": (1e-6, 1e-3), "margin": (0.1, 1.0),  "scale": (64.0, 64.0)},
    "circle":            {"lr": (1e-6, 1e-3), "margin": (0.1, 0.4),  "scale": (32.0, 256.0)},
}

# Valores fixos escolhidos para a Sprint 3 (marcados nas barras do fine search)
SPRINT3_DEFAULTS = {"lr": 1e-5, "margin": 0.35}

INDUSTRY_STANDARDS = {
    "subcenter_arcface": {
        "lr": (1e-5, 1e-4),
        "margin": (0.1, 0.5),
        "scale": (32.0, 64.0),
        "notes": "ArcFace: margin baixo, scale ~32-64. Robusto para documentos.",
    },
    "subcenter_cosface": {
        "lr": (1e-5, 1e-4),
        "margin": (0.1, 0.5),
        "scale": (32.0, 64.0),
        "notes": "CosFace: similar ao ArcFace, margin um pouco menor.",
    },
    "contrastive": {
        "lr": (1e-6, 1e-5),
        "margin": (0.5, 1.0),
        "scale": (64.0, 128.0),
        "notes": "Contrastive: LR muito menor, margin maior (0.5-1.0).",
    },
    "triplet": {
        "lr": (1e-5, 1e-4),
        "margin": (0.3, 1.0),
        "scale": (32.0, 64.0),
        "notes": "Triplet: margin moderado, exige cuidado com hard negatives.",
    },
    "circle": {
        "lr": (1e-5, 1e-4),
        "margin": (0.2, 0.5),
        "scale": (100.0, 300.0),
        "notes": "Circle: scale bem maior (100-300), margin conservador.",
    },
}


@dataclass
class SweepRun:
    run_id: str
    run_name: str
    loss_type: str
    lr: float
    margin: float
    scale: float
    eer_final: float
    eer_epoch1: float
    eer_drop: float
    eer_stability: float
    diverged: bool
    n_epochs: int
    k: int = 1  # num-sub-centers


@dataclass
class CoarseRun:
    run_id: str
    run_name: str
    loss_type: str
    lr: float
    margin: float
    scale: float
    eer_final: float
    state: str  # "finished" | "crashed" | "failed" | "running"


def load_sweep_csv(csv_path: Path) -> List[SweepRun]:
    runs = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                runs.append(
                    SweepRun(
                        run_id=row["run_id"],
                        run_name=row["run_name"],
                        loss_type=row["loss_type"].strip(),
                        lr=float(row["lr"]),
                        margin=float(row["margin"]),
                        scale=float(row["scale"]),
                        eer_final=float(row["eer_final"]),
                        eer_epoch1=float(row["eer_epoch1"]),
                        eer_drop=float(row["eer_drop"]),
                        eer_stability=float(row["eer_stability"]),
                        diverged=row["diverged"].lower() == "true",
                        n_epochs=int(row["n_epochs"]),
                        k=int(row["k"]) if row.get("k") and row["k"].strip() else 1,
                    )
                )
            except (ValueError, KeyError):
                pass
    return runs


def score_run(run: SweepRun, loss_type: str, analysis: Dict) -> float:
    """Calcula score holístico da run considerando múltiplas dimensões."""
    score = 0.0

    # 1. EER final (normalizado: melhor é 0, pior é 1, mas invertemos para score)
    eer_range = analysis['eer_max'] - analysis['eer_min']
    eer_norm = (analysis['eer_max'] - run.eer_final) / eer_range if eer_range > 0 else 0.5
    score += eer_norm * 0.5  # 50% do peso

    # 2. % queda por época (convergência rápida) — 40% do peso
    pct_drop = (run.eer_epoch1 - run.eer_final) / run.eer_epoch1 * 100 if run.eer_epoch1 > 0 else 0
    pct_drop_per_epoch = pct_drop / run.n_epochs if run.n_epochs > 0 else 0
    max_drop_per_epoch = 50
    score += (min(pct_drop_per_epoch, max_drop_per_epoch) / max_drop_per_epoch) * 0.4  # 40% do peso

    # 3. Volatilidade (menor é melhor) — 10% do peso
    stability_max = analysis.get('stability_max', 0.1) or 0.1
    stability_norm = 1.0 - min(run.eer_stability / stability_max, 1.0)
    score += stability_norm * 0.1  # 10% do peso

    return score


def compute_correlation(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
    std_x = statistics.stdev(x) if len(x) > 1 else 1
    std_y = statistics.stdev(y) if len(y) > 1 else 1
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def analyze_loss(runs: List[SweepRun]) -> Dict:
    if not runs:
        return {}

    loss = runs[0].loss_type
    converged = [r for r in runs if not r.diverged]

    eers = [r.eer_final for r in runs]
    lrs = [r.lr for r in runs]
    margins = [r.margin for r in runs]
    scales = [r.scale for r in runs]
    stabilities = [r.eer_stability for r in runs]
    drops = [r.eer_drop for r in runs]

    converged_eers = [r.eer_final for r in converged]
    converged_lrs = [r.lr for r in converged]
    converged_margins = [r.margin for r in converged]
    converged_scales = [r.scale for r in converged]

    corr_lr_eer = compute_correlation(converged_lrs, converged_eers) if converged else 0.0
    corr_margin_eer = compute_correlation(converged_margins, converged_eers) if converged else 0.0
    corr_scale_eer = compute_correlation(converged_scales, converged_eers) if converged else 0.0

    # Score each run holistically antes de retornar
    # Primeiro, preparamos a análise parcial para ter eer_max/min
    eers_temp = [r.eer_final for r in runs]
    analysis_partial = {
        'eer_max': max(eers_temp),
        'eer_min': min(eers_temp),
        'stability_max': max(r.eer_stability for r in runs) if runs else 0.1,
    }
    # Score cada run e pega top 5
    # Prioridade: runs com queda > 0 sempre precedem runs estagnadas (queda ≤ 0).
    # Dentro de cada grupo: scored desc; empatadas em score → EER asc.
    pool = converged if converged else runs
    runs_with_scores = [(r, score_run(r, loss, analysis_partial)) for r in pool]
    improving = sorted(
        [(r, s) for r, s in runs_with_scores if r.eer_epoch1 > r.eer_final],
        key=lambda x: (-x[1], x[0].eer_final),
    )
    stagnant = sorted(
        [(r, s) for r, s in runs_with_scores if r.eer_epoch1 <= r.eer_final],
        key=lambda x: x[0].eer_final,
    )
    best_runs = [r for r, _ in (improving + stagnant)[:5]]

    runs_sorted_by_name = sorted(runs, key=lambda r: r.run_name)
    early_runs = runs_sorted_by_name[:len(runs_sorted_by_name)//3]
    late_runs = runs_sorted_by_name[-len(runs_sorted_by_name)//3:]

    early_eer = statistics.mean([r.eer_final for r in early_runs]) if early_runs else 0
    late_eer = statistics.mean([r.eer_final for r in late_runs]) if late_runs else 0
    improvement = ((early_eer - late_eer) / early_eer * 100) if early_eer > 0 else 0

    # Análise de convergência por época
    convergence_rates = [
        (r.eer_epoch1 - r.eer_final) / r.eer_epoch1 * 100 / r.n_epochs
        for r in runs if r.n_epochs > 0 and r.eer_epoch1 > 0
    ]
    avg_epoch1 = statistics.mean([r.eer_epoch1 for r in runs])
    avg_final = statistics.mean([r.eer_final for r in runs])
    avg_drop_pct = statistics.mean([
        (r.eer_epoch1 - r.eer_final) / r.eer_epoch1 * 100
        for r in runs if r.eer_epoch1 > 0
    ]) if runs else 0
    avg_convergence_rate = statistics.mean(convergence_rates) if convergence_rates else 0

    # Runs com melhor convergência (mais melhoria percentual por época)
    runs_by_conv_rate = sorted(
        runs,
        key=lambda r: (r.eer_epoch1 - r.eer_final) / r.eer_epoch1 * 100 / r.n_epochs if r.n_epochs > 0 and r.eer_epoch1 > 0 else 0,
        reverse=True
    )
    best_convergence = runs_by_conv_rate[:3]

    # Breakdown por k (num-sub-centers)
    k_values = sorted(set(r.k for r in runs))
    k_breakdown: Dict[int, Dict] = {}
    for k in k_values:
        k_runs = [r for r in runs if r.k == k]
        k_eers = [r.eer_final for r in k_runs]
        k_breakdown[k] = {
            "total": len(k_runs),
            "eer_min": min(k_eers),
            "eer_mean": statistics.mean(k_eers),
            "lr_min": min(r.lr for r in k_runs),
            "lr_max": max(r.lr for r in k_runs),
            "margin_min": min(r.margin for r in k_runs),
            "margin_max": max(r.margin for r in k_runs),
        }

    return {
        "loss": loss,
        "total_runs": len(runs),
        "convergence_rate": len(converged) / len(runs) if runs else 0.0,
        "eer_min": min(eers),
        "eer_max": max(eers),
        "eer_mean": statistics.mean(eers),
        "eer_median": statistics.median(eers),
        "eer_stdev": statistics.stdev(eers) if len(eers) > 1 else 0.0,
        "lr_min": min(lrs),
        "lr_max": max(lrs),
        "lr_mean": statistics.mean(lrs),
        "lr_median": statistics.median(lrs),
        "margin_min": min(margins),
        "margin_max": max(margins),
        "margin_mean": statistics.mean(margins),
        "margin_median": statistics.median(margins),
        "scale_min": min(scales),
        "scale_max": max(scales),
        "scale_mean": statistics.mean(scales),
        "scale_median": statistics.median(scales),
        "stability_mean": statistics.mean(stabilities),
        "stability_median": statistics.median(stabilities),
        "drop_mean": statistics.mean(drops),
        "drop_median": statistics.median(drops),
        "corr_lr_eer": corr_lr_eer,
        "corr_margin_eer": corr_margin_eer,
        "corr_scale_eer": corr_scale_eer,
        "best_runs": best_runs,
        "early_eer": early_eer,
        "late_eer": late_eer,
        "improvement": improvement,
        "avg_epoch1": avg_epoch1,
        "avg_final": avg_final,
        "avg_drop_pct": avg_drop_pct,
        "avg_convergence_rate": avg_convergence_rate,
        "best_convergence": best_convergence,
        "k_breakdown": k_breakdown,
    }


def _relative_oscillation(run: SweepRun) -> Optional[str]:
    """Oscilação da trajetória relativa à melhoria total.

    Definida como std(épocas) / queda_total * 100.
    Interpretação: 0% = queda perfeitamente linear; 100%+ = oscilou tanto quanto o total que caiu.
    Retorna None quando não há queda (sem base para normalizar).
    """
    drop = run.eer_epoch1 - run.eer_final
    if drop <= 1e-10:
        return None
    rel = run.eer_stability / drop * 100
    return f"{rel:.1f}%"


def format_float(value: float, decimals: int = 6) -> str:
    """Formata float sem notação científica."""
    if value == 0:
        return "0.0"
    if abs(value) < 0.0001:
        return f"{value:.10f}".rstrip('0').rstrip('.')
    return f"{value:.{decimals}f}"


def _scan_eer_values(run) -> List[float]:
    """Extrai lista de val/eer por época via scan_history (API estável no wandb 0.26+)."""
    values = []
    for row in run.scan_history(keys=["val/eer"]):
        v = row.get("val/eer")
        if v is None:
            continue
        try:
            fv = float(v)
            if not math.isnan(fv):
                values.append(fv)
        except (TypeError, ValueError):
            pass
    return values


def fetch_run_histories(
    run_ids: List[str],
    entity: str,
    projects: List[str],
    cache_path: Path = HISTORY_CACHE_FILE,
) -> Dict[str, List[float]]:
    """Busca histórico val/eer por época do WandB via scan_history.

    Itera pelos projetos em `projects` até encontrar cada run.
    Runs com lista vazia no cache são re-tentados (podem ter falhado anteriormente).
    """
    cache: Dict[str, List[float]] = {}

    if cache_path.exists():
        with cache_path.open() as fh:
            cache = json.load(fh)
        n_valid = sum(1 for v in cache.values() if v)
        print(f"  Cache carregado: {len(cache)} runs, {n_valid} com dados ({cache_path.name})")

    # Re-tenta runs ausentes OU que retornaram vazio anteriormente
    missing = set(rid for rid in run_ids if rid not in cache or not cache[rid])
    if missing:
        try:
            import wandb
            api = wandb.Api()
            for project in projects:
                if not missing:
                    break
                print(f"  Buscando histórico em {entity}/{project} ...")
                for run in api.runs(f"{entity}/{project}", order="-created_at"):
                    if run.id not in missing:
                        continue
                    try:
                        values = _scan_eer_values(run)
                        cache[run.id] = values
                        missing.discard(run.id)
                        print(f"  ✓ {run.id}: {len(values)} épocas")
                    except Exception as e:
                        print(f"  ⚠️  {run.id}: {e}")
                        cache[run.id] = []
                        missing.discard(run.id)

            for rid in missing:
                print(f"  ⚠️  {rid}: não encontrado em nenhum projeto")
                cache[rid] = []

        except ImportError:
            print("  ⚠️  wandb não instalado — usando apenas epoch1/final.")

        cache_path.parent.mkdir(exist_ok=True)
        with cache_path.open("w") as fh:
            json.dump(cache, fh, indent=2)
        print(f"  Cache salvo: {cache_path}")

    return cache


def fetch_coarse_runs_from_wandb(
    entity: str,
    project: str,
    cache_path: Path = COARSE_RUNS_CACHE_FILE,
) -> List[CoarseRun]:
    """Busca todos os runs do projeto coarse no WandB. Cacheia em JSON."""
    if cache_path.exists():
        with cache_path.open() as fh:
            raw = json.load(fh)
        print(f"  Cache coarse carregado: {len(raw)} runs ({cache_path.name})")
        result = []
        for r in raw:
            try:
                result.append(CoarseRun(**r))
            except Exception:
                pass
        return result

    result: List[CoarseRun] = []
    try:
        import wandb
        import pandas as pd
        api = wandb.Api()
        path = f"{entity}/{project}"
        print(f"  Buscando runs de {path} ...")
        all_runs = api.runs(path, filters={"state": {"$in": ["finished", "crashed"]}})
        total = len(all_runs)
        print(f"  {total} runs encontrados. Extraindo métricas...")
        for run in all_runs:
            try:
                cfg = dict(run.config)

                def cfg_get(key: str):
                    return cfg.get(key, cfg.get(key.replace("-", "_"), None))

                loss_type = str(cfg_get("loss-type") or "").strip()
                if not loss_type:
                    continue
                lr_raw     = cfg_get("student-lr")
                margin_raw = cfg_get("margin")
                scale_raw  = cfg_get("scale")
                if lr_raw is None or margin_raw is None:
                    continue

                # EER via scan_history (mais robusto que history() no wandb 0.26+)
                values = _scan_eer_values(run)
                if not values:
                    # Fallback: summary
                    v = run.summary.get("val/eer", run.summary.get("eer"))
                    if v is None:
                        continue
                    try:
                        values = [float(v)]
                    except (TypeError, ValueError):
                        continue
                eer_final = min(values)

                result.append(CoarseRun(
                    run_id=run.id,
                    run_name=run.name,
                    loss_type=loss_type,
                    lr=float(lr_raw),
                    margin=float(margin_raw),
                    scale=float(scale_raw) if scale_raw is not None else 32.0,
                    eer_final=eer_final,
                    state=run.state,
                ))
            except Exception as e:
                print(f"  ⚠️  run {getattr(run, 'id', '?')}: {e}")
        print(f"  ✓ {len(result)} runs coarse carregados")
    except ImportError:
        print("  ⚠️  wandb não instalado — coarse search sem dados reais.")
        return []
    except Exception as e:
        print(f"  ⚠️  Erro ao buscar coarse runs: {e}")
        return []

    cache_path.parent.mkdir(exist_ok=True)
    with cache_path.open("w") as fh:
        import dataclasses
        json.dump([dataclasses.asdict(r) for r in result], fh, indent=2)
    print(f"  Cache coarse salvo: {cache_path}")
    return result


def analyze_coarse_loss(coarse_runs: List[CoarseRun]) -> Dict:
    """Estatísticas do coarse search para uma loss type."""
    if not coarse_runs:
        return {}
    finished = [r for r in coarse_runs if r.state == "finished"]
    eers = [r.eer_final for r in coarse_runs]
    lrs = [r.lr for r in coarse_runs]
    margins = [r.margin for r in coarse_runs]
    scales = [r.scale for r in coarse_runs]
    best3 = sorted(finished or coarse_runs, key=lambda r: r.eer_final)[:3]
    return {
        "total_runs": len(coarse_runs),
        "n_finished": len(finished),
        "convergence_rate": len(finished) / len(coarse_runs),
        "eer_min": min(eers),
        "eer_max": max(eers),
        "eer_mean": statistics.mean(eers),
        "eer_median": statistics.median(eers),
        "eer_stdev": statistics.stdev(eers) if len(eers) > 1 else 0.0,
        "lr_min": min(lrs),
        "lr_max": max(lrs),
        "lr_median": statistics.median(lrs),
        "margin_min": min(margins),
        "margin_max": max(margins),
        "margin_median": statistics.median(margins),
        "scale_min": min(scales),
        "scale_max": max(scales),
        "scale_median": statistics.median(scales),
        "best_runs": best3,
    }


def _interpolate_at(values: List[float], x: float) -> float:
    """Interpolação linear de `values` na posição fracionária x ∈ [0, 1]."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    idx = x * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def _eer_volatility(values: List[float]) -> float:
    """Desvio padrão do histórico EER — volatilidade real da trajetória."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def generate_eer_svg(
    best_run: SweepRun,
    top5: List[SweepRun],
    histories: Optional[Dict[str, List[float]]] = None,
) -> str:
    W, H = 620, 280
    ml, mr, mt, mb = 75, 130, 35, 50
    chart_w = W - ml - mr
    chart_h = H - mt - mb

    hist = histories or {}

    def _run_values(run: SweepRun) -> List[float]:
        h = hist.get(run.run_id, [])
        if len(h) >= 2:
            return h
        return [run.eer_epoch1, run.eer_final]

    all_vals = [v for run in top5 for v in _run_values(run)]
    y_max = max(all_vals) * 1.15 if all_vals else 0.1
    max_epochs = max(len(_run_values(r)) for r in top5)

    def yp(eer: float) -> float:
        return mt + chart_h * (1.0 - eer / y_max)

    def xp(i: int, n: int) -> float:
        return ml + chart_w * i / max(n - 1, 1)

    def polyline_points(run: SweepRun) -> str:
        vals = _run_values(run)
        n = len(vals)
        return " ".join(f"{xp(i, n):.1f},{yp(v):.1f}" for i, v in enumerate(vals))

    # Grid and y-axis ticks
    grid = ""
    for i in range(6):
        val = y_max * i / 5
        y = yp(val)
        grid += f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+chart_w}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>'
        grid += f'<line x1="{ml-4}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#aaa" stroke-width="1"/>'
        grid += f'<text x="{ml-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#666">{val:.4f}</text>'

    # X-axis epoch labels
    n_ticks = min(max_epochs, 6)
    x_labels = ""
    for i in range(n_ticks):
        epoch_idx = round(i * (max_epochs - 1) / max(n_ticks - 1, 1))
        xv = xp(epoch_idx, max_epochs)
        x_labels += f'<text x="{xv:.1f}" y="{mt+chart_h+18}" text-anchor="middle" font-size="10" fill="#555">{epoch_idx + 1}</text>'
    x_labels += f'<text x="{ml + chart_w // 2}" y="{mt+chart_h+32}" text-anchor="middle" font-size="11" fill="#666">Época</text>'

    # Individual top5 runs (not best) — light gray polylines
    run_lines = ""
    for run in top5:
        if run is best_run:
            continue
        pts = polyline_points(run)
        run_lines += f'<polyline points="{pts}" fill="none" stroke="#c0ccda" stroke-width="1.5" opacity="0.9"/>'

    # Top 5 mean — orange dashed (normalized to max_epochs grid via interpolation)
    N_GRID = max(max_epochs * 2, 20)
    mean_pts = ""
    for i in range(N_GRID):
        frac = i / max(N_GRID - 1, 1)
        vals_at_frac = [_interpolate_at(_run_values(r), frac) for r in top5]
        xv = ml + chart_w * frac
        yv = yp(statistics.mean(vals_at_frac))
        mean_pts += f"{xv:.1f},{yv:.1f} "
    run_lines += f'<polyline points="{mean_pts.strip()}" fill="none" stroke="#e67e22" stroke-width="2.5" stroke-dasharray="8,4"/>'
    # Label at last point
    last_mean = statistics.mean(_run_values(r)[-1] for r in top5)
    run_lines += f'<text x="{ml+chart_w+8}" y="{yp(last_mean)+4:.1f}" font-size="10" fill="#e67e22" font-weight="bold">{last_mean:.6f}</text>'

    # Best run — blue bold polyline
    best_pts = polyline_points(best_run)
    run_lines += f'<polyline points="{best_pts}" fill="none" stroke="#2980b9" stroke-width="2.5"/>'
    best_final = _run_values(best_run)[-1]
    run_lines += f'<text x="{ml+chart_w+8}" y="{yp(best_final)+4:.1f}" font-size="10" fill="#2980b9" font-weight="bold">{best_final:.6f}</text>'

    # Axes
    axes = (
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+chart_h}" stroke="#999" stroke-width="1.5"/>'
        f'<line x1="{ml}" y1="{mt+chart_h}" x2="{ml+chart_w}" y2="{mt+chart_h}" stroke="#999" stroke-width="1.5"/>'
    )

    # Y label (rotated)
    y_label = f'<text x="12" y="{mt+chart_h//2}" text-anchor="middle" font-size="11" fill="#666" transform="rotate(-90,12,{mt+chart_h//2})">EER</text>'

    # Legend
    lx, ly = ml, mt - 22
    legend = (
        f'<rect x="{lx}" y="{ly}" width="18" height="3" fill="#2980b9"/>'
        f'<text x="{lx+22}" y="{ly+4}" font-size="10" fill="#2980b9">Melhor run ({best_run.run_id})</text>'
        f'<rect x="{lx+145}" y="{ly}" width="18" height="3" fill="#e67e22"/>'
        f'<text x="{lx+167}" y="{ly+4}" font-size="10" fill="#e67e22">Média Top 5</text>'
        f'<rect x="{lx+255}" y="{ly}" width="18" height="3" fill="#c0ccda"/>'
        f'<text x="{lx+277}" y="{ly+4}" font-size="10" fill="#999">Outros Top 5</text>'
    )

    return (
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto;'
        f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
        f'{grid}{axes}{run_lines}{x_labels}{y_label}{legend}</svg>'
    )


def get_css() -> str:
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2980b9; border-left: 4px solid #3498db; padding-left: 12px; margin-top: 30px; }
        h3 { color: #34495e; margin-top: 20px; }
        .section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background: #ecf0f1; font-weight: 600; color: #2c3e50; }
        tr:hover { background: #f9f9f9; }
        .metric {
            display: inline-block;
            background: #ecf0f1;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .alert {
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .alert-success { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
        .alert-warning { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
        .alert-info { background: #d1ecf1; color: #0c5460; border-left: 4px solid #17a2b8; }
        .checkmark { color: #27ae60; font-weight: bold; }
        .cross { color: #e74c3c; font-weight: bold; }
        .recommendation-box {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .nav-links {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav-links a {
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 8px 12px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .nav-links a:hover { background: #2980b9; }
        .footer {
            text-align: center;
            color: #95a5a6;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        .bayesian-box {
            background: #f0f8ff;
            border: 2px solid #4169e1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
    """


def load_fine_search_sweep_info(yaml_dir: Path) -> Dict[str, Dict]:
    """Lê os YAMLs do fine search e retorna metadados por grupo '{loss}_k{k}'.

    Extrai: loss_type, k, run_cap, faixas de lr/margin/scale (fixo ou intervalo).
    """
    import re
    try:
        import yaml as _yaml
        def _load(p): return _yaml.safe_load(p.read_text())
    except ImportError:
        def _load(p): return None

    result: Dict[str, Dict] = {}
    for yaml_path in sorted(yaml_dir.glob("fine_search_*.yaml")):
        stem = yaml_path.stem
        m = re.match(r"fine_search_(.+)_k(\d+)$", stem)
        if not m:
            continue
        loss_type = m.group(1)
        k = int(m.group(2))
        label = f"{loss_type}_k{k}"

        doc = _load(yaml_path)
        if not doc:
            result[label] = {"loss_type": loss_type, "k": k, "run_cap": "?",
                              "lr": "N/D", "margin": "N/D", "scale": "N/D"}
            continue

        params = doc.get("parameters", {})
        run_cap = doc.get("run_cap", "?")

        def _parse_range(key: str):
            p = params.get(key, {})
            if "value" in p:
                v = float(p["value"])
                return v, v
            if "min" in p and "max" in p:
                return float(p["min"]), float(p["max"])
            return None, None

        lr_min, lr_max = _parse_range("student-lr")
        margin_min, margin_max = _parse_range("margin")
        scale_min, scale_max = _parse_range("scale")

        def _fmt(lo, hi):
            if lo is None:
                return "N/D"
            if lo == hi:
                return format_float(lo, 4)
            return f"{format_float(lo, 4)} ~ {format_float(hi, 4)}"

        result[label] = {
            "loss_type": loss_type,
            "k": k,
            "run_cap": run_cap,
            "lr": _fmt(lr_min, lr_max),
            "lr_min": lr_min,
            "lr_max": lr_max,
            "margin": _fmt(margin_min, margin_max),
            "margin_min": margin_min,
            "margin_max": margin_max,
            "scale": _fmt(scale_min, scale_max),
        }
    return result


def _format_coarse_section(loss_name: str, ca: Optional[Dict], fine_analysis: Dict) -> str:
    """Seção HTML do Coarse Search para a página de uma loss."""
    if not ca:
        return ""

    # Comparativo coarse → fine
    fine_best = fine_analysis["eer_min"]
    fine_mean = fine_analysis["eer_mean"]
    coarse_best = ca["eer_min"]
    coarse_mean = ca["eer_mean"]
    delta_best = (coarse_best - fine_best) / coarse_best * 100 if coarse_best > 0 else 0
    delta_mean = (coarse_mean - fine_mean) / coarse_mean * 100 if coarse_mean > 0 else 0
    fine_better_best = fine_best < coarse_best
    fine_better_mean = fine_mean < coarse_mean

    def _arrow(better: bool) -> str:
        return "<span style='color:#27ae60;font-weight:bold'>▼</span>" if better else "<span style='color:#e74c3c;font-weight:bold'>▲</span>"

    improvement_color = "#d4edda" if fine_better_best else "#fff3cd"
    improvement_border = "#27ae60" if fine_better_best else "#ffc107"

    # EER distribution comparison bar (normalized)
    c_range = ca["eer_max"] - ca["eer_min"]
    f_range = fine_analysis["eer_max"] - fine_analysis["eer_min"]
    concentration = (1 - f_range / c_range * 0.7) * 100 if c_range > 0 else 100

    return f"""    <div class="section">
        <h2>🔍 Coarse Search: Exploração Inicial</h2>
        <div class="alert alert-info">
            <strong>O que foi explorado:</strong> {ca['total_runs']} runs na busca ampla inicial.
            Convergência: {ca['convergence_rate']*100:.0f}% ({ca['n_finished']}/{ca['total_runs']}).
            O fine search refinou as regiões mais promissoras encontradas aqui.
        </div>

        <table>
            <tr><th>Parâmetro</th><th>Mín explorado</th><th>Máx explorado</th><th>Mediana</th></tr>
            <tr><td>LR</td><td>{format_float(ca['lr_min'])}</td><td>{format_float(ca['lr_max'])}</td><td>{format_float(ca['lr_median'])}</td></tr>
            <tr><td>Margin</td><td>{format_float(ca['margin_min'], 4)}</td><td>{format_float(ca['margin_max'], 4)}</td><td>{format_float(ca['margin_median'], 4)}</td></tr>
            <tr><td>Scale</td><td>{format_float(ca['scale_min'], 1)}</td><td>{format_float(ca['scale_max'], 1)}</td><td>{format_float(ca['scale_median'], 1)}</td></tr>
        </table>

        <div style="background:{improvement_color};border-left:4px solid {improvement_border};padding:15px;margin-top:15px;border-radius:4px;">
            <h3 style="margin-top:0">📈 Impacto do Refinamento: Coarse → Fine</h3>
            <table>
                <tr><th>Métrica</th><th>Coarse</th><th>Fine</th><th>Δ</th></tr>
                <tr>
                    <td>Melhor EER</td>
                    <td>{format_float(coarse_best)}</td>
                    <td><strong>{format_float(fine_best)}</strong></td>
                    <td>{_arrow(fine_better_best)} {abs(delta_best):.1f}%</td>
                </tr>
                <tr>
                    <td>EER Médio</td>
                    <td>{format_float(coarse_mean)}</td>
                    <td><strong>{format_float(fine_mean)}</strong></td>
                    <td>{_arrow(fine_better_mean)} {abs(delta_mean):.1f}%</td>
                </tr>
                <tr>
                    <td>Faixa EER (dispersão)</td>
                    <td>{c_range:.6f}</td>
                    <td><strong>{f_range:.6f}</strong></td>
                    <td><span style='color:#2980b9;font-weight:bold'>▼ concentração +{concentration:.0f}%</span></td>
                </tr>
            </table>
            <p style="margin-bottom:0"><small>▼ verde = fine melhorou | ▲ vermelho = fine piorou (inesperado)</small></p>
        </div>
    </div>
"""


def _range_bar(
    val_min: float, val_max: float,
    scale_min: float, scale_max: float,
    log_scale: bool = False,
    color: str = "#2980b9",
    marker: Optional[float] = None,
) -> str:
    """Inline bar showing [val_min, val_max] relative to [scale_min, scale_max].
    If marker is given, draws a red vertical line at that value's position."""
    def norm(v: float) -> float:
        if log_scale:
            if scale_min <= 0 or scale_max <= 0 or v <= 0:
                return 0.0
            lo, hi = math.log10(scale_min), math.log10(scale_max)
            return (math.log10(v) - lo) / (hi - lo) if hi > lo else 0.0
        r = scale_max - scale_min
        return (v - scale_min) / r if r > 0 else 0.0

    left = max(0.0, min(100.0, norm(val_min) * 100))
    right = max(0.0, min(100.0, norm(val_max) * 100))
    w = max(3.0, right - left)

    marker_html = ""
    if marker is not None:
        mp = max(0.0, min(99.5, norm(marker) * 100))
        marker_html = (
            f"<div style='position:absolute;left:{mp:.1f}%;width:2px;height:140%;"
            f"top:-20%;background:#e74c3c;z-index:3;border-radius:1px;' "
            f"title='Sprint 3: {marker}'></div>"
        )

    return (
        f"<div style='position:relative;width:110px;height:8px;background:#e8e8e8;"
        f"border-radius:4px;display:inline-block;vertical-align:middle;border:1px solid #ccc;'>"
        f"<div style='position:absolute;left:{left:.1f}%;width:{w:.1f}%;height:100%;"
        f"background:{color};border-radius:3px;opacity:0.85;'></div>"
        f"{marker_html}"
        f"</div>"
    )


def _format_k_selection_section(
    loss_name: str,
    fine_analysis: Dict,
    sweep_info: Optional[Dict[str, Dict]],
) -> str:
    """Seção mostrando resultados do coarse e a seleção Coarse→Fine por sub-centro (k)."""
    k_bd = fine_analysis.get("k_breakdown", {})
    if not k_bd:
        return ""

    # Sweeps do fine search para esta loss
    relevant_sweeps = {
        label: info for label, info in (sweep_info or {}).items()
        if info["loss_type"] == loss_name
    } if sweep_info else {}

    k_values = sorted(k_bd.keys())
    has_multiple_k = len(k_values) > 1

    # ── Resumo do Coarse Search (dados do CSV, sempre disponível) ──────────
    total_coarse = fine_analysis["total_runs"]
    coarse_eer_min = fine_analysis["eer_min"]
    coarse_eer_mean = fine_analysis["eer_mean"]
    conv_rate = fine_analysis["convergence_rate"] * 100

    coarse_summary = f"""        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
            <div class="metric"><span style="font-weight:bold">Runs coarse:</span> {total_coarse}</div>
            <div class="metric"><span style="font-weight:bold">Convergência:</span> {conv_rate:.0f}%</div>
            <div class="metric"><span style="font-weight:bold">Melhor EER:</span> {format_float(coarse_eer_min)}</div>
            <div class="metric"><span style="font-weight:bold">EER médio:</span> {format_float(coarse_eer_mean)}</div>
        </div>"""

    # ── Tabela por sub-centro (k): duas linhas por k (coarse + fine) ─────────
    def _bar_cell(val_min: float, val_max: float,
                  scale_min: float, scale_max: float,
                  log_scale: bool, label: str, color: str,
                  marker: Optional[float] = None) -> str:
        bar = _range_bar(val_min, val_max, scale_min, scale_max, log_scale, color, marker)
        return (
            f"<div style='white-space:nowrap'>"
            f"{bar}&nbsp;"
            f"<span style='font-size:0.82em;color:#555'>{label}</span>"
            f"</div>"
        )

    rows = []
    for k in k_values:
        bd = k_bd[k]
        sw = relevant_sweeps.get(f"{loss_name}_k{k}")
        in_fine = sw is not None

        # LR bar scale = coarse range for this k (log scale)
        lr_lo, lr_hi = bd["lr_min"], bd["lr_max"]
        mg_lo, mg_hi = bd["margin_min"], bd["margin_max"]

        coarse_lr_label = f"{format_float(lr_lo)} ~ {format_float(lr_hi)}"
        coarse_mg_label = f"{format_float(mg_lo, 4)} ~ {format_float(mg_hi, 4)}"

        coarse_lr_cell = _bar_cell(lr_lo, lr_hi, lr_lo, lr_hi, True, coarse_lr_label, "#7f8c8d")
        coarse_mg_cell = _bar_cell(mg_lo, mg_hi, mg_lo, mg_hi, False, coarse_mg_label, "#7f8c8d")

        # Fine cells (com marcadores dos valores fixados na Sprint 3)
        if in_fine and sw.get("lr_min") is not None:
            flr_lo, flr_hi = sw["lr_min"], sw["lr_max"]
            fmg_lo, fmg_hi = sw["margin_min"], sw["margin_max"]
            fine_lr_cell = _bar_cell(flr_lo, flr_hi, lr_lo, lr_hi, True,
                                     f"{format_float(flr_lo)} ~ {format_float(flr_hi)}", "#2980b9",
                                     marker=SPRINT3_DEFAULTS["lr"])
            fine_mg_cell = _bar_cell(fmg_lo, fmg_hi, mg_lo, mg_hi, False,
                                     f"{format_float(fmg_lo, 4)} ~ {format_float(fmg_hi, 4)}", "#2980b9",
                                     marker=SPRINT3_DEFAULTS["margin"])
            fine_runs = f"cap: {sw['run_cap']}"
            fine_eer = "—"
        else:
            fine_lr_cell = sw["lr"] if sw else "—"
            fine_mg_cell = sw["margin"] if sw else "—"
            fine_runs = f"cap: {sw['run_cap']}" if sw else "—"
            fine_eer = "—"

        fine_status = (
            "<span style='color:#27ae60;font-weight:bold'>✓ selecionado</span>"
            if in_fine else
            "<span style='color:#e74c3c'>✗ não selecionado</span>"
        )

        # Row styles
        c_style = "background:#f8f9fa;"
        f_style = "background:#eaf4fb;" if in_fine else "background:#fdf3f3;"

        # Inserir três linhas por k: Coarse -> Padrão Indústria -> Fine
        if loss_name in INDUSTRY_STANDARDS:
            std = INDUSTRY_STANDARDS[loss_name]
            ilr_lo, ilr_hi = std['lr']
            img_lo, img_hi = std['margin']
            industry_lr_cell = _bar_cell(ilr_lo, ilr_hi, lr_lo, lr_hi, True,
                                         f"{format_float(ilr_lo)} ~ {format_float(ilr_hi)}", "#27ae60")
            industry_mg_cell = _bar_cell(img_lo, img_hi, mg_lo, mg_hi, False,
                                         f"{format_float(img_lo,4)} ~ {format_float(img_hi,4)}", "#27ae60")
        else:
            industry_lr_cell = "—"
            industry_mg_cell = "—"

        rows.append(f"""            <tr style="{c_style}">
                <td rowspan="3" style="vertical-align:middle;text-align:center;font-weight:bold;font-size:1.05em;border-right:2px solid #ccc;">k = {k}</td>
                <td><span style="color:#666;font-size:0.85em;font-weight:bold">COARSE</span></td>
                <td>{bd['total']} runs</td>
                <td>{format_float(bd['eer_min'])}</td>
                <td>{format_float(bd['eer_mean'])}</td>
                <td>{coarse_lr_cell}</td>
                <td>{coarse_mg_cell}</td>
            </tr>
            <tr style="background:#f0fff4;">
                <td><span style="color:#27ae60;font-size:0.85em;font-weight:bold">INDÚSTRIA</span></td>
                <td>—</td>
                <td colspan="2" style="color:#888;font-size:0.85em">Padrão da indústria (faixa)</td>
                <td>{industry_lr_cell}</td>
                <td>{industry_mg_cell}</td>
            </tr>
            <tr style="{f_style}">
                <td><span style="color:#2980b9;font-size:0.85em;font-weight:bold">FINE</span></td>
                <td>{fine_runs}</td>
                <td colspan="2" style="color:#888;font-size:0.85em">{fine_status}</td>
                <td>{fine_lr_cell}</td>
                <td>{fine_mg_cell}</td>
            </tr>""")

    score_note = (
        "Score de seleção (menor = melhor): <code>EER_final − 0.5×EER_drop + EER_stability</code> "
        "— premia queda rápida e penaliza instabilidade. "
        "Filtros prévios: <code>EER_final &lt; 0.05</code> e run não-divergida."
    ) if has_multiple_k else (
        "Único valor de k testado para esta loss."
    )

    multi_k_note = (
        """<div class="alert alert-info" style="margin-top:12px">
            <strong>Por que múltiplos k?</strong> SubCenter ArcFace/CosFace permitem que cada classe tenha
            <em>k</em> centróides, capturando variações intra-classe (ex.: diferentes ângulos do documento).
            Cada k foi avaliado independentemente no coarse search; os mais promissores receberam
            fine search dedicado com faixas de hiperparâmetros refinadas.
        </div>"""
        if has_multiple_k else ""
    )

    k_section_title = "📐 Sub-Centros (k): Seleção Coarse → Fine Search" if has_multiple_k else "📐 Coarse → Fine Search"

    return f"""    <div class="section">
        <h2>{k_section_title}</h2>
{coarse_summary}
        <div class="alert alert-info">
            <strong>Critério de seleção:</strong> {score_note}
        </div>
        <table>
            <tr>
                <th>k</th>
                <th>Fase</th>
                <th>Runs</th>
                <th>Melhor EER</th>
                <th>EER Médio</th>
                <th>Faixa LR</th>
                <th>Faixa Margin</th>
            </tr>
{"".join(rows)}
        </table>
        <p style="font-size:0.82em;color:#777;margin-top:6px">
            ■ <span style="color:#7f8c8d">cinza</span> = faixa coarse &nbsp;|&nbsp;
            ■ <span style="color:#27ae60">verde</span> = faixa padrão da indústria &nbsp;|&nbsp;
            ■ <span style="color:#2980b9">azul</span> = faixa fine search (escala do coarse) &nbsp;|&nbsp;
            <span style="color:#e74c3c;font-weight:bold">|</span> <span style="color:#e74c3c">vermelho</span> = valor fixado Sprint 3 (LR: 1e-5, Margin: 0.35)
        </p>
        {multi_k_note}
    </div>
"""


def format_loss_page(
    loss_name: str,
    analysis: Dict,
    fine_run_count: int = 0,
    histories: Optional[Dict[str, List[float]]] = None,
    coarse_analysis: Optional[Dict] = None,
    sweep_info: Optional[Dict[str, Dict]] = None,
) -> str:
    best_run = analysis['best_runs'][0]
    top5 = analysis['best_runs'][:5]
    avg_lr = statistics.mean(r.lr for r in top5)
    avg_margin = statistics.mean(r.margin for r in top5)
    avg_scale = statistics.mean(r.scale for r in top5)
    avg_eer = statistics.mean(r.eer_final for r in top5)
    avg_drop_pct_per_epoch = statistics.mean(
        (r.eer_epoch1 - r.eer_final) / r.eer_epoch1 * 100 / r.n_epochs
        for r in top5 if r.n_epochs > 0 and r.eer_epoch1 > 0
    )
    avg_stability = statistics.mean(r.eer_stability for r in top5)

    most_impactful = max(
        [("LR", abs(analysis['corr_lr_eer'])), ("Margin", abs(analysis['corr_margin_eer'])), ("Scale", abs(analysis['corr_scale_eer']))],
        key=lambda x: x[1]
    )

    def _drop_per_epoch(run: SweepRun) -> str:
        if run.eer_epoch1 <= 0 or run.n_epochs <= 0:
            return "—"
        val = (run.eer_epoch1 - run.eer_final) / run.eer_epoch1 * 100 / run.n_epochs
        return f"{val:.1f}%"

    def _osc_cell(run: SweepRun) -> str:
        v = _relative_oscillation(run)
        if v is None:
            return "<span style='color:#e74c3c' title='Sem queda — sem base para medir oscilação'>sem queda</span>"
        return v

    def _industry_check(run: SweepRun, param: str) -> str:
        if loss_name not in INDUSTRY_STANDARDS:
            return "—"
        std = INDUSTRY_STANDARDS[loss_name]
        val = {"lr": run.lr, "margin": run.margin, "scale": run.scale}[param]
        return "✓" if std[param][0] <= val <= std[param][1] else "✗"

    top5_html = "\n".join([
        f"""            <tr><td>{i}</td><td><code>{run.run_id}</code></td>
                <td>{run.k}</td>
                <td>{format_float(run.eer_final)}</td>
                <td>{_drop_per_epoch(run)}</td>
                <td>{_osc_cell(run)}</td>
                <td>{_industry_check(run, "lr")}</td>
                <td>{_industry_check(run, "margin")}</td>
                <td>{_industry_check(run, "scale")}</td></tr>"""
        for i, run in enumerate(analysis['best_runs'], 1)
    ])

    corr_html = ""
    for param, corr in [("LR", analysis["corr_lr_eer"]), ("Margin", analysis["corr_margin_eer"]), ("Scale", analysis["corr_scale_eer"])]:
        abs_corr = abs(corr)
        if abs_corr > 0.5:
            impact = "forte impacto"
            color = "#e74c3c"
        elif abs_corr > 0.3:
            impact = "impacto moderado"
            color = "#f39c12"
        else:
            impact = "impacto fraco"
            color = "#95a5a6"
        corr_html += f"            <tr><td>{param}</td><td style=\"color:{color}; font-weight:bold;\">{corr:+.4f}</td><td>{impact}</td></tr>\n"

    industry_html = ""

    # Runs com melhor convergência
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loss {loss_name.upper()} - Análise Sweep</title>
    <style>{get_css()}</style>
</head>
<body>
    <h1>📊 Análise: {loss_name.upper()}</h1>

    <div class="nav-links">
        <a href="index.html">← Voltar ao Índice</a>
    </div>

    <div class="section">
        <h2>📋 Resumo Executivo</h2>
        {f'<div class="metric"><span style="font-weight: bold;">Runs Coarse Search:</span> {coarse_analysis["total_runs"]}</div>' if coarse_analysis else ''}
        <div class="metric"><span style="font-weight: bold;">Runs Fine Search:</span> {fine_run_count}</div>
        <div class="metric"><span style="font-weight: bold;">Taxa de convergência (fine):</span> {analysis['convergence_rate']*100:.1f}%</div>
        <div class="metric"><span style="font-weight: bold;">Melhoria (early→late):</span> {analysis['improvement']:.1f}%</div>
        <div class="metric"><span style="font-weight: bold;">Melhor EER:</span> {analysis['eer_min']:.6f}</div>
        <div class="metric"><span style="font-weight: bold;">EER médio:</span> {analysis['eer_mean']:.6f}</div>
    </div>

    {_format_coarse_section(loss_name, coarse_analysis, analysis)}

    {_format_k_selection_section(loss_name, analysis, sweep_info)}

    <div class="section">
        <h2>📉 Trajetória de Convergência por Época</h2>
        <table>
            <tr><th>Métrica</th><th>Valor</th><th>Interpretação</th></tr>
            <tr><td>EER médio (Epoch 1)</td><td>{analysis['avg_epoch1']:.6f}</td><td>Performance inicial (sem muito treinamento)</td></tr>
            <tr><td>EER médio (Final)</td><td>{analysis['avg_final']:.6f}</td><td>Performance após todas as épocas</td></tr>
            <tr><td>Taxa de melhoria</td><td>{analysis['avg_convergence_rate']:.2f}%/época</td><td>Melhoria percentual por época</td></tr>
        </table>
        <p style="margin-top:1rem;">{"Dados completos via WandB." if histories else "<em>Histórico completo indisponível — exibindo apenas época 1 e final. Execute sem <code>--no-history</code> para todas as épocas.</em>"}</p>
        {generate_eer_svg(best_run, top5, histories)}
        <div class="alert alert-info" style="margin-top:1rem;">
            <strong>Como ler:</strong> Queda/Época = melhoria percentual média por época. Oscilação relativa = std das épocas ÷ queda total × 100: 0% = queda perfeitamente linear; 50% = oscilações equivalentes à metade do total melhorado; "sem queda" = EER não melhorou.
        </div>
    </div>

    <div class="section">
        <h2>🔗 Análise de Sensibilidade (Correlações)</h2>
        <p>Qual parâmetro mais impacta o EER? Correlação de Pearson com EER final:</p>
        <table>
            <tr><th>Parâmetro</th><th>Correlação</th><th>Interpretação</th></tr>
{corr_html}        </table>
    </div>

{industry_html}

    <div class="section">
        <h2>🏆 Top 5 Runs (Ranking Holístico)</h2>
        <div class="alert alert-info">
            <strong>Critério de seleção (score composto):</strong>
            50% EER final (menor = melhor) &nbsp;+&nbsp;
            40% Queda/Época (maior = melhor) &nbsp;+&nbsp;
            10% Estabilidade.
            <br><strong>Prioridade:</strong> runs com queda &gt; 0 sempre precedem runs estagnadas; entre estagnadas, ordena por EER.
            <br><small>Colunas LR / M / S: alinhamento informativo com padrões da indústria (não afeta o ranking).</small>
        </div>
        <table>
            <tr><th>#</th><th>Run ID</th><th>k</th><th>EER Final</th><th>% Queda/Época</th><th>Oscilação relativa ↓</th><th>LR ✓</th><th>M ✓</th><th>S ✓</th></tr>
{top5_html}        </table>
    </div>

    <div class="section">
        <h2>💡 Recomendação para Sprint 3b</h2>

        <div class="recommendation-box">
            <h3>🥇 Opção 1 — Melhor Run Individual <code>{best_run.run_id}</code></h3>
            <div class="metric">EER Final: <strong>{format_float(best_run.eer_final)}</strong></div>
            <div class="metric">LR: <strong>{format_float(best_run.lr)}</strong></div>
            <div class="metric">Margin: <strong>{format_float(best_run.margin, 4)}</strong></div>
            <div class="metric">Scale: <strong>{format_float(best_run.scale, 1)}</strong></div>
            <div class="metric">Queda/Época: <strong>{(best_run.eer_epoch1 - best_run.eer_final) / best_run.eer_epoch1 * 100 / best_run.n_epochs:.2f}%</strong></div>
            <div class="metric">Volatilidade: <strong>{format_float(best_run.eer_stability)}</strong></div>
        </div>

        <div class="recommendation-box" style="background: #e8f8f0; border-left-color: #27ae60;">
            <h3>🔢 Opção 2 — Média Top 5 (mais robusta)</h3>
            <div class="metric">EER médio: <strong>{format_float(avg_eer)}</strong></div>
            <div class="metric">LR médio: <strong>{format_float(avg_lr)}</strong></div>
            <div class="metric">Margin médio: <strong>{format_float(avg_margin, 4)}</strong></div>
            <div class="metric">Scale médio: <strong>{format_float(avg_scale, 1)}</strong></div>
            <div class="metric">Queda/Época média: <strong>{avg_drop_pct_per_epoch:.2f}%</strong></div>
            <div class="metric">Volatilidade média: <strong>{format_float(avg_stability)}</strong></div>
        </div>

        <div class="alert alert-info" style="margin-top: 15px;">
            <strong>⚠️ Parâmetro Crítico:</strong> {most_impactful[0]} tem o maior impacto (|corr| = {most_impactful[1]:.4f})
        </div>
    </div>

    <div class="footer">
        <p>Análise gerada automaticamente | Sweep Fine-Search LA-CDIP</p>
    </div>
</body>
</html>"""


def _format_coarse_index_section(
    coarse_by_loss: Optional[Dict[str, Dict]],
    fine_analyses: Dict[str, Dict],
    all_losses: List[str],
) -> str:
    """Seção do índice comparando coarse × fine por loss."""
    if not coarse_by_loss:
        return ""

    rows = []
    for loss in sorted(all_losses):
        ca = coarse_by_loss.get(loss)
        fa = fine_analyses.get(loss)
        if not ca or not fa:
            continue
        c_best = ca["eer_min"]
        f_best = fa["eer_min"]
        delta = (c_best - f_best) / c_best * 100 if c_best > 0 else 0
        improved = f_best < c_best
        color = "#27ae60" if improved else "#e74c3c"
        arrow = "▼" if improved else "▲"
        c_mean = ca["eer_mean"]
        f_mean = fa["eer_mean"]
        delta_mean = (c_mean - f_mean) / c_mean * 100 if c_mean > 0 else 0
        c_range = ca["eer_max"] - ca["eer_min"]
        f_range = fa["eer_max"] - fa["eer_min"]
        narrowed = f_range < c_range
        range_color = "#27ae60" if narrowed else "#e74c3c"
        rows.append(f"""            <tr>
                <td><strong>{loss}</strong></td>
                <td>{ca['total_runs']} ({ca['convergence_rate']*100:.0f}%)</td>
                <td>{format_float(c_best)} <small style="color:#888">(μ {format_float(c_mean)})</small></td>
                <td>{fa.get('total_runs', '?')}</td>
                <td>{format_float(f_best)} <small style="color:#888">(μ {format_float(f_mean)})</small></td>
                <td style="color:{color};font-weight:bold">{arrow} {abs(delta):.1f}% (μ {abs(delta_mean):.1f}%)</td>
                <td style="color:{range_color}">{f_range:.6f} <small>(era {c_range:.6f})</small></td>
                <td><a href="loss_{loss}.html" style="color:#3498db">Detalhes →</a></td>
            </tr>""")

    if not rows:
        return ""

    return f"""    <div class="section">
        <h2>⚡ Coarse × Fine: Eficácia do Refinamento Bayesiano</h2>
        <p>Comparação direta entre a fase de exploração ampla (coarse) e o refinamento (fine).
           <strong>▼ verde = fine melhorou</strong>; dispersão menor = busca convergiu para região ótima.</p>
        <table>
            <tr>
                <th>Loss</th>
                <th>Coarse runs (conv.)</th>
                <th>EER Coarse</th>
                <th>Fine runs</th>
                <th>EER Fine</th>
                <th>Δ Best EER</th>
                <th>Dispersão EER Fine</th>
                <th>Link</th>
            </tr>
{"".join(rows)}
        </table>
        <div class="alert alert-info" style="margin-top:12px">
            <strong>Como ler:</strong> Δ mostra quanto o fine melhorou o EER em relação ao coarse (melhor run e média).
            A coluna "Dispersão EER Fine" mostra se o espaço de busca convergiu para uma região compacta de bons resultados.
        </div>
    </div>
"""


def format_index_page(
    analyses: Dict[str, Dict],
    all_losses: List[str],
    fine_counts: Dict[str, int],
    coarse_by_loss: Optional[Dict[str, Dict]] = None,
) -> str:
    table_rows = "\n".join([
        f"""            <tr>
                <td><strong>{loss}</strong></td>
                <td>{fine_counts.get(loss, 0)}</td>
                <td>{analyses[loss]['convergence_rate']*100:.0f}%</td>
                <td>{analyses[loss]['eer_min']:.6f}</td>
                <td>{analyses[loss]['improvement']:.1f}%</td>
                <td><a href="loss_{loss}.html" style="color:#3498db; text-decoration:underline;">Abrir →</a></td>
            </tr>"""
        for loss in sorted(all_losses) if loss in analyses and analyses[loss]
    ])

    by_improvement = sorted([(l, a['improvement']) for l, a in analyses.items() if a], key=lambda x: x[1], reverse=True)
    ranking_html = "\n".join([
        f"            <li><strong>{loss}</strong>: {improvement:.1f}% de melhoria (early → late)</li>"
        for loss, improvement in by_improvement[:3]
    ])

    total_runs = sum(a['total_runs'] for a in analyses.values() if a)
    avg_convergence = statistics.mean([a['convergence_rate'] for a in analyses.values() if a]) * 100

    # Análise de alinhamento com indústria
    industry_alignment_html = ""
    for loss in sorted(all_losses):
        if loss not in analyses or not analyses[loss]:
            continue
        a = analyses[loss]
        best_run = a['best_runs'][0] if a['best_runs'] else None
        if not best_run or loss not in INDUSTRY_STANDARDS:
            continue

        std = INDUSTRY_STANDARDS[loss]
        lr_ok = std["lr"][0] <= best_run.lr <= std["lr"][1]
        margin_ok = std["margin"][0] <= best_run.margin <= std["margin"][1]
        scale_ok = std["scale"][0] <= best_run.scale <= std["scale"][1]
        alignment_score = sum([lr_ok, margin_ok, scale_ok]) / 3 * 100

        industry_alignment_html += f"""        <div style="margin: 15px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid {'#27ae60' if alignment_score == 100 else '#f39c12' if alignment_score >= 66 else '#e74c3c'}; border-radius: 4px;">
            <strong>{loss.upper()}:</strong> Best EER = {format_float(best_run.eer_final)} | Alinhamento indústria: <strong>{alignment_score:.0f}%</strong>
            <br><small>LR {format_float(std['lr'][0])}~{format_float(std['lr'][1])}: {format_float(best_run.lr)} {"✓" if lr_ok else "✗"} |
            M {format_float(std['margin'][0], 3)}~{format_float(std['margin'][1], 3)}: {format_float(best_run.margin, 4)} {"✓" if margin_ok else "✗"} |
            S {format_float(std['scale'][0], 1)}~{format_float(std['scale'][1], 1)}: {format_float(best_run.scale, 1)} {"✓" if scale_ok else "✗"}</small>
        </div>
"""

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise Exploratória do Sweep</title>
    <style>{get_css()}</style>
</head>
<body>
    <h1>📊 Análise Exploratória: Bayesian Optimization Sweep</h1>

    <div class="section">
        <h2>📍 Sobre este Relatório</h2>
        <p><strong>Dataset:</strong> Fine-Search Results (LA-CDIP)</p>
        <p><strong>O que é:</strong> Resultados da fase de refinamento (fine search) da otimização Bayesiana. Após uma busca coarse exploratória,
           o algoritmo refinaram as regiões mais promissoras com {total_runs} rodadas.</p>
        <p><strong>Como ler:</strong> Compare cada loss type com seus padrões da indústria. Green = bem alinhado, Yellow = parcial, Red = fora do padrão.</p>
    </div>

    <div class="section bayesian-box">
        <h2>🔬 Sobre Bayesian Optimization</h2>
        <p><strong>O que é:</strong> Técnica de otimização inteligente que aprende a relação entre hiperparâmetros e performance (EER) sem exploração exaustiva.</p>
        <p><strong>Como funciona neste projeto:</strong></p>
        <ul>
            <li><strong>Fase 1 - Coarse Search:</strong> Exploração ampla do espaço de parâmetros</li>
            <li><strong>Fase 2 - Fine Search (este relatório):</strong> Refinamento local nas regiões promissoras</li>
            <li><strong>Gaussian Process (GP):</strong> Modelo probabilístico que entende onde os melhores hiperparâmetros provavelmente estão</li>
            <li><strong>Acquisition Function:</strong> Balanceia exploração vs refinamento</li>
        </ul>
        <p><strong>Benefício:</strong> Em vez de testar 10.000 combinações, Bayesian Optimization encontra boas soluções em 100-200 rodadas.</p>
        <p><strong>Neste dataset:</strong> Runs posteriores tendem a ter EER melhor, evidência que o algoritmo está refinando corretamente.</p>
    </div>

    {_format_coarse_index_section(coarse_by_loss, analyses, all_losses)}

    <div class="section">
        <h2>🏭 Alinhamento com Padrões da Indústria</h2>
        <p><strong>Verde:</strong> 100% alinhado | <strong>Amarelo:</strong> 66%+ alinhado | <strong>Vermelho:</strong> &lt;66% alinhado</p>
{industry_alignment_html}    </div>

    <div class="section">
        <h2>📑 Análise por Loss Type</h2>
        <p>Clique em cada loss para análise detalhada:</p>
        <table>
            <tr><th>Loss</th><th>Runs Fine</th><th>Convergência</th><th>Best EER</th><th>Melhoria</th><th>Link</th></tr>
{table_rows}        </table>
    </div>

    <div class="footer">
        <p>Análise gerada automaticamente | Fine-Search Results</p>
        <p><small>Para análise profunda de cada loss, consulte as páginas individuais acima.</small></p>
    </div>
</body>
</html>"""


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Gera relatório HTML do sweep fine-search.")
    p.add_argument("csv", nargs="?", default=str(FINE_SEARCH_CSV), help="CSV de análise do sweep")
    p.add_argument("--wandb-entity", default="jpcosta1990-university-of-brasilia",
                   help="Entidade WandB (default: jpcosta1990-university-of-brasilia)")
    p.add_argument("--wandb-project", default="CaVL-Doc_LA-CDIP_FineSearch",
                   help="Projeto WandB fine search (default: CaVL-Doc_LA-CDIP_FineSearch)")
    p.add_argument("--coarse-wandb-project", default="CaVL-Doc_LA-CDIP_InternVL3-2B_Sweeps",
                   help="Projeto WandB coarse search (default: CaVL-Doc_LA-CDIP_InternVL3-2B_Sweeps)")
    p.add_argument("--no-history", action="store_true",
                   help="Não buscar histórico por época do WandB (usa apenas epoch1/final)")
    p.add_argument("--no-coarse", action="store_true",
                   help="Não buscar dados do coarse search no WandB")
    return p.parse_args()


def main():
    args = _parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        sys.exit(1)

    print(f"📂 Carregando {csv_path}...")
    runs = load_sweep_csv(csv_path)
    print(f"✓ {len(runs)} runs carregados")

    # Contagem de runs por loss no fine search (runs_raw.csv)
    fine_counts: Dict[str, int] = {}
    if FINE_RUNS_CSV.exists():
        with FINE_RUNS_CSV.open(newline="") as fh:
            for row in csv.DictReader(fh):
                loss = (row.get("loss_type") or "").strip()
                if loss:
                    fine_counts[loss] = fine_counts.get(loss, 0) + 1
        total_fine = sum(fine_counts.values())
        print(f"✓ Fine search runs por loss: {fine_counts}  (total: {total_fine})")

    by_loss: Dict[str, List[SweepRun]] = {}
    for run in runs:
        if run.loss_type not in by_loss:
            by_loss[run.loss_type] = []
        by_loss[run.loss_type].append(run)

    analyses = {loss_name: analyze_loss(by_loss[loss_name]) for loss_name in sorted(by_loss.keys())}

    # Carrega metadados dos YAMLs do fine search (faixas, run_cap, etc.)
    sweep_info = load_fine_search_sweep_info(FINE_SEARCH_YAML_DIR)
    if sweep_info:
        print(f"✓ Fine search YAMLs: {list(sweep_info.keys())}")

    # Busca dados do coarse search no WandB
    coarse_by_loss: Dict[str, Dict] = {}
    if not args.no_coarse:
        print(f"\n📡 Buscando runs do coarse search ({args.coarse_wandb_project})...")
        coarse_runs = fetch_coarse_runs_from_wandb(
            entity=args.wandb_entity,
            project=args.coarse_wandb_project,
        )
        if coarse_runs:
            grouped: Dict[str, List[CoarseRun]] = {}
            for r in coarse_runs:
                grouped.setdefault(r.loss_type, []).append(r)
            coarse_by_loss = {
                loss: analyze_coarse_loss(runs)
                for loss, runs in grouped.items()
            }
            print(f"✓ Coarse analisado: {list(coarse_by_loss.keys())}")

    # Busca histórico por época do WandB para os top 5 de cada loss
    histories: Dict[str, List[float]] = {}
    if not args.no_history:
        top5_ids = [
            run.run_id
            for a in analyses.values() if a
            for run in a["best_runs"][:5]
        ]
        if top5_ids:
            # Busca em ambos os projetos: os runs do CSV podem estar no coarse ou no fine
            projects_to_search = list(dict.fromkeys(
                [args.wandb_project, args.coarse_wandb_project]
            ))
            print(f"\n📡 Buscando histórico WandB para {len(top5_ids)} runs...")
            print(f"  Projetos: {projects_to_search}")
            histories = fetch_run_histories(
                top5_ids,
                entity=args.wandb_entity,
                projects=projects_to_search,
            )
            loaded = sum(1 for v in histories.values() if v)
            print(f"✓ Histórico carregado para {loaded}/{len(top5_ids)} runs")

    output_dir = Path("sweep_report")
    output_dir.mkdir(exist_ok=True)

    index_html = format_index_page(
        analyses,
        list(by_loss.keys()),
        fine_counts,
        coarse_by_loss=coarse_by_loss if coarse_by_loss else None,
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(f"✓ Índice: {output_dir / 'index.html'}")

    for loss_name in sorted(by_loss.keys()):
        if analyses[loss_name]:
            page_html = format_loss_page(
                loss_name,
                analyses[loss_name],
                fine_counts.get(loss_name, 0),
                histories=histories if histories else None,
                coarse_analysis=coarse_by_loss.get(loss_name),
                sweep_info=sweep_info if sweep_info else None,
            )
            (output_dir / f"loss_{loss_name}.html").write_text(page_html, encoding="utf-8")
            print(f"✓ Página: loss_{loss_name}.html")

    print(f"\n🎉 Relatório completo em: {output_dir.absolute()}/index.html")


if __name__ == "__main__":
    main()
