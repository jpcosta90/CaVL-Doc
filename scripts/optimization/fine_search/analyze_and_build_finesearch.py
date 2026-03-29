#!/usr/bin/env python3
"""
analyze_and_build_finesearch.py
==============================================================================
Pipeline multi-fase de Fine Search para o projeto CaVL-Doc.

Fase 1 — Download e Cálculo de Métricas Multi-Objetivo (WandB)
Fase 2 — Poda do Espaço de Busca (Filtros Matemáticos)
Fase 3 — Geração de Sweep YAMLs de Alta Fidelidade (method: bayes)

Uso:
    # RVL-CDIP (padrão):
    python scripts/optimization/fine_search/analyze_and_build_finesearch.py \
        --dataset rvlcdip \
        --entity <seu_wandb_entity>

    # LA-CDIP:
    python scripts/optimization/fine_search/analyze_and_build_finesearch.py \
        --dataset lacdip \
        --entity <seu_wandb_entity>

    # Só análise (sem gerar YAMLs):
    python scripts/optimization/fine_search/analyze_and_build_finesearch.py --dataset rvlcdip --analyze-only

    # Forçar thresholds:
    python scripts/optimization/fine_search/analyze_and_build_finesearch.py \
        --dataset rvlcdip --eer-ceiling 0.40 --top-k 5
==============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURAÇÃO POR DATASET
# ==============================================================================

DATASET_CONFIGS = {
    "rvlcdip": {
        "display_name"     : "RVL-CDIP",
        "wandb_project"    : "CaVL-Doc_RVL-CDIP_InternVL3-2B_Sweeps",
        "output_project"   : "CaVL-Doc_RVL-CDIP_FineSearch",
        "output_dir"       : Path("scripts/optimization/fine_search/configs/rvlcdip"),
        "bash_script"      : Path("scripts/optimization/fine_search/setup_finesearch_rvlcdip.sh"),
        "eer_ceiling"      : 0.35,
        # Parâmetros do command block do YAML gerado
        "dataset_name"     : "RVL-CDIP",
        "pairs_csv"        : "data/RVL-CDIP/train_pairs.csv",
        "base_image_dir"   : "/mnt/data/zs_rvl_cdip/data",
        "val_subset"       : 512,
        "epochs"           : 5,
        "max_steps"        : 100,
        "n_experiments"    : 10,
        "grad_accum"       : 4,
        "pool_size"        : 8,
        "batch_size"       : 8,
    },
    "lacdip": {
        "display_name"     : "LA-CDIP",
        "wandb_project"    : "CaVL-Doc_LA-CDIP_InternVL3-2B_Sweeps",
        "output_project"   : "CaVL-Doc_LA-CDIP_FineSearch",
        "output_dir"       : Path("scripts/optimization/fine_search/configs/lacdip"),
        "bash_script"      : Path("scripts/optimization/fine_search/setup_finesearch_lacdip.sh"),
        "eer_ceiling"      : 0.05,
        # Parâmetros do command block do YAML gerado
        "dataset_name"     : "LA-CDIP",
        "pairs_csv"        : "data/LA-CDIP/train_pairs.csv",
        "base_image_dir"   : "/mnt/data/la-cdip/data",
        "val_subset"       : 512,
        "epochs"           : 5,
        "max_steps"        : 100,
        "n_experiments"    : 10,
        "grad_accum"       : 2,
        "pool_size"        : 8,
        "batch_size"       : 8,
    },
}

# Épocas e steps do Fine Search (alta fidelidade)
FINE_SEARCH_EPOCHS    = 5
FINE_SEARCH_MAX_STEPS = 100   # sobrescrito per-dataset via config


def build_yaml_command_block(cfg: dict) -> str:
    """Monta o bloco 'command:' do YAML com os parâmetros do dataset."""
    lines = [
        "command:",
        "  - ${env}",
        "  - ${interpreter}",
        "  - ${program}",
        '  - "--use-wandb"',
        '  - "--gradient-accumulation-steps"',
        f'  - "{cfg["grad_accum"]}"',
        '  - "--candidate-pool-size"',
        f'  - "{cfg["pool_size"]}"',
        '  - "--student-batch-size"',
        f'  - "{cfg["batch_size"]}"',
        '  - "--dataset-name"',
        f'  - "{cfg["dataset_name"]}"',
        '  - "--val-subset-size"',
        f'  - "{cfg["val_subset"]}"',
        '  - "--pairs-csv"',
        f'  - "{cfg["pairs_csv"]}"',
        '  - "--base-image-dir"',
        f'  - "{cfg["base_image_dir"]}"',
        '  - "--epochs"',
        f'  - "{cfg["epochs"]}"',
        '  - "--weight-decay"',
        '  - "0.05"',
        '  - "--num-workers"',
        '  - "2"',
        '  - "--max-steps-per-epoch"',
        f'  - "{cfg["max_steps"]}"',
        '  - "--model-name"',
        '  - "InternVL3-2B"',
        "  - ${args}",
    ]
    return "\n".join(lines) + "\n"


# ==============================================================================
# FASE 1 — EXTRAÇÃO E CÁLCULO DE MÉTRICAS
# ==============================================================================

def fetch_runs(project: str, entity: str | None) -> pd.DataFrame:
    """
    Baixa todos os runs finalizados do projeto WandB e calcula três métricas:
      - eer_final    : menor val/eer atingido (desempenho)
      - eer_drop     : eer_epoch1 − eer_final  (velocidade de queda)
      - eer_stability: desvio padrão da curva val/eer (estabilidade)

    Retorna um DataFrame com uma linha por run.
    """
    try:
        import wandb
    except ImportError:
        print("❌  wandb não instalado. Execute: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"\n📡 Conectando ao projeto WandB: {path}")

    try:
        runs = api.runs(path, filters={"state": {"$in": ["finished", "crashed"]}})
    except Exception as e:
        print(f"❌  Erro ao acessar o projeto: {e}")
        sys.exit(1)

    records = []
    total = len(runs)
    print(f"   {total} runs encontrados. Calculando métricas…")

    for i, run in enumerate(runs, 1):
        cfg = run.config

        def cfg_get(key):
            return cfg.get(key, cfg.get(key.replace("-", "_"), None))

        loss_type = cfg_get("loss-type")
        lr        = cfg_get("student-lr")
        margin    = cfg_get("margin")
        scale     = cfg_get("scale")
        k         = cfg_get("num-sub-centers")

        if loss_type is None or lr is None:
            continue

        try:
            hist = run.history(keys=["epoch", "val/eer"], pandas=True)
        except Exception:
            continue

        if hist.empty or "val/eer" not in hist.columns:
            continue

        hist = hist.dropna(subset=["val/eer"]).sort_values("epoch").reset_index(drop=True)
        eer_series = hist["val/eer"].values

        if len(eer_series) == 0:
            continue

        eer_final    = float(np.min(eer_series))
        eer_epoch1   = float(eer_series[0])
        eer_drop     = float(eer_epoch1 - eer_final)
        eer_stability = float(np.std(eer_series))

        diverged = bool(
            len(eer_series) >= 2 and eer_series[-1] > eer_series[0]
        )

        records.append({
            "run_id"       : run.id,
            "run_name"     : run.name,
            "loss_type"    : loss_type,
            "lr"           : lr,
            "margin"       : margin,
            "scale"        : scale,
            "k"            : int(k) if k is not None else None,
            "eer_final"    : eer_final,
            "eer_epoch1"   : eer_epoch1,
            "eer_drop"     : eer_drop,
            "eer_stability": eer_stability,
            "diverged"     : diverged,
            "n_epochs"     : len(eer_series),
        })

        if i % 10 == 0:
            print(f"   … {i}/{total} runs processados")

    df = pd.DataFrame(records)
    print(f"\n✅  {len(df)} runs válidos com métricas calculadas.")
    return df


# ==============================================================================
# FASE 2 — PODA DO ESPAÇO DE BUSCA
# ==============================================================================

def prune(df: pd.DataFrame, eer_ceiling: float, top_k: int) -> dict:
    """
    Aplica regras de eliminação e retorna {label: info_dict}.

    Regras:
    1. Remoção de Runs Tóxicos  : EER acima de eer_ceiling OU run divergiu.
    2. Congelamento de Inertes  : Parâmetros com correlação |r| < 0.3 com EER
                                   são fixados ao valor mediano.
    3. Seleção Top-K            : Score composto = EER_final − 0.5×eer_drop + eer_stability
    """
    print(f"\n{'='*60}")
    print(f"  FASE 2 — PODA  (eer_ceiling={eer_ceiling}, top_k={top_k})")
    print(f"{'='*60}")

    before = len(df)
    df = df[~df["diverged"] & (df["eer_final"] < eer_ceiling)]
    print(f"  Regra 1 (Tóxicos):   {before} → {len(df)} runs restantes")

    if df.empty:
        print("  ⚠️  Nenhum run sobreviveu ao filtro básico. Tente aumentar --eer-ceiling.")
        return {}

    group_cols = ["loss_type", "k"]
    finalists = {}

    for (loss, k), grp in df.groupby(group_cols, dropna=False):
        label = f"{loss}_k{k}" if k is not None else loss

        if len(grp) < 2:
            print(f"  ⏩  {label}: menos de 2 runs, pulando.")
            continue

        # Regra 2: Congelamento de Inertes
        frozen_params = {}
        for param in ["margin", "scale"]:
            if param not in grp.columns:
                continue
            col = grp[param].dropna()
            if col.nunique() < 2:
                continue
            pearson_r = abs(grp[param].corr(grp["eer_final"]))
            if pearson_r < 0.3:
                median_val = round(float(col.median()), 4)
                frozen_params[param] = median_val
                print(f"  🔒  {label} | {param} inerte (|r|={pearson_r:.2f}) → fixado em {median_val}")

        # Regra 3: Score Composto
        grp = grp.copy()
        grp["score"] = grp["eer_final"] - 0.5 * grp["eer_drop"] + grp["eer_stability"]
        top = grp.nsmallest(top_k, "score")

        finalists[label] = {
            "df"           : top,
            "loss_type"    : loss,
            "k"            : k,
            "frozen_params": frozen_params,
        }
        print(f"\n  🏆  {label}: Top-{len(top)} finalistas:")
        print(top[["run_name", "lr", "margin", "scale", "eer_final",
                   "eer_drop", "eer_stability", "score"]].to_string(index=False))

    return finalists


# ==============================================================================
# FASE 3 — GERAÇÃO DOS YAMLs DE FINE SEARCH
# ==============================================================================

def _fixed_param_yaml(name: str, value) -> str:
    return f"  {name}:\n    value: {value}\n"


def _uniform_param_yaml(name: str, min_val: float, max_val: float) -> str:
    return (
        f"  {name}:\n"
        f"    distribution: uniform\n"
        f"    min: {min_val}\n"
        f"    max: {max_val}\n"
    )


def _log_uniform_param_yaml(name: str, min_val: float, max_val: float) -> str:
    return (
        f"  {name}:\n"
        f"    distribution: log_uniform_values\n"
        f"    min: {min_val:.2e}\n"
        f"    max: {max_val:.2e}\n"
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _build_data_driven_range(values: list[float],
                             hard_min: float,
                             hard_max: float,
                             min_rel_span: float = 0.20,
                             min_abs_span: float = 0.0) -> tuple[float, float]:
    """
    Gera faixa robusta com base em quantis dos finalistas e uma margem adicional.
    Evita ranges estreitos demais quando há pouca variabilidade.
    """
    if not values:
        return hard_min, hard_max

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return hard_min, hard_max

    q10 = float(np.quantile(arr, 0.10))
    q90 = float(np.quantile(arr, 0.90))
    center = float(np.median(arr))

    observed_span = max(q90 - q10, 1e-12)
    rel_span = max(abs(center) * min_rel_span, min_abs_span)
    target_span = max(observed_span, rel_span)

    low = center - target_span / 2
    high = center + target_span / 2

    low = _clamp(low, hard_min, hard_max)
    high = _clamp(high, hard_min, hard_max)

    if high <= low:
        eps = max(min_abs_span, (hard_max - hard_min) * 0.05)
        low = _clamp(center - eps / 2, hard_min, hard_max)
        high = _clamp(center + eps / 2, hard_min, hard_max)
        if high <= low:
            low, high = hard_min, hard_max

    return float(low), float(high)


def generate_yamls(finalists: dict, output_dir: Path, dataset_cfg: dict) -> list[Path]:
    """
    Gera um YAML (method: grid) por grupo finalista para o Fine Search.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    command_block = build_yaml_command_block(dataset_cfg)

    for label, info in finalists.items():
        top_df    = info["df"]
        loss_type = info["loss_type"]
        k         = info["k"]
        frozen    = info["frozen_params"]

        lr_vals = sorted(top_df["lr"].dropna().unique().tolist())
        # Scale fica FIXO no Fine Search—usar mediana dos finalistas
        margin_vals = sorted(top_df["margin"].dropna().unique().tolist())
        scale_vals  = sorted(top_df["scale"].dropna().unique().tolist())
        scale_median = float(np.median([v for v in scale_vals if pd.notna(v)])) if scale_vals else 32.0

        params_lines = []
        params_lines.append(_fixed_param_yaml("professor-warmup-steps", 100000))
        params_lines.append(_fixed_param_yaml("easy-mining-steps", 0))
        params_lines.append(_fixed_param_yaml("loss-type", f'"{loss_type}"'))
        params_lines.append(_fixed_param_yaml("optimizer-type", '"adamw"'))

        if k is not None:
            params_lines.append(_fixed_param_yaml("num-sub-centers", int(k)))

        # Fine Search com faixa contínua baseada nos melhores runs
        lr_min, lr_max = _build_data_driven_range(
            lr_vals,
            hard_min=1e-7,
            hard_max=1e-3,
            min_rel_span=0.35,
            min_abs_span=1e-6,
        )
        params_lines.append(_log_uniform_param_yaml("student-lr", lr_min, lr_max))

        if "margin" in frozen:
            params_lines.append(_fixed_param_yaml("margin", frozen["margin"]))
        elif margin_vals:
            margin_min, margin_max = _build_data_driven_range(
                margin_vals,
                hard_min=0.05,
                hard_max=0.95,
                min_rel_span=0.30,
                min_abs_span=0.06,
            )
            params_lines.append(_uniform_param_yaml(
                "margin", round(margin_min, 4), round(margin_max, 4)
            ))

        # Scale sempre fixo no Fine Search
        params_lines.append(_fixed_param_yaml("scale", round(scale_median, 1)))

        params_block = "parameters:\n" + "".join(params_lines)

        dataset_label = dataset_cfg["display_name"]
        yaml_content = (
            f"# Fine Search | {dataset_label} | {label}\n"
            f"# Gerado automaticamente por analyze_and_build_finesearch.py\n"
            f"# Finalistas do Sweep Grosseiro refinados com method: bayes\n\n"
            f"program: scripts/training/run_cavl_training.py\n"
            f"method: bayes\n"
            f"run_cap: {dataset_cfg['n_experiments']}\n"
            f"metric:\n"
            f"  name: val/eer\n"
            f"  goal: minimize\n\n"
            f"{params_block}\n"
            f"{command_block}"
        )

        out_path = output_dir / f"fine_search_{label}.yaml"
        out_path.write_text(yaml_content)
        generated.append(out_path)
        print(f"  📄  Gerado: {out_path}")

    return generated


# ==============================================================================
# RELATÓRIO FINAL + BASH SCRIPT
# ==============================================================================

def print_report(df_all: pd.DataFrame, finalists: dict,
                 generated_yamls: list[Path], output_dir: Path,
                 dataset_cfg: dict):
    print(f"\n{'='*60}")
    print("  RELATÓRIO FINAL")
    print(f"{'='*60}")

    print(f"\n  Dataset              : {dataset_cfg['display_name']}")
    print(f"  Runs analisados      : {len(df_all)}")
    print(f"  Grupos de loss+k     : {df_all.groupby(['loss_type','k'], dropna=False).ngroups}")
    print(f"  Grupos finalistas    : {len(finalists)}")
    print(f"  YAMLs gerados        : {len(generated_yamls)}")

    if generated_yamls:
        out_project = dataset_cfg["output_project"]
        print(f"\n  Próximos passos:")
        print(f"  1. Registrar sweeps (projeto: {out_project}):")
        for y in generated_yamls:
            print(f"     wandb sweep {y}  --project {out_project}")
        print(f"\n  2. Lançar agente:")
        print(f"     wandb agent <ENTITY>/{out_project}/<SWEEP_ID> --count {dataset_cfg['n_experiments']}")
        print(f"\n  3. Épocas: {dataset_cfg['epochs']} | "
              f"Max Steps: {dataset_cfg['max_steps']} | "
              f"Val Subset: {dataset_cfg['val_subset']} | "
              f"Experimentos: {dataset_cfg['n_experiments']}")

    csv_out = output_dir / "sweep_analysis.csv"
    df_all.to_csv(csv_out, index=False)
    print(f"\n  📊  Análise completa salva em: {csv_out}")


def generate_bash_script(finalists: dict, generated_yamls: list[Path],
                         output_dir: Path, dataset_cfg: dict):
    """Gera script bash pronto para registrar os sweeps de Fine Search."""
    out_project = dataset_cfg["output_project"]
    bash_path   = dataset_cfg["bash_script"]

    bash_lines = [
        "#!/bin/bash",
        f"# {bash_path.name}",
        "# Gerado automaticamente por analyze_and_build_finesearch.py",
        "",
        f'PROJECT="{out_project}"',
        'export WANDB_PROJECT=$PROJECT',
        "",
        f'echo "🔬 Registrando Fine Search Sweeps em: $PROJECT"',
        'echo "--------------------------------------------------"',
        "",
    ]

    agent_lines = [
        "",
        'echo ""',
        'echo "=================================================================="',
        'echo "✅ USE OS COMANDOS ABAIXO PARA LANÇAR OS AGENTES:"',
        'echo "=================================================================="',
        'echo ""',
    ]

    for label, info in finalists.items():
        yaml_path = output_dir / f"fine_search_{label}.yaml"
        var = f"SWEEP_{label.upper().replace('-','_').replace('.','_')}"
        bash_lines += [
            f'echo "Registrando {label}..."',
            f'OUT=$(wandb sweep {yaml_path} --project $PROJECT 2>&1)',
            f'{var}=$(echo "$OUT" | grep "wandb agent" | sed \'s/.*wandb agent //\')',
            "",
        ]
        n_runs = dataset_cfg["n_experiments"]
        agent_lines.append(f'echo "# {label}"')
        agent_lines.append(f'echo "wandb agent ${var} --count {n_runs}"')
        agent_lines.append('echo ""')

    bash_content = "\n".join(bash_lines + agent_lines) + "\n"
    bash_path.write_text(bash_content)
    bash_path.chmod(0o755)
    print(f"\n  🐚  Script bash gerado: {bash_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Analisa sweeps WandB e gera YAMLs de Fine Search."
    )
    p.add_argument("--dataset", choices=["rvlcdip", "lacdip"], default="rvlcdip",
                   help="Dataset alvo (default: rvlcdip)")
    p.add_argument("--project", default=None,
                   help="Nome do projeto WandB (sobrescreve o padrão do dataset).")
    p.add_argument("--entity", default=None,
                   help="Entidade WandB (user ou org). Deixe vazio para usar o padrão logado.")
    p.add_argument("--eer-ceiling", type=float, default=None,
                   help="Descarta runs com EER final acima deste valor. "
                        "Padrão: 0.35 (rvlcdip) / 0.05 (lacdip).")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K runs por grupo para o Fine Search (default: 5)")
    p.add_argument("--n-experiments", type=int, default=None,
                   help="Quantidade de experimentos por sweep Fine Search (default: 10)")
    p.add_argument("--analyze-only", action="store_true",
                   help="Apenas imprime o relatório, sem gerar YAMLs.")
    p.add_argument("--output-dir", default=None,
                   help="Diretório de saída dos YAMLs (sobrescreve o padrão do dataset).")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_cfg = DATASET_CONFIGS[args.dataset].copy()

    # Sobrescritas opcionais via CLI
    if args.project:
        dataset_cfg["wandb_project"] = args.project
    if args.eer_ceiling is not None:
        dataset_cfg["eer_ceiling"] = args.eer_ceiling
    if args.n_experiments is not None:
        dataset_cfg["n_experiments"] = args.n_experiments
    if args.output_dir:
        dataset_cfg["output_dir"] = Path(args.output_dir)

    output_dir  = dataset_cfg["output_dir"]
    eer_ceiling = dataset_cfg["eer_ceiling"]

    print("=" * 60)
    print(f"  CaVL-Doc — Fine Search | {dataset_cfg['display_name']}")
    print("=" * 60)

    # FASE 1
    df = fetch_runs(dataset_cfg["wandb_project"], args.entity)

    if df.empty:
        print("❌  Nenhum dado disponível. Verifique projeto e entity.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "runs_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"💾  Snapshot bruto salvo: {raw_csv}")

    # FASE 2
    finalists = prune(df, eer_ceiling, args.top_k)

    if not finalists:
        print("❌  Nenhum grupo finalista sobreviveu. Relaxe --eer-ceiling.")
        sys.exit(1)

    if args.analyze_only:
        print_report(df, finalists, [], output_dir, dataset_cfg)
        return

    # FASE 3
    print(f"\n{'='*60}")
    print(f"  FASE 3 — GERAÇÃO DOS YAMLs DE FINE SEARCH")
    print(f"{'='*60}")
    generated = generate_yamls(finalists, output_dir, dataset_cfg)

    generate_bash_script(finalists, generated, output_dir, dataset_cfg)
    print_report(df, finalists, generated, output_dir, dataset_cfg)


if __name__ == "__main__":
    main()
