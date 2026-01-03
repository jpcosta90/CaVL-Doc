#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob
import argparse
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO: ABLATION STUDY - GEOMETRY (RVL-CDIP)
# ==============================================================================
# Objetivo: Analisar restrições geométricas (Margem, Subcentro, Elasticidade).

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")

# Dados
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Experimento
PROTOCOL = "zsl"
SPLITS_TO_RUN = [0, 1, 2, 3]

# BASE CONFIGURATION ("Ours" default for other params)
COMMON_ARGS = [
    "--model-name", "InternVL3-2B",
    "--dataset-name", "RVL-CDIP",
    "--epochs", "5",
    "--student-batch-size", "8",
    "--candidate-pool-size", "64",     # Professor ativado
    "--professor-lr", "1e-4",
    "--use-wandb",
    "--patience", "3",
    "--lr-reduce-factor", "0.5",
    "--student-lr", "1e-5",
    "--cut-layer", "27",
    "--projection-output-dim", "1536",
    "--max-num-image-tokens", "12",
    "--pooler-type", "attention",
    # "--num-queries", "4",  <-- REMOVIDO DA LISTA ESTATICA, SERÁ INJETADO DINAMICAMENTE
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "100",   # RVL Specific
]

# === ABLATION SETTINGS ===
# Updated per user request: [0.15, 0.75] (0.45 is skipped/reused from Architecture Ablation)
MARGINS_TO_TEST = [0.15, 0.75] 
SUBCENTERS_TO_TEST = [1, 3, 5] 
ELASTIC_SIGMA_LIST = [0.0125, 0.05, 0.1]

# ==============================================================================
# AUTO-CONFIGURAÇÃO
# ==============================================================================
def get_auto_config():
    """Lê os CSVs de resultados para encontrar os melhores parâmetros anteriores."""
    config = {
        "queries": 4,   # Default
        "margin": 0.45, # Default
        "k": 1          # Default
    }
    
    # 1. Architecture (Best Q)
    arch_csv = os.path.join(WORKSPACE_ROOT, "results/RVL-CDIP_architecture_ablation_eval.csv")
    if os.path.exists(arch_csv):
        try:
            df = pd.read_csv(arch_csv)
            # Filter for Triplet (since geometry focuses on metric learning properties often common, 
            # or usually we stick to the Loss used. But Architecture Ablation used Triplet/Cosface.
            # Best Q is structural choice. Let's take global best Q across all losses or for CosFace (target here).
            # Usually strict consistency: We use CosFace in Geometry.
            df_cos = df[df['loss'] == 'cosface']
            if not df_cos.empty:
                best_q = df_cos.groupby('queries')['eer'].mean().idxmin()
                config['queries'] = int(best_q)
                print(f"✅ [Auto] Best Q (CosFace): {config['queries']}")
            else:
                # Fallback to global best
                best_q = df.groupby('queries')['eer'].mean().idxmin()
                config['queries'] = int(best_q)
                print(f"✅ [Auto] Best Q (Global): {config['queries']}")
        except Exception as e:
            print(f"⚠️ [Auto] Erro ao ler Q: {e}")
    
    # 2. Geometry (Best M and K)
    geo_csv = os.path.join(WORKSPACE_ROOT, "results/RVL-CDIP_geometry_ablation_eval.csv")
    if os.path.exists(geo_csv):
        try:
            df = pd.read_csv(geo_csv)
            
            # --- Best Margin ---
            m_df = df[df['name'] == 'margin_variation']
            if not m_df.empty:
                best_m = m_df.groupby('margin')['eer'].mean().idxmin()
                config['margin'] = float(best_m)
                print(f"✅ [Auto] Best Margin: {config['margin']}")
            
            # --- Best K (SubCenters) ---
            # Tenta filtrar pelo melhor margin atual
            k_df = df[(df['name'] == 'subcenter_variation') & (df['margin'] == config['margin'])]
            if k_df.empty:
                 k_df = df[df['name'] == 'subcenter_variation'] # Fallback unconstrained
            
            if not k_df.empty:
                best_k = k_df.groupby('k')['eer'].mean().idxmin()
                config['k'] = int(best_k)
                print(f"✅ [Auto] Best K: {config['k']}")
                
        except Exception as e:
             print(f"⚠️ [Auto] Erro ao ler Geometry params: {e}")
             
    return config

# Carrega configuração automática globalmente para uso no COMMON_ARGS se necessário
# Mas COMMON_ARGS é estático. Melhor injetar na chamada do subprocesso.
AUTO_CONFIG = get_auto_config()

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx):
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL-CDIP_{PROTOCOL}_split_{split_idx}")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    return generated_data_dir, pairs_csv

def prepare_data(split_idx, generated_data_dir):
    if os.path.exists(os.path.join(generated_data_dir, "train_pairs.csv")):
        print(f"✅ Dados já existem em: {generated_data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🛠️  Preparando Dados para Split {split_idx} (RVL-CDIP)")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL,
        "--pairs-per-class", "100"
    ]
    subprocess.run(cmd, check=True)

def run_experiment(loss_name, margin, subcenters, split_idx, pairs_csv, std=None):
    # Identificação
    mode_str = "Static"
    if subcenters > 1: mode_str += f" (k={subcenters})"
    if std is not None: mode_str = f"Elastic (std={std})"
    
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: {loss_name} | {mode_str} | m={margin} | Split={split_idx}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    # Naming Convention
    if std is not None:
        base_name = f"RVL_ABLATION_GEO_{loss_name}_m{margin}_k{subcenters}_std{std}_S{split_idx}"
    else:
        base_name = f"RVL_ABLATION_GEO_{loss_name}_m{margin}_k{subcenters}_S{split_idx}"
        
    wandb_project = "CaVL-Ablation-Geometry-RVL"
    
    # Resume Logic
    search_pattern = os.path.join(base_ckpt_path, f"{base_name}_*")
    existing_runs = sorted(glob.glob(search_pattern))
    resume_path = None
    run_name = None
    
    if existing_runs:
        latest_run = existing_runs[-1]
        ckpt_path = os.path.join(latest_run, "last_checkpoint.pt")
        if os.path.exists(ckpt_path):
            print(f"🔄 Retomando de: {ckpt_path}")
            resume_path = ckpt_path
            run_name = os.path.basename(latest_run)
    
    if not run_name:
        timestamp = time.strftime("%Y%m%d-%H%M")
        run_name = f"{base_name}_{timestamp}"

    wandb_id = run_name.replace("/", "-").replace(":", "-")

    cmd = [
        sys.executable, SCRIPT_PATH,
        "--loss-type", loss_name,
        "--margin", str(margin),
        "--num-sub-centers", str(subcenters),
        "--pairs-csv", pairs_csv,
        "--base-image-dir", BASE_IMAGE_DIR,
        "--wandb-project", wandb_project,
        "--wandb-run-name", run_name,
        "--wandb-id", wandb_id,
        "--num-queries", str(AUTO_CONFIG['queries']), # Injeta o Q detectado
    ] + COMMON_ARGS
    
    if std is not None:
        # Passar argumento std para o training script
        # Assumindo que run_cavl_training.py aceita --std (verificado: aceita)
        # O nome do argumento deve bater com o parse_args (verificado: --std)
        cmd += ["--std", str(std)]
    
    if resume_path:
        cmd += ["--resume-from", resume_path]

    print(f"Comando: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Finalizado!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro. Exit code: {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Script de Ablação de Geometria (RVL-CDIP)")
    parser.add_argument("--stage", type=str, required=True, choices=["margin", "subcenter", "elasticity"],
                        help="Qual estágio da ablação rodar")
    parser.add_argument("--best-margin", type=float, default=None,
                        help="Sobrescreve a melhor margem (se não informado, usa auto-detect/default)")
    parser.add_argument("--best-k", type=int, default=None,
                        help="Sobrescreve o melhor K (se não informado, usa auto-detect/default)")
    args = parser.parse_args()

    # Prioridade: CLI > Auto-Config > Defaults (já embutido no Auto-Config)
    eff_margin = args.best_margin if args.best_margin is not None else AUTO_CONFIG['margin']
    eff_k = args.best_k if args.best_k is not None else AUTO_CONFIG['k']

    print(f"\n⚙️  CONFIGURAÇÃO EFETIVA:")
    print(f"   - Queries (Q): {AUTO_CONFIG['queries']} (Automático)")
    print(f"   - Margin (m):  {eff_margin} {'(CLI)' if args.best_margin else '(Auto)'}")
    print(f"   - SubCenters (k): {eff_k} {'(CLI)' if args.best_k else '(Auto)'}")

    print(f"\nIniciando Ablation de Geometria (RVL-CDIP) - Stage: {args.stage}")
    print(f"Splits: {SPLITS_TO_RUN}")
    
    for split_idx in SPLITS_TO_RUN:
        print(f"\n>>> Processando Split {split_idx}...")
        generated_data_dir, pairs_csv = get_paths(split_idx)
        prepare_data(split_idx, generated_data_dir)
        
        # --- Stage 1: Margins ---
        if args.stage == "margin":
            print(f"\n👉 STAGE 1: Testing Margins {MARGINS_TO_TEST} (CosFace, k=1)")
            for m in MARGINS_TO_TEST:
                run_experiment("cosface", m, 1, split_idx, pairs_csv)
                time.sleep(2)
        
        # --- Stage 2: SubCenters ---
        elif args.stage == "subcenter":
            print(f"\n👉 STAGE 2: Testing SubCenters {SUBCENTERS_TO_TEST} (SubCenterCosFace, m={eff_margin})")
            for k in SUBCENTERS_TO_TEST:
                # Se k=1, tecnicamente é um CosFace normal.
                # Se k>1, usamos subcenter_cosface
                loss_name = "subcenter_cosface" if k > 1 else "cosface"
                run_experiment(loss_name, eff_margin, k, split_idx, pairs_csv)
                time.sleep(2)
                
        # --- Stage 3: Elasticity ---
        elif args.stage == "elasticity":
            print(f"\n👉 STAGE 3: Testing Elasticity (ElasticCosFace, m={eff_margin}, k={eff_k}, Sigmas={ELASTIC_SIGMA_LIST})")
            # Assumimos que a elasticidade é aplicada sobre a melhor configuração K.
            # Forçamos K=1 se o loss for elastic_cosface puro, ou adaptamos.
            # Por consistência com o paper: ElasticCosFace -> K=1.
            loss_name = "elastic_cosface"
            k_eff_run = 1 
            
            for std in ELASTIC_SIGMA_LIST:
                run_experiment(loss_name, eff_margin, k_eff_run, split_idx, pairs_csv, std=std)
                time.sleep(2)

    print("\n🎉 Stage concluído!")

if __name__ == "__main__":
    main()
