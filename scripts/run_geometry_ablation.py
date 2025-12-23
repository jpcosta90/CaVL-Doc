#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob

# ==============================================================================
# CONFIGURAÇÃO: ABLATION STUDY - GEOMETRY (LA-CDIP)
# ==============================================================================
# Objetivo: Analisar restrições geométricas (Margem e Elasticidade) para CosFace.

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/prepare_splits.py")

# Dados
RAW_DATA_ROOT = "/mnt/data/la-cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Experimento
PROTOCOL = "zsl"
SPLITS_TO_RUN = [1, 2, 3, 4, 5]

# BASE CONFIGURATION ("Ours" default for other params)
COMMON_ARGS = [
    "--model-name", "InternVL3-2B",
    "--dataset-name", "LA-CDIP",
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
    "--num-queries", "4",               # Fixado Q=4
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "20",
]

# === ABLATION STAGES (Sequential) ===
# O User sugeriu uma abordagem sequencial:
# 1. Testar Margins -> Achar Best M
# 2. Testar SubCenters (fixando Best M) -> Achar Best K
# 3. Testar Elasticity (fixando Best M e K=1) -> Comparar com Static

# Variáveis "Best" (A serem atualizadas conforme resultados da etapa anterior)
# Por padrão, assumimos valores centrais para rodar tudo de uma vez se desejado,
# ou o usuário pode comentar as fases.
BEST_M_DEFAULT = 0.45 

RUN_STAGES = [1, 2, 3] # 1=Margin, 2=SubCenter, 3=Elasticity

# Stage 1: Margins
MARGINS_TO_TEST = [0.35, 0.45, 0.55]

# Stage 2: SubCenters (Uses BEST_M_DEFAULT)
SUBCENTERS_TO_TEST = [1, 3, 5] # k=1 é redundante com etapa 1, mas serve de baseline.
# Nota: Implementação usa subcenter_cosface

# Stage 3: Elasticity (Disabled for now)
# ELASTIC_SIGMA_LIST = [0.0125, 0.05, 0.1]

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx):
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/{PROTOCOL}_split_{split_idx}")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    return generated_data_dir, pairs_csv

def prepare_data(split_idx, generated_data_dir):
    if os.path.exists(os.path.join(generated_data_dir, "train_pairs.csv")):
        print(f"✅ Dados já existem em: {generated_data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🛠️  Preparando Dados para Split {split_idx}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL,
        "--pairs-per-class", "10"
    ]
    subprocess.run(cmd, check=True)

def run_experiment(loss_name, margin, subcenters, split_idx, pairs_csv):
    # Identificação
    mode_str = "Static"
    if subcenters > 1: mode_str += f" (k={subcenters})"
    
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: {loss_name} | {mode_str} | m={margin} | Split={split_idx}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    # Nome único (Elasticity removido)
    base_name = f"ABLATION_GEO_{loss_name}_m{margin}_k{subcenters}_S{split_idx}"
        
    wandb_project = "CaVL-Ablation-Geometry"
    
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
    ] + COMMON_ARGS
    
    if resume_path:
        cmd += ["--resume-from", resume_path]

    print(f"Comando: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Finalizado!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro. Exit code: {e.returncode}")

def main():
    print(f"Iniciando Ablation Sequencial de Geometria (LA-CDIP)")
    print(f"Splits: {SPLITS_TO_RUN}")
    
    for split_idx in SPLITS_TO_RUN:
        print(f"\n>>> Processando Split {split_idx}...")
        generated_data_dir, pairs_csv = get_paths(split_idx)
        prepare_data(split_idx, generated_data_dir)
        
        # --- Stage 1: Find Best Margin ---
        if 1 in RUN_STAGES:
            print(f"\n👉 STAGE 1: Testing Margins {MARGINS_TO_TEST} (CosFace, k=1)")
            for m in MARGINS_TO_TEST:
                run_experiment("cosface", m, 1, split_idx, pairs_csv)
                time.sleep(5)
        
        # --- Stage 2: Find Best Subcenters (using BEST_M_DEFAULT) ---
        if 2 in RUN_STAGES:
            print(f"\n👉 STAGE 2: Testing SubCenters {SUBCENTERS_TO_TEST} (SubCenterCosFace, m={BEST_M_DEFAULT})")
            for k in SUBCENTERS_TO_TEST:
                loss = "subcenter_cosface" if k > 1 else "cosface"
                run_experiment(loss, BEST_M_DEFAULT, k, split_idx, pairs_csv)
                time.sleep(5)
                
        # --- Stage 3: Elasticity (using BEST_M_DEFAULT, k=1, Varying Sigma) ---
        # if 3 in RUN_STAGES:
        #     print(f"\n👉 STAGE 3: Testing Elasticity (ElasticCosFace, m={BEST_M_DEFAULT}, Sigmas={ELASTIC_SIGMA_LIST})")
        #     for std in ELASTIC_SIGMA_LIST:
        #         run_experiment("elastic_cosface", BEST_M_DEFAULT, 1, True, split_idx, pairs_csv, std=std)
        #         time.sleep(5)

    print("\n🎉 Ablation study concluído!")

if __name__ == "__main__":
    main()
