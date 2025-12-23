#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob

# ==============================================================================
# CONFIGURAÇÃO: ABLATION STUDY - SAMPLING STRATEGY (RVL-CDIP)
# ==============================================================================
# Objetivo: Comparar "Random Sampling" (Baseline) vs "Professor Agent" (Ours)
# no dataset RVL-CDIP.

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/prepare_splits.py")

# Dados Originais (RVL-CDIP)
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 

# Configuração do Experimento
PROTOCOL = "zsl"
SPLITS_TO_RUN = [0, 1, 2, 3]  # Todos os splits do RVL-CDIP (CICA)
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# === 1. Escolha das Losses ===
LOSSES_TO_TEST = ["triplet", "cosface"]

# === 2. Definição das Estratégias ===
# Batch Size fixo para o Student
STUDENT_BATCH_SIZE = 8

STRATEGIES = {
    "random_sampling": {
        "description": "Baseline: Sem seleção ativa (Pool = Batch Size)",
        "args": [
            "--professor-lr", "0.0",                 # Desliga o treino do Professor
            "--candidate-pool-size", str(STUDENT_BATCH_SIZE) # Pool == Batch -> Seleção aleatória/sequencial do DataLoader
        ]
    }
    # "professor_agent": JÁ REALIZADO NO SCRIPT run_loss_comparison_rvl.py
}

# Argumentos Comuns (Baseado em run_loss_comparison_rvl.py)
COMMON_ARGS = [
    "--model-name", "InternVL3-2B",
    "--dataset-name", "RVL-CDIP",
    "--epochs", "5",
    "--student-batch-size", str(STUDENT_BATCH_SIZE),
    "--use-wandb",
    "--patience", "3",
    "--lr-reduce-factor", "0.5",
    "--student-lr", "1e-5",
    "--cut-layer", "27",
    "--projection-output-dim", "1536",
    "--max-num-image-tokens", "12",
    "--num-queries", "4",
    "--pooler-type", "attention",
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "100" # Específico do RVL
]

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
    print(f"🛠️  Preparando Dados para Split {split_idx} ({PROTOCOL.upper()}) - RVL-CDIP")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL,
        "--pairs-per-class", "100" # Específico do RVL
    ]
    
    subprocess.run(cmd, check=True)

def run_strategy(strategy_name, split_idx, pairs_csv, loss_type):
    config = STRATEGIES[strategy_name]
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: {strategy_name.upper()} - Loss: {loss_type.upper()} (RVL-CDIP)")
    print(f"Descrição: {config['description']}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    # Nome identificador único
    base_name = f"RVL_ABLATION_SAMPLING_{strategy_name}_S{split_idx}_{loss_type}"
    wandb_project = f"CaVL-Ablation-Sampling-RVL"
    
    # Busca runs anteriores para resume
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

    # IDs do WandB
    wandb_id = run_name.replace("/", "-").replace(":", "-")

    cmd = [
        sys.executable, SCRIPT_PATH,
        "--loss-type", loss_type,
        "--pairs-csv", pairs_csv,
        "--base-image-dir", BASE_IMAGE_DIR,
        "--wandb-project", wandb_project,
        "--wandb-run-name", run_name,
        "--wandb-id", wandb_id,
    ] + COMMON_ARGS + config["args"]
    
    if resume_path:
        cmd += ["--resume-from", resume_path]

    print(f"Comando: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Estratégia {strategy_name} ({loss_type}) finalizada com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro em {strategy_name} ({loss_type}). Exit code: {e.returncode}")

def main():
    print(f"Iniciando Ablation de Sampling RVL-CDIP: {list(STRATEGIES.keys())}")
    print(f"Losses: {LOSSES_TO_TEST}")
    print(f"Splits: {SPLITS_TO_RUN}")
    
    for split_idx in SPLITS_TO_RUN:
        print(f"\n>>> Processando Split {split_idx}...")
        generated_data_dir, pairs_csv = get_paths(split_idx)
        prepare_data(split_idx, generated_data_dir)
        
        # Apenas Random Sampling (Professor Agent nos scripts principais)
        for loss in LOSSES_TO_TEST:
            for strategy in STRATEGIES.keys():
                run_strategy(strategy, split_idx, pairs_csv, loss)
                time.sleep(5)

    print("\n🎉 Ablation study RVL-CDIP concluído!")

if __name__ == "__main__":
    main()
