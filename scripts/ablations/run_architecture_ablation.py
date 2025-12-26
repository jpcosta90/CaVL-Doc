#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob

# ==============================================================================
# CONFIGURAÇÃO: ABLATION STUDY - ARCHITECTURE CAPACITY (LA-CDIP)
# ==============================================================================
# Objetivo: Analisar o impacto do número de queries (Q) no Multi-Query Attention Pooling
# conforme Seção 4.4.3 do paper.

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")

# Dados
RAW_DATA_ROOT = "/mnt/data/la-cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Experimento
PROTOCOL = "zsl"
SPLITS_TO_RUN = [1, 2, 3, 4, 5]
LOSSES_TO_TEST = ["triplet", "cosface"] 

# === Variáveis de Ablação (Q) ===
# O paper explora Q=1 (Global), 4, 8, 16.
QUERIES_TO_TEST = [1, 8]

# Argumentos Fixos ("Ours" Configuration)
# Professor ligado, Batch 8
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
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "20",
]

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

def run_experiment(num_queries, loss_type, split_idx, pairs_csv):
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: Arch Ablation - Q={num_queries} | Loss={loss_type} | Split={split_idx}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    base_name = f"ABLATION_ARCH_Q{num_queries}_S{split_idx}_{loss_type}"
    wandb_project = "CaVL-Ablation-Architecture"
    
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
        "--num-queries", str(num_queries),
        "--loss-type", loss_type,
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
        print(f"\n✅ Experimento Q={num_queries} ({loss_type}) finalizado!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro em Q={num_queries} ({loss_type}). Exit code: {e.returncode}")

def main():
    print(f"Iniciando Ablation de Arquitetura (LA-CDIP)")
    print(f"Queries: {QUERIES_TO_TEST}")
    print(f"Losses: {LOSSES_TO_TEST}")
    print(f"Splits: {SPLITS_TO_RUN}")
    
    for split_idx in SPLITS_TO_RUN:
        print(f"\n>>> Processando Split {split_idx}...")
        generated_data_dir, pairs_csv = get_paths(split_idx)
        prepare_data(split_idx, generated_data_dir)
        
        for loss in LOSSES_TO_TEST:
            for q in QUERIES_TO_TEST:
                run_experiment(q, loss, split_idx, pairs_csv)
                time.sleep(5)

    print("\n🎉 Ablation study concluído!")

if __name__ == "__main__":
    main()
