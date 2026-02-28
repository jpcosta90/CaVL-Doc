#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob
import argparse

# ==============================================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS (LA-CDIP - LARGE SAMPLE / BEST CONFIGS)
# ==============================================================================

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")

# Dados Originais (Read-Only)
RAW_DATA_ROOT = "/mnt/data/la-cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Configurações Geral
PROTOCOL = "zsl"             # LA-CDIP utiliza protocolo ZSL por padrão
SPLITS_TO_RUN = [1, 2, 3, 4, 5] # LA-CDIP possui 5 splits oficiais

# Configurações de Amostra (Intermediária Otimizada)
# Sweep La-CDIP = 2 pares | Ablation Padrão = 10 pares
# Intermediário = 5 pares (~50% do full, similar à lógica do RVL)
PAIRS_PER_CLASS = "5"       
BATCH_SIZE = "8"             
GRAD_ACCUM = "4"             # Effective Batch = 32
POOL_SIZE = "32"             
VAL_SAMPLES = "20"           # Mesma quantidade do Sweep (Padrão)

# Configurações Fixas
NUM_QUERIES = "1"            
WARMUP_STEPS = "99999"       # "Professor desligado"

# ==============================================================================
# 🧪 REGISTRO DE EXPERIMENTOS (BEST CONFIGS)
# ==============================================================================

EXP_REGISTRY = {
    "contrastive": [
        {"margin": 1.0, "scale": 1.0, "lr": 3e-5, "optimizer": "adamw"}
    ],
    "triplet": [
        {"margin": 0.3, "scale": 1.0, "lr": 1e-5, "optimizer": "adamw"}
    ],
    "subcenter_arcface": [
        {"scale": 24.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "k": 2}
    ],
    "subcenter_cosface": [
        {"scale": 32.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "k": 2}
    ],
    "circle": [
        {"scale": 256.0, "margin": 0.25, "lr": 5e-5, "optimizer": "adamw"}
    ]
}

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx):
    folder_name = f"LA-CDIP_{PROTOCOL}_split_{split_idx}_{PAIRS_PER_CLASS}pairs"
    generated_data_dir = os.path.join(WORKSPACE_ROOT, "data/generated_splits", folder_name)
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    
    # Nome do projeto no WandB (Distinto do RVL)
    wandb_project = f"CaVL-LACDIP-BestConfigs-S{split_idx}"
    return generated_data_dir, pairs_csv, wandb_project

def prepare_data(split_idx, generated_data_dir):
    print(f"\n{'='*60}")
    print(f"🛠️  Preparando Dados para Split {split_idx} (LA-CDIP) - {PAIRS_PER_CLASS} pairs/class")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL, # Importante: flag protocol muda para la-cdip
        "--pairs-per-class", PAIRS_PER_CLASS
    ]
    
    print(f"Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✅ Dados preparados com sucesso!\n")

def run_experiment(loss_type, config, split_idx, pairs_csv, wandb_project, dry_run=False):
    lr = config.get("lr", 1e-4) 
    optimizer = config.get("optimizer", "adamw")
    scale = config.get("scale", 64.0)
    margin = config.get("margin", 0.5)
    warmup = WARMUP_STEPS 
    k = config.get("k", 3) 
    
    desc = f"Opt{optimizer}_S{scale}_M{margin}_LR{lr}_NoProf"
    if loss_type.startswith("subcenter"):
        desc += f"_K{k}"
        
    run_name = f"{loss_type}_{desc}"
    wandb_group = f"{loss_type}_CrossVal"

    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: {run_name} (Split {split_idx})")
    print(f"📂 Group: {wandb_group}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    # Folder distinto para LA-CDIP
    ckpt_dir_name = f"LACDIP_S{split_idx}_{run_name}"
    full_ckpt_path = os.path.join(base_ckpt_path, ckpt_dir_name, "last_checkpoint.pt")
    
    resume_args = []
    if os.path.exists(full_ckpt_path):
        print(f"🔄 Checkpoint encontrado: {full_ckpt_path}")
        resume_args = ["--resume-from", full_ckpt_path]
    
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--use-wandb",
        "--wandb-project", wandb_project,
        "--wandb-run-name", run_name,
        "--model-name", "InternVL3-2B",
        "--dataset-name", "LA-CDIP", # Nome do dataset no log
        "--base-image-dir", BASE_IMAGE_DIR,
        "--pairs-csv", pairs_csv,
        "--loss-type", loss_type,
        "--optimizer-type", optimizer,
        "--student-lr", str(lr),
        "--scheduler-type", "constant", 
        "--scale", str(scale),
        "--margin", str(margin),
        "--num-sub-centers", str(k),
        "--professor-warmup-steps", str(warmup),
        "--easy-mining-steps", str(warmup),
        "--epochs", "5",
        "--student-batch-size", BATCH_SIZE,
        "--gradient-accumulation-steps", GRAD_ACCUM,
        "--candidate-pool-size", POOL_SIZE,
        "--patience", "3",
        "--lr-reduce-factor", "0.5",
        "--professor-lr", "0.0", 
        "--cut-layer", "27",
        "--projection-output-dim", "1536",
        "--max-num-image-tokens", "12",
        "--num-queries", NUM_QUERIES, 
        "--pooler-type", "attention",
        "--head-type", "mlp",
        "--baseline-alpha", "0.05",
        "--entropy-coeff", "0.01",
        "--val-samples-per-class", VAL_SAMPLES
    ] + resume_args

    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = wandb_group

    if dry_run:
        print(f"CMD: {' '.join(cmd)}")
        return

    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ Sucesso: {run_name}")
    except subprocess.CalledProcessError:
        print(f"❌ Falha: {run_name}")
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")
        sys.exit(1)
    
    time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="CaVL LA-CDIP Best Configs Runner")
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime os comandos")
    parser.add_argument("--loss", type=str, help="Filtrar uma loss específica para rodar (opcional)")
    args = parser.parse_args()

    print(f"Iniciando Bateria de Melhores Configurações LA-CDIP")
    
    for split_idx in SPLITS_TO_RUN:
        generated_data_dir, pairs_csv, wandb_project = get_paths(split_idx)
        
        if not args.dry_run:
            prepare_data(split_idx, generated_data_dir)
        
        for loss_name, configs in EXP_REGISTRY.items():
            if args.loss and args.loss != loss_name:
                continue

            for conf in configs:
                run_experiment(loss_name, conf, split_idx, pairs_csv, wandb_project, dry_run=args.dry_run)

    print("\n🎉 Fim dos experimentos.")

if __name__ == "__main__":
    main()
