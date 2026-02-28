#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob
import argparse

# ==============================================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS (RVL-CDIP - LARGE SAMPLE / BEST CONFIGS)
# ==============================================================================

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")

# Dados Originais (Read-Only)
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Configurações Geral
PROTOCOL = "zsl"
SPLITS_TO_RUN = [0, 1, 2, 3] # Todos os splits para validação cruzada completa

# Configurações de Amostra (Intermediária Otimizada)
# Objetivo: ~30 min/época (Meio termo entre 15min do Sweep e 1h do Full)
PAIRS_PER_CLASS = "50"       # Reduzido de 100 para 50
BATCH_SIZE = "8"             
GRAD_ACCUM = "4"             # Effective Batch = 32 (Melhor para Contrastive/Triplet)
POOL_SIZE = "32"             # Reduzido de 64 para 32 (Menos custo de mineração)
VAL_SAMPLES = "50"           # Ajustado para acompanhar treino

# Configurações Fixas
NUM_QUERIES = "1"            # Definido pelo usuário
WARMUP_STEPS = "99999"       # "Professor desligado" (Warmup > Total Steps)

# ==============================================================================
# 🧪 REGISTRO DE EXPERIMENTOS (BEST CONFIGS)
# ==============================================================================
# Uma única configuração vencedora por perda.

EXP_REGISTRY = {
    "contrastive": [
        # Baseline Siamese (Contrastive Loss)
        # Margin 1.0 é padrão. LR conservadora para estabilidade.
        {"margin": 1.0, "scale": 1.0, "lr": 3e-5, "optimizer": "adamw"}
    ],
    "triplet": [
        # Baseline Metric Learning (Triplet Loss)
        # Margin 0.3 é padrão (FaceNet).
        {"margin": 0.3, "scale": 1.0, "lr": 1e-5, "optimizer": "adamw"}
    ],
    "subcenter_arcface": [
        # ArcFace com Sub-centros (Lida bem com ruído/intra-class variations)
        # K=3 (3 centros por classe), Margin 0.5 (Padrão ArcFace)
        {"scale": 24.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "k": 2}
    ],
    "subcenter_cosface": [
        # CosFace com Sub-centros
        # K=2, Margin 0.35 (Padrão CosFace)
        {"scale": 24.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "k": 2}
    ],
    "circle": [
        # Circle Loss (Self-paced weighting)
        # Scale 256, Margin 0.25 (Padrão Circle)
        {"scale": 256.0, "margin": 0.25, "lr": 5e-5, "optimizer": "adamw"}
    ]
}

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx):
    # Usa sample size no nome do folder para evitar conflitos e cache incorreto
    folder_name = f"RVL-CDIP_{PROTOCOL}_split_{split_idx}_{PAIRS_PER_CLASS}pairs"
    generated_data_dir = os.path.join(WORKSPACE_ROOT, "data/generated_splits", folder_name)
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    
    # Nome do projeto no WandB
    wandb_project = f"CaVL-RVL-BestConfigs-S{split_idx}"
    return generated_data_dir, pairs_csv, wandb_project

def prepare_data(split_idx, generated_data_dir):
    print(f"\n{'='*60}")
    print(f"🛠️  Preparando Dados para Split {split_idx} - {PAIRS_PER_CLASS} pairs/class")
    print(f"{'='*60}")
    
    # Verifica se já existe, mas como PAIRS_PER_CLASS pode mudar, o ideal seria verificar ou sempre rodar/sobrescrever
    # O script de prep geralmente sobrescreve.
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL,
        "--pairs-per-class", PAIRS_PER_CLASS
    ]
    
    print(f"Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✅ Dados preparados com sucesso!\n")

def run_experiment(loss_type, config, split_idx, pairs_csv, wandb_project, dry_run=False):
    # Extrai configs
    lr = config.get("lr", 1e-4) # Fallback 1e-4
    optimizer = config.get("optimizer", "adamw")
    scale = config.get("scale", 64.0)
    margin = config.get("margin", 0.5)
    # WARMUP fixo e alto para desligar o professor (Random Sampling durante todo treino)
    warmup = WARMUP_STEPS 
    k = config.get("k", 3) # default subcenters
    
    # Constroi nome descritivo para o Run
    desc = f"Opt{optimizer}_S{scale}_M{margin}_LR{lr}_NoProf"
    if loss_type.startswith("subcenter"):
        desc += f"_K{k}"
        
    # run_name = f"{loss_type}_{desc}" # Nome longo para debug
    run_name = f"{loss_type}" # Nome curto se for rodar apenas um por tipo?
    # Melhor usar nome descritivo para garantir que configs diferentes (se houver) não colidam
    run_name = f"{loss_type}_{desc}"

    wandb_group = f"{loss_type}_CrossVal"

    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: {run_name} (Split {split_idx})")
    print(f"📂 Group: {wandb_group}")
    print(f"{'='*60}")

    # Checagem de Checkpoint
    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    ckpt_dir_name = f"RVL_S{split_idx}_{run_name}"
    full_ckpt_path = os.path.join(base_ckpt_path, ckpt_dir_name, "last_checkpoint.pt")
    
    resume_args = []
    if os.path.exists(full_ckpt_path):
        print(f"🔄 Checkpoint encontrado: {full_ckpt_path}")
        resume_args = ["--resume-from", full_ckpt_path]
    
    # Monta comando
    cmd = [
        sys.executable, SCRIPT_PATH,
        # Identificadores WandB
        "--use-wandb",
        "--wandb-project", wandb_project,
        "--wandb-run-name", run_name,
        # "--wandb-group", wandb_group, # Passar via ENV se o script nao aceitar

        # Dataset & Modelo
        "--model-name", "InternVL3-2B",
        "--dataset-name", "RVL-CDIP",
        "--base-image-dir", BASE_IMAGE_DIR,
        "--pairs-csv", pairs_csv,

        # Configurações Específicas da Loss (Variáveis)
        "--loss-type", loss_type,
        "--optimizer-type", optimizer,
        "--student-lr", str(lr),
        "--scheduler-type", "constant", 
        "--scale", str(scale),
        "--margin", str(margin),
        "--num-sub-centers", str(k),
        
        # Curriculum / Warmup ("Desligado")
        "--professor-warmup-steps", str(warmup),
        "--easy-mining-steps", str(warmup),
        
        # Configurações Fixas de Amostra (Grandes)
        "--epochs", "5",
        "--student-batch-size", BATCH_SIZE,
        "--gradient-accumulation-steps", GRAD_ACCUM,
        "--candidate-pool-size", POOL_SIZE,
        # "--max-steps-per-epoch", "2500", # Removido para usar tamanho natural do dataset
        
        # Outros fixos
        "--patience", "3",
        "--lr-reduce-factor", "0.5",
        "--professor-lr", "0.0", # Professor LR zerado para garantir
        "--cut-layer", "27",
        "--projection-output-dim", "1536",
        "--max-num-image-tokens", "12",
        "--num-queries", NUM_QUERIES, # Definido pelo usuário (1)
        "--pooler-type", "attention",
        "--head-type", "mlp",
        "--baseline-alpha", "0.05",
        "--entropy-coeff", "0.01",
        "--val-samples-per-class", VAL_SAMPLES
    ] + resume_args

    # Environment adjustments
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
    parser = argparse.ArgumentParser(description="CaVL RVL-CDIP Best Configs Runner")
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime os comandos")
    parser.add_argument("--loss", type=str, help="Filtrar uma loss específica para rodar (opcional)")
    args = parser.parse_args()

    print(f"Iniciando Bateria de Melhores Configurações RVL-CDIP")
    
    for split_idx in SPLITS_TO_RUN:
        generated_data_dir, pairs_csv, wandb_project = get_paths(split_idx)
        
        # Prepara os dados (garante amostra de 100 pares/classe)
        if not args.dry_run:
            prepare_data(split_idx, generated_data_dir)
        
        # Itera sobre as losses definidas no registro
        for loss_name, configs in EXP_REGISTRY.items():
            if args.loss and args.loss != loss_name:
                continue

            for conf in configs:
                run_experiment(loss_name, conf, split_idx, pairs_csv, wandb_project, dry_run=args.dry_run)

    print("\n🎉 Fim dos experimentos.")

if __name__ == "__main__":
    main()
