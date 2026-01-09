#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob

# ==============================================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS (RVL-CDIP) - REFINED
# ==============================================================================

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")

# Dados Originais (Read-Only)
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 

# Configuração Geral
PROTOCOL = "zsl"
SPLITS_TO_RUN = [0, 1] # Ordem de prioridade ,2, 3
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# ==============================================================================
# CONFIGURAÇÃO ESPECÍFICA POR LOSS (BIBLIOGRAPHIC PRIORS)
# ==============================================================================
# Adaptação de hiperparâmetros baseados em literatura comum de Metric Learning.
# Nota: LR aqui é o inicial. Schedulers ajustam durante o treino.

loss_configs = {
    "contrastive": {
        "description": "Baseline Contrastive Loss",
        "args": [
            "--student-lr", "3e-4", # Ajustado para padrão AdamW
            "--optimizer-type", "adamw",
            "--scheduler-type", "cosine",
            "--margin", "0.5",
        ]
    },
    "triplet": {
        "description": "Triplet Loss (Metric Learning Baseline)",
        "args": [
            "--student-lr", "3e-4", # Ajustado para padrão AdamW
            "--optimizer-type", "adamw",
            "--scheduler-type", "cosine",
            "--margin", "0.5",
        ]
    },
    "arcface": {
        "description": "ArcFace (Additive Angular Margin)",
        "args": [
            "--student-lr", "0.1", # Ajustado para AdamW (0.1 seria muito alto)
            "--optimizer-type", "adamw",
            "--scheduler-type", "cosine",
            "--margin", "0.5",
            "--scale", "64.0",
        ]
    },
    "cosface": {
        "description": "CosFace (Lage Margin Cosine Loss)",
        "args": [
            "--student-lr", "3e-4", # Ajustado para AdamW (0.1 seria muito alto)
            "--optimizer-type", "adamw",
            "--scheduler-type", "cosine",
            "--margin", "0.35",
            "--scale", "64.0",
        ]
    },
    "circle": {
        "description": "Circle Loss",
        "args": [
            "--student-lr", "3e-4", # Ajustado para padrão AdamW
            "--optimizer-type", "adamw",
            "--scheduler-type", "cosine",
            "--margin", "0.25",
            "--scale", "256.0",
        ]
    },
}

LOSSES_TO_TEST = ["arcface", "cosface", "contrastive", "triplet"]

# Argumentos Comuns (Base) - Serão sobrescritos pelos específicos se houver conflito
# Mas como passamos os específicos depois na lista de cmd, eles ganham precedência (dependendo do argparse implementation
# mas geralmente o ultimo vence ou cria lista. No run_cavl_training usamos argparse padrão).
# IMPORTANTE: Definir aqui o que é fixo para TODOS.
COMMON_ARGS = [
    "--model-name", "InternVL3-2B",
    "--dataset-name", "RVL-CDIP",
    "--epochs", "5",
    "--student-batch-size", "8",
    "--candidate-pool-size", "64",
    "--use-wandb",
    "--patience", "3",
    # Professor Params (Geralmente fixos ou menos sensíveis)
    "--professor-lr", "1e-4",
    "--cut-layer", "27",
    "--projection-output-dim", "1536",
    "--max-num-image-tokens", "12",
    "--num-queries", "4",
    "--pooler-type", "attention",
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "100"
]

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx):
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL-CDIP_{PROTOCOL}_split_{split_idx}")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    wandb_project = f"CaVL-Refined-RVL-{PROTOCOL.upper()}-S{split_idx}"
    return generated_data_dir, pairs_csv, wandb_project

def prepare_data(split_idx, generated_data_dir):
    if os.path.exists(os.path.join(generated_data_dir, "train_pairs.csv")):
        print(f"✅ Dados já preparados para Split {split_idx}")
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
        "--pairs-per-class", "100" 
    ]
    
    subprocess.run(cmd, check=True)
    print("✅ Dados preparados com sucesso!\n")

def run_experiment(loss_type, split_idx, pairs_csv, wandb_project, force_restart=False):
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando Experimento: {loss_type.upper()} (Split {split_idx})")
    if loss_type in loss_configs:
        print(f"ℹ️  Config: {loss_configs[loss_type]['description']}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    base_name = f"RVL_Refined_{PROTOCOL}_S{split_idx}_{loss_type}"
    
    # Busca run anterior para resume
    search_pattern = os.path.join(base_ckpt_path, f"{base_name}_*")
    existing_runs = sorted(glob.glob(search_pattern))
    
    resume_path = None
    run_name = None
    
    if existing_runs and not force_restart:
        latest_run = existing_runs[-1]
        ckpt_path = os.path.join(latest_run, "last_checkpoint.pt")
        if os.path.exists(ckpt_path):
            print(f"⚠️  Encontrado run anterior: {os.path.basename(latest_run)}")
            print(f"🔄 Retomando de: {ckpt_path}")
            resume_path = ckpt_path
            run_name = os.path.basename(latest_run)
    elif force_restart and existing_runs:
        print(f"⚠️  Runs anteriores encontrados, mas ignorados devido a --force-restart.")
    
    if not run_name:
        timestamp = time.strftime("%Y%m%d-%H%M")
        run_name = f"{base_name}_{timestamp}"

    # Argumentos específicos da Loss
    loss_specific_args = loss_configs.get(loss_type, {}).get("args", [])
    if not loss_specific_args:
        print(f"⚠️  AVISO: Nenhuma configuração específica encontrada para {loss_type}. Usando defaults.")

    cmd = [
        sys.executable, SCRIPT_PATH,
        "--loss-type", loss_type,
        "--pairs-csv", pairs_csv,
        "--base-image-dir", BASE_IMAGE_DIR,
        "--wandb-project", wandb_project,
        "--wandb-run-name", run_name,
        "--wandb-id", run_name.replace("/", "-").replace(":", "-"), # ID consistente
    ] + COMMON_ARGS + loss_specific_args
    
    if resume_path:
        cmd += ["--resume-from", resume_path]

    print(f"Comando:\n{' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Experimento {loss_type} finalizado!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao executar {loss_type}. Exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupção manual.")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Loss Comparison (Refined)")
    parser.add_argument("--force-restart", action="store_true", help="Ignora checkpoints existentes e começa do zero.")
    parser.add_argument("--splits", type=int, nargs="+", default=SPLITS_TO_RUN, help="Lista de splits para rodar (ex: 0 1)")
    args = parser.parse_args()

    print(f"Iniciando Bateria Refinada RVL-CDIP (Restart: {args.force_restart}, Splits: {args.splits})")
    
    for split_idx in args.splits:
        generated_data_dir, pairs_csv, wandb_project = get_paths(split_idx)
        prepare_data(split_idx, generated_data_dir)
        
        for loss in LOSSES_TO_TEST:
            run_experiment(loss, split_idx, pairs_csv, wandb_project, force_restart=args.force_restart)
            time.sleep(5)

    print("\n🎉 Todos os experimentos concluídos!")

if __name__ == "__main__":
    main()
