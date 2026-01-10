#!/usr/bin/env python3
import subprocess
import sys
import os
import time

# ==============================================================================
# ⚙️ CONFIGURAÇÃO GERAL
# ==============================================================================

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")

# ⚠️ ATENÇÃO: Se você moveu os dados devido ao erro de I/O, altere aqui para "~/dataset_local/..."
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip" 
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR

# Configuração Fixa
PROTOCOL = "zsl"
SPLIT_IDX = 1  
LOSS_TYPE = "arcface"
WANDB_PROJECT = "CaVL-Calibration-Protocol" # V2 para diferenciar

# ==============================================================================
# 🧪 A GRADE DE EXPERIMENTOS (SWEEP GRID COM LR)
# ==============================================================================

experiments = [
    # --- GRUPO 1: O "Golden Standard" (Teoricamente o melhor) ---
    {
        "desc": "Golden_S64_M0.5_LR3e-4",
        "optimizer": "adamw",   # <--- Variável Nova
        "scale": "64.0",
        "margin": "0.5",
        "lr": "3e-4",        # LR Padrão AdamW
        "warmup": "100"
    },

    {
        "desc": "Golden_S32_M0.5_LR3e-4",
        "optimizer": "adamw",   # <--- Variável Nova
        "scale": "32.0",
        "margin": "0.5",
        "lr": "3e-4",        # LR Padrão AdamW
        "warmup": "100"
    },

    {
        "desc": "Golden_S24_M0.5_LR3e-4",
        "optimizer": "adamw",   # <--- Variável Nova
        "scale": "24.0",
        "margin": "0.5",
        "lr": "3e-4",        # LR Padrão AdamW
        "warmup": "100"
    },

    {
        "desc": "Golden_S16_M0.5_LR3e-4",
        "optimizer": "adamw",   # <--- Variável Nova
        "scale": "16.0",
        "margin": "0.5",
        "lr": "3e-4",        # LR Padrão AdamW
        "warmup": "100"
    },

]

# Configurações Globais (SEM OPTIMIZER E SEM LR)
MAX_STEPS = 200       
COMMON_ARGS = [
    "--model-name", "InternVL3-2B",
    "--dataset-name", "RVL-CDIP",
    "--use-wandb",
    "--wandb-project", WANDB_PROJECT,
    
    # Otimização
    "--loss-type", LOSS_TYPE,
    # "--optimizer-type" REMOVIDO (Agora é dinâmico)
    # "--student-lr" REMOVIDO (Agora é dinâmico)
    "--scheduler-type", "constant", 
    
    # Batch & Accumulation (Simulando Batch 32)
    "--student-batch-size", "8",
    "--gradient-accumulation-steps", "4", 
    "--candidate-pool-size", "64",

    # Controles de Duração
    "--epochs", "1",
    "--max-steps-per-epoch", str(MAX_STEPS),
    
    # Professor Head
    "--professor-lr", "1e-4",
    "--cut-layer", "27",
    "--projection-output-dim", "1536",
    
    # Paths
    "--base-image-dir", BASE_IMAGE_DIR,
]

# ==============================================================================
# 🔄 PREPARAÇÃO DE DADOS
# ==============================================================================

generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL-CDIP_{PROTOCOL}_split_{SPLIT_IDX}")
pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")

# Verifica se o arquivo existe (e se o caminho mudou para o local)
if not os.path.exists(pairs_csv):
    print(f"🛠️  Gerando pares de treino...")
    cmd_prep = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(SPLIT_IDX),
        "--protocol", PROTOCOL,
        "--pairs-per-class", "100"
    ]
    subprocess.run(cmd_prep, check=True)
COMMON_ARGS.extend(["--pairs-csv", pairs_csv])

# ==============================================================================
# 🚀 LOOP DE EXECUÇÃO
# ==============================================================================

print(f"\n🎯 Iniciando Protocolo V3 (Optimizer + LR): {len(experiments)} configs.")
print(f"📂 Projeto WandB: {WANDB_PROJECT}")
print("=" * 60)

for i, exp in enumerate(experiments):
    run_id = f"{i+1:02d}_{exp['desc']}"
    print(f"\n▶️  Rodando Exp {i+1}/{len(experiments)}: {run_id}")
    print(f"   Config: Opt={exp['optimizer']} | Scale={exp['scale']} | LR={exp['lr']}")
    
    # Monta os argumentos
    run_args = COMMON_ARGS.copy()
    run_args.extend([
        "--wandb-run-name", run_id,
        "--optimizer-type", exp['optimizer'], # <--- Otimizador Inserido Aqui
        "--student-lr", exp['lr'],
        "--scale", exp['scale'],
        "--margin", exp['margin'],
        "--professor-warmup-steps", exp['warmup'],
        "--easy-mining-steps", exp['warmup'],
    ])
    
    cmd = [sys.executable, SCRIPT_PATH] + run_args
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"✅ Sucesso! Tempo: {duration:.1f}s")
    except subprocess.CalledProcessError:
        print(f"❌ Falha no experimento {run_id}. Passando para o próximo...")
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")
        break
        
    print("-" * 60)
    time.sleep(5)

print("\n🏁 Protocolo V3 finalizado.")