#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob

# ==============================================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS
# ==============================================================================

# Caminhos (Ajuste se necessário)
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/prepare_splits.py")

# Dados Originais (Read-Only)
RAW_DATA_ROOT = "/mnt/data/la-cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") # Imagens estão em /mnt/data/la-cdip/data

# Onde salvar os CSVs gerados para o experimento
PROTOCOL = "zsl"
SPLITS_TO_RUN = [0, 1, 2, 3, 4] # Split 1 já concluído

BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Lista de Losses para Comparar
LOSSES_TO_TEST = [
    "contrastive",      # Baseline
    "triplet",          # Baseline Metric Learning
    "arcface",          # SOTA Clássico
    "cosface",          # SOTA Clássico
    # "expface",          # Foco em exemplos limpos
    # "circle",           # Ponderação dinâmica
    # "subcenter_arcface" # Multi-Centro para classes grandes
    # "iso_arcface",      # Nossa Proposta (Angular)
    # "iso_cosface",      # Nossa Proposta (Aditiva)
    # "iso_circle"        # Nossa Proposta (Circle + Iso)
]

# Argumentos Comuns para todos os experimentos
COMMON_ARGS = [
    "--model-name", "InternVL3-2B",
    "--dataset-name", "LA-CDIP",
    "--epochs", "5",
    "--student-batch-size", "8",
    "--candidate-pool-size", "64",
    "--use-wandb",
    "--patience", "3",
    "--lr-reduce-factor", "0.5",
    # Parâmetros do experimento de referência
    "--student-lr", "1e-5",
    "--professor-lr", "1e-4",
    "--cut-layer", "27",
    "--projection-output-dim", "1536",
    "--max-num-image-tokens", "12",
    "--num-queries", "4",
    "--pooler-type", "attention",
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "20" # Default para LA-CDIP (Few-Shot)
]

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx):
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/{PROTOCOL}_split_{split_idx}")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    wandb_project = f"CaVL-Loss-Comparison-LACDIP-{PROTOCOL.upper()}-S{split_idx}"
    return generated_data_dir, pairs_csv, wandb_project

def prepare_data(split_idx, generated_data_dir):
    print(f"\n{'='*60}")
    print(f"🛠️  Preparando Dados para Split {split_idx} ({PROTOCOL.upper()})")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL,
        "--pairs-per-class", "10" # Reduzido para 20 para viabilizar o treino (Total ~4800 pares)
    ]
    
    print(f"Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✅ Dados preparados com sucesso!\n")

def run_experiment(loss_type, split_idx, pairs_csv, wandb_project):
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando Experimento: {loss_type.upper()} (Split {split_idx})")
    print(f"{'='*60}")

    # Determina onde os checkpoints estão sendo salvos
    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    # Nome base para busca
    base_name = f"LACDIP_{PROTOCOL}_S{split_idx}_{loss_type}"
    
    # Procura por runs anteriores
    search_pattern = os.path.join(base_ckpt_path, f"{base_name}_*")
    existing_runs = sorted(glob.glob(search_pattern))
    
    resume_path = None
    run_name = None
    
    if existing_runs:
        # Pega o mais recente
        latest_run = existing_runs[-1]
        ckpt_path = os.path.join(latest_run, "last_checkpoint.pt")
        
        if os.path.exists(ckpt_path):
            print(f"⚠️  Encontrado run anterior: {os.path.basename(latest_run)}")
            print(f"🔄 Retomando de: {ckpt_path}")
            resume_path = ckpt_path
            run_name = os.path.basename(latest_run)
        else:
            print(f"⚠️  Run anterior encontrado mas sem checkpoint válido: {os.path.basename(latest_run)}")
            # Se não tem checkpoint, melhor criar um novo para não misturar logs quebrados
    
    if not run_name:
        timestamp = time.strftime("%Y%m%d-%H%M")
        run_name = f"{base_name}_{timestamp}"

    # Usar o próprio run_name como ID para garantir consistência no resume
    # WandB IDs devem ser únicos por projeto.
    wandb_id = run_name.replace("/", "-").replace(":", "-")

    cmd = [
        sys.executable, SCRIPT_PATH,
        "--loss-type", loss_type,
        "--pairs-csv", pairs_csv,
        "--base-image-dir", BASE_IMAGE_DIR,
        "--wandb-project", wandb_project,
        "--wandb-run-name", run_name,
        "--wandb-id", wandb_id,
    ] + COMMON_ARGS
    
    if resume_path:
        cmd += ["--resume-from", resume_path]

    # Imprime o comando para debug
    print(f"Comando: {' '.join(cmd)}\n")

    try:
        # Executa o script e aguarda o término
        # check=True lança exceção se o processo retornar erro
        subprocess.run(cmd, check=True)
        print(f"\n✅ Experimento {loss_type} finalizado com sucesso!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao executar {loss_type}. Código de saída: {e.returncode}")
        # Opcional: Parar tudo ou continuar para o próximo?
        # Vamos continuar para garantir que os outros rodem.
    except KeyboardInterrupt:
        print("\n⚠️ Interrupção pelo usuário. Parando script de comparação.")
        sys.exit(1)

def main():
    print(f"Iniciando Bateria de Testes: {len(LOSSES_TO_TEST)} losses x {len(SPLITS_TO_RUN)} splits.")
    
    for split_idx in SPLITS_TO_RUN:
        generated_data_dir, pairs_csv, wandb_project = get_paths(split_idx)
        
        # 1. Prepara os dados para este split
        prepare_data(split_idx, generated_data_dir)
        
        print(f"\n>>> Rodando Split {split_idx}/{len(SPLITS_TO_RUN)}")
        print(f"Projeto WandB: {wandb_project}")
        
        for i, loss in enumerate(LOSSES_TO_TEST):
            print(f"\n>>> Progresso Split {split_idx}: Loss {i+1}/{len(LOSSES_TO_TEST)} ({loss})")
            run_experiment(loss, split_idx, pairs_csv, wandb_project)
            
            # Pequena pausa para garantir que o WandB sincronize/feche corretamente
            time.sleep(5)

    print(f"\n{'='*60}")
    print("🎉 Todos os experimentos foram concluídos!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
