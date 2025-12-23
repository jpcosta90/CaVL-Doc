#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import glob

# ==============================================================================
# CONFIGURAÇÃO: ABLATION STUDY - SAMPLE SIZE (RVL-CDIP ONLY)
# ==============================================================================
# Objetivo: Analisar eficiência de dados treinando com subsets reduzidos.
# Sizes: 2.5k, 5k, 10k.

# Caminhos
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/prepare_splits.py")

# Dados
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip"
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR 

# Experimento
PROTOCOL = "zsl"
SPLITS_TO_RUN = [0, 1, 2, 3]
# SAMPLES_PER_CLASS serve para controlar o tamanho do dataset.
# RVL-CDIP tem 16 classes. Se reservarmos 4 para teste (ZSL), ficam 12 para treino.
# Total Pares = 12 * 2 * SAMPLES_PER_CLASS (sendo metade positivo, metade negativo por 'pacote')
# generate_pairs gera X pares Positivos e X pares Negativos por classe.
# Logo, Total = 12 * 2 * X.
SAMPLES_PER_CLASS_LIST = [50, 200, 300] 

# CONFIGURAÇÃO ÓTIMA (RVL-CDIP)
# De acordo com a Sec 4.4.1 do paper, Triplet foi a melhor loss para RVL.
OPTIMAL_LOSS = "triplet"

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
    "--num-queries", "4",
    "--head-type", "mlp",
    "--baseline-alpha", "0.05",
    "--entropy-coeff", "0.01",
    "--val-samples-per-class", "100",
    "--loss-type", OPTIMAL_LOSS
]

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def get_paths(split_idx, samples_per_class):
    # Cria pasta específica para este tamanho de amostra
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL_SIZE_SPC{samples_per_class}_S{split_idx}")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    return generated_data_dir, pairs_csv

def prepare_data(split_idx, samples_per_class, generated_data_dir):
    if os.path.exists(os.path.join(generated_data_dir, "train_pairs.csv")):
        print(f"✅ Dados já existem em: {generated_data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🛠️  Preparando Dados: SPC={samples_per_class} | Split={split_idx}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, PREP_SCRIPT_PATH,
        "--data-root", RAW_DATA_ROOT,
        "--output-dir", generated_data_dir,
        "--split-idx", str(split_idx),
        "--protocol", PROTOCOL,
        "--pairs-per-class", str(samples_per_class)
    ]
    subprocess.run(cmd, check=True)

def run_experiment(samples_per_class, split_idx, pairs_csv):
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando: Sample Size - Docs/Class={samples_per_class} | Split={split_idx}")
    print(f"{'='*60}")

    base_ckpt_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    base_name = f"RVL_ABLATION_SIZE_SPC{samples_per_class}_S{split_idx}"
    wandb_project = "CaVL-Ablation-SampleSize-RVL"
    
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
        # Removemos --training-sample-size pois o CSV já define o tamanho
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
        print(f"\n✅ Experimento SPC={samples_per_class} finalizado!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro em SPC={samples_per_class}. Exit code: {e.returncode}")

def main():
    print(f"Iniciando Ablation de Tamanho de Amostra (RVL-CDIP)")
    print(f"Samples Per Class: {SAMPLES_PER_CLASS_LIST}")
    print(f"Splits: {SPLITS_TO_RUN}")
    
    for split_idx in SPLITS_TO_RUN:
        print(f"\n>>> Processando Split {split_idx}...")
        
        for spc in SAMPLES_PER_CLASS_LIST:
            generated_data_dir, pairs_csv = get_paths(split_idx, spc)
            prepare_data(split_idx, spc, generated_data_dir)
            
            run_experiment(spc, split_idx, pairs_csv)
            time.sleep(5)

    print("\n🎉 Ablation study RVL-CDIP concluído!")

if __name__ == "__main__":
    main()
