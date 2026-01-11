#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import argparse

# ==============================================================================
# ⚙️ GLOBAL CONFIG
# ==============================================================================

WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip" 
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR

# Common Settings for Protocol V3
PROTOCOL_DEFAULTS = {
    "wandb_project": "CaVL-Calibration-Protocol",
    "epochs": "5",
    "max_steps": "1000",
    "batch_size": "4",
    "grad_accum": "4", # Effective 32
    "pool_size": "8",
    "prof_lr": "1e-4",
    "cut_layer": "27",
    "proj_dim": "1536",
    "pairs": 15
}

# ==============================================================================
# 🧪 EXPERIMENT REGISTRY (SWEEP DEFINITIONS)
# ==============================================================================

EXP_REGISTRY = {
    "arcface": {
        "lr": [
            {"scale": 64.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-2, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-3, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-3, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 3e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "warmup": 100},
        ],
        "scale": [
            {"scale": 64.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 32.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 16.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
        ],
        "margin": [
            {"scale": 24.0, "margin": 0.3, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 24.0, "margin": 0.7, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
        ],
    },
    "cosface": {
        "scale": [
            # Placeholder values for CosFace
            {"scale": 32.0, "margin": 0.2, "lr": 3e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.2, "lr": 3e-4, "optimizer": "adamw", "warmup": 100},
        ]
    },
    "subcenter_arcface": {
        "k": [
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100, "k": 1}, # = Standard ArcFace
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100, "k": 5}, 
        ]
    }
}

# ==============================================================================
# 🛠️ UTILS
# ==============================================================================

def prepare_data(protocol="zsl", split_idx=1):
    num_pairs = PROTOCOL_DEFAULTS['pairs']
    # Adiciona o número de pares ao nome do diretório para evitar cache incorreto
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL-CDIP_{protocol}_split_{split_idx}_{num_pairs}pairs")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    
    if not os.path.exists(pairs_csv):
        print(f"🛠️  Generating training pairs for {protocol} split {split_idx} ({num_pairs} pairs/class)...")
        cmd_prep = [
            sys.executable, PREP_SCRIPT_PATH,
            "--data-root", RAW_DATA_ROOT,
            "--output-dir", generated_data_dir,
            "--split-idx", str(split_idx),
            "--protocol", protocol,
            "--pairs-per-class", str(num_pairs)
        ]
        subprocess.run(cmd_prep, check=True)
    return pairs_csv

def run_suite(experiment_name, sweep_param, experiments, dry_run=False):
    pairs_csv = prepare_data()
    
    # Dynamic Project Name for better organization
    # UNIFIED PROJECT NAME WITH GROUPS
    wandb_project = "CaVL-Protocol-Sweeps" 
    wandb_group = f"{experiment_name}_{sweep_param}" # ex: arcface_lr, arcface_scale
    
    print(f"\n🎯 Running Protocol V3 | Exp: {experiment_name} | Sweep: {sweep_param}")
    print(f"📂 WandB Project: {wandb_project} | Group: {wandb_group}")
    print("=" * 60)

    for i, exp in enumerate(experiments):
        # Auto-generate description keys
        # Format: Arcface_OptSGD_S64.0_M0.5_LR0.0003
        desc = f"Opt{exp.get('optimizer', 'adamw')}_S{exp['scale']}_M{exp['margin']}_LR{exp['lr']}"
        
        # Se tiver SubCenters, adiciona ao nome
        if 'k' in exp:
            desc += f"_K{exp['k']}"

        # REMOVIDO o sweep_param do nome para permitir compartilhamento de checkpoints entre sweeps
        # Adicionei o wandb_group como prefixo APENAS para log visual se quiser, mas o run_name deve ser único por config
        run_name = f"{experiment_name}_{desc}"
        
        print(f"\n▶️  Running {i+1}/{len(experiments)}: {run_name}")
        
        # Check for existing checkpoint to resume
        base_checkpoint_dir = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
        possible_ckpt = os.path.join(base_checkpoint_dir, run_name, "last_checkpoint.pt")
        
        resume_args = []
        if os.path.exists(possible_ckpt):
             print(f"🔄 Checkpoint found! Resuming from: {possible_ckpt}")
             resume_args = ["--resume-from", possible_ckpt]

        # Build Command
        cmd = [
            sys.executable, SCRIPT_PATH,
            "--model-name", "InternVL3-2B",
            "--dataset-name", "RVL-CDIP",
            "--use-wandb",
            "--wandb-project", wandb_project,
            "--wandb-run-name", run_name,
            # ADICIONANDO O ARGUMENTO DE GRUPO (precisa ser suportado no script de treino ou via env var)
            # Como o script principal não tem --wandb-group explícito, vamos passar via env var que o wandb pega
        ]
        
        # Injetando Environment Variable para o Grupo
        env = os.environ.copy()
        env["WANDB_RUN_GROUP"] = wandb_group

        cmd += [
            # Loss & Optimizer
            "--loss-type", experiment_name,
            "--optimizer-type", exp.get('optimizer', 'adamw'),
            "--student-lr", str(exp['lr']),
            "--scheduler-type", "constant",
            
            # Loss Params
            "--scale", str(exp['scale']),
            "--margin", str(exp['margin']),
            "--num-sub-centers", str(exp.get('k', 3)), # Default 3 se não especificado
            
            # Training Params
            "--epochs", PROTOCOL_DEFAULTS['epochs'],
            "--max-steps-per-epoch", PROTOCOL_DEFAULTS['max_steps'],
            "--student-batch-size", PROTOCOL_DEFAULTS['batch_size'],
            "--gradient-accumulation-steps", PROTOCOL_DEFAULTS['grad_accum'],
            "--candidate-pool-size", PROTOCOL_DEFAULTS['pool_size'],
            
            # Curriculum
            "--professor-warmup-steps", str(exp['warmup']),
            "--easy-mining-steps", str(exp['warmup']),
            "--professor-lr", PROTOCOL_DEFAULTS['prof_lr'],
            
            # Model details
            "--cut-layer", PROTOCOL_DEFAULTS['cut_layer'],
            "--projection-output-dim", PROTOCOL_DEFAULTS['proj_dim'],
            
            # Paths
            "--base-image-dir", BASE_IMAGE_DIR,
            "--pairs-csv", pairs_csv
        ] + resume_args

        if dry_run:
            print(f"Env: WANDB_RUN_GROUP={wandb_group}")
            print(" ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True, env=env)  # Passando o env modificado
            print("✅ Success")
        except subprocess.CalledProcessError:
            print("❌ Failed")
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
            break
        
        time.sleep(2)

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaVL Sweep Runner Protocol V3")
    parser.add_argument("--experiment", type=str, required=True, help="Loss type (e.g. arcface, cosface)")
    parser.add_argument("--parameter", type=str, required=True, help="Parameter to sweep (e.g. scale, margin)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    
    args = parser.parse_args()
    
    if args.experiment not in EXP_REGISTRY:
        print(f"Error: Experiment '{args.experiment}' not found in registry.")
        print(f"Available: {list(EXP_REGISTRY.keys())}")
        sys.exit(1)
        
    if args.parameter not in EXP_REGISTRY[args.experiment]:
        print(f"Error: Parameter '{args.parameter}' not defined for {args.experiment}.")
        print(f"Available: {list(EXP_REGISTRY[args.experiment].keys())}")
        sys.exit(1)
        
    experiments = EXP_REGISTRY[args.experiment][args.parameter]
    run_suite(args.experiment, args.parameter, experiments, args.dry_run)
